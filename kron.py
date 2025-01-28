import string
import random
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist


torch._dynamo.config.cache_size_limit = 1_000_000


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        n = torch.tensor(n, dtype=torch.float32)
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


def create_update_buffer(size: int, world_size: int):
    b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
    return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron with simple pipeline sharding example.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode: (string, optional), None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: (bool), whether to send momentum into preconditioner
            update instead of raw gradients.
        precond_lr (float, optional): Learning rate for preconditioner updates. Default: 0.1
        precond_init_scale (float, optional): Initial scale for preconditioner matrices. Default: 1.0
        rank (int): This worker's rank, used in pipeline partitioning.
        world_size (int): Total number of workers, used in pipeline partitioning.
    """

    def __init__(
        self,
        params,
        lr=0.001,
        b1=0.9,
        weight_decay=0.0,
        preconditioner_update_probability=None,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        precond_lr=0.1,
        precond_init_scale=1.0,
        rank=0,
        world_size=1,
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        self.rank = rank
        self.world_size = world_size

        params = list(params)
        sizes = {p.numel() for p in params}
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.numel() == size]
            if size > 0:
                buf_kwargs = create_update_buffer(size, world_size)
            else:
                buf_kwargs = {}
            param_groups.append(
                dict(params=group_params, **buf_kwargs)
            )

        defaults = dict(
            lr=lr,
            b1=b1,
            weight_decay=weight_decay,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.finfo(torch.bfloat16).tiny
        self._prob_step = 0
        self._update_counter = 0
        self.rng = random.Random(42)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(self._prob_step)
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = 0
        self._prob_step += 1

        balance = do_update and self.rng.random() < 0.01

        handle = None
        params_world = None
        def update_prev():
            if params_world is None:
                return
            assert handle is not None
            handle.wait()
            for p_world, g_world in zip(params_world, update_buffer_views):
                p_world.add_(g_world.view_as(p_world), alpha=-self.defaults["lr"])

        for group in self.param_groups:
            update_buffer = group.get("update_buffer")
            if update_buffer is None:
                continue
            update_buffer_views = group["update_buffer_views"]
            params = group["params"]

            handle = None
            params_world = None
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.bfloat16)
                        state["Q"], state["exprs"] = _init_Q_exprs(
                            p,
                            self.defaults["precond_init_scale"],
                            self.defaults["max_size_triangular"],
                            self.defaults["min_ndim_triangular"],
                            self.defaults["memory_save_mode"],
                        )

                    state["step"] += 1
                    
                    momentum_buffer = state["momentum_buffer"]
                    beta = self.defaults["b1"]
                    momentum_buffer.mul_(beta).add_(p.grad, alpha=1 - beta)
                    # keep buffer in bfloat16
                    momentum_buffer.copy_(momentum_buffer.to(dtype=torch.bfloat16))
                    debiased_momentum = momentum_buffer.div(1 - beta ** state["step"])

                    if p.grad.dim() > 1 and balance:
                        _balance_Q(state["Q"])

                    if do_update:
                        _update_precond(
                            state["Q"],
                            state["exprs"],
                            debiased_momentum if self.defaults["momentum_into_precond_update"] else p.grad.to(dtype=torch.bfloat16),
                            self.defaults["precond_lr"],
                            self._tiny,
                        )

                    g = _precond_grad(state["Q"], state["exprs"], debiased_momentum)
                    g.mul_(torch.minimum(torch.tensor(1.0), 1.1 / g.square().mean().sqrt().add(1e-6)))

                    if self.defaults["weight_decay"] > 0 and p.dim() >= 2:
                        g.add_(p, alpha=self.defaults["weight_decay"])
                    
                    assert state["momentum_buffer"].dtype == torch.bfloat16
                    assert debiased_momentum.dtype == torch.bfloat16
                    assert g.dtype == torch.bfloat16
                    assert state["Q"][0].dtype == torch.bfloat16

                    g = g.flatten()
                else:
                    g = update_buffer_views[self.rank]

                update_prev()
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

        return loss


def _init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode):
    """Initialize preconditioner Q and reusable einsum expressions."""
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=torch.bfloat16)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:
        if len(shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!"
            )

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
                "[None, 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if (
                size == 1
                or size > max_size
                or len(shape) < min_ndim_triangular
                or dim_d
            ):
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(size, dtype=torch.bfloat16, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(size, dtype=torch.bfloat16, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = torch.exp(torch.log(norms).mean())
    norms = geometric_mean / norms
    for i, q in enumerate(Q_in):
        q.mul_(norms[i])


def _lb(A: Tensor, max_abs: Tensor):
    A = A / max_abs
    x = A[:, torch.max(torch.sum(A * A, dim=0), 0)[1]] @ A
    return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)


@torch.compiler.disable
def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _solve_triangular_right(X: Tensor, A: Tensor):
    """X @ inv(A) with minimal float32 usage"""
    return torch.linalg.solve_triangular(
        A.float(), 
        X.reshape(-1, X.size(-1)).float(), 
        upper=True, 
        left=False
    ).reshape_as(X).to(dtype=torch.bfloat16)


@torch.compile
def _calc_terms(exprA, exprGs, G, Q):
    order = G.dim()
    conjB = torch.randn(G.shape[1:] + G.shape[:1], dtype=torch.bfloat16, device=G.device)
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)

    A = torch.einsum(exprA, *Q, G)

    terms = []
    for exprG in exprGs:
        term1 = torch.einsum(exprG, A, A)
        term2 = torch.einsum(exprG, conjB, conjB)
        terms.append((term1, term2))
    return terms


def _update_precond(Q, exprs, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs
    
    terms = _calc_terms(exprA, exprGs, G, Q)

    for q, (term1, term2) in zip(Q, terms):
        tmp = term1 - term2
        tmp *= step
        if q.dim() < 2:
            tmp *= q
            tmp /= (term1 + term2).norm(float("inf")) + tiny
            q.sub_(tmp)
        else:
            tmp = torch.triu(tmp)
            tmp /= _norm_lower_bound(term1 + term2) + tiny
            tmp @= q
            q.sub_(tmp)


@torch.compile
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *Q, *Q, G)
