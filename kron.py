"""PSGD Kron for torch with pipeline sharding from Muon and several improvements from heavyball."""

import string
import random
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist


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

    @torch.compile(fullgraph=True, dynamic=False)
    def _schedule(n):
        """Exponential anneal with flat start."""
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


def create_update_buffer(size: int, world_size: int):
    device = torch.device("cuda", torch.cuda.current_device())
    b = torch.empty(world_size, size, dtype=torch.bfloat16, device=device)
    return dict(update_buffer=b, update_buffer_views=b.unbind(0))


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron with simple pipeline sharding example.

    Parameters:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        b1 (float): Momentum parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode (str, optional): Memory saving mode for preconditioners.
            Options:
            None: Set all preconditioners to be triangular
            'smart_one_diag': Sets any large dims that stand out to be diagonal
            'one_diag': Sets the largest or last dim to be diagonal
            'all_diag': Sets all preconditioners to be diagonal

        momentum_into_precond_update (bool): Whether to send momentum into
            preconditioner update instead of raw gradients.
        precond_lr (float, optional): Learning rate for preconditioner updates.
            Default: 0.1
        precond_init_scale (float, optional): Initial scale for preconditioner
            matrices. Default: 1.0
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
        block_size=1024,
        rank=0,
        world_size=1,
    ):
        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        self.rank = rank
        self.world_size = world_size

        params = list(params)
        sizes = {p.numel() for p in params}
        param_groups = [
            dict(
                params=[p for p in params if p.numel() == size],
                **create_update_buffer(size, world_size),
            )
            for size in sizes
        ]

        if world_size < 1:
            raise ValueError(f"Invalid world_size {world_size}, must be ≥1")
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid rank {rank} for world_size {world_size}")

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
            block_size=block_size,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(
            torch.finfo(torch.bfloat16).tiny, dtype=torch.bfloat16, device="cuda"
        )
        self._prob_step = 0
        self._update_counter = 0
        self.rng = random.Random(42)

        self.comm_stream = torch.cuda.Stream()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(
                torch.tensor(self._prob_step, dtype=torch.float32)
            )
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = 0
        self._prob_step += 1

        balance = do_update and self.rng.random() < 0.01

        handles = []
        params_world_list = []

        def update_prev():
            while handles:
                handle = handles.pop(0)
                params_world = params_world_list.pop(0)
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    with torch.cuda.stream(self.comm_stream):
                        update = g_world.view_as(p_world)
                        if group["weight_decay"] > 0 and p_world.dim() >= 2:
                            update.add_(p_world, alpha=group["weight_decay"])
                        p_world.add_(update, alpha=-group["lr"])

        for group in self.param_groups:
            update_buffer = group.get("update_buffer")
            if update_buffer is None:
                continue
            update_buffer_views = group["update_buffer_views"]
            params = group["params"]

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]

                    if grad is None:
                        continue

                    grad = p.grad.bfloat16()

                    # let's merge smaller dims for better memory management
                    orig_shape = grad.shape
                    grad = grad.view(-1, grad.size(-1))

                    state = self.state[p]

                    if "partitioner" not in self.state[p]:

                    if len(state) == 0:
                        state["step"] = 0
                        state["momentum_buffer"] = torch.zeros_like(grad)
                        state["Q"], state["exprs"] = _init_Q_exprs(
                            grad,
                            self.defaults["precond_init_scale"],
                            self.defaults["max_size_triangular"],
                            self.defaults["min_ndim_triangular"],
                            self.defaults["memory_save_mode"],
                        )

                    state["step"] += 1

                    momentum_buffer = state["momentum_buffer"]
                    beta = self.defaults["b1"]
                    momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                    debiased_momentum = momentum_buffer.div(1 - beta ** state["step"])

                    if grad.dim() > 1 and balance:
                        _balance_Q(state["Q"])

                    if do_update:
                        _update_precond(
                            state["Q"],
                            state["exprs"],
                            (
                                debiased_momentum
                                if self.defaults["momentum_into_precond_update"]
                                else grad
                            ),
                            torch.tensor(
                                self.defaults["precond_lr"],
                                dtype=torch.bfloat16,
                                device="cuda",
                            ),
                            self._tiny,
                        )

                    g = _precond_grad(state["Q"], state["exprs"], debiased_momentum)

                    g.mul_(
                        torch.minimum(
                            torch.tensor(1.0), 1.1 / g.square().mean().sqrt().add(1e-6)
                        )
                    )

                    g = g.view(orig_shape)

                    if self.defaults["weight_decay"] > 0 and p.dim() >= 2:
                        g.add_(p, alpha=self.defaults["weight_decay"])

                    g = g.flatten()  # flatten before all_gather
                else:
                    update_buffer_views[self.rank].zero_()
                    g = update_buffer_views[self.rank]

                update_prev()
                with torch.cuda.stream(self.comm_stream):
                    handle = dist.all_gather_into_tensor(
                        update_buffer, g, async_op=True
                    )
                handles.append(handle)
                params_world_list.append(params[base_i : base_i + self.world_size])

            update_prev()
            if handles:
                handles[0].wait()

        torch.cuda.synchronize()
        return loss


def get_dim_diag(memory_save_mode, shape, max_size, min_ndim):
    if memory_save_mode is None:
        dim_diag = [False for _ in shape]
    elif memory_save_mode == "smart_one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [False for _ in shape]
        sorted_shape = sorted(shape)
        if len(shape) > 1 and sorted_shape[-1] > sorted_shape[-2]:
            dim_diag[rev_sorted_dims[0]] = True
    elif memory_save_mode == "one_diag":
        rev_sorted_dims = np.argsort(shape)[::-1]
        dim_diag = [i == rev_sorted_dims[0] for i in range(len(shape))]
    elif memory_save_mode == "all_diag":
        dim_diag = [True for _ in shape]
    else:
        raise ValueError(
            f"Invalid memory_save_mode: {memory_save_mode}, must be one of "
            "[None, 'smart_one_diag', 'one_diag', 'all_diag']"
        )
    if len(shape) < min_ndim:
        return [True for _ in shape]
    for i in range(len(shape)):
        size = shape[i]
        if size == 1 or size > max_size:
            dim_diag[i] = True
    return dim_diag


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

        dim_diag = get_dim_diag(memory_save_mode, shape, max_size, min_ndim_triangular)

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, dim_d in enumerate(dim_diag):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                Q.append(
                    scale * torch.ones(shape[i], dtype=torch.bfloat16, device=t.device)
                )

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
                Q.append(
                    scale * torch.eye(shape[i], dtype=torch.bfloat16, device=t.device)
                )

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


# @torch.compile(fullgraph=True, dynamic=False)
def _balance_Q(Q_in):
    norms = torch.stack([q.norm(float("inf")) for q in Q_in])
    geometric_mean = norms.log().mean().exp()
    norms = geometric_mean / norms
    torch._foreach_mul_(Q_in, list(norms))


def _lb(A: Tensor, max_abs: Tensor):
    A /= max_abs
    a0 = torch.einsum("ij,ij->j", A, A)
    i = torch.argmax(a0)
    x = torch.index_select(A, 1, i).flatten().contiguous()
    x = torch.einsum("i,ij->j", x, A)
    x /= x.norm()
    x = torch.einsum("j,kj->k", x, A)
    x = x.norm()
    x *= max_abs
    return x


def _solve_triangular_right(X: Tensor, A: Tensor):
    """X @ inv(A) with minimal float32 usage"""
    return (
        torch.linalg.solve_triangular(
            A.float(),
            X.reshape(-1, X.size(-1)).float(),
            upper=True,
            left=False,
            unitriangular=False,
        )
        .reshape_as(X)
        .bfloat16()
    )


def _calc_A_and_conjB(exprA, G, Q):
    order = G.dim()
    V = torch.randn_like(G, device=G.device)
    eps = torch.tensor(
        torch.finfo(torch.bfloat16).eps, dtype=torch.bfloat16, device=G.device
    )
    G += eps.sqrt() * G.abs().mean() * V
    conjB = V.permute(*range(1, order), 0)
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    A = torch.einsum(exprA, *Q, G)
    return A, conjB


# @torch.compile(fullgraph=True, dynamic=False)
def _update_precond(Q, exprs, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs
    A, conjB = _calc_A_and_conjB(exprA, G, Q)
    for q, exprG in zip(Q, exprGs):
        term1 = torch.einsum(exprG, A, A)
        term2 = torch.einsum(exprG, conjB, conjB)
        term1, term2 = term1 - term2, term1 + term2
        term1 *= step
        norm = term2.norm(float("inf"))
        if q.dim() < 2:
            term1 *= q / norm.clamp_(min=tiny)
        else:
            torch.triu(term1, out=term1)
            term1 /= torch.where(norm > 0, _lb(term2, norm), norm).clamp_(tiny)
            term1 = torch.mm(term1, q)
        q.sub_(term1)


# @torch.compile(fullgraph=True, dynamic=False)
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    result = torch.einsum(exprs[-1], *Q, *Q, G)
    return result


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.
    
    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    """
    
    def __init__(self, param_shape, block_size, dim_diag):
        assert len(dim_diag) == len(param_shape), "dim_diag must match param_shape length"
        self._shape = param_shape
        self.block_size = block_size
        self._splits = []
        split_sizes = []
        
        # Calculate split points for each dimension
        for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not is_diag:
                # Calculate split indices using numpy-style splitting
                nsplit = (d - 1) // block_size  # Ensures last block isn't empty
                indices = (torch.arange(nsplit, dtype=torch.int32) + 1) * block_size
                sizes = [block_size] * nsplit + [d - nsplit * block_size]
                self._splits.append((i, indices))
                split_sizes.append(torch.tensor(sizes, dtype=torch.int32))
            else:
                split_sizes.append(torch.tensor([d], dtype=torch.int32))
        
        self._split_sizes = split_sizes
        self._padded_stacked_shape = self._calculate_padded_shape()

    def _calculate_padded_shape(self):
        # Calculate padded shape for potential later use
        single_shape = [s[0].item() for s in self._split_sizes]
        padded_single_shape = [
            -(-dim // self.block_size) * self.block_size for dim in single_shape
        ]
        stack_size = max(1, np.prod([len(s) for s in self._split_sizes]))
        return (stack_size,) + tuple(padded_single_shape)

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Split tensor into blocks using precomputed split points."""
        assert tensor.shape == self._shape, "Input tensor shape mismatch"
        blocks = [tensor]
        
        for dim, indices in self._splits:
            new_blocks = []
            for block in blocks:
                # Split along current dimension using exact indices
                split_blocks = torch.tensor_split(block, indices.cpu(), dim=dim)
                new_blocks.extend(split_blocks)
            blocks = new_blocks
            
        return tuple(blocks)

    def merge_partitions(self, partitions):
        """Merge blocks back to original shape using inverse splitting order."""
        blocks = list(partitions)
        
        for dim, indices in reversed(self._splits):
            n = len(indices) + 1  # Number of splits per tensor
            merged = []
            while blocks:
                # Take n blocks and concatenate along original dimension
                to_merge = blocks[:n]
                merged.append(torch.cat(to_merge, dim=dim))
                blocks = blocks[n:]
            blocks = merged
            
        assert len(blocks) == 1, "Merging failed to reconstruct original tensor"
        return blocks[0]
