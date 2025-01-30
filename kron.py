"""PSGD Kron for torch with pipeline sharding from Muon and several improvements from heavyball."""

import string
import random
import numpy as np
import time

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
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grads = p.grad.bfloat16()

                    # merge smaller dims
                    if grads.dim() > 1:
                        if "merged_shape" not in self.state[p]:
                            shape1 = [np.prod(grads.shape[:-1]), grads.shape[-1]]
                            shape2 = [grads.shape[0], np.prod(grads.shape[1:])]
                            shape = (
                                shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                            )
                            state["merged_shape"] = shape
                        else:
                            shape = state["merged_shape"]
                        grads = grads.view(*shape)

                    if "partitioner" not in state:
                        dim_diag = get_dim_diag(
                            self.defaults["memory_save_mode"],
                            grads.shape,
                            self.defaults["max_size_triangular"],
                            self.defaults["min_ndim_triangular"],
                        )
                        state["partitioner"] = BlockPartitioner(
                            grads.shape, self.defaults["block_size"], dim_diag
                        )

                    grads_list = state["partitioner"].partition(grads)
                    precond_grads = []
                    for i, grad in enumerate(grads_list):
                        if f"step_{i}" not in state:
                            state[f"step_{i}"] = 0
                            state[f"momentum_buffer_{i}"] = torch.zeros_like(grad)
                            state[f"Q_{i}"], state[f"exprs_{i}"] = _init_Q_exprs(
                                grad, self.defaults["precond_init_scale"], dim_diag
                            )

                        state[f"step_{i}"] += 1

                        momentum_buffer = state[f"momentum_buffer_{i}"]
                        beta = self.defaults["b1"]
                        momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                        debiased_momentum = momentum_buffer.div(
                            1 - beta ** state[f"step_{i}"]
                        )

                        if grad.dim() > 1 and balance:
                            _balance_Q(state[f"Q_{i}"])

                        if do_update:
                            _update_precond(
                                state[f"Q_{i}"],
                                state[f"exprs_{i}"],
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

                        precond_grads.append(
                            _precond_grad(
                                state[f"Q_{i}"], state[f"exprs_{i}"], debiased_momentum
                            )
                        )

                    grads = state["partitioner"].merge_partitions(precond_grads)

                    g.mul_(
                        torch.minimum(
                            torch.tensor(1.0), 1.1 / g.square().mean().sqrt().add(1e-6)
                        )
                    )

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


def _init_Q_exprs(t, scale, dim_diag):
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
    return torch.einsum(exprs[-1], *Q, *Q, G)


class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py

    Scalable Second Order Optimization for Deep Learning by
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size, dim_diag):
        assert len(dim_diag) == len(
            param_shape
        ), "dim_diag must have same length as param_shape"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._shape = param_shape
        self._split_indices = []
        self._split_dims = []
        for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not is_diag:
                nsplit = (d - 1) // block_size
                if nsplit > 0:
                    self._split_indices.append(
                        [(j + 1) * block_size for j in range(nsplit)]
                    )
                    self._split_dims.append(i)

        self._total_blocks = (
            np.prod([len(indices) + 1 for indices in self._split_indices])
            if self._split_indices
            else 1
        )

    def partition(self, tensor):
        assert tensor.shape == self._shape
        blocks = [tensor]

        for dim, indices in zip(self._split_dims, self._split_indices):
            new_blocks = []
            for block in blocks:
                split_blocks = torch.tensor_split(block, indices, dim=dim)
                new_blocks.extend(split_blocks)
            blocks = new_blocks

        return blocks

    def merge_partitions(self, partitions):
        blocks = list(partitions)

        for dim, indices in zip(
            reversed(self._split_dims), reversed(self._split_indices)
        ):
            n = len(indices) + 1
            merged = []

            for i in range(0, len(blocks), n):
                merged.append(torch.cat(blocks[i : i + n], dim=dim))
            blocks = merged

        return blocks[0]


if __name__ == "__main__":
    print("\nTesting BlockPartitioner...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_cases = [
        ((64, 64), 32),
        ((256, 256), 64),
        ((2048, 2048), 128),
        ((4096, 4096), 256),
        ((16, 32, 64), 32),
    ]

    WARMUP_RUNS = 3
    TIMING_RUNS = 5

    for shape, block_size in test_cases:
        print(f"\nShape: {shape}, block_size: {block_size}")

        x = torch.arange(np.prod(shape), dtype=torch.float32, device=device).reshape(
            shape
        )
        partitioner = BlockPartitioner(x.shape, block_size, [False] * len(shape))

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            blocks = partitioner.partition(x)
            merged = partitioner.merge_partitions(blocks)
            del blocks, merged

        # Timing runs
        times = []
        error = 0
        for _ in range(TIMING_RUNS):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            blocks = partitioner.partition(x)
            merged = partitioner.merge_partitions(blocks)

            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

            error = max(error, (x - merged).abs().max().item())
            del blocks, merged

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        print(
            f"Blocks: {partitioner._total_blocks}, "
            f"Time: {avg_time*1000:.1f}±{std_time*1000:.1f}ms, "
            f"Error: {error:.2e}"
        )

        if device == "cuda":
            torch.cuda.empty_cache()
