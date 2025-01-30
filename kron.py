"""Distributed PSGD Kron for GPU clusters."""

import string
import random
import time
import numpy as np

import torch
from torch import Tensor
import torch.distributed as dist
from torch.backends import opt_einsum

opt_einsum.set_flags(True, "optimal")
print(f"opt_einsum enabled: {opt_einsum.enabled}, opt_einsum strategy: {opt_einsum.strategy}")
torch.set_float32_matmul_precision("high")


def precond_update_prob_schedule(max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=500):
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
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob

    return _schedule


class Kron(torch.optim.Optimizer):
    """Implements PSGD Kron with simple layer-wise pipeline parallelism.

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
        partition_grads (bool): Whether to partition gradients.
        block_size (int): Size of partitions for gradient partitioning.
        dtype (torch.dtype): Data type for parameters and gradients.
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
        partition_grads=True,
        block_size=1024,
        dtype=torch.float32,
        rank=0,
        world_size=1,
    ):
        self.rank = rank
        self.world_size = world_size

        if preconditioner_update_probability is None:
            preconditioner_update_probability = precond_update_prob_schedule()

        params = list(params)
        sizes = {p.numel() for p in params}
        param_groups = [
            dict(
                params=[p for p in params if p.numel() == size],
                **create_update_buffer(size, self.world_size, dtype),
            )
            for size in sizes
        ]

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
            partition_grads=partition_grads,
            block_size=block_size,
            dtype=dtype,
        )
        super().__init__(param_groups, defaults)

        self._tiny = torch.tensor(torch.finfo(dtype).tiny, dtype=dtype, device="cuda")
        self._prob_step = 0
        self._update_counter = 0
        self.rng = random.Random(42)
        self.dtype = dtype

        self.comm_stream = torch.cuda.Stream()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        update_prob = self.param_groups[0]["preconditioner_update_probability"]
        if callable(update_prob):
            update_prob = update_prob(torch.tensor(self._prob_step, dtype=torch.float32))
        self._update_counter += 1
        do_update = self._update_counter >= 1 / update_prob
        if do_update:
            self._update_counter = 0
        self._prob_step += 1

        balance = do_update and self.rng.random() < 0.01

        handles = []
        params_world_list = []

        def process_pending_updates():
            while handles:
                handle = handles.pop(0)
                params_world = params_world_list.pop(0)
                if params_world is None:
                    continue
                handle.wait()
                views = update_buffer_views
                with torch.cuda.stream(self.comm_stream):
                    update = views[self.rank].view_as(params_world)
                    if group["weight_decay"] > 0 and params_world.dim() >= 2:
                        update.add_(params_world, alpha=group["weight_decay"])
                    params_world.add_(update.to(dtype=params_world.dtype), alpha=-group["lr"])

        for group in self.param_groups:
            update_buffer = group.get("update_buffer")
            if update_buffer is None:
                continue

            update_buffer_views = group["update_buffer_views"]
            params = group["params"]

            total_params = len(params)
            for chunk_idx in range((total_params + self.world_size - 1) // self.world_size):
                param_idx = chunk_idx * self.world_size + self.rank
                if param_idx < total_params:
                    p = params[param_idx]
                    if p.grad is None:
                        continue
                    grads = p.grad.to(dtype=self.dtype)
                    state = self.state[p]

                    # merge smaller dims
                    if grads.dim() > 1:
                        if "merged_shape" not in self.state[p]:
                            shape1 = [np.prod(grads.shape[:-1]), grads.shape[-1]]
                            shape2 = [grads.shape[0], np.prod(grads.shape[1:])]
                            shape = shape1 if np.diff(shape1) <= np.diff(shape2) else shape2
                            state["merged_shape"] = shape
                        else:
                            shape = state["merged_shape"]
                        grads = grads.view(*shape)

                    # partition grads
                    if self.defaults["partition_grads"]:
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
                    else:
                        grads_list = [grads]

                    precond_grads = []
                    for i, grad in enumerate(grads_list):
                        if f"step_{i}" not in state:
                            state[f"step_{i}"] = 0
                            state[f"momentum_buffer_{i}"] = torch.zeros_like(grad, dtype=self.dtype)
                            dim_diag = get_dim_diag(
                                self.defaults["memory_save_mode"],
                                grads.shape,
                                self.defaults["max_size_triangular"],
                                self.defaults["min_ndim_triangular"],
                            )
                            state[f"Q_{i}"], state[f"exprs_{i}"] = _init_Q_exprs(
                                grad, self.defaults["precond_init_scale"], dim_diag, self.dtype
                            )

                        state[f"step_{i}"] += 1

                        # momentum
                        momentum_buffer = state[f"momentum_buffer_{i}"]
                        beta = self.defaults["b1"]
                        momentum_buffer.mul_(beta).add_(grad, alpha=1 - beta)
                        debiased_momentum = momentum_buffer.div(1 - beta ** state[f"step_{i}"])

                        # balance Qs
                        if grad.dim() > 1 and balance:
                            _balance_Q(state[f"Q_{i}"])

                        # update preconditioner
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
                                    self.defaults["precond_lr"], dtype=self.dtype, device="cuda"
                                ),
                                self._tiny,
                            )

                        # precondition grads
                        precond_grads.append(
                            _precond_grad(state[f"Q_{i}"], state[f"exprs_{i}"], debiased_momentum)
                        )

                    # merge partitions
                    if self.defaults["partition_grads"]:
                        g = state["partitioner"].merge_partitions(precond_grads)
                    else:
                        g = precond_grads[0]

                    # clip grads at RMS 1.1
                    g.mul_(
                        torch.minimum(
                            torch.tensor(1.0, dtype=self.dtype),
                            1.1 / g.square().mean().sqrt().add(1e-6),
                        )
                    )

                    g = g.flatten()
                    process_pending_updates()
                    compute_event = torch.cuda.Event()
                    compute_event.record()
                    with torch.cuda.stream(self.comm_stream):
                        compute_event.wait()
                        update_buffer_views[self.rank].copy_(g)
                        handle = dist.all_gather_into_tensor(
                            update_buffer, update_buffer_views[self.rank], async_op=True
                        )
                        handles.append(handle)
                        params_world_list.append(
                            params[param_idx] if param_idx < total_params else None
                        )
            process_pending_updates()
            if handles:
                with torch.cuda.stream(self.comm_stream):
                    handles[0].wait()
                torch.cuda.current_stream().wait_stream(self.comm_stream)
        return loss


def create_update_buffer(size: int, world_size: int, dtype: torch.dtype):
    b = torch.empty(world_size, size, dtype=dtype, device="cuda")
    return dict(update_buffer=b, update_buffer_views=b.unbind(0))


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


def _init_Q_exprs(t, scale, dim_diag, dtype):
    """Initialize preconditioner Q and reusable einsum expressions."""
    letters = string.ascii_lowercase + string.ascii_uppercase

    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = [scale * torch.ones_like(t, dtype=dtype)]
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = scale ** (1 / len(shape))

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, dim_d in enumerate(dim_diag):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                Q.append(scale * torch.ones(shape[i], dtype=dtype, device=t.device))

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                Q.append(scale * torch.eye(shape[i], dtype=dtype, device=t.device))

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))]
                )
                piece2 = "".join(
                    [(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))]
                )
                subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P

    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


@torch.compile(fullgraph=True, dynamic=False)
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
    orig_dtype = A.dtype
    return (
        torch.linalg.solve_triangular(
            A.float(),
            X.reshape(-1, X.size(-1)).float(),
            upper=True,
            left=False,
            unitriangular=False,
        )
        .reshape_as(X)
        .to(dtype=orig_dtype)
    )


def _calc_A_and_conjB(exprA, G, Q):
    order = G.dim()
    V = torch.randn_like(G, device=G.device)
    eps = torch.tensor(torch.finfo(torch.float32).eps, dtype=G.dtype, device=G.device)
    G += eps.sqrt() * G.abs().mean() * V
    conjB = V.permute(*range(1, order), 0)
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    A = torch.einsum(exprA, *Q, G)
    return A, conjB


@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
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


@torch.compile(fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")
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
        assert len(dim_diag) == len(param_shape), "dim_diag must have same length as param_shape"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._shape = param_shape
        self._split_indices = []
        self._split_dims = []
        for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not is_diag:
                nsplit = (d - 1) // block_size
                if nsplit > 0:
                    self._split_indices.append([(j + 1) * block_size for j in range(nsplit)])
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
        for dim, indices in zip(reversed(self._split_dims), reversed(self._split_indices)):
            n = len(indices) + 1
            merged = []
            for i in range(0, len(blocks), n):
                merged.append(torch.cat(blocks[i : i + n], dim=dim))
            blocks = merged
        return blocks[0]


if __name__ == "__main__":
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="tcp://localhost:23456", world_size=1, rank=0
        )
        torch.cuda.set_device(0)

    print("\nTesting BlockPartitioner...")
    device = "cuda"

    test_shapes = [((64, 64), 32), ((256, 256), 64), ((2048, 2048), 128), ((4096, 4096), 256)]

    for shape, block_size in test_shapes:
        print(f"\nShape: {shape}, block_size: {block_size}")
        x = torch.arange(np.prod(shape), dtype=torch.float32, device=device).reshape(shape)
        partitioner = BlockPartitioner(x.shape, block_size, [False] * len(shape))

        # warmup
        for _ in range(3):
            blocks = partitioner.partition(x)
            merged = partitioner.merge_partitions(blocks)
            del blocks, merged

        times = []
        max_error = 0
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()

            blocks = partitioner.partition(x)
            merged = partitioner.merge_partitions(blocks)

            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            max_error = max(max_error, (x - merged).abs().max().item())

        avg_time = sum(times) / len(times)
        print(
            f"Blocks: {partitioner._total_blocks}, "
            f"Time: {avg_time*1000:.1f}ms, "
            f"Error: {max_error:.2e}"
        )

    print("\nTesting optimization...")

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 8192), torch.nn.ReLU(), torch.nn.Linear(8192, 1024)
    ).to(device)

    optimizer = Kron(
        model.parameters(),
        lr=0.001,
        block_size=512,
        precond_lr=0.3,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )

    x = torch.randn(512, 1024, device=device) * 0.1
    y = torch.randn(512, 1024, device=device) * 0.1

    # track memory and performance
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    training_start = time.perf_counter()

    losses = []
    for step in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

    training_time = time.perf_counter() - training_start
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    mem_delta = (torch.cuda.memory_allocated() - start_mem) / 1024**2

    print(f"\nResults for rank {dist.get_rank()}:")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Training time: {training_time:.1f}s")
    print(f"Peak memory: {peak_mem:.1f} MB")
    print(f"Memory delta: {mem_delta:.1f} MB")

    assert losses[-1] < losses[0], "Loss did not decrease!"

    dist.destroy_process_group()
