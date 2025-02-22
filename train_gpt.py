import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import wandb

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention
torch._inductor.config.coordinate_descent_tuning = True # turn this off for a faster compile time (but slightly slower run)
torch.set_float32_matmul_precision('high')

from kron import Kron, precond_update_prob_schedule

# -----------------------------------------------------------------------------
# Custom operators : FP8 matmul for lm_head by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.mul(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.mul(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.t(),
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(1 / x_s, dtype=torch.float32),
            scale_b=x.new_tensor(1 / w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.t(), x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(1 / x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(1 / w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(1 / grad_s, dtype=torch.float32)
        grad_f8 = grad.mul(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.t().contiguous().t(),
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        # faster than grad_f8_t @ x_f8, for (d_out, d_in) == (50304, 768)
        grad_w = torch._scaled_mm(
            x_f8.t().contiguous(),
            grad_f8.t().contiguous().t(),
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).t()
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

def lm_head_fp8(x: Tensor, w: Tensor) -> Tensor:
    _x = x.flatten(0, -2)
    out: Tensor = torch.ops.nanogpt.mm(_x, w, x_s=2.0, w_s=32.0, grad_s=2.0**29)[0]
    return out.reshape(*x.shape[:-1], -1)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

USE_WANG_INIT = True

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=False)

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5) # 0.5 is a bit better than the default 1/sqrt(3)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len=65536):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_idx: int, num_layers: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std # improved init scale by @YouJiacheng
        # merged QKV weights: suggested by many, implemented by @fernbear.bsky.social, and further improved by @YouJiacheng
        # https://x.com/hi_tysam/status/1879699187107033311
        self.qkv_w = nn.Parameter(torch.empty(3, dim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(dim // num_heads) # dim // num_heads = head_dim
        self.c_proj = CastedLinear(dim, dim)
        if USE_WANG_INIT:
            std = 2.0 / (num_layers * dim**0.5)
            nn.init.normal_(self.c_proj.weight, mean=0.0, std=std)
        else:
            self.c_proj.weight.detach().zero_()
        self.attn_scale = 0.12

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3*self.num_heads, -1).chunk(3, dim=-2)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v) # @KoszarskyB & @Grad62304977
        else: # skip mid-layers token value embeddings by @YouJiacheng
            v = self.lambdas[0] * v
        q, k = norm(q), norm(k) # QK norm @Grad62304977
        q, k = self.rotary(q), self.rotary(k)
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=self.attn_scale)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        if USE_WANG_INIT:
            std = 2.0 / (num_layers * dim**0.5)
            nn.init.normal_(self.c_proj.weight, mean=0.0, std=std)
        else:
            self.c_proj.weight.detach().zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, layer_idx: int, num_layers: int):
        super().__init__()
        # skip attention of blocks.7 (the 8th layer) by @YouJiacheng
        self.attn = CausalSelfAttention(model_dim, num_heads, layer_idx, num_layers) if layer_idx != 7 else None
        self.mlp = MLP(model_dim, num_layers)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, ve, x0, block_mask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

class ValueEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embed = nn.ModuleList([nn.Embedding(num_embeddings, embedding_dim) for _ in range(3)])

    def forward(self, input_seq) -> list[Tensor | None]:
        ve = [emb(input_seq) for emb in self.embed]
        # 012 ... 012 structure on token value embeddings by @YouJiacheng, improved on @leloykun's U-net structure
        ve = [ve[0], ve[1], ve[2], None, None, None, None, None, None, None, None, None, None, ve[0], ve[1], ve[2]]
        return ve

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim)
        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        self.value_embeds = ValueEmbedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, layer_idx, num_layers) for layer_idx in range(num_layers)])
        # U-net design by @brendanh0gan
        self.num_encoder_layers = num_layers // 2 # Half of the layers for encoder
        self.num_decoder_layers = num_layers - self.num_encoder_layers # Remaining for decoder
        # Add learnable skip connection weights for decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency.
        # suggested to me by @Grad62304977. this originates from Karpathy's experiments.
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128))
        if USE_WANG_INIT:
            std = 2.0 / (num_layers * model_dim**0.5)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        else:
            self.lm_head.weight.detach().zero_()

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        BLOCK_SIZE = 128
        assert input_seq.ndim == 1
        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        docs = (input_seq == 50256).cumsum(0)
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_mask: Tensor):
            num_blocks = dense_mask.sum(dim=-1, dtype=torch.int32)
            indices = dense_mask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        # manual block mask creation by @YouJiacheng
        def create_doc_swc_block_masks(sliding_window_num_blocks: Tensor):
            kv_idx = block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
            q_idx = block_idx[:, None]
            causal_bm = q_idx >= kv_idx
            causal_full_bm = q_idx > kv_idx
            document_bm = (docs_low[:, None] <= docs_high) & (docs_low <= docs_high[:, None])
            document_full_bm = (docs_low[:, None] == docs_high) & (docs_low == docs_high[:, None])
            nonzero_bm = causal_bm & document_bm
            full_bm  = causal_full_bm & document_full_bm
            kv_num_blocks, kv_indices = dense_to_ordered(nonzero_bm & ~full_bm)
            full_kv_num_blocks, full_kv_indices = dense_to_ordered(full_bm)
            def build_bm(sw_num_blocks: Tensor) -> BlockMask:
                return BlockMask.from_kv_blocks(
                    torch.clamp_max(kv_num_blocks, torch.clamp_min(sw_num_blocks - full_kv_num_blocks, 1)),
                    kv_indices,
                    torch.clamp_max(full_kv_num_blocks, sw_num_blocks - 1),
                    full_kv_indices,
                    BLOCK_SIZE=BLOCK_SIZE,
                    mask_mod=document_causal,
                )
            return build_bm(sliding_window_num_blocks), build_bm(sliding_window_num_blocks // 2)

        # Long-short SWA block masks by @leloykun & @YouJiacheng, adapated from suggestion by @Grad62304977, following Gemma 2 paper
        long_bm, short_bm = create_doc_swc_block_masks(sliding_window_num_blocks)

        x = x0 = norm(self.embed(input_seq)[None]) # use of norm here by @Grad62304977
        ve = self.value_embeds(input_seq)
        assert len(ve) == len(self.blocks)
        ve_enc, ve_dec = ve[:self.num_encoder_layers], ve[self.num_encoder_layers:]
        assert len(ve_enc) == self.num_encoder_layers and len(ve_dec) == self.num_decoder_layers

        # Store outputs for U-Net skip connections
        skip_connections = []
        # Encoder pass - process only the first half of the blocks
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm]
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0, block_masks[i])
            skip_connections.append(x)
        # Decoder pass - process the remaining blocks with weighted skip connections
        block_masks.reverse()
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0, block_masks[i])
        x = norm(x)
        logits = lm_head_fp8(x, self.lm_head.weight) if self.training else self.lm_head(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15, @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1)
        logits = 30 * torch.sigmoid(logits.float() / 7.5)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq)
        return loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(f"{file}", False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank : int, world_size : int):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files) # use itertools.cycle(files) instead if you want to do multi-epoch training
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn"t helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin" # input .bin to train on
    val_files = "data/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    # optimization
    batch_size = 8*64*1024 # batch size in tokens
    num_iterations = 7500 # number of iterations to run
    cooldown_frac = 0.4 # fraction of training spent cooling down the learning rate
    # optimizer settings
    optimizer_lr = 0.0003
    warmup_steps = 0
    min_lr_frac = 0.1
    optimizer_momentum = 0.9
    optimizer_ns_steps = 5
    optimizer_weight_decay = 0.3
    max_size_triangular = 10000
    memory_save_mode = None
    precond_lr = 0.5
    precond_init_scale = 1.0
    partition_grads = False
    block_size = 1024
    clip_update_rms = True
    optimizer_dtype = torch.bfloat16
    flat_start = 500
    update_prob = 1 / 10
    # adam optimizer settings
    head_lr = 0.003
    embed_lr = 0.3
    scalar_lr = 0.015
    adam_beta1 = 0.8
    adam_beta2 = 0.95
    adam_eps = 1e-10
    # evaluation and logging
    val_loss_every = 125 # every how many steps to evaluate val loss? 0 for only at the end
    wandb_project = "nanogpt"
    # implementation
    seq_len = 64*1024 # FlexAttention sequence length
    save_checkpoint = False
args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{run_id}.txt"
    print(logfile)
    wandb.init(
        project=args.wandb_project,
        config={
            # data and training
            "batch_size": args.batch_size,
            "num_iterations": args.num_iterations,
            "cooldown_frac": args.cooldown_frac,
            "val_loss_every": args.val_loss_every,
            "seq_len": args.seq_len,
            "val_tokens": args.val_tokens,
            # model architecture
            "vocab_size": 50257,
            "num_layers": 16,
            "num_heads": 8,
            "model_dim": 1024,
            # optimizer
            "optimizer_lr": args.optimizer_lr,
            "warmup_steps": args.warmup_steps,
            "min_lr_frac": args.min_lr_frac,
            "optimizer_momentum": args.optimizer_momentum,
            "optimizer_ns_steps": args.optimizer_ns_steps,
            "optimizer_weight_decay": args.optimizer_weight_decay,
            "max_size_triangular": args.max_size_triangular,
            "memory_save_mode": args.memory_save_mode,
            "precond_lr": args.precond_lr,
            "precond_init_scale": args.precond_init_scale,
            "partition_grads": args.partition_grads,
            "block_size": args.block_size,
            "clip_update_rms": args.clip_update_rms,
            "optimizer_dtype": args.optimizer_dtype,
            "flat_start": args.flat_start,
            "update_prob": args.update_prob,
            # adam optimizer
            "head_lr": args.head_lr,
            "embed_lr": args.embed_lr,
            "scalar_lr": args.scalar_lr,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "adam_eps": args.adam_eps,
            # hardware
            "world_size": world_size,
        }
    )
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
             if console:
                 print(s)
             print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

# load data
train_loader = distributed_data_generator(args.train_files, args.batch_size, rank, world_size)

model = GPT(vocab_size=50257, num_layers=16, num_heads=8, model_dim=1024).cuda()
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()
for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# collect the parameters to optimize
# hidden_matrix_params = [p for p in model.blocks.parameters() if p.ndim >= 2]
# embed_params = [model.embed.weight, *model.value_embeds.parameters()]
# scalar_params = [p for p in model.parameters() if p.ndim < 2]
# head_params = [model.lm_head.weight]

# init the optimizer(s)
# adam_params = [
#     dict(params=head_params, lr=args.head_lr),
#     dict(params=embed_params, lr=args.embed_lr),
#     dict(params=scalar_params, lr=args.scalar_lr)
# ]
# small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
# discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
# optimizer1 = torch.optim.Adam(adam_params, betas=(args.adam_beta1, args.adam_beta2), fused=True, eps=args.adam_eps)
optimizer2 = Kron(
    model.parameters(),
    lr=args.optimizer_lr,
    b1=args.optimizer_momentum,
    weight_decay=args.optimizer_weight_decay,
    preconditioner_update_probability=precond_update_prob_schedule(flat_start=args.flat_start, min_prob=args.update_prob),
    max_size_triangular=args.max_size_triangular,
    memory_save_mode=args.memory_save_mode,
    precond_lr=args.precond_lr,
    precond_init_scale=args.precond_init_scale,
    partition_grads=args.partition_grads,
    block_size=args.block_size,
    clip_update_rms=args.clip_update_rms,
    dtype=args.optimizer_dtype,
    rank=rank,
    world_size=world_size
)
optimizers = [optimizer2]

# learning rate schedule: stable then decay
def get_lr(it: int):
    cooldown_start = args.num_iterations - int(args.num_iterations * args.cooldown_frac)
    if it < args.warmup_steps:
        return it / args.warmup_steps
    elif it >= cooldown_start:
        t = (it - cooldown_start) / (args.num_iterations - cooldown_start)
        return 1.0 - t * (1.0 - args.min_lr_frac)
    else:
        return 1.0

# linear
# def get_lr(it: int):
#     progress = min(it / args.num_iterations, 1.0)
#     return 1.0 - progress * (1.0 - args.min_lr_frac)

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]
@lru_cache(1)
def sw_num_blks(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

model: nn.Module = torch.compile(model)
training_time_ms = 0
train_log_freq = 20
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
start_time = time.time()
time_last_n = 0
step_time = 0
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)
    # This effectively ignores timing first 10 steps, which are slower for weird reasons.
    # Alternately, and slightly more correctly in terms of benchmarking, we could do 10
    # steps with dummy data first, and then re-initialize the model and reset the loader.
    if step == 10:
        training_time_ms = 0
        t0 = time.perf_counter()
    timed_steps = float("nan") if step <= 11 else (step - 10) + 1 # <= 11 to avoid bug in val

    # Linearly increase the block-wise sliding window size over training 128 -> 1792:
    # increase by @fernbear.bsky.social; block-wise by @YouJiacheng
    window_size = next_multiple_of_n(1728 * step / train_steps, n=128)
    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_bs = world_size * args.seq_len
        assert args.val_tokens % val_bs == 0
        val_steps = args.val_tokens // val_bs
        val_loader = distributed_data_generator(args.val_files, val_bs, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for _ in range(val_steps):
                x, y = next(val_loader)
                val_loss += model(x, y, sw_num_blks(window_size))
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms", console=True)
        if master_process:
            to_log = {"val_loss": val_loss.item(), "step": step, "train_time_ms": training_time_ms}
            if timed_steps > 1:
                to_log["avg_step_time_ms"] = training_time_ms/(timed_steps-1)
            wandb.log(to_log, step=step)
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    inputs, targets = next(train_loader)
    for input_seq, target_seq in zip(inputs.split(args.seq_len), targets.split(args.seq_len)):
        loss = model(input_seq, target_seq, sw_num_blks(window_size))
        loss.backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # momentum warmup
    # frac = min(step / 300, 1)
    # for group in optimizer2.param_groups:
    #     group["b1"] = (1 - frac) * 0.85 + frac * args.optimizer_momentum
    # step the optimizers and schedulers
    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    # null the gradients
    model.zero_grad(set_to_none=True)
    # logging
    if step % train_log_freq == 0 and master_process:
        current_lr_multiplier = get_lr(step)
        to_log = {"loss": loss.item(), "lr_multiplier": current_lr_multiplier}
        time_last_n = time.time() - start_time
        step_time = time_last_n * 1000 / train_log_freq
        if step > 1:
            to_log["step_time_ms"] = step_time
        wandb.log(to_log, step=step)
        start_time = time.time()
    approx_time = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_time:.0f}ms step_time:{step_time:.2f}ms step_avg:{approx_time/timed_steps:.2f}ms loss:{loss:.4f}", console=True)

print0(
    f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
    f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
)
if master_process:
    wandb.finish()
dist.destroy_process_group()
