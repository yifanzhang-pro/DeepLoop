"""Training entry point for the ARC-AGI port of HRM (arXiv:2506.21734).

This is a stripped-down counterpart of ``train_adam_finewebedu.py`` that:

  - reads pre-built fixed-length token sequences from ``data/arc-agi/`` (see
    ``data/arc-agi/prepare.py``) instead of streaming text;
  - constructs (idx, targets) pairs with the input region of each row masked
    via ``ignore_index=-1`` so the standard causal-LM loss in
    ``model/gpt-mha-rope-loop.py`` reduces to next-token prediction over the
    900 output-grid tokens, matching HRM's seq2seq objective;
  - reuses the rest of the training stack (DDP, AdamW, ZeRO-1, cosine LR,
    grad clipping, checkpointing) without modification.

Launch:

    torchrun --standalone --nproc_per_node=4 train_adam_arc_agi.py \\
        config/train_gpt_mha_rope_small_loop3_arc_agi_smoke_80g4.py
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, cast

os.environ.setdefault("WANDB_MODE", "offline")

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load_file as load_safetensors_file
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from arc_agi_dataloader import ArcAgiDataloader
from train_adam_config import ArcAgiTrainConfig, load_train_config
from utils.checkpoint_utils import prune_old_checkpoints, update_best_checkpoint
from utils.dist_utils import broadcast_string
from utils.git_utils import get_checkpoint_commit_mismatch_warning, get_repo_git_commit
from utils.optim_utils import get_optimizer_param_groups
from utils.precision_utils import resolve_train_dtype
from utils.seed_utils import seed_everything
from utils.torch_compile import maybe_torch_compile

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Defaults loaded from the Pydantic config; CLI / config-file overrides applied
# below by ``load_train_config``.
_defaults = ArcAgiTrainConfig.model_construct()

data_path = _defaults.data_path
dataset = _defaults.dataset
out_dir = _defaults.out_dir
resume_dir = _defaults.resume_dir
eval_interval = _defaults.eval_interval
log_interval = _defaults.log_interval
eval_iters = _defaults.eval_iters
eval_only = _defaults.eval_only
save_checkpoints = _defaults.save_checkpoints
always_save_checkpoint = _defaults.always_save_checkpoint
keep_last_checkpoints = _defaults.keep_last_checkpoints
init_from = _defaults.init_from

wandb_log = _defaults.wandb_log
wandb_project = _defaults.wandb_project
wandb_run_name = _defaults.wandb_run_name

batch_size = _defaults.batch_size
gradient_accumulation_steps = _defaults.gradient_accumulation_steps
global_batch_size = _defaults.global_batch_size
block_size = _defaults.block_size

seed = _defaults.seed
deterministic = _defaults.deterministic
bitwise_deterministic = _defaults.bitwise_deterministic
data_seed = _defaults.data_seed
eval_seed = _defaults.eval_seed

num_hidden_layers = _defaults.num_hidden_layers
num_attention_heads = _defaults.num_attention_heads
hidden_size = _defaults.hidden_size
head_dim = _defaults.head_dim
tpa_kvrank = _defaults.tpa_kvrank
tpa_qrank = _defaults.tpa_qrank
dropout = _defaults.dropout
bias = _defaults.bias
using_groupnorm = _defaults.using_groupnorm

mup = _defaults.mup
hidden_size_base = _defaults.hidden_size_base
embedding_lr_multiplier = _defaults.embedding_lr_multiplier

use_k_shift = _defaults.use_k_shift
use_v_shift = _defaults.use_v_shift

embedding_init_std = _defaults.embedding_init_std
hidden_init_std_factor = _defaults.hidden_init_std_factor
use_qk_rmsnorm = _defaults.use_qk_rmsnorm
rope_ratio = _defaults.rope_ratio
p_tie_mode = _defaults.p_tie_mode
p_head_dim = _defaults.p_head_dim

block_repeats = _defaults.block_repeats
use_input_embed_rmsnorm = _defaults.use_input_embed_rmsnorm
use_loop_deepnorm = _defaults.use_loop_deepnorm
loop_deepnorm_alpha = _defaults.loop_deepnorm_alpha
loop_deepnorm_beta = _defaults.loop_deepnorm_beta
loop_deepnorm_p = _defaults.loop_deepnorm_p
use_shared_kv_activations = _defaults.use_shared_kv_activations
use_act = _defaults.use_act
act_max_steps = _defaults.act_max_steps
act_threshold = _defaults.act_threshold
act_ponder_lambda = _defaults.act_ponder_lambda
loss_type = _defaults.loss_type
lm_head_tau = _defaults.lm_head_tau

optimizer_name = _defaults.optimizer_name
learning_rate_base = _defaults.learning_rate_base
max_iters = _defaults.max_iters
weight_decay = _defaults.weight_decay
beta1 = _defaults.beta1
beta2 = _defaults.beta2
grad_clip = _defaults.grad_clip
zero_stage = _defaults.zero_stage

decay_lr = _defaults.decay_lr
warmup_iters = _defaults.warmup_iters
lr_decay_iters = _defaults.lr_decay_iters
min_lr_base = _defaults.min_lr_base

backend = _defaults.backend
ddp_timeout_seconds = _defaults.ddp_timeout_seconds
schedule = _defaults.schedule

model_type = _defaults.model_type
num_key_value_heads = _defaults.num_key_value_heads
delta_rule_triton_dtype = _defaults.delta_rule_triton_dtype
gated_delta_rule_triton_dtype = _defaults.gated_delta_rule_triton_dtype

device = _defaults.device
dtype = _defaults.dtype
compile = _defaults.compile  # noqa: A001 (mirrors finewebedu naming)
scale_attn_by_inverse_layer_idx = _defaults.scale_attn_by_inverse_layer_idx

_config, extra_config = load_train_config(ArcAgiTrainConfig, sys.argv[1:])
config = _config.model_dump()
for k, v in config.items():
    globals()[k] = v
configured_wandb_run_name = str(wandb_run_name)
if deterministic:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# -----------------------------------------------------------------------------
model_file = importlib.import_module(f"model.{model_type}")
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT
_ALLOWED_MODEL_MISSING_KEYS: Final[frozenset[str]] = frozenset({"lm_head.weight"})


def _get_num_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get("SLURM_JOB_ID", "0")

ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    print(
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, "
        f"RANK: {os.environ.get('RANK')}, "
        f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}"
    )
    init_process_group(backend=backend, timeout=timedelta(seconds=int(ddp_timeout_seconds)))
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_rank = 0
    world_size = 1

codebase_git_commit = get_repo_git_commit()

requested_global_batch_size: int | None = int(global_batch_size) if global_batch_size is not None else None
if requested_global_batch_size is not None:
    denom = int(batch_size) * int(world_size)
    if requested_global_batch_size % denom != 0:
        raise ValueError(
            f"global_batch_size={requested_global_batch_size} must be divisible by "
            f"batch_size({int(batch_size)}) * world_size({int(world_size)}) = {denom}."
        )
    gradient_accumulation_steps = requested_global_batch_size // denom

effective_global_batch_size: int = int(batch_size) * int(gradient_accumulation_steps) * int(world_size)
if master_process and requested_global_batch_size is not None:
    print(
        "Derived gradient_accumulation_steps from global_batch_size: "
        f"global_batch_size={effective_global_batch_size} batch_size={int(batch_size)} "
        f"world_size={int(world_size)} gradient_accumulation_steps={int(gradient_accumulation_steps)}"
    )

if zero_stage not in (0, 1):
    raise ValueError(f"zero_stage must be 0 or 1, got {zero_stage}")
if zero_stage == 1 and not ddp:
    raise ValueError("zero_stage=1 requires DDP (launch with torchrun)")

# Each row contributes block_size next-token predictions, of which the input
# region is masked. This is the right unit for "tokens per iter" telemetry.
tokens_per_iter = effective_global_batch_size * block_size
total_tokens_B = tokens_per_iter * max_iters / (1000**3)

tokens_trained = 0

seed_everything(seed + seed_offset, deterministic=deterministic)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = not bitwise_deterministic
    torch.backends.cudnn.allow_tf32 = not bitwise_deterministic
if bitwise_deterministic:
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
device_type = torch.device(device).type
dtype = resolve_train_dtype(dtype, device=torch.device(device), log=(print if master_process else None))
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" or dtype == "float32" else torch.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# ARC-AGI dataloaders.
data_dir = os.path.join(data_path, dataset)
train_loader = ArcAgiDataloader(
    data_dir,
    split="train",
    batch_size=batch_size,
    device=device,
    seed=data_seed,
    rank=ddp_rank,
    world_size=world_size,
    shuffle=True,
)
val_loader = ArcAgiDataloader(
    data_dir,
    split="val",
    batch_size=batch_size,
    device=device,
    seed=eval_seed,
    rank=ddp_rank,
    world_size=world_size,
    shuffle=False,
)
if int(train_loader.block_size) != int(block_size):
    raise ValueError(
        f"Config block_size={int(block_size)} does not match dataset block_size={int(train_loader.block_size)}."
    )

meta_vocab_size = train_loader.vocab_size

# -----------------------------------------------------------------------------
# Model.
model_args: dict[str, Any] = {
    "num_hidden_layers": num_hidden_layers,
    "num_attention_heads": num_attention_heads,
    "hidden_size": hidden_size,
    "block_size": block_size,
    "bias": bias,
    "head_dim": head_dim,
    "tpa_kvrank": tpa_kvrank,
    "tpa_qrank": tpa_qrank,
    "using_groupnorm": using_groupnorm,
    "vocab_size": meta_vocab_size,
    "dropout": dropout,
    "scale_attn_by_inverse_layer_idx": scale_attn_by_inverse_layer_idx,
    "embedding_init_std": embedding_init_std,
    "hidden_init_std_factor": hidden_init_std_factor,
    "use_qk_rmsnorm": use_qk_rmsnorm,
    "rope_ratio": rope_ratio,
    "use_k_shift": use_k_shift,
    "use_v_shift": use_v_shift,
    "p_tie_mode": p_tie_mode,
    "p_head_dim": p_head_dim,
    "mup": mup,
    "hidden_size_base": hidden_size_base,
    "embedding_lr_multiplier": embedding_lr_multiplier,
    "delta_rule_triton_dtype": delta_rule_triton_dtype,
    "gated_delta_rule_triton_dtype": gated_delta_rule_triton_dtype,
    "use_input_embed_rmsnorm": use_input_embed_rmsnorm,
    "use_loop_deepnorm": use_loop_deepnorm,
    "loop_deepnorm_alpha": loop_deepnorm_alpha,
    "loop_deepnorm_beta": loop_deepnorm_beta,
    "loop_deepnorm_p": loop_deepnorm_p,
    "use_shared_kv_activations": use_shared_kv_activations,
    "use_act": use_act,
    "act_max_steps": act_max_steps,
    "act_threshold": act_threshold,
    "act_ponder_lambda": act_ponder_lambda,
    "loss_type": loss_type,
}
if block_repeats is not None:
    model_args["block_repeats"] = block_repeats
if lm_head_tau is not None:
    model_args["lm_head_tau"] = lm_head_tau
if "gqa" in model_type:
    model_args["num_key_value_heads"] = num_key_value_heads
for key, value in extra_config.items():
    if key not in model_args:
        model_args[key] = value

try:
    _gptconf_sig = inspect.signature(GPTConfig.__init__)
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in _gptconf_sig.parameters.values()):
        _gptconf_params = set(_gptconf_sig.parameters)
        _gptconf_params.discard("self")
        model_args = {k: v for k, v in model_args.items() if k in _gptconf_params}
except (TypeError, ValueError):
    pass

iter_num = 0
best_val_loss = 1e9
resume_checkpoint_dir: str | None = None
resume_run_dir: str | None = None

if init_from == "scratch":
    if master_process:
        print(f"Initializing a new model from scratch with vocab_size={meta_vocab_size}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    if master_process:
        print(f"Resuming from {resume_dir}")
    base = os.path.basename(resume_dir.rstrip("/"))
    if base.startswith("checkpoint-"):
        resume_checkpoint_dir = resume_dir
    else:
        candidates = sorted(
            (int(name.partition("-")[2]), os.path.join(resume_dir, name))
            for name in os.listdir(resume_dir)
            if name.startswith("checkpoint-") and name.partition("-")[2].isdigit()
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* under {resume_dir!r}")
        resume_checkpoint_dir = candidates[-1][1]
    resume_run_dir = os.path.dirname(resume_checkpoint_dir.rstrip("/"))
    cfg = GPTConfig.from_json_file(os.path.join(resume_checkpoint_dir, "config.json"))
    model = GPT(cfg)
    state_dict = load_safetensors_file(os.path.join(resume_checkpoint_dir, "model.safetensors"), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in load_result.missing_keys if k not in _ALLOWED_MODEL_MISSING_KEYS]
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={missing} unexpected={list(load_result.unexpected_keys)}"
        )
    for k in ("num_hidden_layers", "num_attention_heads", "hidden_size", "block_size", "bias", "vocab_size"):
        model_args[k] = getattr(cfg, k)
    model.tie_weights()
else:
    raise ValueError(f"Unsupported init_from={init_from!r}")

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args["block_size"] = block_size

model.config.tokenizer = "arc-agi"
model.config.nanogpt_next_model_type = model_type
model.to(device)

param_count_m = _get_num_params(model) / 1_000_000

generated_wandb_run_name = configured_wandb_run_name
if init_from != "resume":
    generated_wandb_run_name = (
        f"M_{model_type}_{int(param_count_m)}m_Opt_{optimizer_name}_LR_{learning_rate_base}"
        f"_D_{dataset}_T_{total_tokens_B:.4f}B_time_{current_date}_jobid_{job_id}"
    )
    run_dir_name = (
        f"out_arc_{model_type}_{int(param_count_m)}m_LR_{learning_rate_base}"
        f"_D_{dataset}_time_{current_date}_jobid_{job_id}"
    )
    normalized = os.path.normpath(str(out_dir))
    if normalized == os.path.normpath(_defaults.out_dir):
        out_dir = os.path.join("output", run_dir_name)
    elif not os.path.basename(normalized).startswith("out_"):
        out_dir = os.path.join(normalized, run_dir_name)
elif resume_run_dir is not None and out_dir == _defaults.out_dir:
    out_dir = resume_run_dir


def _resolve_wandb_run_name(*, configured: str, default: str, generated: str) -> str:
    slurm_script = os.environ.get("DEEPLOOP_SLURM_SCRIPT_NAME")
    if slurm_script:
        return slurm_script
    return configured if configured != default else generated


wandb_run_name = _resolve_wandb_run_name(
    configured=configured_wandb_run_name,
    default=str(_defaults.wandb_run_name),
    generated=generated_wandb_run_name,
)

if ddp and init_from != "resume":
    comm_device = device if backend == "nccl" else "cpu"
    out_dir = broadcast_string(out_dir, src=0, device=comm_device)
    wandb_run_name = broadcast_string(wandb_run_name, src=0, device=comm_device)
if master_process:
    os.makedirs(out_dir, exist_ok=True)
if ddp:
    dist.barrier()

# -----------------------------------------------------------------------------
# Optimizer and (no-op for bf16) scaler.
if TYPE_CHECKING:
    from torch.amp.grad_scaler import GradScaler as _GradScaler  # noqa: F401

    scaler = _GradScaler(device=device_type, enabled=(dtype == "float16" and device_type == "cuda"))
else:
    scaler = torch.amp.GradScaler(device=device_type, enabled=(dtype == "float16" and device_type == "cuda"))

mup = bool(getattr(model.config, "mup", mup))
hidden_size_base = int(getattr(model.config, "hidden_size_base", hidden_size_base))
embedding_lr_multiplier = float(getattr(model.config, "embedding_lr_multiplier", embedding_lr_multiplier))
model_hidden_size = float(getattr(model.config, "hidden_size", hidden_size))
hidden_lr_mult = (float(hidden_size_base) / model_hidden_size) if mup else 1.0
embedding_lr_mult = float(embedding_lr_multiplier) if mup else 1.0

param_groups = get_optimizer_param_groups(
    model,
    weight_decay,
    split_embeddings=mup,
    embedding_lr_mult=embedding_lr_mult,
    hidden_lr_mult=hidden_lr_mult,
)
param_groups_for_optimizer = cast(list[dict[str, Any]], param_groups)
use_zero1 = zero_stage == 1
if optimizer_name == "adam_atan2":
    # HRM's optimizer (per their config). exp-free Adam with atan2-based update.
    # Falls back to AdamW if adam_atan2 C-extension isn't available (login-node build limitation).
    try:
        from adam_atan2 import AdamATan2  # noqa: F811
    except (ImportError, ModuleNotFoundError):
        print("WARNING: adam_atan2 not importable; falling back to AdamW", flush=True)
        AdamATan2 = AdamW  # type: ignore
    if use_zero1:
        optimizer = ZeroRedundancyOptimizer(
            param_groups_for_optimizer,
            optimizer_class=AdamATan2,
            lr=learning_rate_base,
            betas=(beta1, beta2),
        )
    else:
        optimizer = AdamATan2(param_groups_for_optimizer, lr=learning_rate_base, betas=(beta1, beta2))
elif use_zero1:
    optimizer = ZeroRedundancyOptimizer(
        param_groups_for_optimizer,
        optimizer_class=AdamW,
        lr=learning_rate_base,
        betas=(beta1, beta2),
        eps=1e-8,
    )
else:
    optimizer = AdamW(param_groups_for_optimizer, lr=learning_rate_base, betas=(beta1, beta2), eps=1e-8)

if init_from == "resume":
    assert resume_checkpoint_dir is not None
    trainer_state_path = os.path.join(resume_checkpoint_dir, "trainer_state.json")
    with open(trainer_state_path, encoding="utf-8") as f:
        trainer_state = json.load(f)
    iter_num = int(trainer_state["global_step"])
    best_val_loss = float(trainer_state.get("best_val_loss", best_val_loss))
    tokens_trained = int(trainer_state.get("tokens_trained", iter_num * int(tokens_per_iter)))
    warning = get_checkpoint_commit_mismatch_warning(
        current_commit=codebase_git_commit,
        checkpoint_commit=trainer_state.get("codebase_git_commit"),
        checkpoint_dir=resume_checkpoint_dir,
    )
    if master_process and warning is not None:
        print(warning)
    optimizer_state = torch.load(
        os.path.join(resume_checkpoint_dir, "optimizer.pt"),
        map_location=("cpu" if use_zero1 else device),
        weights_only=False,
    )
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state

if compile:
    model = maybe_torch_compile(model, enabled=True, log=(print if master_process else None))
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss() -> dict[str, float]:
    out: dict[str, float] = {}
    eval_model = model.module if ddp else model
    eval_model.eval()
    train_loader.reset_eval()
    val_loader.reset_eval()
    for split, loader in (("train", train_loader), ("val", val_loader)):
        loader.reset_eval()
        losses: list[float] = []
        for _ in range(eval_iters):
            batch = loader.next_eval_batch()
            if batch is None:
                break
            x, y = batch
            with ctx:
                _, loss = eval_model(x, y)
            losses.append(loss.item())
        if losses:
            out[split] = float(sum(losses) / len(losses))
        else:
            out[split] = float("nan")
    eval_model.train()
    return out


def get_lr(it: int, schedule: str = "cosine") -> float:
    if it < warmup_iters:
        return learning_rate_base * it / max(1, warmup_iters)
    if it >= lr_decay_iters:
        return min_lr_base
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr_base + coeff * (learning_rate_base - min_lr_base)


# -----------------------------------------------------------------------------
# Training loop.
if wandb_log and master_process:
    import wandb

    wandb_config = {
        "model_args": model_args,
        "training_args": {
            "batch_size": batch_size,
            "global_batch_size": effective_global_batch_size,
            "world_size": world_size,
            "block_size": block_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "tokens_per_iter": tokens_per_iter,
            "max_iters": max_iters,
            "lr_decay_iters": lr_decay_iters,
            "eval_interval": eval_interval,
            "eval_iters": eval_iters,
            "log_interval": log_interval,
            "dataset": dataset,
            "num_train_examples": train_loader.num_examples,
            "num_val_examples": val_loader.num_examples,
        },
        "optimizer_args": {
            "optimizer_name": optimizer_name,
            "learning_rate_base": learning_rate_base,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "grad_clip": grad_clip,
            "decay_lr": decay_lr,
            "warmup_iters": warmup_iters,
            "min_lr_base": min_lr_base,
            "schedule": schedule,
        },
    }
    wandb.init(project=wandb_project, name=wandb_run_name, config=wandb_config)


X, Y = train_loader.next_batch()
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0
clip_time = 0
total_norm_for_log = torch.tensor(0.0)

while True:
    lr = get_lr(iter_num, schedule=schedule) if decay_lr else learning_rate_base
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr * param_group.get("lr_mult", 1.0)

    if eval_only or iter_num % eval_interval == 0:
        save_latest = False
        val_improved = False
        if master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses.get('train', float('nan')):.4f}, val loss {losses.get('val', float('nan')):.4f}")
            if wandb_log:
                wandb.log({"iter": iter_num, "train/loss": losses.get("train"), "val/loss": losses.get("val"), "lr": lr}, step=iter_num)
            val_improved = losses.get("val", float("inf")) < best_val_loss
            if val_improved:
                best_val_loss = losses["val"]
            if iter_num > 0 and (val_improved or always_save_checkpoint):
                save_latest = True

        if ddp:
            comm_device = device if backend == "nccl" else "cpu"
            flag = torch.tensor([1 if save_latest else 0], device=comm_device)
            dist.broadcast(flag, src=0)
            save_latest = bool(flag.item())

        save_snapshot = iter_num % (eval_interval * 5) == 0
        save_checkpoint = save_checkpoints and (save_latest or save_snapshot)
        if save_checkpoint:
            optimizer_state_dict = None
            if use_zero1:
                cast(ZeroRedundancyOptimizer, optimizer).consolidate_state_dict(to=0)
            if master_process:
                optimizer_state_dict = optimizer.state_dict()
                ckpt_dir = os.path.join(out_dir, f"checkpoint-{iter_num}")
                print(f"saving checkpoint to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                raw_model.save_pretrained(ckpt_dir)
                torch.save(optimizer_state_dict, os.path.join(ckpt_dir, "optimizer.pt"))
                trainer_state = {
                    "global_step": iter_num,
                    "best_val_loss": best_val_loss,
                    "tokens_trained": tokens_trained,
                    "codebase_git_commit": codebase_git_commit,
                }
                with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=2, sort_keys=True)
                    f.write("\n")
                if val_improved:
                    update_best_checkpoint(run_dir=out_dir, checkpoint_dir=ckpt_dir)
                if keep_last_checkpoints is not None:
                    prune_old_checkpoints(run_dir=out_dir, keep_last=int(keep_last_checkpoints))
            if ddp:
                dist.barrier()

    if eval_only:
        break
    if iter_num >= max_iters:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            cast(Any, model).require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = train_loader.next_batch()
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm_for_log = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if total_norm_for_log.item() > grad_clip:
            clip_time += 1
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    tokens_trained += tokens_per_iter

    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        tokens_per_sec_M = (tokens_per_iter / dt) / 1_000_000
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, "
            f"mfu {running_mfu * 100:.2f}%, tps (M) {tokens_per_sec_M:.2f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": lossf,
                    "lr": lr,
                    "train/clip_rate": clip_time / (iter_num + 1),
                    "train/grad_norm": total_norm_for_log.item() if grad_clip != 0.0 else 0.0,
                    "train/iter_time_ms": dt * 1000,
                    "train/mfu": running_mfu * 100,
                    "train/tokens_per_sec_M": tokens_per_sec_M,
                },
                step=iter_num,
            )

    iter_num += 1
    local_iter_num += 1

if ddp:
    destroy_process_group()
