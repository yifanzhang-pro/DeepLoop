from __future__ import annotations

from ast import literal_eval
from collections.abc import Mapping, Sequence
import sys
from tokenize import open as open_source
from typing import Literal, TypeVar

try:
    import pydantic
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency: pydantic>=2. Install with `uv pip install -e '.[dev]'` "
        "(recommended) or `pip install 'pydantic>=2.12.5,<3'`."
    ) from exc

try:
    from pydantic import (
        BaseModel,
        ConfigDict,
        NonNegativeInt,
        PositiveInt,
        StrictBool,
        StrictInt,
        StrictStr,
        model_validator,
    )
except ImportError as exc:  # pragma: no cover
    version = getattr(pydantic, "__version__", "unknown")
    pydantic_path = getattr(pydantic, "__file__", "<unknown>")
    raise RuntimeError(
        "nanogpt-next requires Pydantic v2 (`model_validator`, `model_dump`, etc.). "
        f"Found pydantic=={version} from {pydantic_path} under {sys.executable}. "
        "Fix by activating the repo venv and reinstalling deps (recommended): "
        "`uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e '.[dev]'`, "
        "or upgrade in-place: `pip install -U 'pydantic>=2.12.5,<3'`."
    ) from exc

_ConfigModelT = TypeVar("_ConfigModelT", bound=BaseModel)
_CONFIG_OBJECT_KEYS: tuple[str, ...] = ("CONFIG", "config", "TRAIN_CONFIG")
_LEGACY_CONFIG_KEYS: tuple[str, ...] = ("n_embd", "n_embd_base")
_FLA_FORBIDDEN_KEYS: tuple[str, ...] = (
    "fast_weight_lambda",
    "fast_weight_lambda_min",
    "fast_weight_lambda_max",
    "lambda_scale",
)


def _is_fla_model_type(model_type: object) -> bool:
    return isinstance(model_type, str) and "_fla" in model_type


def _parse_override_value(raw: str) -> object:
    try:
        return literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


def _is_jsonish_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_is_jsonish_value(item) for item in value)
    if isinstance(value, tuple):
        return all(_is_jsonish_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonish_value(v) for k, v in value.items())
    return False


def _apply_config_object_overrides(env: dict[str, object]) -> None:
    for key in _CONFIG_OBJECT_KEYS:
        if key not in env:
            continue
        obj = env[key]

        if isinstance(obj, BaseModel):
            overrides: Mapping[str, object] = obj.model_dump(exclude_unset=True)
        elif isinstance(obj, Mapping):
            overrides = obj
        else:
            raise TypeError(f"{key} must be a Pydantic BaseModel or a Mapping[str, object], got {type(obj).__name__}.")

        for override_key, override_value in overrides.items():
            if not isinstance(override_key, str):
                raise TypeError(f"{key} must only contain str keys, got {type(override_key).__name__}.")
            if not _is_jsonish_value(override_value):
                raise TypeError(f"{key}[{override_key!r}] must be JSON-ish, got {type(override_value).__name__}.")
            env[override_key] = override_value


def _reject_legacy_keys(env: Mapping[str, object], *, config_file: str) -> None:
    legacy_keys: list[str] = [key for key in _LEGACY_CONFIG_KEYS if key in env]
    if legacy_keys:
        raise ValueError(
            f"{config_file}: unsupported config key(s) {legacy_keys}; use 'hidden_size' / 'hidden_size_base' instead."
        )

    if "fast_weight_lambda" in env:
        raise ValueError(
            f"{config_file}: unsupported config key(s) ['fast_weight_lambda']; "
            "use 'fast_weight_lambda_min' and 'fast_weight_lambda_max' instead."
        )

    if "fast_weight_alpha" in env:
        raise ValueError(
            f"{config_file}: unsupported config key(s) ['fast_weight_alpha']; "
            "use 'fast_weight_alpha_min' and 'fast_weight_alpha_max' instead."
        )

    if _is_fla_model_type(env.get("model_type")):
        forbidden_fla_keys = [key for key in _FLA_FORBIDDEN_KEYS if key in env]
        if forbidden_fla_keys:
            raise ValueError(
                f"{config_file}: unsupported config key(s) {forbidden_fla_keys} for FLA model variants; "
                "FLA models do not implement lambda decay."
            )


def _exec_config_file(config_file: str, env: dict[str, object], *, print_updates: bool) -> None:
    if print_updates:
        print(f"Overriding config with {config_file}:")
    with open_source(config_file) as f:
        config_text = f.read()
    if print_updates:
        print(config_text)
    env["__file__"] = config_file
    exec(compile(config_text, config_file, "exec"), env)
    _apply_config_object_overrides(env)
    _reject_legacy_keys(env, config_file=config_file)


def load_train_config(
    model_cls: type[_ConfigModelT],
    argv: Sequence[str],
    *,
    print_updates: bool = True,
) -> tuple[_ConfigModelT, dict[str, object]]:
    """
    Load a Pydantic config model using NanoGPT-style config files and CLI overrides.

    Config file conventions:
    - Plain assignments (e.g. `batch_size = 32`) are supported.
    - A config file may also define `CONFIG` (or `config` / `TRAIN_CONFIG`) as either:
        - a Pydantic `BaseModel` instance (only explicitly-set fields are applied), or
        - a `Mapping[str, object]` of overrides.

    Args:
        model_cls: Pydantic model class used for validation.
        argv: Command line arguments (excluding argv[0]).
        print_updates: If True, prints config file contents and applied overrides.

    Returns:
        (config, extra) where:
          - config: validated Pydantic model instance
          - extra: JSON-ish key/value pairs not declared on the model (e.g. model-specific knobs)
    """
    # Use `model_construct()` so derived defaults (e.g. data_seed=-1 -> seed) are
    # computed after config-file/CLI overrides are applied.
    env: dict[str, object] = dict(model_cls.model_construct().model_dump())
    env.setdefault("__name__", "__main__")

    for arg in argv:
        if "=" not in arg:
            if arg.startswith("--"):
                raise ValueError(f"Expected a config file path, got {arg!r}.")
            _exec_config_file(arg, env, print_updates=print_updates)
            continue

        if not arg.startswith("--"):
            raise ValueError(f"Expected an override in the form --name=value, got {arg!r}.")
        key, raw_value = arg[2:].split("=", 1)

        if key not in env:
            raise ValueError(f"Unknown config key: {key}")
        if not _is_jsonish_value(env[key]):
            raise TypeError(f"Unsupported override target {key!r} with type {type(env[key]).__name__}.")

        value = _parse_override_value(raw_value)
        if print_updates:
            print(f"Overriding: {key} = {value}")
        env[key] = value

    model_field_names = set(model_cls.model_fields.keys())
    config_values: dict[str, object] = {key: env[key] for key in model_field_names if key in env}
    extra: dict[str, object] = {
        key: value
        for key, value in env.items()
        if (
            key not in model_field_names
            and key not in _CONFIG_OBJECT_KEYS
            and not key.startswith("_")
            and _is_jsonish_value(value)
        )
    }
    config = model_cls.model_validate(config_values)
    return config, extra


class TrainAdamConfigBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # tokenizer
    tokenizer: Literal["gpt2", "byteoss", "gpt-oss"] = "gpt2"
    byteoss_vocab: Literal["compact", "sparse"] = "compact"

    # I/O
    data_path: StrictStr = "data"
    out_dir: StrictStr = "output/out"
    eval_interval: PositiveInt = 1000
    log_interval: PositiveInt = 1
    eval_iters: PositiveInt = 2000
    eval_only: StrictBool = False
    save_checkpoints: StrictBool = True  # If False, disable all checkpoint saving
    always_save_checkpoint: StrictBool = True
    keep_last_checkpoints: PositiveInt | None = 3
    init_from: StrictStr = "scratch"  # 'scratch' or 'resume'

    # wandb logging
    wandb_log: StrictBool = False
    wandb_project: StrictStr = "nanogpt-pro"
    wandb_run_name: StrictStr = "gpt2"

    # data
    dataset: StrictStr = "openwebtext"
    gradient_accumulation_steps: PositiveInt = 3
    batch_size: PositiveInt = 20
    # If set, training scripts derive `gradient_accumulation_steps` so the effective global batch
    # size (sequences per optimizer step) stays constant across DDP and non-DDP runs.
    global_batch_size: PositiveInt | None = None
    block_size: PositiveInt = 4096
    data_loader: Literal["random_crop", "stream"] = "stream"
    stream_packing: Literal["auto", "static", "dynamic"] = "auto"
    fit_or_cut_threshold: PositiveInt = 100

    # loss masking
    ignore_doc_start_loss: StrictBool = False

    # reproducibility
    seed: StrictInt = 42
    deterministic: StrictBool = False
    bitwise_deterministic: StrictBool = False
    data_seed: StrictInt = -1
    eval_seed: StrictInt = -1
    data_rng_mode: Literal["stateful", "stateless"] = "stateless"

    # model
    num_hidden_layers: PositiveInt = 24
    num_attention_heads: PositiveInt = 8
    hidden_size: PositiveInt = 1024
    head_dim: PositiveInt = 128
    tpa_kvrank: PositiveInt = 2
    tpa_qrank: PositiveInt = 16
    dropout: float = 0.0
    bias: StrictBool = False
    using_groupnorm: StrictBool = False

    # muP
    mup: StrictBool = True
    hidden_size_base: PositiveInt = 1024
    embedding_lr_multiplier: float = 1.0

    # KV shifting
    use_k_shift: StrictBool = False
    use_v_shift: StrictBool = False

    # initialization / normalization knobs
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5
    use_qk_rmsnorm: StrictBool = True
    rope_ratio: float = 1.0
    p_tie_mode: StrictStr = "none"
    p_head_dim: PositiveInt | None = None

    # Looped transformer
    block_repeats: PositiveInt | None = None
    # Input embedding RMSNorm
    use_input_embed_rmsnorm: StrictBool = False
    # LoopDeepNorm
    use_loop_deepnorm: StrictBool = False
    loop_deepnorm_alpha: float | None = None
    loop_deepnorm_beta: float | None = None
    loop_deepnorm_p: float = 0.5
    use_shared_kv_activations: StrictBool = False
    # ACT (Adaptive Computation Time) — sequence-level halting head extends
    # depth past block_repeats up to act_max_steps R-cycles.
    use_act: StrictBool = False
    act_max_steps: PositiveInt = 16
    act_threshold: float = 0.99
    act_ponder_lambda: float = 0.01
    # Loss function: "softmax_ce" (PyTorch CE) or "stablemax_ce" (HRM, fp64 exp-free).
    loss_type: Literal["softmax_ce", "stablemax_ce"] = "softmax_ce"
    # Tying head temperature
    lm_head_tau: float | None = None

    # optimizer
    optimizer_name: Literal["adamw", "adam_atan2"] = "adamw"
    learning_rate_base: float = 2e-3
    max_iters: NonNegativeInt = 100000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # ZeRO (optimizer state sharding)
    zero_stage: Literal[0, 1] = 0

    # learning rate decay settings
    decay_lr: StrictBool = True
    warmup_iters: NonNegativeInt = 2000
    lr_decay_iters: NonNegativeInt = 100000
    min_lr_base: float = 1e-4

    # DDP settings
    backend: StrictStr = "nccl"
    ddp_timeout_seconds: PositiveInt = 3600

    # scheduler
    schedule: Literal["cosine"] = "cosine"

    # model variants
    model_type: StrictStr = "__base_model_placeholder__"
    num_key_value_heads: PositiveInt = 1
    delta_rule_triton_dtype: Literal["auto", "bfloat16", "float32"] = "float32"
    gated_delta_rule_triton_dtype: Literal["auto", "bfloat16", "float32"] = "float32"

    # system
    device: StrictStr = "cuda"
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    compile: StrictBool = True
    scale_attn_by_inverse_layer_idx: StrictBool = True

    @model_validator(mode="after")
    def _apply_derived_defaults(self) -> TrainAdamConfigBase:
        if self.init_from not in ("scratch", "resume"):
            raise ValueError(
                "init_from must be 'scratch' or 'resume' (a directory produced by this repo's "
                f"training scripts), got {self.init_from!r}."
            )
        if self.data_seed < 0:
            self.data_seed = self.seed
        if self.eval_seed < 0:
            self.eval_seed = self.seed
        if self.decay_lr and int(self.lr_decay_iters) < int(self.warmup_iters):
            raise ValueError(
                "lr_decay_iters must be >= warmup_iters when decay_lr=True, got "
                f"warmup_iters={int(self.warmup_iters)} lr_decay_iters={int(self.lr_decay_iters)}."
            )
        if self.bitwise_deterministic:
            self.deterministic = True
            self.compile = False
        if not (0.0 <= float(self.rope_ratio) <= 1.0):
            raise ValueError(f"rope_ratio must be in [0, 1], got {self.rope_ratio}.")
        return self


class ArithmeticsTrainConfig(TrainAdamConfigBase):
    model_config = ConfigDict(extra="forbid")

    dataset: StrictStr = "arithmetics-addition-fixed_digit"
    wandb_log: StrictBool = True
    tokenizer: Literal["gpt2", "byteoss"] = "byteoss"
    data_loader: Literal["random_crop", "stream"] = "stream"
    digit_width: StrictInt = 4

    @model_validator(mode="after")
    def _validate_digit_width(self) -> ArithmeticsTrainConfig:
        if self.digit_width < 1:
            raise ValueError(f"digit_width must be >= 1, got {self.digit_width}.")
        return self


class OpenWebTextTrainConfig(TrainAdamConfigBase):
    data_path: StrictStr = "data"
    dataset: StrictStr = "openwebtext"
    eval_interval: PositiveInt = 2000
    wandb_log: StrictBool = True


class FinewebEduTrainConfig(TrainAdamConfigBase):
    data_path: StrictStr = "data"
    dataset: StrictStr = "fineweb-edu100B"
    resume_dir: StrictStr = "."
    wandb_log: StrictBool = True


class ArcAgiTrainConfig(TrainAdamConfigBase):
    """Config for the ARC-AGI port of HRM (arXiv:2506.21734).

    The dataset is materialized as fixed-length token sequences with vocab=14
    and ``block_size=1801`` (see ``data/arc-agi/prepare.py`` and
    ``arc_agi_dataloader.py``); the streaming-text loader knobs of the base
    class are unused for this entry point.
    """

    data_path: StrictStr = "data"
    dataset: StrictStr = "arc-agi-aug2"
    resume_dir: StrictStr = "."
    block_size: PositiveInt = 1801
    # The streaming-text knobs below have no effect here but are kept on the
    # base schema so the existing ``load_train_config`` machinery accepts them.
    tokenizer: Literal["gpt2", "byteoss", "gpt-oss"] = "gpt2"
    data_loader: Literal["random_crop", "stream"] = "random_crop"
    wandb_log: StrictBool = False
