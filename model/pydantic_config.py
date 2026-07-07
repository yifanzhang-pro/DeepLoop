from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import types
from typing import Any, Annotated, Union, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt, StrictStr, create_model

_L2NORM_KEY_ALIASES: tuple[tuple[str, str], ...] = (
    ("ql2norm", "q_l2norm"),
    ("kl2norm", "k_l2norm"),
    ("vl2norm", "v_l2norm"),
)
_L2NORM_KEYS: tuple[str, ...] = tuple(canonical for _, canonical in _L2NORM_KEY_ALIASES)
_L2NORM_INPUT_KEYS: tuple[str, ...] = _L2NORM_KEYS + tuple(alias for alias, _ in _L2NORM_KEY_ALIASES)
_RMSNORM_KEYS: tuple[str, ...] = ("q_rmsnorm", "k_rmsnorm", "v_rmsnorm")


def _to_pydantic_type(py_type: Any) -> Any:
    origin = get_origin(py_type)
    if origin is None:
        if py_type is bool:
            return StrictBool
        if py_type is int:
            return StrictInt
        if py_type is str:
            return StrictStr
        if py_type is float:
            # Int -> float coercion is convenient in configs (e.g. dropout=0).
            return float
        return py_type

    args = get_args(py_type)
    if not args:
        return py_type

    if origin is Annotated:
        return _to_pydantic_type(args[0])

    if origin in (Union, types.UnionType):
        mapped_args = tuple(_to_pydantic_type(arg) for arg in args)
        return Union[mapped_args]

    if origin is list:
        return list[_to_pydantic_type(args[0])]
    if origin is set:
        return set[_to_pydantic_type(args[0])]
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple[_to_pydantic_type(args[0]), ...]
        return tuple[tuple(_to_pydantic_type(arg) for arg in args)]
    if origin is dict:
        return dict[_to_pydantic_type(args[0]), _to_pydantic_type(args[1])]

    return py_type


@lru_cache(maxsize=None)
def _build_config_model(config_cls: type[object]) -> type[BaseModel]:
    create_dynamic_model = cast(Any, create_model)
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not raw_annotations:
        return cast(
            type[BaseModel],
            create_dynamic_model(
                f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
                __config__=ConfigDict(extra="allow"),
            ),
        )

    resolved = get_type_hints(config_cls, include_extras=True)
    fields: dict[str, tuple[Any, object]] = {}
    for name in raw_annotations:
        if name not in resolved:
            continue
        default = getattr(config_cls, name, ...)
        fields[name] = (_to_pydantic_type(resolved[name]), default)

    return cast(
        type[BaseModel],
        create_dynamic_model(
            f"{config_cls.__module__}.{config_cls.__qualname__}ConfigModel",
            __config__=ConfigDict(extra="allow"),
            **fields,
        ),
    )


def _apply_l2norm_key_aliases(config_cls: type[object], values: dict[str, Any]) -> None:
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not isinstance(raw_annotations, dict):
        return

    for alias, canonical in _L2NORM_KEY_ALIASES:
        if canonical not in raw_annotations:
            continue
        if alias not in values:
            continue
        if canonical in values:
            raise ValueError(
                f"Conflicting GPT config keys: both {alias!r} and {canonical!r} were provided; set only one."
            )
        values[canonical] = values.pop(alias)


def _is_mha_deltanet_module(config_cls: type[object]) -> bool:
    module = getattr(config_cls, "__module__", "")
    if not isinstance(module, str):
        return False
    module_stem = module.rsplit(".", maxsplit=1)[-1]
    return module_stem.startswith("gpt-mha-delta_net")


def _is_fla_module(config_cls: type[object]) -> bool:
    module = getattr(config_cls, "__module__", "")
    if not isinstance(module, str):
        return False
    module_stem = module.rsplit(".", maxsplit=1)[-1]
    return "_fla" in module_stem


def _config_declares_any_key(config_cls: type[object], keys: tuple[str, ...]) -> bool:
    raw_annotations = config_cls.__dict__.get("__annotations__", {})
    if not isinstance(raw_annotations, dict):
        return False
    return any(key in raw_annotations for key in keys)


def validate_pretrained_config_kwargs(config_cls: type[object], values: Mapping[str, Any]) -> dict[str, Any]:
    """
    Validate a Hugging Face `PretrainedConfig` kwarg dict using a generated Pydantic model.

    Notes:
    - Known config fields (declared via type annotations on `config_cls`) are validated.
    - Extra keys are allowed and passed through unchanged, mirroring `PretrainedConfig` behavior.
    """
    values_dict = dict(values)

    if _config_declares_any_key(config_cls, _RMSNORM_KEYS):
        forbidden_l2norm_keys = [key for key in _L2NORM_INPUT_KEYS if key in values_dict]
        if len(forbidden_l2norm_keys) == 1:
            legacy_key = forbidden_l2norm_keys[0]
            raise ValueError(f"Unsupported config key: {legacy_key}. Use q_rmsnorm/k_rmsnorm/v_rmsnorm instead.")
        if forbidden_l2norm_keys:
            raise ValueError(
                f"Unsupported config key(s) {forbidden_l2norm_keys}. Use q_rmsnorm/k_rmsnorm/v_rmsnorm instead."
            )

    _apply_l2norm_key_aliases(config_cls, values_dict)

    deprecated_beta_scale_keys = ("delta_allow_neg_eigval", "gated_deltanet_allow_neg_eigval")
    forbidden_deprecated_beta_scale_keys = [key for key in deprecated_beta_scale_keys if key in values_dict]
    if forbidden_deprecated_beta_scale_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_deprecated_beta_scale_keys}; use 'beta_scale_by_2'."
        )

    if _is_mha_deltanet_module(config_cls):
        forbidden_rmsnorm_keys = [key for key in _RMSNORM_KEYS if key in values_dict]
        if forbidden_rmsnorm_keys:
            raise ValueError(
                f"Unsupported GPT config key(s) {forbidden_rmsnorm_keys} for DeltaNet L2-norm variants; "
                "use q_l2norm/k_l2norm/v_l2norm (legacy: ql2norm/kl2norm/vl2norm) instead."
            )

    size_alias_keys = ("n_embd", "n_embd_base", "d_ff_factor")
    forbidden_size_aliases = [key for key in size_alias_keys if key in values_dict]
    if forbidden_size_aliases:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_size_aliases}; "
            "use 'hidden_size' / 'hidden_size_base' / 'mlp_hidden_mult'."
        )

    deprecated_delta_keys = (
        "delta_beta_init",
        "delta_use_short_conv",
        "delta_conv_size",
        "delta_norm_eps",
        "delta_qk_activation",
        "delta_qk_norm",
    )
    forbidden_deprecated_delta_keys = [key for key in deprecated_delta_keys if key in values_dict]
    if forbidden_deprecated_delta_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_deprecated_delta_keys}; "
            "the deprecated DeltaNet `delta_*` knobs were removed. "
            "Use `q_activation`/`k_activation`/`v_activation`, `q_l2norm`/`k_l2norm`/`v_l2norm`, "
            "the shared `shortconv_*` knobs, and `attention_norm_eps` instead."
        )

    legacy_rank_keys = ("q_rank", "rank")
    forbidden_legacy_rank_keys = [key for key in legacy_rank_keys if key in values_dict]
    if forbidden_legacy_rank_keys:
        raise ValueError(
            f"Unsupported GPT config key(s) {forbidden_legacy_rank_keys}; the legacy attention rank knobs were removed."
        )

    if _is_mha_deltanet_module(config_cls) and "delta_use_gate" in values_dict:
        raise ValueError("Unsupported GPT config key(s) ['delta_use_gate']; use 'use_output_gate'.")

    if "fast_weight_lambda" in values_dict and not _is_fla_module(config_cls):
        raise ValueError(
            "Unsupported GPT config key(s) ['fast_weight_lambda']; "
            "use 'fast_weight_lambda_min' and 'fast_weight_lambda_max'."
        )

    if "fast_weight_alpha" in values_dict:
        raise ValueError(
            "Unsupported GPT config key(s) ['fast_weight_alpha']; "
            "use 'fast_weight_alpha_min' and 'fast_weight_alpha_max'."
        )

    if _is_fla_module(config_cls):
        forbidden_fla_lambda_keys = [
            key
            for key in ("fast_weight_lambda", "fast_weight_lambda_min", "fast_weight_lambda_max", "lambda_scale")
            if key in values_dict
        ]
        if forbidden_fla_lambda_keys:
            raise ValueError(
                f"Unsupported GPT config key(s) {forbidden_fla_lambda_keys} for FLA variants; "
                "FLA models do not implement lambda decay."
            )
    model = _build_config_model(config_cls)
    validated = model.model_validate(values_dict)
    return validated.model_dump()
