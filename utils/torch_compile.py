from __future__ import annotations

from collections.abc import Callable
import traceback
from typing import Any, cast

import torch

_LogFn = Callable[[str], None]
_COMPILE_FAILURE_MARKERS: tuple[str, ...] = (
    "torch.compile is not supported",
    "BackendCompilerFailed",
    "backend='inductor' raised",
    "OutOfResources",
    "s0 is not defined",
)


def _enable_dynamo_error_suppression(*, log: _LogFn | None) -> None:
    """Enable eager fallback when Dynamo/Inductor compilation fails."""
    try:
        import torch._dynamo as torch_dynamo
    except Exception:
        return

    torch_dynamo_config = getattr(cast(Any, torch_dynamo), "config", None)
    if torch_dynamo_config is None:
        return

    if bool(getattr(torch_dynamo_config, "suppress_errors", False)):
        return

    setattr(torch_dynamo_config, "suppress_errors", True)
    if log is not None:
        log("enabled torch._dynamo eager fallback (`suppress_errors=True`).")


def _is_known_compile_failure(exc: BaseException) -> bool:
    message = str(exc)
    if any(marker in message for marker in _COMPILE_FAILURE_MARKERS):
        return True
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return any(marker in formatted for marker in _COMPILE_FAILURE_MARKERS)


def _attach_runtime_compile_fallback(
    compiled_model: torch.nn.Module,
    eager_model: torch.nn.Module,
    *,
    log: _LogFn | None,
) -> torch.nn.Module:
    # Some compile failures are raised lazily on first forward/backward.
    # Keep training alive by switching to eager execution after the first known compile failure.
    fallback_to_eager = False
    compiled_forward = compiled_model.forward
    eager_forward = eager_model.forward

    def forward_with_fallback(*args: Any, **kwargs: Any) -> Any:
        nonlocal fallback_to_eager
        if fallback_to_eager:
            return eager_forward(*args, **kwargs)
        try:
            return compiled_forward(*args, **kwargs)
        except Exception as exc:
            if not _is_known_compile_failure(exc):
                raise
            fallback_to_eager = True
            if log is not None:
                log(f"torch.compile runtime failure; switching to eager execution: {exc}")
            return eager_forward(*args, **kwargs)

    setattr(compiled_model, "forward", cast(Any, forward_with_fallback))
    return compiled_model


def maybe_torch_compile(
    model: torch.nn.Module,
    *,
    enabled: bool,
    log: _LogFn | None = print,
) -> torch.nn.Module:
    """
    Best-effort wrapper around `torch.compile`.

    `torch.compile` is an optional optimization. Some environments (for example, Python 3.14+)
    raise at runtime when compilation is requested. In those cases we continue training with the
    uncompiled model.
    """
    if not enabled:
        return model

    if log is not None:
        log("compiling the model... (takes a ~minute)")

    _enable_dynamo_error_suppression(log=log)

    try:
        compiled = torch.compile(model)
        return _attach_runtime_compile_fallback(
            compiled_model=cast(torch.nn.Module, compiled),
            eager_model=model,
            log=log,
        )
    except Exception as exc:
        if _is_known_compile_failure(exc):
            if log is not None:
                log(f"torch.compile unavailable: {exc}; continuing without compilation.")
            return model
        raise
