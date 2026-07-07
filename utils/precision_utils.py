from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch

TrainDType = Literal["float32", "bfloat16", "float16"]


def resolve_train_dtype(
    dtype: TrainDType,
    *,
    device: torch.device,
    log: Callable[[str], None] | None = None,
    cuda_device_capability: tuple[int, int] | None = None,
) -> TrainDType:
    """Resolve the requested training dtype to a hardware-supported dtype.

    In particular, bf16 Triton/PTX codegen requires NVIDIA Ampere (sm80+) and can fail on
    pre-Ampere GPUs when `torch.compile` / Inductor emits bf16 kernels.
    """

    if dtype != "bfloat16":
        return dtype
    if device.type != "cuda":
        return dtype

    if cuda_device_capability is not None:
        major, minor = cuda_device_capability
    else:
        if not torch.cuda.is_available():
            return dtype
        try:
            major, minor = torch.cuda.get_device_capability(device)
        except Exception:
            return dtype

    if major < 8:
        if log is not None:
            log(
                "dtype='bfloat16' requested on a pre-Ampere CUDA device "
                f"(compute capability sm_{major}{minor}); falling back to float16."
            )
        return "float16"

    return dtype
