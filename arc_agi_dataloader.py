"""Per-example random-batch dataloader for the ARC-AGI causal-LM port.

The ARC-AGI port (see ``data/arc-agi/prepare.py``) materializes every
training and evaluation example as a fixed-length token sequence of the form
``[SEP_IN, in_0..in_899, SEP_OUT, out_0..out_899]`` and writes them to
``train.npy`` / ``val.npy``. This dataloader samples uniformly at random from
those arrays, builds (idx, targets) pairs with the input region masked out of
the loss via ``ignore_index=-1``, and returns CUDA tensors.

The arrays are mmapped so that a single large dataset (e.g., the
``num_aug=1000`` recipe) doesn't have to fit in RAM.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch

DEFAULT_IGNORE_INDEX: int = -1


@dataclass(frozen=True)
class ArcAgiMeta:
    vocab_size: int
    block_size: int  # length of `idx` after the trailing one-shift (== sequence_length - 1)
    sequence_length: int
    grid_size: int
    grid_area: int  # grid_size**2
    pad_id: int
    eos_id: int
    sep_in_id: int
    sep_out_id: int
    digit_offset: int
    ignore_index: int


def load_meta(data_dir: str) -> ArcAgiMeta:
    with open(os.path.join(data_dir, "meta.json")) as f:
        raw = json.load(f)
    grid_size = int(raw["grid_size"])
    return ArcAgiMeta(
        vocab_size=int(raw["vocab_size"]),
        block_size=int(raw["block_size"]),
        sequence_length=int(raw["sequence_length"]),
        grid_size=grid_size,
        grid_area=grid_size * grid_size,
        pad_id=int(raw["pad_id"]),
        eos_id=int(raw["eos_id"]),
        sep_in_id=int(raw["sep_in_id"]),
        sep_out_id=int(raw["sep_out_id"]),
        digit_offset=int(raw["digit_offset"]),
        ignore_index=int(raw.get("ignore_index", DEFAULT_IGNORE_INDEX)),
    )


class ArcAgiDataloader:
    """Random-batch sampler over the prepared ARC-AGI .npy splits.

    The first ``1 + grid_area`` positions of every shifted ``targets`` row are
    set to ``ignore_index`` so that next-token cross-entropy is computed
    exclusively on the 900 output-grid positions, matching HRM's seq2seq loss.
    """

    def __init__(
        self,
        data_dir: str,
        *,
        split: str,
        batch_size: int,
        device: str | torch.device,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        mask_output_pad: bool = True,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if world_size <= 0 or rank < 0 or rank >= world_size:
            raise ValueError(f"Invalid rank/world_size: rank={rank} world_size={world_size}")

        self.meta = load_meta(data_dir)
        path = os.path.join(data_dir, f"{split}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}; run data/arc-agi/prepare.py first.")
        self._array: np.ndarray = np.load(path, mmap_mode="r")
        if self._array.ndim != 2 or self._array.shape[1] != self.meta.sequence_length:
            raise ValueError(
                f"Unexpected shape {self._array.shape} for {path}; "
                f"expected (*, {self.meta.sequence_length})."
            )

        self._batch_size = int(batch_size)
        self._device = torch.device(device)
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._shuffle = bool(shuffle)
        self._mask_output_pad = bool(mask_output_pad)
        self._n = int(self._array.shape[0])
        self._block_size = self.meta.block_size

        self._rng = np.random.default_rng(int(seed) + self._rank)
        # The deterministic test-set walker keeps each rank on its own stride.
        self._eval_cursor = 0
        self._eval_order: np.ndarray | None = None

        # Mask covers [SEP_IN, in_0, ..., in_899] in `targets` (positions 0..grid_area).
        # `targets[i] = tokens[i+1]`, so target index `i` is in the input region iff
        # `i+1 <= grid_area`, i.e. i in [0..grid_area]. That's grid_area+1 = 901 positions.
        self._n_input_targets = self.meta.grid_area + 1

    @property
    def vocab_size(self) -> int:
        return self.meta.vocab_size

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def sequence_length(self) -> int:
        return self.meta.sequence_length

    @property
    def num_examples(self) -> int:
        return self._n

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._device.type == "cuda":
            x = x.pin_memory().to(self._device, non_blocking=True)
            y = y.pin_memory().to(self._device, non_blocking=True)
        else:
            x = x.to(self._device)
            y = y.to(self._device)
        return x, y

    def _build_batch(self, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        # Materialize the rows (np.ndarray copy from mmap).
        rows = np.asarray(self._array[indices], dtype=np.int64)
        idx = torch.from_numpy(rows[:, :-1].copy())
        targets_np = rows[:, 1:].copy()
        targets_np[:, : self._n_input_targets] = self.meta.ignore_index
        if self._mask_output_pad:
            # HRM uses ignore_label_id=PAD: PAD positions in the output grid are
            # excluded from the loss so the model is supervised only on actual
            # cells and EOS boundary markers.
            out_region = targets_np[:, self._n_input_targets :]
            out_region[out_region == self.meta.pad_id] = self.meta.ignore_index
        targets = torch.from_numpy(targets_np)
        return self._to_device(idx, targets)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Random batch sampled with replacement (training mode)."""
        idx = self._rng.integers(0, self._n, size=self._batch_size)
        return self._build_batch(idx)

    def reset_eval(self) -> None:
        self._eval_cursor = 0
        if self._shuffle:
            order = self._rng.permutation(self._n)
        else:
            order = np.arange(self._n, dtype=np.int64)
        self._eval_order = order

    def next_eval_batch(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sequential batch over the current eval order; returns None when exhausted."""
        if self._eval_order is None:
            self.reset_eval()
        assert self._eval_order is not None
        if self._eval_cursor >= self._n:
            return None
        # Each rank takes its own stride across the global order.
        local_start = self._eval_cursor + self._rank * self._batch_size
        local_end = min(local_start + self._batch_size, self._n)
        if local_start >= self._n:
            self._eval_cursor += self._world_size * self._batch_size
            # Try again at the next stride; if still empty, we're done.
            if self._eval_cursor >= self._n:
                return None
            local_start = self._eval_cursor + self._rank * self._batch_size
            local_end = min(local_start + self._batch_size, self._n)
            if local_start >= self._n:
                return None
        idx = self._eval_order[local_start:local_end]
        # Pad short tail batches by re-sampling so sizes stay constant for DDP.
        if idx.size < self._batch_size:
            extra = self._rng.integers(0, self._n, size=self._batch_size - idx.size)
            idx = np.concatenate([idx, extra])
        self._eval_cursor += self._world_size * self._batch_size
        return self._build_batch(idx)
