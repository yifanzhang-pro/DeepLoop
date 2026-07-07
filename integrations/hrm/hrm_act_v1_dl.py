"""HRM ACT V1 + DeepLoop's LoopDeepNorm method (Phase 1).

This is a copy of `hrm_act_v1.py` with three additions from our DeepLoop core
method:

  1. **LoopDeepNorm residual scaling** in each transformer block:
       hidden = rms_norm(alpha * hidden + sublayer(hidden))
     where alpha = (2*M)^p and M is the effective number of layer
     applications across the H/L module cycles.

  2. **Beta-scaled output-projection init** for attention `o_proj` and
     SwiGLU `down_proj`: weight *= beta, beta = (8*M)^-p.

  3. **Optional input-embedding RMSNorm** before the embed_scale multiply.

When `use_loop_deepnorm=False` and `use_input_embed_rmsnorm=False` (defaults),
this module is functionally identical to the original `hrm_act_v1.py` — the
residual scaling collapses to alpha=1.0 and no beta init is applied. Switching
to the patched arch is done by pointing the run config at this module:

  config/arch/hrm_v1_dl.yaml:
    name: hrm.hrm_act_v1_dl@HierarchicalReasoningModel_ACTV1
    use_loop_deepnorm: True
    use_input_embed_rmsnorm: True

p defaults to 0.5 (matches DeepLoop's default LoopDeepNorm exponent).
M defaults to L_layers * H_cycles * L_cycles + H_layers * H_cycles, the
per-forward count of block applications across one inner pass.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # ---- DeepLoop additions (Phase 1) ----
    # Toggle the LoopDeepNorm residual + init scaling. Default False makes
    # this module behave exactly like upstream HRM.
    use_loop_deepnorm: bool = False
    # alpha / beta override the auto-computed values from M and p.
    loop_deepnorm_alpha: Optional[float] = None
    loop_deepnorm_beta: Optional[float] = None
    loop_deepnorm_p: float = 0.5
    # Optional override for the effective depth M used in the alpha/beta
    # formulas. None → auto-compute from H/L layers and cycles.
    loop_deepnorm_effective_M: Optional[int] = None
    # Apply RMSNorm to the token+puzzle+pos embedding before embed_scale.
    # Legacy boolean (True == "after_puzzle"). Use input_embed_rmsnorm_placement
    # to control where RMSNorm is applied relative to puzzle_emb concat.
    use_input_embed_rmsnorm: bool = False
    # When set, overrides use_input_embed_rmsnorm. Choices:
    #   "none"          — no RMSNorm
    #   "after_puzzle"  — Phase 1+ behavior; norm AFTER (token+puzzle+pos)
    #   "before_puzzle" — norm token_emb BEFORE puzzle concat (preserves
    #                     per-puzzle signal magnitude)
    #   "token_only"    — alias for "before_puzzle"
    input_embed_rmsnorm_placement: Optional[str] = None
    # Where to apply LoopDeepNorm scaling: "both" (Phase 1), "h" (H_level only),
    # or "l" (L_level only). Asymmetric application tests if one hierarchy
    # benefits while the other hurts. "none" silently disables.
    loop_deepnorm_apply_to: str = "both"
    # Move rms_norm to BEFORE the sublayer (pre-norm) instead of AFTER
    # (post-norm = HRM upstream). When True the block becomes:
    #     hidden = hidden + sublayer(rms_norm(hidden))
    # This changes gradient flow and is independent of LoopDeepNorm scaling.
    use_pre_norm: bool = False


def _effective_M(config: "HierarchicalReasoningModel_ACTV1Config") -> int:
    """Per-forward effective layer-application count.

    Inside one `_Inner.forward`, L_level is invoked H_cycles*L_cycles times
    (no_grad inner loop minus the last skipped pair, plus one final 1-step-grad
    call), so it nets to exactly H_cycles*L_cycles total invocations. Likewise,
    H_level is invoked H_cycles times. Each invocation runs L_layers (or
    H_layers) blocks in series, so M = L_layers * H_cycles * L_cycles +
    H_layers * H_cycles.
    """
    if config.loop_deepnorm_effective_M is not None:
        return max(1, int(config.loop_deepnorm_effective_M))
    return config.L_layers * config.H_cycles * config.L_cycles + config.H_layers * config.H_cycles


def _resolve_alpha(config: "HierarchicalReasoningModel_ACTV1Config") -> float:
    if config.loop_deepnorm_alpha is not None:
        alpha = float(config.loop_deepnorm_alpha)
    else:
        alpha = math.pow(2.0 * _effective_M(config), config.loop_deepnorm_p)
    if alpha <= 0.0:
        raise ValueError(f"loop_deepnorm_alpha must be > 0, got {alpha}.")
    return alpha


def _resolve_beta(config: "HierarchicalReasoningModel_ACTV1Config") -> float:
    if config.loop_deepnorm_beta is not None:
        beta = float(config.loop_deepnorm_beta)
    else:
        beta = math.pow(8.0 * _effective_M(config), -config.loop_deepnorm_p)
    if beta <= 0.0:
        raise ValueError(f"loop_deepnorm_beta must be > 0, got {beta}.")
    return beta


class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

        # alpha=1.0 reproduces the upstream HRM forward exactly.
        # _Inner.__init__ may override this post-construction for asymmetric
        # H-only / L-only application via config.loop_deepnorm_apply_to.
        self.residual_alpha = _resolve_alpha(config) if config.use_loop_deepnorm else 1.0
        self.use_pre_norm = bool(getattr(config, "use_pre_norm", False))

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_pre_norm:
            # Pre-Norm: hidden + sublayer(norm(hidden))
            # Skips alpha scaling because pre-norm already controls input
            # variance to sublayer; redundant scaling would just rescale identity.
            hidden_states = hidden_states + self.self_attn(
                cos_sin=cos_sin,
                hidden_states=rms_norm(hidden_states, variance_epsilon=self.norm_eps),
            )
            hidden_states = hidden_states + self.mlp(
                rms_norm(hidden_states, variance_epsilon=self.norm_eps),
            )
            return hidden_states
        # Post-Norm (HRM upstream): rms_norm(alpha*hidden + sublayer(hidden))
        # When residual_alpha == 1.0 this matches HRM upstream byte-for-byte.
        hidden_states = rms_norm(self.residual_alpha * hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        hidden_states = rms_norm(self.residual_alpha * hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)

        return hidden_states


class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale  = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.H_layers)])
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(layers=[HierarchicalReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # ---- DeepLoop addition: beta-scaled output-projection init ----
        # Done after both H_level and L_level are constructed so we can walk
        # all blocks in one place. Targets attn.o_proj and mlp.down_proj —
        # the same projections DeepLoop's LoopDeepNorm scales in
        # gpt-mha-rope-loop. `loop_deepnorm_apply_to` gates which level(s)
        # receive the alpha/beta scaling; default "both" matches Phase 1.
        if self.config.use_loop_deepnorm:
            apply_to = (self.config.loop_deepnorm_apply_to or "both").lower()
            if apply_to not in ("both", "h", "l", "none"):
                raise ValueError(f"loop_deepnorm_apply_to must be one of both|h|l|none, got {apply_to}")
            alpha = _resolve_alpha(self.config)
            beta = _resolve_beta(self.config)
            target_levels = []
            if apply_to in ("both", "h"):
                target_levels.append(self.H_level)
            if apply_to in ("both", "l"):
                target_levels.append(self.L_level)
            # Reset alpha=1 on un-targeted level (Block ctor set it via global flag).
            untargeted = []
            if apply_to == "h":
                untargeted.append(self.L_level)
            elif apply_to == "l":
                untargeted.append(self.H_level)
            elif apply_to == "none":
                untargeted = [self.H_level, self.L_level]
            for level in untargeted:
                for layer in level.layers:
                    layer.residual_alpha = 1.0
            with torch.no_grad():
                for level in target_levels:
                    for layer in level.layers:
                        layer.residual_alpha = alpha  # explicit (Block ctor matches but be safe)
                        layer.self_attn.o_proj.weight.mul_(beta)
                        layer.mlp.down_proj.weight.mul_(beta)

        # Resolve effective embed-RMSNorm placement.
        # placement value > legacy bool > default "none".
        if self.config.input_embed_rmsnorm_placement is not None:
            self._embed_rmsnorm_placement = self.config.input_embed_rmsnorm_placement.lower()
        elif self.config.use_input_embed_rmsnorm:
            self._embed_rmsnorm_placement = "after_puzzle"
        else:
            self._embed_rmsnorm_placement = "none"
        if self._embed_rmsnorm_placement not in ("none", "after_puzzle", "before_puzzle", "token_only"):
            raise ValueError(f"input_embed_rmsnorm_placement: unknown value {self._embed_rmsnorm_placement}")
        if self._embed_rmsnorm_placement == "token_only":
            self._embed_rmsnorm_placement = "before_puzzle"

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Optional RMSNorm BEFORE puzzle concat — normalizes only token signal,
        # leaves puzzle_emb at its original (per-puzzle, init_std=0) magnitude.
        if self._embed_rmsnorm_placement == "before_puzzle":
            embedding = rms_norm(embedding, variance_epsilon=self.config.rms_norm_eps)

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Optional RMSNorm AFTER puzzle concat (Phase 1+ legacy behavior).
        if self._embed_rmsnorm_placement == "after_puzzle":
            embedding = rms_norm(embedding, variance_epsilon=self.config.rms_norm_eps)

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L

            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)

                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)

        assert not z_H.requires_grad and not z_L.requires_grad

        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
