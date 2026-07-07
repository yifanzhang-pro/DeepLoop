# Looped GPT with MHA + RoPE
# Repeats K physical transformer blocks R times (block_repeats).
# Effective depth = num_hidden_layers * block_repeats.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, cast
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from .activations import ActivationName, apply_activation, validate_activation_name
from .attention_dtype import AttentionDType
from .gpt_base import GPTBase, PastKeyValue, SharedKVState
from .kv_cache import append_preallocated, get_past_len, maybe_get_cache_len
from .losses import get_loss_fn
from .rmsnorm import RMSNorm
from .kv_shift import ShiftLinear
from .pydantic_config import validate_pretrained_config_kwargs
from .rotary import Rotary, apply_rotary_emb


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.q_activation = validate_activation_name(getattr(config, "q_activation", None), field_name="q_activation")
        self.k_activation = validate_activation_name(getattr(config, "k_activation", None), field_name="k_activation")
        self.v_activation = validate_activation_name(getattr(config, "v_activation", None), field_name="v_activation")
        self.use_k_shift = getattr(config, "use_k_shift", False)
        self.use_v_shift = getattr(config, "use_v_shift", False)
        self.use_output_gate = getattr(config, "use_output_gate", False)
        # projections to per-head dimensions
        self.c_q = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_k_shift:
            self.c_k = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_k = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        if self.use_v_shift:
            self.c_v = ShiftLinear(self.hidden_size, self.n_head * self.head_dim, self.n_head, bias=False)
        else:
            self.c_v = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
        # output projection maps back to embedding dim
        self.o_proj = nn.Linear(self.n_head * self.head_dim, self.hidden_size, bias=False)
        rope_ratio = float(getattr(config, "rope_ratio", 1.0))
        self.rotary = Rotary(self.head_dim, base=getattr(config, "rope_base", 10000.0), rope_ratio=rope_ratio)
        self.using_groupnorm = config.using_groupnorm
        # QK RMSNorm (learnable) flag and layers
        self.use_qk_rmsnorm = getattr(config, "use_qk_rmsnorm", True)
        if self.use_qk_rmsnorm:
            self.q_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            self.k_rms = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        if self.using_groupnorm:
            # Apply RMSNorm to each head's output dimension
            self.subln = RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
            if self.use_output_gate:
                self.g_proj = nn.Linear(self.hidden_size, self.n_head * self.head_dim, bias=False)
                if not self.using_groupnorm:
                    self.o_norm = RMSNorm(
                        self.head_dim, eps=getattr(config, "rms_norm_eps", 1e-5), elementwise_affine=True
                    )

        kv_cache_slot_size = int(getattr(config, "kv_cache_slot_size", 128))
        if kv_cache_slot_size <= 0:
            raise ValueError(f"kv_cache_slot_size must be positive, got {kv_cache_slot_size}.")
        self.kv_cache_slot_size = kv_cache_slot_size

    def _apply_attention(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        *,
        x: torch.Tensor,
        past_len: int,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.size()
        total_len = int(k_t.shape[-2])
        attn_mask: torch.Tensor | None = None
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError(f"attention_mask must have shape (B, S), got {tuple(attention_mask.shape)}")
            if int(attention_mask.shape[0]) != B:
                raise ValueError(f"attention_mask batch mismatch: expected {B}, got {int(attention_mask.shape[0])}")
            if int(attention_mask.shape[1]) != total_len:
                raise ValueError(
                    f"attention_mask sequence mismatch: expected {total_len}, got {int(attention_mask.shape[1])}"
                )
            key_mask = attention_mask.to(dtype=torch.bool)[:, None, None, :]
            attn_mask = key_mask

        use_is_causal = past_len == 0 and attn_mask is None
        if use_is_causal:
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        else:
            if T > 1:
                key_positions = torch.arange(total_len, device=x.device)
                query_positions = past_len + torch.arange(T, device=x.device)
                causal_mask = key_positions <= query_positions[:, None]
            else:
                causal_mask = torch.ones((T, total_len), dtype=torch.bool, device=x.device)
            if attn_mask is not None:
                attn_mask = attn_mask & causal_mask[None, None, :, :]
            else:
                attn_mask = causal_mask[None, None, :, :]
            y = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, is_causal=False)

        if self.using_groupnorm:
            y = self.subln(y)
        elif self.use_output_gate:
            y = self.o_norm(y)

        if self.use_output_gate:
            gate = self.g_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            y = y * F.silu(gate)

        y = y.transpose(1, 2).contiguous().reshape(B, T, self.n_head * self.head_dim)
        return self.o_proj(y)

    def forward(self, x):
        y, _ = self.forward_with_past(x)
        return y

    def forward_with_past(
        self,
        x: torch.Tensor,
        *,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        if self.use_k_shift:
            k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        if self.use_v_shift:
            v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
        else:
            v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        q = apply_activation(q, self.q_activation)
        k = apply_activation(k, self.k_activation)
        v = apply_activation(v, self.v_activation)

        past_len = 0
        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        cache_len = maybe_get_cache_len(past_key_value, batch_size=B)
        past_k_used: torch.Tensor | None = None
        past_v_used: torch.Tensor | None = None
        if past_key_value is not None:
            if len(past_key_value) < 2:
                raise ValueError("past_key_value must have at least 2 tensors: (key, value).")
            past_k = past_key_value[0]
            past_v = past_key_value[1]
            past_len = get_past_len(past_k=past_k, cache_len=cache_len)
            if cache_len is None:
                past_k_used = past_k
                past_v_used = past_v
            else:
                past_k_used = past_k[:, :, :past_len, :]
                past_v_used = past_v[:, :, :past_len, :]

        cos, sin = self.rotary(q, seq_len_offset=past_len)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if self.use_qk_rmsnorm:
            q = self.q_rms(q)
            k = self.k_rms(k)

        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        cache_tensors: list[torch.Tensor] | None = None
        cache_len_out: torch.Tensor | None = None
        if use_cache:
            views, caches, cache_len_out = append_preallocated(
                past=[past_k, past_v] if past_k is not None and past_v is not None else None,
                cache_len=cache_len,
                past_len=past_len,
                new=[k_t, v_t],
                slot_size=self.kv_cache_slot_size,
            )
            k_t, v_t = views
            cache_tensors = caches
        elif past_k_used is not None and past_v_used is not None:
            k_t = torch.cat([past_k_used, k_t], dim=-2)
            v_t = torch.cat([past_v_used, v_t], dim=-2)

        y = self._apply_attention(q_t, k_t, v_t, x=x, past_len=past_len, attention_mask=attention_mask)
        present: PastKeyValue | None = None
        if use_cache:
            if cache_tensors is None or cache_len_out is None:
                raise RuntimeError("KV cache append did not return expected cache tensors.")
            present = (cache_tensors[0], cache_tensors[1], cache_len_out)
        return y, present

    def forward_with_shared_kv(
        self,
        x: torch.Tensor,
        *,
        shared_kv_state: SharedKVState | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SharedKVState]:
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        q = apply_activation(q, self.q_activation)

        if shared_kv_state is None:
            if self.use_k_shift:
                k = self.c_k(x, None).view(B, T, self.n_head, self.head_dim)
            else:
                k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
            if self.use_v_shift:
                v = self.c_v(x, None).view(B, T, self.n_head, self.head_dim)
            else:
                v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

            k = apply_activation(k, self.k_activation)
            v = apply_activation(v, self.v_activation)

            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            if self.use_qk_rmsnorm:
                q = self.q_rms(q)
                k = self.k_rms(k)

            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            next_shared_kv_state: SharedKVState = (k_t, v_t)
        else:
            if len(shared_kv_state) != 2:
                raise ValueError("shared_kv_state must have exactly 2 tensors: (key, value).")
            k_t, v_t = shared_kv_state
            if k_t.ndim != 4 or v_t.ndim != 4:
                raise ValueError("shared_kv_state tensors must have shape (B, H, T, D).")
            if int(k_t.shape[0]) != B or int(v_t.shape[0]) != B:
                raise ValueError(
                    "shared_kv_state batch mismatch: "
                    f"expected batch {B}, got key batch {int(k_t.shape[0])} and value batch {int(v_t.shape[0])}."
                )
            if int(k_t.shape[-2]) != T or int(v_t.shape[-2]) != T:
                raise ValueError(
                    "shared_kv_state sequence mismatch: "
                    f"expected {T}, got key length {int(k_t.shape[-2])} and value length {int(v_t.shape[-2])}."
                )

            cos, sin = self.rotary(q)
            q = apply_rotary_emb(q, cos, sin)
            if self.use_qk_rmsnorm:
                q = self.q_rms(q)

            next_shared_kv_state = shared_kv_state

        q_t = q.transpose(1, 2)
        y = self._apply_attention(q_t, k_t, v_t, x=x, past_len=0, attention_mask=attention_mask)
        return y, next_shared_kv_state


# -----------------------------------------------------------------------------

@dataclass
class GPTConfig(PretrainedConfig):
    model_type = "nanogpt-pro"
    vocab_size: int = 50304
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    hidden_size: int = 768
    head_dim: int = 128
    block_size: int = 1024
    bias: bool = False
    dropout: float = 0.0
    scale_attn_by_inverse_layer_idx: bool = False
    using_groupnorm: bool = False
    use_qk_rmsnorm: bool = True
    use_k_shift: bool = False
    use_v_shift: bool = False
    use_output_gate: bool = False

    q_activation: ActivationName | None = None
    k_activation: ActivationName | None = None
    v_activation: ActivationName | None = None
    attention_dtype: AttentionDType = "auto"

    rope_ratio: float = 1.0
    rope_base: float = 10000.0
    mlp_hidden_mult: float = 8 / 3
    embedding_init_std: float = 0.02
    hidden_init_std_factor: float = 0.5
    mup: bool = False
    hidden_size_base: int = 1024
    embedding_lr_multiplier: float = 1.0

    # Loop config
    block_repeats: int = 1

    # Input embedding RMSNorm
    use_input_embed_rmsnorm: bool = False

    # LoopDeepNorm
    use_loop_deepnorm: bool = False
    loop_deepnorm_alpha: float | None = None
    loop_deepnorm_beta: float | None = None
    loop_deepnorm_p: float = 0.5
    use_shared_kv_activations: bool = False

    # ACT (Adaptive Computation Time, Graves 2016) — sequence-level halting.
    # When enabled, forward() iterates the L blocks for up to `act_max_steps`
    # R-cycles. After each R-cycle the halting head outputs a per-sequence
    # halting probability. The final logits are a halt-prob-weighted sum of
    # each step's logits. Training adds a ponder cost λ · mean(n_steps) to
    # encourage early halting. Mutually exclusive with KV cache (each ACT
    # step would otherwise need its own cache slot).
    use_act: bool = False
    act_max_steps: int = 16
    act_threshold: float = 0.99
    act_ponder_lambda: float = 0.01

    # Loss function: "softmax_ce" (default, std PyTorch CE) or "stablemax_ce"
    # (HRM's exp-free stable variant in fp64).
    loss_type: str = "softmax_ce"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**validate_pretrained_config_kwargs(type(self), kwargs))


class GPT(GPTBase):
    config_class = GPTConfig
    attention_cls = CausalSelfAttention

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.block_repeats = max(1, int(getattr(config, "block_repeats", 1)))
        self.use_shared_kv_activations = bool(getattr(config, "use_shared_kv_activations", False))
        self.use_act = bool(getattr(config, "use_act", False))
        self.act_max_steps = max(1, int(getattr(config, "act_max_steps", 16)))
        self.act_threshold = float(getattr(config, "act_threshold", 0.99))
        self.act_ponder_lambda = float(getattr(config, "act_ponder_lambda", 0.01))
        if self.use_act:
            self.halting_head = nn.Linear(int(getattr(config, "hidden_size")), 1, bias=True)
        else:
            self.halting_head = None
        self.loss_fn = get_loss_fn(str(getattr(config, "loss_type", "softmax_ce")))

    def forward(
        self,
        idx: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        return_logits: bool = True,
        output_all_seq: bool = False,
        *,
        input_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[PastKeyValue, ...] | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor | None, torch.Tensor | None]:
        del cache_position, kwargs

        if (idx is None) == (input_ids is None):
            raise ValueError("Exactly one of `idx` or `input_ids` must be provided.")
        if idx is None:
            idx = input_ids

        if labels is not None and targets is not None:
            raise ValueError("Only one of `labels` or `targets` can be provided.")
        if targets is None:
            targets = labels

        use_cache = bool(use_cache) if use_cache is not None else False
        return_dict = bool(return_dict) if return_dict is not None else False
        output_hidden_states = bool(output_hidden_states) if output_hidden_states is not None else False
        output_attentions = bool(output_attentions) if output_attentions is not None else False

        if output_attentions:
            raise NotImplementedError("output_attentions=True is not supported.")

        if (use_cache or past_key_values is not None) and self.use_shared_kv_activations:
            raise NotImplementedError(
                "KV cache and shared_kv_activations are mutually exclusive — shared-KV reuses "
                "first-iteration K/V across loop repeats, which conflicts with per-step caching."
            )

        if self.use_act and (use_cache or past_key_values is not None):
            raise NotImplementedError(
                "ACT and KV cache are mutually exclusive — each ACT step recomputes the loop, "
                "which would require its own cache slot per (token, step)."
            )
        if self.use_act and self.use_shared_kv_activations:
            raise NotImplementedError("ACT is incompatible with shared_kv_activations.")

        L = len(self.transformer.h)
        n_virtual_layers = L * self.block_repeats
        if past_key_values is not None and len(past_key_values) != n_virtual_layers:
            raise ValueError(
                f"past_key_values must have length L*block_repeats={n_virtual_layers}, "
                f"got {len(past_key_values)}."
            )

        x_emb = self.transformer.wte(idx)
        if self.input_embed_rmsnorm is not None:
            x_emb = self.input_embed_rmsnorm(x_emb)

        hidden_states: tuple[torch.Tensor, ...] | None = (x_emb,) if output_hidden_states else None

        logits_scale = 1.0
        if bool(getattr(self.config, "mup", False)):
            logits_scale = float(getattr(self.config, "hidden_size_base", 1024)) / float(self.config.hidden_size)

        # ----- ACT path (sequence-level halting, multi-pass refinement) -----
        if self.use_act:
            assert self.halting_head is not None
            B = int(idx.shape[0])
            cum_halt = torch.zeros(B, device=x_emb.device, dtype=torch.float32)
            n_steps = torch.zeros(B, device=x_emb.device, dtype=torch.float32)
            weighted_logits: torch.Tensor | None = None
            x = x_emb
            done = torch.zeros(B, device=x_emb.device, dtype=torch.bool)
            for step_idx in range(self.act_max_steps):
                # One refinement pass = one R-cycle through the L blocks. We
                # treat the whole block_repeats=R loop as one "ACT step" so
                # the model sees the same effective depth at training and
                # inference; ACT extends depth past R.
                for _r in range(self.block_repeats):
                    for block in self.transformer.h:
                        x = block(x)
                if output_hidden_states:
                    assert hidden_states is not None
                    hidden_states = (*hidden_states, x)
                x_norm = self.ln_f(x)
                # Sequence-level halting: pool the final hidden state across
                # the sequence (mean over T) to get a per-batch halting prob.
                seq_pool = x_norm.mean(dim=1).float()
                halt_logit = self.halting_head(seq_pool).squeeze(-1)
                h_t = torch.sigmoid(halt_logit)
                is_last = step_idx == self.act_max_steps - 1
                if is_last:
                    halt_step = (1.0 - cum_halt).clamp(min=0.0)
                else:
                    halt_step = h_t * (1.0 - cum_halt)
                # Once a sequence has halted (cum_halt > threshold), pin its
                # contribution to zero for later steps so total sums to 1.
                halt_step = torch.where(done, torch.zeros_like(halt_step), halt_step)
                cum_halt = cum_halt + halt_step
                n_steps = n_steps + halt_step * float(step_idx + 1)
                step_logits = (self.lm_head(x_norm) * logits_scale).float()
                contribution = halt_step.view(B, 1, 1) * step_logits
                if weighted_logits is None:
                    weighted_logits = contribution
                else:
                    weighted_logits = weighted_logits + contribution
                done = done | (cum_halt > self.act_threshold)
                if bool(done.all().item()) and not is_last:
                    # Force-halt remainder for any not-yet-fully-halted sequence.
                    remainder = (1.0 - cum_halt).clamp(min=0.0)
                    if bool((remainder > 0).any().item()):
                        weighted_logits = weighted_logits + remainder.view(B, 1, 1) * step_logits
                        n_steps = n_steps + remainder * float(step_idx + 1)
                    break
            assert weighted_logits is not None
            logits_full = weighted_logits

            if targets is not None:
                ce = self.loss_fn(logits_full, targets, ignore_index=-1)
                ponder = (n_steps - 1.0).mean()
                loss = ce + self.act_ponder_lambda * ponder
                logits = logits_full
            else:
                loss = None
                logits = logits_full if (output_all_seq or return_dict) else logits_full[:, [-1], :]
            if not return_logits:
                logits = None
            past_out_act: tuple[PastKeyValue, ...] | None = None  # ACT path doesn't emit cache
            if not return_dict:
                return logits, loss
            return CausalLMOutputWithPast(
                loss=cast(torch.FloatTensor | None, loss),
                logits=logits,
                past_key_values=cast(Any, past_out_act),
                hidden_states=cast(tuple[torch.FloatTensor, ...] | None, hidden_states),
                attentions=None,
            )

        # ----- non-ACT path (original looped forward) -----
        x = x_emb
        shared_kv_states: list[SharedKVState | None] | None = None
        if self.use_shared_kv_activations and self.block_repeats > 1:
            shared_kv_states = [None] * L

        present_key_values: list[PastKeyValue] | None = [] if use_cache else None

        for repeat_idx in range(self.block_repeats):
            for layer_idx, block in enumerate(self.transformer.h):
                if shared_kv_states is not None:
                    x, next_shared_kv_state = block.forward_with_shared_kv(
                        x,
                        shared_kv_state=shared_kv_states[layer_idx],
                        attention_mask=attention_mask,
                    )
                    if shared_kv_states[layer_idx] is None:
                        shared_kv_states[layer_idx] = next_shared_kv_state
                elif use_cache or past_key_values is not None or attention_mask is not None:
                    virtual_idx = repeat_idx * L + layer_idx
                    past = past_key_values[virtual_idx] if past_key_values is not None else None
                    x, present = block.forward_with_past(
                        x,
                        past_key_value=past,
                        use_cache=use_cache,
                        attention_mask=attention_mask,
                    )
                    if use_cache:
                        if present is None:
                            raise NotImplementedError(
                                f"{type(block.attn).__name__} does not implement `forward_with_past` "
                                "needed for KV cache."
                            )
                        assert present_key_values is not None
                        present_key_values.append(present)
                else:
                    x = block(x)

            if output_hidden_states:
                assert hidden_states is not None
                hidden_states = (*hidden_states, x)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x).float() * logits_scale
            loss = self.loss_fn(logits, targets, ignore_index=-1)
        else:
            loss = None
            if output_all_seq or return_dict:
                logits = self.lm_head(x) * logits_scale
            else:
                logits = self.lm_head(x[:, [-1], :]).float() * logits_scale

        if not return_logits:
            logits = None

        past_out: tuple[PastKeyValue, ...] | None = None
        if use_cache:
            assert present_key_values is not None
            past_out = tuple(present_key_values)

        if not return_dict:
            return logits, loss

        return CausalLMOutputWithPast(
            loss=cast(torch.FloatTensor | None, loss),
            logits=logits,
            past_key_values=cast(Any, past_out),
            hidden_states=cast(tuple[torch.FloatTensor, ...] | None, hidden_states),
            attentions=None,
        )

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        N = self.get_num_params()
        cfg = self.config
        L = int(getattr(cfg, "num_hidden_layers")) * self.block_repeats
        H = int(getattr(cfg, "num_attention_heads"))
        head_dim = int(getattr(cfg, "head_dim", int(getattr(cfg, "hidden_size")) // H))
        T = int(getattr(cfg, "block_size"))
        flops_per_token = 6 * N + 12 * L * H * head_dim * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised
