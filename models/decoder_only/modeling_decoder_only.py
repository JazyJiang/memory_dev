from __future__ import annotations

from enum import Enum
import math
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
try:
    from transformers.generation.utils import GenerationMixin
except Exception:  # pragma: no cover
    from transformers.generation import GenerationMixin

from pkm.memory import HashingMemory
from .configuration_decoder_only import DecoderOnlyConfig


class InitStdFactor(Enum):
    DISABLED = "disabled"
    GLOBAL_DEPTH = "global_depth"
    CURRENT_DEPTH = "current_depth"
    DIM_RATIO = "dim_ratio"


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()
    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (x.shape[seq_dim], x.shape[-3], 2, 2), f"{freqs_cis.shape=} vs {x.shape=}"
    shape = [d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, seq_dim).float()
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()
        self.theta = float(theta)
        self.head_dim = int(head_dim)
        self.max_seqlen = int(max_seqlen)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=self.head_dim, end=self.max_seqlen, theta=self.theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(dim=self.head_dim, end=self.max_seqlen, theta=self.theta)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)) * self.weight

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)


class Attention(nn.Module):
    def __init__(self, dim: int, head_dim: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        self.dim = int(dim)
        self.head_dim = int(head_dim)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)

        assert self.n_heads % self.n_kv_heads == 0
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, seq_len, _ = x.shape

        xq = self.wq(x.view_as(x)).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x.view_as(x)).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x.view_as(x)).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        past_len = 0
        past_k = past_v = None
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k is None or past_v is None:
                past_key_value = None
                past_k = past_v = None
            else:
                past_len = int(past_k.shape[1])

        xq, xk = apply_rotary_emb(
            xq,
            xk,
            1,
            freq_cis[past_len : past_len + seq_len],
        )

        if past_key_value is not None:
            xk = torch.cat([past_k, xk], dim=1)
            xv = torch.cat([past_v, xv], dim=1)

        present = (xk, xv) if use_cache else None

        xk = repeat_kv(xk, self.heads_per_group)
        xv = repeat_kv(xv, self.heads_per_group)

        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))  # [B,H,T,D]

        attn_mask = None
        if attention_mask is not None:
            key_mask = attention_mask.to(torch.bool)  # [B, K]
            total_k = int(xk.shape[2])
            if key_mask.shape[1] != total_k:
                key_mask = key_mask[:, -total_k:]
            additive = torch.zeros((bsz, 1, 1, total_k), device=x.device, dtype=torch.float32)
            additive = additive.masked_fill(~key_mask[:, None, None, :], float("-1e9"))
            attn_mask = additive.to(dtype=xq.dtype)

        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.wo(out), present

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        init_std = init_std or (self.dim ** (-0.5))
        init_std = init_std / float(factor)
        for w in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.trunc_normal_(w.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)


class FeedForward(nn.Module):
    def __init__(self, dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(float(ffn_dim_multiplier) * hidden_dim)
        hidden_dim = int(multiple_of) * ((hidden_dim + int(multiple_of) - 1) // int(multiple_of))

        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)

        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        return self.w2(F.silu(x1) * x3)

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std / float(factor)
        out_init_std = out_init_std / float(factor)

        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(w.weight, mean=0.0, std=in_init_std, a=-3 * in_init_std, b=3 * in_init_std)
        nn.init.trunc_normal_(self.w2.weight, mean=0.0, std=out_init_std, a=-3 * out_init_std, b=3 * out_init_std)


class TransformerBlock(nn.Module):
    def __init__(self, config: DecoderOnlyConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = int(layer_id)

        head_dim = config.head_dim or (config.d_model // config.n_heads)
        n_heads = config.n_heads or (config.d_model // head_dim)
        n_kv_heads = config.n_kv_heads or n_heads
        assert n_heads % n_kv_heads == 0

        self.attention = Attention(
            dim=config.d_model,
            head_dim=head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )

        use_pkm = config.pk_is_enabled and (self.layer_id in set(config.pk_layers))
        if use_pkm:
            self.feed_forward = HashingMemory(
                input_dim=config.d_model,
                output_dim=config.d_model,
                mem_n_keys=config.pk_mem_n_keys,
                mem_heads=config.pk_mem_heads,
                mem_knn=config.pk_mem_knn,
                mem_share_values=config.pk_mem_share_values,
                mem_k_dim=config.pk_mem_k_dim,
                mem_v_dim=config.pk_mem_v_dim,
                swilu_projection=config.pk_swilu_projection,
                value_fixed_lr=config.pk_value_fixed_lr,
                mem_gated=config.pk_mem_gated,
                peer_variant=config.pk_peer_variant,
            )
            self.feed_forward.layer_id = self.layer_id
        else:
            self.feed_forward = FeedForward(
                dim=config.d_model,
                multiple_of=config.multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
            )

        self.attention_norm = RMSNorm(config.d_model, eps=1e-6)
        self.ffn_norm = RMSNorm(config.d_model, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present = self.attention(
            self.attention_norm(x),
            freq_cis,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = x + attn_out
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, present

    def init_weights(self, init_std=None, factor: float = 1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        if hasattr(self.feed_forward, "reset_parameters"):
            self.feed_forward.reset_parameters(init_std, factor)  # HashingMemory/FeedForward both have it
        self.ffn_norm.reset_parameters()


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: DecoderOnlyConfig) -> None:
        super().__init__()
        self.config = config

        head_dim = config.head_dim or (config.d_model // config.n_heads)
        self.rope_embeddings = RotaryEmbedding(
            theta=config.rope_theta,
            head_dim=head_dim,
            max_seqlen=config.max_seq_len,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = float(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model, eps=1e-6)

    def init_weights(self):
        self.rope_embeddings.reset_parameters()

        dim = float(self.config.d_model)
        std = self.config.init_base_std if self.config.init_base_std is not None else (dim ** (-0.5))
        mode = InitStdFactor(self.config.init_std_factor)

        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=std)

        for depth, layer in enumerate(self.blocks):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.blocks) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: (self.config.d_model / 4096),
                InitStdFactor.DISABLED: 1.0,
            }[mode]
            layer.init_weights(std, factor)

        self.norm.reset_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]]:
        x = self.embed_tokens(input_ids)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        seq_len = int(input_ids.shape[1])
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            pkv0 = past_key_values[0]
            if (
                isinstance(pkv0, (tuple, list))
                and len(pkv0) == 2
                and pkv0[0] is not None
                and pkv0[1] is not None
            ):
                past_len = int(pkv0[0].shape[1])

        freq_cis = self.rope_embeddings(seqlen=past_len + seq_len)

        presents: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            pkv = None
            if past_key_values is not None:
                pkv = past_key_values[i]
                if not (
                    pkv is not None
                    and isinstance(pkv, (tuple, list))
                    and len(pkv) == 2
                    and pkv[0] is not None
                    and pkv[1] is not None
                ):
                    pkv = None

            x, present = block(
                x,
                freq_cis,
                attention_mask=attention_mask,
                past_key_value=pkv,
                use_cache=use_cache,
            )
            if use_cache:
                assert present is not None
                presents.append(present)

        x = self.norm(x)
        return x, (tuple(presents) if use_cache else None)


class DecoderOnlyForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DecoderOnlyConfig

    def __init__(self, config: DecoderOnlyConfig) -> None:
        super().__init__(config)

        self.transformer = DecoderOnlyTransformer(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.transformer.init_weights()
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=(config.d_model ** -0.5))

        self.lm_head.weight = self.transformer.embed_tokens.weight

        self.post_init()

    def _init_weights(self, module):
        return

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.transformer.embed_tokens = new_embeddings
        self.lm_head.weight = self.transformer.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        has_real_past = False
        if past_key_values is not None:
            try:
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    pkv0 = past_key_values[0]
                    if (
                        isinstance(pkv0, (tuple, list))
                        and len(pkv0) == 2
                        and pkv0[0] is not None
                        and pkv0[1] is not None
                        and int(pkv0[0].shape[1]) > 0
                    ):
                        has_real_past = True
            except Exception:
                has_real_past = True

        if not has_real_past:
            past_key_values = None

        if has_real_past:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        reordered = []
        for layer_past in past_key_values:
            k, v = layer_past
            reordered.append((k.index_select(0, beam_idx), v.index_select(0, beam_idx)))
        return tuple(reordered)

    def _build_full_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_token_id = int(getattr(self.config, "pad_token_id", 0))

        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id).to(dtype=torch.long)
        else:
            attention_mask = attention_mask.to(dtype=torch.long)

        prompt_lens = attention_mask.sum(dim=1).tolist()

        response_input_ids = labels.clone()
        response_input_ids[response_input_ids == -100] = pad_token_id
        response_lens = (labels != -100).to(dtype=torch.long).sum(dim=1).tolist()

        max_seq_len = int(getattr(self.config, "max_seq_len", input_ids.size(1)))

        full_input_ids_list: List[torch.Tensor] = []
        full_attention_mask_list: List[torch.Tensor] = []
        full_labels_list: List[torch.Tensor] = []

        for i in range(input_ids.size(0)):
            p_len = int(prompt_lens[i])
            r_len = int(response_lens[i])

            prompt_ids_i = input_ids[i, :p_len]
            response_ids_i = response_input_ids[i, :r_len]
            response_labels_i = labels[i, :r_len]

            seq_ids = torch.cat([prompt_ids_i, response_ids_i], dim=0)
            seq_labels = torch.cat(
                [
                    torch.full((p_len,), -100, dtype=labels.dtype, device=labels.device),
                    response_labels_i,
                ],
                dim=0,
            )
            seq_attn = torch.ones((seq_ids.size(0),), dtype=torch.long, device=input_ids.device)

            if seq_ids.size(0) > max_seq_len:
                seq_ids = seq_ids[-max_seq_len:]
                seq_labels = seq_labels[-max_seq_len:]
                seq_attn = seq_attn[-max_seq_len:]

            full_input_ids_list.append(seq_ids)
            full_labels_list.append(seq_labels)
            full_attention_mask_list.append(seq_attn)

        batch_max_len = max(x.size(0) for x in full_input_ids_list)

        full_input_ids = input_ids.new_full((input_ids.size(0), batch_max_len), pad_token_id)
        full_attention_mask = attention_mask.new_zeros((input_ids.size(0), batch_max_len))
        full_labels = labels.new_full((input_ids.size(0), batch_max_len), -100)

        for i in range(input_ids.size(0)):
            L = full_input_ids_list[i].size(0)
            full_input_ids[i, :L] = full_input_ids_list[i]
            full_attention_mask[i, :L] = full_attention_mask_list[i]
            full_labels[i, :L] = full_labels_list[i]

        return full_input_ids, full_attention_mask, full_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ):
        if labels is not None:
            full_input_ids, full_attention_mask, full_labels = self._build_full_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            hidden, _ = self.transformer(
                full_input_ids,
                attention_mask=full_attention_mask,
                past_key_values=None,
                use_cache=False,
            )
            logits = self.lm_head(hidden)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = full_labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", True))

        hidden, present = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden)
        return CausalLMOutputWithPast(logits=logits, past_key_values=present)