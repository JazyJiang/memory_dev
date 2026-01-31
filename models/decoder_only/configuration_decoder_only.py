from __future__ import annotations

from typing import List, Optional

from transformers import PretrainedConfig


class DecoderOnlyConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 32128,
        pad_token_id: int = 0,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        max_seq_len: int = 2048,
        d_model: int = 512,
        n_layers: int = 6,
        head_dim: Optional[int] = None,
        n_heads: Optional[int] = 8,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        init_base_std: Optional[float] = None,
        init_std_factor: str = "disabled",
        pk_is_enabled: bool = False,
        pk_layers: Optional[List[int]] = None,
        pk_mem_n_keys: int = 128,
        pk_mem_heads: int = 4,
        pk_mem_knn: Optional[int] = None,
        pk_mem_share_values: bool = True,
        pk_mem_k_dim: int = 512,
        pk_mem_v_dim: int = -1,
        pk_swilu_projection: bool = True,
        pk_value_fixed_lr: Optional[float] = 0.001,
        pk_value_weight_decay: float = 0.0,
        pk_mem_gated: bool = False,
        pk_peer_variant: bool = False,
        # --- backward-compat (old flags kept) ---
        ffn_dim: int = 2048,
        pk_topk: Optional[int] = 8,
        pk_mem_dim: Optional[int] = None,
        pk_use_gating: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)

        self.d_model = int(d_model)
        self.n_layers = int(n_layers)

        self.head_dim = int(head_dim) if head_dim is not None else None
        self.n_heads = int(n_heads) if n_heads is not None else None
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads is not None else None

        self.multiple_of = int(multiple_of)
        self.ffn_dim_multiplier = float(ffn_dim_multiplier) if ffn_dim_multiplier is not None else None

        self.dropout = float(dropout)
        self.rope_theta = float(rope_theta)

        self.init_base_std = float(init_base_std) if init_base_std is not None else None
        self.init_std_factor = str(init_std_factor)

        self.pk_is_enabled = bool(pk_is_enabled)
        self.pk_layers = list(pk_layers) if pk_layers is not None else []

        self.pk_mem_n_keys = int(pk_mem_n_keys)
        self.pk_mem_heads = int(pk_mem_heads)

        if pk_mem_knn is None:
            pk_mem_knn = int(pk_topk) if pk_topk is not None else 8
        self.pk_mem_knn = int(pk_mem_knn)

        self.pk_mem_share_values = bool(pk_mem_share_values)
        self.pk_mem_k_dim = int(pk_mem_k_dim)

        self.pk_mem_v_dim = int(pk_mem_v_dim)

        self.pk_swilu_projection = bool(pk_swilu_projection)
        self.pk_value_fixed_lr = float(pk_value_fixed_lr) if pk_value_fixed_lr is not None else None
        self.pk_value_weight_decay = float(pk_value_weight_decay)

        self.use_cache = bool(use_cache)
        self.is_encoder_decoder = False
        self.is_decoder = True

        self.pk_mem_gated = bool(pk_mem_gated or pk_use_gating)
        self.pk_peer_variant = bool(pk_peer_variant)

        self.ffn_dim = int(ffn_dim)
        self.pk_topk = int(pk_topk) if pk_topk is not None else 8
        self.pk_mem_dim = int(pk_mem_dim) if pk_mem_dim is not None else None
        self.pk_use_gating = bool(pk_use_gating)