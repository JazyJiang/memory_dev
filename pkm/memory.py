from __future__ import annotations

import math
from logging import getLogger
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = getLogger()


class xFormerEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if indices.dim() != 2 or scores.dim() != 2:
            raise ValueError(f"expected [B, K] indices/scores; got {indices.shape=} {scores.shape=}")
        if indices.shape != scores.shape:
            raise ValueError(f"indices/scores shape mismatch: {indices.shape} vs {scores.shape}")

        emb = self.weight.index_select(0, indices.reshape(-1)).view(
            indices.shape[0], indices.shape[1], self.weight.shape[1]
        )
        return (emb * scores.unsqueeze(-1)).sum(dim=1)


class QueryMLP(nn.Module):
    def __init__(self, input_dim, heads, k_dim, sizes, bias=False, batchnorm=False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.heads = int(heads)
        self.k_dim = int(k_dim)
        self.sizes = sizes

        assert sizes[0] == self.input_dim
        assert sizes[-1] == (self.heads * self.k_dim)

        self.query_mlps = QueryMLP.mlp(list(sizes), bias=bias, batchnorm=batchnorm)

    @staticmethod
    def mlp(sizes, bias=True, batchnorm=True):
        assert len(sizes) >= 2
        pairs = [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        layers = []
        for i, (dim_in, dim_out) in enumerate(pairs):
            layers.append(nn.Linear(dim_in, dim_out, bias=bias))
            if batchnorm:
                layers.append(nn.BatchNorm1d(dim_out))
            if i < len(pairs) - 1:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[-1] == self.input_dim
        input = input.contiguous().view(-1, self.input_dim) if input.dim() > 2 else input
        bs = len(input)

        outputs = [m(input) for m in self.query_mlps]
        query = torch.cat(outputs, 1) if len(outputs) > 1 else outputs[0]
        assert query.shape == (bs, self.heads * self.k_dim)
        return query.view(bs * self.heads, self.k_dim)


class HashingMemory(nn.Module):
    VALUES = None
    EVAL_MEMORY = True

    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        *,
        value_fixed_lr: Optional[float] = 0.001,
        mem_k_dim: int = 512,
        mem_v_dim: int = -1,
        mem_heads: int = 4,
        mem_knn: int = 32,
        mem_share_values: bool = True,
        mem_n_keys: int = 1024,
        mem_query_bias: bool = True,
        mem_query_batchnorm: bool = False,
        mem_gated: bool = False,
        mem_input_dropout: float = 0.0,
        mem_query_dropout: float = 0.0,
        mem_value_dropout: float = 0.0,
        peer_variant: bool = False,
        swilu_projection: bool = True,
        # --- backward-compat shim (old minimal API) ---
        model_dim: Optional[int] = None,
        topk: Optional[int] = None,
        mem_dim: Optional[int] = None,
        use_gating: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__()

        if kwargs:
            raise TypeError(f"Unexpected kwargs: {sorted(kwargs.keys())}")

        if model_dim is not None:
            input_dim = int(model_dim) if input_dim is None else int(input_dim)
            output_dim = int(model_dim) if output_dim is None else int(output_dim)
        if input_dim is None or output_dim is None:
            raise ValueError("HashingMemory requires input_dim/output_dim (or model_dim for backward-compat).")

        if topk is not None:
            mem_knn = int(topk)
        if mem_dim is not None:
            mem_v_dim = int(mem_dim)
        if use_gating is not None:
            mem_gated = bool(use_gating)

        assert mem_k_dim >= 2 and mem_k_dim % 2 == 0
        assert mem_heads >= 2
        assert 0 <= mem_input_dropout < 1
        assert 0 <= mem_query_dropout < 1
        assert 0 <= mem_value_dropout < 1
        assert not (peer_variant and mem_v_dim > 0), "PEER requires mem_v_dim=-1"

        self.use_peer_variant = peer_variant

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.size = int(mem_n_keys) ** 2
        self.k_dim = int(mem_k_dim)

        self.v_dim = int(mem_v_dim) if mem_v_dim > 0 else self.output_dim
        self.heads = int(mem_heads)
        self.knn = int(mem_knn)

        self.mem_share_values = bool(mem_share_values)
        self.original = (not self.mem_share_values) or (HashingMemory.VALUES is None)

        self.swilu_proj = bool(swilu_projection)
        self.v_proj = (mem_v_dim > 0) or self.swilu_proj
        self.value_fixed_lr = value_fixed_lr

        self.input_dropout = float(mem_input_dropout)
        self.query_dropout = float(mem_query_dropout)
        self.value_dropout = float(mem_value_dropout)

        # keys: (2 * heads * n_keys, half)
        self.keys = nn.Parameter(torch.empty(2 * self.heads * int(self.size**0.5), self.k_dim // 2))

        if self.original:
            if not self.use_peer_variant:
                self.values = xFormerEmbeddingBag(self.size, self.v_dim)
                HashingMemory.VALUES = self.values
            else:
                self.values_u = nn.Embedding(self.size, self.v_dim)
                self.values_v = nn.Embedding(self.size, self.v_dim)
                HashingMemory.VALUES = (self.values_u, self.values_v)
        else:
            if not self.use_peer_variant:
                self.values = None
            else:
                self.values_u = None
                self.values_v = None

        if self.v_proj:
            proj_input = mem_v_dim
            if self.swilu_proj and proj_input < 0:
                proj_input = self.output_dim
            self.value_proj = nn.Linear(proj_input, self.output_dim)
        else:
            self.value_proj = None

        if self.swilu_proj:
            proj_input = mem_v_dim if mem_v_dim > 0 else self.output_dim
            self.swilu_projection = nn.Linear(self.input_dim, proj_input)
        else:
            self.swilu_projection = None

        self.gating = nn.Linear(self.input_dim, 1) if mem_gated else None

        l_sizes = (self.input_dim, self.heads * self.k_dim)
        self.query_proj = QueryMLP(
            self.input_dim,
            self.heads,
            self.k_dim,
            l_sizes,
            bias=mem_query_bias,
            batchnorm=mem_query_batchnorm,
        )

        self.reset_parameters()

        if self.original and self.value_fixed_lr is not None:
            if self.use_peer_variant:
                for p in self.values_u.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
                for p in self.values_v.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True
            else:
                for p in self.values.parameters():
                    p.fixed_lr = self.value_fixed_lr
                    p.pk_value_param = True

    def reset_parameters(self, init_std=None, factor: float = 1.0):
        bound = 1 / math.sqrt(self.k_dim)
        nn.init.uniform_(self.keys, a=-bound, b=bound)

        if self.original:
            if not self.use_peer_variant:
                nn.init.normal_(self.values.weight, mean=0, std=self.v_dim**-0.5)
            else:
                nn.init.normal_(self.values_u.weight, mean=0, std=self.v_dim**-0.5)
                nn.init.normal_(self.values_v.weight, mean=0, std=self.v_dim**-0.5)

        layer_std = init_std or (self.output_dim ** (-0.5))
        layer_std = layer_std / float(factor)

        if len(self.query_proj.query_mlps) > 0 and hasattr(self.query_proj.query_mlps[0], "weight"):
            nn.init.trunc_normal_(
                self.query_proj.query_mlps[0].weight,
                mean=0.0,
                std=layer_std,
                a=-3 * layer_std,
                b=3 * layer_std,
            )
            if getattr(self.query_proj.query_mlps[0], "bias", None) is not None:
                nn.init.zeros_(self.query_proj.query_mlps[0].bias)

        if self.value_proj is not None:
            nn.init.trunc_normal_(
                self.value_proj.weight,
                mean=0.0,
                std=layer_std,
                a=-3 * layer_std,
                b=3 * layer_std,
            )
            if self.value_proj.bias is not None:
                nn.init.zeros_(self.value_proj.bias)

        if self.swilu_projection is not None:
            nn.init.trunc_normal_(
                self.swilu_projection.weight,
                mean=0.0,
                std=layer_std,
                a=-3 * layer_std,
                b=3 * layer_std,
            )
            if self.swilu_projection.bias is not None:
                nn.init.zeros_(self.swilu_projection.bias)

        if self.gating is not None:
            nn.init.trunc_normal_(
                self.gating.weight,
                mean=0.0,
                std=layer_std,
                a=-3 * layer_std,
                b=3 * layer_std,
            )
            if self.gating.bias is not None:
                nn.init.zeros_(self.gating.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, C = input.shape
        input = input.view(-1, self.input_dim)
        assert input.shape[-1] == self.input_dim
        prefix_shape = input.shape[:-1]
        bs = int(np.prod(prefix_shape))

        input = F.dropout(input, p=self.input_dropout, training=self.training)
        query = self.query_proj(input)
        query = F.dropout(query, p=self.query_dropout, training=self.training)
        assert query.shape == (bs * self.heads, self.k_dim)

        scores, indices = self.get_indices(query, self.knn)

        scores = F.softmax(scores.float(), dim=-1).type_as(scores)

        with torch.no_grad():
            p = scores.float().clamp_min(1e-9)
            ent = (-(p * p.log()).sum(dim=-1)).mean()
            ent_norm = ent / math.log(float(self.knn)) if self.knn > 1 else ent
            uniq_ratio = float(indices.unique().numel()) / float(indices.numel())
            self._last_stats = {
                "score_entropy": float(ent.item()),
                "score_entropy_norm": float(ent_norm.item()),
                "indices_unique_ratio": float(uniq_ratio),
                "score_max": float(scores.float().max().item()),
            }

        indices = indices.view(bs, self.heads * self.knn)
        scores = scores.view(bs, self.heads * self.knn)

        if not self.use_peer_variant:
            values_module = self.values if self.original else HashingMemory.VALUES
            output = values_module(indices, scores)

            if self.swilu_projection is not None:
                output = self.value_proj(output * F.silu(self.swilu_projection(input)))
            elif self.value_proj is not None:
                output = self.value_proj(output)
        else:
            values_u, values_v = (self.values_u, self.values_v) if self.original else HashingMemory.VALUES
            u = values_u(indices)
            x = torch.einsum("bh, blh->bl", input, u)
            x = F.gelu(x)
            v = values_v(indices)
            x = x * scores
            output = torch.einsum("bl, blh->bh", x, v)

        output = F.dropout(output, p=self.value_dropout, training=self.training)

        if len(prefix_shape) >= 2:
            output = output.view(prefix_shape + (self.v_dim,))

        if self.gating is not None:
            output = torch.sigmoid(self.gating(input)) * output

        output = output.view(B, T, -1)

        with torch.no_grad():
            out_rms = output.float().pow(2).mean().sqrt()
            self._last_stats = dict(getattr(self, "_last_stats", {}))
            self._last_stats.update(
                {
                    "out_rms": float(out_rms.item()),
                }
            )

        return output

    def get_indices(self, query: torch.Tensor, knn: int):
        assert query.dim() == 2 and query.size(1) == self.k_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.k_dim)

        half = self.k_dim // 2
        keys = self.keys.view(self.heads, 2, -1, half)
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

        q1 = query[:, :, :half]
        q2 = query[:, :, half:]

        scores1 = torch.einsum("blh, lkh->blk", q1, keys1)
        scores2 = torch.einsum("blh, lkh->blk", q2, keys2)

        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)

        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(bs, self.heads, -1)

        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn) * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(bs, self.heads, -1)

        scores, best_indices = torch.topk(all_scores, k=knn, dim=2, largest=True, sorted=True)
        indices = all_indices.gather(2, best_indices)

        assert scores.shape == indices.shape == (bs, self.heads, knn)
        return scores.view(bs * self.heads, knn), indices.view(bs * self.heads, knn)