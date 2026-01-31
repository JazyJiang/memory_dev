from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ProductKeyArgs:
    is_enabled: bool = False
    pk_layers: List[int] = None

    mem_n_keys: int = 128
    topk: int = 8
    mem_dim: int = 256

    use_gating: bool = False

    def __post_init__(self) -> None:
        if self.pk_layers is None:
            self.pk_layers = []
        if self.mem_n_keys <= 0:
            raise ValueError("mem_n_keys must be > 0")
        if self.topk <= 0:
            raise ValueError("topk must be > 0")
        if self.mem_dim <= 0:
            raise ValueError("mem_dim must be > 0")