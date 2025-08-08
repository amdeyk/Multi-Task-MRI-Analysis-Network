"""Kolmogorov–Arnold Network (KAN) heads implemented in PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor, nn

try:  # pragma: no cover - optional dependency
    from kan import KANLinear  # type: ignore
except Exception:  # pragma: no cover - fall back to simple linear layer
    KANLinear = None


class SOTAKANHead(nn.Module):
    """Simple wrapper around ``KANLinear`` or a PyTorch linear fallback."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        if KANLinear is not None:  # pragma: no cover - optional path
            self.linear = KANLinear(in_dim, out_dim)
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        flat = x.view(-1, d)
        out = self.linear(flat)
        return out.view(b, n, -1)
