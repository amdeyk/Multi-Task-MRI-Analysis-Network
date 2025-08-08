"""Simple residual transformer block."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualTransformerBlock(nn.Module):
    """Placeholder transformer block using PyTorch operations.

    The block merely mixes the current input with the previous residual to
    keep the example lightweight.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.1
    ) -> None:  # noqa: D401
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, prev_residual: Tensor | None = None) -> Tensor:
        if prev_residual is None:
            return x
        return 0.5 * (x + prev_residual)
