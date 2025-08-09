"""Residual transformer block with attention and feed-forward layers.

This module implements a standard Transformer encoder block tailored for the
3D medical imaging setting.  It exposes a residual connection with an optional
`prev_residual` input that can be used to pass features from the previous
layer.  The implementation follows the architecture introduced in
"Attention is All You Need" with multi-head self-attention, GELU activated
feed-forward networks, layer normalisation and configurable dropout.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class ResidualTransformerBlock(nn.Module):
    """Transformer encoder block with residual connections.

    Parameters
    ----------
    dim: int
        Dimensionality of the input embeddings.
    num_heads: int
        Number of attention heads.
    mlp_ratio: int, optional
        Expansion ratio for the feed-forward network.  Default is ``2``.
    dropout: float, optional
        Dropout probability applied after attention and feed-forward layers.
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 2, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: Tensor,
        prev_residual: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply the transformer block.

        Parameters
        ----------
        x: Tensor
            Input tensor of shape ``(batch, seq_len, dim)``.
        prev_residual: Tensor, optional
            Residual from the previous block.  When provided, the new residual
            connection is formed with this tensor; otherwise ``x`` is used.
        attn_mask: Tensor, optional
            Optional attention mask broadcastable to ``(batch, seq_len, seq_len)``.
        """

        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D) tensor, got {tuple(x.shape)}")
        residual = x if prev_residual is None else prev_residual
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q, attn_mask=attn_mask)
        x = residual + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x
