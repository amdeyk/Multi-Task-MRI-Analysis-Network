"""Optimized attention mechanisms."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FlashAttention(nn.Module):
    """Memory-efficient attention using PyTorch's SDP kernels."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=True
        ):
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class SparseAttention(nn.Module):
    """Sparse attention with learnable patterns."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.register_parameter(
            "attention_bias", nn.Parameter(torch.zeros(num_heads, window_size, window_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_weights = []
        for i in range(0, n, self.window_size):
            end = min(i + self.window_size, n)
            q_w = q[:, :, i:end]
            k_w = k[:, :, i:end]
            v_w = v[:, :, i:end]
            attn = (q_w @ k_w.transpose(-2, -1)) * (self.head_dim**-0.5)
            attn = attn + self.attention_bias[:, : end - i, : end - i]
            attn = F.softmax(attn, dim=-1)
            out = attn @ v_w
            attn_weights.append(out)
        out = torch.cat(attn_weights, dim=2)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.proj(out)
