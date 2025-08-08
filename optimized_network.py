"""Optimized MRI analysis network with advanced modules."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from attention import FlashAttention
from advanced_kan import ParallelKANHead
from differential import DifferentialFeatureExtractor


class OptimizedCubeEmbedding(nn.Module):
    """3D cube embedding with optional multi-scale overlap."""

    def __init__(self, cube_size: int = 8, embed_dim: int = 128, overlap: float = 0.25, multi_scale: bool = True) -> None:
        super().__init__()
        self.cube_size = cube_size
        self.stride = int(cube_size * (1 - overlap))
        self.multi_scale = multi_scale
        if multi_scale:
            self.embeddings = nn.ModuleList(
                [
                    nn.Conv3d(1, embed_dim // 3, kernel_size=cube_size, stride=self.stride),
                    nn.Conv3d(1, embed_dim // 3, kernel_size=cube_size // 2, stride=max(1, self.stride // 2)),
                    nn.Conv3d(1, embed_dim // 3, kernel_size=cube_size * 2, stride=self.stride * 2),
                ]
            )
        else:
            self.embeddings = nn.ModuleList(
                [nn.Conv3d(1, embed_dim, kernel_size=cube_size, stride=self.stride)]
            )
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        feats = []
        for layer in self.embeddings:
            emb = layer(x.reshape(b * c, 1, d, h, w))
            emb = rearrange(emb, "(b c) e d h w -> b c (d h w) e", b=b, c=c)
            feats.append(emb)
        if self.multi_scale:
            target = feats[0].shape[2]
            aligned = []
            for emb in feats:
                if emb.shape[2] != target:
                    emb = F.interpolate(emb.transpose(-1, -2), size=target, mode="linear").transpose(-1, -2)
                aligned.append(emb)
            x = torch.cat(aligned, dim=-1)
        else:
            x = feats[0]
        x = x + self.pos_embed[:, : x.shape[2], :]
        return x.reshape(b, -1, x.shape[-1])


class MixtureOfExperts(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, num_experts: int = 4) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(dim * mlp_ratio, dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = F.softmax(self.gate(x), dim=-1)
        expert_out = torch.stack([e(x) for e in self.experts], dim=-2)
        return torch.einsum("bne,bnec->bnc", gates, expert_out)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.empty(x.shape[0], 1, 1, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob


class EnhancedTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        if use_moe:
            self.mlp: nn.Module = MixtureOfExperts(dim, mlp_ratio, num_experts)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * mlp_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * mlp_ratio, dim),
                nn.Dropout(dropout),
            )
        self.drop_path = StochasticDepth(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, prev_layer: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x))
        if prev_layer is not None:
            attn_out = attn_out + 0.1 * prev_layer
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SOTAMRINetwork(nn.Module):
    """State-of-the-art MRI analysis network."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.diff_extractor = DifferentialFeatureExtractor()
        self.cube_embed = OptimizedCubeEmbedding(
            cube_size=config.model.cube_size,
            embed_dim=config.model.embed_dim,
            overlap=config.data.overlap,
            multi_scale=True,
        )
        self.transformers = nn.ModuleList(
            [
                EnhancedTransformerBlock(
                    config.model.embed_dim,
                    config.model.num_heads,
                    dropout=config.model.dropout,
                    use_moe=config.model.use_moe,
                    num_experts=config.model.num_experts,
                )
                for _ in range(config.model.num_layers)
            ]
        )
        self.heads = ParallelKANHead(
            config.model.embed_dim,
            {
                "segmentation": 2,
                "classification": 4,
                "edge": 1,
                "tumor": 5,
            },
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.diff_extractor(x)
        x = self.cube_embed(x)
        prev = None
        for layer in self.transformers:
            x = layer(x, prev)
            prev = x
        outputs = self.heads(x)
        return outputs
