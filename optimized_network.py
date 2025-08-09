"""Optimized MRI analysis network with advanced modules."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from attention import FlashAttention
from shape_utils import ensure_bcdhw


class AdaptivePatchEmbedding(nn.Module):
    """Memory-aware 3D patch embedding with sliding convolutions.

    The module replaces earlier cube-based extraction with a sliding window
    approach built on ``nn.Conv3d``.  Volumes are processed in depth chunks
    to fit available memory and, when distributed training is initialised,
    slices are divided across ranks and gathered after embedding.
    """

    def __init__(
        self,
        cube_size: int = 8,
        embed_dim: int = 128,
        overlap: float = 0.25,
        multi_scale: bool = True,
    ) -> None:
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

    def _free_mem(self, device: torch.device) -> int:
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return int(free)
        return int(1e12)

    def _apply_layer(self, layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply ``layer`` to ``x`` in depth chunks based on free memory."""

        b, c, d, h, w = x.shape
        device = x.device
        bytes_per_voxel = x.element_size() * layer.out_channels
        free_mem = self._free_mem(device) * 0.8
        max_depth = max(1, int(free_mem / (bytes_per_voxel * h * w * b)))

        outputs: list[torch.Tensor] = []
        for start in range(0, d, max_depth):
            chunk = x[:, :, start : start + max_depth]
            out = layer(chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        x = ensure_bcdhw(x, "x")
        b, c, d, h, w = x.shape
        pad = self.cube_size - self.stride
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad, 0, pad))

        if dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
            chunk = (d + world - 1) // world
            start = rank * chunk
            end = min(start + chunk, d)
            x_local = x[:, :, start:end]
            if end - start < chunk:
                x_local = F.pad(x_local, (0, 0, 0, 0, 0, chunk - (end - start)))
        else:
            x_local = x

        feats = []
        for layer in self.embeddings:
            emb = self._apply_layer(layer, x_local.reshape(b * c, 1, x_local.shape[2], x_local.shape[3], x_local.shape[4]))
            emb = rearrange(emb, "(b c) e d h w -> b c (d h w) e", b=b, c=c)
            feats.append(emb)

        if self.multi_scale:
            target = feats[0].shape[2]
            aligned = []
            for emb in feats:
                if emb.shape[2] != target:
                    emb = F.interpolate(
                        emb.transpose(-1, -2), size=target, mode="linear"
                    ).transpose(-1, -2)
                aligned.append(emb)
            out = torch.cat(aligned, dim=-1)
        else:
            out = feats[0]

        if dist.is_initialized():
            gather_list = [torch.zeros_like(out) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_list, out)
            out = torch.cat(gather_list, dim=2)
            d_tokens = (d + pad - self.cube_size) // self.stride + 1
            h_tokens = (h + pad - self.cube_size) // self.stride + 1
            w_tokens = (w + pad - self.cube_size) // self.stride + 1
            tokens = d_tokens * h_tokens * w_tokens
            out = out[:, :, :tokens]

        out = out + self.pos_embed[:, : out.shape[2], :]
        return out.reshape(b, -1, out.shape[-1])


# Backwards compatibility
OptimizedCubeEmbedding = AdaptivePatchEmbedding


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
        checkpoint: bool | str = False,
        checkpoint_mlp: bool = True,
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
        self.checkpoint = checkpoint
        self.checkpoint_mlp = checkpoint_mlp

    def _should_checkpoint(self) -> bool:
        if self.checkpoint == "auto":
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return free / total < 0.5
            return False
        return bool(self.checkpoint)

    def forward(self, x: torch.Tensor, prev_layer: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, N, D) tensor, got {tuple(x.shape)}")
        use_ckpt = self._should_checkpoint()

        def attn_fn(t: torch.Tensor) -> torch.Tensor:
            return self.attn(self.norm1(t))

        if use_ckpt:
            attn_out = torch.utils.checkpoint.checkpoint(attn_fn, x)
        else:
            attn_out = attn_fn(x)

        if prev_layer is not None:
            attn_out = attn_out + 0.1 * prev_layer
        x = x + self.drop_path(attn_out)

        def mlp_fn(t: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm2(t))

        if use_ckpt and self.checkpoint_mlp:
            mlp_out = torch.utils.checkpoint.checkpoint(mlp_fn, x)
        else:
            mlp_out = mlp_fn(x)

        x = x + self.drop_path(mlp_out)
        return x


from mri_network import BaseMRINet


class SOTAMRINetwork(BaseMRINet):
    """State-of-the-art MRI analysis network using advanced modules."""

    def __init__(self, config) -> None:
        head_channels = {"segmentation": 2, "classification": 4, "edge": 1, "tumor": 5}
        cube_kwargs = {
            "overlap": config.data.overlap,
            "multi_scale": True,
            "cube_size": config.model.cube_size,
            "embed_dim": config.model.embed_dim,
        }
        trans_kwargs = {
            "mlp_ratio": 4,
            "dropout": config.model.dropout,
            "use_moe": config.model.use_moe,
            "num_experts": config.model.num_experts,
        }
        super().__init__(
            in_channels=config.model.in_channels,
            cube_size=config.model.cube_size,
            embed_dim=config.model.embed_dim,
            num_heads=config.model.num_heads,
            num_layers=config.model.num_layers,
            head_channels=head_channels,
            cube_embed_cls=OptimizedCubeEmbedding,
            transformer_block_cls=EnhancedTransformerBlock,
            cube_embed_kwargs=cube_kwargs,
            transformer_kwargs=trans_kwargs,
            parallel_heads=True,
        )
