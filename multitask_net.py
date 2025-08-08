from __future__ import annotations

import torch
from torch import Tensor, nn

from differential import DifferentialFeatureExtractor
from cube_embed import CubeSplitter3D
from residual_transformer import ResidualTransformerBlock
from sota_kan import SOTAKANHead


class MultiTaskMRINet(nn.Module):
    """Minimal PyTorch implementation of the multi-task MRI network."""

    def __init__(
        self,
        in_channels: int,
        cube_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        n_tasks: int,
        face_embed: bool = True,
    ) -> None:
        super().__init__()
        self.diff_feat = DifferentialFeatureExtractor()
        self.cube_embed = CubeSplitter3D(cube_size=cube_size, face_embed=face_embed)
        self.transformer_layers = nn.ModuleList(
            [ResidualTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.seg_head = SOTAKANHead(embed_dim, 2)
        self.cls_head = SOTAKANHead(embed_dim, 1)
        self.edge_head = SOTAKANHead(embed_dim, 1)
        self.tumor_head = SOTAKANHead(embed_dim, 3)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = torch.as_tensor(x, dtype=torch.float32)
        x = self.diff_feat(x)
        x = self.cube_embed(x)
        prev = None
        for layer in self.transformer_layers:
            x = layer(x, prev)
            prev = x
        seg = self.seg_head(x)
        edge = self.edge_head(x)
        tumor = self.tumor_head(x)
        cls = self.cls_head(x.mean(dim=1, keepdim=True))
        return {
            "segmentation": seg,
            "classification": cls,
            "edge": edge,
            "tumor": tumor,
        }
