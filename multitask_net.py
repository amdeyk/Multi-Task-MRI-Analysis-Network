import numpy as np
from differential import DifferentialFeatureExtractor
from cube_embed import CubeSplitter3D
from residual_transformer import ResidualTransformerBlock
from sota_kan import SOTAKANHead

class MultiTaskMRINet:
    """Minimal NumPy implementation of the multi-task MRI network."""

    def __init__(self, in_channels: int, cube_size: int, embed_dim: int,
                 num_heads: int, num_layers: int, n_tasks: int,
                 face_embed: bool = True) -> None:
        self.diff_feat = DifferentialFeatureExtractor()
        self.cube_embed = CubeSplitter3D(cube_size=cube_size, face_embed=face_embed)
        self.transformer_layers = [
            ResidualTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ]
        self.seg_head = SOTAKANHead(embed_dim, 2)
        self.cls_head = SOTAKANHead(embed_dim, 1)
        self.edge_head = SOTAKANHead(embed_dim, 1)
        self.tumor_head = SOTAKANHead(embed_dim, 3)

    def forward(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = self.diff_feat.forward(x)
        x = self.cube_embed.forward(x)
        prev = None
        for layer in self.transformer_layers:
            x = layer.forward(x, prev)
            prev = x
        seg = self.seg_head.forward(x)
        edge = self.edge_head.forward(x)
        tumor = self.tumor_head.forward(x)
        cls = self.cls_head.forward(x.mean(axis=1, keepdims=True))
        return {
            "segmentation": seg,
            "classification": cls,
            "edge": edge,
            "tumor": tumor,
        }
