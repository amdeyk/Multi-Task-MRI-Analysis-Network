"""Unified configurable MRI analysis network.

The :class:`BaseMRINet` provides a common backbone that can be extended with
optional modules to trade off complexity and performance.  All variants operate
on 3D volumes in ``(B, C, D, H, W)`` format.

Typical usage is via the factory functions ``create_basic_net`` for lightweight
experiments and ``create_sota_net`` for the state-of-the-art configuration.
"""
from __future__ import annotations

from typing import Dict, Type, Optional

import torch
from torch import Tensor, nn

from differential import DifferentialFeatureExtractor
from cube_embed import CubeSplitter3D
from residual_transformer import ResidualTransformerBlock
from sota_kan import SOTAKANHead
from shape_utils import ensure_bcdhw, check_bcdhw, trace_shape

try:  # optional dependency
    from advanced_kan import ParallelKANHead
except Exception:  # pragma: no cover - fallback to per-task heads
    ParallelKANHead = None  # type: ignore


class BaseMRINet(nn.Module):
    """Configurable MRI analysis network backbone.

    Parameters
    ----------
    in_channels: int
        Number of input channels in the MRI volume.
    cube_size: int
        Edge length of cubic patches used for tokenisation.
    embed_dim: int
        Dimensionality of the token embeddings.
    num_heads: int
        Number of attention heads per transformer block.
    num_layers: int
        Number of transformer blocks in the model.
    head_channels: Dict[str, int]
        Mapping from task name to number of output channels.
    cube_embed_cls: Type[nn.Module], optional
        Module class used for cube embedding. Defaults to :class:`CubeSplitter3D`.
    transformer_block_cls: Type[nn.Module], optional
        Class implementing a transformer block. Defaults to
        :class:`ResidualTransformerBlock`.
    cube_embed_kwargs: dict, optional
        Additional keyword arguments passed to ``cube_embed_cls``.
    transformer_kwargs: dict, optional
        Additional keyword arguments passed to ``transformer_block_cls``.
    parallel_heads: bool, optional
        If ``True`` and :class:`~advanced_kan.ParallelKANHead` is available, a
        single parallel head processes all tasks simultaneously.
    trace_shapes: bool, optional
        When enabled, shapes at key points in the forward pass are printed for
        debugging.
    """

    def __init__(
        self,
        in_channels: int,
        cube_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        head_channels: Dict[str, int],
        *,
        cube_embed_cls: Type[nn.Module] = CubeSplitter3D,
        transformer_block_cls: Type[nn.Module] = ResidualTransformerBlock,
        cube_embed_kwargs: Optional[Dict] = None,
        transformer_kwargs: Optional[Dict] = None,
        parallel_heads: bool = False,
        trace_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.trace_shapes = trace_shapes
        self.parallel_heads = parallel_heads and ParallelKANHead is not None

        self.diff_feat = DifferentialFeatureExtractor()
        cube_kwargs = {"cube_size": cube_size, "embed_dim": embed_dim}
        if cube_embed_kwargs:
            cube_kwargs.update(cube_embed_kwargs)
        self.cube_embed = cube_embed_cls(**cube_kwargs)

        trans_kwargs: Dict = {"dim": embed_dim, "num_heads": num_heads}
        if transformer_kwargs:
            trans_kwargs.update(transformer_kwargs)
        self.transformers = nn.ModuleList(
            [transformer_block_cls(**trans_kwargs) for _ in range(num_layers)]
        )

        if self.parallel_heads:
            self.heads = ParallelKANHead(embed_dim, head_channels)  # type: ignore[arg-type]
        else:
            self.heads = nn.ModuleDict(
                {t: SOTAKANHead(embed_dim, c) for t, c in head_channels.items()}
            )

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Run the network on ``x``.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape ``(B, C, D, H, W)``.  Channels-last inputs
            are automatically rearranged.
        """

        x = ensure_bcdhw(x, "input")
        if self.trace_shapes:
            trace_shape("input", x)

        x = self.diff_feat(x)
        check_bcdhw(x, "diff_feat")
        if self.trace_shapes:
            trace_shape("after_diff", x)

        x = self.cube_embed(x)
        if self.trace_shapes:
            trace_shape("after_embed", x)

        prev: Optional[Tensor] = None
        for i, layer in enumerate(self.transformers):
            x = layer(x, prev)
            prev = x
            if self.trace_shapes:
                trace_shape(f"layer_{i}", x)

        if self.parallel_heads:
            outputs = self.heads(x)  # type: ignore[operator]
        else:
            outputs = {}
            for name, head in self.heads.items():
                inp = x if name != "classification" else x.mean(dim=1, keepdim=True)
                outputs[name] = head(inp)
        return outputs


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_basic_net(**kwargs: int) -> BaseMRINet:
    """Return a lightweight baseline network.

    Use this variant when experimenting on limited hardware or when rapid
    prototyping is desired.
    """
    head_channels = {"segmentation": 2, "classification": 1, "edge": 1, "tumor": 3}
    return BaseMRINet(head_channels=head_channels, **kwargs)


def create_sota_net(config) -> BaseMRINet:
    """Return the state-of-the-art network configuration.

    Parameters
    ----------
    config: Config
        Application configuration object providing model and data settings.
    """
    from optimized_network import OptimizedCubeEmbedding, EnhancedTransformerBlock

    head_channels = {"segmentation": 2, "classification": 4, "edge": 1, "tumor": 5}
    cube_kwargs = {
        "cube_size": config.model.cube_size,
        "embed_dim": config.model.embed_dim,
        "overlap": config.data.overlap,
        "multi_scale": True,
    }
    trans_kwargs = {
        "mlp_ratio": 4,
        "dropout": config.model.dropout,
        "use_moe": config.model.use_moe,
        "num_experts": config.model.num_experts,
    }
    return BaseMRINet(
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
