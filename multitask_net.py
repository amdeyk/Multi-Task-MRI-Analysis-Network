"""Baseline multi-task MRI network wrapper.

This module exposes :class:`MultiTaskMRINet`, a thin convenience wrapper around
:class:`mri_network.BaseMRINet` configured for the lightweight architecture.  It
maintains the original class name for backwards compatibility while delegating
all heavy lifting to the unified implementation.
"""
from __future__ import annotations

from mri_network import BaseMRINet


class MultiTaskMRINet(BaseMRINet):
    """Baseline network built from :class:`BaseMRINet`.

    Parameters mirror those of :class:`BaseMRINet` but default to a configuration
    suitable for small-scale experimentation.
    """

    def __init__(
        self,
        in_channels: int,
        cube_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        n_tasks: int | None = None,
        face_embed: bool = True,
    ) -> None:
        head_channels = {"segmentation": 2, "classification": 1, "edge": 1, "tumor": 3}
        super().__init__(
            in_channels=in_channels,
            cube_size=cube_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            head_channels=head_channels,
            cube_embed_kwargs={"face_embed": face_embed},
        )


def create_multitask_net(**kwargs) -> BaseMRINet:
    """Factory function returning :class:`MultiTaskMRINet`."""
    return MultiTaskMRINet(**kwargs)

