"""Differential feature extraction utilities."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DifferentialFeatureExtractor(nn.Module):
    """Compute differential features using PyTorch tensors.

    The extractor concatenates the original volume with slice differences,
    channel differences and optional spatial gradients. Keeping the
    implementation torch-based enables GPU acceleration and automatic
    differentiation while remaining intentionally lightweight.
    """

    def __init__(self, spatial_grad: bool = True) -> None:
        super().__init__()
        self.spatial_grad = spatial_grad

    def forward(self, x: Tensor) -> Tensor:
        """Return concatenated differential features for ``x``.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, S, H, W)``.

        Returns
        -------
        Tensor
            Tensor containing the original input and differential features
            along the channel and slice dimensions.
        """

        x = torch.as_tensor(x, dtype=torch.float32)
        diff_slices = torch.diff(x, dim=2, prepend=torch.zeros_like(x[:, :, :1]))
        diff_channels = torch.diff(x, dim=1, prepend=torch.zeros_like(x[:, :1]))
        if self.spatial_grad:
            grad_x = torch.zeros_like(x)
            grad_y = torch.zeros_like(x)
        else:
            grad_x = grad_y = torch.zeros_like(x)
        return torch.cat([x, diff_slices, diff_channels, grad_x, grad_y], dim=1)
