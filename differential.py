"""Differential feature extraction utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DifferentialFeatureExtractor(nn.Module):
    """Compute differential features using efficient tensor operations.

    The extractor concatenates the original volume with slice differences,
    channel differences and optional spatial gradients computed with Sobel
    filters. Results are cached when possible to avoid redundant work.
    """

    def __init__(self, spatial_grad: bool = True, method: str = "sobel") -> None:
        super().__init__()
        self.spatial_grad = spatial_grad
        self.method = method
        self.cache: Dict[Tuple[int, ...], Tuple[Tensor, Tensor]] = {}

        if spatial_grad:
            kx = torch.tensor(
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
            )
            ky = torch.tensor(
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
            )
            kx3 = torch.zeros(1, 1, 3, 3, 3)
            ky3 = torch.zeros(1, 1, 3, 3, 3)
            kx3[0, 0, 1] = kx
            ky3[0, 0, 1] = ky
            self.register_buffer("kernel_x", kx3)
            self.register_buffer("kernel_y", ky3)

    def forward(self, x: Tensor) -> Tensor:
        """Return concatenated differential features for ``x``.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, C, S, H, W)``.
        """

        x = torch.as_tensor(x, dtype=torch.float32)
        b, c, s, h, w = x.shape

        diff_slices = x.new_zeros(b, c, s, h, w)
        diff_slices[:, :, 1:].copy_(x[:, :, 1:] - x[:, :, :-1])
        diff_channels = x.new_zeros(b, c, s, h, w)
        diff_channels[:, 1:].copy_(x[:, 1:] - x[:, :-1])

        if self.spatial_grad:
            key = (*x.shape, x.device.index if x.is_cuda else -1)
            if key in self.cache:
                grad_x, grad_y = self.cache[key]
            else:
                weight_x = self.kernel_x.repeat(c, 1, 1, 1, 1)
                weight_y = self.kernel_y.repeat(c, 1, 1, 1, 1)
                grad_x = F.conv3d(x, weight_x, padding=1, groups=c)
                grad_y = F.conv3d(x, weight_y, padding=1, groups=c)
                self.cache[key] = (grad_x, grad_y)
        else:
            grad_x = grad_y = x.new_zeros(b, c, s, h, w)

        return torch.cat([x, diff_slices, diff_channels, grad_x, grad_y], dim=1)
