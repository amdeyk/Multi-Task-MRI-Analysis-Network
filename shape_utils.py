"""Utility functions for enforcing a common tensor shape convention.

All 3D volumes in the project use the ``(B, C, D, H, W)`` layout.  The helpers
in this module provide lightweight validation and optional debugging utilities
for tracing tensor shapes through the network.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import torch

BCDHW = Tuple[int, int, int, int, int]


def check_bcdhw(x: torch.Tensor, name: str = "tensor") -> None:
    """Validate that ``x`` has 5 dimensions in ``(B, C, D, H, W)`` order.

    Parameters
    ----------
    x: torch.Tensor
        Tensor to validate.
    name: str, optional
        Human readable name used in error messages.
    """
    if x.dim() != 5:
        raise ValueError(f"{name} must be rank 5 (B, C, D, H, W); got {tuple(x.shape)}")


def ensure_bcdhw(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Return ``x`` in ``(B, C, D, H, W)`` order.

    The function accepts tensors with channels-last layout ``(B, D, H, W, C)`` and
    automatically permutes them.  Four dimensional inputs are interpreted as
    missing the channel dimension and a singleton one is inserted.
    """
    if x.dim() == 4:  # (B, D, H, W)
        x = x.unsqueeze(1)
    elif x.dim() == 5 and x.shape[1] > 4 and x.shape[-1] <= 4:
        # Heuristic for channels-last volumes
        x = x.permute(0, 4, 1, 2, 3)
    check_bcdhw(x, name)
    return x


def trace_shape(name: str, x: torch.Tensor) -> None:
    """Print a debug statement with ``x``'s shape."""
    print(f"{name} shape: {tuple(x.shape)}")
