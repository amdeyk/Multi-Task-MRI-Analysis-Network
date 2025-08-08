"""Utilities for embedding 3D volumes into cube representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CubeSplitter3D(nn.Module):
    """Split volumes into non-overlapping cubes and project them."""

    def __init__(
        self, cube_size: int = 8, embed_dim: int = 32, face_embed: bool = False
    ) -> None:
        super().__init__()
        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.face_embed = face_embed
        self.proj = nn.Parameter(torch.randn(cube_size**3, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Return embedded cubes for ``x``."""

        x = torch.as_tensor(x, dtype=torch.float32)
        b, c, s, h, w = x.shape
        cubes: list[Tensor] = []
        cs = self.cube_size
        for bi in range(b):
            for ci in range(c):
                for zz in range(0, s, cs):
                    for yy in range(0, h, cs):
                        for xx in range(0, w, cs):
                            cube = x[bi, ci, zz : zz + cs, yy : yy + cs, xx : xx + cs]
                            if cube.shape == (cs, cs, cs):
                                cubes.append(cube.reshape(-1))
        if not cubes:
            return x.new_zeros((b, 0, self.embed_dim))
        cubes = torch.stack(cubes, dim=0) @ self.proj
        n_per_sample = cubes.shape[0] // b
        return cubes.view(b, n_per_sample, -1)
