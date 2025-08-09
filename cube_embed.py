"""Utilities for embedding 3D volumes into cube representations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from shape_utils import ensure_bcdhw


class CubeSplitter3D(nn.Module):
    """Split volumes into cubes and project them with vectorised ops."""

    def __init__(
        self,
        cube_size: int = 8,
        embed_dim: int = 32,
        face_embed: bool = False,
        overlap: int = 0,
        adaptive_batch: bool = True,
    ) -> None:
        super().__init__()
        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.face_embed = face_embed
        self.overlap = overlap
        self.stride = cube_size - overlap
        self.adaptive_batch = adaptive_batch
        self.proj = nn.Parameter(torch.randn(cube_size**3, embed_dim))

    # ------------------------------------------------------------------ utils
    def _to_tensor(self, x: Union[Tensor, np.ndarray, str, Path]) -> Tensor:
        """Load tensor ``x`` supporting memory-mapped files and lazy loading."""

        if isinstance(x, (str, Path)):
            x = np.load(str(x), mmap_mode="r")
        if isinstance(x, np.memmap):
            x = torch.from_numpy(x)
        return torch.as_tensor(x, dtype=torch.float32)

    def _available_memory(self, device: torch.device) -> int:
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            return int(free)
        return int(1e12)  # effectively infinity on CPU

    # ------------------------------------------------------------------ forward
    def forward(self, x: Union[Tensor, np.ndarray, str, Path]) -> Tensor:
        """Return embedded cubes for ``x``.

        Parameters
        ----------
        x:
            Input volume in ``(B, C, D, H, W)`` format.  Non-conforming inputs are
            automatically reshaped using :func:`~shape_utils.ensure_bcdhw`.

        The operation is fully vectorised using :meth:`Tensor.unfold` and
        supports overlapping patches. Large volumes residing on disk can be
        passed as filenames or ``np.memmap`` instances to enable lazy loading.
        ``adaptive_batch`` processes patches in chunks based on free GPU memory.
        """

        x = ensure_bcdhw(self._to_tensor(x), "x")
        b, c, s, h, w = x.shape
        cs, stride = self.cube_size, self.stride
        patches = (
            x.unfold(2, cs, stride)
            .unfold(3, cs, stride)
            .unfold(4, cs, stride)
            .contiguous()
        )
        patches = patches.view(b, c, -1, cs**3)
        patches = patches.view(b * c, -1, cs**3)

        if self.adaptive_batch and patches.is_cuda:
            free_mem = self._available_memory(patches.device)
            bytes_per_patch = cs**3 * patches.element_size() + self.embed_dim * 4
            max_patches = max(1, free_mem // bytes_per_patch)
            chunks = patches.split(max_patches, dim=1)
            embed_chunks = [chunk @ self.proj for chunk in chunks]
            cubes = torch.cat(embed_chunks, dim=1)
        else:
            cubes = patches @ self.proj

        cubes = cubes.view(b, c, -1, self.embed_dim)
        return cubes.view(b, -1, self.embed_dim)

    # ---------------------------------------------------------------- reconstruct
    def reconstruct(self, patches: Tensor, volume_shape: Iterable[int]) -> Tensor:
        """Reconstruct volume from embedded ``patches`` with overlap averaging."""

        b, c, s, h, w = volume_shape
        cs, stride = self.cube_size, self.stride
        n_per_chan = patches.shape[1] // c
        cubes = (patches.view(b * c, n_per_chan, self.embed_dim) @ self.proj.t()).view(
            b, c, n_per_chan, cs, cs, cs
        )

        nz = (s - cs) // stride + 1
        ny = (h - cs) // stride + 1
        nx = (w - cs) // stride + 1
        vol = torch.zeros(b, c, s, h, w, device=patches.device)
        count = torch.zeros_like(vol)

        idx = 0
        for zz in range(nz):
            for yy in range(ny):
                for xx in range(nx):
                    zs, ys, xs = zz * stride, yy * stride, xx * stride
                    vol[:, :, zs : zs + cs, ys : ys + cs, xs : xs + cs] += cubes[:, :, idx]
                    count[:, :, zs : zs + cs, ys : ys + cs, xs : xs + cs] += 1
                    idx += 1
        return vol / count.clamp_min(1)
