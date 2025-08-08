from __future__ import annotations

import torch
from torch import Tensor


class DummyMRIDataset:
    """Return synthetic MRI samples for prototyping."""

    def __init__(
        self,
        n_samples: int = 20,
        contrasts: int = 2,
        slices: int = 16,
        height: int = 64,
        width: int = 64,
    ) -> None:
        self.n_samples = n_samples
        self.contrasts = contrasts
        self.slices = slices
        self.height = height
        self.width = width

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Tensor]:  # pragma: no cover - simple
        mri = torch.randn(self.contrasts, self.slices, self.height, self.width)
        seg = torch.randint(
            0, 2, (self.slices, self.height, self.width), dtype=torch.float32
        )
        cls = torch.randint(0, 2, (1,), dtype=torch.float32)
        edge = torch.randint(
            0, 2, (self.slices, self.height, self.width), dtype=torch.float32
        )
        tumor = torch.randint(
            0, 3, (self.slices, self.height, self.width), dtype=torch.int64
        )
        return {"mri": mri, "seg": seg, "cls": cls, "edge": edge, "tumor": tumor}


def get_samples(n: int = 2) -> list[dict[str, Tensor]]:
    dataset = DummyMRIDataset(n_samples=n)
    return [dataset[i] for i in range(len(dataset))]
