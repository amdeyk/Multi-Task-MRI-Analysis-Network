from __future__ import annotations

import torch
from torch import Tensor, nn
from typing import Sequence


def train_one_epoch(model: nn.Module, data: Sequence[dict[str, Tensor]]) -> float:
    """Very small dummy training loop used for examples."""

    loss = 0.0
    model.train()
    for batch in data:
        mri = torch.as_tensor(batch["mri"][None], dtype=torch.float32)
        out = model(mri)
        loss += float(out["classification"].mean())
    return loss / max(len(data), 1)


def validate(model: nn.Module, data: Sequence[dict[str, Tensor]]) -> float:
    """Dummy validation pass."""

    model.eval()
    with torch.no_grad():
        return train_one_epoch(model, data)
