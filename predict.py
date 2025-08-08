"""Prediction utilities for the MRI network."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def predict(model: nn.Module, mri_vol: Tensor) -> dict[str, Tensor]:
    """Run a forward pass and apply basic activations."""

    mri_vol = torch.as_tensor(mri_vol, dtype=torch.float32)
    outputs = model(mri_vol)
    outputs["segmentation"] = torch.softmax(outputs["segmentation"], dim=-1)
    outputs["classification"] = torch.sigmoid(outputs["classification"])
    outputs["edge"] = torch.sigmoid(outputs["edge"])
    outputs["tumor"] = torch.softmax(outputs["tumor"], dim=-1)
    return outputs
