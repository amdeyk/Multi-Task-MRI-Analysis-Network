"""Training utilities for the MRI analysis network."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    clip_grad: float = 0.0,
) -> float:
    """Train ``model`` for a single epoch.

    The function performs a standard training loop with optional mixed
    precision, gradient clipping and learning rate scheduling.  It returns the
    average training loss over the epoch.
    """

    model.train()
    epoch_loss = 0.0
    num_batches = 0
    for batch in loader:
        mri = batch["mri"].to(device)
        targets = {
            k: v.to(device) for k, v in batch.items() if k in {"seg", "cls", "edge", "tumor"}
        }
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(mri)
                losses = loss_fn(outputs, targets)
                loss = losses["total"]
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            epoch_loss += float(loss.detach())
            num_batches += 1
        except RuntimeError as exc:  # pragma: no cover - hardware specific
            if "out of memory" in str(exc):
                torch.cuda.empty_cache()
                continue
            raise
    if scheduler is not None:
        scheduler.step()
    return epoch_loss / max(1, num_batches)


def _dice_coefficient(pred: Tensor, target: Tensor) -> float:
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return 2 * intersection / (union + 1e-8)


def validate(
    model: nn.Module,
    loader: Iterable[Dict[str, Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate ``model`` on ``loader`` without gradient computation."""

    model.eval()
    val_loss = 0.0
    dice_scores: list[float] = []
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            mri = batch["mri"].to(device)
            targets = {
                k: v.to(device) for k, v in batch.items() if k in {"seg", "cls", "edge", "tumor"}
            }
            outputs = model(mri)
            losses = loss_fn(outputs, targets)
            val_loss += float(losses["total"].detach())
            pred = outputs["segmentation"].argmax(dim=1).cpu()
            dice_scores.append(_dice_coefficient(pred, targets["seg"].cpu()))
            num_batches += 1
    return {
        "loss": val_loss / max(1, num_batches),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
    }
