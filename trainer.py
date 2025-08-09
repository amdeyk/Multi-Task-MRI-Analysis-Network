"""Comprehensive training framework for MRI-KAN."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from torch import Tensor, nn

from experiment_tracking import ExperimentTracker
from monitoring import setup_logging


@dataclass
class Trainer:
    """Handle a single epoch of training or validation."""

    model: nn.Module
    loss_fn: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    amp: bool = False
    amp_dtype: torch.dtype = torch.float16
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    clip_grad: float = 0.0
    tracker: Optional[ExperimentTracker] = None

    def __post_init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.model.to(self.device)
        if self.amp and self.scaler is None and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

    def _forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        mri = batch["mri"].to(self.device)
        return self.model(mri)

    def train_step(self, batch: Dict[str, Tensor]) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        targets = {k: v.to(self.device) for k, v in batch.items() if k != "mri"}
        with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
            outputs = self._forward(batch)
            losses = self.loss_fn(outputs, targets)
            loss = losses["total"]
        if self.scaler is not None:
            prev_scale = self.scaler.get_scale()
            self.scaler.scale(loss).backward()
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < prev_scale:
                self.logger.warning("Gradient overflow detected, reducing loss scale")
        else:
            loss.backward()
            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
        if torch.cuda.is_available():  # pragma: no cover - device specific
            self.logger.debug(
                "mem_allocated=%d", torch.cuda.memory_allocated(self.device)
            )
        return float(loss.detach())

    @torch.no_grad()
    def validate_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.model.eval()
        with torch.cuda.amp.autocast(enabled=self.amp, dtype=self.amp_dtype):
            outputs = self._forward(batch)
        targets = {k: v.to(self.device) for k, v in batch.items() if k != "mri"}
        losses = self.loss_fn(outputs, targets)
        pred = outputs["segmentation"].argmax(dim=1).float()
        dice = self._dice_coefficient(pred, targets["seg"].long())
        return {"loss": float(losses["total"]), "dice": float(dice)}

    def _dice_coefficient(self, pred: Tensor, target: Tensor) -> float:
        pred = pred.float()
        target = target.float()
        intersection = (pred * target).sum().item()
        union = pred.sum().item() + target.sum().item()
        return 2 * intersection / (union + 1e-8)


@dataclass
class TrainingManager:
    """Manage multi-epoch training with validation and checkpoints."""

    trainer: Trainer
    train_loader: Iterable[Dict[str, Tensor]]
    val_loader: Iterable[Dict[str, Tensor]]
    epochs: int
    out_dir: Path
    patience: int = 10
    tracker: Optional[ExperimentTracker] = None
    curriculum: Optional[object] = None
    progressive_resize: Optional[object] = None
    start_epoch: int = 0
    best_loss: float = float("inf")
    logger: logging.Logger = field(default_factory=lambda: setup_logging())

    def fit(self, resume: Optional[str] = None) -> None:
        if resume:
            self.load_checkpoint(resume)
        if self.tracker:
            self.tracker.log_params({"epochs": self.epochs})
        epochs_no_improve = 0
        for epoch in range(self.start_epoch, self.epochs):
            train_loss = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            improved = val_metrics["loss"] < self.best_loss
            if improved:
                self.best_loss = val_metrics["loss"]
                epochs_no_improve = 0
                self.save_checkpoint(epoch, best=True)
            else:
                epochs_no_improve += 1
            self.save_checkpoint(epoch)
            if self.tracker:
                self.tracker.log_metrics({"train_loss": train_loss, **val_metrics}, step=epoch)
                self.tracker.update_version(improved)
            if epochs_no_improve >= self.patience:
                self.logger.info("Early stopping triggered")
                break
        if self.tracker:
            self.tracker.finish()

    def _train_epoch(self, epoch: int) -> float:
        if hasattr(self.train_loader, "sampler") and isinstance(
            self.train_loader.sampler, torch.utils.data.distributed.DistributedSampler
        ):
            self.train_loader.sampler.set_epoch(epoch)

        losses = []
        for batch in self.train_loader:
            try:
                loss = self.trainer.train_step(batch)
                losses.append(loss)
            except RuntimeError as exc:  # pragma: no cover - hardware specific
                if "out of memory" in str(exc).lower():
                    self.logger.warning("OOM encountered, skipping batch")
                    torch.cuda.empty_cache()
                    continue
                raise
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.step()
        return float(sum(losses) / max(len(losses), 1))

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        metrics = []
        for batch in self.val_loader:
            metrics.append(self.trainer.validate_step(batch))
        loss = float(sum(m["loss"] for m in metrics) / max(len(metrics), 1))
        dice = float(sum(m["dice"] for m in metrics) / max(len(metrics), 1))
        self.logger.info("Epoch %d - val_loss: %.4f dice: %.4f", epoch + 1, loss, dice)
        return {"loss": loss, "dice": dice}

    def save_checkpoint(self, epoch: int, best: bool = False) -> None:
        state = {
            "model": self.trainer.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "scheduler": self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
            "scaler": self.trainer.scaler.state_dict() if self.trainer.scaler else None,
            "epoch": epoch,
            "best_loss": self.best_loss,
        }
        ckpt_path = self.out_dir / ("best.pt" if best else f"epoch_{epoch}.pt")
        torch.save(state, ckpt_path)
        if self.tracker:
            self.tracker.log_artifact(ckpt_path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.trainer.device)
        self.trainer.model.load_state_dict(ckpt["model"])
        self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler") and self.trainer.scheduler:
            self.trainer.scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and self.trainer.scaler:
            self.trainer.scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.logger.info("Resumed training from %s", path)
