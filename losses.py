"""Advanced loss functions for multi-task learning."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LossConfig


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, pred.shape[1]).permute(0, 3, 1, 2)
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, theta0: int = 3, theta: int = 5) -> None:
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones(1, 1, 3, 3, 3, device=pred.device)
        target_boundary = F.conv3d(target.unsqueeze(1).float(), kernel, padding=1).squeeze(1)
        target_boundary = (target_boundary > 0) & (target_boundary < 27)
        pred_sigmoid = torch.sigmoid(pred)
        return F.binary_cross_entropy(pred_sigmoid[target_boundary], target[target_boundary].float())


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.seg_loss = nn.ModuleList([
            DiceLoss() if config.use_dice else nn.CrossEntropyLoss(),
            BoundaryLoss() if config.use_boundary else None,
        ])
        self.cls_loss = FocalLoss() if config.use_focal else nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )
        self.edge_loss = nn.BCEWithLogitsLoss()
        self.tumor_loss = FocalLoss() if config.use_focal else nn.CrossEntropyLoss()
        self.log_vars = nn.Parameter(torch.zeros(4))

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        seg_loss = 0
        for loss_fn in self.seg_loss:
            if loss_fn is not None:
                seg_loss += loss_fn(outputs["segmentation"], targets["seg"])
        losses["seg"] = seg_loss * torch.exp(-self.log_vars[0]) + self.log_vars[0]
        losses["cls"] = self.cls_loss(outputs["classification"].squeeze(), targets["cls"]) * torch.exp(
            -self.log_vars[1]
        ) + self.log_vars[1]
        losses["edge"] = self.edge_loss(outputs["edge"].squeeze(), targets["edge"].float()) * torch.exp(
            -self.log_vars[2]
        ) + self.log_vars[2]
        losses["tumor"] = self.tumor_loss(outputs["tumor"], targets["tumor"]) * torch.exp(
            -self.log_vars[3]
        ) + self.log_vars[3]
        losses["total"] = sum(losses.values())
        return losses
