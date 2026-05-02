import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight_tensor = torch.where(
            targets == 1,
            torch.full_like(targets, self.pos_weight),
            torch.ones_like(targets),
        )
        return F.binary_cross_entropy_with_logits(predictions, targets, weight=weight_tensor)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=predictions.device),
        )
        probabilities = torch.sigmoid(predictions)
        p_t = targets * probabilities + (1 - targets) * (1 - probabilities)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class FocalLossWithOHEM(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0, ohem_ratio: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.ohem_ratio = ohem_ratio

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            predictions,
            targets,
            reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=predictions.device),
        )
        probabilities = torch.sigmoid(predictions)
        p_t = targets * probabilities + (1 - targets) * (1 - probabilities)
        focal_weight = (1 - p_t) ** self.gamma
        focal_losses = focal_weight * bce

        positive_mask = targets == 1
        negative_mask = targets == 0

        positive_loss = focal_losses[positive_mask]

        if negative_mask.sum() > 0:
            negative_losses = focal_losses[negative_mask]
            n_hard = max(1, int(negative_mask.sum().item() * self.ohem_ratio))
            hard_negative_losses, _ = torch.topk(negative_losses, k=n_hard)
            negative_loss = hard_negative_losses
        else:
            negative_loss = focal_losses[negative_mask]

        all_losses = torch.cat([positive_loss, negative_loss]) if positive_loss.numel() > 0 else negative_loss
        return all_losses.mean()


def get_loss_function(loss_name: str, pos_weight: Optional[float] = None, gamma: float = 2.0, ohem_ratio: float = 0.5) -> nn.Module:
    if pos_weight is None:
        pos_weight = 1.0

    if loss_name == "bce":
        logger.info("Using standard BCE loss")
        return nn.BCEWithLogitsLoss()

    if loss_name == "weighted_bce":
        logger.info(f"Using weighted BCE loss with pos_weight={pos_weight:.2f}")
        return WeightedBCELoss(pos_weight=pos_weight)

    if loss_name == "focal":
        logger.info(f"Using focal loss with gamma={gamma}, pos_weight={pos_weight:.2f}")
        return FocalLoss(gamma=gamma, pos_weight=pos_weight)

    if loss_name == "focal_ohem":
        logger.info(f"Using focal loss with OHEM, gamma={gamma}, pos_weight={pos_weight:.2f}, ohem_ratio={ohem_ratio}")
        return FocalLossWithOHEM(gamma=gamma, pos_weight=pos_weight, ohem_ratio=ohem_ratio)

    raise ValueError(f"Unknown loss function: {loss_name}. Choose from: bce, weighted_bce, focal, focal_ohem")


def compute_pos_weight(n_negative: int, n_positive: int) -> float:
    if n_positive == 0:
        raise ValueError("Cannot compute pos_weight with zero positive samples")
    weight = n_negative / n_positive
    logger.info(f"Computed pos_weight={weight:.2f} from {n_positive} positive and {n_negative} negative samples")
    return weight


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    batch_size = 32
    predictions = torch.randn(batch_size)
    targets = torch.zeros(batch_size)
    targets[:2] = 1.0

    pos_weight = compute_pos_weight(n_negative=30, n_positive=2)

    for loss_name in ["bce", "weighted_bce", "focal", "focal_ohem"]:
        loss_fn = get_loss_function(loss_name, pos_weight=pos_weight)
        loss = loss_fn(predictions, targets)
        print(f"{loss_name}: {loss.item():.4f}")