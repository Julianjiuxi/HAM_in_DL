"""Loss function builders."""

import torch.nn as nn


def build_loss(name: str = "cross_entropy", class_weights=None):
    """Build classification loss.

    Args:
        name: Loss name (currently only ``"cross_entropy"``).
        class_weights: Optional tensor of per-class weights for imbalanced data.
    """
    if name != "cross_entropy":
        raise ValueError(f"Unsupported loss: {name}")
    return nn.CrossEntropyLoss(weight=class_weights)
