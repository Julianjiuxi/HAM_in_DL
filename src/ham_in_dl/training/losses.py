"""Loss function builders."""

import torch.nn as nn


def build_loss(class_weights=None):
    """Build classification loss."""
    return nn.CrossEntropyLoss(weight=class_weights)
