"""Class imbalance utilities."""

import torch


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """TODO: Compute inverse-frequency class weights."""
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights
