"""Transfer learning model builders."""

import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False):
    """Build a ResNet18 classifier."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
