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


def build_convnext_tiny(num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False):
    """Build a ConvNeXt-Tiny classifier (~28.6M params)."""
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    model = models.convnext_tiny(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # ConvNeXt classifier head is a Sequential: [LayerNorm2d, Flatten, Linear]
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
