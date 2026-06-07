"""Model factory."""

from ham_in_dl.models.baseline_cnn import BaselineCNN
from ham_in_dl.models.transfer import build_convnext_tiny, build_resnet18


def build_model(
    model_name: str,
    *,
    num_classes: int = 7,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Build a model by name.

    Args:
        model_name: ``"baseline_cnn"``, ``"resnet18"``, or ``"convnext_tiny"``.
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights (transfer models only).
        freeze_backbone: Freeze all layers except the classifier head.
    """
    name = str(model_name).lower()

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)

    if name in ("resnet18", "resnet_18"):
        return build_resnet18(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    if name in ("convnext_tiny", "convnext-tiny"):
        return build_convnext_tiny(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unknown model name: {name}. "
        "Supported: baseline_cnn, resnet18, convnext_tiny"
    )
