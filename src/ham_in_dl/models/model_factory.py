"""Model factory."""

from ham_in_dl.models.baseline_cnn import BaselineCNN
from ham_in_dl.models.transfer import build_resnet18


def build_model(config: dict):
    """Build model from config."""
    model_cfg = config["model"]
    name = model_cfg["name"].lower()
    num_classes = model_cfg.get("num_classes", 7)

    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
        )

    raise ValueError(f"Unknown model name: {name}")
