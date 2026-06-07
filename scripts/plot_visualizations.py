"""Generate experiment summary visualizations from training history and predictions.

Output:
    outputs/figures/loss_curves.png
    outputs/figures/accuracy_curves.png
    outputs/figures/parameter_comparison.png
    outputs/figures/per_class_f1.png
    outputs/figures/overfitting_gap.png
"""

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.models.model_factory import build_model
from ham_in_dl.interpretation.visualization import (
    plot_loss_curves,
    plot_accuracy_curves,
    plot_parameter_comparison,
    plot_per_class_f1,
    plot_overfitting_gap,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"

HISTORY = {
    "baseline_cnn": OUTPUTS_DIR / "history_baseline.csv",
    "resnet18": OUTPUTS_DIR / "history_resnet18.csv",
    "convnext_tiny": OUTPUTS_DIR / "history_convnext_tiny.csv",
}

PREDICTIONS = {
    "baseline_cnn": OUTPUTS_DIR / "predictions_baseline.csv",
    "resnet18": OUTPUTS_DIR / "predictions_resnet18.csv",
    "convnext_tiny": OUTPUTS_DIR / "predictions_convnext_tiny.csv",
}

FIG_DIR = OUTPUTS_DIR / "figures"


def _count_params(model_name: str) -> int:
    """Load a model from its definition and count trainable parameters."""
    model = build_model(
        model_name,
        num_classes=7,
        pretrained=False,  # weight values don't matter for counting
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loss curves ...")
    plot_loss_curves(HISTORY, FIG_DIR / "loss_curves.png")

    print("[2/5] Accuracy curves ...")
    plot_accuracy_curves(HISTORY, FIG_DIR / "accuracy_curves.png")

    print("[3/5] Parameter comparison ...")
    params = {k: _count_params(k) for k in HISTORY}
    for k, v in params.items():
        print(f"  {k}: {v / 1e6:.1f}M params")
    plot_parameter_comparison(params, FIG_DIR / "parameter_comparison.png")

    print("[4/5] Per-class F1 ...")
    plot_per_class_f1(PREDICTIONS, FIG_DIR / "per_class_f1.png")

    print("[5/5] Overfitting gap ...")
    plot_overfitting_gap(HISTORY, FIG_DIR / "overfitting_gap.png")

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
