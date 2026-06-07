"""Visualization helpers for model interpretation and experiment summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from PIL import Image

# ── Shared style constants ──────────────────────────────────────────────
MODEL_COLORS = {"baseline_cnn": "#E74C3C", "resnet18": "#3498DB", "convnext_tiny": "#2ECC71"}
MODEL_LABELS = {"baseline_cnn": "Baseline CNN", "resnet18": "ResNet18", "convnext_tiny": "ConvNeXt-Tiny"}
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def _setup_style():
    """Set consistent matplotlib style for report-quality figures."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ── Grad-CAM overlay (existing) ────────────────────────────────────────

def overlay_heatmap(
    image: Image.Image,
    heatmap,
    *,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> Image.Image:
    """Overlay a normalised heatmap on a PIL image."""
    image = image.convert("RGB")
    heatmap_array = np.asarray(heatmap, dtype=np.float32)
    heatmap_array = np.clip(heatmap_array, 0.0, 1.0)

    if heatmap_array.shape[:2] != (image.height, image.width):
        heatmap_image = Image.fromarray((heatmap_array * 255).astype(np.uint8))
        heatmap_image = heatmap_image.resize(image.size, resample=Image.BILINEAR)
        heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    cmap = cm.get_cmap(colormap)
    colored_heatmap = (cmap(heatmap_array)[..., :3] * 255).astype(np.uint8)
    base = np.asarray(image, dtype=np.float32)
    overlay = (1.0 - alpha) * base + alpha * colored_heatmap.astype(np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# ── Experiment summary visualizations ──────────────────────────────────

def plot_loss_curves(
    history_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    title: str = "Training & Validation Loss",
) -> None:
    """Plot training and validation loss curves for multiple models in one figure.

    Args:
        history_paths: Dict mapping model key to history CSV path.
        output_path: Where to save the figure.
    """
    _setup_style()
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))

    for key, path in history_paths.items():
        df = pd.read_csv(path)
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, "black")
        ax_train.plot(df["epoch"], df["train_loss"], color=color, alpha=0.9, label=label)
        ax_val.plot(df["epoch"], df["val_loss"], color=color, alpha=0.7, label=f"{label} (val)")
    for ax, name in [(ax_train, "Train Loss"), (ax_val, "Validation Loss")]:
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_curves(
    history_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    title: str = "Training & Validation Accuracy",
) -> None:
    """Plot accuracy curves for multiple models in one figure."""
    _setup_style()
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))

    for key, path in history_paths.items():
        df = pd.read_csv(path)
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, "black")
        ax_train.plot(df["epoch"], df["train_accuracy"], color=color, alpha=0.9, label=label)
        ax_val.plot(df["epoch"], df["val_accuracy"], color=color, alpha=0.7, label=f"{label} (val)")
    for ax, name in [(ax_train, "Train Accuracy"), (ax_val, "Validation Accuracy")]:
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_comparison(
    model_params: dict[str, int],
    output_path: str | Path,
    *,
    title: str = "Model Parameter Count",
) -> None:
    """Bar chart comparing trainable parameters across models.

    Args:
        model_params: Dict mapping model key to parameter count (int).
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(model_params.keys())
    counts = list(model_params.values())
    colors = [MODEL_COLORS.get(k, "gray") for k in models]
    labels = [MODEL_LABELS.get(k, k) for k in models]

    bars = ax.bar(labels, [c / 1e6 for c in counts], color=colors, edgecolor="white", width=0.55)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{count / 1e6:.1f}M", ha="center", va="bottom", fontweight="bold")

    ax.set_title(title)
    ax.set_ylabel("Parameters (millions)")
    ax.set_ylim(0, max(counts) / 1e6 * 1.25)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(
    predictions_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    title: str = "Per-Class F1 Score by Model",
) -> None:
    """Grouped bar chart of per-class F1 scores across models.

    Args:
        predictions_paths: Dict mapping model key to predictions CSV path.
    """
    from sklearn.metrics import f1_score

    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25
    n_models = len(predictions_paths)

    for i, (key, path) in enumerate(predictions_paths.items()):
        df = pd.read_csv(path)
        per_class = f1_score(df["true_label"], df["pred_label"], labels=CLASS_NAMES,
                             average=None, zero_division=0)
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, "black")
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, per_class, width, label=label,
                       color=color, edgecolor="white")
        for bar, val in zip(bars, per_class):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(title)
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_overfitting_gap(
    history_paths: dict[str, str | Path],
    output_path: str | Path,
    *,
    title: str = "Overfitting Gap (Train F1 − Val F1) Over Epochs",
) -> None:
    """Plot train-val macro-F1 gap to quantify overfitting per model.

    A gap near zero suggests underfitting; a widening gap indicates overfitting.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, path in history_paths.items():
        df = pd.read_csv(path)
        gap = df["train_macro_f1"] - df["val_macro_f1"]
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key, "black")
        ax.plot(df["epoch"], gap, color=color, label=label, marker="o", markersize=3)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train F1 − Val F1")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
