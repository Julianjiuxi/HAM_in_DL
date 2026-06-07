"""Plotting utilities for evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Confusion Matrix",
) -> None:
    """Save a labelled confusion matrix figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_df.to_numpy()
    class_names = confusion_df.index.tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    threshold = matrix.max() / 2 if matrix.size and matrix.max() > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
