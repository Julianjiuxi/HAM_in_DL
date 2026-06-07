"""Evaluation metrics and persistence helpers."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> dict[str, float]:
    """Compute project-required classification metrics."""
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "num_examples": int(len(y_true)),
        "num_classes": int(len(class_names)),
    }


def classification_report_dataframe(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    """Return sklearn classification report as a tidy DataFrame."""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    return (
        pd.DataFrame(report)
        .transpose()
        .reset_index()
        .rename(columns={"index": "label"})
    )


def confusion_matrix_dataframe(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> pd.DataFrame:
    """Return confusion matrix with class labels on rows and columns."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
    )
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def save_metrics_json(metrics: dict[str, float], path: str | Path) -> None:
    """Save metrics as human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
