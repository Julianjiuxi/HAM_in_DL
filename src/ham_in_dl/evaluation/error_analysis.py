"""Successful and failed prediction analysis."""

import pandas as pd


def select_success_and_failure_cases(
    predictions: pd.DataFrame,
    *,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select high-confidence correct and failed predictions."""
    required = {"correct", "confidence"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions CSV is missing required columns: {sorted(missing)}")

    correct_mask = predictions["correct"].astype(bool)
    top_correct = (
        predictions.loc[correct_mask]
        .sort_values("confidence", ascending=False)
        .head(top_k)
        .copy()
    )
    top_failed = (
        predictions.loc[~correct_mask]
        .sort_values("confidence", ascending=False)
        .head(top_k)
        .copy()
    )
    return top_correct, top_failed


def most_common_confusions(
    predictions: pd.DataFrame,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the most common true/predicted label pairs among mistakes."""
    wrong = predictions.loc[~predictions["correct"].astype(bool)].copy()
    if wrong.empty:
        return pd.DataFrame(columns=["true_label", "pred_label", "count"])
    return (
        wrong.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_k)
    )
