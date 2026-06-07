"""Data split helpers."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ham_in_dl.constants import CLASS_NAMES

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}


def class_counts(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """Return class counts ordered by the project class list."""
    return df[label_col].value_counts().reindex(CLASS_NAMES, fill_value=0)


def create_splits(
    metadata_csv: str | Path,
    split_dir: str | Path,
    *,
    val_size: float = 0.2,
    seed: int = 42,
    force: bool = False,
    label_col: str = "label",
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation splits from HAM10000 metadata.

    The external test set is intentionally not touched here.
    """
    metadata_csv = Path(metadata_csv)
    split_dir = Path(split_dir)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if train_csv.exists() and val_csv.exists() and not force:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        return train_csv, val_csv, train_df, val_df

    df = pd.read_csv(metadata_csv)
    if label_col not in df.columns:
        raise ValueError(f"{metadata_csv} is missing required column: {label_col}")

    labels = df[label_col].astype(str).str.upper()
    invalid = sorted(set(labels) - set(CLASS_NAMES))
    if invalid:
        raise ValueError(f"Invalid labels in {metadata_csv}: {invalid}")

    df = df.copy()
    df[label_col] = labels

    # Add label_idx column if not present
    if "label_idx" not in df.columns:
        df["label_idx"] = df[label_col].map(LABEL_TO_INDEX)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df[label_col],
    )

    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.sort_index().to_csv(train_csv, index=False)
    val_df.sort_index().to_csv(val_csv, index=False)
    return train_csv, val_csv, train_df, val_df
