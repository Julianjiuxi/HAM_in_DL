"""Data split helpers."""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ham_in_dl.constants import CLASS_NAMES

LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}


def class_counts(df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """Return class counts ordered by the project class list."""
    return df[label_col].value_counts().reindex(CLASS_NAMES, fill_value=0)


def _check_lesion_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame) -> int:
    """Return the number of lesion_ids shared between train and val."""
    if "lesion_id" not in train_df.columns or "lesion_id" not in val_df.columns:
        return -1
    train_lesions = set(train_df["lesion_id"].dropna())
    val_lesions = set(val_df["lesion_id"].dropna())
    return len(train_lesions & val_lesions)


def create_splits(
    metadata_csv: str | Path,
    split_dir: str | Path,
    *,
    val_size: float = 0.2,
    seed: int = 42,
    force: bool = False,
    label_col: str = "label",
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Create lesion-level stratified train/validation splits.

    Splits are performed by lesion_id so that all images of the same
    skin lesion stay in the same split (no leakage). If ``lesion_id`` is
    not present the function falls back to image-level stratified split.

    The external test set is intentionally not touched here.
    """
    metadata_csv = Path(metadata_csv)
    split_dir = Path(split_dir)
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"

    if train_csv.exists() and val_csv.exists() and not force:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        print(f"Loaded existing splits: {train_csv} / {val_csv}")
        leakage = _check_lesion_leakage(train_df, val_df)
        if leakage > 0:
            print(
                f"WARNING: {leakage} lesion_ids appear in both train and val. "
                "Re-run with --force to regenerate group-aware splits."
            )
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

    if "label_idx" not in df.columns:
        df["label_idx"] = df[label_col].map(LABEL_TO_INDEX)

    # Group-aware split by lesion_id
    if "lesion_id" in df.columns and df["lesion_id"].nunique() > 1:
        lesion_groups = (
            df.groupby("lesion_id")
            .agg(
                {
                    label_col: lambda x: x.mode().iloc[0]
                    if not x.mode().empty
                    else x.iloc[0],
                    "lesion_id": "first",
                }
            )
            .reset_index(drop=True)
        )

        train_lesions, val_lesions = train_test_split(
            lesion_groups,
            test_size=val_size,
            random_state=seed,
            stratify=lesion_groups[label_col],
        )

        train_lesion_ids = set(train_lesions["lesion_id"])
        val_lesion_ids = set(val_lesions["lesion_id"])

        train_df = df[df["lesion_id"].isin(train_lesion_ids)].copy()
        val_df = df[df["lesion_id"].isin(val_lesion_ids)].copy()

        overlap = train_lesion_ids & val_lesion_ids
        if overlap:
            print(f"WARNING: {len(overlap)} lesion_ids still appear in both splits")
        else:
            print("Lesion-level split confirmed: 0 overlapping lesion_ids")
    else:
        print(
            "WARNING: no lesion_id column found; "
            "falling back to image-level stratified split"
        )
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
