"""Tests for HAMDataset and DataLoader construction."""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.data.dataset import (
    LABEL_TO_INDEX,
    HAMDataset,
    build_dataloader,
    build_dataset_from_csv,
)


def test_label_to_index_has_seven_classes():
    assert len(LABEL_TO_INDEX) == 7


def test_ham_dataset_construction_from_items(tmp_path):
    items = [("img_0.jpg", "NV"), ("img_1.jpg", "MEL")]
    ds = HAMDataset(items=items)
    assert len(ds) == 2


def test_build_dataset_from_csv_requires_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"wrong_column": []}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="required columns"):
        HAMDataset(csv_path=path)


def test_build_dataloader_returns_batches():
    ds = HAMDataset(items=[("nonexistent.jpg", "NV")])
    loader = build_dataloader(ds, batch_size=2, shuffle=False)
    assert loader.batch_size == 2
