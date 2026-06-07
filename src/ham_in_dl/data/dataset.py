"""Dataset definitions for HAM10000 / ISIC-style image classification."""

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ham_in_dl.constants import CLASS_NAMES
from ham_in_dl.data.transforms import build_eval_transforms, build_train_transforms


LABEL_TO_INDEX = {label: index for index, label in enumerate(CLASS_NAMES)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def _normalise_label(label: str | int) -> int:
    """Convert a string label or integer-like label to the canonical class index."""
    if isinstance(label, int):
        if label not in INDEX_TO_LABEL:
            raise ValueError(f"Invalid class index: {label}")
        return label

    text = str(label).strip().upper()
    if text.isdigit():
        index = int(text)
        if index in INDEX_TO_LABEL:
            return index
    if text not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label {label!r}; expected one of {CLASS_NAMES}")
    return LABEL_TO_INDEX[text]


def resolve_image_path(root_dir: str | Path, image_path: str | Path) -> Path:
    """Resolve CSV image paths, preserving absolute paths when provided."""
    path = Path(image_path)
    if path.is_absolute():
        return path
    return Path(root_dir) / path


class HAMDataset(Dataset):
    """Image classification dataset backed by a CSV metadata file.

    The CSV must contain at least ``image_path`` and ``label`` columns. Image
    paths may be absolute or relative to ``root_dir``.
    """

    def __init__(
        self,
        csv_path: str | Path | None = None,
        root_dir: str | Path = ".",
        transform: Callable | None = None,
        image_col: str = "image_path",
        label_col: str = "label",
        items: list[tuple[str, int]] | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

        if csv_path is None and items is None:
            raise ValueError("Either csv_path or items must be provided.")

        if csv_path is not None:
            self.csv_path = Path(csv_path)
            self.dataframe = pd.read_csv(self.csv_path)
            self._validate_columns()
            self.image_paths = [
                resolve_image_path(self.root_dir, p)
                for p in self.dataframe[self.image_col].astype(str).tolist()
            ]
            self.labels = [
                _normalise_label(label)
                for label in self.dataframe[self.label_col].tolist()
            ]
        else:
            self.csv_path = None
            self.image_paths = [Path(path) for path, _ in items or []]
            self.labels = [_normalise_label(label) for _, label in items or []]
            self.dataframe = pd.DataFrame(
                {
                    self.image_col: [str(path) for path in self.image_paths],
                    self.label_col: [INDEX_TO_LABEL[label] for label in self.labels],
                }
            )

    def _validate_columns(self) -> None:
        missing = [
            col
            for col in (self.image_col, self.label_col)
            if col not in self.dataframe.columns
        ]
        if missing:
            raise ValueError(f"{self.csv_path} is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def build_dataset_from_csv(
    csv_path: str | Path,
    root_dir: str | Path,
    *,
    train: bool = False,
    image_size: int = 224,
    transform: Callable | None = None,
) -> HAMDataset:
    """Build a HAMDataset with the standard train/eval transforms."""
    if transform is None:
        transform = (
            build_train_transforms(image_size)
            if train
            else build_eval_transforms(image_size)
        )
    return HAMDataset(csv_path=csv_path, root_dir=root_dir, transform=transform)


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Build a DataLoader with sensible CUDA pin-memory defaults."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
