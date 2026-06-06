"""Dataset definitions for HAM10000 / ISIC-style image classification."""

from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


class HAMDataset(Dataset):
    """TODO: Implement dataset loading from image paths and labels."""

    def __init__(self, items: list[tuple[str, int]], transform: Callable | None = None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        image_path, label = self.items[index]
        image = Image.open(Path(image_path)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
