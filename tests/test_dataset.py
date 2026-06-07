"""Tests for dataset loading."""

import pandas as pd
from PIL import Image

from ham_in_dl.data.dataset import HAMDataset, LABEL_TO_INDEX


def test_ham_dataset_loads_csv_rows(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.jpg"
    Image.new("RGB", (16, 16), color=(128, 64, 32)).save(image_path)

    csv_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "image_path": ["images/sample.jpg"],
            "label": ["MEL"],
        }
    ).to_csv(csv_path, index=False)

    dataset = HAMDataset(csv_path=csv_path, root_dir=tmp_path)
    image, label = dataset[0]

    assert image.size == (16, 16)
    assert label.item() == LABEL_TO_INDEX["MEL"]
