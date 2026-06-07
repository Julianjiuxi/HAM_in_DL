"""Create train/validation splits from HAM10000 processed metadata."""

import argparse
import sys
from pathlib import Path

import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.data.split import class_counts, create_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_csv, val_csv, train_df, val_df = create_splits(
        config["data"]["metadata_csv"],
        config["data"]["split_dir"],
        val_size=config["training"].get("val_size", 0.2),
        seed=config["seed"],
        force=args.force,
    )

    print(f"train: {len(train_df)} samples -> {train_csv}")
    print("train class distribution:")
    print(class_counts(train_df))
    print(f"\nval:   {len(val_df)} samples -> {val_csv}")
    print("val class distribution:")
    print(class_counts(val_df))


if __name__ == "__main__":
    main()
