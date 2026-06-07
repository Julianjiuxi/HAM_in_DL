"""Create or verify data splits."""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ham_in_dl.config import load_config
from ham_in_dl.data.split import class_counts, create_splits


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing split CSVs.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    seed = config.get("project", {}).get("seed", 42)

    train_csv, val_csv, train_df, val_df = create_splits(
        repo_path(data_cfg["ham10000_metadata_csv"]),
        repo_path(data_cfg["split_dir"]),
        val_size=data_cfg.get("val_size", 0.2),
        seed=seed,
        force=args.force,
    )

    print("Saved splits:")
    print(f"- train: {train_csv}")
    print(f"- val: {val_csv}")
    print()
    print("Train class counts:")
    print(class_counts(train_df).to_string())
    print()
    print("Val class counts:")
    print(class_counts(val_df).to_string())
    print()
    print(f"Test set remains separate: {repo_path(data_cfg['test_csv'])}")


if __name__ == "__main__":
    main()
