"""Download HAM10000 dataset via kagglehub to the repo-local downloads directory."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handle",
        type=str,
        default="kmader/skin-cancer-mnist-ham10000",
        help="Kaggle dataset handle, e.g. owner/dataset-slug.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(Path("data/raw/_downloads/kagglehub_cache")),
        help="kagglehub cache directory (preferred, avoids copying huge files).",
    )
    parser.add_argument(
        "--write-path-file",
        type=str,
        default=str(Path("data/raw/_downloads/kagglehub_ham10000_path.txt")),
        help="Where to write the resolved dataset path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = (repo_root / args.cache_dir).resolve()
    path_file = (repo_root / args.write_path_file).resolve()

    cache_dir.mkdir(parents=True, exist_ok=True)
    path_file.parent.mkdir(parents=True, exist_ok=True)

    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    import kagglehub

    dataset_path = kagglehub.dataset_download(args.handle)

    path_file.write_text(str(dataset_path), encoding="utf-8")
    print("Path to dataset files:", dataset_path)
    print("Cache dir:", cache_dir)
    print("Saved path file:", path_file)


if __name__ == "__main__":
    main()
