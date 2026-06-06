"""Extract HAM10000 archives into repo-local folders."""

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle-archive",
        type=str,
        default="data/raw/_downloads/kagglehub_cache/archive.zip",
    )
    parser.add_argument(
        "--kaggle-out",
        type=str,
        default="data/raw/_downloads/ham10000_kaggle",
    )
    parser.add_argument(
        "--testset-zip",
        type=str,
        default="data/raw/_downloads/HAM10000_TestSet.zip",
    )
    parser.add_argument(
        "--testset-out",
        type=str,
        default="data/raw/_downloads/ham10000_testset",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def extract_zip(zip_path: Path, out_dir: Path, *, force: bool) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(str(zip_path))

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    kaggle_archive = (repo_root / args.kaggle_archive).resolve()
    kaggle_out = (repo_root / args.kaggle_out).resolve()
    testset_zip = (repo_root / args.testset_zip).resolve()
    testset_out = (repo_root / args.testset_out).resolve()

    print("Extracting:")
    print(" - Kaggle archive:", kaggle_archive)
    print("   to:", kaggle_out)
    extract_zip(kaggle_archive, kaggle_out, force=args.force)

    print(" - TestSet zip:", testset_zip)
    print("   to:", testset_out)
    extract_zip(testset_zip, testset_out, force=args.force)

    print("Done.")


if __name__ == "__main__":
    main()
