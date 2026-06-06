"""Extract HAM10000 archives into repo-local folders."""

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=["all", "kaggle", "testset"],
        help="Which archives to extract.",
    )
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
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip missing archives instead of failing.",
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

    def maybe_extract(name: str, zip_path: Path, out_dir: Path) -> None:
        if not zip_path.exists():
            if args.skip_missing:
                print(f" - {name}: missing, skipped:", zip_path)
                return
            raise FileNotFoundError(str(zip_path))
        print(f" - {name}:", zip_path)
        print("   to:", out_dir)
        extract_zip(zip_path, out_dir, force=args.force)

    if args.source in {"all", "kaggle"}:
        maybe_extract("Kaggle archive", kaggle_archive, kaggle_out)

    if args.source in {"all", "testset"}:
        maybe_extract("TestSet zip", testset_zip, testset_out)

    print("Done.")


if __name__ == "__main__":
    main()
