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


def _find_first(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    for path in root.rglob("*"):
        if path.name in candidates:
            return path
    return None


def _copy_or_extract_image_part(src: Path, out_dir: Path, part_name: str) -> None:
    dst = out_dir / part_name
    if src.resolve() == dst.resolve():
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return

    with zipfile.ZipFile(src) as z:
        names = [name for name in z.namelist() if not name.endswith("/")]
        has_part_dir = any(name.startswith(f"{part_name}/") for name in names)
        target_dir = out_dir if has_part_dir else dst
        target_dir.mkdir(parents=True, exist_ok=True)
        z.extractall(target_dir)


def normalise_kagglehub_download(dataset_dir: Path, out_dir: Path, *, force: bool) -> None:
    """Convert a kagglehub download directory into the expected raw layout."""
    if not dataset_dir.exists():
        raise FileNotFoundError(str(dataset_dir))

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = _find_first(dataset_dir, ["archive.zip"])
    if archive is not None:
        extract_zip(archive, out_dir, force=False)

    metadata = _find_first(dataset_dir, ["HAM10000_metadata.csv"])
    if metadata is not None:
        shutil.copy2(metadata, out_dir / "HAM10000_metadata.csv")

    for part_name in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        source = _find_first(dataset_dir, [part_name, f"{part_name}.zip"])
        if source is None:
            source = _find_first(out_dir, [part_name, f"{part_name}.zip"])
        if source is not None:
            _copy_or_extract_image_part(source, out_dir, part_name)

    required = [
        out_dir / "HAM10000_metadata.csv",
        out_dir / "HAM10000_images_part_1",
        out_dir / "HAM10000_images_part_2",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Kaggle download did not contain the expected HAM10000 files: "
            + ", ".join(missing)
        )


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    kaggle_archive = (repo_root / args.kaggle_archive).resolve()
    kaggle_out = (repo_root / args.kaggle_out).resolve()
    kaggle_path_file = (
        repo_root / "data/raw/_downloads/kagglehub_ham10000_path.txt"
    ).resolve()
    testset_zip = (repo_root / args.testset_zip).resolve()
    testset_out = (repo_root / args.testset_out).resolve()

    print("Extracting:")

    def maybe_extract_zip(name: str, zip_path: Path, out_dir: Path) -> None:
        if not zip_path.exists():
            if args.skip_missing:
                print(f" - {name}: missing, skipped:", zip_path)
                return
            raise FileNotFoundError(str(zip_path))
        print(f" - {name}:", zip_path)
        print("   to:", out_dir)
        extract_zip(zip_path, out_dir, force=args.force)

    def maybe_extract_kaggle() -> None:
        if kaggle_archive.exists():
            print(" - Kaggle archive:", kaggle_archive)
            print("   to:", kaggle_out)
            extract_zip(kaggle_archive, kaggle_out, force=args.force)
            return
        if kaggle_path_file.exists():
            dataset_dir = Path(kaggle_path_file.read_text(encoding="utf-8").strip())
            print(" - KaggleHub dataset directory:", dataset_dir)
            print("   to:", kaggle_out)
            normalise_kagglehub_download(dataset_dir, kaggle_out, force=args.force)
            return
        if args.skip_missing:
            print(f" - Kaggle archive: missing, skipped: {kaggle_archive}")
            return
        raise FileNotFoundError(
            f"{kaggle_archive} not found and {kaggle_path_file} not found. "
            "Run scripts/datadownload/download_ham10000_kagglehub.py first."
        )

    if args.source in {"all", "kaggle"}:
        maybe_extract_kaggle()

    if args.source in {"all", "testset"}:
        maybe_extract_zip("TestSet zip", testset_zip, testset_out)

    print("Done.")


if __name__ == "__main__":
    main()
