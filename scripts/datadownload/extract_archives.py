"""Extract datasets from zip archive and/or kagglehub download directory."""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where raw archives / downloads live
DEFAULT_DOWNLOADS_DIR = REPO_ROOT / "data" / "raw" / "_downloads"

# Known archives and extracted layout
SOURCES = {
    "kaggle": {
        "zip": (
            DEFAULT_DOWNLOADS_DIR / "kagglehub_cache" / "archive.zip"
        ),
        "dir": (
            DEFAULT_DOWNLOADS_DIR / "kagglehub_cache" / "ham10000_download"
        ),
        "target": DEFAULT_DOWNLOADS_DIR / "ham10000_kaggle",
    },
    "testset": {
        "zip": DEFAULT_DOWNLOADS_DIR / "HAM10000_TestSet.zip",
        "dir": None,
        "target": DEFAULT_DOWNLOADS_DIR / "ham10000_testset",
    },
}


def normalise_kagglehub_download(src_dir: Path, target_dir: Path) -> None:
    """Flatten kagglehub download output so all images are in one place."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_path in src_dir.rglob("*"):
        if file_path.is_file():
            rel_dest = target_dir / file_path.name
            if not rel_dest.exists():
                shutil.copy2(file_path, rel_dest)


def extract_zip(zip_path: Path, target_dir: Path, *, force: bool = False) -> None:
    """Extract a zip file to target_dir."""
    if target_dir.exists() and not force:
        print(f"  already exists: {target_dir} (use --force to overwrite)")
        return
    if target_dir.exists() and force:
        shutil.rmtree(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    # Remove __MACOSX if present
    macos_dir = target_dir / "__MACOSX"
    if macos_dir.exists():
        shutil.rmtree(macos_dir)
    print(f"  extracted to {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Kaggle HAM10000 and teacher-provided TestSet zip archives."
    )
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "kaggle", "testset"],
        help="which archive(s) to extract (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="remove existing extracted directory before extraction",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="skip missing archives instead of failing",
    )
    args = parser.parse_args()

    if args.source in ("all", "kaggle"):
        print("[kaggle]")
        zip_path = SOURCES["kaggle"]["zip"]
        download_dir = SOURCES["kaggle"]["dir"]
        target = SOURCES["kaggle"]["target"]

        if zip_path and zip_path.exists():
            extract_zip(zip_path, target, force=args.force)
        elif download_dir and download_dir.exists():
            print(f"  using kagglehub download directory: {download_dir}")
            normalise_kagglehub_download(download_dir, target)
        elif args.skip_missing:
            print("  skipping: no kaggle archive or download directory found")
        else:
            print(
                f"ERROR: no kaggle archive found at {zip_path} "
                f"or download dir {download_dir}. Use --skip-missing to continue."
            )
            sys.exit(1)

    if args.source in ("all", "testset"):
        print("[testset]")
        zip_path = SOURCES["testset"]["zip"]
        target = SOURCES["testset"]["target"]

        if zip_path and zip_path.exists():
            extract_zip(zip_path, target, force=args.force)
        elif args.skip_missing:
            print("  skipping: TestSet archive not found")
        else:
            print(
                f"ERROR: TestSet archive not found at {zip_path}. "
                "Use --skip-missing to continue."
            )
            sys.exit(1)

    print("done.")


if __name__ == "__main__":
    main()
