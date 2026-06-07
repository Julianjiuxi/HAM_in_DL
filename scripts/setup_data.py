"""One-command data setup orchestration.

Usage:
    python scripts/setup_data.py --force
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true",
                        help="Explicitly download kaggle training data via kagglehub.")
    parser.add_argument(
        "--extract-source",
        type=str,
        default="all",
        choices=["all", "kaggle", "testset"],
    )
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip missing archives during extraction.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing extracted / processed data.")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _kaggle_data_ready(repo_root: Path) -> bool:
    """Check whether extracted kaggle data (part1 + part2 + metadata) exists."""
    kaggle_dir = repo_root / "data" / "raw" / "_downloads" / "ham10000_kaggle"
    return (
        (kaggle_dir / "HAM10000_images_part_1").exists()
        and (kaggle_dir / "HAM10000_images_part_2").exists()
        and (kaggle_dir / "HAM10000_metadata.csv").exists()
    )


def _build_extract_cmd(repo_root: Path, py: str, args) -> list[str]:
    """Build the extract_archives.py command line."""
    cmd = [
        py,
        str(repo_root / "scripts/datadownload/extract_archives.py"),
        "--source", args.extract_source,
    ]
    if args.force:
        cmd.append("--force")
    if args.skip_missing:
        cmd.append("--skip-missing")
    return cmd


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    # ── 1. Explicit --download (user asked for it) ──
    if args.download:
        run([py, str(repo_root / "scripts/datadownload/download_ham10000_kagglehub.py")])

    # ── 2. Extract archives ──
    #    Auto-skip missing unless download was just performed (data should exist then).
    extract_cmd = _build_extract_cmd(repo_root, py, args)
    if not args.download and not args.skip_missing:
        extract_cmd.append("--skip-missing")
    run(extract_cmd)

    # ── 3. Auto-download kaggle if still missing ──
    if not _kaggle_data_ready(repo_root):
        print("[setup_data] Kaggle training data not found; downloading via kagglehub ...")
        try:
            run([py, str(repo_root / "scripts/datadownload/download_ham10000_kagglehub.py")])
            # Re-run extract for kaggle source only (overwrite partial results)
            run([
                py,
                str(repo_root / "scripts/datadownload/extract_archives.py"),
                "--source", "kaggle",
                "--force",
            ])
        except subprocess.CalledProcessError:
            print("[setup_data] ERROR: kaggle download / extract failed. "
                  "Place kaggle data manually and re-run, or use --skip-missing.")
            sys.exit(1)

        if not _kaggle_data_ready(repo_root):
            print("[setup_data] ERROR: kaggle data still not ready after download. "
                  "Check network / disk space and re-run.")
            sys.exit(1)

    # ── 4. Prepare processed data ──
    prepare_cmd = [
        py,
        str(repo_root / "scripts/dataprocess/prepare_processed_data.py"),
        "--mode", args.mode,
    ]
    if args.force:
        prepare_cmd.append("--force")
    if args.strict:
        prepare_cmd.append("--strict")
    run(prepare_cmd)

    # ── 5. Validate ──
    validate_cmd = [
        py,
        str(repo_root / "scripts/dataprocess/validate_data.py"),
    ]
    if args.strict:
        validate_cmd.append("--strict")
    run(validate_cmd)


if __name__ == "__main__":
    main()

