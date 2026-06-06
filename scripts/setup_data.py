"""One-command data setup orchestration."""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--extract-source",
        type=str,
        default="all",
        choices=["all", "kaggle", "testset"],
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--mode", type=str, default="copy", choices=["copy", "move"])
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    if args.all:
        args.download = False

    if args.download:
        run([py, str(repo_root / "scripts/datadownload/download_ham10000_kagglehub.py")])

    extract_cmd = [
        py,
        str(repo_root / "scripts/datadownload/extract_archives.py"),
        "--source",
        args.extract_source,
    ]
    if args.force:
        extract_cmd.append("--force")
    if args.skip_missing:
        extract_cmd.append("--skip-missing")
    run(extract_cmd)

    prepare_cmd = [
        py,
        str(repo_root / "scripts/dataprocess/prepare_processed_data.py"),
        "--mode",
        args.mode,
    ]
    if args.force:
        prepare_cmd.append("--force")
    if args.strict:
        prepare_cmd.append("--strict")
    run(prepare_cmd)

    validate_cmd = [
        py,
        str(repo_root / "scripts/dataprocess/validate_data.py"),
    ]
    if args.strict:
        validate_cmd.append("--strict")
    run(validate_cmd)


if __name__ == "__main__":
    main()

