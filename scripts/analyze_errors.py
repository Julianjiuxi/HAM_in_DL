"""Analyze successful and failed prediction examples."""

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ham_in_dl.evaluation.error_analysis import (
    most_common_confusions,
    select_success_and_failure_cases,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, required=True, help="Path to prediction CSV.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="outputs/predictions")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def main():
    args = parse_args()
    predictions_path = repo_path(args.predictions)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(predictions_path)
    top_correct, top_failed = select_success_and_failure_cases(
        predictions,
        top_k=args.top_k,
    )
    confusion_pairs = most_common_confusions(predictions, top_k=args.top_k)

    top_correct_path = output_dir / "top_correct_examples.csv"
    top_failed_path = output_dir / "top_failed_examples.csv"
    top_correct.to_csv(top_correct_path, index=False)
    top_failed.to_csv(top_failed_path, index=False)

    correct_count = int(predictions["correct"].astype(bool).sum())
    total = len(predictions)
    wrong_count = total - correct_count
    accuracy = correct_count / total if total else 0.0

    print(f"Total examples: {total}")
    print(f"Correct count: {correct_count}")
    print(f"Wrong count: {wrong_count}")
    print(f"Accuracy from CSV: {accuracy:.4f}")
    print(f"Saved top correct examples: {top_correct_path}")
    print(f"Saved top failed examples: {top_failed_path}")
    print()
    print("Most common confusion pairs:")
    if confusion_pairs.empty:
        print("No wrong predictions found.")
    else:
        print(confusion_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
