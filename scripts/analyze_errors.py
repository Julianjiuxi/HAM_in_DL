"""Analyse prediction errors."""

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.evaluation.error_analysis import (
    most_common_confusions,
    select_success_and_failure_cases,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(args.predictions)

    top_correct, top_failed = select_success_and_failure_cases(
        predictions, top_k=args.top_k
    )
    confusions = most_common_confusions(predictions, top_k=args.top_k)

    top_correct.to_csv(output_dir / "top_correct.csv", index=False)
    top_failed.to_csv(output_dir / "top_failed.csv", index=False)
    confusions.to_csv(output_dir / "most_common_confusions.csv", index=False)

    print(f"top-{args.top_k} correct  -> {output_dir / 'top_correct.csv'}")
    print(f"top-{args.top_k} failed   -> {output_dir / 'top_failed.csv'}")
    print(f"top-{args.top_k} confusions -> {output_dir / 'most_common_confusions.csv'}")


if __name__ == "__main__":
    main()
