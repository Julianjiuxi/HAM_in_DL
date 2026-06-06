"""Analyze successful and failed prediction examples."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=str, required=True, help="Path to prediction CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: Implement error analysis.
    print(f"TODO: analyze predictions at {args.predictions}")


if __name__ == "__main__":
    main()
