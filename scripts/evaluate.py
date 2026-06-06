"""Evaluate a trained model on validation or test split."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: Implement evaluation pipeline.
    print(f"TODO: evaluate checkpoint={args.checkpoint} on split={args.split}")


if __name__ == "__main__":
    main()
