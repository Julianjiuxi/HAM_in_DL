"""Train a transfer learning model."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: Implement script logic.
    print(f"TODO: run train_transfer with config={args.config}")


if __name__ == "__main__":
    main()
