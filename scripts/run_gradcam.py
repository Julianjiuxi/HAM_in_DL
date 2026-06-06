"""Run Grad-CAM for a trained model and selected image."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # TODO: Implement Grad-CAM pipeline.
    print(f"TODO: run Grad-CAM for image={args.image_path}")


if __name__ == "__main__":
    main()
