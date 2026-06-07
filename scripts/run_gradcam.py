"""Run Grad-CAM for a trained model and selected image."""

import argparse
import copy
import sys
from pathlib import Path

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ham_in_dl.config import load_config
from ham_in_dl.constants import CLASS_NAMES
from ham_in_dl.data.transforms import build_eval_transforms
from ham_in_dl.interpretation.gradcam import generate_gradcam
from ham_in_dl.interpretation.visualization import overlay_heatmap
from ham_in_dl.models.model_factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    config = load_config(args.config)
    model_name = config["model"]["name"].lower()
    if model_name != "resnet18":
        raise ValueError("This Grad-CAM script currently supports ResNet18 only.")

    device = get_device()
    model_config = copy.deepcopy(config)
    model_config["model"]["pretrained"] = False
    model = build_model(model_config)
    checkpoint = torch.load(repo_path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    image_path = repo_path(args.image_path)
    image = Image.open(image_path).convert("RGB")
    image_size = config["data"].get("image_size", 224)
    transform = build_eval_transforms(image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    target_layer = model.layer4[-1].conv2
    heatmap, target_class, confidence = generate_gradcam(
        model,
        input_tensor,
        target_layer,
    )

    resized_image = image.resize((image_size, image_size), resample=Image.BILINEAR)
    overlay = overlay_heatmap(resized_image, heatmap.numpy())

    if args.output is None:
        output_dir = repo_path(config["outputs"].get("gradcam_dir", "outputs/gradcam"))
        output_path = output_dir / f"{image_path.stem}_gradcam.png"
    else:
        output_path = repo_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    print(f"Predicted class: {CLASS_NAMES[target_class]} ({confidence:.4f})")
    print(f"Saved Grad-CAM overlay to {output_path}")


if __name__ == "__main__":
    main()
