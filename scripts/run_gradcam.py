"""Generate Grad-CAM visualisations."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.constants import NUM_CLASSES
from ham_in_dl.data.dataset import HAMDataset
from ham_in_dl.data.transforms import build_eval_transforms
from ham_in_dl.interpretation.gradcam import generate_gradcam
from ham_in_dl.interpretation.visualization import overlay_heatmap
from ham_in_dl.models.model_factory import build_model
from ham_in_dl.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--predictions-csv", required=True)
    parser.add_argument(
        "--case",
        default="failed",
        choices=("correct", "failed", "all"),
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/gradcam")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        config["model"],
        num_classes=NUM_CLASSES,
        pretrained=False,  # load from checkpoint, not pretrained weights
    )
    checkpoint_path = (
        Path(config["checkpoint"]["dir"])
        / f"{config['model']}_{config['seed']}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Disable inplace ReLU to avoid view/inplace conflicts with Grad-CAM hooks
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False

    # Resolve target layer by model architecture
    model_name = config["model"]
    if model_name in ("convnext_tiny", "convnext-tiny"):
        # ConvNeXt: hook into last depthwise conv inside the final CNBlock
        target_layer = model.features[-1][-1].block[5]
    elif model_name in ("resnet18", "resnet_18"):
        target_layer = model.layer4[-1]
    elif model_name == "baseline_cnn":
        # BaselineCNN: third and final conv layer in self.features (index 6)
        target_layer = model.features[6]
    else:
        raise ValueError(
            f"Unsupported model for Grad-CAM: {model_name}. "
            "Supported: resnet18, convnext_tiny, baseline_cnn"
        )
    transform = build_eval_transforms(config["data"]["image_size"])

    predictions = pd.read_csv(args.predictions_csv)
    mask = predictions["correct"].astype(bool)
    if args.case == "failed":
        subset = predictions.loc[~mask]
    elif args.case == "correct":
        subset = predictions.loc[mask]
    else:
        subset = predictions
    subset = subset.sort_values("confidence", ascending=False).head(args.num_samples)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_, row) in enumerate(subset.iterrows()):
        img_path = Path(row["image_path"])
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        heatmap, target_class, conf = generate_gradcam(
            model, input_tensor, target_layer
        )
        overlay = overlay_heatmap(image, heatmap.numpy())

        image.save(output_dir / f"{args.case}_{idx+1}_original.jpg")
        overlay.save(output_dir / f"{args.case}_{idx+1}_gradcam.jpg")

        pred_label = row.get("pred_label", f"class_{target_class}")
        true_label = row.get("true_label", pred_label)
        print(
            f"{args.case} {idx+1}: {img_path.name} "
            f"true={true_label} pred={pred_label} conf={conf:.4f}"
        )


if __name__ == "__main__":
    main()
