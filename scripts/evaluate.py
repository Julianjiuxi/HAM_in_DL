"""Run model evaluation on validation/test split."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.constants import CLASS_NAMES, INDEX_TO_LABEL, NUM_CLASSES
from ham_in_dl.data.dataset import build_dataloader, build_dataset_from_csv
from ham_in_dl.data.split import create_splits
from ham_in_dl.evaluation.evaluate import predict
from ham_in_dl.evaluation.metrics import (
    classification_report_dataframe,
    compute_metrics,
    confusion_matrix_dataframe,
    save_metrics_json,
)
from ham_in_dl.evaluation.plots import plot_confusion_matrix
from ham_in_dl.models.model_factory import build_model
from ham_in_dl.seed import set_seed


def build_predictions_dataframe(
    targets: list[int],
    predictions: list[int],
    confidences: list[float],
    dataset,
    class_names: list[str],
) -> pd.DataFrame:
    """Assemble a prediction results DataFrame with required metadata."""
    correct = [t == p for t, p in zip(targets, predictions)]

    rows = []
    for i in range(len(targets)):
        row = {
            "image_id": dataset.dataframe.iloc[i].get("image_id", ""),
            "dataset": dataset.dataframe.iloc[i].get("dataset", ""),
            "image_path": str(dataset.image_paths[i]),
            "true_label": class_names[targets[i]],
            "pred_label": class_names[predictions[i]],
            "true_index": targets[i],
            "pred_index": predictions[i],
            "confidence": confidences[i],
            "correct": correct[i],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.yaml")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.split == "test":
        csv_path = config["data"]["test_csv"]
        root_dir = config["data"]["test_image_dir"]
    else:
        _, csv_path, _, _ = create_splits(
            config["data"]["metadata_csv"],
            config["data"]["split_dir"],
            val_size=config["training"].get("val_size", 0.2),
            seed=config["seed"],
            force=False,
        )
        root_dir = config["data"]["image_dir"]

    dataset = build_dataset_from_csv(
        csv_path,
        root_dir,
        train=False,
        image_size=config["data"]["image_size"],
    )
    loader = build_dataloader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    model = build_model(
        config["model"],
        num_classes=NUM_CLASSES,
        pretrained=config.get("pretrained", True),
        freeze_backbone=False,  # evaluate always loads from checkpoint
    )
    checkpoint_path = (
        Path(config["checkpoint"]["dir"])
        / f"{config['model']}_{config['seed']}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    targets, preds, confs = predict(model, loader, device)

    metrics = compute_metrics(targets, preds, CLASS_NAMES)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    df = build_predictions_dataframe(
        targets, preds, confs, dataset, CLASS_NAMES
    )

    preds_csv = Path(config["output"]["predictions_csv"])
    preds_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(preds_csv, index=False)
    print(f"predictions saved to {preds_csv}")

    cm = confusion_matrix_dataframe(targets, preds, CLASS_NAMES)
    cm_path = preds_csv.parent / f"{args.split}_confusion_matrix.csv"
    cm.to_csv(cm_path)
    print(f"confusion matrix saved to {cm_path}")

    cm_plot_path = preds_csv.parent / f"{args.split}_confusion_matrix.png"
    plot_confusion_matrix(cm, cm_plot_path, title=f"{config['model']} Confusion Matrix ({args.split})")
    print(f"confusion matrix plot saved to {cm_plot_path}")

    cr = classification_report_dataframe(targets, preds, CLASS_NAMES)
    cr_path = preds_csv.parent / f"{args.split}_classification_report.csv"
    cr.to_csv(cr_path, index=False)
    print(f"classification report saved to {cr_path}")

    metrics_path = preds_csv.parent / f"{args.split}_metrics.json"
    save_metrics_json(metrics, metrics_path)
    print(f"metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
