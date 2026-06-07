"""Evaluate a trained model on validation or test split."""

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ham_in_dl.config import load_config
from ham_in_dl.constants import CLASS_NAMES
from ham_in_dl.data.dataset import build_dataloader, build_dataset_from_csv
from ham_in_dl.evaluation.evaluate import evaluate_model
from ham_in_dl.evaluation.metrics import (
    classification_report_dataframe,
    compute_metrics,
    confusion_matrix_dataframe,
    save_metrics_json,
)
from ham_in_dl.evaluation.plots import plot_confusion_matrix
from ham_in_dl.models.model_factory import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
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


def split_paths(config: dict, split: str) -> tuple[Path, Path]:
    data_cfg = config["data"]
    if split == "train":
        return repo_path(data_cfg["train_csv"]), repo_path(data_cfg["ham10000_root"])
    if split == "val":
        return repo_path(data_cfg["val_csv"]), repo_path(data_cfg["ham10000_root"])
    return repo_path(data_cfg["test_csv"]), repo_path(data_cfg["testset_root"])


def main():
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    outputs_cfg = config["outputs"]
    model_name = config["model"]["name"].lower()
    device = get_device()
    print(f"Using device: {device}")

    model_config = copy.deepcopy(config)
    model_config["model"]["pretrained"] = False
    model = build_model(model_config)

    checkpoint = torch.load(repo_path(args.checkpoint), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    csv_path, root_dir = split_paths(config, args.split)
    dataset = build_dataset_from_csv(
        csv_path,
        root_dir,
        train=False,
        image_size=data_cfg.get("image_size", 224),
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=config["training"].get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    y_true, y_pred, confidences = evaluate_model(model, dataloader, device)
    metrics = compute_metrics(y_true, y_pred, CLASS_NAMES)
    report_df = classification_report_dataframe(y_true, y_pred, CLASS_NAMES)
    confusion_df = confusion_matrix_dataframe(y_true, y_pred, CLASS_NAMES)

    table_dir = repo_path(outputs_cfg["table_dir"])
    figure_dir = repo_path(outputs_cfg["figure_dir"])
    prediction_dir = repo_path(outputs_cfg["prediction_dir"])
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{args.split}_{model_name}"
    metrics_path = table_dir / f"{prefix}_metrics.json"
    report_path = table_dir / f"{prefix}_classification_report.csv"
    confusion_csv_path = table_dir / f"{prefix}_confusion_matrix.csv"
    confusion_png_path = figure_dir / f"{prefix}_confusion_matrix.png"
    predictions_path = prediction_dir / f"{prefix}_predictions.csv"

    save_metrics_json(metrics, metrics_path)
    report_df.to_csv(report_path, index=False)
    confusion_df.to_csv(confusion_csv_path)
    plot_confusion_matrix(
        confusion_df,
        confusion_png_path,
        title=f"{args.split.title()} {model_name} Confusion Matrix",
    )

    predictions_df = pd.DataFrame(
        {
            "image_path": [str(p) for p in dataset.dataframe["image_path"].tolist()],
            "true_label": [CLASS_NAMES[index] for index in y_true],
            "pred_label": [CLASS_NAMES[index] for index in y_pred],
            "true_index": y_true,
            "pred_index": y_pred,
            "confidence": confidences,
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        }
    )
    predictions_df.to_csv(predictions_path, index=False)

    print("Saved evaluation outputs:")
    print(f"- metrics: {metrics_path}")
    print(f"- classification report: {report_path}")
    print(f"- confusion matrix CSV: {confusion_csv_path}")
    print(f"- confusion matrix PNG: {confusion_png_path}")
    print(f"- predictions: {predictions_path}")
    print(
        f"Accuracy={metrics['accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
