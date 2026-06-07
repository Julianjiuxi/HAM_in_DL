"""Train baseline CNN."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

# Ensure src is on path.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.constants import CLASS_NAMES, NUM_CLASSES
from ham_in_dl.data.dataset import build_dataloader, build_dataset_from_csv
from ham_in_dl.data.imbalance import compute_class_weights
from ham_in_dl.data.split import create_splits
from ham_in_dl.models.model_factory import build_model
from ham_in_dl.seed import set_seed
from ham_in_dl.training.checkpoint import load_checkpoint
from ham_in_dl.training.losses import build_loss
from ham_in_dl.training.trainer import fit, save_history


def build_optimizer(model, config):
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline_cnn.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create train/val splits
    train_csv, val_csv, _, _ = create_splits(
        config["data"]["metadata_csv"],
        config["data"]["split_dir"],
        val_size=config["training"].get("val_size", 0.2),
        seed=config["seed"],
        force=config.get("force_splits", False),
    )

    train_ds = build_dataset_from_csv(
        train_csv,
        config["data"]["image_dir"],
        train=True,
        image_size=config["data"]["image_size"],
    )
    val_ds = build_dataset_from_csv(
        val_csv,
        config["data"]["image_dir"],
        train=False,
        image_size=config["data"]["image_size"],
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    model = build_model(
        config["model"],
        num_classes=NUM_CLASSES,
        pretrained=config.get("pretrained", True),
        freeze_backbone=config.get("freeze_backbone", False),
    )
    model = model.to(device)

    # Compute class weights from training split
    class_weights = None
    if config["training"].get("class_weights", False):
        import pandas as pd
        train_df = pd.read_csv(train_csv)
        label_col = "label"
        if "label_idx" in train_df.columns:
            label_vals = train_df["label_idx"].tolist()
        else:
            from ham_in_dl.data.dataset import LABEL_TO_INDEX
            label_vals = [LABEL_TO_INDEX[l.upper()] for l in train_df[label_col].tolist()]
        class_weights = compute_class_weights(label_vals, NUM_CLASSES).to(device)

    criterion = build_loss("cross_entropy", class_weights=class_weights)
    optimizer = build_optimizer(model, config)

    checkpoint_path = (
        Path(config["checkpoint"]["dir"]) / f"baseline_cnn_{config['seed']}.pt"
    )

    history, best_metric = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=config["training"]["epochs"],
        checkpoint_path=checkpoint_path,
        config=config,
        class_names=CLASS_NAMES,
        early_stopping_patience=config["training"].get("early_stopping_patience"),
    )

    save_history(history, config["output"]["history_csv"])
    print(f"training finished; best val macro-F1: {best_metric:.4f}")


if __name__ == "__main__":
    main()
