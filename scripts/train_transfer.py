"""Train a transfer learning model."""

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ham_in_dl.config import load_config
from ham_in_dl.constants import CLASS_NAMES
from ham_in_dl.data.dataset import build_dataloader, build_dataset_from_csv
from ham_in_dl.data.imbalance import compute_class_weights
from ham_in_dl.models.transfer import build_resnet18
from ham_in_dl.seed import set_seed
from ham_in_dl.training.losses import build_loss
from ham_in_dl.training.trainer import fit, save_history


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
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


def build_optimizer(model, training_cfg: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    name = training_cfg.get("optimizer", "adamw").lower()
    lr = training_cfg.get("learning_rate", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=training_cfg.get("momentum", 0.9),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("project", {}).get("seed", 42))

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    outputs_cfg = config["outputs"]

    device = get_device()
    print(f"Using device: {device}")

    model = build_resnet18(
        num_classes=model_cfg.get("num_classes", len(CLASS_NAMES)),
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    ).to(device)

    train_dataset = build_dataset_from_csv(
        repo_path(data_cfg["train_csv"]),
        repo_path(data_cfg["ham10000_root"]),
        train=True,
        image_size=data_cfg.get("image_size", 224),
    )
    val_dataset = build_dataset_from_csv(
        repo_path(data_cfg["val_csv"]),
        repo_path(data_cfg["ham10000_root"]),
        train=False,
        image_size=data_cfg.get("image_size", 224),
    )
    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
    )

    class_weights = None
    if training_cfg.get("use_class_weights", False):
        class_weights = compute_class_weights(
            train_dataset.labels,
            model_cfg.get("num_classes", len(CLASS_NAMES)),
        ).to(device)

    criterion = build_loss(class_weights=class_weights)
    optimizer = build_optimizer(model, training_cfg)

    checkpoint_path = repo_path(outputs_cfg["checkpoint_dir"]) / "best_resnet18.pt"
    history_path = repo_path(outputs_cfg["table_dir"]) / "resnet18_history.csv"
    history, best_metric = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=training_cfg.get("epochs", 30),
        checkpoint_path=checkpoint_path,
        config=config,
        class_names=CLASS_NAMES,
        early_stopping_patience=training_cfg.get("early_stopping_patience"),
    )
    save_history(history, history_path)
    print(f"Saved history to {history_path}")
    print(f"Best validation macro-F1: {best_metric:.4f}")


if __name__ == "__main__":
    main()
