"""Training and validation loops."""

from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from ham_in_dl.training.metrics import RunningClassificationMetrics


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> dict[str, float]:
    """Run one training epoch."""
    model.train()
    running = RunningClassificationMetrics()

    for images, labels in tqdm(dataloader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running.update(loss.item(), logits, labels)

    return running.compute()


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device) -> dict[str, float]:
    """Run one validation or evaluation epoch."""
    model.eval()
    running = RunningClassificationMetrics()

    for images, labels in tqdm(dataloader, desc="val", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        running.update(loss.item(), logits, labels)

    return running.compute()


def fit(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    *,
    epochs: int,
    checkpoint_path: str | Path,
    config: dict,
    class_names: list[str],
    early_stopping_patience: int | None = None,
) -> tuple[list[dict], float]:
    """Train a model and save the best checkpoint by validation macro-F1."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_metric = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(row)

        print(
            "  "
            f"train loss={train_metrics['loss']:.4f} "
            f"acc={train_metrics['accuracy']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        current_metric = val_metrics["macro_f1"]
        if current_metric > best_metric:
            best_metric = current_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "config": config,
                    "class_names": class_names,
                },
                checkpoint_path,
            )
            print(f"  saved best checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if (
            early_stopping_patience is not None
            and early_stopping_patience > 0
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                "  early stopping: "
                f"no validation macro-F1 improvement for {early_stopping_patience} epochs"
            )
            break

    return history, best_metric


def save_history(history: list[dict], path: str | Path) -> None:
    """Save epoch-level training history as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)
