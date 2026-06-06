"""Checkpoint saving and loading."""

from pathlib import Path

import torch


def save_checkpoint(model, path: str | Path, extra: dict | None = None) -> None:
    """Save model checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model, path: str | Path, map_location="cpu"):
    """Load model checkpoint into model."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
