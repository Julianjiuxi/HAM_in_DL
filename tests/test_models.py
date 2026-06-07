"""Tests for model construction."""

import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ham_in_dl.constants import NUM_CLASSES
from ham_in_dl.models.model_factory import build_model


def test_baseline_cnn_output_shape():
    model = build_model("baseline_cnn", num_classes=NUM_CLASSES)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, NUM_CLASSES)


def test_resnet18_output_shape():
    model = build_model("resnet18", num_classes=NUM_CLASSES)
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    assert out.shape == (2, NUM_CLASSES)
