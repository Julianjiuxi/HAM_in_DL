"""Tests for model construction."""

from ham_in_dl.models.model_factory import build_model


def test_build_baseline_model():
    model = build_model({"model": {"name": "baseline_cnn", "num_classes": 7}})
    assert model.classifier.out_features == 7


def test_build_resnet18_without_pretrained_download():
    model = build_model(
        {
            "model": {
                "name": "resnet18",
                "num_classes": 7,
                "pretrained": False,
                "freeze_backbone": False,
            }
        }
    )
    assert model.fc.out_features == 7
