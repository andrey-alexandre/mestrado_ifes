"""Tests for src/segmentation/model_factory.py (stage 4, architecture
factory). No real training, no I/O -- just construction and routing.

STATUS: implemented, awaiting researcher execution (pytest). Not run by
the assistant/fork that wrote them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.deeplabv3plus_model import DeepLabV3PlusResNet34  # noqa: E402
from src.segmentation.model_factory import build_segmentation_model  # noqa: E402
from src.segmentation.unet_model import UNet  # noqa: E402


def test_build_unet_returns_unet_instance():
    model = build_segmentation_model("unet")
    assert isinstance(model, UNet)


def test_build_deeplabv3plus_returns_deeplabv3plus_instance():
    model = build_segmentation_model("deeplabv3plus")
    assert isinstance(model, DeepLabV3PlusResNet34)


def test_unknown_architecture_raises_value_error():
    with pytest.raises(ValueError):
        build_segmentation_model("resnet_unknown_arch")
