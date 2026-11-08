"""Tests for src/segmentation/deeplabv3plus_model.py (stage 4). Fake small
tensors only -- no real images, no training.

STATUS: implemented, awaiting researcher execution (pytest). Not run by
the assistant/fork that wrote them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.deeplabv3plus_model import (  # noqa: E402
    HIGH_LEVEL_IN_CHANNELS,
    LOW_LEVEL_IN_CHANNELS,
    build_deeplabv3plus,
)


def test_forward_output_shape_matches_input_spatial_size():
    model = build_deeplabv3plus(img_ch=3, output_ch=1)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)
    assert tuple(logits.shape) == (1, 1, 256, 256)


def test_encoder_layer4_has_512_channels_not_2048():
    """ResNet34 uses BasicBlock (expansion=1) -- layer4 output is 512
    channels, unlike ResNet50/101's Bottleneck (expansion=4, 2048
    channels). This is a direct check that the encoder is really
    ResNet34, not a different ResNet variant."""
    model = build_deeplabv3plus(img_ch=3, output_ch=1)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        low = model.stem(x)
        low = model.layer1(low)
        high = model.layer2(low)
        high = model.layer3(high)
        high = model.layer4(high)
    assert low.shape[1] == LOW_LEVEL_IN_CHANNELS == 64
    assert high.shape[1] == HIGH_LEVEL_IN_CHANNELS == 512
    assert high.shape[1] != 2048


def test_logits_are_not_probabilities():
    """Raw logits convention -- sigmoid is the caller's job, never inside
    forward(). A deliberately unnormalized small input/weights can still
    produce values outside [0, 1] in general; this test just confirms the
    model does not apply sigmoid internally by checking the output is not
    already bounded to [0, 1] when a bias is pushed far out."""
    model = build_deeplabv3plus(img_ch=3, output_ch=1)
    with torch.no_grad():
        model.classifier.bias.fill_(50.0)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(x)
    assert logits.max().item() > 1.0  # would be impossible if sigmoid were applied internally
