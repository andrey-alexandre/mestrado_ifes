from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
nn = torch.nn

from src.cnn_baseline.gradcam import (  # noqa: E402
    STATUS_MAP_EMPTY_ALL_ZERO,
    STATUS_OK,
    compute_gradcam,
)


class _TinyModel(nn.Module):
    """Modelo minusculo imitando a interface real (model.features[-1] =
    ultima camada convolucional; model(x) -> logit unico), sem baixar
    nenhum peso pre-treinado -- so para exercitar os hooks do Grad-CAM."""

    def __init__(self, zero_weights: bool = False):
        super().__init__()
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        if zero_weights:
            with torch.no_grad():
                conv.weight.zero_()
                conv.bias.zero_()
        self.features = nn.Sequential(conv, nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def test_compute_gradcam_ok_status_and_shape():
    model = _TinyModel()
    model.eval()
    image_tensor = torch.randn(1, 3, 16, 16)
    result = compute_gradcam(model, image_tensor, target_image_size=(16, 16))
    assert result.status in (STATUS_OK, "mapa_constante_nao_zero", STATUS_MAP_EMPTY_ALL_ZERO)
    assert result.raw_logit is not None
    assert result.sigmoid_score is not None
    if result.status == STATUS_OK:
        assert result.cam_normalized_upsampled.shape == (16, 16)
        assert result.cam_normalized_upsampled.min() >= 0.0 - 1e-6
        assert result.cam_normalized_upsampled.max() <= 1.0 + 1e-6


def test_compute_gradcam_detects_empty_map_when_activations_are_all_zero():
    model = _TinyModel(zero_weights=True)
    model.eval()
    image_tensor = torch.randn(1, 3, 16, 16)
    result = compute_gradcam(model, image_tensor, target_image_size=(16, 16))
    assert result.status == STATUS_MAP_EMPTY_ALL_ZERO
    assert result.cam_normalized_upsampled is None


def test_model_weights_unchanged_after_gradcam():
    model = _TinyModel()
    model.eval()
    before = [p.clone() for p in model.parameters()]
    image_tensor = torch.randn(1, 3, 16, 16)
    compute_gradcam(model, image_tensor, target_image_size=(16, 16))
    after = list(model.parameters())
    for b, a in zip(before, after):
        assert torch.equal(b, a)  # gradcam nunca atualiza pesos (sem optimizer.step())


def test_failure_is_captured_not_raised():
    model = _TinyModel()
    model.eval()
    bad_tensor = torch.randn(2, 3, 16, 16)  # batch=2 -> logit.squeeze() nao escalar -> falha tratada
    result = compute_gradcam(model, bad_tensor, target_image_size=(16, 16))
    assert result.status == "falha"
    assert result.error is not None
