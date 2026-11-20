"""Grad-CAM para a CNN EfficientNetV2-S congelada (checkpoint SHA-256
``42c385a36845bdfeb33c308862973dc55b1d92c0a1779bdf3193ebd9e6ede0c3``, ver
``docs/reconstruction/29_cnn_stage3_full_closure_s0_eval.md``).

Classe-alvo: SEMPRE melanoma. O classificador tem um único neurônio de
saída (``model.classifier[-1]`` = ``nn.Linear(in_features, 1)``,
construído em ``src.cnn_baseline.efficientnet_training.build_model``) --
não há duas classes/softmax. Na convenção de treinamento
(``src.cnn_baseline.bcn20000_dataset.BINARY_TARGET_TO_LABEL``),
rótulo 1 = melanoma, e o sigmoide desse único logit é a probabilidade de
melanoma usada em toda a avaliação (S0, limiar de Youden). O tensor-alvo
do Grad-CAM é portanto **o logit pré-sigmoide desse único neurônio**
(``model(x).squeeze()`` -- o MESMO tensor que, após sigmoide, já é usado
pela avaliação S0 -- nunca uma classe reconstruída por softmax, que não
existe aqui).

Camada-alvo: ``model.features[-1]`` -- o último bloco convolucional da
EfficientNetV2-S do torchvision (um ``Conv2dNormActivation`` 1x1 que
expande para 1280 canais), último mapa espacial ANTES do
global-average-pool + classificador. Escolha padrão para Grad-CAM em
arquiteturas EfficientNet (último mapa de ativação com resolução
espacial > 1x1, mais próximo da decisão final). Documentado aqui porque
o pedido exige essa documentação explícita; nenhuma outra camada foi
testada ou comparada.

Modo: ``model.eval()`` (BatchNorm/Dropout em modo de inferência, igual à
avaliação S0 -- nunca treino), mas com gradientes HABILITADOS
(``torch.set_grad_enabled(True)``, nunca ``torch.no_grad()``) só para
esta chamada -- os pesos do modelo nunca são atualizados (nenhum
``optimizer.step()``/``loss.backward()`` sobre parâmetros, só
``logit.backward()`` para obter o gradiente da ATIVAÇÃO, consumido e
descartado).

Normalização do mapa: min-max POR IMAGEM (cada mapa é normalizado
independentemente para o próprio intervalo [min, max] -> [0, 1]) --
NUNCA compare magnitude absoluta entre mapas de imagens diferentes
normalizados desta forma; a intensidade relativa dentro de uma imagem é
interpretável, a escala absoluta entre imagens não é (documentado
explicitamente, conforme exigido).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

STATUS_OK = "ok"
STATUS_MAP_EMPTY_ALL_ZERO = "mapa_vazio_zero_apos_relu"
STATUS_MAP_CONSTANT_NONZERO = "mapa_constante_nao_zero"
STATUS_FAILURE = "falha"

TARGET_LAYER_ATTR_PATH = "features[-1]"  # documentado, nao configuravel por chamador
NORMALIZATION_METHOD = "min_max_por_imagem_independente"


@dataclass
class GradCamResult:
    status: str
    cam_raw: Optional[np.ndarray]  # mapa (H_ativacao, W_ativacao), antes de upsample/normalizacao, preservado para auditoria
    cam_normalized_upsampled: Optional[np.ndarray]  # (H_img, W_img) em [0,1], ou None se status != ok
    raw_logit: Optional[float]
    sigmoid_score: Optional[float]
    cam_min_before_norm: Optional[float]
    cam_max_before_norm: Optional[float]
    error: Optional[str]


def compute_gradcam(model, image_tensor, target_image_size: tuple[int, int]) -> GradCamResult:
    """``model``: já em ``.eval()``, pesos carregados, no device correto.
    ``image_tensor``: já pré-processado (mesma transformação da
    avaliação S0), shape ``(1, C, H, W)``, no mesmo device do modelo.
    ``target_image_size``: ``(H, W)`` da imagem ORIGINAL (antes do
    pré-processamento), para o upsample final do mapa.

    Nunca lança excecao para fora -- qualquer falha (forward/backward,
    ativacao/gradiente ausente) e capturada e devolvida como
    ``GradCamResult(status=STATUS_FAILURE, ...)``, para que o chamador
    (``scripts/generate_gradcam_gallery.py``) registre o status por
    imagem sem interromper a galeria inteira."""
    import torch
    import torch.nn.functional as F

    activations: dict[str, "torch.Tensor"] = {}
    gradients: dict[str, "torch.Tensor"] = {}

    target_layer = model.features[-1]

    def _forward_hook(_module, _input, output):
        activations["value"] = output.detach()

    def _backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    handle_fwd = target_layer.register_forward_hook(_forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            logit = model(image_tensor).squeeze()
            if logit.dim() != 0:
                raise ValueError(f"logit esperado escalar (batch=1, 1 saida), obtido shape {tuple(logit.shape)}")
            sigmoid_score = float(torch.sigmoid(logit).detach().cpu().item())
            raw_logit = float(logit.detach().cpu().item())
            logit.backward()

        if "value" not in activations or "value" not in gradients:
            return GradCamResult(STATUS_FAILURE, None, None, raw_logit, sigmoid_score, None, None,
                                  error="ativacao ou gradiente nao capturado pelos hooks")

        act = activations["value"][0]  # (C, H, W)
        grad = gradients["value"][0]  # (C, H, W)
        weights = grad.mean(dim=(1, 2))  # (C,) -- media global do gradiente por canal (Grad-CAM original, Selvaraju et al. 2017)
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # (H, W)
        cam_np = cam.cpu().numpy().astype(np.float64)

        cam_min, cam_max = float(cam_np.min()), float(cam_np.max())
        if cam_max <= 1e-12:
            return GradCamResult(STATUS_MAP_EMPTY_ALL_ZERO, cam_np, None, raw_logit, sigmoid_score, cam_min, cam_max,
                                  error=None)
        if (cam_max - cam_min) < 1e-12:
            return GradCamResult(STATUS_MAP_CONSTANT_NONZERO, cam_np, None, raw_logit, sigmoid_score, cam_min, cam_max,
                                  error=None)

        cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        cam_upsampled = _resize_to(cam_norm, target_image_size)
        return GradCamResult(STATUS_OK, cam_np, cam_upsampled, raw_logit, sigmoid_score, cam_min, cam_max, error=None)
    except Exception as exc:  # noqa: BLE001 -- registrado por imagem, nunca propagado para a galeria inteira
        return GradCamResult(STATUS_FAILURE, None, None, None, None, None, None, error=f"{type(exc).__name__}: {exc}")
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _resize_to(array_2d: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    import cv2  # opencv-python-headless, ja dependencia do projeto -- nenhuma nova dependencia

    h, w = target_hw
    return cv2.resize(array_2d.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap_on_image(original_rgb_uint8: np.ndarray, cam_normalized: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """``original_rgb_uint8``: (H,W,3) uint8 RGB. ``cam_normalized``:
    (H,W) float em [0,1], já no mesmo tamanho da imagem. Colormap fixo
    (JET, convenção comum de Grad-CAM) e alpha fixo -- consistentes para
    toda a galeria (nenhum mapa recebe colormap/alpha diferente)."""
    import cv2

    heatmap_uint8 = (cam_normalized * 255).astype(np.uint8)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    overlay = (original_rgb_uint8.astype(np.float32) * (1 - alpha) + heatmap_color_rgb.astype(np.float32) * alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)
