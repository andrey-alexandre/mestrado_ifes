"""Common factory to select a segmentation architecture by configuration
(stage 4, architecture-comparison pilot). Routes to the already-approved
``build_unet`` or the new ``build_deeplabv3plus`` -- does not reimplement
or duplicate either model's logic.
"""
from __future__ import annotations

import torch.nn as nn

from src.segmentation.unet_model import build_unet
from src.segmentation.deeplabv3plus_model import build_deeplabv3plus

ARCHITECTURES = ("unet", "deeplabv3plus")


def build_segmentation_model(architecture: str, img_ch: int = 3, output_ch: int = 1) -> nn.Module:
    """Builds a fresh segmentation model for the given architecture name.

    Args:
        architecture: ``"unet"`` or ``"deeplabv3plus"``.
        img_ch: Number of input image channels.
        output_ch: Number of output channels.

    Returns:
        A freshly constructed ``nn.Module`` returning raw logits.

    Raises:
        ValueError: if ``architecture`` is not one of ``ARCHITECTURES``.
    """
    if architecture == "unet":
        return build_unet(img_ch=img_ch, output_ch=output_ch)
    if architecture == "deeplabv3plus":
        return build_deeplabv3plus(img_ch=img_ch, output_ch=output_ch)
    raise ValueError(
        f"Unknown segmentation architecture: '{architecture}'. "
        f"Choose from: {list(ARCHITECTURES)}"
    )
