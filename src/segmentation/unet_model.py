"""Standard U-Net architecture (Ronneberger et al., 2015) for the
segmentation baseline (stage 4, ISIC2018 Task1 development).

Implemented directly in ``torch.nn`` rather than imported from
``Image_Segmentation.code.network`` (referenced by
``src/models/model.py``) because that external package
(LeeJunHyun/Image_Segmentation) is NOT installed in this environment
(confirmed: ``import Image_Segmentation`` raises ``ModuleNotFoundError``)
and no checkpoint or vendored copy exists anywhere in the repository or
on disk (confirmed by search). This is the SAME architecture the
researcher already approved as the baseline -- it is a direct
implementation of it to remove the blocked external dependency, not a
new/different architecture. ``src/models/model.py`` and its
``MODEL_CLASSES`` dict are left untouched; this module is independent
and does not attempt to patch or replace that import.

Encoder-decoder with skip connections, 4 downsampling/upsampling stages
(64-128-256-512 channels, 1024 at the bottleneck) -- the standard U-Net
channel progression. Output head: a single 1x1 conv to ``output_ch``
channels (binary mask here, ``output_ch=1``), returning raw logits --
sigmoid is applied by the caller when a probability/mask is needed
(training loss uses ``BCEWithLogitsLoss`` on the raw logits directly, for
numerical stability), never inside the model, same convention already
used in ``src/cnn_baseline/efficientnet_training.py``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net: 4 encoder stages (maxpool downsampling), a
    bottleneck, and 4 decoder stages (transposed-conv upsampling with
    skip-connection concatenation from the matching encoder stage).

    Input spatial size must be divisible by 16 (2**4, from the 4 pooling
    stages) for the skip-connection shapes to line up exactly -- the
    dataset module (``isic2018_dataset.py``) resizes to a fixed size
    chosen to satisfy this.
    """

    CHANNELS = (64, 128, 256, 512)

    def __init__(self, img_ch: int = 3, output_ch: int = 1) -> None:
        super().__init__()
        c1, c2, c3, c4 = self.CHANNELS
        bottleneck_ch = c4 * 2  # 1024

        self.enc1 = DoubleConv(img_ch, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c4, bottleneck_ch)

        self.up4 = nn.ConvTranspose2d(bottleneck_ch, c4, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(bottleneck_ch, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c4, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c3, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c2, c1)

        self.out_conv = nn.Conv2d(c1, output_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # raw logits, shape (N, output_ch, H, W)


def build_unet(img_ch: int = 3, output_ch: int = 1) -> UNet:
    """Construct a fresh (randomly initialized) U-Net. No pretrained
    weights exist for this architecture/task (ImageNet init is a
    classification-backbone concept, not applicable to a from-scratch
    encoder-decoder segmentation net trained on ISIC2018 Task1 directly)."""
    return UNet(img_ch=img_ch, output_ch=output_ch)
