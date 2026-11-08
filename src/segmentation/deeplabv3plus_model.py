"""DeepLabV3+ (Chen et al., 2018) with a ResNet34 encoder, for the
segmentation architecture-comparison pilot (stage 4, ISIC2018 Task1
development).

Implemented directly in ``torch``/``torchvision`` -- no new dependency
(``segmentation_models_pytorch`` is NOT installed in this environment and
would require an install, which this task may not perform).

Encoder: ``torchvision.models.resnet34(weights=None)``. ``weights=None`` is
deliberate, not a placeholder -- this session has no network access to
download ImageNet weights, and the researcher's instruction for THIS pilot
is random initialization for both architectures being compared (see
``docs/reconstruction/13_segmentation_deeplabv3plus_pilot.md``). This does
NOT decide the pretraining policy for the final experiment.

output_stride = 32 (no atrous/dilated stages), NOT the DeepLabV3+ paper's
usual output_stride=16. This is a confirmed environment constraint, not a
convenience choice: ``torchvision.models.resnet.BasicBlock.__init__``
(.venv/Lib/site-packages/torchvision/models/resnet.py:78-79) raises
``NotImplementedError("Dilation > 1 not supported in BasicBlock")``.
ResNet34 is built entirely from ``BasicBlock`` (unlike ResNet50/101, which
use ``Bottleneck`` and do support dilation) -- so the standard "replace the
last stage's stride with dilation to get output_stride=16" trick used by
``torchvision.models.segmentation.deeplabv3_resnet50`` is structurally
unavailable for ResNet34. ResNet34 + DeepLabV3+ remains viable; it only
changes the encoder's output_stride from 16 to 32 (standard ResNet34
forward, every stage uses its normal stride). Do not swap to ResNet50/101
to "fix" this -- ResNet34 was the approved encoder; output_stride=32 is the
correct response to a confirmed library limitation, not a workaround that
changes the methodological decision.

Channel/stride bookkeeping for ResNet34 (BasicBlock, expansion=1 --
different from ResNet50/101's Bottleneck, expansion=4):
  conv1+bn1+relu+maxpool -> stride 4,  64 ch   (low-level features, here)
  layer1                  -> stride 4,  64 ch   (same spatial size as above)
  layer2                  -> stride 8,  128 ch
  layer3                  -> stride 16, 256 ch
  layer4                  -> stride 32, 512 ch  (high-level features, here)
Low-level features are taken after ``layer1`` (stride 4, 64 channels) and
high-level features after ``layer4`` (stride 32, 512 channels) -- 512, not
2048 as it would be for ResNet50/101's Bottleneck-based layer4.

ASPP atrous rates (3, 6, 9): the original paper's rates (6, 12, 18) are
tuned for output_stride=16. Scaled down proportionally for output_stride=32
(half the output_stride ratio relative to 16, i.e. a feature map half the
linear size in each dimension) to keep the atrous receptive fields
meaningful relative to the smaller feature map (e.g. 256x256 input ->
8x8 feature map at stride 32, vs. 16x16 at stride 16) -- not an arbitrary
choice, a direct proportional scaling of the original rates.

Returns raw logits (sigmoid applied by the caller), same convention as
``src.segmentation.unet_model.build_unet``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

ASPP_RATES = (3, 6, 9)
ASPP_OUT_CHANNELS = 256
LOW_LEVEL_PROJECT_CHANNELS = 48
LOW_LEVEL_IN_CHANNELS = 64  # resnet34 layer1 output channels (BasicBlock, expansion=1)
HIGH_LEVEL_IN_CHANNELS = 512  # resnet34 layer4 output channels (BasicBlock, expansion=1)


class _ASPPConvBranch(nn.Sequential):
    """One atrous (or 1x1, when dilation=1) branch of the ASPP module."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        if dilation == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=3,
                padding=dilation, dilation=dilation, bias=False,
            )
        super().__init__(conv, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class _ASPPPoolingBranch(nn.Module):
    """Global-average-pooling branch of the ASPP module (image-level context)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        y = self.pool(x)
        y = self.relu(self.bn(self.conv(y)))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: 1x1 + len(rates) atrous 3x3 branches +
    a global-average-pooling branch, concatenated and projected to
    ``out_channels`` via a 1x1 conv + BN + ReLU + dropout."""

    def __init__(self, in_channels: int, out_channels: int = ASPP_OUT_CHANNELS, rates=ASPP_RATES) -> None:
        super().__init__()
        branches = [_ASPPConvBranch(in_channels, out_channels, dilation=1)]
        for rate in rates:
            branches.append(_ASPPConvBranch(in_channels, out_channels, dilation=rate))
        branches.append(_ASPPPoolingBranch(in_channels, out_channels))
        self.branches = nn.ModuleList(branches)

        n_branches = len(branches)
        self.project = nn.Sequential(
            nn.Conv2d(n_branches * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3PlusResNet34(nn.Module):
    """DeepLabV3+ with a ResNet34 encoder (output_stride=32 -- see module
    docstring for why). Returns raw logits, shape (N, output_ch, H, W),
    matching the input spatial size exactly."""

    def __init__(self, img_ch: int = 3, output_ch: int = 1) -> None:
        super().__init__()
        backbone = resnet34(weights=None)
        if img_ch != 3:
            backbone.conv1 = nn.Conv2d(
                img_ch, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Split the encoder into the stem+layer1 (low-level, stride 4) and
        # layer2-4 (high-level, stride 32) stages -- using the real
        # submodules directly (not the private ``_forward_impl``) so the
        # intermediate low-level activation can be captured explicitly.
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # stride 4,  64 ch (low-level features)
        self.layer2 = backbone.layer2  # stride 8,  128 ch
        self.layer3 = backbone.layer3  # stride 16, 256 ch
        self.layer4 = backbone.layer4  # stride 32, 512 ch (high-level features)

        self.aspp = ASPP(HIGH_LEVEL_IN_CHANNELS, ASPP_OUT_CHANNELS)

        self.low_level_project = nn.Sequential(
            nn.Conv2d(LOW_LEVEL_IN_CHANNELS, LOW_LEVEL_PROJECT_CHANNELS, kernel_size=1, bias=False),
            nn.BatchNorm2d(LOW_LEVEL_PROJECT_CHANNELS),
            nn.ReLU(inplace=True),
        )

        decoder_in_channels = ASPP_OUT_CHANNELS + LOW_LEVEL_PROJECT_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )
        self.classifier = nn.Conv2d(256, output_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        low = self.stem(x)
        low = self.layer1(low)  # stride 4, 64 ch -- low-level features

        high = self.layer2(low)
        high = self.layer3(high)
        high = self.layer4(high)  # stride 32, 512 ch -- high-level features

        aspp_out = self.aspp(high)
        aspp_out = F.interpolate(aspp_out, size=low.shape[-2:], mode="bilinear", align_corners=False)

        low_proj = self.low_level_project(low)
        decoder_in = torch.cat([aspp_out, low_proj], dim=1)
        decoded = self.decoder(decoder_in)

        logits = self.classifier(decoded)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        return logits


def build_deeplabv3plus(img_ch: int = 3, output_ch: int = 1) -> DeepLabV3PlusResNet34:
    """Construct a fresh (randomly initialized, ``weights=None``) DeepLabV3+
    with a ResNet34 encoder. No pretrained weights -- see module docstring."""
    return DeepLabV3PlusResNet34(img_ch=img_ch, output_ch=output_ch)
