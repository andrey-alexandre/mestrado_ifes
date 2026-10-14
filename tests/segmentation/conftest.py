"""Fixtures sintéticas: sem GPU, sem datasets reais, sem pacote externo da U-Net."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── U-Net miniatura com a mesma nomenclatura de blocos da U_Net legada ─────────

def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super().__init__()
        c = [4, 6, 8, 8, 8]
        self.Maxpool = nn.MaxPool2d(2, 2)
        self.Conv1, self.Conv2 = _block(img_ch, c[0]), _block(c[0], c[1])
        self.Conv3, self.Conv4, self.Conv5 = _block(c[1], c[2]), _block(c[2], c[3]), _block(c[3], c[4])
        self.Up5 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c[4], c[3], 3, padding=1))
        self.Up_conv5 = _block(c[3] * 2, c[3])
        self.Up4 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c[3], c[2], 3, padding=1))
        self.Up_conv4 = _block(c[2] * 2, c[2])
        self.Up3 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c[2], c[1], 3, padding=1))
        self.Up_conv3 = _block(c[1] * 2, c[1])
        self.Up2 = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(c[1], c[0], 3, padding=1))
        self.Up_conv2 = _block(c[0] * 2, c[0])
        self.Conv_1x1 = nn.Conv2d(c[0], output_ch, 1)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Conv2(self.Maxpool(x1))
        x3 = self.Conv3(self.Maxpool(x2))
        x4 = self.Conv4(self.Maxpool(x3))
        x5 = self.Conv5(self.Maxpool(x4))
        d5 = self.Up_conv5(torch.cat((x4, self.Up5(x5)), 1))
        d4 = self.Up_conv4(torch.cat((x3, self.Up4(d5)), 1))
        d3 = self.Up_conv3(torch.cat((x2, self.Up3(d4)), 1))
        d2 = self.Up_conv2(torch.cat((x1, self.Up2(d3)), 1))
        return self.Conv_1x1(d2)


def _install_fake_network():
    try:
        import Image_Segmentation.code.network  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    pkg = types.ModuleType("Image_Segmentation")
    code = types.ModuleType("Image_Segmentation.code")
    net = types.ModuleType("Image_Segmentation.code.network")
    net.U_Net = net.R2U_Net = net.AttU_Net = net.R2AttU_Net = TinyUNet
    pkg.code, code.network = code, net
    sys.modules.update({"Image_Segmentation": pkg, "Image_Segmentation.code": code,
                        "Image_Segmentation.code.network": net})


_install_fake_network()


# ── geração de dados sintéticos ─────────────────────────────────────────────────

def _lesion(h, w, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    cy, cx = h / 2 + rng.integers(-3, 4), w / 2 + rng.integers(-3, 4)
    ry, rx = h * rng.uniform(0.22, 0.34), w * rng.uniform(0.22, 0.34)
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    img[mask] = (170 + rng.integers(-10, 10), 90, 60)
    img = np.clip(img + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    return img, mask


@pytest.fixture
def make_isic(tmp_path):
    def _make(n=24, with_groups=False, h=48, w=64):
        img_dir, mask_dir = tmp_path / "isic_img", tmp_path / "isic_mask"
        img_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        for i in range(n):
            img, mask = _lesion(h, w, seed=i)
            Image.fromarray(img).save(img_dir / f"ISIC_{i:07d}.jpg", quality=95)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_dir / f"ISIC_{i:07d}_segmentation.png")
        groups = None
        if with_groups:
            groups = tmp_path / "groups.csv"
            groups.write_text("image_id,lesion_id\n" + "\n".join(
                f"ISIC_{i:07d},L{i // 2}" for i in range(n)), encoding="utf-8")
        return img_dir, mask_dir, groups
    return _make


@pytest.fixture
def make_ph2(tmp_path):
    """30 imagens: 10 nevo comum (0), 10 atípico (1), 10 melanoma (2)."""
    def _make(n_per_class=10, h=40, w=56):
        root = tmp_path / "PH2"
        lines = ["||   Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry | Colors ||"]
        k = 0
        for clin in (0, 1, 2):
            for _ in range(n_per_class):
                name = f"IMD{k + 100:03d}"
                (root / name / f"{name}_Dermoscopic_Image").mkdir(parents=True)
                (root / name / f"{name}_lesion").mkdir(parents=True)
                img, mask = _lesion(h, w, seed=1000 + k)
                Image.fromarray(img).save(root / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp")
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    root / name / f"{name}_lesion" / f"{name}_lesion.bmp")
                hist = "Melanoma" if (clin == 2 and k % 3 != 0) else ""  # histologia parcialmente preenchida
                lines.append(f"|| {name} || {hist:>12} || {clin} || 0 | 3 ||")
                k += 1
        meta = tmp_path / "PH2_dataset.txt"
        meta.write_text("\n".join(lines) + "\n\nLegends for Clinical Diagnosis:\n0 - Common Nevus;\n",
                        encoding="utf-8")
        return root, meta
    return _make


@pytest.fixture
def cfg_factory(tmp_path):
    from src.segmentation import config as cfgmod

    def _make(isic=None, ph2=None, **overrides):
        sec = {
            "datasets": {"isic2018": {"val_fraction": 0.25}, "ph2": {}},
            "segmentation": {
                "image_size": 32, "batch_size": 4, "max_epochs": 2, "learning_rate": 5e-3,
                "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0, "monitor": "val_dice"},
                "augmentation": {"hflip": True, "vflip": True, "rotation_degrees": 10, "color_jitter": 0.05},
                "seed": 7, "device": "cpu",
            },
            "transfer_learning": {"max_epochs": 2, "inner_val_fraction": 0.25, "learning_rate": 1e-3,
                                  "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0,
                                                     "monitor": "val_dice"}},
            "evaluation": {"n_folds": 5, "bootstrap": {"n_resamples": 300, "confidence": 0.95, "seed": 1},
                           "n_examples": 2},
            "output": {"root_dir": str(tmp_path / "out")},
        }
        if isic:
            sec["datasets"]["isic2018"].update({"image_dir": str(isic[0]), "mask_dir": str(isic[1])})
            if len(isic) > 2 and isic[2]:
                sec["datasets"]["isic2018"]["lesion_groups_csv"] = str(isic[2])
        full = {"paths": {}, "segmentation_experiment": sec, "llm": {"model_name": "fake-llm", "temperature": 0}}
        if ph2:
            sec["datasets"]["ph2"].update({"dataset_dir": str(ph2[0]), "metadata_txt": str(ph2[1])})
        for dotted, val in overrides.items():
            cur = sec
            *head, last = dotted.split("__")
            for k in head:
                cur = cur.setdefault(k, {})
            cur[last] = val
        return cfgmod.build_config(full)
    return _make


# ── mundo sintético + execução das etapas 1–5 ──────────────────────────────────

@pytest.fixture
def world(make_isic, make_ph2, cfg_factory):
    isic, ph2 = make_isic(24), make_ph2()
    return cfg_factory(isic=isic, ph2=ph2), isic, ph2


@pytest.fixture
def full_run(world):
    """Executa etapas 1–5 com dados sintéticos (poucas épocas, CPU)."""
    from src.segmentation import artifacts as A, comparison, stages
    cfg, *_ = world
    store = A.StageStore(cfg["output"]["results_dir"])
    stages.run_train_isic(cfg, store)
    stages.run_make_folds(cfg, store)
    stages.run_direct_transfer(cfg, store)
    stages.run_finetune(cfg, store)
    comparison.run_comparison(cfg, store)
    return cfg, store
