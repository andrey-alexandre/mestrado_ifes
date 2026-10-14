"""Fixtures sintéticas: sem datasets reais, sem APIs, sem GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset_validation import config as cfgmod  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    return cfgmod.build_config({"paths": {}, "llm": {"temperature": 0}}, {
        "sample": {"n_per_class": 3},
        "output": {"root_dir": str(tmp_path / "out")},
        "datasets": {"ph2": {"dataset_dir": str(tmp_path / "ph2"), "metadata_txt": str(tmp_path / "ph2.txt")},
                     "ham10000": {"image_dir": str(tmp_path / "ham"), "mask_dir": str(tmp_path / "hammask")}},
    })


def _img(path, size=(40, 30), color=(180, 120, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _mask(path, size=(40, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((size[1], size[0]), np.uint8)
    m[8:22, 10:30] = 255
    Image.fromarray(m).save(path)


@pytest.fixture
def ph2_records(cfg, tmp_path):
    recs = {}
    for i in range(12):
        name = f"IMD{i:03d}"
        code = 2 if i < 5 else (0 if i < 9 else 1)
        recs[name] = {"name": name, "clinical_diagnosis": code,
                      "histological_diagnosis": "Melanoma" if i < 2 else "",
                      "asymmetry": "0", "pigment_network": "T", "dots_globules": "A", "streaks": "A",
                      "regression_areas": "A", "blue_whitish_veil": "A", "colors": "3"}
        base = tmp_path / "ph2" / name
        _img(base / f"{name}_Dermoscopic_Image" / f"{name}.bmp")
        _mask(base / f"{name}_lesion" / f"{name}_lesion.bmp")
    return recs


@pytest.fixture
def ham_meta(cfg, tmp_path):
    rows = []
    n = 0
    for dx, count in (("mel", 8), ("nv", 12), ("bkl", 3)):
        for j in range(count):
            iid = f"ISIC_{n:07d}"
            rows.append({"lesion_id": f"HAM_{dx}{j // 2}", "image_id": iid, "dx": dx,
                         "dx_type": "histo" if dx == "mel" else "follow_up",
                         "age": "50", "sex": "male", "localization": "back",
                         "dataset": "vidir_modern" if j % 2 else "rosendahl"})
            _img(tmp_path / "ham" / f"{iid}.jpg")
            _mask(tmp_path / "hammask" / f"{iid}_segmentation.png")
            n += 1
    return rows
