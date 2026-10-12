"""ABCD A asymmetry extraction using the frozen Kasmi and Mokrani path."""
import math
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from src.features.geometry import principal_axis as _shared_principal_axis
from src.features.schemas import EvidenceEnvelope, EvidenceStatus
from src.features.utils import rgb_to_luminance


SOURCE = "Kasmi and Mokrani 2016 ABCD asymmetry operationalization"
BLOCK_SIZE = 20


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float64) / 255.0
    linear = np.where(arr > 0.04045, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    x = linear[..., 0] * 0.4124564 + linear[..., 1] * 0.3575761 + linear[..., 2] * 0.1804375
    y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    z = linear[..., 0] * 0.0193339 + linear[..., 1] * 0.1191920 + linear[..., 2] * 0.9503041
    xyz = np.stack([x / 0.95047, y, z / 1.08883], axis=-1)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([l, a, b], axis=-1)


def principal_axis(mask: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    return _shared_principal_axis(mask)


def block_has_required_lesion_coverage(block_mask: np.ndarray, threshold: float = 0.75) -> bool:
    return bool(block_mask.mean() >= threshold)


def color_symmetric_from_deltas(delta_e_values, threshold: float = 6.0) -> bool:
    if not delta_e_values:
        return False
    symmetric = sum(1 for value in delta_e_values if value <= threshold)
    asymmetric = len(delta_e_values) - symmetric
    return bool(symmetric > asymmetric)


def _rotate_pair(rgb: np.ndarray, mask: np.ndarray, angle_radians: float):
    angle_degrees = -math.degrees(angle_radians)
    img = Image.fromarray(rgb).rotate(angle_degrees, resample=Image.BICUBIC, expand=False)
    msk = Image.fromarray(mask.astype(np.uint8) * 255).rotate(
        angle_degrees,
        resample=Image.NEAREST,
        expand=False,
    )
    return np.asarray(img, dtype=np.uint8), np.asarray(msk) > 128


def _axis_metrics(rgb: np.ndarray, mask: np.ndarray, axis: str) -> Dict:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("No lesion pixels available.")
    cx = float(xs.mean())
    cy = float(ys.mean())
    lum = rgb_to_luminance(rgb)
    lab = _rgb_to_lab(rgb)
    if axis == "vertical":
        neg = mask & (np.indices(mask.shape)[1] < cx)
        pos = mask & (np.indices(mask.shape)[1] >= cx)
    elif axis == "horizontal":
        neg = mask & (np.indices(mask.shape)[0] < cy)
        pos = mask & (np.indices(mask.shape)[0] >= cy)
    else:
        raise ValueError("axis must be vertical or horizontal")

    area = int(mask.sum())
    shape_diff = abs(int(neg.sum()) - int(pos.sum())) / area if area else 1.0
    mean_lum = float(lum[mask].mean()) if area else 0.0
    if neg.any() and pos.any() and mean_lum > 0:
        brightness_diff = abs(float(lum[neg].mean()) - float(lum[pos].mean())) / mean_lum
    else:
        brightness_diff = 1.0

    color_deltas = []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    for y in range(y0, y1, BLOCK_SIZE):
        for x in range(x0, x1, BLOCK_SIZE):
            if axis == "vertical":
                mx = int(round(2 * cx - (x + BLOCK_SIZE)))
                my = y
            else:
                mx = x
                my = int(round(2 * cy - (y + BLOCK_SIZE)))
            if mx < 0 or my < 0 or mx + BLOCK_SIZE > mask.shape[1] or my + BLOCK_SIZE > mask.shape[0]:
                continue
            patch_a = mask[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE]
            patch_b = mask[my : my + BLOCK_SIZE, mx : mx + BLOCK_SIZE]
            if not block_has_required_lesion_coverage(patch_a) or not block_has_required_lesion_coverage(patch_b):
                continue
            lab_a = lab[y : y + BLOCK_SIZE, x : x + BLOCK_SIZE][patch_a].mean(axis=0)
            lab_b = lab[my : my + BLOCK_SIZE, mx : mx + BLOCK_SIZE][patch_b].mean(axis=0)
            color_deltas.append(float(np.linalg.norm(lab_a - lab_b)))
    symmetric_pair_count = sum(1 for value in color_deltas if value <= 6.0)
    asymmetric_pair_count = len(color_deltas) - symmetric_pair_count
    color_delta = float(max(color_deltas)) if color_deltas else None

    return {
        "shape_difference_fraction": shape_diff,
        "brightness_difference_fraction": brightness_diff,
        "max_block_delta_e": color_delta,
        "mirrored_block_pairs": len(color_deltas),
        "color_symmetric_pair_count": symmetric_pair_count,
        "color_asymmetric_pair_count": asymmetric_pair_count,
        "block_lesion_coverage_threshold": 0.75,
        "color_axis_rule": "color-symmetric when symmetric mirrored block pairs outnumber asymmetric block pairs",
        "shape_symmetric": shape_diff < 0.02,
        "brightness_symmetric": brightness_diff < 0.03,
        "color_symmetric": color_symmetric_from_deltas(color_deltas),
    }


def extract_abcd_asymmetry(rgb: np.ndarray, lesion_mask: np.ndarray) -> EvidenceEnvelope:
    try:
        angle, centroid = principal_axis(lesion_mask)
        aligned_rgb, aligned_mask = _rotate_pair(rgb, lesion_mask, angle)
        axes = {
            "major_axis": _axis_metrics(aligned_rgb, aligned_mask, "vertical"),
            "minor_axis": _axis_metrics(aligned_rgb, aligned_mask, "horizontal"),
        }
        symmetric_axes = {
            name: bool(
                metrics["shape_symmetric"]
                and metrics["brightness_symmetric"]
                and metrics["color_symmetric"]
            )
            for name, metrics in axes.items()
        }
        asymmetric_axis_count = 2 - sum(1 for value in symmetric_axes.values() if value)
        return EvidenceEnvelope(
            status=EvidenceStatus.DETERMINISTIC_CLINICAL,
            value=int(asymmetric_axis_count),
            evidence={
                "center_of_gravity_xy": [centroid[0], centroid[1]],
                "principal_axis_degrees": math.degrees(angle),
                "axis_symmetry": symmetric_axes,
                "axis_metrics": axes,
                "rotation_interpolation": "AUTHOR_IMPLEMENTATION_CONVENTION: RGB uses bicubic interpolation; mask uses nearest-neighbor interpolation.",
                "axis_pixel_assignment": "AUTHOR_IMPLEMENTATION_CONVENTION: pixels on or above/right of the centroid line are assigned to the positive half.",
                "thresholds": {
                    "color_delta_e_max": 6,
                    "brightness_difference_fraction_lt": 0.03,
                    "shape_difference_fraction_lt": 0.02,
                },
            },
            limitations=[
                "Published computational implementation thresholds, not universal clinical thresholds.",
                "No PH2 calibration.",
            ],
            source=SOURCE,
            extractor_id="ABCD-A-001",
        )
    except Exception as exc:
        return EvidenceEnvelope(
            status=EvidenceStatus.UNAVAILABLE,
            value=None,
            source=SOURCE,
            extractor_id="ABCD-A-001",
            failure_reason=str(exc),
        )
