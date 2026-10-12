"""Debug visualizations for frozen Feature Extraction outputs."""
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

from src.features.asymmetry import principal_axis
from src.features.blue_white_veil import compute_skin_reference, majority_filter
from src.features.utils import boundary_mask, load_rgb_and_mask


def _overlay(rgb: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float64).copy()
    layer = np.array(color, dtype=np.float64)
    out[mask] = (1.0 - alpha) * out[mask] + alpha * layer
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_axis(image: Image.Image, center, angle, color):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    length = max(w, h)
    cx, cy = center
    dx = np.cos(angle) * length
    dy = np.sin(angle) * length
    draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color, width=2)


def save_feature_debug_visualization(image_path: str, mask_path: str, output_dir: str, image_id: Optional[str] = None) -> str:
    rgb, lesion_mask = load_rgb_and_mask(image_path, mask_path)
    image_id = image_id or Path(image_path).stem
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    candidate_blotch = lesion_mask & (rgb[:, :, 0] < 120) & (rgb[:, :, 1] < 60)
    try:
        skin_rgb = compute_skin_reference(rgb, lesion_mask)
        arr = rgb.astype(np.float64)
        channel_sum = arr.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            f3 = np.where(channel_sum > 0, arr[:, :, 2] / channel_sum, 0.0)
        f10 = arr[:, :, 0] - float(skin_rgb[0])
        veil = majority_filter(lesion_mask & (f3 > 0.3) & (f10 > -194) & (f10 <= -51)) & lesion_mask
    except Exception:
        veil = np.zeros(lesion_mask.shape, dtype=bool)

    panel = np.concatenate(
        [
            rgb,
            _overlay(rgb, boundary_mask(lesion_mask), (255, 255, 0), 0.85),
            _overlay(rgb, candidate_blotch, (255, 0, 0), 0.65),
            _overlay(rgb, veil, (0, 130, 255), 0.65),
        ],
        axis=1,
    )
    img = Image.fromarray(panel)
    try:
        angle, center = principal_axis(lesion_mask)
        _draw_axis(img, center, angle, (0, 255, 0))
        _draw_axis(img, center, angle + np.pi / 2.0, (0, 180, 255))
    except Exception:
        pass
    path = output / f"{image_id}_feature_debug.png"
    img.save(path)
    return str(path)
