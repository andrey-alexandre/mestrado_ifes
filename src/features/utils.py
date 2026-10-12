"""Utility functions shared by dermoscopic feature extractors."""
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def load_rgb_and_mask(image_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    rgb = np.asarray(image, dtype=np.uint8)
    lesion_mask = np.asarray(mask) > 128
    if not lesion_mask.any():
        raise ValueError("Lesion mask contains no positive pixels.")
    return rgb, lesion_mask


def image_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float64)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def binary_erosion_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    out = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def binary_dilation_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    out = np.zeros(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    return mask.astype(bool) & ~binary_erosion_3x3(mask)


def connected_components(mask: np.ndarray, connectivity: int = 8):
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        pixels = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(pixels)
    return components


def binary_mask_rle(mask: np.ndarray) -> dict:
    flat = mask.astype(np.uint8).ravel(order="C")
    if flat.size == 0:
        return {"shape": list(mask.shape), "order": "C", "starts": [], "lengths": []}
    padded = np.concatenate([[0], flat, [0]])
    changes = np.nonzero(padded[1:] != padded[:-1])[0]
    starts = changes[0::2]
    ends = changes[1::2]
    return {
        "shape": list(mask.shape),
        "order": "C",
        "starts": starts.astype(int).tolist(),
        "lengths": (ends - starts).astype(int).tolist(),
    }
