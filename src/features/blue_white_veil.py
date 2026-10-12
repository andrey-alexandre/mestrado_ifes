"""Celebi et al. 2008 blue-white veil pixel detector."""
import numpy as np
from scipy.ndimage import distance_transform_edt

from src.features.schemas import EvidenceEnvelope, EvidenceStatus
from src.features.utils import binary_mask_rle


SOURCE = "Celebi et al. 2008, Figure 5 pixel-level C4.5 tree"


def majority_filter(mask: np.ndarray, size: int = 5) -> np.ndarray:
    if size != 5:
        raise ValueError("Celebi postprocessing uses a frozen 5x5 majority filter.")
    pad = size // 2
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
    votes = np.zeros(mask.shape, dtype=np.uint16)
    for dy in range(size):
        for dx in range(size):
            votes += padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return votes > (size * size // 2)


def compute_skin_reference(rgb: np.ndarray, lesion_mask: np.ndarray):
    # Celebi omits an outside area equal to 10% of lesion area, then samples
    # the next outside region equal to 20% of lesion area.
    if not lesion_mask.any():
        raise ValueError("No lesion pixels available.")
    outside = ~lesion_mask
    distances = distance_transform_edt(outside)
    ys, xs = np.nonzero(outside)
    if xs.size == 0:
        raise ValueError("No outside-lesion pixels available for skin reference.")
    order = np.lexsort((xs, ys, distances[ys, xs]))
    lesion_area = int(lesion_mask.sum())
    omit = min(int(round(0.10 * lesion_area)), order.size)
    take = min(int(round(0.20 * lesion_area)), max(0, order.size - omit))
    selected_y = ys[order[omit : omit + take]]
    selected_x = xs[order[omit : omit + take]]
    if selected_y.size == 0:
        raise ValueError("Skin reference band is empty.")
    band_pixels = rgb[selected_y, selected_x]
    skin = (
        (band_pixels[:, 0] > 90)
        & (band_pixels[:, 0] > band_pixels[:, 2])
        & (band_pixels[:, 0] > band_pixels[:, 1])
    )
    if not skin.any():
        raise ValueError("No Celebi-valid skin pixels available in reference band.")
    return band_pixels[skin].astype(np.float64).mean(axis=0)


def extract_blue_white_veil(rgb: np.ndarray, lesion_mask: np.ndarray) -> EvidenceEnvelope:
    try:
        skin_rgb = compute_skin_reference(rgb, lesion_mask)
        lesion_rgb = rgb.astype(np.float64)
        channel_sum = lesion_rgb.sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            f3 = np.where(channel_sum > 0, lesion_rgb[:, :, 2] / channel_sum, 0.0)
        f10 = lesion_rgb[:, :, 0] - float(skin_rgb[0])
        initial = lesion_mask & (f3 > 0.3) & (f10 > -194) & (f10 <= -51)
        filtered = majority_filter(initial) & lesion_mask
        lesion_count = int(lesion_mask.sum())
        veil_count = int(filtered.sum())
        s1 = veil_count / lesion_count if lesion_count else None
        return EvidenceEnvelope(
            status=EvidenceStatus.RAW_EVIDENCE,
            value=None,
            evidence={
                "veil_pixel_count": veil_count,
                "lesion_pixel_count": lesion_count,
                "veil_area_fraction": s1,
                "s1": s1,
                "initial_veil_mask_rle": binary_mask_rle(initial),
                "veil_mask_rle": binary_mask_rle(filtered),
                "skin_rgb": skin_rgb.tolist(),
                "figure_5_rule": "F3 > 0.3 AND -194 < F10 <= -51",
                "postprocessing": "5x5 majority filter",
                "skin_reference_distance": "Euclidean distance transform",
                "skin_reference_tie_break": "AUTHOR_IMPLEMENTATION_CONVENTION: pixels at identical distance are ordered by row, then column when a partial equal-distance ring crosses the 10%/20% area boundary.",
            },
            limitations=[
                "Pixel-level detector only; no 7PCL/Menzies lesion-level binary mapping.",
                "Do not use S1=0.009 or S1>0 as a veil-presence rule.",
                "Skin-reference equal-distance tie-breaking may slightly change skin RGB when a partial distance ring is needed.",
                "No PH2 calibration.",
            ],
            source=SOURCE,
            extractor_id="VEIL-DETECT-001",
        )
    except Exception as exc:
        return EvidenceEnvelope(
            status=EvidenceStatus.UNAVAILABLE,
            value=None,
            source=SOURCE,
            extractor_id="VEIL-DETECT-001",
            failure_reason=str(exc),
        )
