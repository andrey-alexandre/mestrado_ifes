"""ABCD C color extraction using frozen Majumder and Ullah thresholds."""
from typing import Dict

import numpy as np

from src.features.schemas import EvidenceEnvelope, EvidenceStatus


COLOR_THRESHOLDS_SOURCE = "Majumder and Ullah 2019, normalized RGB table; 5% lesion-area rule"


def classify_abcd_colors(rgb: np.ndarray, mask: np.ndarray) -> EvidenceEnvelope:
    pixels = rgb[mask].astype(np.float64) / 255.0
    n = int(pixels.shape[0])
    if n == 0:
        return EvidenceEnvelope(
            status=EvidenceStatus.UNAVAILABLE,
            value=None,
            source=COLOR_THRESHOLDS_SOURCE,
            extractor_id="ABCD-C-001",
            failure_reason="No lesion pixels available.",
        )

    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    classes = {
        "white": (r >= 0.8) & (g >= 0.8) & (b >= 0.8),
        "red": (r >= 0.588) & (g < 0.2) & (b < 0.2),
        "light_brown": (r >= 0.588) & (r <= 0.94) & (g > 0.196) & (g <= 0.588) & (b > 0) & (b < 0.392),
        "dark_brown": (r > 0.243) & (r < 0.56) & (g >= 0) & (g < 0.392) & (b > 0) & (b < 0.392),
        "blue_gray": (r >= 0) & (r <= 0.588) & (g >= 0.392) & (g <= 0.588) & (b >= 0.490) & (b <= 0.588),
        "black": (r <= 0.243) & (g <= 0.243) & (b <= 0.243),
    }

    counts: Dict[str, int] = {name: int(values.sum()) for name, values in classes.items()}
    fractions = {name: counts[name] / n for name in classes}
    present = {name: fractions[name] > 0.05 for name in classes}
    color_count = sum(1 for is_present in present.values() if is_present)

    return EvidenceEnvelope(
        status=EvidenceStatus.DETERMINISTIC_CLINICAL,
        value=int(color_count),
        evidence={
            "lesion_pixel_count": n,
            "pixel_count_per_color": counts,
            "fraction_per_color": fractions,
            "present_per_color": present,
            "presence_rule": "fraction > 0.05",
        },
        limitations=[
            "Published computational/dataset-derived thresholds, not universal clinical thresholds.",
            "No PH2 calibration.",
        ],
        source=COLOR_THRESHOLDS_SOURCE,
        extractor_id="ABCD-C-001",
    )

