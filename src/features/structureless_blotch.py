"""Stoecker et al. 2005 absolute-color blotch evidence."""
import numpy as np

from src.features.schemas import EvidenceEnvelope, EvidenceStatus
from src.features.utils import binary_mask_rle, connected_components


SOURCE = "Stoecker et al. 2005 absolute-color path"


def extract_structureless_blotch(
    rgb: np.ndarray,
    lesion_mask: np.ndarray,
    connectivity: int = 8,
    min_area: int = 800,
) -> EvidenceEnvelope:
    try:
        candidate = lesion_mask & (rgb[:, :, 0] < 120) & (rgb[:, :, 1] < 60)
        components = connected_components(candidate, connectivity=connectivity)
        areas = [len(component) for component in components]
        retained_areas = [area for area in areas if area > min_area]
        retained_mask = np.zeros(candidate.shape, dtype=bool)
        for component in components:
            if len(component) > min_area:
                ys, xs = zip(*component)
                retained_mask[list(ys), list(xs)] = True
        lesion_area = int(lesion_mask.sum())
        total_area = int(sum(retained_areas))
        largest_area = int(max(retained_areas)) if retained_areas else 0
        total_fraction = total_area / lesion_area if lesion_area else None
        largest_fraction = largest_area / lesion_area if lesion_area else None
        structureless_present = bool(total_fraction is not None and total_fraction >= 0.10)
        return EvidenceEnvelope(
            status=EvidenceStatus.RAW_EVIDENCE,
            value=None,
            evidence={
                "candidate_pixel_count": int(candidate.sum()),
                "absolute_color_candidate_mask_rle": binary_mask_rle(candidate),
                "retained_blotch_mask_rle": binary_mask_rle(retained_mask),
                "component_pixel_areas": areas,
                "retained_component_pixel_areas": retained_areas,
                "retained_component_count": len(retained_areas),
                "total_retained_blotch_area": total_area,
                "total_blotch_fraction": total_fraction,
                "largest_retained_component_area": largest_area,
                "largest_blotch_fraction": largest_fraction,
                "abcd_structureless_area": {
                    "status": "DETERMINISTIC_CLINICAL_WITH_LIMITATION",
                    "value": structureless_present,
                    "threshold": "total_blotch_fraction >= 0.10",
                },
                "absolute_color_rule": "R < 120 AND G < 60",
                "component_filter": f"A_blotch > {min_area} pixels",
                "connected_component_connectivity": connectivity,
                "connectivity_origin": "AUTHOR_IMPLEMENTATION_CONVENTION",
            },
            limitations=[
                "Stoecker relative-color path is blocked and not implemented.",
                "A_blotch > 800 pixels is source-scale-specific and not rescaled for PH2.",
                "No PH2 calibration.",
            ],
            source=SOURCE,
            extractor_id="STRUCTURELESS-BLOTCH-001",
        )
    except Exception as exc:
        return EvidenceEnvelope(
            status=EvidenceStatus.UNAVAILABLE,
            value=None,
            source=SOURCE,
            extractor_id="STRUCTURELESS-BLOTCH-001",
            failure_reason=str(exc),
        )
