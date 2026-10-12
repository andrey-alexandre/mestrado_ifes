"""Build the frozen Feature Extraction package for one image/mask pair."""
from typing import Dict, Optional

from src.features.asymmetry import extract_abcd_asymmetry
from src.features.blue_white_veil import extract_blue_white_veil
from src.features.border import extract_abcd_border
from src.features.color import classify_abcd_colors
from src.features.pigment_network import extract_pigment_network_raw
from src.features.schemas import (
    ABCDFeaturePackage,
    EvidenceEnvelope,
    EvidenceStatus,
    FeatureExtractionResult,
    MenziesFeaturePackage,
    SevenPointFeaturePackage,
    SharedEvidencePackage,
)
from src.features.structureless_blotch import extract_structureless_blotch
from src.features.utils import image_id_from_path, load_rgb_and_mask


UNAVAILABLE_SOURCE = "ADR-009 and ADR-010 frozen scientific readiness"


def unavailable(extractor_id: str, reason: str) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        status=EvidenceStatus.UNAVAILABLE,
        value=None,
        evidence={},
        limitations=["null denotes unresolved/unavailable, not clinical absence."],
        source=UNAVAILABLE_SOURCE,
        extractor_id=extractor_id,
        failure_reason=reason,
    )


def raw_forwarded(source: EvidenceEnvelope, extractor_id: str, limitation: str) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        status=EvidenceStatus.RAW_EVIDENCE,
        value=None,
        evidence=source.evidence,
        limitations=[limitation, *source.limitations],
        source=source.source,
        extractor_id=extractor_id,
        failure_reason=source.failure_reason,
    )


def _structureless_abcd(blotch: EvidenceEnvelope) -> EvidenceEnvelope:
    if blotch.status == EvidenceStatus.UNAVAILABLE:
        return unavailable(
            "ABCD-D-STRUCTURELESS-AREA-UNAVAILABLE",
            blotch.failure_reason or "Structureless-area source evidence unavailable.",
        )
    raw = blotch.evidence.get("abcd_structureless_area", {})
    return EvidenceEnvelope(
        status=EvidenceStatus.DETERMINISTIC_CLINICAL_WITH_LIMITATION,
        value=raw.get("value"),
        evidence={
            "source_feature": "shared.structureless_blotch",
            "total_blotch_fraction": blotch.evidence.get("total_blotch_fraction"),
            "threshold": raw.get("threshold", "total_blotch_fraction >= 0.10"),
            "status_in_source_feature": raw.get("status"),
            "raw_intermediate_outputs_preserved": True,
        },
        limitations=[
            "ABCD structureless-area evidence only; complete ABCD D remains unresolved.",
            "Clinical-consensus 10% structureless-area mapping is separate from Stoecker blotch segmentation.",
            *blotch.limitations,
        ],
        source="ADR-010 clinical-consensus structureless-area mapping over Stoecker absolute-color evidence",
        extractor_id="ABCD-D-STRUCTURELESS-AREA-001",
        failure_reason=blotch.failure_reason,
    )


def build_feature_package(
    image_path: str,
    mask_path: Optional[str],
    experiment_condition: str = "image_plus_fe",
    config: Optional[Dict] = None,
) -> FeatureExtractionResult:
    config = config or {}
    image_id = image_id_from_path(image_path)
    metadata = {
        "ph2_calibration_performed": False,
        "scientific_decision_source": "frozen feature extraction specification / ADR-009 / ADR-010",
        "published_thresholds_are_not_universal_clinical_thresholds": True,
        "experiment_condition": experiment_condition,
    }
    if not mask_path:
        reason = "Feature extraction requires a lesion mask; no mask_path was provided."
        unavailable_all = unavailable("FE-MASK-REQUIRED", reason)
        return FeatureExtractionResult(
            image_id=image_id,
            image_path=image_path,
            mask_path=mask_path,
            experiment_condition=experiment_condition,
            abcd=ABCDFeaturePackage(
                asymmetry=unavailable_all,
                border=unavailable_all,
                color=unavailable_all,
                dermoscopic_structures_d=unavailable("ABCD-D-UNRESOLVED", "ABCD_D = null / STILL_UNRESOLVED."),
                pigment_network_presence=unavailable("ABCD-PN-PRESENCE-UNRESOLVED", "No deterministic ABCD pigment-network presence mapping."),
                structureless_area=unavailable_all,
                branched_streaks=unavailable("ABCD-BRANCHED-STREAKS-UNRESOLVED", "No implementation-ready extractor."),
                dots=unavailable("ABCD-DOTS-UNRESOLVED", "No implementation-ready extractor."),
                globules=unavailable("ABCD-GLOBULES-UNRESOLVED", "No implementation-ready extractor."),
            ),
            seven_point=SevenPointFeaturePackage(
                atypical_pigment_network=unavailable("7PCL-APN-UNRESOLVED", "No deterministic 7PCL-r atypical pigment-network mapping."),
                blue_whitish_veil=unavailable("7PCL-BWV-BINARY-UNRESOLVED", "No deterministic lesion-level 7PCL blue-whitish veil binary mapping."),
                irregular_streaks=unavailable("7PCL-STREAKS-UNRESOLVED", "No deterministic 7PCL-r irregular streaks extractor."),
                irregular_pigmentation=unavailable("7PCL-PIGMENTATION-UNRESOLVED", "No deterministic 7PCL-r irregular pigmentation mapping."),
                irregular_dots_globules=unavailable("7PCL-DOTS-GLOBULES-UNRESOLVED", "No deterministic 7PCL-r dots/globules mapping."),
            ),
            menzies=MenziesFeaturePackage(
                broadened_network=unavailable("MENZIES-BROADENED-NETWORK-UNRESOLVED", "No deterministic Menzies-r broadened-network mapping."),
                blue_white_veil=unavailable("MENZIES-BWV-BINARY-UNRESOLVED", "No deterministic lesion-level Menzies blue-white veil binary mapping."),
                blue_gray_dots=unavailable("MENZIES-BLUE-GRAY-DOTS-UNRESOLVED", "No implementation-ready extractor."),
                peripheral_black_dots_globules=unavailable("MENZIES-PERIPHERAL-DOTS-GLOBULES-UNRESOLVED", "No implementation-ready extractor."),
                radial_streaming=unavailable("MENZIES-RADIAL-STREAMING-UNRESOLVED", "No implementation-ready extractor."),
                pseudopods=unavailable("MENZIES-PSEUDOPODS-UNRESOLVED", "No implementation-ready extractor."),
                color_evidence=unavailable_all,
            ),
            shared=SharedEvidencePackage(
                pigment_network=unavailable_all,
                blue_white_veil=unavailable_all,
                structureless_blotch=unavailable_all,
            ),
            metadata=metadata,
        )

    rgb, lesion_mask = load_rgb_and_mask(image_path, mask_path)
    asymmetry = extract_abcd_asymmetry(rgb, lesion_mask)
    border = extract_abcd_border(lesion_mask)
    color = classify_abcd_colors(rgb, lesion_mask)
    pigment_network = extract_pigment_network_raw(rgb, lesion_mask)
    blue_white_veil = extract_blue_white_veil(rgb, lesion_mask)
    blotch = extract_structureless_blotch(
        rgb,
        lesion_mask,
        connectivity=int(config.get("stoecker_connectivity", 8)),
        min_area=int(config.get("stoecker_min_area", 800)),
    )

    abcd = ABCDFeaturePackage(
        asymmetry=asymmetry,
        border=border,
        color=color,
        dermoscopic_structures_d=unavailable("ABCD-D-STILL-UNRESOLVED", "ABCD_D = null / STILL_UNRESOLVED; no complete ABCD D score is emitted."),
        pigment_network_presence=unavailable("ABCD-PN-PRESENCE-UNRESOLVED", "Raw pigment-network evidence has no frozen deterministic clinical presence mapping."),
        structureless_area=_structureless_abcd(blotch),
        branched_streaks=unavailable("ABCD-BRANCHED-STREAKS-UNRESOLVED", "No implementation-ready extractor."),
        dots=unavailable("ABCD-DOTS-UNRESOLVED", "No implementation-ready extractor."),
        globules=unavailable("ABCD-GLOBULES-UNRESOLVED", "No implementation-ready extractor."),
    )
    seven_point = SevenPointFeaturePackage(
        atypical_pigment_network=unavailable("7PCL-APN-UNRESOLVED", "Raw pigment-network evidence has no deterministic 7PCL-r atypical-network mapping."),
        blue_whitish_veil=raw_forwarded(blue_white_veil, "7PCL-BWV-RAW-EVIDENCE-ONLY", "Raw pixel-level BWV evidence only; lesion-level 7PCL binary remains unavailable."),
        irregular_streaks=unavailable("7PCL-STREAKS-UNRESOLVED", "No deterministic 7PCL-r irregular streaks extractor."),
        irregular_pigmentation=unavailable("7PCL-PIGMENTATION-UNRESOLVED", "No deterministic 7PCL-r irregular pigmentation mapping."),
        irregular_dots_globules=unavailable("7PCL-DOTS-GLOBULES-UNRESOLVED", "No deterministic 7PCL-r dots/globules mapping."),
    )
    menzies = MenziesFeaturePackage(
        broadened_network=unavailable("MENZIES-BROADENED-NETWORK-UNRESOLVED", "Raw pigment-network evidence has no deterministic Menzies-r broadened-network mapping."),
        blue_white_veil=raw_forwarded(blue_white_veil, "MENZIES-BWV-RAW-EVIDENCE-ONLY", "Raw pixel-level BWV evidence only; lesion-level Menzies binary remains unavailable."),
        blue_gray_dots=unavailable("MENZIES-BLUE-GRAY-DOTS-UNRESOLVED", "No implementation-ready extractor."),
        peripheral_black_dots_globules=unavailable("MENZIES-PERIPHERAL-DOTS-GLOBULES-UNRESOLVED", "No implementation-ready extractor."),
        radial_streaming=unavailable("MENZIES-RADIAL-STREAMING-UNRESOLVED", "No implementation-ready extractor."),
        pseudopods=unavailable("MENZIES-PSEUDOPODS-UNRESOLVED", "No implementation-ready extractor."),
        color_evidence=raw_forwarded(color, "MENZIES-COLOR-RAW-EVIDENCE", "ABCD C color counts are shared as evidence only for Menzies; no new Menzies color formula is introduced."),
    )
    shared = SharedEvidencePackage(
        pigment_network=pigment_network,
        blue_white_veil=blue_white_veil,
        structureless_blotch=blotch,
    )
    return FeatureExtractionResult(
        image_id=image_id,
        image_path=image_path,
        mask_path=mask_path,
        experiment_condition=experiment_condition,
        abcd=abcd,
        seven_point=seven_point,
        menzies=menzies,
        shared=shared,
        metadata=metadata,
    )
