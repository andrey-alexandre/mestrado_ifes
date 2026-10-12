"""Pigment-network extractor state after scientific conformance audit.

The frozen documentation approves Anantha et al. as RAW_EVIDENCE only when the
published NGLDM/Laws operation can be faithfully reproduced. The repository
documentation records the required parameters, but not enough executable detail
for the NGLDM matrix/rules, Laws masks, and final block classifier. Therefore no
custom proxy is emitted under the Anantha method name.
"""
from src.features.schemas import EvidenceEnvelope, EvidenceStatus


SOURCE = "Anantha et al. 2004 fallback; frozen documentation lacks full executable classifier detail"


def extract_pigment_network_raw(rgb, lesion_mask, block_size: int = 64) -> EvidenceEnvelope:
    return EvidenceEnvelope(
        status=EvidenceStatus.UNAVAILABLE,
        value=None,
        evidence={
            "requested_block_size": block_size,
            "documented_parameters_not_executed": {
                "luminance": "0.29R + 0.59G + 0.11B",
                "ngldm_a": 4,
                "ngldm_d": 5,
                "tn1": 5800,
                "laws_masks": ["9x9", "11x11", "13x13"],
                "laws_weights": "40% 9x9 and 60% 11x11",
                "tl1": 30000,
                "tl2": 30000,
                "tl3": 130,
                "tl4": 180,
            },
        },
        limitations=[
            "SCIENTIFIC_SPECIFICATION_GAP: frozen docs do not fully specify the executable Anantha NGLDM/Laws classifier.",
            "No approximation is emitted as pigment-network evidence.",
            "ABCD D, 7PCL atypical network, and Menzies broadened network remain unavailable.",
            "No PH2 calibration.",
        ],
        source=SOURCE,
        extractor_id="PIGMENT-NETWORK-RAW-001",
        failure_reason=(
            "UNAVAILABLE_DUE_TO_SPECIFICATION_GAP: previous variance/neighbor-difference "
            "proxy was removed because it was not the published Anantha NGLDM/Laws method."
        ),
    )
