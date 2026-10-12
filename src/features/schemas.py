"""Pydantic schemas for deterministic dermoscopic feature extraction."""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EvidenceStatus(str, Enum):
    DETERMINISTIC_CLINICAL = "DETERMINISTIC_CLINICAL"
    DETERMINISTIC_CLINICAL_WITH_LIMITATION = "DETERMINISTIC_CLINICAL_WITH_LIMITATION"
    RAW_EVIDENCE = "RAW_EVIDENCE"
    UNAVAILABLE = "UNAVAILABLE"


class EvidenceEnvelope(BaseModel):
    """Common envelope that distinguishes clinical absence from unavailability."""

    status: EvidenceStatus
    value: Optional[Any] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    limitations: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    extractor_id: Optional[str] = None
    failure_reason: Optional[str] = None


class ABCDFeaturePackage(BaseModel):
    asymmetry: EvidenceEnvelope
    border: EvidenceEnvelope
    color: EvidenceEnvelope
    dermoscopic_structures_d: EvidenceEnvelope
    pigment_network_presence: EvidenceEnvelope
    structureless_area: EvidenceEnvelope
    branched_streaks: EvidenceEnvelope
    dots: EvidenceEnvelope
    globules: EvidenceEnvelope


class SevenPointFeaturePackage(BaseModel):
    atypical_pigment_network: EvidenceEnvelope
    blue_whitish_veil: EvidenceEnvelope
    irregular_streaks: EvidenceEnvelope
    irregular_pigmentation: EvidenceEnvelope
    irregular_dots_globules: EvidenceEnvelope


class MenziesFeaturePackage(BaseModel):
    broadened_network: EvidenceEnvelope
    blue_white_veil: EvidenceEnvelope
    blue_gray_dots: EvidenceEnvelope
    peripheral_black_dots_globules: EvidenceEnvelope
    radial_streaming: EvidenceEnvelope
    pseudopods: EvidenceEnvelope
    color_evidence: EvidenceEnvelope


class SharedEvidencePackage(BaseModel):
    pigment_network: EvidenceEnvelope
    blue_white_veil: EvidenceEnvelope
    structureless_blotch: EvidenceEnvelope
    unavailable_structures: Dict[str, EvidenceEnvelope] = Field(default_factory=dict)


class FeatureExtractionResult(BaseModel):
    schema_version: str = "fe_v1"
    image_id: str
    image_path: str
    mask_path: Optional[str] = None
    experiment_condition: str
    abcd: ABCDFeaturePackage
    seven_point: SevenPointFeaturePackage
    menzies: MenziesFeaturePackage
    shared: SharedEvidencePackage
    metadata: Dict[str, Any] = Field(default_factory=dict)

