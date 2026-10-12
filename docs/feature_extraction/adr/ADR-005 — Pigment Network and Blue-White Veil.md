# ADR-005 — Pigment Network and Blue-White Veil

**Status:** Accepted  
**Date:** 2026-09-08

**Supersession notice:** Current implementation readiness is governed by ADR-009 and ADR-010. The blue-white veil scientific approval below authorizes the shared Celebi pixel-level detector and raw veil evidence only after ADR-010 confirmation; it must not be interpreted as authorization to emit deterministic lesion-level binary 7PCL-r or Menzies-r blue-white-veil values. Those clinical mappings remain `UNRESOLVED`.

## Pigment Network

### Decision

Adopt Sadeghi et al. (2011) as the preferred low-level pigment-network operationalization because its graph/cyclic-hole representation is semantically aligned with network morphology and supports reusable structural measurements.

`NETWORK-DETECT-001 = HUMAN_APPROVED`

Implementation is blocked until the complete primary source provides sufficient methodological detail to reproduce all required equations, graph construction rules, parameters and thresholds.

Anantha, Moss and Stoecker (2004) remains a secondary reproducible alternative, but its fixed block/pixel parameters create transferability concerns across image scales.

### Algorithm mappings

**ABCD pigment-network presence:** approved conceptually, conditional on reproducible implementation of the published detector.

**7PCL atypical pigment network:** `UNRESOLVED`. The literature supports the clinical definition but no approved deterministic mapping from raw network morphology to typical/atypical has been identified.

**Menzies broadened network:** `UNRESOLVED`. Focal broadening is clinically defined, but no approved numerical width threshold has been identified.

Raw network measurements are approved for VLM evidence.

No PH² calibration is allowed.

PH² typical/atypical pigment-network annotations may be used for post-hoc validation only.

---

## Blue-White Veil

### Decision

Adopt Celebi et al. (2008) as the shared blue-white veil detector.

`VEIL-DETECT-001 = HUMAN_APPROVED`

The published C4.5 classifier and its parameters may be reproduced exactly.

Classifier-derived thresholds are explicitly classified as `CLASSIFIER_DERIVED`/published computational parameters and must not be described as universal clinical thresholds.

The complete pixel-level decision-tree splits and any required surrounding-skin-annulus parameters must be recovered from the primary source before implementation. Missing values must not be inferred.

### Algorithm mappings

**7PCL blue-whitish veil:** `HUMAN_APPROVED`.

**Menzies blue-white veil:** `HUMAN_APPROVED`.

The same low-level detector may serve both criteria because the reviewed clinical definitions are compatible.

**ABCD:** blue-white veil must not independently increment ABCD D. It may be retained as auxiliary raw evidence relevant to color/structure.

PH² veil annotations may be used for validation only.
