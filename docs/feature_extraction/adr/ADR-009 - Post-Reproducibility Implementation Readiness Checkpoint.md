# ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint

**Status:** Accepted  
**Date:** 2026-09-09  
**Scope:** Feature Extraction implementation readiness after primary-source reproducibility resolution

**Supersession note:** ADR-010 records the final human confirmation after Celebi Figure 5 recovery and the final Stoecker implementation scope. Where this ADR says Celebi implementation readiness required human confirmation, ADR-010 updates the Celebi pixel-level detector to `IMPLEMENTATION_READY` while keeping 7PCL-r and Menzies-r lesion-level blue-white-veil binary mappings unresolved. Where this ADR broadly describes Stoecker raw blotch segmentation/features as implementation-ready, ADR-010 narrows that readiness to the absolute-color path `R < 120 AND G < 60` followed by `A_blotch > 800 pixels`; Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. ADR-011 governs the final ABCD-B executable formulation: exact source implementation remains unavailable due to the `length(y)` gap, while the dissertation author operationalization is implemented with principal-axis-aligned sectors, dimensionless smoothing, Euclidean RMSE, `L_smooth`, and `E_smooth > 0.05`.

## Context

The primary-source reproducibility stage separated scientific method selection from the ability to faithfully implement a selected method. Several methods remain scientifically selected by prior ADRs, but their available primary-source materials do not provide all equations, parameters, thresholds, classifier artifacts, or figure values required for implementation without invention.

This ADR supersedes earlier `PENDING_HUMAN_APPROVAL`, `READY_FOR_APPROVAL`, and implementation-readiness language only where it conflicts with the decisions below. It does not reject the prior scientific-method selections.

## Decision

Scientific approval, reproducibility status, and implementation readiness are distinct:

- `methodological/scientific selection`: the method is selected as the dissertation's scientifically preferred or accepted source.
- `reproducibility status`: primary-source details are sufficient, partially sufficient, or insufficient for faithful reproduction.
- `implementation readiness`: the method may be implemented without inventing scientific methodology.

A scientifically selected method whose implementation would require inventing missing scientific details must be marked:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

Earlier `HUMAN_APPROVED` decisions do not authorize inference of missing formulas, thresholds, preprocessing steps, classifier splits, classifier weights, scale conversions, or clinical mappings.

Retraining unpublished classifiers is rejected:

`RETRAIN_UNPUBLISHED_CLASSIFIER = REJECTED`

Weka SimpleLogistic, C4.5, neural networks, or other published classifiers must not be retrained merely because original model artifacts, training data, folds, weights, or complete training configuration are unavailable. Training a new classifier on another dataset is a new method, not reproduction. PH2 must never be used to train, recalibrate, or recover these classifiers.

## RAW_EVIDENCE Eligibility

`RAW_EVIDENCE` is allowed only when the mechanism producing the raw measurement is itself reproducible and scientifically approved.

The active state model is:

- `DETERMINISTIC_CLINICAL`: the extractor is reproducible and an approved mapping exists from extractor output to the algorithm-specific clinical criterion.
- `DETERMINISTIC_CLINICAL_WITH_LIMITATION`: the extractor is reproducible and approved, but reporting must preserve a specified methodological limitation.
- `RAW_EVIDENCE`: the extractor itself is reproducible and scientifically approved, but the final algorithm-specific clinical mapping is unresolved.
- `UNAVAILABLE`: the required extractor cannot be faithfully reproduced, the image does not permit the measurement, or required information is unavailable.
- `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`: the method is scientifically selected, but primary-source details are insufficient for implementation without invention.

An irreproducible approximation must not be classified as `RAW_EVIDENCE`. `null` must never mean clinical absence.

## Method Decisions

### Pigment Network

Scientific preference remains Sadeghi et al. (2011), but the selected Sadeghi implementation is:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

Anantha, Moss and Stoecker (2004) is accepted as the practical reproducible fallback for pigment-network evidence.

Fallback role:

`RAW_EVIDENCE`

The Anantha fallback must not be described as reproducing Sadeghi, resolving atypical pigment network, resolving broadened network, or deriving an ABCD pigment-network presence rule from positive blocks.

Still unresolved:

- ABCD D pigment-network presence;
- 7PCL-r atypical pigment network;
- Menzies-r broadened network.

No PH2 calibration is authorized.

### Blue-White Veil

Celebi et al. (2008) remains the approved scientific method.

Implementation status at the ADR-009 checkpoint:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

One final focused reproducibility action is authorized: manually inspect the original/full-text Celebi et al. (2008) PDF Figure 5 and attempt to transcribe the complete pixel-level C4.5 decision tree. Every split and threshold must be visibly legible and independently verified a second time. Ambiguous values remain `UNRESOLVED_REPRODUCIBILITY_GAP`.

Do not retrain C4.5, derive replacement thresholds, or calibrate on PH2.

### Streaks / Radial Structures

Sadeghi et al. (2013) remains scientifically selected, but its implementation status is:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

Do not reconstruct missing LoG scales, invent angular thresholds, invent residual thresholds, retrain SimpleLogistic, approximate the detector and call it Sadeghi, or create a new streak detector merely to complete the feature set.

Until new explicit human approval, no deterministic streak extractor is implementation-ready. 7PCL irregular streaks are not implementation-ready despite prior methodological approval with limitation. ABCD branched streaks, Menzies radial streaming, and Menzies pseudopods remain unresolved.

### Dots / Globules

Maglogiannis and Delibasis (2015) remains the selected scientific reference, but its implementation status is:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

Do not reconstruct the inverse nonlinear diffusion method, substitute LoG/DoG/OpenCV blob detection without a new methodological decision, invent size thresholds, infer physical-to-pixel conversion, assume a pixel equivalent for the clinical approximately 0.1 mm distinction, or calibrate using PH2.

Dots/globules are currently `UNAVAILABLE` as implemented deterministic or raw FE unless a separately reproducible approved extractor is later selected.

### Structureless Areas / Blotches

Stoecker et al. (2005) is approved for implementation of reproducible raw blotch segmentation and features.

Use the published source-scale threshold unchanged on original-resolution PH2 images:

`A_blotch > 800 pixels`

Do not rescale it, derive a PH2-specific threshold, calibrate it, resize PH2 solely to emulate the source dataset, or infer physical equivalence.

Classify this parameter as a `PUBLISHED_COMPUTATIONAL`, source-scale-specific threshold. Reporting must state that physical equivalence on PH2 is not guaranteed because the threshold was developed for the source image scale/resolution.

The approximately 10% lesion-area clinical rule remains separate from the computational 800-pixel candidate/component rule. The original diagnostic neural network does not need to be reproduced and must not be retrained.

### ABCD D

The complete ABCD D remains unresolved:

`ABCD_D = null`

Do not calculate a partial D and present it as complete ABCD D.

Current component readiness:

- pigment network: `RAW_EVIDENCE` through Anantha fallback; final ABCD mapping unresolved;
- structureless area: reproducible/approved path available with documented limitations;
- branched streaks: `UNAVAILABLE` / unresolved;
- dots: `UNAVAILABLE` / unresolved;
- globules: `UNAVAILABLE` / unresolved.

## Historical ADR-009 Implementation-Readiness Snapshot

This snapshot records the state before final human confirmation of the recovered Celebi Figure 5 tree. ADR-010 is authoritative for the final state before coding.

| Feature | ADR-009 checkpoint state |
|---|---|
| ABCD A | `DETERMINISTIC_CLINICAL` |
| ABCD B | `DETERMINISTIC_CLINICAL_WITH_LIMITATION` |
| ABCD C | `DETERMINISTIC_CLINICAL` |
| Pigment network | `RAW_EVIDENCE` through Anantha fallback |
| Blue-white veil | `READY_FOR_HUMAN_CONFIRMATION` after Celebi Figure 5 extraction; no code until human confirmation |
| Streaks | `UNAVAILABLE` / implementation blocked |
| Dots/globules | `UNAVAILABLE` / implementation blocked |
| Structureless/blotch | `RAW_EVIDENCE` plus approved ABCD structureless-area use with documented limitations |
| Complete ABCD D | `null` / unresolved |
| Complete deterministic 7PCL-r FE | `UNAVAILABLE` |
| Complete deterministic Menzies-r FE | `UNAVAILABLE` |

## Experimental Interpretation

The dissertation extension must not be described as a complete deterministic computational implementation of ABCD, 7PCL and Menzies.

Preferred framing:

**deterministic extraction of dermoscopic evidence grounded in ABCD, Seven-Point Checklist and Menzies criteria for VLM grounding**

The FE package may be incomplete by design where the literature does not support reproducible deterministic operationalization.

The three experimental conditions remain:

1. image only;
2. image + FE;
3. FE only.

For FE-only, the VLM receives only scientifically supported structured evidence that is actually available. Missing or unresolved criteria must not be synthesized or guessed. For image + FE, FE remains auxiliary evidence and the VLM may disagree with it.

## Addendum: Celebi Figure 5 Extraction Result

The authorized Figure 5 extraction was completed from the Celebi et al. (2008) arXiv author manuscript PDF. The figure was inspected visually from a 300 dpi page render and independently rechecked from a 600 dpi page render.

The pixel-level C4.5 tree in Figure 5 is transcribed as:

1. `F3 <= 0.3 ?`
2. yes: `Non-Veil`
3. no: `F10 > -51 ?`
4. yes: `Non-Veil`
5. no: `F10 <= -194 ?`
6. yes: `Non-Veil`
7. no: `Veil`

Equivalent positive veil interval:

`F3 > 0.3 AND -194 < F10 <= -51`

This is a transcription/restatement of the primary-source figure, not a newly derived threshold. At the ADR-009 checkpoint, the Celebi pixel-level detector was `RESOLVED_BY_PRIMARY_SOURCE` and still required human confirmation before coding. ADR-010 records that final human confirmation.
