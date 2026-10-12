# Research Sprint Review

## Audit Scope

This review checks whether the proposed recommendations are supported by the cited sources, whether segmentation was confused with FE, whether classifier-derived features were overinterpreted as dermatological definitions, and whether thresholds were treated too broadly.

Post-reproducibility checkpoints: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md` are the active source of truth for implementation readiness. Scientific approval, reproducibility status, and implementation readiness are separate.

## Recommendation Audit

| Recommendation | Reviewer Status | Issue |
|---|---|---|
| ABCD A via center of gravity, moment principal axis, CIELAB block color symmetry, luminance symmetry, and shape area symmetry from Kasmi and Mokrani (2016) | `HUMAN_APPROVED` with `PARTIALLY_SUPPORTED` evidence | Approved for implementation with published parameters and no PH2 calibration. Thresholds are EDRA-tested computational thresholds, not universal clinical thresholds. PH2 asymmetry annotations are validation ground truth only. |
| ABCD color count using six PH2/ABCD colors | `SUPPORTED` for clinical color set; `PARTIALLY_SUPPORTED` for implementation | ABCD six-color rule and PH2 color annotations are well supported. Exact RGB thresholds and minimum area must be verified/approved. |
| ABCD C using Majumder and Ullah (2019) normalized RGB table and 5% color-presence rule | `HUMAN_APPROVED` with `PARTIALLY_SUPPORTED` evidence | Approved for implementation with published parameters and no PH2 calibration. Thresholds are computational/dataset-derived, not universal clinical thresholds. PH2 color annotations are validation ground truth only. |
| ABCD B source-guided author operationalization based on Kasmi and Mokrani (2016) | `HUMAN_APPROVED_WITH_LIMITATION`; exact source implementation `UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`; author operationalization implemented | Source supports eight 45-degree sectors, cubic spline comparison, 5% threshold form, and `B=0..8`. The executable `length(y)` convention is under-specified, so the implementation uses principal-axis-aligned sectors, ordered parametric contour, dimensionless smoothing, Euclidean RMSE, `L_smooth`, and `E_smooth > 0.05` as documented author conventions. |
| ABCD strict abrupt pigment cutoff in eight sectors | `NEEDS_MORE_RESEARCH` | Kaya et al. (2016) and Erol et al. (2017) quantify abruptness from inner-border homogeneity, but they do not output eight ABCD sector scores. |
| ABCD D as differential structures, not diameter | `SUPPORTED` | Original dermoscopic ABCD sources support D as differential structures. |
| Complete ABCD D score from deterministic extractors | Historical `PENDING_SHARED_EXTRACTOR_RESEARCH`; active state `ABCD_D = null` / `STILL_UNRESOLVED` | Shared-extractor research has been completed for this dissertation. Do not approve or implement final ABCD D; no further shared-extractor research is planned within this dissertation. |
| Shared pigment-network low-level extractor (`NETWORK-DETECT-001`) | Sadeghi scientific preference `HUMAN_APPROVED`; Sadeghi implementation `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; Anantha fallback `RAW_EVIDENCE` | Anantha et al. (2004) is accepted as reproducible raw evidence only. It must not be described as reproducing Sadeghi, resolving atypical/broadened network, or deriving ABCD pigment-network presence from positive blocks. |
| 7PCL atypical pigment network interpretation (`7PCL-NETWORK-ATYPICAL-001`) | `STILL_UNRESOLVED`; clinical decision `UNAVAILABLE` | Argenziano et al. (1998) and Dermoscopedia define prominent/broad irregular network, but no deterministic thresholded mapping from raw network features was identified. |
| Menzies broadened network interpretation (`MENZIES-NETWORK-BROADENED-001`) | `STILL_UNRESOLVED`; clinical decision `UNAVAILABLE` | Braun and Menzies define focal cord/grid broadening, but a numeric width threshold is missing. It must not be treated as identical to 7PCL atypical network. |
| Shared blue-white veil low-level extractor (`VEIL-DETECT-001`) | `HUMAN_APPROVED` scientifically; pixel detector `IMPLEMENTATION_READY`; `S1`/veil-area fraction `RAW_EVIDENCE` | Celebi et al. (2008) remains approved. Figure 5 pixel tree was transcribed, independently verified from 300 dpi and 600 dpi renders, and human-confirmed for implementation. The lesion-level S1/S3 diagnostic tree is not a 7PCL-r or Menzies-r veil-presence mapping. |
| Shared peripheral radial-structure extractor (`STREAK-DETECT-001`) | `HUMAN_APPROVED` scientifically; `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | Sadeghi et al. (2013) remains selected, but missing LoG scales, angular thresholds, residual thresholds, and classifier artifacts block faithful implementation. |
| 7PCL irregular streaks interpretation | `HUMAN_APPROVED_WITH_LIMITATION` scientifically; not implementation-ready | PH2 labels only streaks present/absent, and Sadeghi implementation gaps block deterministic output. |
| Menzies radial streaming and pseudopods interpretations | `STILL_UNRESOLVED`; `UNAVAILABLE` | Dermoscopedia distinguishes linear streaming from bulbous pseudopods, but no numeric terminal morphology or angular distribution threshold was identified. |
| Shared dots/globules low-level extractor (`BLOB-DETECT-001`) | scientific reference selected; `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | Maglogiannis and Delibasis (2015) lacks recovered reproducible method details. No LoG/DoG/OpenCV substitute or physical-to-pixel inference is authorized. |
| 7PCL irregular dots/globules interpretation | `STILL_UNRESOLVED`; `UNAVAILABLE` | Argenziano et al. (1998) define irregular distribution, but no deterministic threshold was identified. |
| Menzies blue-gray dots and peripheral black dots/globules interpretations | `STILL_UNRESOLVED`; `UNAVAILABLE` | Braun and Menzies define color/spatial constraints, but PH2 has only partial validation labels and no separate subtype annotations. |
| Shared structureless/blotch/irregular-pigmentation extractor (`STRUCTURELESS-BLOTCH-001`) | `HUMAN_APPROVED`; absolute-color `RAW_EVIDENCE`; ABCD structureless-area path available with limitation | Stoecker et al. (2005) absolute-color path is implementation-ready: `R < 120 AND G < 60`, followed by `A_blotch > 800 pixels` unchanged on original-resolution PH2 as a source-scale-specific published computational threshold. Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. 7PCL irregular pigmentation and Menzies color non-uniformity remain unresolved/auxiliary. |
| Menzies pattern asymmetry reusing ABCD asymmetry | `PARTIALLY_SUPPORTED` | Reuse is acceptable only as partial evidence; Menzies pattern symmetry includes colors/structures. |
| Menzies single color / 5-6 colors using shared color extractor | `PARTIALLY_SUPPORTED` | Menzies color count is clinically supported, but color set and minimum area differ from ABCD. |
| Menzies blue-white veil sharing with 7PCL | `SUPPORTED` for shared raw detector; binary mapping `STILL_UNRESOLVED` | Definitions are compatible enough for shared Celebi pixel evidence. No lesion-level checklist binary mapping is approved. |
| Menzies broadened network sharing with 7PCL atypical network | `PARTIALLY_SUPPORTED` | Shared low-level network detector is justified; broadened network is not identical to atypical pigment network. |
| Menzies radial streaming/pseudopods sharing with 7PCL streaks | `PARTIALLY_SUPPORTED` | Shared detector family is justified; subtype decisions need geometry thresholds. |
| Blanket DullRazor preprocessing | `REJECTED` | Hair removal can destroy or mask pigment network/streak-like features; PH2 does not justify universal hair removal. |
| Selective artifact masks/QC | `SUPPORTED` | Conservative exclusion is compatible with FE and avoids altering diagnostic structures. |
| Global color constancy for PH2-only | `REJECTED` as required preprocessing | PH2 controlled acquisition weakens necessity; applying normalization without retuned thresholds risks color-feature drift. |

## Cross-Checks

Segmentation was not counted as FE. The future stage should use PH2 GT masks only.

Classifier-derived or dataset-derived thresholds were not promoted to universal dermatological thresholds. Kasmi and Mokrani (2016) thresholds for A and the 5% threshold form for B, plus Majumder and Ullah (2019) RGB thresholds for C, are human-approved published computational parameters adopted without PH2 calibration. For ABCD-B, exact executable source reproduction remains unavailable because `length(y)` is under-specified; the implemented method is an author operationalization with synthetic geometric validation. Structure-extractor readiness is governed by ADR-009 and remains classified separately from scientific selection.

No recommendation depends on a new supervised model unless explicitly marked for human approval. Raw feature extractors are preferred over final binary decisions where evidence is incomplete.

ADR-010 accepts the raw-only evidence package as sufficient for the planned dissertation experiment. Complete deterministic 7PCL-r and Menzies-r FE remain unavailable by design.

PH2 structure annotations were treated only as validation ground truth. The reviewed specification does not use PH2 pigment network, blue-whitish veil, streaks, dots/globules, color, regression, asymmetry, or diagnosis annotations as Feature Extraction inputs or calibration targets.

Retraining unpublished classifiers is rejected. Weka SimpleLogistic, C4.5, neural networks, or other classifier models must not be retrained to replace missing source artifacts. Such retraining would be a new methodological method.

## Rejected Recommendations

- Blanket DullRazor/hair inpainting before all extractors.
- Treating compactness/fractal dimension alone as the ABCD border score.
- Replacing ABCD `D` with diameter.
- Treating `shape OR color` per asymmetry axis as validated without evidence.
- Treating global color normalization as mandatory for PH2-only FE.
- Using PH2 expert color annotations as ABCD C feature inputs. They may be used as validation ground truth only.
- Using PH2 expert structure annotations as inputs to pigment-network, blue-white veil, streak, dots/globules, or structureless/blotch extractors.
- Sharing final binary thresholds across ABCD D, 7PCL, and Menzies merely because their low-level detectors can share raw measurements.
- Emitting `RAW_EVIDENCE` from an irreproducible approximation.
- Retraining unpublished classifier models to replace missing model artifacts or training data.
- Rescaling Stoecker `A_blotch > 800 pixels` to PH2 or inferring physical equivalence.
- Inventing Stoecker relative-color background-skin sampling extent or substituting Celebi's skin annulus.

## Method Searches Out of Scope for This Dissertation

ADR-010 records the human decision that no new literature or method-selection stage will be conducted within this dissertation for streaks, branched streaks, radial streaming, pseudopods, dots, globules, irregular dots/globules, irregular pigmentation, broadened network, or atypical pigment network. Their unresolved or unavailable states are accepted limitations unless a future explicit human decision expands scope.

## Sources Audited

See source inventories in `abcd.md`, `seven_point.md`, `menzies.md`, `preprocessing.md`, and `shared_extractors.md`.
