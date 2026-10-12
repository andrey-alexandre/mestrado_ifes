# Reproducibility Matrix

## Post-Reproducibility Checkpoint

Authoritative decisions: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md`, `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md`, and `docs/feature_extraction/adr/ADR-011 - ABCD B Author Operationalization Closure.md`.

Scientific method selection, reproducibility status, and implementation readiness are separate. `RAW_EVIDENCE` is allowed only when the mechanism producing the raw measurement is reproducible and scientifically approved. Irreproducible approximations must be `UNAVAILABLE` or `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

## Cross-Method Readiness

| Method | Primary source full text recovered? | Reproducibility verdict | Implementation readiness without invention | Main blocking issue |
|---|---:|---|---|---|
| Sadeghi et al. 2011 pigment network graph/cyclic holes | No journal full text; same-author chapter recovered | `NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE` | `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | Classifier model/decision boundary for 94.3% result and several low-level thresholds are missing. |
| Anantha, Moss and Stoecker 2004 fallback pigment network | Yes, PMC manuscript recovered | `REPRODUCIBLE_WITH_EXPLICIT_GAPS` | `RAW_EVIDENCE` fallback authorized by ADR-009 | Detects gross block texture, not Sadeghi graph holes or atypical/broadened network; no ABCD pigment-network presence rule may be derived from positive blocks. |
| Celebi et al. 2008 blue-white veil | Yes, author manuscript recovered and Figure 5 visually transcribed | `REPRODUCIBLE_WITH_EXPLICIT_GAPS` | Pixel detector `IMPLEMENTATION_READY`; `veil_mask` and `S1` are implementation-ready `RAW_EVIDENCE` | Pixel-level C4.5 tree recovered and human-confirmed from Figure 5; retraining remains forbidden and unpublished training pixels remain unavailable. Lesion-level S1/S3 diagnostic tree is not a 7PCL/Menzies veil-presence mapping. |
| Sadeghi et al. 2013 streaks | Partial full-text cache recovered | `NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE` | `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | LoG scale values, valid-streak angular threshold, several equations, and trained classifier artifacts are missing. |
| Maglogiannis and Delibasis 2015 dots/globules | No | `NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE` | `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | Full inverse nonlinear diffusion method, thresholds, dot/globule separation, and feature vector are missing. |
| Stoecker et al. 2005 blotches | Yes, PMC manuscript recovered | `REPRODUCIBLE_WITH_EXPLICIT_GAPS` | absolute-color path `RAW_EVIDENCE`; approved ABCD structureless-area use with documented limitations | `R < 120 AND G < 60` and `A_blotch > 800 pixels` are implementation-ready unchanged on original-resolution PH2. Relative-color path is blocked because the background-skin sampling outer extent is unspecified. |

## Final Implementation-Readiness Snapshot Before Coding

| Feature | Final state before coding |
|---|---|
| ABCD A | `DETERMINISTIC_CLINICAL` |
| ABCD B | `DETERMINISTIC_CLINICAL_WITH_LIMITATION`; exact source implementation `UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`; author operationalization implemented and documented in `docs/abcd_border_methodology.md` |
| ABCD C | `DETERMINISTIC_CLINICAL` |
| Pigment network | `RAW_EVIDENCE` through Anantha fallback |
| Blue-white veil pixel mask | reproducible / `IMPLEMENTATION_READY` |
| Blue-white veil area fraction | `RAW_EVIDENCE` / `IMPLEMENTATION_READY` |
| 7PCL blue-whitish veil binary | `UNRESOLVED` |
| Menzies blue-white veil binary | `UNRESOLVED` |
| Streaks | `UNAVAILABLE` / implementation blocked |
| Dots/globules | `UNAVAILABLE` / implementation blocked |
| Structureless/blotch | absolute-color `RAW_EVIDENCE` plus approved ABCD structureless-area use with documented limitations / `IMPLEMENTATION_READY` |
| Complete ABCD D | `null` / unresolved |
| Complete deterministic 7PCL-r FE | `UNAVAILABLE` |
| Complete deterministic Menzies-r FE | `UNAVAILABLE` |

## Algorithm-Specific Resolution Matrix

| Criterion | Resolution |
|---|---|
| ABCD D pigment network presence | `STILL_UNRESOLVED` |
| ABCD D structureless areas | `RESOLVED_BY_PRIMARY_SOURCE` for Stoecker absolute-color raw blotch evidence plus Dermoscopedia 10% rule; use `R < 120 AND G < 60` and `A_blotch > 800 pixels` unchanged with source-scale limitation; full ABCD D remains `null` / `STILL_UNRESOLVED`. |
| ABCD D branched streaks | `STILL_UNRESOLVED` |
| ABCD D dots | `STILL_UNRESOLVED` |
| ABCD D globules | `STILL_UNRESOLVED` |
| Complete `D in {0,...,5}` | `STILL_UNRESOLVED`; cannot be computed without invention. |
| 7PCL-r atypical pigment network | `STILL_UNRESOLVED` |
| 7PCL-r blue-whitish veil | Pixel detector `IMPLEMENTATION_READY`; `veil_mask` and `S1` are `RAW_EVIDENCE`; binary clinical criterion `STILL_UNRESOLVED`. |
| 7PCL-r irregular streaks | `STILL_UNRESOLVED` for exact implementation despite ADR approval with limitation. |
| 7PCL-r irregular pigmentation | `STILL_UNRESOLVED` |
| 7PCL-r irregular dots/globules | `STILL_UNRESOLVED` |
| Menzies-r broadened network | `STILL_UNRESOLVED` |
| Menzies-r blue-white veil | Pixel detector `IMPLEMENTATION_READY`; `veil_mask` and `S1` are `RAW_EVIDENCE`; binary clinical criterion `STILL_UNRESOLVED`. |
| Menzies-r blue-gray dots | `STILL_UNRESOLVED` |
| Menzies-r peripheral black dots/globules | `STILL_UNRESOLVED` |
| Menzies-r radial streaming | `STILL_UNRESOLVED` |
| Menzies-r pseudopods | `STILL_UNRESOLVED` |

## PH2 Validation Only

PH2 expert annotations remain validation-only. No row above authorizes PH2-derived calibration, classifier training, threshold selection, model selection, filter tuning, or physical-to-pixel conversion.
