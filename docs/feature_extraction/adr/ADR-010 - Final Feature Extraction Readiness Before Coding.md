# ADR-010 - Final Feature Extraction Readiness Before Coding

**Status:** Accepted  
**Date:** 2026-09-09  
**Scope:** Final human confirmation after Celebi Figure 5 recovery and before Feature Extraction coding

**Supersession note for ABCD-B:** ADR-011 records the post-experimental ABCD-B closure. The final ABCD-B extractor remains `DETERMINISTIC_CLINICAL_WITH_LIMITATION` at runtime, but exact Kasmi and Mokrani source reproduction is `UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`; the implemented path is the documented author operationalization.

## Context

The scientific selection and primary-source reproducibility stages are complete for the dissertation Feature Extraction package. ADR-009 recorded the post-reproducibility checkpoint and the recovered Celebi et al. (2008) Figure 5 pixel-level C4.5 tree, but left Celebi implementation readiness pending human confirmation.

This ADR is the final scientific documentation correction before coding. It preserves the chronology:

1. initial scientific selection;
2. reproducibility investigation;
3. ADR-009 post-reproducibility checkpoint;
4. Celebi Figure 5 recovery;
5. final human confirmation.

## Decision

The recovered Celebi et al. (2008) Figure 5 pixel-level blue-white veil detector is sufficiently reproducible and is approved for implementation.

The following Celebi pixel-level components are:

`IMPLEMENTATION_READY`

- `F3` computation;
- `F10` computation;
- Figure 5 C4.5 pixel decision tree;
- 5 x 5 majority-filter postprocessing;
- resulting pixel-level `veil_mask`;
- `S1` / `veil_area_fraction`.

The recovered published pixel rule is:

`F3 > 0.3 AND -194 < F10 <= -51`

This rule is explicitly traceable to Celebi et al. (2008), Figure 5. It is a classifier-derived published computational rule, not a universal clinical threshold. Do not retrain C4.5, modify the thresholds, recalibrate the thresholds on PH2, or use PH2 annotations as Feature Extraction inputs.

## Lesion-Level Interpretation Correction

The primary source resolves the pixel-level blue-white veil detector, but it does not establish a reproduced deterministic lesion-level presence/absence mapping for the 7PCL or Menzies clinical blue-white veil criterion.

The recovered Celebi lesion-level tree:

- `S1 < 0.009 -> benign`;
- otherwise `S3 > 0.979 -> benign`;
- otherwise melanoma.

This is a melanoma/benign diagnostic classifier. It is not a threshold defining whether blue-white veil is clinically present in a lesion.

Therefore:

- do not use `S1 = 0.009` as a blue-white-veil clinical presence threshold;
- do not interpret the Celebi lesion diagnostic tree as the 7PCL-r or Menzies-r blue-white-veil criterion;
- do not create `S1 > 0 -> veil_present = true` without a separately approved source;
- do not invent a new aggregation threshold.

## Final Blue-White Veil State

| Output or criterion | Final state before coding |
|---|---|
| Celebi pixel-level detector | `IMPLEMENTATION_READY` |
| `veil_mask` | Reproducible deterministic computational output / implementation ready |
| `veil_area_fraction` / `S1` | `RAW_EVIDENCE` / implementation ready |
| 7PCL-r blue-whitish veil binary criterion | `STILL_UNRESOLVED` |
| Menzies-r blue-white veil binary criterion | `STILL_UNRESOLVED` |

The VLM may receive reproducible Celebi veil evidence, but Feature Extraction must not fabricate final binary checklist values for 7PCL-r or Menzies-r.

## Scope Closure

No new literature or method-selection stage will be conducted for the currently unresolved extractors within this dissertation. Do not search for or substitute new methods for:

- streaks / branched streaks;
- radial streaming;
- pseudopods;
- dots;
- globules;
- irregular dots/globules;
- irregular pigmentation;
- broadened network;
- atypical pigment network.

Their unresolved or unavailable states are accepted limitations of the dissertation Feature Extraction extension. New methods would constitute a scope expansion and require a new explicit human decision.

## Approved Experiment Package

The scientifically reproducible Feature Extraction package is sufficient to proceed with the dissertation experiment even though deterministic 7PCL-r and Menzies-r are incomplete.

### Final Stoecker Blotch Scope

For this dissertation, the deterministic implementation-ready Stoecker blotch segmentation path is the fully specified absolute-color method:

`R < 120 AND G < 60`

followed by the published minimum candidate/component rule:

`A_blotch > 800 pixels`

This threshold must be used unchanged on original-resolution PH2 images, with the documented source-scale limitation. Do not rescale it, calibrate it on PH2, resize PH2 to emulate the source dataset, or infer physical equivalence.

The Stoecker relative-color segmentation remains scientifically documented but is:

`IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`

because the primary source does not fully specify the outer extent of the background-skin sampling region. Do not invent that outer extent and do not silently substitute Celebi's skin-annulus construction into Stoecker.

Only outputs that can be produced without an unresolved implementation convention may be marked implementation-ready. If a raw Stoecker feature depends on an unspecified connected-component connectivity convention, perimeter estimator, or other implementation-sensitive operation, the implementation must identify the unresolved convention before coding rather than silently choosing a library default.

Implementation scope may include:

- deterministic clinical ABCD A;
- deterministic clinical ABCD B, with the approved limitation;
- deterministic clinical ABCD C;
- pigment-network raw evidence through the approved Anantha fallback;
- blue-white-veil mask and quantitative raw evidence through Celebi et al. (2008);
- structureless/blotch raw evidence through Stoecker et al. (2005) absolute-color path, plus approved ABCD structureless-area use with limitations.

Unavailable:

- streak-based evidence from the selected Sadeghi method;
- dots/globules evidence from the selected Maglogiannis method;
- unresolved algorithm-specific clinical mappings.

Complete ABCD D remains:

`null`

Complete deterministic 7PCL-r FE remains unavailable. Complete deterministic Menzies-r FE remains unavailable. This incompleteness is intentional and scientifically documented.

## Output Semantics

Runtime Feature Extraction outputs must preserve these state semantics:

- `DETERMINISTIC_CLINICAL`;
- `DETERMINISTIC_CLINICAL_WITH_LIMITATION`;
- `RAW_EVIDENCE`;
- `UNAVAILABLE`.

Where useful internally, `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` may remain a documentation/method status. Runtime outputs must distinguish evidence availability from clinical absence. A missing or unresolved criterion must never become `false` or `0` merely because it could not be computed. Use `null` with the appropriate status. Clinical absence is a valid computed result and must remain distinguishable from unavailable evidence.

## Experimental Scope

The three experimental conditions remain:

1. image only;
2. image + FE;
3. FE only.

In image + FE, deterministic clinical outputs and raw evidence are auxiliary evidence; the VLM may disagree with them. Raw evidence must not be presented as an established checklist decision.

In FE only, provide only evidence actually produced by reproducible approved extractors. Unavailable criteria remain explicitly unavailable. Do not synthesize missing information.

Scientific framing:

**deterministic extraction of dermoscopic evidence grounded in ABCD, Seven-Point Checklist and Menzies criteria for VLM grounding**

Do not describe the extension as a complete computational implementation of all three clinical algorithms.

## Final Readiness Snapshot

| Feature | Final state before coding |
|---|---|
| ABCD A | `DETERMINISTIC_CLINICAL` / implementation ready |
| ABCD B | `DETERMINISTIC_CLINICAL_WITH_LIMITATION`; exact source implementation `UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`; author operationalization implemented |
| ABCD C | `DETERMINISTIC_CLINICAL` / implementation ready |
| Pigment network | `RAW_EVIDENCE` via Anantha / implementation ready |
| Blue-white veil pixel mask | reproducible / implementation ready |
| Blue-white veil area fraction | `RAW_EVIDENCE` / implementation ready |
| 7PCL blue-whitish veil binary | `UNRESOLVED` |
| Menzies blue-white veil binary | `UNRESOLVED` |
| Streaks | `UNAVAILABLE` |
| Dots/globules | `UNAVAILABLE` |
| Structureless/blotch | absolute-color `RAW_EVIDENCE` plus approved ABCD use with limitations / implementation ready |
| Complete ABCD D | `null` / unresolved |
| Complete deterministic 7PCL-r FE | unavailable |
| Complete deterministic Menzies-r FE | unavailable |

## References

- Celebi ME, Iyatomi H, Stoecker WV, Moss RH, Rabinovitz HS, Argenziano G, Soyer HP. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." Computerized Medical Imaging and Graphics. 2008;32(8):670-677. DOI `10.1016/j.compmedimag.2008.08.003`; PMID `18804955`.
- Internal ADR-009: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md`.
