# Feature Extraction Scientific Specification

## Project Constraints

The future FE stage should run before VLM calls, use PH2 only, use the original RGB image and PH2 ground-truth lesion mask, preserve sequential execution, expose parameters through `config.yaml`, return raw values and final candidate scores, use Pydantic schemas, return `null` when not calculable, and generate debug visualizations. The current ABCD, Seven-Point Checklist, Menzies, summarizer, result JSON flow, and evaluation flow should remain minimally changed.

The existing evaluation computes accuracy, sensitivity, specificity, TP, TN, FP, and FN from `validation.diagnostico`; the future experiment will add F1 as the primary metric and persist predictions for later bootstrap/statistical analysis.

## Active Post-Reproducibility Readiness

Authoritative checkpoints: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md`.

Scientific method selection, reproducibility status, and implementation readiness are separate. A method can remain scientifically selected while implementation is blocked because reproducing it would require inventing missing source details.

State model:

- `DETERMINISTIC_CLINICAL`: reproducible extractor plus approved algorithm-specific clinical mapping.
- `DETERMINISTIC_CLINICAL_WITH_LIMITATION`: reproducible and approved, with mandatory methodological limitation in reporting.
- `RAW_EVIDENCE`: reproducible and scientifically approved raw measurement, but unresolved final clinical mapping.
- `UNAVAILABLE`: extractor cannot be faithfully reproduced, image information is unavailable, or measurement cannot be calculated.
- `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`: scientifically selected, but primary-source details are insufficient for implementation without invention.
- `IMPLEMENTATION_READY`: documentation status indicating that a reproducible approved extractor can be coded without further scientific decisions.

`RAW_EVIDENCE` is allowed only when the mechanism producing the raw measurement is itself reproducible and scientifically approved. An irreproducible approximation must not be emitted as raw evidence. `null` never means clinical absence.

| Feature | Final state before coding |
|---|---|
| ABCD A | `DETERMINISTIC_CLINICAL` |
| ABCD B | `DETERMINISTIC_CLINICAL_WITH_LIMITATION` |
| ABCD C | `DETERMINISTIC_CLINICAL` |
| Pigment network | `RAW_EVIDENCE` through Anantha fallback; Sadeghi preference remains `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` |
| Blue-white veil pixel mask | reproducible / `IMPLEMENTATION_READY` |
| Blue-white veil area fraction | `RAW_EVIDENCE` / `IMPLEMENTATION_READY` |
| 7PCL blue-whitish veil binary | `UNRESOLVED` |
| Menzies blue-white veil binary | `UNRESOLVED` |
| Streaks | `UNAVAILABLE`; Sadeghi 2013 implementation blocked |
| Dots/globules | `UNAVAILABLE`; Maglogiannis and Delibasis 2015 implementation blocked |
| Structureless/blotch | absolute-color `RAW_EVIDENCE` plus approved ABCD structureless-area use with documented limitations / `IMPLEMENTATION_READY` |
| Complete ABCD D | `null` / unresolved |
| Complete deterministic 7PCL-r FE | `UNAVAILABLE` |
| Complete deterministic Menzies-r FE | `UNAVAILABLE` |

Forbidden actions: do not retrain unpublished classifiers, do not calibrate on PH2, do not invent thresholds or scale conversions, do not compute partial ABCD D as complete `D`, and do not describe the dissertation extension as a complete deterministic implementation of ABCD, 7PCL, and Menzies.

Preferred framing: deterministic extraction of dermoscopic evidence grounded in ABCD, Seven-Point Checklist and Menzies criteria for VLM grounding.

## Decision Matrix

| Algorithm | Valence | Candidate Method | Metric | Threshold | Preprocessing | Evidence | Reviewer | Status |
|---|---|---|---|---|---|---|---|---|
| ABCD | A - asymmetry | Center of gravity + moment principal axis from Kasmi and Mokrani (2016); score by color, brightness, and shape symmetry thresholds | Delta E block symmetry; luminance half-difference; area half-difference | Delta E <= 6; brightness < 3%; shape < 2%; published computational thresholds, not universal clinical thresholds; no PH2 calibration | GT mask; original RGB; CIELAB conversion for color asymmetry | `[DERMOSCOPEDIA_ABCD_2023]`, `[MURADIA_2022]`, `[KASMI_2016]` | `PARTIALLY_SUPPORTED` | `HUMAN_APPROVED` |
| ABCD | B - border | Source-guided author operationalization: GT mask, ordered contour, principal-axis-aligned eight 45-degree sectors, dimensionless cubic spline, Euclidean RMSE, `L_smooth`, fixed 0.05 threshold | Per-sector `E_smooth = RMSE_geom / L_smooth`; threshold margin; final `B` | 5% form source-specified, but executable `length(y)` is under-specified; no PH2 calibration; not universal clinical threshold | GT mask; ordered lesion contour | `[DERMOSCOPEDIA_ABCD_2023]`, `[MURADIA_2022]`, `[KASMI_2016]`, `[KAYA_2016]`, `[EROL_2017]` | `PARTIALLY_SUPPORTED`; exact source implementation unavailable, author operationalization implemented | `HUMAN_APPROVED_WITH_LIMITATION` |
| ABCD | C - color | Deterministic six-color RGB classifier on lesion pixels; PH2 annotations reserved for validation, not FE input | Normalized RGB threshold classes; color pixel fractions | Majumder and Ullah (2019) RGB table; 5% lesion-pixel rule; published computational/dataset-derived thresholds, not universal clinical thresholds; no PH2 calibration | GT mask; original RGB; optional highlight exclusion | `[DERMOSCOPEDIA_ABCD_2023]`, `[MAJUMDER_2019]`, `[KASMI_2016]`, `[PH2_2013]` | `PARTIALLY_SUPPORTED` | `HUMAN_APPROVED` |
| ABCD | D - dermoscopic structures | Shared extractor family documented in `shared_extractors.md`; complete ABCD D must remain `null` | Raw evidence only where extractor is reproducible; no partial D score | `UNRESOLVED`; no final D score approved | GT mask; original RGB; selective artifact masks only | `[DERMOSCOPEDIA_ABCD_2023]`, `[ANANTHA_2004]`, `[SADEGHI_PN_2011]`, `[CELEBI_2008]`, `[SADEGHI_STREAKS_2013]`, `[MAGLOGIANNIS_2015]`, `[STOECKER_2005]`, `[ADR-009]`, `[ADR-010]` | Complete score `UNSUPPORTED`; component readiness varies | `ABCD_D=null`; `STILL_UNRESOLVED` |
| 7PCL | Atypical pigment network | Scientific preference remains Sadeghi; practical Anantha fallback may provide raw network evidence only | Anantha block/network texture evidence; no typical/atypical decision | `UNRESOLVED`; no PH2 calibration | GT mask; original RGB; selective hair QC | `[ARGENZIANO_1998]`, `[ANANTHA_2004]`, `[SADEGHI_PN_2011]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[ADR-009]` | 7PCL mapping unresolved | `RAW_EVIDENCE` only for network evidence; atypical decision `UNAVAILABLE` |
| 7PCL | Blue-whitish veil | Celebi et al. remains approved scientific method; Figure 5 pixel detector is implementation-ready | Veil mask and `S1` / veil area fraction are reproducible from the published C4.5 tree; no binary 7PCL checklist value | Celebi pixel thresholds are `CLASSIFIER_DERIVED`; `S1 = 0.009` is not a veil-presence threshold; PH2 calibration prohibited | GT mask; original RGB; skin annulus; highlight exclusion | `[CELEBI_2008]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[ADR-009]`, `[ADR-010]` | pixel detector `IMPLEMENTATION_READY`; clinical binary mapping unresolved | `RAW_EVIDENCE` for mask/fraction; 7PCL binary `UNRESOLVED` |
| 7PCL | Irregular streaks | Sadeghi et al. 2013 remains scientifically selected; implementation blocked | No deterministic or raw streak extractor until source gaps resolved or new decision made | Exact thresholds `UNRESOLVED`; do not reconstruct/retrain | GT mask; original RGB; peripheral band only if source-reproduced later | `[SADEGHI_STREAKS_2013]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[ADR-009]` | not implementation-ready despite prior approval with limitation | `UNAVAILABLE` |
| 7PCL | Irregular pigmentation | Stoecker absolute-color raw blotch evidence may be emitted; deterministic 7PCL irregular-pigmentation mapping unresolved | Dark/structureless region area and implementation-ready area fractions from the absolute-color path only | 10% structureless/blotch area is `CLINICAL_CONSENSUS`; `R < 120 AND G < 60`; `A_blotch > 800 pixels`; 7PCL threshold `UNRESOLVED` | GT mask; original RGB; highlight exclusion; no Stoecker relative-color skin annulus | `[STOECKER_2005]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[ADR-009]`, `[ADR-010]` | raw absolute-color evidence supported; relative-color path blocked; 7PCL mapping unresolved | `RAW_EVIDENCE` only; clinical decision `UNAVAILABLE` |
| 7PCL | Irregular dots/globules | Maglogiannis and Delibasis remains selected scientific reference; implementation blocked | None until reproducible detector is selected | Dot/globule 0.1 mm boundary is clinical only; no pixel conversion approved | GT mask only if future approved extractor exists | `[MAGLOGIANNIS_2015]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[ADR-009]` | not implementation-ready | `UNAVAILABLE` |
| Menzies | Pattern asymmetry | Shared asymmetry plus color/structure divergence | Best-axis pattern divergence | `UNRESOLVED` | GT mask; shared structure/color masks | `[MENZIES_1996]`, `[DERMOSCOPEDIA_ABCD_2023]`, `[KASMI_2016]` | `PARTIALLY_SUPPORTED` | `UNRESOLVED` |
| Menzies | Single color / non-uniformity | Shared color classifier with Menzies color set | Menzies color count, entropy | `count == 1` concept supported; area threshold `UNRESOLVED` | GT mask | `[MENZIES_1996]`, `[MAJUMDER_2019]`, `[KASMI_2016]`, `[ADR-009]` | `PARTIALLY_SUPPORTED`; complete Menzies-r unavailable | `PARTIAL_COLOR_EVIDENCE`; final Menzies-r `UNAVAILABLE` |
| Menzies | Blue-white veil | Shared Celebi blue-white veil detector | Veil mask and `S1` / veil area fraction; no binary Menzies checklist value | Figure 5 thresholds are `CLASSIFIER_DERIVED`; `S1 = 0.009` is not a veil-presence threshold; PH2 calibration prohibited | GT mask; skin annulus; highlight exclusion | `[BRAUN_MENZIES_2017]`, `[CELEBI_2008]`, `[ARGENZIANO_1998]`, `[ADR-009]`, `[ADR-010]` | pixel detector `IMPLEMENTATION_READY`; clinical binary mapping unresolved | `RAW_EVIDENCE` for mask/fraction; Menzies binary `UNRESOLVED` |
| Menzies | 5-6 colors | Shared color classifier, Menzies count | Count of red/tan/dark brown/black/gray/blue | `>=5` supported; area threshold `UNRESOLVED` | GT mask | `[MENZIES_1996]`, `[ADR-009]` | `PARTIALLY_SUPPORTED`; complete Menzies-r unavailable | `PARTIAL_COLOR_EVIDENCE`; final Menzies-r `UNAVAILABLE` |
| Menzies | Blue-gray dots | Maglogiannis-selected blob method is implementation-blocked | None until reproducible detector is selected | `UNRESOLVED`; no PH2 calibration or pixel-size inference | GT mask only if future approved extractor exists | `[BRAUN_MENZIES_2017]`, `[MAGLOGIANNIS_2015]`, `[PH2_DATABASE]`, `[ADR-009]` | `NEEDS_MORE_RESEARCH` | `UNAVAILABLE` |
| Menzies | Broadened network | Anantha fallback can provide raw network evidence only; broadened-network decision unresolved | Raw network texture evidence only | `UNRESOLVED`; no focal-width threshold | GT mask; original RGB; hair QC | `[BRAUN_MENZIES_2017]`, `[ANANTHA_2004]`, `[SADEGHI_PN_2011]`, `[ADR-009]` | `NEEDS_MORE_RESEARCH` | `RAW_EVIDENCE` only; clinical decision `UNAVAILABLE` |
| Menzies | Peripheral black dots/globules | Maglogiannis-selected blob method is implementation-blocked | None until reproducible detector is selected | `UNRESOLVED`; PH2 dots/globules GT is partial validation only | GT mask only if future approved extractor exists | `[BRAUN_MENZIES_2017]`, `[MAGLOGIANNIS_2015]`, `[PH2_DATABASE]`, `[ADR-009]` | `NEEDS_MORE_RESEARCH` | `UNAVAILABLE` |
| Menzies | Radial streaming / pseudopods | Sadeghi-selected radial method is implementation-blocked; subtype interpretation remains separate | None until reproducible detector is selected | `UNRESOLVED`; subtype thresholds missing | GT mask only if future approved extractor exists | `[BRAUN_MENZIES_2017]`, `[SADEGHI_STREAKS_2013]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[ADR-009]` | `PARTIALLY_SUPPORTED` | `UNAVAILABLE` |

## Criteria Specifications

### ABCD A - Asymmetry

Clinical concept: asymmetry across two perpendicular axes considering contour, colors, and structures.

Candidate implementation: compute the lesion center of gravity and the first principal axis using second-order image moments as described by Kasmi and Mokrani (2016). Rotate the mask/image by `-phi`, evaluate the rotated horizontal and vertical axes, and classify an axis as symmetric only if color, brightness, and shape are all symmetric under that method.

Input: original RGB image, PH2 lesion mask.

Raw outputs: center of gravity, moment-derived principal axis angle `phi`, rotated mask, color Delta E per mirrored block pair, count of symmetric/asymmetric block pairs by axis, mean luminance difference by axis, lesion-area half-difference by axis.

Algorithm output: approved ABCD `A` score using Kasmi and Mokrani (2016): `0` if both axes are symmetric, `1` if exactly one axis is symmetric, `2` otherwise.

Formula: `phi` is computed from second-order central moments `m11`, `m20`, and `m02` following Kasmi and Mokrani (2016), Section 4.1, equations 7-9. Color asymmetry uses CIELAB `Delta E` between mirrored 20 x 20 blocks. Shape asymmetry uses the area difference between opposite halves. Brightness asymmetry uses the average luminance difference between opposite halves.

Parameters: center of gravity; moment axis angle `phi`; 20 x 20 block size; at least 75% of block area inside lesion; CIELAB Delta E threshold `6`; brightness threshold `3%` of total average lesion luminance; shape threshold `2%` of lesion area.

Threshold: `Delta E <= 6`, brightness difference `<3%`, and shape difference `<2%` are adopted from Kasmi and Mokrani (2016). These are published computational thresholds from a prior automatic ABCD implementation and are not claimed to be universal clinical thresholds. No PH2 calibration will be performed.

Preprocessing: GT mask; no required hair removal.

Shared extractor: Menzies pattern asymmetry.

Scientific traceability:

* `[DERMOSCOPEDIA_ABCD_2023]` - `CLINICAL_DEFINITION` - supports ABCD A as 0/1/2 based on the number of asymmetric perpendicular axes, considering contour, colors, and structures.
* `[MURADIA_2022]` - `CLINICAL_DEFINITION` - supports placing the first clinical axis at the division of "most symmetry" and the second at 90 degrees; does not provide a computational optimization method.
* `[KASMI_2016]` - `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION` - supports center of gravity, moment-derived principal axis, rotation by `-phi`, CIELAB block comparison, 20 x 20 blocks, 75% lesion-coverage filter, Delta E 6, 3% brightness threshold, 2% shape threshold, and A=0/1/2 conversion.
* No sufficient supporting reference identified for a published computational method that explicitly searches all possible axes to reproduce the clinical "most symmetry" procedure.

Human decision: `APPROVED`. Adopt Kasmi and Mokrani (2016) for ABCD A, including axis selection, color symmetry, brightness symmetry, shape symmetry, published thresholds, and `A=0/1/2` conversion. Use the published parameters without PH2 calibration. PH2 asymmetry annotations, if used, are validation ground truth only and must never be FE inputs.

Rationale: most reproducible PH2-compatible shape metric, but not complete clinical asymmetry.

Known limitations: does not fully capture color/structure asymmetry.

Review status: `PARTIALLY_SUPPORTED`.

Decision: `HUMAN_APPROVED`.

### ABCD B - Border

Clinical concept: number of eight sectors with abrupt pigment cutoff.

Final implementation: divide the ordered lesion contour into eight 45-degree sectors in a principal-axis-aligned frame and score each sector using a source-guided third-order spline fitting-error operationalization. Treat this as an automatic geometric border-irregularity proxy for ABCD B, not as a fully resolved pigment-transition abruptness detector.

Input: RGB image, lesion mask.

Raw outputs: lesion centroid, principal-axis diagnostics, ordered contour, eight sector assignments, contiguous sub-contour runs per sector, normalized arc-length samples, local chord scale and fallback flag, third-order spline fit per run, Euclidean RMSE per sector, `L_smooth`, `E_smooth`, threshold margin, binary sector irregularity, and `B=sum(binary_sectors)`. Auxiliary pigment-transition homogeneity may be stored separately from Kaya et al. (2016) or Erol et al. (2017), but it does not produce ABCD B.

Algorithm output: `B=0..8` as a source-guided author operationalization based on Kasmi and Mokrani (2016). This remains approved with limitation as a geometric sub-contour irregularity proxy for the clinical concept of abrupt pigment cutoff.

Formula: each octant sub-contour is approximated by a third-order spline; a sector is positive when fitting error exceeds `0.05 * length(y)`, where `y` is the sub-contour, as stated in Kasmi and Mokrani (2016), Section 4.2.

Implementation-resolution note: the exact executable meaning of `y` and `length(y)` is under-specified in the source and remains a preserved source gap. The final dissertation implementation uses an ordered external 2D contour, lesion-pixel second-moment principal axis, principal-axis-relative sector assignment, contiguous ordered sub-contour runs, cumulative arc-length parameterization, dimensionless cubic smoothing splines with `alpha=1.0`, Euclidean geometric RMSE, and `E_smooth = RMSE / L_smooth`. A sector is positive when `E_smooth > 0.05`. These are author implementation conventions evaluated on controlled synthetic geometric cases, not PH2 calibration and not universal clinical thresholds. Detailed validation is documented in `docs/abcd_border_methodology.md`.

Parameters: lesion centroid, moment-derived principal axis `phi`, 45-degree sectors, ordered contour extraction, third-order smoothing spline, dimensionless smoothing `spline_s = alpha * n_samples` with `alpha=1.0`, local chord normalization, smoothed-contour denominator `L_smooth`, and fixed threshold `0.05`.

Threshold: the 5% threshold form is source-specified by Kasmi and Mokrani (2016), but `RMSE / L_smooth > 0.05` is the dissertation author operationalization because `length(y)` is not executable from the source alone. No PH2 calibration will be performed.

Preprocessing: GT mask; exclude highlights/hair if confounding border transition.

Shared extractor: none direct. Kaya et al. (2016) and Erol et al. (2017) abruptness homogeneity features may be stored as auxiliary border-abruptness evidence.

Scientific traceability:

* `[DERMOSCOPEDIA_ABCD_2023]` - `CLINICAL_DEFINITION` - supports B as the number of eight segments with a sharp abrupt ending of pigment pattern.
* `[MURADIA_2022]` - `CLINICAL_DEFINITION` - supports counting eight pie-shaped pieces with abrupt perimeter cutoff, range 0-8.
* `[KASMI_2016]` - `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION` - supports eight 45-degree octants, third-order spline approximation, fitting-error threshold form `0.05 * length(y)`, one point per positive sub-contour, and `B=0..8`; it does not fully specify the executable contour-vector and `length(y)` conventions used here.
* `[KAYA_2016]` - `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION` - supports continuous abruptness quantification by contracted/scaled lesion borders and texture homogeneity, but explicitly does not output eight ABCD sector scores.
* `[EROL_2017]` - `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION` - supports inner-border level-set contraction and homogeneity/mean/std feature vectors for abruptness, but not ABCD `B=0..8`.
* No sufficient supporting reference identified for a fully specified, PH2-validated, eight-sector lesion-to-skin color-transition threshold.

Human decision: `APPROVED_WITH_LIMITATION`. Adopt Kasmi and Mokrani (2016) as the source basis for ABCD B while preserving the `length(y)` specification gap. The executable dissertation method is an implemented author operationalization using principal-axis-aligned sectors, ordered parametric contour runs, dimensionless smoothing, Euclidean RMSE, `L_smooth`, and the fixed 0.05 threshold. Document the method as a computational proxy based on geometric sub-contour irregularity. Do not claim direct lesion-to-skin pigment-transition measurement. Do not combine Kaya et al. (2016) or Erol et al. (2017) with Kasmi and Mokrani (2016) to create a new unpublished ABCD B formula; Kaya/Erol remain auxiliary evidence only and are not required for the initial implementation.

Rationale: preserves the clinical eight-sector score while making clear that the currently traceable automatic implementation is a contour irregularity proxy.

Known limitations: abrupt pigment cutoff is not the same as contour irregularity. The primary source does not fully specify every low-level spline, contour-vector, or `length(y)` convention; exact source reproduction remains unavailable. Synthetic validation supports the adopted author conventions for controlled geometric behavior, but does not establish clinical validity. Binary sector thresholding has edge behavior near `E_smooth=0.05`, documented by the localized-three-octants P3 diagnostic.

Review status: `PARTIALLY_SUPPORTED`.

Decision: `HUMAN_APPROVED_WITH_LIMITATION`.

### ABCD C / Menzies Color

Clinical concept: ABCD counts six colors; Menzies checks single color and multiple 5-6 colors with a closely related but not identical set.

Candidate implementation: classify only lesion pixels from the original RGB image using Majumder and Ullah's normalized RGB threshold table for the six ABCD colors. Use PH2 expert color annotations only as validation ground truth, not as extractor input.

Input: RGB image, lesion mask; optional skin annulus.

Raw outputs: valid lesion-pixel count; per-color pixel count and lesion fraction for white, red, light brown, dark brown, blue-gray, and black; ABCD color count; pixels not assigned to any ABCD color; optional highlight-confounded flag.

Algorithm output: approved ABCD color count `C=1..6` using the Majumder and Ullah (2019) normalized RGB thresholds and the 5% lesion-area color-presence rule.

Formula: `color_present_i = area(color_i pixels)/area(valid lesion pixels) >= tau_color`.

Parameters: normalized RGB in `[0,1]`; color thresholds from Majumder and Ullah (2019), Table 1; presence threshold `tau_color=0.05`.

Threshold: Majumder and Ullah (2019), Table 1, defines normalized RGB ranges: white `R>=0.8, G>=0.8, B>=0.8`; red `R>=0.588, G<0.2, B<0.2`; light brown `0.588<=R<=0.94, 0.196<G<=0.588, 0<B<0.392`; dark brown `0.243<R<0.56, 0<=G<0.392, 0<B<0.392`; blue-gray `0<=R<=0.588, 0.392<=G<=0.588, 0.490<=B<=0.588`; black `R<=0.243, G<=0.243, B<=0.243`. The 5% presence rule is adopted as a published computational/dataset-derived threshold. These thresholds are not claimed to be universal clinical thresholds. No PH2 calibration will be performed.

Preprocessing: original PH2 RGB; no global normalization unless thresholds are calibrated after it; highlight exclusion.

Shared extractor: ABCD C; Menzies single/multiple colors; irregular pigmentation support.

Scientific traceability:

* `[DERMOSCOPEDIA_ABCD_2023]` - `CLINICAL_DEFINITION` - supports the six ABCD colors and the statement that white must be whiter than adjacent normal skin.
* `[MAJUMDER_2019]` - `COMPUTATIONAL_OPERATIONALIZATION`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION` - supports the normalized RGB threshold table and the 5% lesion-pixel presence rule; thresholds were computed from 200 input images.
* `[KASMI_2016]` - `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION` - supports normalized Euclidean distance to RGB reference colors, `T=0.4`, and the same 5% presence rule on 200 EDRA images.
* `[PH2_2013]` - `DATASET_VALIDATION` - supports PH2 as a 200-image dermoscopy dataset with lesion masks and dermoscopic annotations; it does not support using annotations as FE input.

Human decision: `APPROVED`. Adopt Majumder and Ullah (2019) normalized RGB thresholds and the published 5% lesion-area color-presence rule. Use published parameters without PH2 calibration. PH2 expert color annotations must be used exclusively as validation ground truth and never as Feature Extraction input.

Rationale: PH2 explicitly provides color annotations and controlled acquisition.

Known limitations: acquisition-dependent thresholds; white is relative to surrounding skin.

Review status: `PARTIALLY_SUPPORTED`.

Decision: `HUMAN_APPROVED`.

### ABCD D - Dermoscopic Structures Pending

Historical note: this section preserves the earlier `PENDING_SHARED_EXTRACTOR_RESEARCH` state. It is superseded for active implementation readiness by ADR-009 and ADR-010. The shared-extractor research stage has been completed for this dissertation, no further shared-extractor research is planned, and the active state is `ABCD_D = null` / `STILL_UNRESOLVED`.

Clinical concept: ABCD `D` is the differential dermoscopic-structures component. The final score depends on structure categories such as pigment network, structureless areas, streaks, dots, and globules.

Human decision: historical `PENDING`. Do not approve or implement the final ABCD D score. The completed shared-extractor research stage records candidate low-level detectors and algorithm-specific interpretations only; it does not approve the final ABCD D formula.

Methodological limitation: no final ABCD D scoring formula, threshold set, or binary structure-presence rule is approved in this document. Existing structure sections provide reproducible raw evidence only where explicitly marked implementation-ready.

Historical decision: `PENDING_SHARED_EXTRACTOR_RESEARCH`.

Active decision: `ABCD_D = null`; `STILL_UNRESOLVED`.

## Shared Dermoscopic Structure Extractors Stage

Detailed evidence cards, source-specific limitations, threshold classifications, algorithm-specific interpretation candidates, and PH2 validation availability are recorded in `docs/feature_extraction/shared_extractors.md`.

This stage is documentation only. It does not implement Feature Extraction, does not modify the current VLM pipeline, does not reopen the human-approved ABCD A/B/C decisions, and does not use PH2 annotations as extractor inputs.

### Shared Extractor Matrix

| Shared extractor | Raw outputs to preserve | ABCD interpretation | 7PCL interpretation | Menzies interpretation | Low-level sharing | Final-threshold sharing |
|---|---|---|---|---|---|---|
| Pigment network | Candidate network mask, block response, cyclic holes, graph density, line width, mesh/hole heterogeneity, color variation | Pigment network presence for ABCD D, final scoring pending | Atypical pigment network requires prominent/broad and irregular network evidence | Broadened network requires focal cord/grid width increase | Allowed | Not approved |
| Blue-white veil | Candidate veil mask, area fraction, component geometry, absolute/relative color, texture/context features, confound flags | Auxiliary evidence only; ABCD D final use pending | Raw evidence only; binary criterion unresolved | Raw evidence only; binary criterion unresolved | Allowed | Not approved |
| Peripheral radial structures | Candidate radial mask, count, length, width, radiality, angular coverage, branching, terminal bulb/stalk descriptors | Branched streaks candidate | Irregular streaks candidate | Radial streaming and pseudopods interpreted separately | Allowed | Not approved |
| Dots/globules | Candidate blob mask, count, area, circularity, color class, size variability, peripheral fraction, spatial entropy | Dots and globules candidates | Irregular dots/globules candidate | Multiple brown dots, blue-gray dots, peripheral black dots/globules interpreted separately | Allowed | Not approved |
| Structureless/blotch/irregular pigmentation | Candidate structureless/blotch masks, area fractions, largest component, eccentricity, irregularity, color class, spatial distribution | Structureless areas candidate | Irregular diffuse pigmentation/blotches candidate | Color non-uniformity support only | Allowed | Not approved |

### Stage-Wide Constraints

PH2 annotations are validation ground truth only. PH2 pigment network, blue-whitish veil, streaks, dots/globules, color, regression, asymmetry, and diagnosis annotations must never be used as Feature Extraction inputs or calibration targets in the initial specification.

Threshold origins must be recorded using the approved categories: `CLINICAL_CONSENSUS`, `PUBLISHED_COMPUTATIONAL`, `DATASET_DERIVED`, `CLASSIFIER_DERIVED`, `AUTHOR_DEFINED`, and `UNRESOLVED`.

All raw intermediate outputs listed above must be preserved for validation and debugging before any future binary clinical decision is computed.

Shared-extractor decisions from the earlier stage are superseded for implementation readiness by ADR-009 and ADR-010. Historical `READY_FOR_APPROVAL`, `PENDING_HUMAN_APPROVAL`, or `PENDING_SHARED_EXTRACTOR_RESEARCH` labels mean only that the evidence package was awaiting review at that time; the active readiness states are the ones listed in this specification's final pre-coding table and ADR-010.

### Blue-Whitish Veil

Clinical concept: irregular, structureless, confluent blue to whitish-blue area with white ground-glass haze.

Candidate implementation: shared 7PCL/Menzies blue-white candidate mask using relative color and contextual texture features.

Input: RGB image, lesion mask, surrounding skin annulus.

Raw outputs: blue-white candidate mask, area fraction, component count, component irregularity, mean relative color features, highlight-confounded flag.

Algorithm output: Celebi pixel-level detector is `IMPLEMENTATION_READY` after final human confirmation. It may produce the pixel-level `veil_mask` and `S1` / veil area fraction as `RAW_EVIDENCE`. Binary 7PCL-r or Menzies-r clinical presence must not be produced.

Formula: `veil_fraction = area(veil_candidate)/area(valid lesion)`.

Parameters: annulus widths, color feature definitions, neighborhood size, post-filter size, and Figure 5 pixel tree.

Threshold: Celebi et al. (2008) Figure 5 pixel-level tree is `F3 <= 0.3 -> Non-Veil`; otherwise `F10 > -51 -> Non-Veil`; otherwise `F10 <= -194 -> Non-Veil`; otherwise `Veil`. Equivalent positive interval: `F3 > 0.3 AND -194 < F10 <= -51`. The recovered lesion-level tree, `S1 < 0.009 -> benign`; otherwise `S3 > 0.979 -> benign`; otherwise melanoma, is a melanoma/benign diagnostic classifier and must not be used as a 7PCL or Menzies blue-white-veil presence threshold. Do not use `S1 = 0.009` or `S1 > 0` as a clinical veil-presence rule. No PH2 calibration will be performed.

Preprocessing: highlight exclusion; skin annulus if using relative color.

Shared extractor: 7PCL blue-whitish veil; Menzies blue-white veil; ABCD blue/white color and structureless evidence.

Evidence: `[CELEBI_2008]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[PH2_DATABASE]`, `[ADR-010]`.

PH2 validation: direct present/absent blue-whitish veil label exists; no PH2 pixel-level veil mask is described in the public PH2 summary. PH2 labels are validation ground truth only.

Rationale: strongest single structure-specific FE evidence found; Figure 5 pixel-tree values were recovered by primary-source visual inspection and received final human confirmation for pixel-level implementation.

Known limitations: overlaps with regression and blue-gray dots; may confuse reflections.

Review status: `SUPPORTED` for raw features.

Decision: `HUMAN_APPROVED` scientifically; pixel detector `IMPLEMENTATION_READY`; `veil_mask` and `S1`/`veil_area_fraction` are implementation-ready `RAW_EVIDENCE`; 7PCL-r and Menzies-r clinical binary mappings remain `UNRESOLVED`.

### Pigment Network / Broadened Network

Clinical concept: pigmented grid of lines and lighter holes; atypical/broadened networks have irregular, thickened, variably colored lines and irregular meshes.

Candidate implementation: Sadeghi et al. (2011) remains the scientific preference, but implementation is blocked by reproducibility gaps. Anantha, Moss and Stoecker (2004) is the approved practical fallback for pigment-network `RAW_EVIDENCE` only.

Input: RGB image, lesion mask.

Raw outputs: Anantha fallback may emit block/network texture evidence and intermediate thresholds. Sadeghi graph/cyclic-hole outputs must not be emitted until the selected method is reproducible.

Algorithm output: pigment-network `RAW_EVIDENCE` through Anantha fallback only. ABCD pigment-network presence, 7PCL atypical network, and Menzies broadened network remain `null` / unresolved.

Formula: method-specific texture/graph features; no single resolved formula.

Parameters: Laws/Gabor/edge filters, block size, graph construction.

Threshold: Anantha et al. (2004) thresholds are `PUBLISHED_COMPUTATIONAL`, `AUTHOR_DEFINED`, and `DATASET_DERIVED` for their image set. Sadeghi et al. (2011) graph-density threshold was not fully available from the accessible abstract. Atypical-network and broadened-network thresholds remain `UNRESOLVED`. No PH2 calibration will be performed.

Preprocessing: selective hair QC only.

Shared extractor: 7PCL atypical pigment network; Menzies broadened network; ABCD pigment network.

Evidence: `[ANANTHA_2004]`, `[SADEGHI_PN_2011]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[BRAUN_MENZIES_2017]`, `[PH2_2013]`, `[PH2_DATABASE]`.

PH2 validation: direct pigment-network label exists as typical/atypical; PH2 does not provide low-level line, hole, mesh, or network mask annotations. PH2 labels are validation ground truth only.

Rationale: Sadeghi is semantically preferred but implementation-blocked; Anantha is reproducible enough for raw evidence but not for final clinical mappings.

Known limitations: hair-removal false positives can damage network; typical/atypical thresholds vary by image scale.

Review status: `PARTIALLY_SUPPORTED`.

Decision: Sadeghi scientific preference `HUMAN_APPROVED`; Sadeghi implementation `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; Anantha fallback `RAW_EVIDENCE`.

### Streaks / Radial Streaming / Pseudopods

Clinical concept: pigmented peripheral radial projections; radial streaming is linear, pseudopods have bulbous ends attached to the lesion.

Candidate implementation: shared peripheral-band radial structure detector with graph/skeleton features.

Input: RGB image, lesion mask.

Raw outputs: candidate streak mask, line count, length, width, radial orientation score, angular sector coverage, terminal bulb/stalk descriptors.

Algorithm output: `UNAVAILABLE` under the current checkpoint. No raw or deterministic streak extractor may be implemented from Sadeghi et al. (2013) until source gaps are resolved or a new human methodological decision is made.

Formula: no single resolved formula; orientation can be stored as angle difference between candidate skeleton tangent and radial vector from centroid.

Parameters: peripheral band width, line enhancement scale, skeleton pruning, length/width thresholds.

Threshold: Sadeghi et al. (2013) classifier outputs are `CLASSIFIER_DERIVED` and `PUBLISHED_COMPUTATIONAL`; exact low-level implementation thresholds and PH2-specific radial subtype thresholds are `UNRESOLVED`. No PH2 calibration will be performed.

Preprocessing: hair QC; preserve raw image.

Shared extractor: 7PCL irregular streaks; Menzies radial streaming/pseudopods; ABCD branched streaks.

Evidence: `[SADEGHI_STREAKS_2013]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[BRAUN_MENZIES_2017]`, `[DERMOSCOPEDIA_ABCD_2023]`, `[PH2_DATABASE]`.

PH2 validation: direct streaks present/absent label exists; PH2 does not separately annotate irregular streaks, branched streaks, radial streaming, or pseudopods. PH2 labels are validation ground truth only.

Rationale: Sadeghi et al. (2013) remains scientifically selected, but missing implementation-critical source details block faithful reproduction.

Known limitations: PH2 labels streaks only present/absent; hair can mimic streaks.

Review status: `PARTIALLY_SUPPORTED`.

Decision: scientific method `HUMAN_APPROVED`; implementation readiness `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

### Dots / Globules / Blue-Gray Dots / Peripheral Black Dots

Clinical concept: dots/globules are round or oval pigmented structures; irregularity depends on variable size/color/shape and asymmetric/peripheral distribution.

Candidate implementation: shared dark/blue-gray blob detector plus spatial distribution features.

Input: RGB image, lesion mask.

Raw outputs: blob mask, count, area distribution, circularity, color class per blob, nearest-neighbor distances, ring/sector distribution, peripheral fraction.

Algorithm output: `UNAVAILABLE` under the current checkpoint. No deterministic or raw dots/globules FE may be emitted until a reproducible approved extractor is selected.

Formula: `circularity = 4*pi*A/P^2`; `peripheral_fraction = count(blobs in peripheral band)/count(all blobs)`.

Parameters: blob scale, size thresholds, circularity cutoff, peripheral-band definition.

Threshold: the dot/globule boundary around 0.1 mm is `CLINICAL_CONSENSUS`. Maglogiannis and Delibasis (2015) support a `PUBLISHED_COMPUTATIONAL` detector concept, but accessible evidence does not expose reproducible size, circularity, color, or count thresholds. Implementation thresholds are `UNRESOLVED`. No PH2 calibration will be performed.

Preprocessing: reflection/hair QC; avoid confusing network intersections with blobs.

Shared extractor: 7PCL irregular dots/globules; Menzies blue-gray dots, multiple brown dots, peripheral black dots/globules; ABCD dots/globules.

Evidence: `[MAGLOGIANNIS_2015]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[BRAUN_MENZIES_2017]`, `[PH2_2013]`, `[PH2_DATABASE]`.

PH2 validation: direct dots/globules label exists as absent/atypical/typical; PH2 does not provide separate dot masks, globule masks, blue-gray dot labels, or peripheral black dot/globule labels. PH2 labels are validation ground truth only.

Rationale: Maglogiannis and Delibasis (2015) remains the selected reference, but its implementation-critical details were not recovered.

Known limitations: thresholds are scale-dependent; peppering and tiny dots overlap.

Review status: `PARTIALLY_SUPPORTED`.

Decision: scientific reference selected; implementation readiness `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

### Irregular Pigmentation / Structureless Areas

Clinical concept: irregular diffuse pigmentation/blotches are black, brown, or gray structureless pigmented regions with irregular shape/distribution.

Candidate implementation: Stoecker et al. (2005) absolute-color blotch segmentation after excluding artifact masks. The relative-color segmentation remains scientifically documented but is not implementation-ready.

Input: RGB image, lesion mask, color masks.

Raw outputs: absolute-color blotch candidate mask after `R < 120 AND G < 60` and `A_blotch > 800 pixels`; implementation-ready area outputs that do not require unresolved conventions; debug masks and candidate/component areas. Features depending on connected-component connectivity, perimeter estimation, or other implementation-sensitive conventions must be identified before coding rather than silently using library defaults.

Algorithm output: Stoecker absolute-color raw blotch `RAW_EVIDENCE`; ABCD structureless-area use is approved with documented limitations. Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. 7PCL irregular pigmentation and Menzies color nonuniformity remain unresolved/auxiliary.

Formula: implementation-ready segmentation is `R < 120 AND G < 60` inside the lesion mask followed by `A_blotch > 800 pixels` on original-resolution PH2 images. Candidate `total_blotch_fraction = sum(area(C_i))/area(M)` and `largest_blotch_fraction = area(C_max)/area(M)` are supported if the component convention is specified before coding. `irregularity = P_max/sqrt(area(C_max))` remains implementation-sensitive because the perimeter estimator is not specified.

Parameters: absolute-color thresholds `R < 120 AND G < 60`; minimum candidate/component rule `A_blotch > 800 pixels`; original-resolution PH2 images; no PH2 calibration, rescaling, or physical-equivalence inference.

Threshold: the 10% structureless/blotch area rule is `CLINICAL_CONSENSUS`. Stoecker et al. (2005) support the `PUBLISHED_COMPUTATIONAL` absolute-color rule `R < 120 AND G < 60` and minimum candidate/component rule `A_blotch > 800 pixels`. The 800-pixel threshold is source-scale-specific and must be used unchanged on original-resolution PH2 images. Stoecker relative-color thresholds remain scientifically documented but blocked because the outer extent of the background-skin sampling region is unspecified. Final 7PCL/Menzies thresholds are `UNRESOLVED`. No PH2 calibration will be performed.

Preprocessing: highlight exclusion. Do not substitute Celebi's skin-annulus construction into Stoecker's relative-color method.

Shared extractor: 7PCL irregular pigmentation; ABCD structureless areas; Menzies color nonuniformity support.

Evidence: `[STOECKER_2005]`, `[ARGENZIANO_1998]`, `[BRAUN_KERL_STRUCTURES_2018]`, `[PH2_2013]`, `[PH2_DATABASE]`.

PH2 validation: no direct PH2 irregular-pigmentation, blotch, or structureless-area label is described in the public PH2 summary. Color, regression, and lesion-mask labels are partial/related validation evidence only and must never be extractor inputs.

Rationale: Stoecker et al. (2005) absolute-color segmentation and the 800-pixel candidate/component rule are reproducible enough for implementation; the relative-color path is blocked by the unspecified background-skin sampling extent.

Known limitations: PH2 lacks direct irregular-pigmentation label. Connected-component connectivity and perimeter estimator are not specified in the primary source and must be explicitly identified before coding for any dependent outputs.

Review status: `PARTIALLY_SUPPORTED`.

Decision: `HUMAN_APPROVED` for Stoecker absolute-color raw blotch segmentation, `R < 120 AND G < 60`, `A_blotch > 800 pixels`, and approved ABCD structureless-area use with limitations. Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

## Human Decisions Required

- ADR-010 confirms the fully transcribed Celebi Figure 5 pixel tree is implementation-ready for coding as a pixel-level detector only.
- ABCD D remains `null`; do not approve or implement the final ABCD D score.
- No new methodological search will be conducted within this dissertation for currently unresolved or unavailable structure extractors. Their incompleteness is an accepted limitation.
- Implementation schemas must expose `UNAVAILABLE` and `RAW_EVIDENCE` without conflating `null` with clinical absence.

## Bibliographic References

`[NACHBAR_1994]` Nachbar F, Stolz W, Merkle T, Cognetta AB, Vogt T, Landthaler M, Bilek P, Braun-Falco O, Plewig G. "The ABCD rule of dermatoscopy. High prospective value in the diagnosis of doubtful melanocytic skin lesions." Journal of the American Academy of Dermatology. 1994;30(4):551-559. PMID `8157780`. https://pubmed.ncbi.nlm.nih.gov/8157780/

`[DERMOSCOPEDIA_ABCD_2023]` Kunz M, Stolz W. "ABCD rule." Dermoscopedia. Last edited 2023. https://dermoscopedia.org/ABCD_rule

`[MURADIA_2022]` Muradia I, Khunger N, Yadav AK. "Real-time Intraoperative Dermoscopic Monitoring of Common Acquired Melanocytic Nevi Following Shave Excision." Journal of Cutaneous and Aesthetic Surgery. 2022;15(3):230-236. DOI `10.4103/JCAS.JCAS_174_21`. https://jcasonline.com/real-time-intraoperative-dermoscopic-monitoring-of-common-acquired-melanocytic-nevi-following-shave-excision/

`[KASMI_2016]` Kasmi R, Mokrani K. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing. 2016;10(6):448-455. DOI `10.1049/iet-ipr.2015.0385`. https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2015.0385

`[KAYA_2016]` Kaya S, Bayraktar M, Kockara S, Mete M, Halic T, Field HE, Wong HK. "Abrupt skin lesion border cutoff measurement for malignancy detection in dermoscopy images." BMC Bioinformatics. 2016;17(Suppl 13):367. DOI `10.1186/s12859-016-1221-4`; PMID `27766942`. https://pubmed.ncbi.nlm.nih.gov/27766942/

`[EROL_2017]` Erol R, Bayraktar M, Kockara S, Kaya S, Halic T. "Texture based skin lesion abruptness quantification to detect malignancy." BMC Bioinformatics. 2017;18(Suppl 14):484. DOI `10.1186/s12859-017-1892-5`; PMID `29297290`. https://pubmed.ncbi.nlm.nih.gov/29297290/

`[MAJUMDER_2019]` Majumder S, Ullah MA. "A Computational Approach to Pertinent Feature Extraction for Diagnosis of Melanoma Skin Lesion." Pattern Recognition and Image Analysis. 2019;29(3):503-514. DOI `10.1134/S1054661819030131`. https://impact.ornl.gov/en/publications/a-computational-approach-to-pertinent-feature-extraction-for-diag/

`[PH2_2013]` Mendonca T, Ferreira PM, Marques JS, Marcal ARS, Rozeira J. "PH2 - A dermoscopic image database for research and benchmarking." 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 2013. DOI `10.1109/EMBC.2013.6610779`; PMID `24110966`. https://pubmed.ncbi.nlm.nih.gov/24110966/

`[PH2_DATABASE]` ADDI Project. "PH2 Database." Universidade do Porto. https://www.fc.up.pt/addi/ph2%20database.html

`[ARGENZIANO_1998]` Argenziano G, Fabbrocini G, Carli P, De Giorgi V, Sammarco E, Delfino M. "Epiluminescence microscopy for the diagnosis of doubtful melanocytic skin lesions." Archives of Dermatology. 1998;134(12):1563-1570. PMID `9875194`. https://jamanetwork.com/journals/jamadermatology/fullarticle/189703

`[MENZIES_1996]` Menzies SW, Ingvar C, Crotty KA, McCarthy WH. "Frequency and morphologic characteristics of invasive melanomas lacking specific surface microscopic features." Archives of Dermatology. 1996;132(10):1178-1182. PMID `8859028`. https://pubmed.ncbi.nlm.nih.gov/8859028/

`[KITTLER_2016]` Kittler H, et al. "Standardization of terminology in dermoscopy/dermatoscopy." https://pmc.ncbi.nlm.nih.gov/articles/PMC5551974/

`[BRAUN_KERL_STRUCTURES_2018]` Braun RP, Kerl K. "Dermoscopic structures." Dermoscopedia. Status update 2018. https://dermoscopedia.org/Dermoscopic_structures

`[BRAUN_MENZIES_2017]` Braun RP, Menzies SW. "Menzies Method." Dermoscopedia. Status update 2017. https://dermoscopedia.org/Menzies_Method

`[JAIMES_2016]` Jaimes N, et al. "Using Dermoscopy to Identify Melanoma and Improve Diagnostic Discrimination." https://pmc.ncbi.nlm.nih.gov/articles/PMC6375419/

`[CELEBI_2008]` Celebi ME, et al. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." Computerized Medical Imaging and Graphics. 2008. PMID `18804955`. https://pubmed.ncbi.nlm.nih.gov/18804955/

`[ANANTHA_2004]` Anantha M, Moss RH, Stoecker WV. "Detection of pigment network in dermatoscopy images using texture analysis." Computerized Medical Imaging and Graphics. 2004. PMID `15249068`. https://pubmed.ncbi.nlm.nih.gov/15249068/

`[SADEGHI_PN_2011]` Sadeghi M, Razmara M, Lee TK, Atkins MS. "A novel method for detection of pigment network in dermoscopic images using graphs." Computerized Medical Imaging and Graphics. 2011. PMID `20724109`. https://pubmed.ncbi.nlm.nih.gov/20724109/

`[SADEGHI_STREAKS_2013]` Sadeghi M, Lee TK, McLean D, Lui H, Atkins MS. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." IEEE Transactions on Medical Imaging. 2013. PMID `23335664`. https://pubmed.ncbi.nlm.nih.gov/23335664/

`[STOECKER_2005]` Stoecker WV, et al. "Detection of asymmetric blotches..." Skin Research and Technology. 2005. DOI `10.1111/j.1600-0846.2005.00117.x`. https://doi.org/10.1111/j.1600-0846.2005.00117.x

`[MAGLOGIANNIS_2015]` Maglogiannis I, Delibasis KK. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." Computer Methods and Programs in Biomedicine. 2015. PMID `25540998`. https://pubmed.ncbi.nlm.nih.gov/25540998/

`[LEE_1997]` Lee T, Ng V, Gallagher R, Coldman A, McLean D. "DullRazor: a software approach to hair removal from images." Computers in Biology and Medicine. 1997. PMID `9437554`. https://pubmed.ncbi.nlm.nih.gov/9437554/

`[BARATA_2014]` Barata C, Celebi ME, Marques JS. "Improving Dermoscopy Image Classification Using Color Constancy." IEEE Journal of Biomedical and Health Informatics. DOI `10.1109/JBHI.2014.2336473`. https://doi.org/10.1109/JBHI.2014.2336473

`[DANESHJOU_2022]` Daneshjou R, et al. "Robustness of dermatology image classifiers to artifacts." Dermatology Practical & Conceptual. DOI `10.5826/dpc.1203a126`. https://doi.org/10.5826/dpc.1203a126

`[ADR-009]` "ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint." Internal project decision record. `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md`

`[ADR-010]` "ADR-010 - Final Feature Extraction Readiness Before Coding." Internal project decision record. `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md`
