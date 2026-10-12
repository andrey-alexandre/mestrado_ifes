# Stoecker et al. 2005 Blotches Reproducibility

## Scope

Primary selected method: William V. Stoecker, Kapil Gupta, R. Joe Stanley, Randy H. Moss, Bijaya Shrestha. "Detection of asymmetric blotches (asymmetric structureless areas) in dermoscopy images of malignant melanoma using relative color." Skin Research and Technology. 2005;11(3):179-184. DOI `10.1111/j.1600-0846.2005.00117.x`; PMID `15998328`. Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC3196558/

Authoritative clinical structure source for 10% rule: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018. https://dermoscopedia.org/Dermoscopic_structures

ADR status: `STRUCTURELESS-BLOTCH-001 = HUMAN_APPROVED`; ADR-010 freezes the implementation-ready Stoecker path as the absolute-color method `R < 120 AND G < 60` followed by `A_blotch > 800 pixels` on original-resolution PH2 images. ABCD structureless area is approved conditional on this faithful detector reproduction and supported 10% criterion.

Post-reproducibility human checkpoint: Stoecker et al. (2005) is implementation-ready for the fully specified absolute-color raw blotch segmentation path only. Use `R < 120 AND G < 60`, then apply the published `A_blotch > 800 pixels` threshold unchanged on original-resolution PH2 images. Classify the 800-pixel threshold as a `PUBLISHED_COMPUTATIONAL`, source-scale-specific threshold. Do not rescale, derive a PH2-specific threshold, calibrate it, resize PH2 solely to emulate the source dataset, or infer physical equivalence. The original diagnostic neural network does not need to be reproduced and must not be retrained.

The Stoecker relative-color segmentation remains scientifically documented but is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` because the primary source does not fully specify the outer extent of the background-skin sampling region. Do not invent that outer extent and do not silently substitute Celebi's skin-annulus construction into Stoecker.

PH2 rule: PH2 colors/regression/diagnosis must not tune thresholds or classifier behavior. PH2 has no direct irregular-pigmentation/blotch/structureless label in the public summary.

## Reproducibility Table

| Component | Exact implementation | Value/equation | Origin class | Primary source | Exact source location | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|
| Dataset | 512 dermoscopy images with biopsy-confirmed diagnoses | 165 melanomas, 347 benign lesions; EDRA and clinical sources; JPEG converted to TIFF; manual borders under WVS | `DATASET_DERIVED` | Stoecker et al. 2005 | Methods, lines 133-136 | Yes | Not PH2-specific. |
| Blotch clinical definition used by paper | Dark structureless areas | Black, brown, and/or gray structureless areas with symmetrical or asymmetric distribution | `CLINICAL_CONSENSUS` | Stoecker et al. 2005 citing consensus | Introduction, lines 117-118 | Yes | Does not include 10% rule. |
| Absolute color segmentation | Threshold RGB values inside lesion | `R < 120 AND G < 60` | `AUTHOR_DEFINED`, `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Absolute color blotches, lines 137-140 | Yes | Blue channel not used in final absolute rule. |
| Background skin reference | Average skin color over lesion perimeter | Omit area of 10% lesion size around lesion before sampling | `AUTHOR_DEFINED` | Stoecker et al. 2005 | Relative color blotches, lines 140-142 | Partial | Exact outer band size is not specified in this paper. |
| Nonskin exclusion for background | Exclude frames/rulers/reflections/bubbles | `R > 90`, `R > B`, `R > G`, `R < 250`, `G < 250`, `B < 250` | `AUTHOR_DEFINED` | Stoecker et al. 2005 | Relative color blotches, lines 141-142 | Yes | Used only for background sampling. |
| Relative color segmentation | Threshold against skin averages | `(R_skin - R > 140) OR ((R_skin - R > 110) AND (G_skin - G > 140))` | `AUTHOR_DEFINED`, `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Relative color blotches, lines 142-143 | Blocked | Thresholds are documented, but the background-skin sampling outer extent is unspecified; `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. |
| Globule elimination / minimum blotch size | Remove smaller dark structures | `A_blotch > 800` pixels; estimated for typical 1024 x 768 images; approximately 0.2 mm diameter | `PUBLISHED_COMPUTATIONAL`, `AUTHOR_DEFINED`, `DATASET_DERIVED` | Stoecker et al. 2005 | Globule elimination, lines 144-146 | Yes with limitation | ADR-010 authorizes using this unchanged on original-resolution PH2 images after the absolute-color rule; physical equivalence on PH2 is not guaranteed. |
| Blotch count | Connected blotch components | `n` = number of blotches | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Blotch features, lines 147-162 | Implementation-sensitive | Component-connectivity convention is not explicitly stated and must be identified before coding. |
| Eccentricity index | Scaled distance between largest blotch centroid and lesion centroid | `E = D / sqrt(A)` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Blotch features, lines 149-152, Eq. 1 | Yes | `D` Euclidean centroid distance; `A` lesion area. |
| Total blotch fraction | Sum all blotch areas over lesion area | `R = sum(B_i) / A` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Lines 152-155, Eq. 2 | Yes | `B_i` is area of blotch `i`. |
| Largest blotch fraction | Largest blotch over lesion area | `S = B_max / A` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Lines 155-158, Eq. 3 | Yes | Scale invariant. |
| Largest blotch size | Pixel area of largest blotch | `B_max` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Lines 157-158 | Yes | Pixel-scale feature. |
| Irregularity index | Perimeter-normalized largest blotch shape | `I = P_max / sqrt(B_max)` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Lines 158-162, Eq. 4 | Implementation-sensitive | Perimeter estimator is not specified and must be identified before coding. |
| Classifier inputs | Neural-network feature vector | Six inputs: `E`, `R`, `S`, `n`, `B_max`, `I` | `PUBLISHED_COMPUTATIONAL` | Stoecker et al. 2005 | Lines 163-164 | Yes | The FE outputs are directly reproducible. |
| Classifier architecture | Backpropagation neural network | `6 x 4 x 2 x 1`; train up to 10 epochs; convergence RMSE `<0.15` | `CLASSIFIER_DERIVED`, `AUTHOR_DEFINED` | Stoecker et al. 2005 | Lines 163-166 | Partial | Initialization, learning rate, and trained weights are not published. |
| Transfer functions | Neural-network activations | Sigmoid at input and hidden layers; linear at output | `AUTHOR_DEFINED` | Stoecker et al. 2005 | Lines 164-165 | Yes | Unusual "input layer" sigmoid wording should be copied cautiously. |
| Evaluation | Leave-one-out classification and ROC | Leave-one-out; ROC by sweeping NN output threshold | `CLASSIFIER_DERIVED` | Stoecker et al. 2005 | Lines 165-166 | Partial | Exact trained models are not reproducible. |
| Reported performance | Diagnostic accuracy | Relative-color method about 77% diagnostic accuracy | `DATASET_DERIVED`, `CLASSIFIER_DERIVED` | Stoecker et al. 2005 | Abstract lines 105-110; Fig. 3 lines 194-198 | Yes as reported | Not an FE threshold. |
| Clinical 10% structureless rule | Structureless area size | At least 10% of lesion total surface area | `CLINICAL_CONSENSUS` | Dermoscopedia | Lines 52-54 | Yes | Do not attribute this to Stoecker. |
| Clinical 10% blotch rule | Blotch size | At least 10% lesion surface and heavily pigmented | `CLINICAL_CONSENSUS` | Dermoscopedia | Lines 61-70 | Yes | Separate from Stoecker's 800 px computational filter. |

## Reproducibility Verdict

`REPRODUCIBLE_WITH_EXPLICIT_GAPS`

The absolute-color blotch segmentation rule and the 800-pixel minimum candidate/component rule are reproducible from the recovered primary source and are implementation-ready. The relative-color segmentation path is scientifically documented but blocked because the background-skin sampling outer extent is not specified. The original diagnostic neural-network result is not exactly reproducible because trained weights, random initialization, learning rate, and implementation details are not provided.

Explicit gaps:

- Exact outer extent for the "perimeter" background-skin sampling area is not specified, blocking the relative-color path.
- Connected-component connectivity and perimeter estimator are not specified; any dependent output must identify the convention before coding.
- `A_blotch > 800` pixels is tied to typical 1024 x 768 images. ADR-010 authorizes using it unchanged on original-resolution PH2 images after the absolute-color rule and requires explicit reporting that physical equivalence on PH2 is not guaranteed.
- Neural-network weights and training details are insufficient for exact diagnostic-classifier reproduction.

## Final Implementation-Ready Outputs

Implementation-ready Stoecker outputs:

- absolute-color candidate mask inside the lesion from `R < 120 AND G < 60`;
- absolute-color blotch mask after `A_blotch > 800 pixels` on original-resolution PH2 images;
- candidate/component pixel areas;
- `total_blotch_fraction = sum(B_i) / A`, provided the connected-component convention is explicitly identified before coding;
- `largest_blotch_fraction = B_max / A`, provided the connected-component convention is explicitly identified before coding;
- `B_max`, provided the connected-component convention is explicitly identified before coding;
- ABCD structureless-area evidence using the approved absolute-color mask plus the separately documented 10% clinical rule, with limitations.

Blocked or implementation-sensitive Stoecker outputs:

- relative-color segmentation and any relative-color-derived feature: `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`;
- background-skin-relative thresholds in execution, unless a future explicit decision defines the missing outer sampling extent;
- `n` / blotch count until connected-component connectivity is explicitly identified before coding;
- `E = D / sqrt(A)` until connected-component connectivity and centroid convention are explicitly identified before coding;
- `I = P_max / sqrt(B_max)` until connected-component connectivity and perimeter estimator are explicitly identified before coding;
- original diagnostic neural-network output.

## Clinical 10% Rule Verification

The 10% structureless/blotch rule is supported by Dermoscopedia, not by Stoecker et al. (2005). Dermoscopedia states that structureless areas should occupy at least 10% of lesion surface and that blotches are heavily pigmented areas of at least 10% lesion surface.

## Algorithm-Specific Implications

| Mapping | Resolution |
|---|---|
| ABCD D structureless areas | `RESOLVED_BY_PRIMARY_SOURCE` for the absolute-color Stoecker blotch mask plus Dermoscopedia 10% clinical rule, using `R < 120 AND G < 60` and `A_blotch > 800 pixels` unchanged with source-scale limitation. |
| 7PCL-r irregular pigmentation | `STILL_UNRESOLVED`; Stoecker provides features but no deterministic 7PCL irregular-pigmentation threshold. |
| Menzies-r color non-uniformity | `STILL_UNRESOLVED`; ADR-007 keeps blotch evidence auxiliary only. |
