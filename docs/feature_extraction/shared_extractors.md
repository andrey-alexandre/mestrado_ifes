# Shared Dermoscopic Structure Extractors

This focused stage specifies reusable low-level detectors for ABCD D, the reduced Seven-Point Checklist, and the reduced Menzies method. A shared detector means shared image-processing primitives and raw measurements only. It does not mean that final algorithm-specific clinical decisions or thresholds are equivalent.

PH2 expert annotations must never be used as feature-extraction inputs. They may be used only as validation ground truth.

## Post-Reproducibility Human Checkpoint

Authoritative updates: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md`.

Scientific method selection, reproducibility status, and implementation readiness are distinct. Earlier `PENDING_HUMAN_APPROVAL` and `READY_FOR_APPROVAL` language in this document is superseded where it conflicts with ADR-009 or ADR-010.

`RAW_EVIDENCE` may be emitted only when the mechanism producing the raw measurement is reproducible and scientifically approved. An irreproducible approximation must be `UNAVAILABLE` or `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`, not raw evidence.

Current checkpoint summary:

- pigment network: Sadeghi et al. (2011) remains scientifically preferred but is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; Anantha, Moss and Stoecker (2004) is accepted as `RAW_EVIDENCE` fallback only.
- blue-white veil: Celebi et al. (2008) remains scientifically approved; Figure 5 pixel-level tree has been transcribed, human-confirmed, and is `IMPLEMENTATION_READY` for pixel-level `veil_mask` and `S1` / veil-area fraction only. 7PCL-r and Menzies-r binary clinical mappings remain `UNRESOLVED`.
- streaks/radial structures: Sadeghi et al. (2013) remains scientifically selected but is `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.
- dots/globules: Maglogiannis and Delibasis (2015) remains selected but is `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.
- structureless/blotch: Stoecker et al. (2005) is implementation-ready for the absolute-color raw blotch path only: `R < 120 AND G < 60`, followed by `A_blotch > 800 pixels` unchanged on original-resolution PH2 images with source-scale limitation. The relative-color path is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.
- complete ABCD D: must remain `null`.

## Shared Extractor Matrix

| Shared extractor | Raw measurement | ABCD use | 7PCL use | Menzies use | Can low-level code be shared? | Can final threshold be shared? |
|---|---|---|---|---|---|---|
| Pigment network | Anantha fallback block/network texture evidence only | ABCD pigment-network presence remains unresolved | Atypical pigment network remains unresolved | Broadened network remains unresolved | Yes, through Anantha fallback only | No |
| Blue-white veil | Celebi candidate pixels, veil area fraction, component geometry, color/context features | Auxiliary to color/structure only; does not increment ABCD D | `RAW_EVIDENCE` only; binary clinical criterion unresolved | `RAW_EVIDENCE` only; binary clinical criterion unresolved | Yes, pixel detector is implementation-ready | No |
| Peripheral radial structures | None under current checkpoint | Branched streaks unresolved | Irregular streaks not implementation-ready | Radial streaming and pseudopods unresolved | Not yet | No |
| Dots/globules | None under current checkpoint | Dots/globules unresolved | Irregular dots/globules unresolved | Menzies dot/globule subtypes unresolved | Not yet | No |
| Structureless/blotch/irregular pigmentation | Stoecker absolute-color raw blotch mask and implementation-ready area evidence | ABCD structureless-area path available with documented limitations; complete D still null | Irregular pigmentation unresolved | Color non-uniformity auxiliary only | Yes, absolute-color path only | No |

## Evidence Cards

### Decision ID: `NETWORK-DETECT-001`

Decision status: Sadeghi et al. (2011) scientific preference is `HUMAN_APPROVED`; Sadeghi implementation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; Anantha, Moss and Stoecker (2004) fallback is `RAW_EVIDENCE`.

#### Proposed decision

Use Sadeghi et al. (2011) as the scientifically preferred method, but do not implement it until its reproducibility gaps are resolved. Use Anantha, Moss and Stoecker (2004) as the practical reproducible fallback for pigment-network raw evidence. Do not claim Anantha reproduces Sadeghi, resolves atypical pigment network, resolves broadened network, or supplies an ABCD pigment-network presence rule from positive blocks.

#### References

##### Reference 1

* Full citation: Murali Anantha, Randy H. Moss, William V. Stoecker. "Detection of pigment network in dermatoscopy images using texture analysis." Computerized Medical Imaging and Graphics. 2004;28(5):225-234.
* DOI/PMID: DOI `10.1016/j.compmedimag.2004.04.002`; PMID `15249068`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: Sections 2, 3, 4.2, 4.3; Figures 2-3; PubMed abstract.
* What it supports: Laws energy masks and NGLDM large-number emphasis are used for pigment-network detection on dermoscopy images; 64 x 64 blocks, Laws masks at 9 x 9, 11 x 11, 13 x 13, fully enclosed tumor blocks, NGLDM parameters `a=4`, `d=5`, NGLDM threshold `TN1=5800`, Laws thresholds `TL1=TL2=30000`, `TL3=130`, `TL4=180`; 155-image optimization and approximately 10-pixel/0.22 mm texel.
* What it does NOT support: It detects gross pigment-network presence; it explicitly does not identify network irregularities or atypical/broadened network decisions.

##### Reference 2

* Full citation: Maryam Sadeghi, Majid Razmara, Tim K. Lee, M. Stella Atkins. "A novel method for detection of pigment network in dermoscopic images using graphs." Computerized Medical Imaging and Graphics. 2011;35(2):137-143.
* DOI/PMID: DOI `10.1016/j.compmedimag.2010.07.002`; PMID `20724109`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: PubMed abstract lines 264-267.
* What it supports: Image enhancement and edge detection produce a binary edge image; the edge image is converted to a graph; cyclic subgraphs correspond to network holes; subgraphs are filtered to remove globules, dots and oil bubbles by size and color; a high-level graph uses holes as nodes; final present/absent classification uses graph density ratio; validation on 500 atlas images achieved 94.3% accuracy.
* What it does NOT support: It does not provide a clinically validated threshold for atypical pigment network or Menzies broadened network in PH2.

#### Formula

Anantha et al. compute Laws energy by convolving luminance blocks with Level/Edge masks and averaging squared filtered values. Sadeghi et al. use graph density ratio after cyclic-subgraph extraction; the abstract does not expose the exact density-ratio equation.

#### Parameters

Anantha et al.: 64 x 64 blocks; fully enclosed lesion blocks; Laws 9 x 9, 11 x 11, 13 x 13; NGLDM `a=4`, `d=5`; thresholds `TN1=5800`, `TN2=80`, `TN3=0`, `TN4=150` or `TN2=235`, `TN3=100`, `TN4=200`; Laws `TL1=TL2=30000`, `TL3=130`, `TL4=180`.

#### Threshold

`PUBLISHED_COMPUTATIONAL`, `AUTHOR_DEFINED`, `DATASET_DERIVED`: Anantha et al. thresholds were optimized for their dermoscopy image set. Sadeghi et al. density-ratio threshold is not fully available from PubMed abstract; mark implementation threshold `UNRESOLVED` unless full text is consulted.

#### Original validation

Anantha et al.: 155 dermatoscopy images; 2037 full pigment-network blocks and 4585 non-network blocks reported in the PMC text. Sadeghi et al.: 500 atlas dermoscopy images; 94.3% present/absent accuracy.

#### Generalizability

Dermoscopy-specific but not PH2-specific. Anantha block sizes assume scale near 512 x 480 images; PH2 is 768 x 560 and requires explicit scale handling or documented adoption without calibration.

#### Evidence assessment

`MODERATE`

#### Recommendation

`RAW_EVIDENCE` through Anantha fallback. Sadeghi et al. (2011) remains `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. Final ABCD pigment-network presence, 7PCL atypical network, and Menzies broadened network remain `UNRESOLVED`.

#### Remaining uncertainty

No PH2-validated threshold was identified for low-level network presence, atypical network, or broadened network.

### Decision ID: `7PCL-NETWORK-ATYPICAL-001`

Decision status: `HUMAN_APPROVED` clinical mapping, but `STILL_UNRESOLVED` / not implementation-ready.

#### Proposed decision

Use the shared network detector outputs to support atypical pigment network, but require higher-level features for irregularity: line width variability, pigmentation variability, hole area/shape heterogeneity, foci of broader/darker lines, abrupt network ending, and asymmetry.

#### References

##### Reference 1

* Full citation: Giuseppe Argenziano, Gabriella Fabbrocini, Paolo Carli, Vincenzo De Giorgi, Elena Sammarco, Mario Delfino. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Archives of Dermatology. 1998;134(12):1563-1570.
* DOI/PMID: DOI `10.1001/archderm.134.12.1563`; PMID `9875194`.
* Evidence role: `CLINICAL_DEFINITION`, `DATASET_VALIDATION`.
* Evidence location: Table 2 lines 230-233; Table 3 lines 247-257; Table 4 lines 266-299.
* What it supports: Atypical pigment network is a major 7PCL criterion; definition is prominent/broad and irregular network; total 7PCL score validation on 342 lesions reported sensitivity 95% and specificity 75%.
* What it does NOT support: It does not give an image-processing formula, feature threshold, or PH2-specific validation.

##### Reference 2

* Full citation: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018.
* DOI/PMID: URL `https://dermoscopedia.org/Dermoscopic_structures`.
* Evidence role: `CLINICAL_DEFINITION`, `LIMITATION`.
* Evidence location: Pigment Network section lines 27-43.
* What it supports: Typical network is regular, uniform in width and color, and fades at the periphery; atypical network has irregular mesh, variable line width/pigmentation, heterogeneous holes, broader/darker foci, abrupt peripheral ending, and may break into branched streaks.
* What it does NOT support: It does not provide computational thresholds.

#### Formula

No sufficient supporting reference identified for a published deterministic formula that converts low-level network measurements to 7PCL atypical network.

#### Parameters

Candidate raw measurements: line thickness coefficient of variation, hole area/shape coefficient of variation, line color standard deviation, network-area asymmetry. Parameter values are `UNRESOLVED`.

#### Threshold

`UNRESOLVED`

#### Original validation

Clinical 7PCL validation: Argenziano et al. 342 lesions. No PH2-specific computational validation identified for this higher-level decision.

#### Generalizability

Clinical definition is general. Higher-level computational interpretation remains unresolved.

#### Evidence assessment

`MODERATE` for clinical definition, `INSUFFICIENT` for deterministic threshold.

#### Recommendation

`UNRESOLVED`

#### Remaining uncertainty

Need a thresholded mapping from network raw features to "prominent/broad and irregular" network.

### Decision ID: `MENZIES-NETWORK-BROADENED-001`

Decision status: `HUMAN_APPROVED` scientific criterion, but `STILL_UNRESOLVED` / not implementation-ready.

#### Proposed decision

Use the shared network detector only for raw network-line/mesh measurements. Do not equate Menzies broadened network with 7PCL atypical pigment network. A broadened-network decision requires focal increase in cord/grid width.

#### References

##### Reference 1

* Full citation: Ralph P. Braun, Scott Menzies. "Menzies Method." Dermoscopedia. Status update 2017.
* DOI/PMID: URL `https://dermoscopedia.org/Menzies_Method`.
* Evidence role: `CLINICAL_DEFINITION`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: Lines 64-66.
* What it supports: Broadened network is an increase in the width of grids or cords of the pigmented net, usually focal in melanoma; reported 35% frequency and 86% specificity for invasive melanoma.
* What it does NOT support: It does not define a numeric line-width threshold.

#### Formula

No sufficient supporting reference identified.

#### Parameters

Candidate parameters: line-width map and focal-width increase relative to the lesion's own network width. Parameter values are `UNRESOLVED`.

#### Threshold

`UNRESOLVED`

#### Original validation

Menzies method original training/test design is summarized by Braun and Menzies and in Menzies et al. (1996): training set 62 invasive melanomas and 159 atypical nonmelanomas; test set 45 invasive melanomas and 119 atypical nonmelanomas; sensitivity 92%, specificity 71%.

#### Generalizability

Clinical criterion is general. Quantitative operationalization is unresolved.

#### Evidence assessment

`INSUFFICIENT`

#### Recommendation

`UNRESOLVED`

#### Remaining uncertainty

Need published width threshold or an approved validation plan.

### Decision ID: `VEIL-DETECT-001`

Decision status: `HUMAN_APPROVED` scientific method; Figure 5 pixel detector `IMPLEMENTATION_READY`; `veil_mask` and `S1` / veil-area fraction are implementation-ready `RAW_EVIDENCE`; 7PCL-r and Menzies-r clinical binary mappings remain `UNRESOLVED`.

#### Proposed decision

Use Celebi et al. (2008) as the primary low-level blue-white veil detector: compute the required contextual color features, apply the published Figure 5 C4.5 pixel decision tree, apply the published 5 x 5 majority-filter postprocessing, and store candidate `veil_mask`, `S1` / veil-area fraction, component geometry, and confound flags. Do not retrain C4.5.

#### References

##### Reference 1

* Full citation: M. Emre Celebi, Hitoshi Iyatomi, William V. Stoecker, Randy H. Moss, Harold S. Rabinovitz, Giuseppe Argenziano, H. Peter Soyer. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." Computerized Medical Imaging and Graphics. 2008;32(8):670-677.
* DOI/PMID: DOI `10.1016/j.compmedimag.2008.08.003`; PMID `18804955`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: PMC Section 4 and Figures 5 and 8; results paragraph below Figure 8.
* What it supports: Pixel classification uses a C4.5 decision tree with features from 5 x 5 neighborhoods; pruning confidence `C=0.1`, minimum leaf samples `M=100`; induced tree used absolute blue chromaticity `F3` and relative red difference `F10`; Figure 5 resolves the pixel-level veil rule; pixel test sensitivity 84.33% and specificity 96.19%; lesion tree used blue-white area fraction `S1`, circularity `S2`, ellipticity `S3`; whole-set validation on 545 images reported sensitivity 69.35%, specificity 89.97%, accuracy 82.94%.
* What it does NOT support: It does not establish a universal binary clinical threshold or a deterministic lesion-level 7PCL-r/Menzies-r blue-white-veil presence mapping. The lesion-level `S1 < 0.009` / `S3 > 0.979` tree is a melanoma/benign diagnostic classifier, not a veil-presence rule.

##### Reference 2

* Full citation: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018.
* DOI/PMID: URL `https://dermoscopedia.org/Dermoscopic_structures`.
* Evidence role: `CLINICAL_DEFINITION`, `LIMITATION`.
* Evidence location: Blue-White Veil section lines 145-148.
* What it supports: Blue-white veil is confluent blue pigmentation with an overlying white ground-glass haze; in melanoma it is focal, ill-defined, and nonuniform; distinction from regression may require clinical palpability information unavailable in PH2 images.
* What it does NOT support: It does not provide computational thresholds.

#### Formula

`veil_fraction = area(veil_candidate_pixels) / area(valid_lesion_pixels)`.

#### Parameters

5 x 5 contextual neighborhood; C4.5 parameters `C=0.1`, `M=100`; lesion-level features `S1`, `S2`, `S3`.

#### Threshold

`CLASSIFIER_DERIVED`, `PUBLISHED_COMPUTATIONAL`: Figure 5 pixel tree: `F3 <= 0.3 -> Non-Veil`; otherwise `F10 > -51 -> Non-Veil`; otherwise `F10 <= -194 -> Non-Veil`; otherwise `Veil`. Equivalent positive interval: `F3 > 0.3 AND -194 < F10 <= -51`. The lesion-level tree, `S1 < 0.009 -> benign`; otherwise `S3 > 0.979 -> benign`; otherwise melanoma, is not a blue-white-veil presence threshold. Do not use `S1 = 0.009` or `S1 > 0` as a clinical veil-presence rule.

#### Original validation

Celebi et al.: 545 dermoscopy images; pixel test sensitivity 84.33%, specificity 96.19%; lesion-level sensitivity 69.35%, specificity 89.97%, accuracy 82.94%.

#### Generalizability

Dermoscopy-specific; not PH2-specific. Figure 5 pixel-level classifier thresholds are human-approved for implementation; no lesion-level 7PCL-r or Menzies-r clinical binary mapping is approved.

#### Evidence assessment

`STRONG` for low-level raw detector candidate; `MODERATE` for binary decision if classifier-derived thresholds are allowed.

#### Recommendation

`IMPLEMENTATION_READY` for the Celebi pixel-level detector, `veil_mask`, and `S1` / veil-area fraction as `RAW_EVIDENCE`. `UNRESOLVED` for 7PCL-r and Menzies-r binary clinical veil criteria.

#### Remaining uncertainty

PH2 has blue-whitish veil present/absent labels but no pixel-level veil masks in the public summary.

### Decision ID: `STREAK-DETECT-001`

Decision status: `HUMAN_APPROVED` scientific method; `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

#### Proposed decision

Use a reusable peripheral radial-structure detector based on Sadeghi et al. (2013): detect valid streak lines, build a graphical representation, compute geometry/distribution/coverage/orientation features, and store raw measures for later algorithm-specific interpretation.

#### References

##### Reference 1

* Full citation: Maryam Sadeghi, Tim K. Lee, David McLean, Harvey Lui, M. Stella Atkins. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." IEEE Transactions on Medical Imaging. 2013;32(5):849-861.
* DOI/PMID: DOI `10.1109/TMI.2013.2239307`; PMID `23335664`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: PubMed abstract.
* What it supports: Method identifies absence/presence of streaks, performs three-way absent/regular/irregular classification, analyzes detected line orientation, models valid streaks and their distribution/coverage with graphical representation, and combines valid-streak features with lesion color/texture features. Validation on 945 images achieved 76.1% accuracy and weighted AUC 85% for absent/regular/irregular; absent/present accuracy 78.3%, AUC 83.2%; regular/irregular accuracy 83.6%, AUC 88.9%.
* What it does NOT support: It does not provide a PH2-specific threshold or separate validated rules for ABCD branched streaks, Menzies radial streaming, and Menzies pseudopods.

##### Reference 2

* Full citation: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018.
* DOI/PMID: URL `https://dermoscopedia.org/Dermoscopic_structures`.
* Evidence role: `CLINICAL_DEFINITION`, `LIMITATION`.
* Evidence location: Streaks section lines 95-102.
* What it supports: Streaks are peripheral linear pigmented projections; pseudopods have bulbous tips; radial streaming lacks bulbous tips; melanoma streaks are focal/asymmetric, while symmetric circumferential streaks suggest Spitz/Reed nevi; pseudopods must be connected to the main tumor mass and have a knob larger than the originating network.
* What it does NOT support: It does not provide a computational detector threshold.

##### Reference 3

* Full citation: Giuseppe Argenziano, Gabriella Fabbrocini, Paolo Carli, Vincenzo De Giorgi, Elena Sammarco, Mario Delfino. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Archives of Dermatology. 1998;134(12):1563-1570.
* DOI/PMID: DOI `10.1001/archderm.134.12.1563`; PMID `9875194`.
* Evidence role: `CLINICAL_DEFINITION`, `DATASET_VALIDATION`.
* Evidence location: Table 2 lines 230-237; Table 3 lines 247-257.
* What it supports: 7PCL streaks are radially and asymmetrically arranged linear or bulbous extensions at the lesion edge, scored as a minor criterion.
* What it does NOT support: It does not separate a deterministic computational rule for radial streaming versus pseudopods.

#### Formula

No exact public formula was recovered from the accessible abstract. Store radiality as a candidate raw measurement: angular difference between the local candidate-line direction and radial vector from lesion centroid; do not treat this as a sourced final threshold.

#### Parameters

Peripheral band width, line detector scale, graph construction, length/width filters, angular coverage, terminal bulb rule are `UNRESOLVED`.

#### Threshold

`CLASSIFIER_DERIVED`, `PUBLISHED_COMPUTATIONAL` for Sadeghi et al. classifier outputs; exact low-level thresholds are `UNRESOLVED` from accessible evidence.

#### Original validation

Sadeghi et al.: 945 images from atlases and internet, plus 300-image cleaned subset; metrics above.

#### Generalizability

Dermoscopy-specific but not PH2-specific. Classifier-derived absent/regular/irregular decision may be proposed later, but subtype thresholds are not shared.

#### Evidence assessment

`MODERATE`

#### Recommendation

`UNAVAILABLE` for current implementation. Do not reconstruct missing LoG scales, invent thresholds, retrain SimpleLogistic, or approximate the method and call it Sadeghi et al. (2013).

#### Remaining uncertainty

No published PH2-specific peripheral-band width, line length, angular focality, terminal bulb, or branching threshold was identified.

### Decision ID: `BLOB-DETECT-001`

Decision status: `HUMAN_APPROVED_PENDING_SOURCE_DETAIL`; current implementation state `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

#### Proposed decision

Use a reusable dots/globules detector that produces candidate dark/blue-gray/brown/black round components and raw spatial features. Maglogiannis and Delibasis (2015) is the primary computational candidate, but implementation thresholds remain unresolved unless full method details are available.

#### References

##### Reference 1

* Full citation: Ilias Maglogiannis, Konstantinos K. Delibasis. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." Computer Methods and Programs in Biomedicine. 2015;118(2):124-133.
* DOI/PMID: DOI `10.1016/j.cmpb.2014.12.001`; PMID `25540998`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: PubMed abstract.
* What it supports: A methodology detects, segments, and counts dark dots and globules from dermoscopy images; segmentation uses a multi-resolution approach based on inverse nonlinear diffusion; features extracted from segmented dots/globules improved melanoma/non-malignant nevus classification when combined with other region descriptors.
* What it does NOT support: The abstract does not provide exact size, circularity, color, or threshold parameters.

##### Reference 2

* Full citation: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018.
* DOI/PMID: URL `https://dermoscopedia.org/Dermoscopic_structures`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Dots section lines 73-85.
* What it supports: Dots are round structures less than 0.1 mm; melanin-related dots may be black, brown, or blue-gray; irregular dots vary in size, shape and color and are often peripheral or scattered.
* What it does NOT support: It does not provide an image-processing threshold beyond the clinical size boundary.

##### Reference 3

* Full citation: Giuseppe Argenziano, Gabriella Fabbrocini, Paolo Carli, Vincenzo De Giorgi, Elena Sammarco, Mario Delfino. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Archives of Dermatology. 1998;134(12):1563-1570.
* DOI/PMID: DOI `10.1001/archderm.134.12.1563`; PMID `9875194`.
* Evidence role: `CLINICAL_DEFINITION`, `DATASET_VALIDATION`.
* Evidence location: Table 2 lines 230-237; Table 3 lines 247-257.
* What it supports: 7PCL irregular dots/globules are black, brown, or blue round structures irregularly distributed within the lesion and form a minor criterion.
* What it does NOT support: It does not distinguish computational dot/globule thresholds.

#### Formula

`circularity = 4*pi*area/perimeter^2`; `peripheral_fraction = count(peripheral_blobs)/count(all_blobs)`. These are standard raw features but no source above establishes their final clinical thresholds for this project.

#### Parameters

Dot diameter `<0.1 mm` is a clinical definition from Dermoscopedia. Globule diameter `>0.1 mm` is described by Dermoscopedia in the Globules section. Pixel conversion requires PH2 scale metadata or an approved image-scale assumption; therefore PH2 pixel threshold is `UNRESOLVED`.

#### Threshold

`CLINICAL_CONSENSUS` for dot/globule size boundary around 0.1 mm as a clinical descriptor. `UNRESOLVED` for pixel diameter, circularity, eccentricity, count, color, and peripheral-band thresholds.

#### Original validation

Maglogiannis and Delibasis used dermoscopy images with histopathology; accessible abstract does not provide sample size. PH2 offers dots/globules labels as absent/atypical/typical.

#### Generalizability

Dermoscopy-specific; low-level blob extraction can be shared. Final ABCD, 7PCL, and Menzies decisions cannot share thresholds.

#### Evidence assessment

`MODERATE` for raw detector concept; `INSUFFICIENT` for reproducible thresholded implementation.

#### Recommendation

`UNAVAILABLE` for deterministic or raw FE until a separately reproducible approved extractor is selected. Do not substitute LoG/DoG/OpenCV blob detection without a new methodological decision.

#### Remaining uncertainty

Need reproducible implementation details for Maglogiannis and Delibasis or an approved alternative primary source.

### Decision ID: `STRUCTURELESS-BLOTCH-001`

Decision status: `HUMAN_APPROVED` for Stoecker absolute-color raw blotch segmentation and approved ABCD structureless-area use. Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`.

#### Proposed decision

Use a reusable structureless/blotch/irregular-pigmentation extractor based on Stoecker's absolute-color dark component segmentation: `R < 120 AND G < 60`, followed by `A_blotch > 800 pixels`. Store only outputs that can be produced without unresolved implementation conventions. Do not equate ABCD structureless areas, 7PCL irregular pigmentation, and Menzies color non-uniformity.

#### References

##### Reference 1

* Full citation: William V. Stoecker, Kapil Gupta, R. Joe Stanley, Randy H. Moss, Bijaya Shrestha. "Detection of asymmetric blotches (asymmetric structureless areas) in dermoscopy images of malignant melanoma using relative color." Skin Research and Technology. 2005;11(3):179-184.
* DOI/PMID: DOI `10.1111/j.1600-0846.2005.00117.x`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: Wiley abstract Methods/Results.
* What it supports: Blotches are detected automatically using thresholds in red and green color planes with absolute and relative color; the fully specified absolute-color path uses `R < 120 AND G < 60`; computed features include scaled distance between largest blotch centroid and lesion centroid, total blotch-area ratio, largest-blotch ratio, blotch count, largest-blotch size, and largest-blotch irregularity; 512-image validation contained 165 melanomas and 347 benign lesions; relative-color features achieved about 77% diagnostic accuracy.
* What it does NOT support: It does not provide a final 7PCL irregular-pigmentation binary threshold. The relative-color path does not fully specify the outer extent of the background-skin sampling region.

##### Reference 2

* Full citation: Ralph P. Braun, Katrin Kerl. "Dermoscopic structures." Dermoscopedia. Status update 2018.
* DOI/PMID: URL `https://dermoscopedia.org/Dermoscopic_structures`.
* Evidence role: `CLINICAL_DEFINITION`, `THRESHOLD`.
* Evidence location: Structureless Areas and Blotches sections lines 52-70.
* What it supports: Structureless areas lack network or other structures and should occupy at least 10% of lesion surface; blotches are heavily pigmented structureless areas of at least 10% of lesion surface; irregular blotches are asymmetric, off-center, irregular-contoured, and/or heterogeneous in dark hue.
* What it does NOT support: It does not provide image-processing color thresholds.

##### Reference 3

* Full citation: Giuseppe Argenziano, Gabriella Fabbrocini, Paolo Carli, Vincenzo De Giorgi, Elena Sammarco, Mario Delfino. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Archives of Dermatology. 1998;134(12):1563-1570.
* DOI/PMID: DOI `10.1001/archderm.134.12.1563`; PMID `9875194`.
* Evidence role: `CLINICAL_DEFINITION`, `DATASET_VALIDATION`.
* Evidence location: Table 2 lines 230-237; Table 3 lines 247-257.
* What it supports: 7PCL irregular diffuse pigmentation/blotches are brown, gray, and black diffuse pigmentation with irregular shape or distribution and abrupt end; it is a minor criterion.
* What it does NOT support: It does not define deterministic thresholds.

#### Formula

Implementation-ready segmentation formula: `R < 120 AND G < 60` inside the lesion, followed by `A_blotch > 800 pixels`. Area measurements may include `total_blotch_fraction = sum(area(blotch_components))/area(lesion)` and `largest_blotch_fraction = area(largest_blotch)/area(lesion)` once the connected-component convention is explicitly identified before coding. `eccentricity_raw = distance(centroid_largest_blotch, centroid_lesion)/sqrt(area(lesion))` and `irregularity_raw = perimeter(largest_blotch)/sqrt(area(largest_blotch))` are implementation-sensitive until centroid/connectivity/perimeter conventions are identified.

#### Parameters

Clinical structureless/blotch area threshold `>=10%` of lesion surface from Dermoscopedia. Use Stoecker et al. (2005) absolute-color threshold `R < 120 AND G < 60` followed by `A_blotch > 800 pixels` unchanged on original-resolution PH2 images as a `PUBLISHED_COMPUTATIONAL`, source-scale-specific threshold. Its physical equivalence on PH2 is not guaranteed.

#### Threshold

`CLINICAL_CONSENSUS` for 10% structureless/blotch area clinical descriptor. `PUBLISHED_COMPUTATIONAL` and source-scale-specific for Stoecker `R < 120 AND G < 60` and `A_blotch > 800 pixels`. Stoecker relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` because the background-skin sampling outer extent is unspecified. Final binary 7PCL/Menzies thresholds are `UNRESOLVED`.

#### Original validation

Stoecker et al.: 512 images, 165 melanomas and 347 benign lesions; relative-color blotch method gave about 77% diagnostic accuracy.

#### Generalizability

Dermoscopy-specific; not PH2-specific. PH2 has color and regression labels but not a direct irregular-pigmentation label.

#### Evidence assessment

`MODERATE` for absolute-color raw blotch/structureless evidence; `INSUFFICIENT` for relative-color implementation readiness and final algorithm-specific thresholds.

#### Recommendation

Implementation-ready as `RAW_EVIDENCE` for the Stoecker absolute-color blotch mask, implementation-ready area evidence, and approved ABCD structureless-area use with documented limitations. Relative-color segmentation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. `UNRESOLVED` for 7PCL irregular pigmentation and Menzies color non-uniformity mapping.

#### Remaining uncertainty

Before coding any feature that depends on connected components, centroids, or perimeter, explicitly identify the connectivity, centroid, and perimeter-estimation conventions. Do not silently choose library defaults.

## Algorithm-Specific Interpretation Summary

| Decision ID | Algorithm-specific criterion | Recommendation | Main reason |
|---|---|---|---|
| `ABCD-NETWORK-PRESENCE-001` | ABCD pigment network presence | `UNRESOLVED`; Anantha fallback may emit raw evidence only | Sadeghi remains implementation-blocked; Anantha does not establish ABCD lesion-level presence. |
| `7PCL-NETWORK-ATYPICAL-001` | 7PCL atypical pigment network | `UNRESOLVED` | Clinical definition is clear in Argenziano et al. and Dermoscopedia; deterministic threshold is missing. |
| `MENZIES-NETWORK-BROADENED-001` | Menzies broadened network | `UNRESOLVED` | Braun and Menzies define focal cord/grid broadening; numeric width threshold is missing. |
| `7PCL-VEIL-001` | 7PCL blue-whitish veil | pixel detector `IMPLEMENTATION_READY`; binary clinical criterion `UNRESOLVED` | Celebi Figure 5 tree has been transcribed and human-confirmed for pixel-level evidence only. |
| `MENZIES-VEIL-001` | Menzies blue-white veil | pixel detector `IMPLEMENTATION_READY`; binary clinical criterion `UNRESOLVED` | Same Celebi pixel-level evidence as 7PCL veil; no lesion-level checklist mapping approved. |
| `ABCD-STREAKS-001` | ABCD branched streaks | `UNRESOLVED` | Low-level radial structures can be detected, but branched streak rule is not separately thresholded. |
| `7PCL-STREAKS-001` | 7PCL irregular streaks | `UNAVAILABLE` / `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY` | Sadeghi et al. is approved with limitation, but missing source details block implementation. |
| `MENZIES-RADIAL-STREAMING-001` | Menzies radial streaming | `UNRESOLVED` | Clinical linear subtype is distinct; numeric threshold is missing. |
| `MENZIES-PSEUDOPOD-001` | Menzies pseudopods | `UNRESOLVED` | Clinical bulb/stalk rule is distinct; numeric terminal morphology threshold is missing. |
| `ABCD-DOTS-GLOBULES-001` | ABCD dots and globules | `UNAVAILABLE` | Maglogiannis method is implementation-blocked and no substitute detector is approved. |
| `7PCL-DOTS-GLOBULES-001` | 7PCL irregular dots/globules | `UNRESOLVED` | Argenziano et al. define irregular distribution; reproducible thresholds are missing. |
| `MENZIES-DOTS-001` | Menzies multiple brown/blue-gray dots and peripheral black dots/globules | `UNRESOLVED` | Braun and Menzies define color/spatial constraints; exact count/peripheral thresholds are missing. |
| `ABCD-STRUCTURELESS-001` | ABCD structureless areas | `RESOLVED_BY_PRIMARY_SOURCE` for Stoecker absolute-color raw evidence plus 10% clinical rule, with limitations | Absolute-color segmentation is reproducible; `R < 120 AND G < 60` and `A_blotch > 800 pixels` are transferred unchanged as source-scale-specific. |
| `7PCL-IRREGULAR-PIGMENTATION-001` | 7PCL irregular pigmentation/blotches | `UNRESOLVED` | Argenziano et al. define shape/distribution/abrupt end; color segmentation thresholds unresolved. |
| `MENZIES-COLOR-NONUNIFORMITY-001` | Menzies color non-uniformity support | `UNRESOLVED` | Menzies color rule was approved separately for color count; blotch/structureless detector is only auxiliary. |

## PH2 Validation Availability

| Extractor | Direct PH2 ground truth | Partial/related PH2 ground truth | No appropriate PH2 ground truth |
|---|---|---|---|
| Pigment network | Pigment Network: `AT` atypical, `T` typical | Lesion mask | Low-level line/hole masks are absent |
| Blue-white veil | Blue Whitish Veil: `A` absent, `P` present | Color labels may support analysis | Pixel-level veil masks are not described in PH2 public summary |
| Streaks/radial structures | Streaks: `A` absent, `P` present | Lesion mask | Regular/irregular, branched streaks, radial streaming, pseudopods are not separately annotated |
| Dots/globules | Dots/Globules: `A` absent, `AT` atypical, `T` typical | Colors and lesion mask | Separate dot vs globule masks, blue-gray dots, peripheral black dots/globules are not separately annotated |
| Structureless/blotch/irregular pigmentation | None as named structure in PH2 public summary | Color segmentation, regression areas, lesion mask | Irregular pigmentation, blotch, and structureless areas are not separately annotated |

## Bibliographic References

`[ANANTHA_2004]` Anantha M, Moss RH, Stoecker WV. "Detection of pigment network in dermatoscopy images using texture analysis." Computerized Medical Imaging and Graphics. 2004;28(5):225-234. DOI `10.1016/j.compmedimag.2004.04.002`; PMID `15249068`. https://pubmed.ncbi.nlm.nih.gov/15249068/

`[SADEGHI_PN_2011]` Sadeghi M, Razmara M, Lee TK, Atkins MS. "A novel method for detection of pigment network in dermoscopic images using graphs." Computerized Medical Imaging and Graphics. 2011;35(2):137-143. DOI `10.1016/j.compmedimag.2010.07.002`; PMID `20724109`. https://pubmed.ncbi.nlm.nih.gov/20724109/

`[CELEBI_2008]` Celebi ME, Iyatomi H, Stoecker WV, Moss RH, Rabinovitz HS, Argenziano G, Soyer HP. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." Computerized Medical Imaging and Graphics. 2008;32(8):670-677. DOI `10.1016/j.compmedimag.2008.08.003`; PMID `18804955`. https://pubmed.ncbi.nlm.nih.gov/18804955/

`[SADEGHI_STREAKS_2013]` Sadeghi M, Lee TK, McLean D, Lui H, Atkins MS. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." IEEE Transactions on Medical Imaging. 2013;32(5):849-861. DOI `10.1109/TMI.2013.2239307`; PMID `23335664`. https://pubmed.ncbi.nlm.nih.gov/23335664/

`[MAGLOGIANNIS_2015]` Maglogiannis I, Delibasis KK. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." Computer Methods and Programs in Biomedicine. 2015;118(2):124-133. DOI `10.1016/j.cmpb.2014.12.001`; PMID `25540998`. https://pubmed.ncbi.nlm.nih.gov/25540998/

`[STOECKER_2005]` Stoecker WV, Gupta K, Stanley RJ, Moss RH, Shrestha B. "Detection of asymmetric blotches (asymmetric structureless areas) in dermoscopy images of malignant melanoma using relative color." Skin Research and Technology. 2005;11(3):179-184. DOI `10.1111/j.1600-0846.2005.00117.x`. https://doi.org/10.1111/j.1600-0846.2005.00117.x

`[ARGENZIANO_1998]` Argenziano G, Fabbrocini G, Carli P, De Giorgi V, Sammarco E, Delfino M. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Archives of Dermatology. 1998;134(12):1563-1570. DOI `10.1001/archderm.134.12.1563`; PMID `9875194`. https://jamanetwork.com/journals/jamadermatology/fullarticle/189703

`[MENZIES_1996]` Menzies SW, Ingvar C, Crotty KA, McCarthy WH. "Frequency and morphologic characteristics of invasive melanomas lacking specific surface microscopic features." Archives of Dermatology. 1996;132(10):1178-1182. PMID `8859028`. https://pubmed.ncbi.nlm.nih.gov/8859028/

`[BRAUN_KERL_STRUCTURES_2018]` Braun RP, Kerl K. "Dermoscopic structures." Dermoscopedia. Status update 2018. https://dermoscopedia.org/Dermoscopic_structures

`[BRAUN_MENZIES_2017]` Braun RP, Menzies SW. "Menzies Method." Dermoscopedia. Status update 2017. https://dermoscopedia.org/Menzies_Method

`[PH2_2013]` Mendonca T, Ferreira PM, Marques JS, Marcal ARS, Rozeira J. "PH2 - A dermoscopic image database for research and benchmarking." 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society. 2013:5437-5440. DOI `10.1109/EMBC.2013.6610779`; PMID `24110966`. https://pubmed.ncbi.nlm.nih.gov/24110966/

`[PH2_DATABASE]` ADDI Project. "PH2 Database." Universidade do Porto. https://www.fc.up.pt/addi/ph2%20database.html
