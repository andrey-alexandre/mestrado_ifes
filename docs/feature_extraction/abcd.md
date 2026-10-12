# ABCD Feature Extraction Research

## Compatibility with the Current Project

The current pipeline sends a masked/segmented dermoscopic image to `ABCDDiagnosticAgent`, then stores a structured Pydantic answer with free-text fields for `assimetria`, `bordas`, `cor`, `estruturas`, and `explicacao`. The future feature-extraction stage should therefore produce both raw deterministic measurements and the ABCD-facing scores that can be inserted into a minimally changed prompt.

For PH2, the compatible inputs are the original BMP image and the expert lesion mask returned by `PH2Dataset` as `image_path` and `seg_path`. Feature extraction should run before VLM calls and should not require the multiagent graph.

## Focused Evidence Cards for ABCD A/B/C

### Decision ID: `ABCD-A-AXIS-001`

#### Proposed Decision

Use the lesion-mask center of gravity as the center, estimate the principal symmetry axis from second-order central image moments, rotate the mask by `-phi`, and use the rotated horizontal/vertical axes as the two perpendicular axes for ABCD A computation.

#### References

##### Reference 1

* Full citation: Ishika Muradia, Niti Khunger, Amit Kumar Yadav. "Real-time Intraoperative Dermoscopic Monitoring of Common Acquired Melanocytic Nevi Following Shave Excision." Journal of Cutaneous and Aesthetic Surgery, 2022.
* DOI/PMID: DOI `10.4103/JCAS.JCAS_174_21`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Table 1, lines 747-752 in the online article.
* What it supports: The first asymmetry axis should attempt to bisect the lesion at the division of "most symmetry"; the second axis is placed at 90 degrees; contour, colors, and structures count for symmetry.
* What it does NOT support: It does not provide a computational search, moment formula, or numeric threshold.

##### Reference 2

* Full citation: Rachid Kasmi, Kamel Mokrani. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing, 2016, 10(6), 448-455.
* DOI/PMID: DOI `10.1049/iet-ipr.2015.0385`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`.
* Evidence location: Section 4.1 "Asymmetry", equations 7-10, Figures 5-7; Sections 4.1.1-4.1.3; Section 5; Tables 3-4.
* What it supports: The center of gravity is used; the first principal axis crosses the center of gravity with angle `phi` computed from second-order moments `m11`, `m20`, and `m02`; the lesion is rotated by `-phi`; color symmetry is measured in CIELAB blocks of 20 x 20 pixels with at least 75% lesion coverage and Delta E; brightness and shape symmetry thresholds are specified; A is converted to 0/1/2 from the number of symmetric axes.
* What it does NOT support: It does not implement the clinical "most symmetry" axis by exhaustive angular minimization; it uses an EDRA dataset, not PH2; its thresholds are author-defined/dataset-tested rather than PH2-specific.

#### Evidence Assessment

`MODERATE`

#### Generalizability

The clinical definition is general. Kasmi & Mokrani is a computational ABCD implementation validated on 200 EDRA images, not PH2-specific. The moment-axis and block/threshold choices are method-specific.

#### Recommendation

`HUMAN_APPROVED`. Adopt Kasmi and Mokrani (2016) for computational axis selection. The exact clinical "most symmetry" search remains a limitation, but no PH2 calibration will be performed.

#### Remaining Uncertainty

No sufficient supporting reference identified for an exhaustive angular search that explicitly minimizes ABCD asymmetry and then maps that search result to the clinical A score. PH2 asymmetry annotations, if used, are validation ground truth only and must never be FE inputs.

### Decision ID: `ABCD-A-SCORE-001`

#### Proposed Decision

For each axis, classify it as symmetric only if it is simultaneously color-symmetric, brightness-symmetric, and shape-symmetric according to Kasmi & Mokrani; set `A=0` if both axes are symmetric, `A=1` if exactly one axis is symmetric, and `A=2` otherwise.

#### References

##### Reference 1

* Full citation: Rachid Kasmi, Kamel Mokrani. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing, 2016, 10(6), 448-455.
* DOI/PMID: DOI `10.1049/iet-ipr.2015.0385`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`.
* Evidence location: Sections 4.1.1-4.1.3; Section 6; Tables 3-4.
* What it supports: Color symmetry uses CIELAB Delta E between mirrored 20 x 20 lesion blocks and JND threshold `Delta E <= 6`; an axis is color-symmetric when symmetric block pairs outnumber asymmetric block pairs. Brightness asymmetry uses difference between average luminance of opposite halves with threshold 3% of total average lesion luminance. Shape asymmetry uses area difference across the axis with threshold 2% of lesion area. The final A score is 0, 1, or 2 from axis symmetry.
* What it does NOT support: It does not prove the thresholds are universal or PH2-calibrated.

##### Reference 2

* Full citation: Michael Kunz, Wilhelm Stolz. "ABCD rule." Dermoscopedia, last edited 2023.
* DOI/PMID: URL `https://dermoscopedia.org/ABCD_rule`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Asymmetry section, lines 26-27.
* What it supports: ABCD A is assigned from the number of asymmetric axes, considering contour, colors, and structures.
* What it does NOT support: It does not provide continuous-to-binary thresholds.

#### Evidence Assessment

`MODERATE`

#### Generalizability

Classifier-independent but method-specific; thresholds are EDRA-tested, not PH2-specific.

#### Recommendation

`HUMAN_APPROVED`. Adopt Kasmi and Mokrani (2016) for ABCD A color, brightness, and shape symmetry thresholds and `A=0/1/2` conversion. The thresholds are published computational thresholds and are not claimed to be universal clinical thresholds. No PH2 calibration will be performed.

#### Remaining Uncertainty

No PH2-specific validation of the 2% shape threshold, 3% brightness threshold, or Delta E 6 majority rule was identified; this is documented as a methodological limitation accepted by human decision.

### Decision ID: `ABCD-B-OCTANTS-001`

#### Proposed Decision

Construct eight 45-degree sectors around the lesion center and count one point per sector only when a published abrupt/irregular border criterion is positive.

#### References

##### Reference 1

* Full citation: Michael Kunz, Wilhelm Stolz. "ABCD rule." Dermoscopedia, last edited 2023.
* DOI/PMID: URL `https://dermoscopedia.org/ABCD_rule`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Border section, lines 28-30.
* What it supports: ABCD B is the number of eight lesion segments with sharp abrupt ending of pigment pattern versus gradual indistinct cutoff.
* What it does NOT support: It does not define center, initial sector orientation, color space, ring width, or numeric abruptness threshold.

##### Reference 2

* Full citation: Ishika Muradia, Niti Khunger, Amit Kumar Yadav. "Real-time Intraoperative Dermoscopic Monitoring of Common Acquired Melanocytic Nevi Following Shave Excision." Journal of Cutaneous and Aesthetic Surgery, 2022.
* DOI/PMID: DOI `10.4103/JCAS.JCAS_174_21`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Table 1, lines 747-752.
* What it supports: B score counts the number of eight pie-shaped pieces with abrupt perimeter cutoff, range 0-8.
* What it does NOT support: It does not provide an image-processing implementation.

##### Reference 3

* Full citation: Rachid Kasmi, Kamel Mokrani. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing, 2016, 10(6), 448-455.
* DOI/PMID: DOI `10.1049/iet-ipr.2015.0385`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`.
* Evidence location: Section 4.2 "Border", equations 10-12, Figure 8; Section 5; Tables 3-4.
* What it supports: The lesion is divided into eight equal octants of 45 degrees; each sub-contour is approximated by a third-order spline; if fitting error exceeds `0.05 x length(y)` for the sub-contour, that sub-contour receives one point; total B ranges 0-8. Validation used 200 EDRA dermoscopic images.
* What it does NOT support: It measures sub-contour irregularity and does not directly measure lesion-to-skin pigment abruptness by color transition. It does not fully specify the executable meaning of `y` or `length(y)`, principal-axis border-sector alignment, ordered parametric 2D contour representation, Euclidean residual, dimensionless smoothing, or `L_smooth` normalization.

#### Evidence Assessment

`MODERATE`

#### Generalizability

The eight-sector scoring definition is clinical/general. Kasmi & Mokrani's spline-error threshold is method-specific and EDRA-tested.

#### Recommendation

`HUMAN_APPROVED_WITH_LIMITATION`. Adopt Kasmi and Mokrani (2016) as the source basis for automatic `B=0..8` using eight sectors, third-order spline sub-contour fitting, and the 5% threshold form. Because executable `length(y)` details remain under-specified, the dissertation implementation is a source-guided author operationalization, not exact source reproduction. This is approved only as a geometric sub-contour irregularity proxy for the clinical concept of abrupt pigment cutoff. It must not be described as directly measuring lesion-to-skin pigment transition.

#### Implementation Resolution

The final dissertation implementation uses the GT binary lesion mask, ordered external 2D contour, lesion centroid, lesion-pixel second-moment principal-axis orientation shared with ABCD-A, principal-axis-relative eight-sector assignment, ordered local parametric contour runs, cumulative arc-length parameterization, dimensionless cubic smoothing splines with `alpha=1.0`, Euclidean geometric RMSE, `L_smooth` fitted-contour normalization, and `E_smooth > 0.05`. No PH2 calibration is performed. Radial radius-over-angle representation, sample-count normalization, and fixed image-axis sectors are historical comparators, not the final production method. See `docs/abcd_border_methodology.md`.

#### Remaining Uncertainty

No source identified a fully specified, PH2-validated, eight-sector color-transition implementation of abrupt pigment cutoff. No PH2 calibration will be performed for the approved Kasmi and Mokrani proxy. Exact source reproduction remains unavailable because `y` and `length(y)` are not executable from the source alone. Principal-axis alignment, ordered parametric contours, dimensionless smoothing, Euclidean residuals, `L_smooth`, and threshold-edge behavior are explicitly documented implementation conventions and synthetic experiment findings.

### Decision ID: `ABCD-B-ABRUPTNESS-001`

#### Proposed Decision

Do not replace ABCD B with compactness, solidity, circularity, convexity, or fractal dimension. Treat them only as auxiliary geometry unless a source explicitly maps them to eight-sector ABCD B.

#### References

##### Reference 1

* Full citation: Sertan Kaya, Mustafa Bayraktar, Sinan Kockara, Mutlu Mete, Tansel Halic, Halle E. Field, Henry K. Wong. "Abrupt skin lesion border cutoff measurement for malignancy detection in dermoscopy images." BMC Bioinformatics, 2016, 17(Suppl 13), 367.
* DOI/PMID: DOI `10.1186/s12859-016-1221-4`; PMID `27766942`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: Abstract Methods/Results; Methods "Dermoscopic image analysis"; Figures 13-18 and equations 2.17-2.18 in the online article.
* What it supports: Abruptness can be quantified by contracting/scaling lesion borders, sampling inner border regions, and computing pigment-pattern homogeneity/statistical texture features across scales; RGB blue channel was most informative; validation used 100 images.
* What it does NOT support: It explicitly does not analyze the eight virtual pies; it does not output ABCD `B=0..8`.

##### Reference 2

* Full citation: Recep Erol, Mustafa Bayraktar, Sinan Kockara, Sertan Kaya, Tansel Halic. "Texture based skin lesion abruptness quantification to detect malignancy." BMC Bioinformatics, 2017, 18(Suppl 14), 484.
* DOI/PMID: DOI `10.1186/s12859-017-1892-5`; PMID `29297290`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `DATASET_VALIDATION`, `LIMITATION`.
* Evidence location: Abstract; Methods "Dermoscopic image analysis"; Figure 2.
* What it supports: The improved method uses level set propagation to contract the lesion border and computes homogeneity, mean, and standard deviation of pixel values in the region between the original and contracted border; lower homogeneity indicates sharper cutoffs; evaluated on 1000 images from ISIC 2016 and EDRA.
* What it does NOT support: It does not produce eight binary sector scores or a direct ABCD B score.

#### Evidence Assessment

`STRONG` for rejecting geometric substitution; `MODERATE` for using homogeneity as auxiliary abruptness evidence.

#### Generalizability

Kaya/Erol methods are computational abruptness methods, classifier-derived for malignancy detection, not PH2-specific and not ABCD-sector output methods.

#### Recommendation

`REJECTED` for using compactness/solidity/fractal dimension as ABCD B. Kaya/Erol abruptness methods may remain documented as auxiliary evidence, but they are not required for the initial implementation and must not be combined with Kasmi and Mokrani (2016) to create a new unpublished ABCD B formula.

#### Remaining Uncertainty

The literature found supports continuous abruptness quantification and ABCD-style octant scoring separately, but not a fully specified PH2-ready method combining both.

### Decision ID: `ABCD-C-COLOR-001`

#### Proposed Decision

Use a deterministic RGB six-color classifier on lesion pixels only; do not use PH2 expert color annotations as FE input. Treat PH2 annotations, if used, as validation labels.

#### References

##### Reference 1

* Full citation: Michael Kunz, Wilhelm Stolz. "ABCD rule." Dermoscopedia, last edited 2023.
* DOI/PMID: URL `https://dermoscopedia.org/ABCD_rule`.
* Evidence role: `CLINICAL_DEFINITION`.
* Evidence location: Color section, lines 31-41.
* What it supports: ABCD counts white, red, light brown, dark brown, blue-gray, and black; white should be whiter than adjacent normal skin.
* What it does NOT support: It does not provide RGB thresholds or minimum area rules.

##### Reference 2

* Full citation: Sharmin Majumder, Muhammad Ahsan Ullah. "A Computational Approach to Pertinent Feature Extraction for Diagnosis of Melanoma Skin Lesion." Pattern Recognition and Image Analysis, 2019, 29(3), 503-514.
* DOI/PMID: DOI `10.1134/S1054661819030131`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`.
* Evidence location: Section 2.4 "Color feature", Table 1.
* What it supports: Six ABCD/PH2 colors are thresholded in normalized RGB. Table 1 gives: white `R>=0.8, G>=0.8, B>=0.8`; red `R>=0.588, G<0.2, B<0.2`; light brown `0.588<=R<=0.94, 0.196<G<=0.588, 0<B<0.392`; dark brown `0.243<R<0.56, 0<=G<0.392, 0<B<0.392`; blue-gray `0<=R<=0.588, 0.392<=G<=0.588, 0.490<=B<=0.588`; black `R<=0.243, G<=0.243, B<=0.243`. A color is present when its pixels exceed 5% of lesion pixels. Threshold ranges were computed from 200 input images.
* What it does NOT support: It does not establish the 5% rule as expert-derived or universal; the paper's ABCD variant uses D as diameter in a related Springer version, so only the color threshold table is used here.

##### Reference 3

* Full citation: Rachid Kasmi, Kamel Mokrani. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing, 2016, 10(6), 448-455.
* DOI/PMID: DOI `10.1049/iet-ipr.2015.0385`.
* Evidence role: `COMPUTATIONAL_OPERATIONALIZATION`, `FORMULA`, `PARAMETER`, `THRESHOLD`, `DATASET_VALIDATION`.
* Evidence location: Section 4.3 "Colour", Figure 9; Section 5; Tables 3-4.
* What it supports: Six suspicious ABCD colors are represented as RGB reference values obtained from a manually segmented lesion; each lesion pixel is assigned by normalized Euclidean distance to a reference color when distance is less than `T=0.4`; a color is present if its pixel count exceeds 5% of lesion pixels. Validation used 200 EDRA images.
* What it does NOT support: It does not provide PH2-calibrated thresholds and does not justify 5% as a dermatologist-derived universal threshold.

#### Evidence Assessment

`MODERATE`

#### Generalizability

Majumder & Ullah's thresholds are dataset-derived from 200 input images and therefore dataset-specific. Kasmi & Mokrani's RGB reference-distance method is dataset-tested on EDRA and method-specific. Neither is a general clinical threshold.

#### Recommendation

`HUMAN_APPROVED`. Adopt the Majumder and Ullah (2019) normalized RGB table and the published 5% lesion-area color-presence rule. These are published computational/dataset-derived parameters and are not claimed to be universal clinical thresholds. No PH2 calibration will be performed.

#### Remaining Uncertainty

The exact origin of the 5% rule is not established as expert consensus. It appears as an author rule in both Kasmi & Mokrani and Majumder & Ullah. No primary source identified a PH2-expert-derived minimum area threshold for color presence. PH2 expert color annotations must be used exclusively as validation ground truth and never as Feature Extraction input.

## A - Asymmetry

### Clinical Definition

The dermoscopic ABCD rule scores asymmetry as `0`, `1`, or `2` axes. The lesion is bisected along two perpendicular axes, and asymmetry is assessed from contour, colors, and dermoscopic structures. The original ABCD total dermoscopy score is `TDS = 1.3A + 0.1B + 0.5C + 0.5D`, with `<4.75` benign, `4.75/4.76-5.45` suspicious, and `>5.45` highly suggestive of melanoma.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Reflection non-overlap on lesion mask | Align lesion, reflect halves around candidate axes, compare non-overlap. Common formula: `A_axis = area(mask XOR reflected_mask) / area(mask)`. | Binary GT mask. | Raw asymmetry per axis. | Axis choice; interpolation/reflection method. | `UNRESOLVED` for mapping raw score to clinical `A_axis present`. | Majumder and Ullah use XOR area ratios in PH2-style ABCD FE; automatic symmetry work uses Jaccard/reflection. |
| Jaccard/Dice symmetry | Reflect one half over an axis and compute overlap similarity. | Binary GT mask. | Jaccard/Dice by axis. | Candidate-axis grid or principal axes. | `UNRESOLVED`. | Automatic symmetry in dermoscopic images formalizes Jaccard reflection. |
| PCA/moment axes | Use centroid and principal inertia axes to orient the lesion before reflection. | Binary mask. | Major/minor axes and asymmetry metrics. | Moment/PCA implementation. | `UNRESOLVED`. | Common in CAD asymmetry literature; aligns with original perpendicular-axis assessment but not identical to "lowest possible clinical score". |
| Color/structure asymmetry | Compare color histograms or detected structures between mirrored halves. | Original RGB + mask + color/structure masks. | Distribution divergence per axis. | Color space, bins, metric. | `UNRESOLVED`; no support found for `shape OR color` per axis followed by `A=A_x+A_y`. | Original ABCD includes contour, color, and structures, but computational OR rule lacks direct support. |

### Comparison

Shape reflection is reproducible, deterministic, PH2-compatible, and easy to test with synthetic masks. It is incomplete clinically because ABCD asymmetry also includes colors and structures. Color/structure asymmetry is conceptually closer but depends on unresolved color and structure detectors.

### Provisional Recommendation

`PARTIALLY_SUPPORTED`: implement raw shape asymmetry first with centroid/PCA axes and reflection non-overlap/Jaccard. Report both axes as raw values. Do not automatically convert to `A in {0,1,2}` unless a human approves a threshold or PH2 expert asymmetry annotations are used to calibrate a prespecified threshold.

The rule `shape OR color` per axis then `A=A_x+A_y` is `UNRESOLVED`.

### Gaps

No universal numeric threshold was found for converting reflection metrics to ABCD `0/1/2`. The exact clinical instruction chooses axes yielding the lowest asymmetry score; PCA axes may not reproduce that rule exactly.

## B - Border

### Clinical Definition

ABCD border score counts the number of eight peripheral sectors showing an abrupt, sharp ending of the pigment pattern. Score range is `0-8`, weighted by `0.1`.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Octant border scoring | Divide lesion around centroid into 8 angular sectors and compute per-sector abruptness/irregularity. | RGB + mask. | 8 raw sector values and count. | Centroid, sector definition, inner/outer rings. | Clinical count exists; computational per-sector cutoff is `UNRESOLVED`. | Original ABCD requires eight sectors; border abruptness studies implement sector/rotation logic. |
| Compactness/circularity | `compactness = P^2 / (4*pi*A)` or inverse circularity `4*pi*A/P^2`. | Mask contour. | Scalar geometric irregularity. | Perimeter estimator. | `UNRESOLVED`. | Widely used but less faithful to abrupt pigment cutoff. |
| Convexity/solidity | Compare lesion area/perimeter to convex hull. | Mask. | Convexity/solidity. | Hull method. | `UNRESOLVED`. | Jaworek-Korjakowska (2015) cites radial-distance and contour-function approaches for border irregularity; this row remains auxiliary and is not a direct ABCD B implementation. |
| Fractal dimension | Box-counting estimate: `D = lim_{e->0} log N(e) / log(1/e)`. | Boundary mask. | Boundary complexity. | Box sizes. | `UNRESOLVED`. | Border irregularity papers report larger values for irregular borders. |
| Borderline extrema/rotation | Rotate lesion, build borderline function, smooth and detect irregularities. | Mask contour. | Irregularity locations/score. | Gaussian smoothing, extrema rules. | Study-specific. | Jaworek-Korjakowska reports 350-image validation and 92% accuracy. |

### Comparison

Octant scoring is closest to ABCD but requires an explicit published abruptness criterion. Compactness, solidity, and fractal dimension are reproducible raw descriptors but measure geometric contour complexity, not necessarily pigment-pattern cutoff.

### Provisional Recommendation

`PARTIALLY_SUPPORTED`: compute raw geometric descriptors and an eight-sector evidence table, but keep ABCD `B` score `UNRESOLVED` until the per-sector abruptness rule is approved. Avoid claiming compactness alone is the ABCD border score.

### Gaps

No universal threshold maps compactness, fractal dimension, or radial variance to the original eight-sector clinical score. Abrupt cutoff requires color transition at the lesion-skin interface, not only mask geometry.

## C - Color

### Clinical Definition

ABCD color score counts the presence of six colors: white, red, light brown, dark brown, blue-gray, and black. Each present color contributes `1`; score range is `1-6`, weighted by `0.5`.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| PH2 color masks/annotations | Use PH2 expert color masks/labels if available. | PH2 metadata/color masks. | Presence of six colors. | None beyond PH2 parsing. | Expert label. | PH2 explicitly annotates six color classes and locations. |
| RGB thresholding | Threshold pixels into six classes; count class present when area exceeds a minimum lesion fraction. | RGB + lesion mask. | Pixel count and color presence. | RGB intervals, minimum area. | One PH2 paper uses `>5%` area, but RGB ranges are dataset-derived. | Majumder and Ullah report PH2-derived RGB thresholds. |
| Clustering/quantization | Cluster lesion pixels and map cluster centers to canonical colors. | RGB or Lab + mask. | Cluster centers, mapped color count. | `k`, stopping rule, palette, distance. | `UNRESOLVED` unless calibrated. | Used in CAD literature; mapping to ABCD colors remains implementation-dependent. |
| Relative skin color | Compare lesion pixels with surrounding skin mean. | RGB + mask + skin annulus. | Relative color features. | Annulus width, skin pixel filters. | Study-specific. | Blue-white veil/blotch studies show relative color helps control illumination/skin tone. |

### Comparison

Expert PH2 color labels are scientifically strongest for PH2-only validation but may be considered annotation use rather than image-derived FE. RGB thresholding is deterministic and compatible with PH2 if thresholds are explicitly taken from a PH2 study, but portability is limited. CIELAB/Delta E is perceptually attractive, but no resolved ABCD six-color thresholds were found.

### Provisional Recommendation

`HUMAN_APPROVED`: use the original RGB image inside the GT mask and the Majumder and Ullah (2019) normalized RGB thresholds with the published 5% lesion-area rule. Store raw per-color pixel fractions for validation/debugging. Do not use PH2 expert color annotations as FE input; use them only as validation ground truth. No PH2 calibration will be performed.

### Gaps

Exact RGB intervals from paywalled/secondary sources must be verified before implementation. White should be treated relative to adjacent normal skin when possible, per the clinical rule.

## D - Dermoscopic Structures

### Clinical Definition

The ABCD `D` component is differential dermoscopic structures, not diameter. The five counted structures are pigment network, structureless areas, branched streaks, dots, and globules. Score range is `1-5`, weighted by `0.5`.

### Computational Methods Found

| Structure | Candidate Method | Threshold | Shared With |
|---|---|---|---|
| Pigment network | Texture/Laws/NGLDM or graph/cyclic-hole detector. | Network presence has study thresholds; atypical/broadened classification `UNRESOLVED`. | 7PCL atypical pigment network; Menzies broadened network. |
| Structureless areas | Segment homogeneous/blotch/color regions after excluding network/dots/streaks. | `UNRESOLVED`. | 7PCL irregular pigmentation; Menzies color/nonuniformity and blue-white veil. |
| Branched streaks | Peripheral radial line detector, skeleton/graph features. | Presence/irregularity `UNRESOLVED`. | 7PCL irregular streaks; Menzies radial streaming/pseudopods. |
| Dots | Blob detector with size/color filters. | Size/count thresholds `UNRESOLVED`. | 7PCL irregular dots/globules; Menzies blue-gray and peripheral black dots. |
| Globules | Blob detector with larger size/color filters. | Size/count thresholds `UNRESOLVED`. | 7PCL irregular dots/globules; Menzies peripheral black dots/globules. |

### Comparison

No single complete deterministic ABCD-D extractor with fully specified thresholds was found. Individual structure detectors exist, but several rely on training, private datasets, or study-specific thresholds.

### Provisional Recommendation

`UNRESOLVED`: do not implement a final ABCD `D` score until the component detectors are individually approved. It is acceptable to specify reusable raw detectors for network, streaks, dots/globules, structureless areas, and blue-white/irregular pigmentation, then pass raw evidence to the VLM.

### Gaps

Need approved thresholds for structure presence, distinction between dots and globules at PH2 scale, and distinction between structureless areas, blotches, regression, and blue-white veil.

## Sources

1. Nachbar F, Stolz W, Merkle T, et al. "The ABCD rule of dermatoscopy." J Am Acad Dermatol. 1994. https://pubmed.ncbi.nlm.nih.gov/8157780/
2. Dermoscopedia. "ABCD rule." https://dermoscopedia.org/ABCD_rule
3. Mendonca T, Ferreira PM, Marques JS, Marcal ARS, Rozeira J. "PH2 - A dermoscopic image database for research and benchmarking." EMBC 2013. https://pubmed.ncbi.nlm.nih.gov/24110966/
4. PH2 database page. https://www.fc.up.pt/addi/ph2%20database.html
5. Majumder S, Ullah MA. "Feature extraction from dermoscopy images for melanoma diagnosis." SN Applied Sciences. 2019. https://doi.org/10.1007/s42452-019-0786-8
6. Jaworek-Korjakowska J. "Novel Method for Border Irregularity Assessment in Dermoscopic Color Images." 2015. https://doi.org/10.1155/2015/496202
7. Anantha M, Moss RH, Stoecker WV. "Detection of pigment network in dermatoscopy images using texture analysis." 2004. https://pubmed.ncbi.nlm.nih.gov/15249068/
8. Sadeghi M, Razmara M, Lee TK, Atkins MS. "A novel method for detection of pigment network in dermoscopic images using graphs." 2011. https://pubmed.ncbi.nlm.nih.gov/20724109/
9. Sadeghi M, Lee TK, McLean D, Lui H, Atkins MS. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." 2013. https://pubmed.ncbi.nlm.nih.gov/23335664/
10. Maglogiannis I, Delibasis KK. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." 2015. https://pubmed.ncbi.nlm.nih.gov/25540998/
