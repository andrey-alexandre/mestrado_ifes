# Sadeghi et al. 2013 Streaks Reproducibility

## Scope

Primary selected method: Maryam Sadeghi, Tim K. Lee, David McLean, Harvey Lui, M. Stella Atkins. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." IEEE Transactions on Medical Imaging. 2013;32(5):849-861. DOI `10.1109/TMI.2013.2239307`; PMID `23335664`. PubMed: https://pubmed.ncbi.nlm.nih.gov/23335664/

Full-text access used: ResearchGate HTML/PDF text cache. https://www.researchgate.net/publication/234704377_Detection_and_Analysis_of_Irregular_Streaks_in_Dermoscopic_Images_of_Skin_Lesions

ADR status: `STREAK-DETECT-001 = HUMAN_APPROVED`; implementation blocked until source details are sufficient. 7PCL irregular streaks are approved with limitation. ABCD branched streaks, Menzies radial streaming, and Menzies pseudopods remain unresolved.

Post-reproducibility human checkpoint: Sadeghi et al. (2013) remains scientifically selected, but implementation status is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. Do not reconstruct missing LoG scales, invent angular thresholds, invent residual thresholds, retrain SimpleLogistic, approximate the detector and call it Sadeghi, or create a new streak detector merely to complete the feature set. Current implementation state is `UNAVAILABLE`.

PH2 rule: PH2 streak annotations validate presence/absence only; they must not tune thresholds, classifiers, subtype mappings, or geometric parameters.

## Reproducibility Table

| Component | Exact implementation | Value/equation | Origin class | Primary source | Exact source location | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|
| Clinical target | Streaks include radial streaming and pseudopods | Radial streaming is linear extension at periphery; pseudopods are finger-like dark projections | `CLINICAL_CONSENSUS` | Sadeghi et al. 2013 | Introduction, lines 240-264 | Yes | Paper treats them as streak patterns but does not separate final subtype classifiers. |
| Irregularity concept | Irregular streaks are not regularly/symmetrically distributed | Not clearly attached to pigment network lines | `CLINICAL_CONSENSUS` | Sadeghi et al. 2013 | Lines 294-304 | Yes | Used to design features, not a standalone threshold. |
| Preprocessing | Segment, orient, standardize, convert color | Wighton segmentation; major axis parallel to x-axis; sizes standardized; RGB to L*a*b*; L* used | `PUBLISHED_COMPUTATIONAL`, `DATASET_DERIVED` | Sadeghi et al. 2013 | Previous method outline, lines 412-423 | Partial | Exact size standardization dimensions are not visible. PH2 GT mask may substitute segmentation under ADR-002. |
| Candidate line detection | Multi-scale LoG blob/line detection near border | Four LoG filters with different sizes; Gaussian sigma `0.1`; 25 experiments with five sigmas and five filter sizes | `PUBLISHED_COMPUTATIONAL`, `DATASET_DERIVED` | Sadeghi et al. 2013 | Lines 427-449 | No | The four selected filter sizes are not recovered from accessible text. |
| Multi-scale result | Union of four scales | Union of results from four scales | `AUTHOR_DEFINED` | Sadeghi et al. 2013 | Lines 444-449 | Yes | Requires missing scale values. |
| Orientation estimation | Averaged squared gradient flow | ASGF; squared gradient doubles angle and squares magnitude; reliability calculated per block | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 451-486, Eq. 3-6 | Partial | Equations are garbled in the accessible text. |
| Reliability threshold | Filter orientation vectors | Keep vectors with reliability greater than `0.5` | `AUTHOR_DEFINED` | Sadeghi et al. 2013 | Lines 472-486 | Yes | One explicit threshold recovered. |
| Ridge frequency | Estimate frequency of detected linear structures | Rotate block vertical; project columns; frequency from first-last peak distance divided by peak count minus one | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 487-495 | Partial | Peak-detection details and block size are not recovered. |
| Gabor enhancement | Enhance candidate streak structures | Gabor filter tuned to ridge frequency and orientation | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 497-505, Eq. 7-8 | Partial | Equations are not legible in accessible text. |
| Two-class previous classifier | Absent/present classifier | 25 features; 12 candidate-line features and 13 lesion color/texture features; Weka SimpleLogistic | `CLASSIFIER_DERIVED` | Sadeghi et al. 2013 | Lines 506-529 | No | Exact 25 features and model coefficients are not fully specified in recovered text. |
| Valid streak geometry | Ellipse/foci model | Generalize lesion as ellipse with same normalized second central moments; compute two foci; compare candidate orientation to expected bisector | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 551-594; Fig. 3 | Partial | Orientation threshold symbol/value is missing in accessible text. |
| Valid streak threshold | Reject non-streak line segments | Constant orientation-difference threshold | `AUTHOR_DEFINED` | Sadeghi et al. 2013 | Lines 574-583 | No | Numeric threshold missing in accessible text. |
| Streak STR features | 18 streak features | Three structural, three geometric, six orientation, six chromatic | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 600-619 | Partial | Complete list is partly recoverable, equations not fully legible. |
| Graph node construction | Convert streak lines to nodes | Number of nodes per line based on line length divided by average streak length; shorter-than-average lines contribute one node | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 647-684, Eq. 9, Fig. 4 | Partial | Formula is described but exact Eq. 9 rendering is missing. |
| Streak graph density | Density-style graph feature | Standard graph density Eq. 10 and modified lesion-normalized density Eq. 11 | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 686-703, Eq. 10-11 | Partial | Eq. 11 not legible in accessible text. |
| Coverage | Pattern coverage across lesion arcs | Histogram bins across lesion arcs; coverage is fraction of bins with more than two streak lines; lesion divided into 12 areas in Fig. 5 | `PUBLISHED_COMPUTATIONAL`, `AUTHOR_DEFINED` | Sadeghi et al. 2013 | Lines 742-767; Fig. 5 | Partial | Bin range notation is missing; 12-area figure supports bin count. |
| Completeness | Connectedness of graph | Pattern completeness based on number of disconnected subgraphs/components | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 776-791, Eq. 12 | Partial | Exact equation not legible. |
| Orientation regularity | Linear regression over ordered streak orientations | Slope, RMSE, first-derivative RMSE, outlier fraction, entropy of orientations/residuals | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 793-931, Eq. 13-16, Fig. 6 | Partial | Numeric outlier range threshold is missing in accessible text. |
| Chromatic features | Streak color features | Mean/std of S and V in HSV and chosen luminance channel L* over segmented streak lines | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 932-942 | Yes | L* outperformed other channels. |
| Structural features | Streak size/count features | Mean streak length, total line count, streak-pixel/lesion-pixel ratio | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 944-957 | Yes | Useful raw outputs. |
| Lesion color texture features | LCT 13 features | Mean/std/reciprocal CV for HSV S,V and L*a*b* L*; GLCM energy, contrast, correlation, homogeneity | `PUBLISHED_COMPUTATIONAL` | Sadeghi et al. 2013 | Lines 959-976 | Partial | GLCM details are not fully visible. |
| Three-class classifier | Absent/Regular/Irregular | 31 features into Weka SimpleLogistic / LogitBoost | `CLASSIFIER_DERIVED` | Sadeghi et al. 2013 | Lines 1013-1030 | No | Model coefficients and Weka version/defaults are not provided. |
| Evaluation | 945 inclusive images and 300 clean subset | 10-fold cross-validation; L* best channel; 76.1% accuracy and weighted AUC 85% for three-class Set1 | `DATASET_DERIVED`, `CLASSIFIER_DERIVED` | Sadeghi et al. 2013 | PubMed abstract; lines 1211-1285 | Yes for reported result | Exact reproduction requires unavailable trained folds/data. |

## Reproducibility Verdict

`NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE`

Blocking gaps:

- Four selected LoG filter sizes are not recovered.
- Size standardization parameters are not recovered.
- Several equations are not legible in the accessible full-text cache.
- Constant orientation-difference threshold for valid streaks is missing.
- Outlier residual range threshold is missing.
- SimpleLogistic trained models, coefficients, Weka version/defaults, folds, and source data are not published.

## Operationalization of Branching, Radial Streaming, Pseudopods

The paper describes radial streaming and pseudopods clinically and shows examples. It does not publish separate deterministic classifiers for:

- branching;
- radial streaming as a distinct Menzies subtype;
- pseudopods as a distinct Menzies subtype;
- bulbous terminal morphology threshold.

The reported classifier is for Absent/Present, Regular/Irregular, and Absent/Regular/Irregular streak classes.

## Algorithm-Specific Implications

| Mapping | Resolution |
|---|---|
| ABCD D branched streaks | `STILL_UNRESOLVED` |
| 7PCL-r irregular streaks | `STILL_UNRESOLVED` for exact implementation until classifier/model details are recovered, despite ADR approval with limitation. |
| Menzies-r radial streaming | `STILL_UNRESOLVED` |
| Menzies-r pseudopods | `STILL_UNRESOLVED` |
