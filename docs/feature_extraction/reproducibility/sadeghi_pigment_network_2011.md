# Sadeghi et al. 2011 Pigment Network Reproducibility

## Scope

Primary selected method: Maryam Sadeghi, Majid Razmara, Tim K. Lee, M. Stella Atkins. "A novel method for detection of pigment network in dermoscopic images using graphs." Computerized Medical Imaging and Graphics. 2011;35(2):137-143. DOI `10.1016/j.compmedimag.2010.07.002`; PMID `20724109`. https://pubmed.ncbi.nlm.nih.gov/20724109/

Supplementary same-author methodological source located: Sadeghi et al. "Pigment Network Detection and Analysis" web-accessible chapter/page. https://studylib.net/doc/11565416/pigment-network-detection-and-analysis

ADR status: `NETWORK-DETECT-001 = HUMAN_APPROVED`, but implementation is blocked until primary-source details are sufficient.

Post-reproducibility human checkpoint: Sadeghi et al. (2011) remains the preferred scientific method, but its implementation state is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. Anantha, Moss and Stoecker (2004) is accepted as a practical reproducible fallback for pigment-network `RAW_EVIDENCE` only. The fallback must not be described as reproducing Sadeghi, resolving atypical pigment network, resolving broadened network, or providing an ABCD pigment-network presence rule from positive blocks.

PH2 rule: PH2 pigment-network annotations are validation ground truth only and must not tune parameters, thresholds, classifier models, or scale conversions.

## Reproducibility Table

| Component | Exact implementation | Value/equation | Origin class | Primary source | Exact source location | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|
| Lesion segmentation | Use manual or automatic segmentation before analysis | Manual segmentation or Wighton method in supplementary source | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Pre-processing, lines 191-194 | Partial | PH2 GT mask can substitute under ADR-002 because masks are allowed FE inputs, but the original automatic border method is not reproduced. |
| Sharpening | MATLAB Image Processing Toolbox unsharp mask | High-pass filter, alpha `3`, MATLAB default parameters | `PUBLISHED_COMPUTATIONAL`, `AUTHOR_DEFINED` | Sadeghi chapter | Pre-processing, lines 194-203, Eq. 1 | Yes | Exact MATLAB default behavior must be matched if implemented. |
| Channel selection | Convert color image to one plane; choose green channel | Green channel selected after training/validation | `DATASET_DERIVED` | Sadeghi chapter | Pre-processing, lines 210-215; Table 1 lines 438-455 | Yes | The PubMed abstract does not expose this; same-author chapter does. |
| Hole edge detection | Detect sharp intensity changes with LoG | MATLAB LoG edge detection using zero crossings | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Hole Detection, lines 218-239 | Partial | LoG sigma/window for hole detection is not explicitly recovered. |
| Binary image to graph | Convert binary connected components to graph | 8-connected neighbors; each pixel is a node with coordinate label | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Hole Detection, lines 243-249 | Yes | Exact graph construction is described. |
| Cyclic subgraph detection | Detect all cyclic subgraphs | Iterative Loop Counting Algorithm, tree transform plus depth-first search | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Hole Detection, lines 250-252 | Partial | ILCA reference is cited but algorithm details are not reproduced in the accessible source. |
| Cycle size filter | Remove cycles outside supported range | Remove cycles shorter than 7 pixels and longer than 150 pixels | `AUTHOR_DEFINED`, `DATASET_DERIVED` | Sadeghi chapter | Hole Detection, lines 261-267 | Yes | Authors state values depend on scale/magnification; no PH2 recalibration allowed. |
| Hole/globule/dot color-intensity filter | Compare inside intensity and border intensity | Green if `0.01 < (I - B) < 0.1`; white if `(I - B) > 0.1` and any inside pixel `PI > 0.8`; red otherwise | `AUTHOR_DEFINED`, `DATASET_DERIVED` | Sadeghi chapter | Hole Detection, lines 255-291, Eq. 2 | Yes | `I` is mean inside intensity; `B` is mean border/outside intensity. |
| Oil bubble / white cyst removal | Remove bright false holes | Same Eq. 2 white condition | `AUTHOR_DEFINED`, `DATASET_DERIVED` | Sadeghi chapter | Hole Detection, lines 272-278 | Yes | Preserves false structures only as rejected candidates. |
| High-level hole graph | Build graph of pigment-network holes | Nodes are centers of accepted holes; connect nodes within MDT | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Lines 295-303 | Yes | No minimum node distance threshold. |
| Maximum distance threshold | Distance threshold for connecting holes | `MDT = 3 * average_hole_diameter` | `AUTHOR_DEFINED`, `DATASET_DERIVED` | Sadeghi chapter | Lines 297-303 | Yes | Alpha set to `3`. |
| Density ratio feature | Quantify hole graph density | `Density = |E| / (|V| * log(LesionSize))` | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Geometric Features, lines 374-392, Eq. 3 | Yes | This is the feature used for absent/present classification in the same-author chapter. |
| Classifier for 94.3% result | Feed density ratio into Weka SimpleLogistic | SimpleLogistic with LogitBoost base learners | `CLASSIFIER_DERIVED` | Sadeghi chapter and PubMed abstract | Evaluation, lines 419-430; PubMed abstract | No | Trained model coefficients, decision boundary, Weka version/defaults, and random/training state are not provided. |
| Reported 94.3% output | Present/Absent classification | 500 atlas images; 250 present, 250 absent; green channel accuracy 94.3% | `DATASET_DERIVED`, `CLASSIFIER_DERIVED` | Sadeghi chapter; PubMed | Lines 423-455; PubMed abstract | No | The exact output is the SimpleLogistic classification, not the raw density ratio alone. |
| Net mask extraction | Apply LoG to green channel to identify net | Sigma `0.15`; LoG window half average hole size; average window 11 px; automatic threshold; skeletonize net mask | `PUBLISHED_COMPUTATIONAL`, `DATASET_DERIVED` | Sadeghi chapter | Net Detection, lines 314-335 | Partial | Automatic threshold method is not specified. |
| Structural features | Net and hole geometry | Mean/std/CV of net size and length; thickness distance to skeleton; 6-bin thickness histogram; hole-size stats | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Lines 351-369 | Partial | Histogram bin ranges are not recovered. |
| Chromatic features | HSV plus green channel features | Mean/std/CV over HSV channels and green; atypicality measure is green intensity over net mask normalized by net pixels | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Lines 398-408 | Partial | Full 37-feature list not fully recovered in accessible text. |
| Texture features | Haralick GLCM features | Entropy, energy, contrast, correlation, homogeneity; 4 directions, distance 4 px, 8 gray levels; lesion and pigment-network GLCMs | `PUBLISHED_COMPUTATIONAL` | Sadeghi chapter | Lines 409-418 | Yes | Useful for raw evidence, not necessary for the 94.3% density-only result. |

## Reproducibility Verdict

`NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE`

Blocking gaps:

- The 2011 journal full text was not recovered; PubMed provides only the abstract.
- The exact SimpleLogistic model used for the reported 94.3% result is not published as coefficients, tree/rule, serialized model, or fixed decision threshold.
- The LoG sigma/window for hole edge detection is not fully specified in the recovered same-author chapter.
- The automatic threshold used to produce the net mask is not specified.
- ILCA details are referenced but not included in the recovered source.

## Fallback Feasibility: Anantha, Moss and Stoecker 2004

Anantha, Moss and Stoecker (2004) was recovered in full via PMC and is a reproducible alternative for gross pigment-network detection using Laws/NGLDM texture blocks. It should not automatically replace Sadeghi et al. (2011), because ADR-005 selected Sadeghi as preferred and treats Anantha as secondary.

Fallback reproducibility: `REPRODUCIBLE_WITH_EXPLICIT_GAPS`.

Recovered Anantha details include 512 x 480 24-bit RGB images, DullRazor preprocessing, 64 x 64 fully enclosed lesion blocks, luminance `0.29R + 0.59G + 0.11B`, NGLDM `a=4`, `d=5`, `TN1=5800`, Laws masks 9 x 9 / 11 x 11 / 13 x 13, weighted Laws combination 40% 9 x 9 and 60% 11 x 11, `TL1=TL2=30000`, `TL3=130`, `TL4=180`, and scale `512 px = 11.9 mm`.

Fallback implementation decision: Anantha, Moss and Stoecker (2004) may be used as pigment-network `RAW_EVIDENCE` on original-resolution PH2 images. No PH2 calibration is authorized.

Fallback limitation: the method detects gross pigment-network texture on blocks and explicitly does not identify pigment-network irregularities. It must not be used to compute ABCD D pigment-network presence, 7PCL atypical pigment network, or Menzies broadened network.

## Algorithm-Specific Implications

| Mapping | Resolution |
|---|---|
| ABCD D pigment network presence | `STILL_UNRESOLVED` for Sadeghi exact reproduction because classifier model/threshold is missing. |
| 7PCL-r atypical pigment network | `STILL_UNRESOLVED`; clinical atypicality mapping remains unsupported by a deterministic threshold. |
| Menzies-r broadened network | `STILL_UNRESOLVED`; no numerical focal-width threshold recovered. |
