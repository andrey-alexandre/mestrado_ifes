# Seven-Point Checklist Feature Extraction Research

## Post-Reproducibility Status Note

This document preserves the earlier research-stage evidence and provisional recommendations. For current implementation readiness, `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md` supersede any older `READY_FOR_APPROVAL` or `PENDING_HUMAN_APPROVAL` language here.

Active checkpoint summary: pigment network may emit `RAW_EVIDENCE` only through the Anantha fallback; blue-whitish veil has an implementation-ready Celebi pixel-level detector and `RAW_EVIDENCE` area fraction, but the 7PCL-r binary clinical criterion remains `UNRESOLVED`; streaks are `UNAVAILABLE`; dots/globules are `UNAVAILABLE`; irregular pigmentation may receive Stoecker raw blotch evidence only, while the 7PCL clinical mapping remains unresolved.

## Compatibility with the Current Project

The current `SPCLDiagnosticAgent` implements a reduced Seven-Point Checklist workflow in text: atypical pigment network, blue-whitish veil, irregular streaks, irregular pigmentation, and irregular dots/globules are in scope for this sprint. Atypical vessels, regression, and other variants exist clinically but are not part of the reduced requested implementation.

PH2 directly annotates pigment network, dots/globules, streaks, regression areas, blue-whitish veil, colors, asymmetry, and lesion masks. It does not directly annotate "irregular pigmentation" as a named PH2 field.

## Atypical Pigment Network

### Clinical Definition

The original 7-point checklist assigns 2 points to atypical pigment network. Argenziano defines it as a prominent, hyperpigmented or broad, irregular network. Consensus terminology describes pigment network as pigmented lines surrounding lighter holes; atypicality reflects irregular line thickness/color/spacing and asymmetry.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Texture analysis | Divide lesion into blocks and use Laws energy masks or NGLDM large-number emphasis on luminance. | RGB + mask. | Block-level network map. | 64x64 blocks; Laws 9x9/11x11; texel approx. 10 px/0.22 mm in source images. | Study thresholds, not PH2-universal. | Anantha et al. studied 155 images and 2037 full-network blocks. |
| Graph/cyclic holes | Enhance edges, convert binary edge image to graph, find cyclic subgraphs corresponding to network holes. | RGB + mask. | Network present/absent and density ratio. | Edge/enhancement parameters. | Classifier threshold not directly portable. | Sadeghi et al. reported 94.3% present/absent accuracy on 500 atlas images. |
| Gabor/directional filters | Emphasize oriented reticular lines, then classify typical/atypical. | RGB + mask. | Network mask and type. | Filter bank; classifier. | `UNRESOLVED` for deterministic fixed cutoff. | PH2-based work reports good PN classification but uses classifier logic. |

### Comparison

Texture and graph approaches are credible for network detection. Atypicality is harder because it requires irregularity of mesh, thickness, color, and distribution. PH2 can validate typical versus atypical labels, but thresholds should be approved.

### Provisional Recommendation

ADR-009 checkpoint status: `RAW_EVIDENCE` only through the Anantha fallback. Sadeghi remains scientifically preferred but `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. Automatic atypical/typical pigment-network classification remains `UNAVAILABLE`; do not derive it from Anantha-positive blocks or PH2 calibration.

### Gaps

No universal cutoff maps network texture/graph features to "atypical pigment network" without training/calibration.

## Blue-Whitish Veil

### Clinical Definition

Blue-white veil is an irregular, structureless area of confluent blue to whitish-blue pigmentation with an overlying white ground-glass haze. In the 7-point checklist it is a major criterion worth 2 points.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Contextual pixel classifier | Compute 18 pixel features in 5x5 neighborhoods: RGB chromaticity, relative RGB ratios/differences against skin, and GLCM entropy/contrast/correlation; classify blue-white pixels. | RGB + lesion mask + skin annulus. | Veil mask and area ratio. | 5x5 neighborhood; skin annulus; majority filter. | Lesion-level tree includes diagnostic classifier splits `S1 < 0.009` and `S3 > 0.979`; these are not 7PCL-r blue-whitish-veil presence thresholds. | Celebi et al. used 545 dermoscopy images; sensitivity 69.35%, specificity 89.97%. |
| Discrete color matching | Convert sRGB to CIELAB and match segments to a blue-white palette. | RGB + segmentation. | Probability/mask. | Mean-shift segmentation parameters; Lab distance tolerance. | Study-specific. | Reported in later blue-white veil color matching work. |

### Comparison

Celebi et al. is the strongest directly relevant computational source. It uses a supervised decision tree, so the exact classifier is not a purely hand-coded dermatological threshold. However, the raw features and area ratio are deterministic and reusable.

### Provisional Recommendation

ADR-010 checkpoint status: Celebi et al. remains `HUMAN_APPROVED` scientifically, and the Figure 5 pixel-level tree is `IMPLEMENTATION_READY`. The extractor may emit `veil_mask` and `S1` / veil-area fraction as `RAW_EVIDENCE`. The 7PCL-r binary blue-whitish-veil criterion remains `UNRESOLVED`; do not use `S1 = 0.009`, the lesion-level diagnostic tree, or `S1 > 0` as a clinical presence rule.

### Gaps

Need distinction from regression/peppering and reflections. Need decision whether to allow a trained decision tree or only deterministic thresholding.

## Irregular Streaks

### Clinical Definition

Streaks include radial streaming and pseudopods: pigmented linear or bulbous projections at the lesion periphery. Irregular streaks are focal/asymmetric rather than symmetrically distributed.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Graph/tubular streak detection | Detect valid streak lines, model their geometry and distribution, combine with lesion color/texture. | RGB + mask. | Absent/regular/irregular label; orientation/distribution features. | Line detector and graph rules. | Classifier-derived. | Sadeghi et al. used 945 images; 3-way accuracy 76.1%, weighted AUC 85%. |
| Peripheral band skeletonization | Analyze peripheral band, threshold dark structures, skeletonize, measure radial orientation, length, thickness, angular coverage. | RGB + mask. | Candidate streak count/coverage. | Band width, length/thickness filters. | `UNRESOLVED`. | Conceptually aligned with clinical definition; exact thresholds not found. |

### Comparison

The graph/tubular approach best matches streak morphology but is not threshold-complete for a deterministic PH2 implementation. PH2 only labels streaks present/absent, not regular/irregular.

### Provisional Recommendation

ADR-009 checkpoint status: `UNAVAILABLE`. Sadeghi et al. remains scientifically selected, but implementation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; do not compute a deterministic or raw streak extractor by reconstructing missing thresholds, scales, equations, or classifier artifacts.

### Gaps

Need PH2-scale band width, minimum streak length, angular focality rule, and hair differentiation rule.

## Irregular Pigmentation

### Clinical Definition

In the 7-point checklist, irregular pigmentation corresponds to irregular diffuse pigmentation or blotches: brown, gray, or black areas with irregular shape, distribution, and often abrupt ending.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Relative-color blotch detection | Detect dark blotches using absolute/relative red and green drops compared with background skin. | RGB + mask + skin annulus. | Blotch mask and asymmetry features. | Skin estimate; color-drop thresholds; component area. | Example rules include `R<120 and G<60`, relative drops, and area `>800 px` at 1024x768; not universal. | Stoecker et al. studied 512 images. |
| Color-mask distribution | Use black/dark-brown/blue-gray regions, compute area fraction, component eccentricity, irregularity, spatial entropy. | RGB or PH2 color masks + lesion mask. | Heterogeneity/asymmetry features. | Color-classifier thresholds, component rules. | `UNRESOLVED`. | PH2 color annotations make validation feasible but do not label irregular pigmentation directly. |

### Comparison

Relative-color approaches are more robust than absolute RGB and align with the lesion-versus-skin concept. The feature overlaps with ABCD color, Menzies multiple colors, and structureless/blotch features.

### Provisional Recommendation

ADR-009 checkpoint status: Stoecker raw blotch evidence may be emitted as `RAW_EVIDENCE` with documented limitations. The final 7PCL irregular-pigmentation clinical mapping remains `UNAVAILABLE`; PH2 does not provide a direct irregular-pigmentation FE input or calibration target.

### Gaps

PH2 lacks a direct irregular-pigmentation ground-truth label; clinical overlap with blotches/regression must be documented.

## Irregular Dots/Globules

### Clinical Definition

Dots and globules are sharply circumscribed round or oval black, brown, or blue-gray structures; globules are larger than dots. Irregularity reflects variation in size, shape, color, spacing, and asymmetric/peripheral distribution.

### Computational Methods Found

| Method | Description | Inputs | Output | Parameters | Threshold | Evidence |
|---|---|---|---|---|---|---|
| Multi-resolution dark blob segmentation | Segment dark dots/globules using inverse nonlinear diffusion, then extract count, area, size variance, spatial features. | RGB + mask. | Blob mask and features. | Diffusion/scales. | Study-specific. | Maglogiannis and Delibasis report automatic dot/globule segmentation improves classification. |
| LoG/DoG/top-hat blob detection | Enhance small dark circular structures and filter by size, circularity, and color. | RGB + mask. | Candidate dot/globule components. | Scale, circularity, size. | `UNRESOLVED`. | Consistent with blob morphology but thresholds are scale-dependent. |
| Spatial distribution metrics | Divide lesion into sectors/rings; compute count entropy, peripheral fraction, nearest-neighbor statistics, size/color CV. | Blob candidates + mask. | Irregularity evidence. | Sector/ring definitions. | `UNRESOLVED`. | ICIP 2013 and PH2 labels support this direction. |

### Comparison

This is one of the most PH2-testable 7PCL criteria because PH2 labels dots/globules as absent/typical/atypical. Still, a deterministic detector needs approved size and circularity thresholds.

### Provisional Recommendation

ADR-009 checkpoint status: `UNAVAILABLE`. Maglogiannis and Delibasis remains the selected scientific reference, but implementation is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`; do not substitute LoG/DoG/OpenCV blob detection or infer PH2 pixel sizes without a new human methodological decision.

### Gaps

Need dot/globule size threshold at PH2 resolution, handling of peppering, and separation from pigment network intersections.

## Sources

1. Argenziano G, Fabbrocini G, Carli P, et al. "Epiluminescence Microscopy for the Diagnosis of Doubtful Melanocytic Skin Lesions." Arch Dermatol. 1998. https://jamanetwork.com/journals/jamadermatology/fullarticle/189703
2. Kittler H, et al. "Standardization of terminology in dermoscopy/dermatoscopy." https://pmc.ncbi.nlm.nih.gov/articles/PMC5551974/
3. Jaimes N, et al. "Using Dermoscopy to Identify Melanoma and Improve Diagnostic Discrimination." https://pmc.ncbi.nlm.nih.gov/articles/PMC6375419/
4. PH2 database page. https://www.fc.up.pt/addi/ph2%20database.html
5. Mendonca T, et al. "PH2 - A dermoscopic image database for research and benchmarking." https://pubmed.ncbi.nlm.nih.gov/24110966/
6. Celebi ME, et al. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." https://pubmed.ncbi.nlm.nih.gov/18804955/
7. Anantha M, Moss RH, Stoecker WV. "Detection of pigment network in dermatoscopy images using texture analysis." https://pubmed.ncbi.nlm.nih.gov/15249068/
8. Sadeghi M, et al. "Detection and analysis of irregular streaks in dermoscopic images of skin lesions." https://pubmed.ncbi.nlm.nih.gov/23335664/
9. Stoecker WV, et al. "Detection of asymmetric blotches..." https://doi.org/10.1111/j.1600-0846.2005.00117.x
10. Maglogiannis I, Delibasis KK. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." https://pubmed.ncbi.nlm.nih.gov/25540998/
