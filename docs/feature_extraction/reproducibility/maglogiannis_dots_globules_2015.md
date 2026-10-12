# Maglogiannis and Delibasis 2015 Dots/Globules Reproducibility

## Scope

Primary selected method: Ilias Maglogiannis, Konstantinos K. Delibasis. "Enhancing classification accuracy utilizing globules and dots features in digital dermoscopy." Computer Methods and Programs in Biomedicine. 2015;118(2):124-133. DOI `10.1016/j.cmpb.2014.12.001`; PMID `25540998`. https://pubmed.ncbi.nlm.nih.gov/25540998/

Repository metadata located: University of Thessaly record. https://ir.lib.uth.gr/xmlui/handle/11615/30488

ADR status: `BLOB-DETECT-001 = HUMAN_APPROVED_PENDING_SOURCE_DETAIL`; implementation blocked until the primary source provides sufficient reproducibility parameters.

Post-reproducibility human checkpoint: Maglogiannis and Delibasis (2015) remains the selected scientific reference, but implementation status is `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`. Dots/globules are currently `UNAVAILABLE` as deterministic or raw FE. Do not reconstruct the inverse nonlinear diffusion method, substitute LoG/DoG/OpenCV blob detection without a new methodological decision, invent size thresholds, infer physical-to-pixel conversion, assume a pixel equivalent for the clinical approximately 0.1 mm distinction, or calibrate using PH2.

PH2 rule: PH2 dots/globules annotations validate post hoc only; they must not define pixel sizes, train classifiers, tune thresholds, or resolve dot/globule physical-scale conversions.

## Reproducibility Table

| Component | Exact implementation | Value/equation | Origin class | Primary source | Exact source location | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|
| Target structures | Dark dots and globules | Detect, segment, and count dark dots/globules | `PUBLISHED_COMPUTATIONAL` | Maglogiannis and Delibasis 2015 | PubMed abstract | No | Abstract confirms target but not implementation details. |
| Segmentation family | Multi-resolution inverse nonlinear diffusion | Multi-resolution approach based on inverse nonlinear diffusion | `PUBLISHED_COMPUTATIONAL` | Maglogiannis and Delibasis 2015 | PubMed abstract | No | Equations, diffusion coefficients, levels, time steps, and stopping rules are not recovered. |
| Candidate detection | Locate dark dot/globule candidates | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking gap. |
| Preprocessing | Any color conversion, masking, artifact handling | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking gap. |
| Size filtering | Dot/globule size thresholds | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking gap. |
| Shape filtering | Circularity/eccentricity/roundness thresholds | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking gap. |
| Color filtering | Dark/brown/black/blue-gray constraints | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking gap. |
| Dot definition | Computational dot definition | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Clinical 0.1 mm boundary comes from Dermoscopedia, not this paper. |
| Globule definition | Computational globule definition | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | No operational separation recovered. |
| Physical-to-pixel conversion | Convert clinical size to image pixels | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | No defensible PH2 pixel conversion can be introduced. |
| Minimum count rules | Define "multiple" or present/absent | Not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | Not available in abstract | No | Blocking for Menzies "multiple" terms. |
| Raw downstream features | Features passed to classification | A number of features extracted from segmented dots/globules; exact list not recovered | `UNRESOLVED` | Maglogiannis and Delibasis 2015 | PubMed abstract | No | Abstract states diagnostic value but not the feature vector. |
| Validation | Classification improvement | Applied to images with known histopathology; features improved melanoma/non-malignant nevus classification when combined with region descriptors | `DATASET_DERIVED`, `CLASSIFIER_DERIVED` | Maglogiannis and Delibasis 2015 | PubMed abstract | Partial | Sample size and classifier details not recovered. |
| Clinical dot/globule size | Dots less than 0.1 mm; globules larger than 0.1 mm | 0.1 mm boundary | `CLINICAL_CONSENSUS` | Dermoscopedia | Dermoscopic structures, lines 73-88 | Yes as clinical definition | Not a computational pixel threshold. |

## Reproducibility Verdict

`NOT_REPRODUCIBLE_FROM_AVAILABLE_PRIMARY_SOURCE`

Blocking gaps:

- Full text was not recovered from accessible primary sources.
- Inverse nonlinear diffusion equations and parameters are missing.
- Multi-resolution levels and scale parameters are missing.
- Candidate segmentation, size, circularity, color, and postprocessing rules are missing.
- Dot/globule separation rule is missing.
- Physical-to-pixel conversion is not provided.
- Raw feature vector and classifier inputs are not recovered.

## Physical Size Conversion

The primary paper evidence recovered does not provide a defensible method for converting the clinical approximately 0.1 mm dot/globule boundary into image pixels. Therefore the PH2 pixel conversion is:

`UNRESOLVED_REPRODUCIBILITY_GAP`

## Algorithm-Specific Implications

| Mapping | Resolution |
|---|---|
| ABCD D dots | `STILL_UNRESOLVED` |
| ABCD D globules | `STILL_UNRESOLVED` |
| 7PCL-r irregular dots/globules | `STILL_UNRESOLVED` |
| Menzies-r blue-gray dots | `STILL_UNRESOLVED` |
| Menzies-r peripheral black dots/globules | `STILL_UNRESOLVED` |
