# Celebi et al. 2008 Blue-White Veil Reproducibility

## Scope

Primary selected method: M. Emre Celebi, Hitoshi Iyatomi, William V. Stoecker, Randy H. Moss, Harold S. Rabinovitz, Giuseppe Argenziano, H. Peter Soyer. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." Computerized Medical Imaging and Graphics. 2008;32(8):670-677. DOI `10.1016/j.compmedimag.2008.08.003`; PMID `18804955`. Open author manuscript: https://arxiv.org/pdf/1009.1013

ADR status: `VEIL-DETECT-001 = HUMAN_APPROVED`; ADR-010 confirms the Celebi Figure 5 pixel-level detector is `IMPLEMENTATION_READY`. 7PCL-r blue-whitish veil and Menzies-r blue-white veil binary clinical mappings remain `STILL_UNRESOLVED`.

Post-reproducibility human checkpoint: Celebi et al. (2008) remains the approved scientific method. ADR-009 authorized one final focused primary-source action: visually inspect PDF Figure 5 and transcribe the complete pixel-level C4.5 tree only if every split is unambiguously legible and independently verified.

Figure 5 extraction result: completed from the arXiv author manuscript PDF rendered at 300 dpi and independently rechecked at 600 dpi. The three split values are visibly legible in Figure 5. Final human review confirms the recovered pixel-level detector is approved for implementation.

PH2 rule: PH2 blue-whitish veil annotations are validation ground truth only and must not tune thresholds, classifier splits, annulus definitions, or component rules.

## Reproducibility Table

| Component | Exact implementation | Value/equation | Origin class | Primary source | Exact source location | Reproducible? | Notes |
|---|---|---|---|---|---|---|---|
| Dataset | 545 digital dermoscopy images from two atlases | Typical resolution 768 x 512; diagnoses listed in paper | `DATASET_DERIVED` | Celebi et al. 2008 | Section 2.1, lines 60-68 | Yes | Not PH2-specific. |
| Lesion border | Manual border supervised by WVS | Border points connected by second-order B-spline; closed curve flood-filled to binary mask | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Section 2.2, lines 70-79 | Yes | PH2 GT mask can substitute under ADR-002. |
| Skin reference exclusion band | Omit immediate outside-lesion area | Outside area equal to 10% of lesion area omitted | `AUTHOR_DEFINED` | Celebi et al. 2008 | Section 2.2, lines 80-87; Fig. 4 legend lines 349-350 | Yes | Computed with Euclidean distance transform. |
| Skin reference band | Compute background skin color | Average over next outside-lesion region with area equal to 20% of lesion area | `AUTHOR_DEFINED` | Celebi et al. 2008 | Section 2.2, lines 80-87 | Yes | Required for relative features. |
| Non-skin exclusion for skin reference | Exclude frames/rulers/hairs/bubbles | Skin rule: `R > 90 AND R > B AND R > G` | `AUTHOR_DEFINED` | Celebi et al. 2008 | Section 2.2, lines 83-86 | Yes | Saturation exclusions are not included here, unlike Stoecker blotch paper. |
| Training/test pixel selection | Manual circular veil/non-veil regions | 100 images; 43 with sizeable pure veil regions, 62 with sizeable pure non-veil regions | `DATASET_DERIVED` | Celebi et al. 2008 | Section 2.2, lines 88-93 | Partial | Exact selected pixels/regions are not published. |
| Neighborhood aggregation | Contextual pixel features | 5 x 5 neighborhood; median feature value assigned to center pixel | `AUTHOR_DEFINED` | Celebi et al. 2008 | Section 3, lines 100-108 | Yes | Median search implementation may vary but not scientifically critical. |
| Absolute color features | Chromaticity coordinates | `F1 = R_L/(R_L+G_L+B_L)`, `F2 = G_L/(R_L+G_L+B_L)`, `F3 = B_L/(R_L+G_L+B_L)` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Table 1, lines 373-379 | Yes | Table OCR is garbled but formulas are recoverable. |
| Relative color ratios | Lesion color divided by skin color | `F4 = R_L/R_S`, `F5 = G_L/G_S`, `F6 = B_L/B_S` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Table 1, lines 380-383 | Yes | Denominators are average background skin RGB. |
| Normalized relative ratios | Normalize F4-F6 by their sum | `F7 = F4/(F4+F5+F6)`, `F8 = F5/(...)`, `F9 = F6/(...)` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Table 1, lines 384-390 | Yes | Formula recovered despite OCR compression. |
| Relative color differences | Lesion minus skin channels | `F10 = R_L - R_S`, `F11 = G_L - G_S`, `F12 = B_L - B_S` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Table 1, lines 391-392 | Yes | Sign is lesion minus skin in table. |
| Normalized relative differences | Normalize F10-F12 by their sum | `F13 = F10/(F10+F11+F12)`, `F14 = F11/(...)`, `F15 = F12/(...)` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Table 1, lines 392-397 | Partial | Denominator with negative sums may need exact source rendering. |
| Texture features | GLCM statistics | Entropy `F16`, contrast `F17`, correlation `F18`; normalized GLCM averaged over 0, 45, 90, 135 degrees | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Section 3.3, lines 124-130 | Partial | Gray quantization and offset distance are not specified in the visible text. |
| Pixel classifier | C4.5 decision tree | C4.5 with pruning confidence `C=0.1` and minimum leaf samples `M=100` | `AUTHOR_DEFINED`, `CLASSIFIER_DERIVED` | Celebi et al. 2008 | Section 4, lines 133-150 | Yes for configuration | Exact training data and induced split values are needed for reproduction. |
| Pixel decision tree | Final pixel rule uses only F3 and F10 | If `F3 <= 0.3`, classify `Non-Veil`; else if `F10 > -51`, classify `Non-Veil`; else if `F10 <= -194`, classify `Non-Veil`; else classify `Veil` | `CLASSIFIER_DERIVED`, `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Figure 5, "Pixel classification tree"; Section 4, lines 146-155 | Yes | Transcribed visually from Figure 5 at 300 dpi, independently verified at 600 dpi, and human-confirmed for implementation. |
| Pixel postprocessing | Smooth initial veil mask | 5 x 5 majority filter | `AUTHOR_DEFINED` | Celebi et al. 2008 | Section 4, lines 155-158; Fig. 6 legend | Yes | Replaces each pixel label by majority label in 5 x 5 neighborhood. |
| Lesion-level feature S1 | Blue-white area fraction | `S1 = area(detected blue-white veil) / area(lesion)` | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Section 5, lines 163-170, Eq. 1 | Yes | Implementation-ready `RAW_EVIDENCE`; not a clinical binary veil-presence threshold. |
| Lesion-level feature S2 | Circularity descriptor | Boundary radial-distance coefficient of variation formula in Eq. 2 | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Section 5, lines 172-202, Eq. 2 | Partial | Formula is present but OCR rendering is difficult; original PDF figure/text must be inspected manually before implementation. |
| Lesion-level feature S3 | Ellipticity descriptor | Moment-based ellipticity formula in Eq. 3 | `PUBLISHED_COMPUTATIONAL` | Celebi et al. 2008 | Section 5, lines 202-244, Eq. 3 | Partial | Formula is present but OCR rendering is difficult; original PDF figure/text must be inspected manually before implementation. |
| Lesion-level classifier | C4.5 with 10-fold cross-validation | Features S1, S2, S3; induced tree retained S1 and S3 | `CLASSIFIER_DERIVED` | Celebi et al. 2008 | Section 5, lines 245-254 | Yes | Logic is stated in prose. |
| Lesion-level decision tree | Classify benign/melanoma from S1/S3 | If `S1 < 0.009`, benign; else if `S3 > 0.979`, benign; else melanoma | `CLASSIFIER_DERIVED` | Celebi et al. 2008 | Section 5, lines 250-254 | Yes | Diagnostic classifier only; not a 7PCL-r or Menzies-r blue-white-veil presence mapping. |
| Reported performance | Pixel and lesion results | Pixel sensitivity 84.33%, specificity 96.19%; lesion sensitivity 69.35%, specificity 89.97%, accuracy 82.94% | `DATASET_DERIVED`, `CLASSIFIER_DERIVED` | Celebi et al. 2008 | Lines 149-150; 256-262 | Yes | Depends on unpublished pixel training/test selections. |

## Reproducibility Verdict

`REPRODUCIBLE_WITH_EXPLICIT_GAPS`

The pixel-level blue-white veil decision tree is now reproducible from primary-source Figure 5 without retraining or inference. The executable pixel detector is the published C4.5 tree applied to `F3` and `F10`, followed by the published 5 x 5 majority filter. ADR-010 confirms this pixel-level detector is `IMPLEMENTATION_READY`.

Explicit gaps:

- The manual training/test pixel regions are not published, so retraining would not reproduce the published C4.5 model.
- Texture-feature gray quantization and pixel-distance parameter are not specified in the visible text.
- Equations for S2/S3 require manual inspection of the PDF rendering before faithful implementation.

Non-blocking context:

- Skin-annulus construction is explicit.
- The 18-feature family is largely recoverable, but the published application phase requires only `F3` and `F10`.
- The lesion-level decision tree is recoverable: `S1 < 0.009` benign; otherwise `S3 > 0.979` benign; otherwise melanoma. This is not a 7PCL-r or Menzies-r blue-white-veil presence rule.

## Figure 5 Transcription

Source: Celebi et al. (2008), arXiv author manuscript PDF, Figure 5 "Pixel classification tree". Visual check 1: 300 dpi page render. Visual check 2: 600 dpi page render.

Complete tree:

1. Test `F3 <= 0.3 ?`
2. If yes: `Non-Veil`.
3. If no: test `F10 > -51 ?`
4. If yes: `Non-Veil`.
5. If no: test `F10 <= -194 ?`
6. If yes: `Non-Veil`.
7. If no: `Veil`.

Equivalent positive veil interval:

`F3 > 0.3 AND -194 < F10 <= -51`

This equivalent interval is only a restatement of the transcribed Figure 5 tree, not a newly derived threshold.

## Algorithm-Specific Implications

| Mapping | Resolution |
|---|---|
| Celebi pixel-level detector | `IMPLEMENTATION_READY` for F3, F10, Figure 5 C4.5 rule, 5 x 5 majority filter, `veil_mask`, and `S1` / veil-area fraction. |
| 7PCL-r blue-whitish veil | Pixel-level detector and `S1` raw evidence are implementation-ready; binary clinical criterion remains `STILL_UNRESOLVED`. |
| Menzies-r blue-white veil | Pixel-level detector and `S1` raw evidence are implementation-ready; binary clinical criterion remains `STILL_UNRESOLVED`. |
| ABCD D | `STILL_UNRESOLVED`; ADR-005 states veil must not independently increment ABCD D. |
