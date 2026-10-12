# Remaining Reproducibility Gaps

Authoritative checkpoints: `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md`.

Resolved human-decision questions from the prior version:

- Anantha, Moss and Stoecker (2004) is accepted as a pigment-network `RAW_EVIDENCE` fallback, but not as Sadeghi reproduction and not as an ABCD/7PCL/Menzies clinical mapping.
- Celebi et al. (2008) Figure 5 manual extraction was completed from the primary PDF, independently verified from 300 dpi and 600 dpi renders, and human-confirmed for pixel-level implementation.
- Stoecker et al. (2005) absolute-color path `R < 120 AND G < 60`, followed by `A_blotch > 800 pixels`, is to be used unchanged on original-resolution PH2 images as a `PUBLISHED_COMPUTATIONAL`, source-scale-specific implementation-ready path.
- `RETRAIN_UNPUBLISHED_CLASSIFIER = REJECTED`.
- `RAW_EVIDENCE` is allowed only when the raw-measurement mechanism is reproducible and scientifically approved.
- No new methodological search will be conducted for currently unresolved extractors within this dissertation; their unavailable/unresolved states are accepted limitations unless a new explicit human decision changes scope.

## 1. Blocking Implementation Gaps

- `UNRESOLVED_REPRODUCIBILITY_GAP`: Sadeghi et al. (2011) pigment-network SimpleLogistic model or decision boundary for the reported 94.3% present/absent result is not published in recovered primary material.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Sadeghi et al. (2011) hole-detection LoG sigma/window and net-mask automatic threshold are not fully recovered.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Sadeghi et al. (2013) four selected LoG scales/filter sizes are not recovered.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Sadeghi et al. (2013) valid-streak angular threshold and outlier residual threshold are not recovered.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Sadeghi et al. (2013) SimpleLogistic trained models, Weka version/defaults, folds, and source data are not published.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Maglogiannis and Delibasis (2015) full inverse nonlinear diffusion formulation and all implementation thresholds are not recovered.
- `UNRESOLVED_REPRODUCIBILITY_GAP`: Maglogiannis and Delibasis (2015) provide no recovered physical-to-pixel conversion for dot/globule size. No PH2 pixel equivalent for the clinical approximately 0.1 mm distinction is approved.
- `IMPLEMENTATION_BLOCKED_BY_REPRODUCIBILITY`: Stoecker et al. (2005) relative-color segmentation does not fully specify the outer extent of the background-skin sampling region. Do not invent it and do not substitute Celebi's skin annulus.

## 2. Unresolved Clinical Mappings

- ABCD D pigment network presence remains unresolved for the selected Sadeghi method.
- ABCD D branched streaks remain unresolved.
- ABCD D dots remain unresolved.
- ABCD D globules remain unresolved.
- Complete ABCD `D in {0,...,5}` remains unresolved and must remain `null`.
- 7PCL-r atypical pigment network remains unresolved.
- 7PCL-r blue-whitish veil binary clinical mapping remains unresolved; Celebi resolves the pixel-level detector and `S1` raw evidence only.
- 7PCL-r irregular pigmentation remains unresolved.
- 7PCL-r irregular dots/globules remain unresolved.
- Menzies-r broadened network remains unresolved.
- Menzies-r blue-white veil binary clinical mapping remains unresolved; Celebi resolves the pixel-level detector and `S1` raw evidence only.
- Menzies-r blue-gray dots remain unresolved.
- Menzies-r peripheral black dots/globules remain unresolved.
- Menzies-r radial streaming remains unresolved.
- Menzies-r pseudopods remain unresolved.
- Complete deterministic 7PCL-r FE remains unavailable.
- Complete deterministic Menzies-r FE remains unavailable.

## 3. Non-Blocking Reproducibility Limitations

- Stoecker et al. (2005) absolute-color raw blotch mask and area evidence are implementation-ready, but the original diagnostic neural-network result is not exactly reproducible without unpublished weights/training details.
- Stoecker et al. (2005) `A_blotch > 800` pixels is tied to typical 1024 x 768 source images. ADR-010 authorizes using it unchanged on original-resolution PH2 images after the absolute-color rule and requires explicit reporting that physical equivalence on PH2 is not guaranteed.
- Stoecker et al. (2005) background-skin sampling omits a 10% lesion-size area around the lesion, but the exact outer perimeter/sampling extent is not fully specified; therefore the relative-color path is blocked.
- Stoecker connected-component connectivity, centroid convention, and perimeter estimator are not specified; any dependent output must identify the convention before coding.
- Anantha, Moss and Stoecker (2004) is reproducible as a fallback for gross pigment-network block texture, but it is not the ADR-preferred graph/cyclic-hole method and does not resolve atypical or broadened network.
- Celebi et al. (2008) Figure 5 pixel-level tree is recovered and human-confirmed for implementation. The recovered lesion-level `S1`/`S3` tree is a melanoma/benign diagnostic classifier, not a 7PCL-r or Menzies-r blue-white-veil presence mapping.
- Celebi et al. (2008) pixel training/test regions are unpublished, so retraining the C4.5 tree would not reproduce the published model; retraining is rejected by ADR-009 and is not required to apply the transcribed published tree.
## 4. Final Human Decisions Resolved

- The fully transcribed Celebi Figure 5 pixel-level tree is implementation-ready for coding as a pixel-level detector.
- No new methodological search/selection is required within this dissertation for streaks, dots/globules, 7PCL irregular pigmentation, broadened network, atypical pigment network, or unresolved Menzies criteria.
- The raw-only evidence package is sufficient for the planned VLM-grounding experiment despite incomplete deterministic 7PCL-r and Menzies-r FE.
- Future implementation must expose unavailable criteria in schemas while preserving `null != absent`.
