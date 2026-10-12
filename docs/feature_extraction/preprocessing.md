# Preprocessing Research

## Compatibility with PH2 and Planned Feature Extraction

The future FE stage should use the original PH2 image resolution and the PH2 expert lesion mask. PH2 images are 8-bit RGB dermoscopic images acquired under consistent conditions with 20x magnification and 768x560 resolution. PH2 provides manual lesion segmentation and expert annotations for colors and several dermoscopic structures.

The main preprocessing principle is conservative masking/QC, not blanket image alteration. Any operation that changes color, line texture, or small structures can contaminate the exact FEs under study.

## Post-Reproducibility Status Note

This document preserves preprocessing research-stage recommendations. For current Feature Extraction implementation readiness, `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` is authoritative; older `READY_FOR_APPROVAL` language here means evidence-readiness for human review, not permission to implement code during the current checkpoint.

## Hair

### Evidence

DullRazor was proposed to detect dark hairs with morphological operations, replace hair pixels by interpolation, and smooth the result. It was designed largely to prevent hair from interfering with lesion analysis and segmentation.

Later hair-removal work improves dark/light hair handling but also reports important false positives: narrow hair-like dermoscopic structures, pigment-network lines, bubbles, and other artifacts can be marked as hair. This is especially dangerous for pigment network, streaks/radial structures, and dots/globules.

### Recommendation

`PARTIALLY_SUPPORTED`: do not apply global hair removal to every PH2 image. Add hair detection and a per-feature artifact mask. If hair intersects the lesion, exclude masked pixels from color statistics and use inpainting only for extractors that require line continuity. Preserve raw-image measurements and debug masks.

### Necessary for PH2?

Selective only. PH2 is controlled and masked; blanket DullRazor is not justified for all images.

### Thresholds

Hair-detection thresholds are `UNRESOLVED` for this project. Published algorithms are acceptable candidates, but their parameters must be copied exactly from source or approved.

## Specular Highlights / Reflections

### Evidence

Specular highlights and bright reflection artifacts can mimic white structures, inflate white color counts, disrupt blue-white veil detection, and create artificial high gradients or texture gaps. Dermoscopy preprocessing literature commonly handles highlights by detection, masking/exclusion, and sometimes inpainting.

### Recommendation

`READY_FOR_APPROVAL`: implement a reflection/highlight mask as QC and exclusion evidence. Use inpainting only for downstream line/texture continuity when explicitly needed. Do not classify saturated reflections as ABCD white, regression, or blue-white veil.

### Necessary for PH2?

Yes as QC/exclusion. Not necessarily as image repair.

### Thresholds

Exact saturated-bright/low-saturation threshold is `UNRESOLVED`.

## Illumination and Color Normalization

### Evidence

Color constancy and calibration are important in multi-source dermoscopy because devices and lighting shift diagnostic colors. PH2, however, was acquired under controlled and consistent conditions, so global color normalization is less necessary for a PH2-only experiment.

Relative color methods, especially for blue-white veil and blotches, estimate surrounding skin color from an annulus around the lesion. Celebi et al. omit a ring adjacent to the lesion and use a surrounding skin band to compute relative RGB features.

### Recommendation

`PARTIALLY_SUPPORTED`: for PH2-only deterministic FE, avoid global color normalization unless all downstream thresholds are defined after that normalization. Prefer fixed color-space conversion plus optional relative-to-skin features where the cited method requires them.

### Necessary for PH2?

Not globally. Necessary only for extractors whose published method relies on background skin color.

### Thresholds

Annulus definitions are method-specific; global color-constancy parameters for PH2 are `UNRESOLVED`.

## Other Artifacts

### Evidence

Dermoscopic images can include rulers, dark corners/vignetting, ink markings, air bubbles, and reflections. Robustness studies explicitly discuss artifact influence in datasets including PH2, HAM10000, and 7-point datasets.

### Recommendation

`READY_FOR_APPROVAL`: add artifact QC flags and per-feature exclusion masks. Ignore artifacts outside the lesion mask. If artifacts intersect the lesion, return raw feature values plus `artifact_confounded=true` or return `null` when calculation is not defensible.

### Necessary for PH2?

QC is necessary; universal repair is not.

## Sources

1. Lee T, Ng V, Gallagher R, Coldman A, McLean D. "DullRazor: a software approach to hair removal from images." Comput Biol Med. 1997. https://pubmed.ncbi.nlm.nih.gov/9437554/
2. Maglogiannis I, Delibasis K. "Hair removal on dermoscopy images." EMBC 2015. https://pubmed.ncbi.nlm.nih.gov/26736913/
3. Kiani K, Sharafat AR. "E-shaver: an improved DullRazor..." Comput Biol Med. 2011. https://doi.org/10.1016/j.compbiomed.2011.01.003
4. Kiani K, et al. "SharpRazor..." Skin Research and Technology. https://doi.org/10.1111/srt.13203
5. Celebi ME, et al. "Automatic Detection of Blue-White Veil..." https://pubmed.ncbi.nlm.nih.gov/18804955/
6. Barata C, Celebi ME, Marques JS. "Improving Dermoscopy Image Classification Using Color Constancy." IEEE J Biomed Health Inform. https://doi.org/10.1109/JBHI.2014.2336473
7. Mendonca T, et al. "PH2 - A dermoscopic image database for research and benchmarking." https://pubmed.ncbi.nlm.nih.gov/24110966/
8. Daneshjou R, et al. "Robustness of dermatology image classifiers to artifacts." Dermatology Practical & Conceptual. https://doi.org/10.5826/dpc.1203a126
