# Menzies Feature Extraction Research

## Post-Reproducibility Status Note

This document preserves the earlier research-stage evidence and provisional recommendations. For current implementation readiness, `docs/feature_extraction/adr/ADR-009 - Post-Reproducibility Implementation Readiness Checkpoint.md` and `docs/feature_extraction/adr/ADR-010 - Final Feature Extraction Readiness Before Coding.md` supersede any older `READY_FOR_APPROVAL` or `PENDING_HUMAN_APPROVAL` language here.

Active checkpoint summary: complete deterministic Menzies-r FE is `UNAVAILABLE`; blue-white veil has an implementation-ready Celebi pixel-level detector and `RAW_EVIDENCE` area fraction, but the Menzies-r binary clinical criterion remains `UNRESOLVED`; network, radial structures, and dots/globules remain unavailable or raw-only as specified in ADR-009 and ADR-010; Stoecker blotch evidence is auxiliary only for Menzies color non-uniformity.

## Compatibility with the Current Project

The current `MenziesDiagnosticAgent` asks for negative criteria (`simetria`, `cor_unica`) and positive criteria including blue-white veil, brown dots, pseudopods, radial streaming, scar-like depigmentation, peripheral black dots/globules, multiple colors, blue-gray dots, and branched/broadened network-like lines. The Pydantic schema currently omits some positive fields that the prompt names; future implementation should reconcile the schema and prompt, but this sprint does not modify code.

The Menzies rule is a gated rule: melanoma is suggested when both negative features are absent and at least one positive feature is present. Original sources report high sensitivity but moderate specificity.

## Negative Criteria

### Pattern Asymmetry

Clinical concept: Menzies pattern symmetry refers to symmetry of color and dermoscopic structures, not merely external contour.

Candidate implementation: reuse ABCD asymmetry extractor only as partial evidence. A Menzies-compatible extractor should compare mirrored color and structure distributions across axes through the lesion center.

Raw outputs: shape reflection asymmetry, color histogram divergence by axis, structure-mask divergence by axis, best-axis symmetry score.

Threshold: `UNRESOLVED`. Clinical rule is binary, but no universal numeric threshold was found.

Review status: `PARTIALLY_SUPPORTED`.

### Color Uniformity / Non-Uniformity

Clinical concept: single color is a negative feature. Menzies color set commonly includes black, gray, blue, red, dark brown, and tan; white is not counted as a Menzies color in several descriptions.

Candidate implementation: share a six-color classifier with ABCD but expose a Menzies-specific count that excludes white and maps light brown to tan.

Raw outputs: per-color pixel fractions, color count including/excluding white, dominant color, entropy.

Threshold: `single_color = count == 1` is clinically supported. Minimum pixel area for "present color" is `UNRESOLVED`.

Review status: `PARTIALLY_SUPPORTED`.

## Positive Criteria

| Criterion | Clinical Concept | Candidate Implementation | Threshold | Shared Extractor | Review |
|---|---|---|---|---|---|
| Blue-white veil | Irregular structureless confluent blue-white pigmentation with haze. | Reuse the implementation-ready Celebi pixel-level detector from 7PCL. Store candidate mask and `S1` / area ratio as raw evidence. | Celebi thresholds are classifier-derived and not universal clinical thresholds; `S1 = 0.009` and `S1 > 0` are not approved clinical presence rules; no PH2 calibration. | 7PCL blue-whitish veil; ABCD color/structureless evidence. | Pixel detector `IMPLEMENTATION_READY`; Menzies-r binary criterion `UNRESOLVED`. |
| Multiple colors / 5-6 colors | Presence of 5 or 6 Menzies colors. | Reuse color classifier; count Menzies colors excluding white. | `count >= 5` supported; minimum area per color `UNRESOLVED`. | ABCD color. | `PARTIALLY_SUPPORTED`. |
| Multiple blue-gray dots | Multiple pepper-like blue-gray dots. | Reuse dots/globules detector plus blue-gray color classification. | Count/size threshold `UNRESOLVED`. | 7PCL dots/globules; regression/peppering. | `UNRESOLVED`. |
| Broadened network | Thickened/broadened pigment network. | Reuse pigment-network detector; add line-thickness/hole-size stats. | Broadening threshold `UNRESOLVED`. | 7PCL atypical pigment network; ABCD pigment network. | `UNRESOLVED`. |
| Peripheral black dots/globules | Black round/oval structures near lesion periphery. | Reuse blob detector; classify black; compute distance-to-border/peripheral band. | Peripheral band/count threshold `UNRESOLVED`. | 7PCL irregular dots/globules; ABCD dots/globules. | `UNRESOLVED`. |
| Radial streaming | Radial linear extensions at lesion edge. | Reuse streak detector; separate linear from bulbous endings. | Presence/focality threshold `UNRESOLVED`. | 7PCL irregular streaks; ABCD branched streaks. | `UNRESOLVED`. |
| Pseudopods | Bulbous projections attached to lesion edge by stalk. | Reuse streak detector plus terminal bulb/stalk geometry. | Presence/focality threshold `UNRESOLVED`. | 7PCL irregular streaks; ABCD branched streaks. | `UNRESOLVED`. |
| Multiple brown dots | Multiple brown dots distributed in lesion. | Reuse blob detector plus brown color classification. | Count/area threshold `UNRESOLVED`. | Dots/globules. | `UNRESOLVED`. |
| Scar-like depigmentation | White scar-like regression area. | Could share white/structureless/regression candidate masks. | `UNRESOLVED`; current reduced requested FE does not prioritize it. | 7PCL regression if later restored. | `NEEDS_MORE_RESEARCH`. |

## Comparison

Menzies shares most extractors with ABCD and 7PCL, but not all definitions are equivalent. Blue-white veil is the strongest direct shared feature. Color count is shared with ABCD but uses a different color set and decision rule. Pattern asymmetry should not be reduced to shape-only ABCD asymmetry. Broadened network is narrower than "atypical pigment network"; radial streaming and pseudopods are subtypes of streaks.

## Provisional Recommendation

Use Menzies as a consumer of reusable extractors rather than as a separate extractor family:

- `asymmetry`: consume raw shape/color/structure asymmetry; decision remains `UNRESOLVED`.
- `color`: consume ABCD/Menzies color-count extractor; decision is partially supported.
- `blue-white veil`: consume 7PCL detector; raw outputs ready for approval.
- `dots/globules`, `network`, and `streaks`: consume shared raw detectors; final Menzies positives remain unresolved without approved thresholds.

## Gaps

The largest unresolved issue is binary conversion. Menzies criteria are clinically binary, but most computational papers provide classifier-derived or dataset-derived cutoffs. Public PH2 annotations do not label every Menzies subtype, particularly broadened network, pseudopods, radial streaming, multiple brown dots, and peripheral black dots/globules.

## Sources

1. Menzies SW, et al. "Frequency and morphologic characteristics of invasive melanomas lacking specific surface microscopic features." Arch Dermatol. 1996. https://doi.org/10.1001/archderm.1996.03890340038007
2. Menzies SW. "A method for the diagnosis of primary cutaneous melanoma using surface microscopy." Dermatol Clin. 2001. https://doi.org/10.1016/S0733-8635(05)70267-9
3. DermNet NZ. "Other algorithms for melanocytic lesions." https://dermnetnz.org/cme/dermoscopy-course/other-algorithms-for-melanocytic-lesions
4. Jaimes N, et al. "Using Dermoscopy to Identify Melanoma and Improve Diagnostic Discrimination." https://pmc.ncbi.nlm.nih.gov/articles/PMC6375419/
5. Argenziano A, et al. "The Significance of Eccentric and Central Hyperpigmentation..." https://jamanetwork.com/journals/DERM/articlepdf/420122/dst70069_1440_1444.pdf
6. Celebi ME, et al. "Automatic Detection of Blue-White Veil and Related Structures in Dermoscopy Images." https://pubmed.ncbi.nlm.nih.gov/18804955/
7. Anantha M, et al. "Detection of pigment network in dermatoscopy images using texture analysis." https://pubmed.ncbi.nlm.nih.gov/15249068/
8. Sadeghi M, et al. "Detection and analysis of irregular streaks..." https://pubmed.ncbi.nlm.nih.gov/23335664/
9. Kittler H, et al. "Standardization of terminology in dermoscopy/dermatoscopy." https://pmc.ncbi.nlm.nih.gov/articles/PMC5551974/
