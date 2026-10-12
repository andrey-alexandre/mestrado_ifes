# ADR-011 - ABCD B Author Operationalization Closure

**Status:** Accepted  
**Date:** 2026-09-10  
**Scope:** Final ABCD-B implementation and documentation reconciliation after synthetic methodological experiments

## Context

ADR-003 approved Kasmi and Mokrani (2016) as the ABCD-B source basis with limitation. Subsequent implementation-resolution experiments showed that exact executable reproduction remains blocked by a source specification gap: the source states `error > 0.05 x length(y)` but does not fully define the executable contour-vector meaning of `y` or `length(y)`.

The dissertation therefore must distinguish source-specified components from author implementation conventions and synthetic empirical validation.

## Decision

ABCD-B exact source reproduction is:

`SOURCE_EXACT_IMPLEMENTATION = UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`

The dissertation ABCD-B extractor is:

`AUTHOR_OPERATIONALIZATION = IMPLEMENTED_WITH_AUTHOR_OPERATIONALIZATION`

Runtime FE status remains:

`DETERMINISTIC_CLINICAL_WITH_LIMITATION`

The final implemented pipeline is:

1. GT binary lesion mask.
2. Ordered external 2D lesion contour.
3. Lesion centroid.
4. Lesion principal-axis orientation from second-order central moments through `principal_axis_analysis(mask)`.
5. Principal-axis-relative eight 45-degree sector assignment.
6. Ordered local parametric contour runs.
7. Cumulative arc-length parameterization.
8. Dimensionless cubic smoothing splines with local chord normalization and `spline_s = alpha * n_samples`, `alpha = 1.0`.
9. Euclidean geometric RMSE in image space.
10. `L_smooth` normalization.
11. `E_smooth > 0.05`.
12. `B = count(irregular sectors)`, bounded to `0..8`.

## Provenance

`SOURCE_SPECIFIED`:

- eight 45-degree sectors;
- cubic/third-order spline concept;
- comparison of real border with fitted spline using fitting error/RMSE;
- 5% threshold form;
- B obtained by counting positive sectors;
- final B range 0-8.

`AUTHOR_IMPLEMENTATION_CONVENTION`:

- GT binary mask input;
- ordered external 2D contour extraction;
- centroid-based angular location;
- principal-axis-aligned sector frame;
- second-order central-moment orientation shared with ABCD-A;
- contiguous ordered contour runs;
- cumulative arc-length parameterization;
- parametric x(s), y(s) contour representation;
- Euclidean geometric RMSE;
- dimensionless coordinate normalization;
- local chord `D_i`;
- deterministic degenerate-chord fallback;
- `alpha = 1.0`;
- `spline_s = alpha * n_samples`;
- `L_smooth` denominator;
- modulo/indexing conventions.

`EMPIRICALLY_VALIDATED_CONVENTION` means validated only on controlled synthetic geometric cases. It applies to:

- ordered parametric contour avoiding radial-collapse failures in deep concavities;
- rejection of RMSE/n because of sampling-density dependence;
- evidence favoring `L_smooth` over `L_real` because `L_real` grows with irregularity and partially absorbs signal;
- dimensionless smoothing as dimensionally coherent and sampling-normalized;
- principal-axis alignment materially improving rotation robustness;
- 180-degree and 90-degree axis ambiguity preserving total B in tests;
- stable B in tested regular-circle underdetermined-axis control;
- scale stability for tested irregular synthetic cases;
- localized-three-octants `B=2` explained as threshold-edge behavior with no implementation defect.

## Experimental Closure

Final experiment labels:

- `ADOPT_DIMENSIONLESS_SMOOTHING`
- `ADOPT_PRINCIPAL_AXIS_SECTOR_ALIGNMENT`
- `ROTATION_ROBUSTNESS_MATERIALLY_IMPROVED`
- `NORMALIZATION_EVIDENCE_FOR_L_SMOOTH_CAN_NOW_BE_REEVALUATED`
- `B2_THRESHOLD_EDGE_EFFECT`
- `NO_IMPLEMENTATION_DEFECT_FOUND`

The ABCD-B experimental phase is closed for this dissertation. Do not tune alpha, sector phase, threshold, denominator, synthetic perturbations, or principal-axis handling without a new explicit human decision.

## Limitations

The method is a geometric contour-irregularity proxy for abrupt pigment cutoff, not a direct lesion-to-skin pigment-transition measurement. It depends on GT mask quality. Synthetic validation does not establish clinical validity. Principal-axis orientation is underdetermined for rotationally symmetric lesions. Binary thresholding near `E_smooth = 0.05` can change sector classification after legitimate sector repartitioning or spline response.

## References

- Kasmi R, Mokrani K. "Classification of malignant melanoma and benign skin lesions: implementation of automatic ABCD rule." IET Image Processing. 2016;10(6):448-455. DOI `10.1049/iet-ipr.2015.0385`.
- Methodology record: `docs/abcd_border_methodology.md`.
