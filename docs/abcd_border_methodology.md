# ABCD B Border Methodology

## Final Status

ABCD-B is implemented as a source-guided computational operationalization, not as an exact executable reproduction of Kasmi and Mokrani (2016). Runtime status remains `DETERMINISTIC_CLINICAL_WITH_LIMITATION`; scientific status is:

- `SOURCE_EXACT_IMPLEMENTATION`: `UNAVAILABLE_DUE_TO_SPECIFICATION_GAP`
- `AUTHOR_OPERATIONALIZATION`: `IMPLEMENTED_WITH_AUTHOR_OPERATIONALIZATION`

The source supports the eight-sector spline-error structure and `B in {0,...,8}`, but does not fully specify the executable meaning of `y` and `length(y)` in `RMSE > 0.05 * length(y)`. The adopted implementation preserves that source gap explicitly and uses deterministic author conventions tested on controlled synthetic geometric cases.

## Clinical And Source Background

Clinical ABCD border scoring concerns border irregularity and abrupt pigment cutoff. This implementation measures geometric contour irregularity from a binary lesion mask, so it is a computational proxy and does not directly measure lesion-to-skin pigment transition.

Kasmi and Mokrani (2016) support dividing the lesion contour into eight 45-degree sectors, fitting third-order splines to sub-contours, comparing the real border with the fitted spline using fitting error/RMSE, applying a 5% `length(y)` threshold form, and summing positive sectors to produce `B` from 0 to 8.

Kaya et al. (2016) and Erol et al. (2017) remain auxiliary evidence for pigment-transition abruptness concepts only. They are not combined with the Kasmi and Mokrani path to create a new ABCD-B formula.

## Adopted Pipeline

```text
GT binary lesion mask
  -> ordered 2D lesion contour
  -> lesion centroid
  -> lesion principal-axis orientation
  -> principal-axis-aligned angular reference
  -> eight 45-degree sectors
  -> ordered local parametric contour runs
  -> cumulative arc-length parameterization
  -> dimensionless cubic smoothing spline
  -> Euclidean geometric RMSE
  -> normalization by smoothed local contour length
  -> E_smooth > 0.05
  -> B = count(irregular sectors), B in {0,...,8}
```

The principal-axis orientation is shared with ABCD-A through `src/features/geometry.py::principal_axis_analysis(mask)` and is derived from lesion-pixel second-order central moments:

```text
phi = 0.5 * atan2(2 * mu11, mu20 - mu02)
```

This is mathematically equivalent to the principal eigenvector orientation of the lesion-pixel covariance matrix. No PCA library is used.

Sector assignment uses the original contour coordinates and changes only sector membership:

```text
theta_image = atan2(y - cy, x - cx)
theta_relative = mod(theta_image - phi, 2*pi)
sector = floor(theta_relative / (pi/4))
```

## Parametric Spline

For each contiguous ordered local contour run, cumulative Euclidean arc length is:

```text
s_j = cumulative length along the run
u_j = s_j / s_max
```

The primary local scale is the sub-contour chord:

```text
D_i = ||p_end - p_start||
```

Coordinates are normalized by:

```text
x'_j = (x_j - x_ref) / D_i
y'_j = (y_j - y_ref) / D_i
```

Degenerate chords use the deterministic fallback already implemented: local arc length is used as the scale and the fallback is recorded in raw evidence.

Cubic smoothing splines are fit as `x'(u)` and `y'(u)` with:

```text
spline_s = alpha * n_samples
alpha = 1.0
```

Fitted coordinates are transformed back to image space. The residual is Euclidean:

```text
d_j = sqrt((x_j - x_hat_j)^2 + (y_j - y_hat_j)^2)
RMSE_geom = sqrt(mean(d_j^2))
```

The final normalization is:

```text
L_smooth = local arc length of the fitted smoothed contour
E_smooth = RMSE_geom / L_smooth
irregular_i = E_smooth > 0.05
B = sum(irregular_i)
```

The threshold is source-guided by the 5% form, but `RMSE / L_smooth > 0.05` is an author operationalization because the source does not uniquely define executable `length(y)`.

## Provenance

| Component | Final implementation | Provenance | Evidence/status |
|---|---|---|---|
| Sector count | 8 sectors of 45 degrees | `SOURCE_SPECIFIED` | Kasmi and Mokrani |
| Cubic spline concept | cubic/third-order spline per sector | `SOURCE_SPECIFIED` | executable details incomplete |
| Spline error concept | compare real border with fitted spline using RMSE/fitting error | `SOURCE_SPECIFIED` | Kasmi and Mokrani |
| Final B aggregation | count positive sectors, `B in {0,...,8}` | `SOURCE_SPECIFIED` | Kasmi and Mokrani |
| GT mask input | binary lesion mask as geometry source | `AUTHOR_IMPLEMENTATION_CONVENTION` | PH2 masks used as inputs, not calibration labels |
| Contour representation | ordered external 2D contour from marching squares | `AUTHOR_IMPLEMENTATION_CONVENTION` | `EMPIRICALLY_VALIDATED_CONVENTION` on concavity tests |
| Sector orientation | principal-axis-aligned angular frame | `AUTHOR_IMPLEMENTATION_CONVENTION` | `EMPIRICALLY_VALIDATED_CONVENTION` on rotation tests |
| Principal axis | second-order central-moment orientation shared with ABCD-A | `AUTHOR_IMPLEMENTATION_CONVENTION` | code-shared geometry utility |
| Contour parameter | cumulative arc length `s`, normalized to `u` | `AUTHOR_IMPLEMENTATION_CONVENTION` | sampling-density tests |
| Spline scaling | dimensionless coordinates and `spline_s = alpha * n_samples` | `AUTHOR_IMPLEMENTATION_CONVENTION` | density/scale experiment |
| Residual | Euclidean geometric RMSE in image space | `AUTHOR_IMPLEMENTATION_CONVENTION` | parametric geometry tests |
| Denominator | `L_smooth` fitted local contour length | `AUTHOR_IMPLEMENTATION_CONVENTION` | L_real/L_smooth comparison |
| Threshold | `E_smooth > 0.05` | `AUTHOR_OPERATIONALIZATION` | source-guided 5% form; exact `length(y)` gap preserved |
| Legacy radial path | comparator only | `HISTORICAL_EXPERIMENTAL_COMPARATOR` | rejected for geometry limitations |

`EMPIRICALLY_VALIDATED_CONVENTION` means validated on controlled synthetic geometric cases only. It does not imply clinical validation.

## Synthetic Evidence Summary

Radial angle-radius representation was rejected as the final path because it can collapse duplicate angular contour information and cannot faithfully represent deep concavities.

The ordered parametric contour preserves the 2D contour and avoids requiring unique angular or unique x/y representation.

Dimensionless smoothing was adopted because it is dimensionally coherent and sampling-normalized. At `alpha = 1.0`, the legacy comparator `spline_s = alpha * n_samples * polyline_length` and the dimensionless formulation produced numerically identical outputs on tested synthetic cases, consistent with a smoothing plateau; adoption is not based on improved B scores.

Sampling-density tests showed `RMSE/n` changes strongly with contour resampling density, while geometric ratios remained approximately stable. This rejects sample count as the final geometric normalization scale.

The L_real/L_smooth comparison showed `L_real / L_smooth` increases with irregularity:

| Shape | L_real/L_smooth |
|---|---:|
| regular ellipse | about 1.058 |
| localized irregular | about 1.138 |
| moderate irregular | about 1.192 |
| strong serrated | about 1.287 |

Because `L_real` grows partly from the same border irregularity being measured, dividing by `L_real` partially absorbs the target signal. Normalization evidence favors `L_smooth`; this is not a claim of universal optimality.

Principal-axis sector alignment materially improved rotation robustness across rotations 0 to 45 degrees in 5-degree steps:

| Shape | IMAGE_FIXED B sequence | PRINCIPAL_AXIS_ALIGNED B sequence |
|---|---|---|
| localized | `[3,2,1,0,0,0,0,1,2,3]` | `[2,2,2,2,2,2,2,2,2,2]` |
| moderate | `[3,1,1,2,2,1,1,1,1,3]` | `[3,3,3,3,3,3,3,3,3,3]` |
| strong | `[8,6,4,4,4,4,4,4,4,8]` | `[8,8,8,8,8,8,8,8,8,8]` |

Axis equivariance error remained below approximately 0.4 degrees in these principal irregular synthetic cases. The 180-degree principal-axis ambiguity and 90-degree axis exchange only permuted sector labels and preserved total B in tests.

Near-isotropic controls showed the expected axis confidence behavior: clearly elongated ellipse anisotropy about 0.5510, near-circular ellipse about 0.0669, circle 0.0000, and near-circular localized irregularity about 0.0431. The perfect circle is reported as `PRINCIPAL_AXIS_UNDERDETERMINED`, but B remained stable in the tested regular-circle control. No anisotropy switching threshold was adopted.

The localized three-perturbation diagnostic found that the PCA-aligned synthetic score `B=2` is a threshold-edge effect, not an implementation defect. P1 and P2 exceed threshold; P3 is fully represented in sector 6 but falls below the fixed 0.05 threshold:

| Perturbation | Sector | E_smooth | Margin vs 0.05 | Classification |
|---|---:|---:|---:|---|
| P1 | 1 | about 0.055708 | about +0.005708 | irregular |
| P2 | 4 | about 0.089125 | about +0.039125 | irregular |
| P3 | 6 | about 0.046046 | about -0.003954 | not irregular |

Diagnostic classification: `B2_THRESHOLD_EDGE_EFFECT`; audit result: `NO_IMPLEMENTATION_DEFECT_FOUND`.

Detailed artifacts:

- `results/abcd_border_parametric_diagnostics/`
- `results/abcd_border_smooth_length_experiment/`
- `results/abcd_border_dimensionless_smoothing_experiment/`
- `results/abcd_border_principal_axis_orientation_experiment/`
- `results/abcd_border_localized_three_octants_diagnostic/`

## Limitations

1. Source operationalization gap: `RMSE > 0.05 * length(y)` is insufficiently specified for exact reproduction.
2. Clinical approximation: the implementation measures binary-mask geometric contour irregularity, not direct pigment-transition abruptness.
3. GT mask dependence: segmentation uncertainty is not evaluated by this feature.
4. Threshold edge effect: values near `E_smooth = 0.05` can change classification after legitimate sector repartitioning or spline response; P3 is the observed synthetic example.
5. Principal-axis underdetermination: rotationally symmetric lesions can have undefined principal orientation; stability is not guaranteed for all isotropic irregular lesions.
6. Synthetic validation scope: experiments demonstrate geometric and numerical behavior only, not clinical validity.
7. No exact Kasmi reproduction claim: the final implementation is a source-guided operationalization with explicit author conventions.

## Output Contract

`extract_abcd_border(mask)` returns:

- `status = DETERMINISTIC_CLINICAL_WITH_LIMITATION`;
- `value = B`, an integer in `0..8`;
- raw evidence including centroid, principal-axis diagnostics, sector orientation, contour count, spline convention, local chord scale, per-sector sample count, RMSE, `L_smooth`, threshold margin, irregular flag, and insufficient-points flag;
- limitations preserving the source gap, geometric-proxy status, no PH2 calibration, and synthetic-only validation scope.
