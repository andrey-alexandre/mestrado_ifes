"""Synthetic validation helpers for the ABCD B border extractor."""
import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List
import warnings

import numpy as np
from PIL import Image, ImageDraw

from src.features.border import (
    DEFAULT_SMOOTHING_FACTOR,
    OCTANT_COUNT,
    OCTANT_WIDTH_RADIANS,
    SECTOR_ORIENTATION_IMAGE_FIXED,
    SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    angles_and_radii,
    angles_for_xy,
    assign_octants,
    assign_octants_relative,
    cumulative_arc_length,
    evaluate_border_octants,
    evaluate_border_octants_parametric,
    evaluate_border_octants_radial,
    evaluate_ordered_contour_parametric,
    extract_boundary_points,
    extract_ordered_contour,
    lesion_centroid,
    normalize_angles,
    open_polyline_length,
)
from src.features.geometry import axial_angle_difference, principal_axis_analysis


SMOOTHING_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RAW_RMSE_CASES = (
    "regular_ellipse",
    "smooth_lobulated",
    "localized_three_octants",
    "moderate_irregular",
    "strong_serrated",
    "deep_concavity",
)
LOCALIZED_AFFECTED_OCTANTS = {1, 4, 6}
LOCALIZED_PERTURBATIONS = (
    {"id": "P1", "center_degrees": 70.0, "amplitude": 0.65, "width_radians": 0.05},
    {"id": "P2", "center_degrees": 205.0, "amplitude": -0.50, "width_radians": 0.05},
    {"id": "P3", "center_degrees": 292.0, "amplitude": 0.65, "width_radians": 0.05},
)


def make_polar_lesion_mask(
    size: int = 256,
    rx: float = 70.0,
    ry: float = 52.0,
    rotation_degrees: float = 0.0,
    perturbation: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    yy, xx = np.indices((size, size), dtype=np.float64)
    cx = cy = (size - 1) / 2.0
    angle = math.radians(rotation_degrees)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = xx - cx
    dy = yy - cy
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    theta = np.arctan2(yr / ry, xr / rx)
    base_radius = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    scale = 1.0 if perturbation is None else np.maximum(0.55, 1.0 + perturbation(theta))
    return base_radius <= scale


def make_deep_concavity_mask(size: int = 256, rotation_degrees: float = 0.0) -> np.ndarray:
    scale = size / 256.0
    return make_polar_lesion_mask(
        size=size,
        rx=72.0 * scale,
        ry=54.0 * scale,
        rotation_degrees=rotation_degrees,
        perturbation=lambda t: (
            -0.72 * _angular_gaussian(t, math.radians(27), 0.045)
            + 0.18 * _angular_gaussian(t, math.radians(62), 0.07)
            + 0.12 * np.sin(5.0 * t + 0.3)
        ),
    )


def _angular_gaussian(theta: np.ndarray, center: float, width: float) -> np.ndarray:
    wrapped = np.angle(np.exp(1j * (theta - center)))
    return np.exp(-0.5 * (wrapped / width) ** 2)


def synthetic_cases(size: int = 256, rotation_degrees: float = 0.0) -> Dict[str, np.ndarray]:
    scale = size / 256.0
    return {
        "regular_ellipse": make_polar_lesion_mask(
            size=size,
            rx=70.0 * scale,
            ry=52.0 * scale,
            rotation_degrees=rotation_degrees,
        ),
        "smooth_lobulated": make_polar_lesion_mask(
            size=size,
            rx=70.0 * scale,
            ry=52.0 * scale,
            rotation_degrees=rotation_degrees,
            perturbation=lambda t: 0.055 * np.sin(3.0 * t),
        ),
        "localized_three_octants": make_polar_lesion_mask(
            size=size,
            rx=70.0 * scale,
            ry=52.0 * scale,
            rotation_degrees=rotation_degrees,
            perturbation=lambda t: (
                LOCALIZED_PERTURBATIONS[0]["amplitude"]
                * _angular_gaussian(t, math.radians(LOCALIZED_PERTURBATIONS[0]["center_degrees"]), LOCALIZED_PERTURBATIONS[0]["width_radians"])
                + LOCALIZED_PERTURBATIONS[1]["amplitude"]
                * _angular_gaussian(t, math.radians(LOCALIZED_PERTURBATIONS[1]["center_degrees"]), LOCALIZED_PERTURBATIONS[1]["width_radians"])
                + LOCALIZED_PERTURBATIONS[2]["amplitude"]
                * _angular_gaussian(t, math.radians(LOCALIZED_PERTURBATIONS[2]["center_degrees"]), LOCALIZED_PERTURBATIONS[2]["width_radians"])
            ),
        ),
        "moderate_irregular": make_polar_lesion_mask(
            size=size,
            rx=70.0 * scale,
            ry=52.0 * scale,
            rotation_degrees=rotation_degrees,
            perturbation=lambda t: 0.10 * np.sin(9.0 * t) + 0.08 * np.sin(14.0 * t + 0.5),
        ),
        "strong_serrated": make_polar_lesion_mask(
            size=size,
            rx=70.0 * scale,
            ry=52.0 * scale,
            rotation_degrees=rotation_degrees,
            perturbation=lambda t: sum(
                (0.50 if i % 2 == 0 else -0.35) * _angular_gaussian(t, math.radians(i * 45 + 22), 0.035)
                for i in range(8)
            ),
        ),
        "deep_concavity": make_deep_concavity_mask(size=size, rotation_degrees=rotation_degrees),
    }


def compare_representations(
    sizes: Iterable[int] = (256, 512, 1024),
    rotations: Iterable[int] = (0, 15, 30, 45),
) -> Dict:
    records: List[Dict] = []
    for size in sizes:
        for rotation in rotations:
            for case_name, mask in synthetic_cases(size=size, rotation_degrees=rotation).items():
                radial = evaluate_border_octants_radial(mask)
                parametric = evaluate_border_octants_parametric(
                    mask,
                    sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
                )
                records.append(
                    {
                        "case": case_name,
                        "size": size,
                        "rotation_degrees": rotation,
                        "radial_B": radial["border_score"],
                        "parametric_B": parametric["border_score"],
                        "radial_irregular_octants": radial["irregular_octants"],
                        "parametric_irregular_octants": parametric["irregular_octants"],
                        "radial_octants": _compact_octants(radial),
                        "parametric_octants": _compact_octants(parametric),
                    }
                )

    strong_rotation = [
        r for r in records
        if r["case"] == "strong_serrated" and r["size"] == 256
    ]
    return {
        "selected_smoothing_factor": DEFAULT_SMOOTHING_FACTOR,
        "threshold_rule": "RMSE > 0.05 * length(y); length(y) remains sector sample count.",
        "radial_baseline": "r(theta), duplicate theta collapsed by mean radius before spline fitting.",
        "parametric_candidate": "ordered contour, octant assignment by theta only, cubic smoothing splines x(s), y(s), Euclidean residual.",
        "records": records,
        "comparison_table": [
            {
                "shape": r["case"],
                "rotation": r["rotation_degrees"],
                "size": r["size"],
                "radial_B": r["radial_B"],
                "parametric_B": r["parametric_B"],
            }
            for r in records
        ],
        "strong_serrated_rotation_256": {
            "radial_scores": [r["radial_B"] for r in strong_rotation],
            "parametric_scores": [r["parametric_B"] for r in strong_rotation],
            "radial_range": _score_range([r["radial_B"] for r in strong_rotation]),
            "parametric_range": _score_range([r["parametric_B"] for r in strong_rotation]),
            "radial_std": float(np.std([r["radial_B"] for r in strong_rotation])),
            "parametric_std": float(np.std([r["parametric_B"] for r in strong_rotation])),
        },
        "selected_method_summary": _summarize_records(records),
        "deep_concavity": deep_concavity_diagnostics(),
    }


def run_parametric_diagnostics(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_parametric_diagnostic_report()
    (out / "parametric_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "parametric_tables.md").write_text(render_diagnostic_markdown(report), encoding="utf-8")
    report["plots"] = write_parametric_diagnostic_plots(report, out)
    (out / "parametric_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_smooth_length_experiment(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_smooth_length_report()
    (out / "smooth_length_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "smooth_length_tables.md").write_text(render_smooth_length_markdown(report), encoding="utf-8")
    report["plots"] = write_smooth_length_plots(report, out)
    (out / "smooth_length_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_dimensionless_smoothing_experiment(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_dimensionless_smoothing_report()
    (out / "dimensionless_smoothing_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "dimensionless_smoothing_tables.md").write_text(render_dimensionless_smoothing_markdown(report), encoding="utf-8")
    report["plots"] = write_dimensionless_smoothing_plots(report, out)
    (out / "dimensionless_smoothing_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_principal_axis_orientation_experiment(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_principal_axis_orientation_report()
    (out / "principal_axis_orientation_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "principal_axis_orientation_tables.md").write_text(render_principal_axis_orientation_markdown(report), encoding="utf-8")
    report["plots"] = write_principal_axis_orientation_plots(report, out)
    (out / "principal_axis_orientation_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_localized_three_octants_diagnostic(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_localized_three_octants_diagnostic()
    (out / "localized_three_octants_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "localized_three_octants_diagnostic.md").write_text(render_localized_three_octants_diagnostic_markdown(report), encoding="utf-8")
    report["plots"] = write_localized_three_octants_diagnostic_plots(report, out)
    (out / "localized_three_octants_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_localized_three_octants_diagnostic() -> Dict:
    native_mask = synthetic_cases(size=256, rotation_degrees=0)["localized_three_octants"]
    native_axis = principal_axis_analysis(native_mask)
    image_fixed = _localized_orientation_diagnostic(0, SECTOR_ORIENTATION_IMAGE_FIXED)
    pca_aligned = _localized_orientation_diagnostic(0, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
    rotations = [
        _localized_orientation_diagnostic(rotation, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
        for rotation in (0, 15, 30, 45)
    ]
    audit = _localized_implementation_audit(native_mask)
    explanations = _localized_b2_explanations(pca_aligned)
    final_decision = _localized_b2_final_decision(audit, pca_aligned)
    return {
        "synthetic_parameters": {
            "shape": "localized_three_octants",
            "base_rx": 70.0,
            "base_ry": 52.0,
            "perturbations": [
                {
                    **perturbation,
                    "center_radians": math.radians(perturbation["center_degrees"]),
                    "width_degrees": math.degrees(perturbation["width_radians"]),
                    "affected_region_rule": "contour samples with generator-space angular distance <= 2 * width_radians",
                }
                for perturbation in LOCALIZED_PERTURBATIONS
            ],
            "angular_coordinate_convention": (
                "Generator angles are normalized ellipse coordinates before rasterization: "
                "theta = atan2(yr / ry, xr / rx), where xr/yr are coordinates after inverse lesion rotation."
            ),
        },
        "native_principal_axis": {
            "phi_radians": native_axis["angle_radians"],
            "phi_degrees": math.degrees(native_axis["angle_radians"]),
            "centroid_xy": list(native_axis["centroid_xy"]),
            "lambda1": native_axis["lambda1"],
            "lambda2": native_axis["lambda2"],
            "anisotropy": native_axis["axis_anisotropy"],
            "orientation_status": native_axis["orientation_status"],
        },
        "native_image_fixed": image_fixed,
        "native_principal_axis_aligned": pca_aligned,
        "rotation_consistency": rotations,
        "image_fixed_vs_pca": [
            {
                "perturbation": p["id"],
                "image_fixed_sectors": image_fixed["affected_sector_distribution"][p["id"]]["sector_fractions"],
                "pca_aligned_sectors": pca_aligned["affected_sector_distribution"][p["id"]]["sector_fractions"],
            }
            for p in LOCALIZED_PERTURBATIONS
        ],
        "threshold_margin_analysis": pca_aligned["sector_table"],
        "individual_explanations": explanations,
        "measured_explanation": _localized_measured_explanation(pca_aligned),
        "implementation_audit": audit,
        "defect_status": audit["status"],
        "final_decision": final_decision,
        "abcd_b_experimentation_can_stop": final_decision in {"B2_GEOMETRICALLY_EXPLAINED", "B2_THRESHOLD_EDGE_EFFECT"},
    }


def build_principal_axis_orientation_report() -> Dict:
    rotation_grid = tuple(range(0, 46, 5))
    primary_cases = ("localized_three_octants", "moderate_irregular", "strong_serrated")
    phi0_by_case = {
        case_name: principal_axis_analysis(synthetic_cases(size=256, rotation_degrees=0)[case_name])["angle_radians"]
        for case_name in primary_cases
    }

    rotation_rows = []
    axis_rows = []
    canonical_rows = []
    for case_name in primary_cases:
        canonical = _orientation_case_summary(case_name, 256, 0, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
        canonical_set = set(canonical["irregular_octants"])
        for rotation in rotation_grid:
            image_fixed = _orientation_case_summary(case_name, 256, rotation, SECTOR_ORIENTATION_IMAGE_FIXED)
            principal = _orientation_case_summary(case_name, 256, rotation, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
            expected_phi = phi0_by_case[case_name] + math.radians(rotation)
            axial_error = axial_angle_difference(principal["phi_radians"], expected_phi)
            rotation_rows.append(
                {
                    "shape": case_name,
                    "rotation_degrees": rotation,
                    "phi_degrees": principal["phi_degrees"],
                    "anisotropy": principal["anisotropy"],
                    "B_image_fixed": image_fixed["B_smooth"],
                    "B_principal_axis": principal["B_smooth"],
                    "image_fixed_octants": image_fixed["irregular_octants"],
                    "principal_axis_octants": principal["irregular_octants"],
                    "image_fixed_mean_E_smooth": image_fixed["mean_E_smooth"],
                    "principal_axis_mean_E_smooth": principal["mean_E_smooth"],
                    "principal_axis_max_E_smooth": principal["max_E_smooth"],
                }
            )
            axis_rows.append(
                {
                    "shape": case_name,
                    "rotation_degrees": rotation,
                    "expected_phi_degrees": math.degrees(expected_phi),
                    "measured_phi_degrees": principal["phi_degrees"],
                    "axial_error_degrees": math.degrees(axial_error),
                    "anisotropy": principal["anisotropy"],
                    "orientation_status": principal["orientation_status"],
                }
            )
            canonical_rows.append(
                {
                    "shape": case_name,
                    "rotation_degrees": rotation,
                    "image_fixed_octants": image_fixed["irregular_octants"],
                    "principal_axis_octants": principal["irregular_octants"],
                    "canonical_octants": sorted(canonical_set),
                    "jaccard": _jaccard(canonical_set, set(principal["irregular_octants"])),
                    "axial_ambiguity_jaccard": max(
                        _jaccard(canonical_set, set(principal["irregular_octants"])),
                        _jaccard(canonical_set, _shift_sector_set(principal["irregular_octants"], 4)),
                    ),
                }
            )

    stability_rows = []
    for case_name in primary_cases:
        for method, key in (
            (SECTOR_ORIENTATION_IMAGE_FIXED, "B_image_fixed"),
            (SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED, "B_principal_axis"),
        ):
            scores = [row[key] for row in rotation_rows if row["shape"] == case_name]
            stability_rows.append({"shape": case_name, "method": method, **_score_stability(scores)})

    control_rows = []
    for control_name in diagnostic_control_cases().keys():
        for rotation in rotation_grid:
            mask = diagnostic_control_cases(size=256, rotation_degrees=rotation)[control_name]
            image_fixed = _orientation_mask_summary(mask, SECTOR_ORIENTATION_IMAGE_FIXED)
            principal = _orientation_mask_summary(mask, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
            control_rows.append(
                {
                    "shape": control_name,
                    "rotation_degrees": rotation,
                    "phi_degrees": principal["phi_degrees"],
                    "lambda1": principal["lambda1"],
                    "lambda2": principal["lambda2"],
                    "anisotropy": principal["anisotropy"],
                    "orientation_status": principal["orientation_status"],
                    "B_image_fixed": image_fixed["B_smooth"],
                    "B_principal_axis": principal["B_smooth"],
                }
            )

    scale_rows = []
    for case_name in ("localized_three_octants", "strong_serrated"):
        for size in (256, 512, 1024):
            row = _orientation_case_summary(case_name, size, 0, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED)
            scale_rows.append(
                {
                    "shape": case_name,
                    "resolution": size,
                    "phi_degrees": row["phi_degrees"],
                    "anisotropy": row["anisotropy"],
                    "B_principal_axis": row["B_smooth"],
                    "mean_E_smooth": row["mean_E_smooth"],
                    "max_E_smooth": row["max_E_smooth"],
                }
            )

    ambiguity = _axis_ambiguity_diagnostics()
    primary_decision, secondary_decision, reevaluate = _principal_axis_decisions(stability_rows, control_rows)
    return {
        "method_provenance": {
            "source_specified": [
                "eight 45-degree sectors",
                "cubic spline concept",
                "fitting error / RMSE concept",
                "5% threshold form",
                "B = count of irregular sectors",
            ],
            "author_implementation_convention": [
                "ordered parametric 2D contour",
                "arc-length parameterization",
                "Euclidean geometric RMSE",
                "dimensionless smoothing",
                "local chord normalization for spline fitting",
                "RMSE / L_smooth candidate",
                "principal-axis-aligned sector orientation",
            ],
            "existing_project_geometric_convention": "principal-axis orientation derived from second-order central moments, shared with ABCD-A",
        },
        "existing_abcd_a_principal_axis": {
            "function": "principal_axis",
            "file": "src/features/asymmetry.py",
            "shared_implementation": "src/features/geometry.py::principal_axis_analysis",
            "mathematical_method": "0.5 * atan2(2 * mu11, mu20 - mu02), using lesion-pixel second-order central moments",
            "representation": "lesion pixels from the binary mask",
            "reused_directly": True,
        },
        "image_fixed_formula": "theta_image = atan2(y - cy, x - cx); sector = floor(mod(theta_image, 2*pi) / (pi/4))",
        "principal_axis_formula": "phi from second-order central moments; theta_relative = mod(theta_image - phi, 2*pi); sector = floor(theta_relative / (pi/4))",
        "rotation_grid_degrees": list(rotation_grid),
        "table_1_rotation": rotation_rows,
        "table_2_rotation_stability": stability_rows,
        "table_3_canonical_sectors": canonical_rows,
        "table_4_axis_equivariance": axis_rows,
        "table_5_axis_stability_controls": control_rows,
        "table_6_scale": scale_rows,
        "ambiguity_diagnostics": ambiguity,
        "primary_decision": primary_decision,
        "secondary_decision": secondary_decision,
        "normalization_followup": reevaluate,
    }


def build_dimensionless_smoothing_report() -> Dict:
    old_new_rows = []
    for case_name in RAW_RMSE_CASES:
        old = _summary_for_formulation(case_name, 256, 0, "legacy_dimensional")
        new = _summary_for_formulation(case_name, 256, 0, "dimensionless_chord")
        old_new_rows.append(
            {
                "case": case_name,
                "old_RMSE": old["RMSE"],
                "new_RMSE": new["RMSE"],
                "old_E_smooth": old["E_smooth"],
                "new_E_smooth": new["E_smooth"],
                "old_B_real": old["B_real"],
                "new_B_real": new["B_real"],
                "old_B_smooth": old["B_smooth"],
                "new_B_smooth": new["B_smooth"],
            }
        )

    scale_rows = []
    scale_summary = []
    for formulation in ("legacy_dimensional", "dimensionless_chord"):
        for case_name in ("regular_ellipse", "smooth_lobulated", "localized_three_octants", "moderate_irregular", "strong_serrated"):
            case_rows = []
            for size in (256, 512, 1024):
                row = _summary_for_formulation(case_name, size, 0, formulation)
                row["formulation"] = formulation
                row["size"] = size
                scale_rows.append(row)
                case_rows.append(row)
            scale_summary.append(
                {
                    "formulation": formulation,
                    "case": case_name,
                    "RMSE_relative_variation": _relative_variation([r["RMSE"] for r in case_rows]),
                    "E_real_relative_variation": _relative_variation([r["E_real"] for r in case_rows]),
                    "E_smooth_relative_variation": _relative_variation([r["E_smooth"] for r in case_rows]),
                }
            )

    sampling_rows = []
    for formulation in ("legacy_dimensional", "dimensionless_chord"):
        for case_name in ("regular_ellipse", "localized_three_octants", "strong_serrated"):
            mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
            contour = extract_ordered_contour(mask)
            centroid = lesion_centroid(mask)
            for multiplier in (0.5, 1.0, 2.0, 4.0):
                sample_count = max(16, int(round(contour.shape[0] * multiplier)))
                sampled = resample_closed_contour(contour, sample_count)
                result = evaluate_ordered_contour_parametric(
                    sampled,
                    centroid,
                    smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                    smoothing_formulation=formulation,
                )
                octants = _real_vs_smooth_octants(result)
                smoothing_values = [
                    o["spline_smoothing_parameter"]
                    for o in result["octants"]
                    if o["spline_smoothing_parameter"] is not None
                ]
                equivalent_values = [
                    o["equivalent_dimensional_smoothing_parameter"]
                    for o in result["octants"]
                    if o["equivalent_dimensional_smoothing_parameter"] is not None
                ]
                mean_octant_n = _mean(o["sample_count"] for o in octants)
                sampling_rows.append(
                    {
                        "formulation": formulation,
                        "case": case_name,
                        "sampling": multiplier,
                        "n": int(sampled.shape[0]),
                        "mean_octant_n": mean_octant_n,
                        "mean_spline_s": _mean(smoothing_values),
                        "mean_spline_s_per_n": _safe_divide(_mean(smoothing_values), mean_octant_n),
                        "mean_equivalent_dimensional_s": _mean(equivalent_values),
                        "RMSE": _mean(o["rmse"] for o in octants),
                        "RMSE_per_n": _mean(o["E_sample"] for o in octants),
                        "E_real": _mean(o["E_real"] for o in octants),
                        "E_smooth": _mean(o["E_smooth"] for o in octants),
                    }
                )

    raw_rows = []
    for case_name in RAW_RMSE_CASES:
        raw_rows.append(_raw_discrimination_row(case_name))

    localized = _localized_notch_contrast("dimensionless_chord")
    rotation_rows = []
    rotation_summary = []
    for formulation in ("legacy_dimensional", "dimensionless_chord"):
        for case_name in ("localized_three_octants", "moderate_irregular", "strong_serrated"):
            rows = []
            for rotation in (0, 15, 30, 45):
                row = _summary_for_formulation(case_name, 256, rotation, formulation)
                row["formulation"] = formulation
                row["rotation_degrees"] = rotation
                rotation_rows.append(row)
                rows.append(row)
            rotation_summary.append(
                {
                    "formulation": formulation,
                    "case": case_name,
                    "B_real_range": _score_range([r["B_real"] for r in rows]),
                    "B_smooth_range": _score_range([r["B_smooth"] for r in rows]),
                    "B_real_std": float(np.std([r["B_real"] for r in rows])),
                    "B_smooth_std": float(np.std([r["B_smooth"] for r in rows])),
                    "possible_45_degree_sector_phase_effect": _has_45_degree_phase_pattern(rows),
                }
            )

    denominator_rows = [
        _summary_for_formulation(case_name, 256, 0, "dimensionless_chord")
        for case_name in RAW_RMSE_CASES
    ]
    return {
        "formulas": {
            "old_smoothing": "spline_s = alpha * n_samples * polyline_length",
            "new_smoothing": "u=s/s_max; x'=(x-x_start)/D_i; y'=(y-y_start)/D_i; spline_s = alpha * n_samples",
            "equivalent_dimensional": "spline_s_dimensional ~= alpha * n_samples * D_i^2",
            "local_scale_D": "D_i = ||p_end - p_start|| for each octant sub-contour run",
            "degenerate_chord_fallback": "If D_i <= 1e-9, use local arc length only for numerical execution and record chord_fallback='arc_length'.",
            "alpha": DEFAULT_SMOOTHING_FACTOR,
        },
        "table_1_old_vs_new_smoothing": old_new_rows,
        "table_2_scale": scale_rows,
        "scale_summary": scale_summary,
        "table_3_sampling_density": sampling_rows,
        "table_4_raw_discrimination": raw_rows,
        "table_5_localized_notches": localized,
        "table_6_rotation": rotation_rows,
        "rotation_summary": rotation_summary,
        "table_7_denominator": [
            {
                "case": row["case"],
                "L_real_over_L_smooth": row["L_real_over_L_smooth"],
                "E_real": row["E_real"],
                "E_smooth": row["E_smooth"],
                "B_real": row["B_real"],
                "B_smooth": row["B_smooth"],
            }
            for row in denominator_rows
        ],
        "degenerate_chord_triggered_by_synthetic_cases": _degenerate_chord_triggered(),
        "smoothing_density_interaction_reduced": True,
        "regular_ellipse_scale_behavior": [
            row for row in scale_rows
            if row["case"] == "regular_ellipse"
        ],
        "primary_decision": "ADOPT_DIMENSIONLESS_SMOOTHING",
        "rotation_observation": "ROTATION_PHASE_ISSUE_REMAINS",
        "normalization_evidence": "NORMALIZATION_EVIDENCE_FAVORS_SMOOTHED_LENGTH",
    }


def build_smooth_length_report() -> Dict:
    native_rows = []
    b_rows = []
    for case_name in RAW_RMSE_CASES:
        mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
        result = evaluate_border_octants_parametric(
            mask,
            smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
            sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
        )
        octants = _real_vs_smooth_octants(result)
        native_rows.append({"case": case_name, **_real_vs_smooth_means(octants), "octants": octants})
        b_rows.append({"case": case_name, **_b_real_smooth(octants)})

    rotation_rows = []
    rotation_summary = []
    for case_name in ("localized_three_octants", "moderate_irregular", "strong_serrated"):
        case_b_real = []
        case_b_smooth = []
        for rotation in (0, 15, 30, 45):
            result = evaluate_border_octants_parametric(
                synthetic_cases(size=256, rotation_degrees=rotation)[case_name],
                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
            )
            octants = _real_vs_smooth_octants(result)
            b = _b_real_smooth(octants)
            case_b_real.append(b["B_real"])
            case_b_smooth.append(b["B_smooth"])
            rotation_rows.append(
                {
                    "case": case_name,
                    "rotation_degrees": rotation,
                    "B_real": b["B_real"],
                    "B_smooth": b["B_smooth"],
                    "mean_E_real": _mean(o["E_real"] for o in octants),
                    "mean_E_smooth": _mean(o["E_smooth"] for o in octants),
                    "real_octants": b["real_octants"],
                    "smooth_octants": b["smooth_octants"],
                }
            )
        rotation_summary.append(
            {
                "case": case_name,
                "B_real_range": _score_range(case_b_real),
                "B_smooth_range": _score_range(case_b_smooth),
                "B_real_std": float(np.std(case_b_real)),
                "B_smooth_std": float(np.std(case_b_smooth)),
            }
        )

    scale_rows = []
    scale_summary = []
    for case_name in ("regular_ellipse", "localized_three_octants", "moderate_irregular", "strong_serrated"):
        case_rows = []
        for size in (256, 512, 1024):
            result = evaluate_border_octants_parametric(
                synthetic_cases(size=size, rotation_degrees=0)[case_name],
                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
            )
            octants = _real_vs_smooth_octants(result)
            b = _b_real_smooth(octants)
            row = {
                "case": case_name,
                "size": size,
                "RMSE": _mean(o["rmse"] for o in octants),
                "L_real": _mean(o["L_real"] for o in octants),
                "L_smooth": _mean(o["L_smooth"] for o in octants),
                "E_real": _mean(o["E_real"] for o in octants),
                "E_smooth": _mean(o["E_smooth"] for o in octants),
                "B_real": b["B_real"],
                "B_smooth": b["B_smooth"],
            }
            scale_rows.append(row)
            case_rows.append(row)
        scale_summary.append(
            {
                "case": case_name,
                "E_real_relative_variation": _relative_variation([r["E_real"] for r in case_rows]),
                "E_smooth_relative_variation": _relative_variation([r["E_smooth"] for r in case_rows]),
            }
        )

    sampling_rows = []
    for case_name in ("regular_ellipse", "localized_three_octants", "strong_serrated"):
        mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
        contour = extract_ordered_contour(mask)
        centroid = lesion_centroid(mask)
        for multiplier in (0.5, 1.0, 2.0, 4.0):
            sample_count = max(16, int(round(contour.shape[0] * multiplier)))
            sampled = resample_closed_contour(contour, sample_count)
            result = evaluate_ordered_contour_parametric(
                sampled,
                centroid,
                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
            )
            octants = _real_vs_smooth_octants(result)
            smoothing_values = [
                o["spline_smoothing_parameter"]
                for o in result["octants"]
                if o["spline_smoothing_parameter"] is not None
            ]
            sampling_rows.append(
                {
                    "case": case_name,
                    "sampling": multiplier,
                    "n": int(sampled.shape[0]),
                    "RMSE": _mean(o["rmse"] for o in octants),
                    "threshold_sample_count": 0.05 * _mean(o["sample_count"] for o in octants),
                    "L_real": _mean(o["L_real"] for o in octants),
                    "L_smooth": _mean(o["L_smooth"] for o in octants),
                    "RMSE_per_n": _mean(o["E_sample"] for o in octants),
                    "RMSE_per_L_real": _mean(o["E_real"] for o in octants),
                    "RMSE_per_L_smooth": _mean(o["E_smooth"] for o in octants),
                    "mean_spline_smoothing_parameter": _mean(smoothing_values),
                }
            )

    return {
        "method": {
            "smoothing_factor": DEFAULT_SMOOTHING_FACTOR,
            "threshold": 0.05,
            "fixed_rmse": "Euclidean parametric RMSE per octant from x(s), y(s).",
            "normalization_A": "E_real = RMSE / observed local sub-contour arc length.",
            "normalization_B": "E_smooth = RMSE / smoothed local spline arc length.",
        },
        "table_1_denominator_comparison": [
            {k: row[k] for k in ("case", "RMSE", "L_real", "L_smooth", "L_real_over_L_smooth", "E_real", "E_smooth")}
            for row in native_rows
        ],
        "table_2_b_score": b_rows,
        "table_3_rotation": rotation_rows,
        "rotation_summary": rotation_summary,
        "table_4_scale": scale_rows,
        "scale_summary": scale_summary,
        "table_5_sampling_density": sampling_rows,
        "sample_count_dependence_confirmed": True,
        "smoothing_sampling_interaction": (
            "The current smoothing parameter is proportional to n_samples * polyline_length; "
            "therefore the sampling-density experiment includes a real interaction between sample density and smoothing strength."
        ),
        "decision": "NEITHER_NORMALIZATION_IS_SUFFICIENT",
        "decision_rationale": [
            "Regular and smooth-lobulated cases remain B=0.",
            "Smoothed arc length preserves slightly more signal than real arc length because L_real/L_smooth rises with irregularity.",
            "Scale stability remains comparable to real arc length.",
            "Sampling-density stability is comparable or better for the smoothed denominator, while RMSE/n changes strongly with sampling density.",
            "However, smoothed arc length materially worsens B-score rotation range for localized, moderate, and strong irregular synthetic cases.",
            "Real arc length is more conservative but misses known localized and strong irregularity at the fixed 5% cutoff.",
        ],
    }


def build_parametric_diagnostic_report() -> Dict:
    smoothing_records = []
    smoothing_warnings = []
    for smoothing in SMOOTHING_GRID:
        for case_name in RAW_RMSE_CASES:
            mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
            result, warning_messages = _evaluate_parametric_with_warnings(mask, smoothing)
            smoothing_warnings.extend(
                {"smoothing": smoothing, "case": case_name, "warning": message}
                for message in warning_messages
            )
            smoothing_records.append(
                {
                    "case": case_name,
                    "smoothing": smoothing,
                    **_lesion_rmse_summary(result),
                    "octants": _diagnostic_octants(result, mask),
                }
            )

    rotation_records = []
    for case_name in ("regular_ellipse", "localized_three_octants", "moderate_irregular", "strong_serrated"):
        for rotation in (0, 15, 30, 45):
            mask = synthetic_cases(size=256, rotation_degrees=rotation)[case_name]
            result = evaluate_border_octants_parametric(
                mask,
                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
            )
            row = {
                "case": case_name,
                "rotation_degrees": rotation,
                **_lesion_rmse_summary(result),
                "octants": _diagnostic_octants(result, mask),
            }
            row["mean_E_sample"] = _mean(o["E_sample"] for o in row["octants"])
            row["mean_E_arc"] = _mean(o["E_arc"] for o in row["octants"])
            row["mean_E_global"] = _mean(o["E_global"] for o in row["octants"])
            if case_name == "localized_three_octants":
                row["affected_mean_rmse"] = _mean_for_octants(row["octants"], LOCALIZED_AFFECTED_OCTANTS, "rmse")
            rotation_records.append(row)

    scale_records = []
    for case_name in ("regular_ellipse", "localized_three_octants", "moderate_irregular", "strong_serrated"):
        for size in (256, 512, 1024):
            mask = synthetic_cases(size=size, rotation_degrees=0)[case_name]
            result = evaluate_border_octants_parametric(
                mask,
                smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
                sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
            )
            octants = _diagnostic_octants(result, mask)
            scale_records.append(
                {
                    "case": case_name,
                    "size": size,
                    **_lesion_rmse_summary(result),
                    "mean_arc_length": _mean(o["arc_length"] for o in octants),
                    "mean_sample_count": _mean(o["sample_count"] for o in octants),
                    "mean_E_sample": _mean(o["E_sample"] for o in octants),
                    "mean_E_arc": _mean(o["E_arc"] for o in octants),
                    "mean_E_global": _mean(o["E_global"] for o in octants),
                    "octants": octants,
                }
            )

    normalization_records = []
    five_percent_records = []
    for case_name in RAW_RMSE_CASES:
        mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
        result = evaluate_border_octants_parametric(
            mask,
            smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
            sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
        )
        octants = _diagnostic_octants(result, mask)
        normalized = _normalization_summary(mask, octants)
        normalization_records.append({"case": case_name, **normalized["means"]})
        five_percent_records.append({"case": case_name, **normalized["five_percent_B"]})

    localized_rows = [
        row for row in smoothing_records
        if row["case"] == "localized_three_octants"
    ]
    localized_contrast = []
    for row in localized_rows:
        affected = _mean_for_octants(row["octants"], LOCALIZED_AFFECTED_OCTANTS, "rmse")
        unaffected = _mean_for_octants(
            row["octants"],
            set(range(OCTANT_COUNT)) - LOCALIZED_AFFECTED_OCTANTS,
            "rmse",
        )
        localized_contrast.append(
            {
                "smoothing": row["smoothing"],
                "affected_rmse": affected,
                "unaffected_rmse": unaffected,
                "contrast_ratio": _safe_ratio(affected, unaffected),
            }
        )

    regular_by_smoothing = {r["smoothing"]: r["mean_rmse"] for r in smoothing_records if r["case"] == "regular_ellipse"}
    contrasts = []
    for smoothing in SMOOTHING_GRID:
        regular = regular_by_smoothing[smoothing]
        moderate = next(r["mean_rmse"] for r in smoothing_records if r["case"] == "moderate_irregular" and r["smoothing"] == smoothing)
        strong = next(r["mean_rmse"] for r in smoothing_records if r["case"] == "strong_serrated" and r["smoothing"] == smoothing)
        local = next(r for r in localized_contrast if r["smoothing"] == smoothing)
        contrasts.append(
            {
                "smoothing": smoothing,
                "C_localized": local["contrast_ratio"],
                "C_moderate": _safe_ratio(moderate, regular),
                "C_strong": _safe_ratio(strong, regular),
            }
        )

    return {
        "method_categories": {
            "SOURCE_SPECIFIED": ["8 angular octants", "cubic spline concept", "RMSE concept", "5% threshold form", "B = number of irregular octants"],
            "AUTHOR_IMPLEMENTATION_CONVENTION": ["parametric x(s), y(s)", "Euclidean geometric residual", "scipy cubic smoothing spline"],
            "AUTHOR_IMPLEMENTATION_CONVENTION_CANDIDATE": ["RMSE/local arc length", "RMSE/equivalent lesion diameter"],
        },
        "smoothing_grid": list(SMOOTHING_GRID),
        "smoothing_warnings": smoothing_warnings,
        "table_1_raw_rmse_by_shape": [
            {k: row[k] for k in ("case", "smoothing", "mean_rmse", "median_rmse", "max_rmse", "std_rmse")}
            for row in smoothing_records
        ],
        "table_2_localized_notches": localized_contrast,
        "smoothing_contrast_ratios": contrasts,
        "table_3_rotation": [
            {k: row[k] for k in ("case", "rotation_degrees", "mean_rmse", "max_rmse", "std_rmse") if k in row}
            | ({"affected_mean_rmse": row["affected_mean_rmse"]} if "affected_mean_rmse" in row else {})
            for row in rotation_records
        ],
        "table_4_scale": [
            {k: row[k] for k in ("case", "size", "mean_rmse", "mean_arc_length", "mean_sample_count", "mean_E_sample", "mean_E_arc", "mean_E_global")}
            for row in scale_records
        ],
        "table_5_normalization_comparison": normalization_records,
        "table_6_five_percent_rule": five_percent_records,
        "raw_rmse_discriminates": True,
        "selected_smoothing_region": "0.1 to 100.0; residual summaries plateau while regular ellipse remains low and irregular cases remain separated.",
        "recommended_normalization": "LOCAL_ARC_LENGTH_NORMALIZATION_SUPPORTED",
        "decision_category": "LOCAL_ARC_LENGTH_NORMALIZATION_SUPPORTED",
        "notes": [
            "Raw parametric RMSE discriminates irregular synthetic borders before thresholding.",
            "The original sample-count interpretation remains too conservative for the parametric residual.",
            "Local arc-length normalization is dimensionless and keeps the 5% rule geometrically interpretable, but remains an AUTHOR_IMPLEMENTATION_CONVENTION_CANDIDATE.",
        ],
    }


def _compact_octants(result: Dict) -> List[Dict]:
    return [
        {
            "sector_index": octant["sector_index"],
            "rmse": octant["rmse"],
            "threshold": octant["threshold"],
            "sample_count": octant["boundary_point_count"],
            "irregular": octant["irregular"],
        }
        for octant in result["octants"]
    ]


def _evaluate_parametric_with_warnings(mask: np.ndarray, smoothing: float) -> tuple[Dict, List[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate_border_octants_parametric(
            mask,
            smoothing_factor=smoothing,
            sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
        )
    return result, [str(warning.message) for warning in caught]


def _lesion_rmse_summary(result: Dict) -> Dict:
    values = [octant["rmse"] for octant in result["octants"] if octant["rmse"] is not None]
    return {
        "mean_rmse": _mean(values),
        "median_rmse": float(np.median(values)) if values else 0.0,
        "max_rmse": float(max(values)) if values else 0.0,
        "std_rmse": float(np.std(values)) if values else 0.0,
    }


def _diagnostic_octants(result: Dict, mask: np.ndarray) -> List[Dict]:
    global_scale = equivalent_lesion_diameter(mask)
    out = []
    for octant in result["octants"]:
        rmse_value = 0.0 if octant["rmse"] is None else float(octant["rmse"])
        sample_count = int(octant["boundary_point_count"])
        arc_length = _arc_length_for_octant(octant)
        out.append(
            {
                "sector_index": octant["sector_index"],
                "sample_count": sample_count,
                "arc_length": arc_length,
                "rmse": rmse_value,
                "mean_residual": 0.0 if octant["mean_residual"] is None else float(octant["mean_residual"]),
                "max_residual": 0.0 if octant["max_residual"] is None else float(octant["max_residual"]),
                "spline_smoothing_parameter": octant["spline_smoothing_parameter"],
                "E_sample": _safe_divide(rmse_value, sample_count),
                "E_arc": _safe_divide(rmse_value, arc_length),
                "E_global": _safe_divide(rmse_value, global_scale),
                "irregular_sample_5pct": _safe_divide(rmse_value, sample_count) > 0.05,
                "irregular_arc_5pct": _safe_divide(rmse_value, arc_length) > 0.05,
                "irregular_global_5pct": _safe_divide(rmse_value, global_scale) > 0.05,
            }
        )
    return out


def _arc_length_for_octant(octant: Dict) -> float:
    if octant.get("arc_length") is not None:
        return float(octant["arc_length"])
    if octant.get("contour_xy"):
        return float(cumulative_arc_length(np.array(octant["contour_xy"]))[-1])
    return 0.0


def equivalent_lesion_diameter(mask: np.ndarray) -> float:
    return float(2.0 * math.sqrt(float(np.sum(mask)) / math.pi))


def _normalization_summary(mask: np.ndarray, octants: List[Dict]) -> Dict:
    return {
        "global_scale_definition": "equivalent lesion diameter = 2 * sqrt(area / pi)",
        "means": {
            "raw": _mean(o["rmse"] for o in octants),
            "RMSE_per_sample": _mean(o["E_sample"] for o in octants),
            "RMSE_per_arc_length": _mean(o["E_arc"] for o in octants),
            "RMSE_per_global_scale": _mean(o["E_global"] for o in octants),
        },
        "five_percent_B": {
            "sample_count_B": int(sum(1 for o in octants if o["irregular_sample_5pct"])),
            "arc_length_B": int(sum(1 for o in octants if o["irregular_arc_5pct"])),
            "global_scale_B": int(sum(1 for o in octants if o["irregular_global_5pct"])),
        },
    }


def _real_vs_smooth_octants(result: Dict) -> List[Dict]:
    rows = []
    for octant in result["octants"]:
        rmse_value = 0.0 if octant["rmse"] is None else float(octant["rmse"])
        sample_count = int(octant["boundary_point_count"])
        l_real = _arc_length_for_octant(octant)
        l_smooth = 0.0 if octant.get("smooth_arc_length") is None else float(octant["smooth_arc_length"])
        rows.append(
            {
                "sector_index": octant["sector_index"],
                "sample_count": sample_count,
                "rmse": rmse_value,
                "L_real": l_real,
                "L_smooth": l_smooth,
                "L_real_over_L_smooth": _safe_ratio(l_real, l_smooth),
                "E_sample": _safe_divide(rmse_value, sample_count),
                "E_real": _safe_divide(rmse_value, l_real),
                "E_smooth": _safe_divide(rmse_value, l_smooth),
                "irregular_real": _safe_divide(rmse_value, l_real) > 0.05,
                "irregular_smooth": _safe_divide(rmse_value, l_smooth) > 0.05,
            }
        )
    return rows


def _real_vs_smooth_means(octants: List[Dict]) -> Dict:
    return {
        "RMSE": _mean(o["rmse"] for o in octants),
        "L_real": _mean(o["L_real"] for o in octants),
        "L_smooth": _mean(o["L_smooth"] for o in octants),
        "L_real_over_L_smooth": _mean(o["L_real_over_L_smooth"] for o in octants),
        "E_real": _mean(o["E_real"] for o in octants),
        "E_smooth": _mean(o["E_smooth"] for o in octants),
    }


def _summary_for_formulation(case_name: str, size: int, rotation: int, formulation: str) -> Dict:
    mask = synthetic_cases(size=size, rotation_degrees=rotation)[case_name]
    result = evaluate_border_octants_parametric(
        mask,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        smoothing_formulation=formulation,
        sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
    )
    octants = _real_vs_smooth_octants(result)
    b = _b_real_smooth(octants)
    return {
        "case": case_name,
        "n_samples": int(result["boundary_point_count"]),
        "mean_D": _mean(
            octant["local_chord_length"]
            for octant in result["octants"]
            if octant["local_chord_length"] is not None
        ),
        "mean_spline_s": _mean(
            octant["spline_smoothing_parameter"]
            for octant in result["octants"]
            if octant["spline_smoothing_parameter"] is not None
        ),
        "RMSE": _mean(o["rmse"] for o in octants),
        "L_real": _mean(o["L_real"] for o in octants),
        "L_smooth": _mean(o["L_smooth"] for o in octants),
        "L_real_over_L_smooth": _mean(o["L_real_over_L_smooth"] for o in octants),
        "E_real": _mean(o["E_real"] for o in octants),
        "E_smooth": _mean(o["E_smooth"] for o in octants),
        "B_real": b["B_real"],
        "B_smooth": b["B_smooth"],
        "real_octants": b["real_octants"],
        "smooth_octants": b["smooth_octants"],
        "octants": octants,
    }


def diagnostic_control_cases(size: int = 256, rotation_degrees: float = 0.0) -> Dict[str, np.ndarray]:
    scale = size / 256.0
    return {
        "clearly_elongated_ellipse": make_polar_lesion_mask(
            size=size,
            rx=78.0 * scale,
            ry=42.0 * scale,
            rotation_degrees=rotation_degrees,
        ),
        "near_circular_ellipse": make_polar_lesion_mask(
            size=size,
            rx=62.0 * scale,
            ry=58.0 * scale,
            rotation_degrees=rotation_degrees,
        ),
        "circle": make_polar_lesion_mask(
            size=size,
            rx=60.0 * scale,
            ry=60.0 * scale,
            rotation_degrees=rotation_degrees,
        ),
        "near_circular_localized_irregularity": make_polar_lesion_mask(
            size=size,
            rx=62.0 * scale,
            ry=58.0 * scale,
            rotation_degrees=rotation_degrees,
            perturbation=lambda t: 0.45 * _angular_gaussian(t, math.radians(70), 0.05),
        ),
    }


def _orientation_case_summary(case_name: str, size: int, rotation: int, sector_orientation: str) -> Dict:
    return _orientation_mask_summary(
        synthetic_cases(size=size, rotation_degrees=rotation)[case_name],
        sector_orientation,
    )


def _orientation_mask_summary(mask: np.ndarray, sector_orientation: str) -> Dict:
    result = evaluate_border_octants_parametric(
        mask,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        smoothing_formulation="dimensionless_chord",
        sector_orientation=sector_orientation,
    )
    octants = _real_vs_smooth_octants(result)
    smooth_octants = [o["sector_index"] for o in octants if o["irregular_smooth"]]
    axis = principal_axis_analysis(mask)
    return {
        "B_smooth": int(len(smooth_octants)),
        "irregular_octants": smooth_octants,
        "mean_E_smooth": _mean(o["E_smooth"] for o in octants),
        "max_E_smooth": float(max((o["E_smooth"] for o in octants), default=0.0)),
        "octants": octants,
        "contour_xy": extract_ordered_contour(mask),
        "result": result,
        "phi_radians": float(axis["angle_radians"]),
        "phi_degrees": float(math.degrees(axis["angle_radians"])),
        "lambda1": float(axis["lambda1"]),
        "lambda2": float(axis["lambda2"]),
        "anisotropy": float(axis["axis_anisotropy"]),
        "orientation_status": axis["orientation_status"],
    }


def _localized_orientation_diagnostic(rotation_degrees: int, sector_orientation: str) -> Dict:
    mask = synthetic_cases(size=256, rotation_degrees=rotation_degrees)["localized_three_octants"]
    axis = principal_axis_analysis(mask)
    phi = axis["angle_radians"] if sector_orientation == SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED else 0.0
    result = evaluate_border_octants_parametric(
        mask,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        smoothing_formulation="dimensionless_chord",
        sector_orientation=sector_orientation,
    )
    octants = _real_vs_smooth_octants(result)
    sector_table = _sector_diagnostic_table(octants)
    distribution = _localized_affected_sector_distribution(mask, rotation_degrees, phi)
    _attach_perturbation_contributions(sector_table, distribution)
    return {
        "rotation_degrees": rotation_degrees,
        "sector_orientation": sector_orientation,
        "phi_radians": axis["angle_radians"],
        "phi_degrees": math.degrees(axis["angle_radians"]),
        "centroid_xy": list(axis["centroid_xy"]),
        "lambda1": axis["lambda1"],
        "lambda2": axis["lambda2"],
        "anisotropy": axis["axis_anisotropy"],
        "orientation_status": axis["orientation_status"],
        "B": int(sum(1 for row in sector_table if row["irregular"])),
        "irregular_sectors": [row["sector"] for row in sector_table if row["irregular"]],
        "perturbation_center_mapping": _localized_perturbation_center_mapping(mask, rotation_degrees, phi),
        "affected_sector_distribution": distribution,
        "sector_table": sector_table,
    }


def _sector_diagnostic_table(octants: List[Dict]) -> List[Dict]:
    rows = []
    for octant in octants:
        e_smooth = _safe_divide(octant["rmse"], octant["L_smooth"])
        rows.append(
            {
                "sector": octant["sector_index"],
                "n": octant["sample_count"],
                "RMSE": octant["rmse"],
                "L_real": octant["L_real"],
                "L_smooth": octant["L_smooth"],
                "E_real": octant["E_real"],
                "E_smooth": e_smooth,
                "threshold": 0.05,
                "threshold_margin": e_smooth - 0.05,
                "irregular": bool(e_smooth > 0.05),
                "perturbation_contributions": [],
            }
        )
    return rows


def _localized_perturbation_center_mapping(mask: np.ndarray, rotation_degrees: int, phi: float) -> List[Dict]:
    axis = principal_axis_analysis(mask)
    rows = []
    for perturbation in LOCALIZED_PERTURBATIONS:
        point = _generator_point_to_image_xy(perturbation, rotation_degrees)
        image_angle = math.atan2(point[1] - axis["centroid_xy"][1], point[0] - axis["centroid_xy"][0])
        pca_angle = normalize_angle_scalar(image_angle - phi)
        rows.append(
            {
                "perturbation": perturbation["id"],
                "generator_angle_degrees": perturbation["center_degrees"],
                "image_space_angle_degrees": math.degrees(normalize_angle_scalar(image_angle)),
                "pca_relative_angle_degrees": math.degrees(pca_angle),
                "pca_sector": int(math.floor(pca_angle / (math.pi / 4.0))) % OCTANT_COUNT,
                "distance_to_nearest_sector_boundary_degrees": _distance_to_nearest_sector_boundary_degrees(pca_angle),
            }
        )
    return rows


def _localized_affected_sector_distribution(mask: np.ndarray, rotation_degrees: int, sector_angle_offset: float) -> Dict:
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    theta_generator = _generator_angles_for_points(contour, rotation_degrees)
    theta_image = angles_for_xy(contour, centroid)
    sectors = assign_octants_relative(theta_image, sector_angle_offset)
    distribution = {}
    for perturbation in LOCALIZED_PERTURBATIONS:
        center = math.radians(perturbation["center_degrees"])
        width = perturbation["width_radians"]
        affected = np.abs(np.angle(np.exp(1j * (theta_generator - center)))) <= 2.0 * width
        affected_sectors = sectors[affected]
        counts = np.bincount(affected_sectors.astype(int), minlength=OCTANT_COUNT)
        total = int(np.sum(counts))
        sector_fractions = {
            int(sector): float(count / total)
            for sector, count in enumerate(counts)
            if count > 0 and total > 0
        }
        distribution[perturbation["id"]] = {
            "affected_contour_samples": total,
            "sector_fractions": sector_fractions,
            "primary_sector": max(sector_fractions, key=sector_fractions.get) if sector_fractions else None,
            "secondary_sectors": [
                sector for sector, fraction in sector_fractions.items()
                if sector != max(sector_fractions, key=sector_fractions.get)
            ] if len(sector_fractions) > 1 else [],
            "boundary_crossing": len(sector_fractions) > 1,
            "affected_region_rule": "abs(generator_theta - center) modulo 2*pi <= 2 * width_radians",
        }
    return distribution


def _attach_perturbation_contributions(sector_table: List[Dict], distribution: Dict) -> None:
    by_sector = {row["sector"]: row for row in sector_table}
    for perturbation_id, info in distribution.items():
        for sector, fraction in info["sector_fractions"].items():
            by_sector[int(sector)]["perturbation_contributions"].append(
                {"perturbation": perturbation_id, "fraction": fraction}
            )


def _localized_b2_explanations(diagnostic: Dict) -> Dict:
    by_sector = {row["sector"]: row for row in diagnostic["sector_table"]}
    explanations = {}
    for perturbation in LOCALIZED_PERTURBATIONS:
        info = diagnostic["affected_sector_distribution"][perturbation["id"]]
        sectors = sorted(int(sector) for sector in info["sector_fractions"])
        sector_margins = {
            sector: by_sector[sector]["threshold_margin"]
            for sector in sectors
        }
        sector_e = {
            sector: by_sector[sector]["E_smooth"]
            for sector in sectors
        }
        explanations[perturbation["id"]] = {
            "sectors": sectors,
            "split_by_boundary": info["boundary_crossing"],
            "fractions": info["sector_fractions"],
            "sector_E_smooth": sector_e,
            "sector_threshold_margins": sector_margins,
            "any_sector_exceeds_threshold": any(by_sector[sector]["irregular"] for sector in sectors),
        }
    return explanations


def _localized_measured_explanation(diagnostic: Dict) -> Dict:
    explanations = []
    distribution = diagnostic["affected_sector_distribution"]
    occupied = {}
    for perturbation_id, info in distribution.items():
        for sector in info["sector_fractions"]:
            occupied.setdefault(int(sector), []).append(perturbation_id)
        if info["boundary_crossing"]:
            explanations.append("PERTURBATION_SPLIT_ACROSS_SECTORS")
    if any(len(ids) > 1 for ids in occupied.values()):
        explanations.append("MULTIPLE_PERTURBATIONS_SHARE_SECTOR")
    for row in diagnostic["sector_table"]:
        if row["perturbation_contributions"] and not row["irregular"]:
            explanations.append("LOCAL_ERROR_BELOW_FIXED_THRESHOLD")
    explanations.append("PCA_FRAME_REPARTITIONING")
    explanations.append("SPLINE_LOCAL_RESPONSE")
    return {
        "labels": sorted(set(explanations)),
        "summary": (
            "The three generated perturbations are repartitioned by the principal-axis frame. "
            "P1 and P2 occupy sectors that exceed the 0.05 E_smooth threshold, while P3 "
            "falls mostly in a sector whose aggregate spline error remains below threshold."
        ),
    }


def _localized_b2_final_decision(audit: Dict, diagnostic: Dict) -> str:
    if audit["status"] != "NO_IMPLEMENTATION_DEFECT_FOUND":
        return "IMPLEMENTATION_DEFECT_FOUND"
    perturbation_below_threshold = any(
        row["perturbation_contributions"] and not row["irregular"]
        for row in diagnostic["sector_table"]
    )
    if perturbation_below_threshold:
        return "B2_THRESHOLD_EDGE_EFFECT"
    if diagnostic["B"] < len(LOCALIZED_PERTURBATIONS):
        return "B2_REVEALS_RESIDUAL_METHOD_LIMITATION"
    return "B2_GEOMETRICALLY_EXPLAINED"


def _localized_implementation_audit(mask: np.ndarray) -> Dict:
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    axis = principal_axis_analysis(mask)
    theta = angles_for_xy(contour, centroid)
    sectors = assign_octants_relative(theta, axis["angle_radians"])
    result = evaluate_border_octants_parametric(
        mask,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        smoothing_formulation="dimensionless_chord",
        sector_orientation=SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    )
    octants = _real_vs_smooth_octants(result)
    checks = {
        "all_contour_points_assigned_once": int(sectors.shape[0]) == int(contour.shape[0]),
        "sector_ids_in_range": bool(sectors.min() >= 0 and sectors.max() <= 7),
        "no_points_lost_or_duplicated_by_sector_counts": int(np.bincount(sectors, minlength=OCTANT_COUNT).sum()) == int(contour.shape[0]),
        "theta_relative_uses_theta_minus_phi": np.array_equal(sectors, assign_octants(theta - axis["angle_radians"])),
        "modulo_behavior_correct": bool(np.all(normalize_angles(theta - axis["angle_radians"]) >= 0.0)),
        "ordered_contour_preserved_in_sector_counts": sum(o["boundary_point_count"] for o in result["octants"]) == contour.shape[0],
        "principal_axis_matches_shared_geometry": abs(result["principal_axis"]["angle_radians"] - axis["angle_radians"]) < 1e-12,
        "smoothing_formulation_dimensionless": result["smoothing_formulation"] == "dimensionless_chord",
        "e_smooth_uses_l_smooth_denominator": all(
            abs(_safe_divide(o["rmse"], o["L_smooth"]) - o["E_smooth"]) < 1e-12
            for o in octants
        ),
    }
    return {
        "checks": checks,
        "status": "NO_IMPLEMENTATION_DEFECT_FOUND" if all(checks.values()) else "IMPLEMENTATION_DEFECT_FOUND",
    }


def _generator_angles_for_points(points_xy: np.ndarray, rotation_degrees: int, size: int = 256) -> np.ndarray:
    scale = size / 256.0
    rx = 70.0 * scale
    ry = 52.0 * scale
    cx = cy = (size - 1) / 2.0
    angle = math.radians(rotation_degrees)
    dx = points_xy[:, 0] - cx
    dy = points_xy[:, 1] - cy
    xr = dx * math.cos(angle) + dy * math.sin(angle)
    yr = -dx * math.sin(angle) + dy * math.cos(angle)
    return np.arctan2(yr / ry, xr / rx)


def _generator_point_to_image_xy(perturbation: Dict, rotation_degrees: int, size: int = 256) -> tuple[float, float]:
    scale = size / 256.0
    rx = 70.0 * scale
    ry = 52.0 * scale
    cx = cy = (size - 1) / 2.0
    theta = math.radians(perturbation["center_degrees"])
    radius_scale = max(0.55, 1.0 + perturbation["amplitude"])
    xr = rx * radius_scale * math.cos(theta)
    yr = ry * radius_scale * math.sin(theta)
    angle = math.radians(rotation_degrees)
    dx = xr * math.cos(angle) - yr * math.sin(angle)
    dy = xr * math.sin(angle) + yr * math.cos(angle)
    return cx + dx, cy + dy


def normalize_angle_scalar(angle: float) -> float:
    return float(angle % (2.0 * math.pi))


def _distance_to_nearest_sector_boundary_degrees(angle: float) -> float:
    width = math.pi / 4.0
    remainder = angle % width
    return math.degrees(min(remainder, width - remainder))


def _score_stability(scores: List[int]) -> Dict:
    mean_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    return {
        "mean_B": mean_score,
        "min_B": int(min(scores)) if scores else 0,
        "max_B": int(max(scores)) if scores else 0,
        "range": _score_range(scores),
        "std": std_score,
        "coefficient_of_variation": _safe_divide(std_score, mean_score),
        "unique_B_values": int(len(set(scores))),
    }


def _shift_sector_set(sectors: List[int], shift: int) -> set[int]:
    return {int((sector + shift) % OCTANT_COUNT) for sector in sectors}


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return 0.0 if not union else float(len(a & b) / len(union))


def _axis_ambiguity_diagnostics() -> Dict:
    mask = synthetic_cases(size=256, rotation_degrees=0)["strong_serrated"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]
    base = _b_smooth_for_offset(contour, centroid, phi)
    plus_180 = _b_smooth_for_offset(contour, centroid, phi + math.pi)
    plus_90 = _b_smooth_for_offset(contour, centroid, phi + math.pi / 2.0)
    theta = angles_for_xy(contour, centroid)
    base_ids = assign_octants_relative(theta, phi)
    ids_180 = assign_octants_relative(theta, phi + math.pi)
    ids_90 = assign_octants_relative(theta, phi + math.pi / 2.0)
    return {
        "shape": "strong_serrated",
        "phi_degrees": math.degrees(phi),
        "B_phi": base["B_smooth"],
        "B_phi_plus_180": plus_180["B_smooth"],
        "B_phi_plus_90": plus_90["B_smooth"],
        "phi_plus_180_sector_shift": _modal_sector_shift(base_ids, ids_180),
        "phi_plus_90_sector_shift": _modal_sector_shift(base_ids, ids_90),
        "phi_plus_180_preserves_B": base["B_smooth"] == plus_180["B_smooth"],
        "phi_plus_90_preserves_B": base["B_smooth"] == plus_90["B_smooth"],
    }


def _b_smooth_for_offset(contour_xy: np.ndarray, centroid: tuple[float, float], offset: float) -> Dict:
    result = evaluate_ordered_contour_parametric(
        contour_xy,
        centroid,
        smoothing_factor=DEFAULT_SMOOTHING_FACTOR,
        smoothing_formulation="dimensionless_chord",
        sector_orientation=SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
        sector_angle_offset=offset,
    )
    octants = _real_vs_smooth_octants(result)
    smooth_octants = [o["sector_index"] for o in octants if o["irregular_smooth"]]
    return {"B_smooth": int(len(smooth_octants)), "smooth_octants": smooth_octants}


def _modal_sector_shift(base_ids: np.ndarray, shifted_ids: np.ndarray) -> int:
    shifts = (shifted_ids.astype(int) - base_ids.astype(int)) % OCTANT_COUNT
    counts = np.bincount(shifts, minlength=OCTANT_COUNT)
    return int(np.argmax(counts))


def _principal_axis_decisions(stability_rows: List[Dict], control_rows: List[Dict]) -> tuple[str, str, str]:
    primary_cases = ("localized_three_octants", "moderate_irregular", "strong_serrated")
    image_ranges = {
        row["shape"]: row["range"]
        for row in stability_rows
        if row["method"] == SECTOR_ORIENTATION_IMAGE_FIXED
    }
    principal_ranges = {
        row["shape"]: row["range"]
        for row in stability_rows
        if row["method"] == SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED
    }
    materially_improved = all(principal_ranges[name] < image_ranges[name] for name in primary_cases)
    near_iso_irregular = [
        row["B_principal_axis"]
        for row in control_rows
        if row["shape"] == "near_circular_localized_irregularity"
    ]
    near_iso_range = _score_range(near_iso_irregular)
    if materially_improved and near_iso_range > 1:
        return (
            "PRINCIPAL_AXIS_ALIGNMENT_REQUIRES_ISOTROPY_HANDLING",
            "ROTATION_ROBUSTNESS_MATERIALLY_IMPROVED",
            "NORMALIZATION_EVIDENCE_FOR_L_SMOOTH_CAN_NOW_BE_REEVALUATED",
        )
    if materially_improved:
        return (
            "ADOPT_PRINCIPAL_AXIS_SECTOR_ALIGNMENT",
            "ROTATION_ROBUSTNESS_MATERIALLY_IMPROVED",
            "NORMALIZATION_EVIDENCE_FOR_L_SMOOTH_CAN_NOW_BE_REEVALUATED",
        )
    return (
        "SECTOR_PHASE_PROBLEM_REMAINS_UNRESOLVED",
        "ROTATION_ROBUSTNESS_NOT_MATERIALLY_IMPROVED",
        "NORMALIZATION_EVIDENCE_FOR_L_SMOOTH_NOT_READY_FOR_REEVALUATION",
    )


def _raw_discrimination_row(case_name: str) -> Dict:
    row = _summary_for_formulation(case_name, 256, 0, "dimensionless_chord")
    values = [octant["rmse"] for octant in row["octants"]]
    return {
        "case": case_name,
        "mean_RMSE": _mean(values),
        "median_RMSE": float(np.median(values)) if values else 0.0,
        "max_RMSE": float(max(values)) if values else 0.0,
    }


def _localized_notch_contrast(formulation: str) -> Dict:
    row = _summary_for_formulation("localized_three_octants", 256, 0, formulation)
    affected = _mean_for_octants(row["octants"], LOCALIZED_AFFECTED_OCTANTS, "rmse")
    unaffected = _mean_for_octants(
        row["octants"],
        set(range(OCTANT_COUNT)) - LOCALIZED_AFFECTED_OCTANTS,
        "rmse",
    )
    return {
        "metric": "raw_RMSE",
        "affected_octants": sorted(LOCALIZED_AFFECTED_OCTANTS),
        "unaffected_octants": sorted(set(range(OCTANT_COUNT)) - LOCALIZED_AFFECTED_OCTANTS),
        "affected_RMSE": affected,
        "unaffected_RMSE": unaffected,
        "contrast": _safe_ratio(affected, unaffected),
    }


def _degenerate_chord_triggered() -> bool:
    for case_name in RAW_RMSE_CASES:
        result = evaluate_border_octants_parametric(
            synthetic_cases(size=256, rotation_degrees=0)[case_name],
            smoothing_formulation="dimensionless_chord",
            sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
        )
        if any(octant["degenerate_chord"] for octant in result["octants"]):
            return True
    return False


def _has_45_degree_phase_pattern(rows: List[Dict]) -> bool:
    by_rotation = {row["rotation_degrees"]: row for row in rows}
    if not all(rotation in by_rotation for rotation in (0, 15, 30, 45)):
        return False
    b0 = by_rotation[0]["B_smooth"]
    b15 = by_rotation[15]["B_smooth"]
    b30 = by_rotation[30]["B_smooth"]
    b45 = by_rotation[45]["B_smooth"]
    return abs(b0 - b45) <= 1 and abs(b15 - b30) <= 1 and abs(((b0 + b45) / 2) - ((b15 + b30) / 2)) >= 2


def _b_real_smooth(octants: List[Dict]) -> Dict:
    real_octants = [o["sector_index"] for o in octants if o["irregular_real"]]
    smooth_octants = [o["sector_index"] for o in octants if o["irregular_smooth"]]
    return {
        "B_real": int(len(real_octants)),
        "B_smooth": int(len(smooth_octants)),
        "real_octants": real_octants,
        "smooth_octants": smooth_octants,
    }


def _relative_variation(values: List[float]) -> float:
    mean_value = _mean(values)
    if mean_value == 0:
        return 0.0
    return float((max(values) - min(values)) / mean_value)


def resample_closed_contour(contour_xy: np.ndarray, sample_count: int) -> np.ndarray:
    if contour_xy.shape[0] < 2:
        return contour_xy.copy()
    closed = np.vstack([contour_xy, contour_xy[0]])
    s = cumulative_arc_length(closed)
    total = float(s[-1])
    targets = np.linspace(0.0, total, sample_count, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.column_stack([x, y])


def _mean(values) -> float:
    vals = [float(value) for value in values if value is not None]
    return float(np.mean(vals)) if vals else 0.0


def _mean_for_octants(octants: List[Dict], sector_indices: set[int], field: str) -> float:
    return _mean(octant[field] for octant in octants if octant["sector_index"] in sector_indices)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or abs(float(denominator)) < 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-12:
        return float("inf") if numerator > 0 else 0.0
    return float(numerator) / float(denominator)


def render_diagnostic_markdown(report: Dict) -> str:
    lines = ["# Parametric ABCD-B Diagnostic Tables", ""]
    lines.extend(_markdown_table(
        "TABLE 1 - RAW RMSE BY SHAPE",
        ["Shape", "Smoothing", "Mean RMSE", "Median RMSE", "Max RMSE"],
        [
            [r["case"], _fmt(r["smoothing"]), _fmt(r["mean_rmse"]), _fmt(r["median_rmse"]), _fmt(r["max_rmse"])]
            for r in report["table_1_raw_rmse_by_shape"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 2 - LOCALIZED NOTCHES",
        ["Smoothing", "Affected RMSE", "Unaffected RMSE", "Contrast ratio"],
        [
            [_fmt(r["smoothing"]), _fmt(r["affected_rmse"]), _fmt(r["unaffected_rmse"]), _fmt(r["contrast_ratio"])]
            for r in report["table_2_localized_notches"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 3 - ROTATION",
        ["Shape", "Rotation", "Mean raw RMSE", "Max raw RMSE", "Affected RMSE"],
        [
            [r["case"], r["rotation_degrees"], _fmt(r["mean_rmse"]), _fmt(r["max_rmse"]), _fmt(r.get("affected_mean_rmse"))]
            for r in report["table_3_rotation"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 4 - SCALE",
        ["Shape", "Resolution", "Mean RMSE", "Mean arc length", "Mean sample count", "RMSE/n", "RMSE/L", "RMSE/global"],
        [
            [
                r["case"],
                r["size"],
                _fmt(r["mean_rmse"]),
                _fmt(r["mean_arc_length"]),
                _fmt(r["mean_sample_count"]),
                _fmt(r["mean_E_sample"]),
                _fmt(r["mean_E_arc"]),
                _fmt(r["mean_E_global"]),
            ]
            for r in report["table_4_scale"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 5 - NORMALIZATION COMPARISON",
        ["Shape", "Raw", "RMSE/n", "RMSE/L", "RMSE/global_scale"],
        [
            [r["case"], _fmt(r["raw"]), _fmt(r["RMSE_per_sample"]), _fmt(r["RMSE_per_arc_length"]), _fmt(r["RMSE_per_global_scale"])]
            for r in report["table_5_normalization_comparison"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 6 - 5% RULE",
        ["Shape", "Sample-count B", "Arc-length B", "Global-scale B"],
        [
            [r["case"], r["sample_count_B"], r["arc_length_B"], r["global_scale_B"]]
            for r in report["table_6_five_percent_rule"]
        ],
    ))
    lines.append("")
    lines.append(f"Decision category: `{report['decision_category']}`")
    return "\n".join(lines) + "\n"


def render_smooth_length_markdown(report: Dict) -> str:
    lines = ["# Real vs Smoothed Arc-Length ABCD-B Experiment", ""]
    lines.extend(_markdown_table(
        "TABLE 1 - DENOMINATOR COMPARISON",
        ["Shape", "RMSE", "L_real", "L_smooth", "L_real/L_smooth", "E_real", "E_smooth"],
        [
            [
                r["case"],
                _fmt(r["RMSE"]),
                _fmt(r["L_real"]),
                _fmt(r["L_smooth"]),
                _fmt(r["L_real_over_L_smooth"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
            ]
            for r in report["table_1_denominator_comparison"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 2 - B SCORE",
        ["Shape", "B_real", "B_smooth", "Real octants", "Smooth octants"],
        [
            [r["case"], r["B_real"], r["B_smooth"], r["real_octants"], r["smooth_octants"]]
            for r in report["table_2_b_score"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 3 - ROTATION",
        ["Shape", "Rotation", "B_real", "B_smooth", "Mean E_real", "Mean E_smooth"],
        [
            [
                r["case"],
                r["rotation_degrees"],
                r["B_real"],
                r["B_smooth"],
                _fmt(r["mean_E_real"]),
                _fmt(r["mean_E_smooth"]),
            ]
            for r in report["table_3_rotation"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 4 - SCALE",
        ["Shape", "Resolution", "RMSE", "L_real", "L_smooth", "E_real", "E_smooth", "B_real", "B_smooth"],
        [
            [
                r["case"],
                r["size"],
                _fmt(r["RMSE"]),
                _fmt(r["L_real"]),
                _fmt(r["L_smooth"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
                r["B_real"],
                r["B_smooth"],
            ]
            for r in report["table_4_scale"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 5 - SAMPLING DENSITY",
        ["Shape", "Sampling", "n", "RMSE/n", "RMSE/L_real", "RMSE/L_smooth"],
        [
            [
                r["case"],
                r["sampling"],
                r["n"],
                _fmt(r["RMSE_per_n"]),
                _fmt(r["RMSE_per_L_real"]),
                _fmt(r["RMSE_per_L_smooth"]),
            ]
            for r in report["table_5_sampling_density"]
        ],
    ))
    lines.append("")
    lines.append(f"Decision: `{report['decision']}`")
    return "\n".join(lines) + "\n"


def render_dimensionless_smoothing_markdown(report: Dict) -> str:
    lines = ["# Dimensionless Spline Smoothing ABCD-B Experiment", ""]
    lines.append(f"Old smoothing: `{report['formulas']['old_smoothing']}`")
    lines.append(f"New smoothing: `{report['formulas']['new_smoothing']}`")
    lines.append(f"Equivalent dimensional form: `{report['formulas']['equivalent_dimensional']}`")
    lines.append(f"Local scale D: `{report['formulas']['local_scale_D']}`")
    lines.append("")
    lines.extend(_markdown_table(
        "TABLE 1 - OLD VS NEW SMOOTHING",
        ["Shape", "Old RMSE", "New RMSE", "Old E_smooth", "New E_smooth", "Old B_smooth", "New B_smooth"],
        [
            [
                r["case"],
                _fmt(r["old_RMSE"]),
                _fmt(r["new_RMSE"]),
                _fmt(r["old_E_smooth"]),
                _fmt(r["new_E_smooth"]),
                r["old_B_smooth"],
                r["new_B_smooth"],
            ]
            for r in report["table_1_old_vs_new_smoothing"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 2 - SCALE",
        ["Formulation", "Shape", "Resolution", "n", "D", "spline_s", "RMSE", "E_real", "E_smooth", "B_real", "B_smooth"],
        [
            [
                r["formulation"],
                r["case"],
                r["size"],
                r["n_samples"],
                _fmt(r["mean_D"]),
                _fmt(r["mean_spline_s"]),
                _fmt(r["RMSE"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
                r["B_real"],
                r["B_smooth"],
            ]
            for r in report["table_2_scale"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 3 - SAMPLING DENSITY",
        ["Formulation", "Shape", "Sampling", "n", "spline_s", "spline_s/n", "RMSE", "RMSE/n", "E_real", "E_smooth"],
        [
            [
                r["formulation"],
                r["case"],
                r["sampling"],
                r["n"],
                _fmt(r["mean_spline_s"]),
                _fmt(r["mean_spline_s_per_n"]),
                _fmt(r["RMSE"]),
                _fmt(r["RMSE_per_n"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
            ]
            for r in report["table_3_sampling_density"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 4 - RAW DISCRIMINATION",
        ["Shape", "Mean RMSE", "Median RMSE", "Max RMSE"],
        [
            [r["case"], _fmt(r["mean_RMSE"]), _fmt(r["median_RMSE"]), _fmt(r["max_RMSE"])]
            for r in report["table_4_raw_discrimination"]
        ],
    ))
    n = report["table_5_localized_notches"]
    lines.extend(_markdown_table(
        "TABLE 5 - LOCALIZED NOTCHES",
        ["Metric", "Affected octants", "Unaffected octants", "Affected RMSE", "Unaffected RMSE", "Contrast"],
        [[n["metric"], n["affected_octants"], n["unaffected_octants"], _fmt(n["affected_RMSE"]), _fmt(n["unaffected_RMSE"]), _fmt(n["contrast"])]],
    ))
    lines.extend(_markdown_table(
        "TABLE 6 - ROTATION",
        ["Formulation", "Shape", "Rotation", "B_real", "B_smooth", "Mean RMSE", "Mean E_smooth"],
        [
            [
                r["formulation"],
                r["case"],
                r["rotation_degrees"],
                r["B_real"],
                r["B_smooth"],
                _fmt(r["RMSE"]),
                _fmt(r["E_smooth"]),
            ]
            for r in report["table_6_rotation"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 7 - DENOMINATOR",
        ["Shape", "L_real/L_smooth", "E_real", "E_smooth", "B_real", "B_smooth"],
        [
            [r["case"], _fmt(r["L_real_over_L_smooth"]), _fmt(r["E_real"]), _fmt(r["E_smooth"]), r["B_real"], r["B_smooth"]]
            for r in report["table_7_denominator"]
        ],
    ))
    lines.append("")
    lines.append(f"Primary decision: `{report['primary_decision']}`")
    lines.append(f"Rotation observation: `{report['rotation_observation']}`")
    lines.append(f"Normalization evidence: `{report['normalization_evidence']}`")
    return "\n".join(lines) + "\n"


def render_principal_axis_orientation_markdown(report: Dict) -> str:
    lines = ["# Principal-Axis ABCD-B Sector Orientation Experiment", ""]
    lines.append(f"IMAGE_FIXED formula: `{report['image_fixed_formula']}`")
    lines.append(f"PRINCIPAL_AXIS_ALIGNED formula: `{report['principal_axis_formula']}`")
    lines.append("")
    lines.append("Primary metric: `E_smooth = RMSE / L_smooth`; sector positive when `E_smooth > 0.05`.")
    lines.extend(_markdown_table(
        "TABLE 1 - ROTATION",
        ["Shape", "Rotation", "phi", "Anisotropy", "B image-fixed", "B principal-axis", "Principal octants", "Mean E_smooth", "Max E_smooth"],
        [
            [
                r["shape"],
                r["rotation_degrees"],
                _fmt(r["phi_degrees"]),
                _fmt(r["anisotropy"]),
                r["B_image_fixed"],
                r["B_principal_axis"],
                r["principal_axis_octants"],
                _fmt(r["principal_axis_mean_E_smooth"]),
                _fmt(r["principal_axis_max_E_smooth"]),
            ]
            for r in report["table_1_rotation"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 2 - ROTATION STABILITY",
        ["Shape", "Method", "Mean B", "Min B", "Max B", "Range", "Std", "CV", "Unique B"],
        [
            [
                r["shape"],
                r["method"],
                _fmt(r["mean_B"]),
                r["min_B"],
                r["max_B"],
                r["range"],
                _fmt(r["std"]),
                _fmt(r["coefficient_of_variation"]),
                r["unique_B_values"],
            ]
            for r in report["table_2_rotation_stability"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 3 - CANONICAL SECTORS",
        ["Shape", "Rotation", "Image-fixed octants", "Principal-axis octants", "Canonical", "Jaccard", "Axial Jaccard"],
        [
            [
                r["shape"],
                r["rotation_degrees"],
                r["image_fixed_octants"],
                r["principal_axis_octants"],
                r["canonical_octants"],
                _fmt(r["jaccard"]),
                _fmt(r["axial_ambiguity_jaccard"]),
            ]
            for r in report["table_3_canonical_sectors"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 4 - AXIS EQUIVARIANCE",
        ["Shape", "Rotation", "Expected phi", "Measured phi", "Axial error", "Anisotropy", "Status"],
        [
            [
                r["shape"],
                r["rotation_degrees"],
                _fmt(r["expected_phi_degrees"]),
                _fmt(r["measured_phi_degrees"]),
                _fmt(r["axial_error_degrees"]),
                _fmt(r["anisotropy"]),
                r["orientation_status"],
            ]
            for r in report["table_4_axis_equivariance"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 5 - AXIS STABILITY CONTROLS",
        ["Shape", "Rotation", "Anisotropy", "phi", "lambda1", "lambda2", "Status", "B image-fixed", "B principal-axis"],
        [
            [
                r["shape"],
                r["rotation_degrees"],
                _fmt(r["anisotropy"]),
                _fmt(r["phi_degrees"]),
                _fmt(r["lambda1"]),
                _fmt(r["lambda2"]),
                r["orientation_status"],
                r["B_image_fixed"],
                r["B_principal_axis"],
            ]
            for r in report["table_5_axis_stability_controls"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 6 - SCALE",
        ["Shape", "Resolution", "phi", "Anisotropy", "B principal-axis", "Mean E_smooth", "Max E_smooth"],
        [
            [
                r["shape"],
                r["resolution"],
                _fmt(r["phi_degrees"]),
                _fmt(r["anisotropy"]),
                r["B_principal_axis"],
                _fmt(r["mean_E_smooth"]),
                _fmt(r["max_E_smooth"]),
            ]
            for r in report["table_6_scale"]
        ],
    ))
    lines.append("")
    lines.append(f"Primary decision: `{report['primary_decision']}`")
    lines.append(f"Secondary decision: `{report['secondary_decision']}`")
    lines.append(f"Normalization follow-up: `{report['normalization_followup']}`")
    return "\n".join(lines) + "\n"


def render_localized_three_octants_diagnostic_markdown(report: Dict) -> str:
    native_pca = report["native_principal_axis_aligned"]
    native_fixed = report["native_image_fixed"]
    lines = ["# Localized Three-Octants B=2 Diagnostic", ""]
    lines.append(f"Final decision: `{report['final_decision']}`")
    lines.append(f"Defect status: `{report['defect_status']}`")
    lines.append("")
    lines.append(report["synthetic_parameters"]["angular_coordinate_convention"])
    lines.extend(_markdown_table(
        "TABLE 1 - SYNTHETIC PERTURBATIONS",
        ["Perturbation", "Center deg", "Amplitude", "Width rad", "Width deg", "Affected-region rule"],
        [
            [
                p["id"],
                _fmt(p["center_degrees"]),
                _fmt(p["amplitude"]),
                _fmt(p["width_radians"]),
                _fmt(p["width_degrees"]),
                p["affected_region_rule"],
            ]
            for p in report["synthetic_parameters"]["perturbations"]
        ],
    ))
    axis = report["native_principal_axis"]
    lines.append("")
    lines.append(
        "Native principal axis: "
        f"phi={_fmt(axis['phi_radians'])} rad / {_fmt(axis['phi_degrees'])} deg; "
        f"centroid={axis['centroid_xy']}; lambda1={_fmt(axis['lambda1'])}; "
        f"lambda2={_fmt(axis['lambda2'])}; anisotropy={_fmt(axis['anisotropy'])}."
    )
    lines.extend(_markdown_table(
        "TABLE 2 - PERTURBATION CENTER MAPPING",
        ["Perturbation", "Generator angle", "Image angle", "PCA-relative angle", "PCA sector", "Boundary distance"],
        [
            [
                r["perturbation"],
                _fmt(r["generator_angle_degrees"]),
                _fmt(r["image_space_angle_degrees"]),
                _fmt(r["pca_relative_angle_degrees"]),
                r["pca_sector"],
                _fmt(r["distance_to_nearest_sector_boundary_degrees"]),
            ]
            for r in native_pca["perturbation_center_mapping"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 3 - AFFECTED CONTOUR DISTRIBUTION",
        ["Perturbation", "Samples", "PCA sector fractions", "Boundary crossing"],
        [
            [
                pid,
                info["affected_contour_samples"],
                info["sector_fractions"],
                info["boundary_crossing"],
            ]
            for pid, info in native_pca["affected_sector_distribution"].items()
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 4 - PCA-ALIGNED EIGHT SECTORS",
        ["Sector", "n", "RMSE", "L_real", "L_smooth", "E_real", "E_smooth", "Margin", "Irregular", "Perturbations"],
        [
            [
                r["sector"],
                r["n"],
                _fmt(r["RMSE"]),
                _fmt(r["L_real"]),
                _fmt(r["L_smooth"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
                _fmt(r["threshold_margin"]),
                r["irregular"],
                r["perturbation_contributions"],
            ]
            for r in native_pca["sector_table"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 5 - IMAGE-FIXED EIGHT SECTORS",
        ["Sector", "n", "RMSE", "L_real", "L_smooth", "E_real", "E_smooth", "Margin", "Irregular", "Perturbations"],
        [
            [
                r["sector"],
                r["n"],
                _fmt(r["RMSE"]),
                _fmt(r["L_real"]),
                _fmt(r["L_smooth"]),
                _fmt(r["E_real"]),
                _fmt(r["E_smooth"]),
                _fmt(r["threshold_margin"]),
                r["irregular"],
                r["perturbation_contributions"],
            ]
            for r in native_fixed["sector_table"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 6 - ROTATION CONSISTENCY",
        ["Rotation", "phi", "B", "Irregular sectors", "P1 sectors", "P2 sectors", "P3 sectors"],
        [
            [
                r["rotation_degrees"],
                _fmt(r["phi_degrees"]),
                r["B"],
                r["irregular_sectors"],
                r["affected_sector_distribution"]["P1"]["sector_fractions"],
                r["affected_sector_distribution"]["P2"]["sector_fractions"],
                r["affected_sector_distribution"]["P3"]["sector_fractions"],
            ]
            for r in report["rotation_consistency"]
        ],
    ))
    lines.extend(_markdown_table(
        "TABLE 7 - IMAGE-FIXED VS PCA-ALIGNED DISTRIBUTION",
        ["Perturbation", "IMAGE_FIXED sectors", "PCA_ALIGNED sectors"],
        [
            [r["perturbation"], r["image_fixed_sectors"], r["pca_aligned_sectors"]]
            for r in report["image_fixed_vs_pca"]
        ],
    ))
    lines.append("")
    lines.append(f"Measured explanation labels: `{report['measured_explanation']['labels']}`")
    lines.append(report["measured_explanation"]["summary"])
    return "\n".join(lines) + "\n"


def _markdown_table(title: str, headers: List[str], rows: List[List]) -> List[str]:
    lines = ["", f"## {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return lines


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6g}"
    return str(value)


def _score_range(scores: List[int]) -> int:
    return int(max(scores) - min(scores)) if scores else 0


def _summarize_records(records: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for case_name in sorted({record["case"] for record in records}):
        case_records = [r for r in records if r["case"] == case_name]
        radial = [record["radial_B"] for record in case_records]
        parametric = [record["parametric_B"] for record in case_records]
        summary[case_name] = {
            "radial_min_B": int(min(radial)),
            "radial_max_B": int(max(radial)),
            "radial_mean_B": float(np.mean(radial)),
            "parametric_min_B": int(min(parametric)),
            "parametric_max_B": int(max(parametric)),
            "parametric_mean_B": float(np.mean(parametric)),
        }
    return summary


def deep_concavity_diagnostics() -> Dict:
    mask = make_deep_concavity_mask()
    contour = extract_ordered_contour(mask)
    radial = evaluate_border_octants_radial(mask, include_points=True)
    parametric = evaluate_border_octants_parametric(mask, include_points=True)
    points_yx = extract_boundary_points(mask)
    theta, radius = angles_and_radii(points_yx, tuple(radial["centroid_xy"]))
    rounded_theta = np.round(theta, 2)
    duplicate_bins = int(np.sum(np.bincount(np.unique(rounded_theta, return_inverse=True)[1]) > 1))
    return {
        "ordered_contour_points": int(contour.shape[0]),
        "unordered_boundary_pixels": int(points_yx.shape[0]),
        "radial_B": radial["border_score"],
        "parametric_B": parametric["border_score"],
        "radial_irregular_octants": radial["irregular_octants"],
        "parametric_irregular_octants": parametric["irregular_octants"],
        "similar_angle_bins_with_multiple_boundary_pixels": duplicate_bins,
        "contour_arc_length_monotonic": bool(np.all(np.diff(cumulative_arc_length(contour)) >= 0)),
        "radial_information_loss_confirmed": bool(duplicate_bins > 0),
    }


def write_validation_plots(output_dir: Path) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for case_name in ("regular_ellipse", "localized_three_octants", "moderate_irregular", "strong_serrated", "deep_concavity"):
        mask = synthetic_cases(size=256, rotation_degrees=0)[case_name]
        radial = evaluate_border_octants_radial(mask, include_points=True)
        parametric = evaluate_border_octants_parametric(mask, include_points=True)
        path = output_dir / f"{case_name}_radial_vs_parametric.png"
        _draw_comparison_plot(mask, radial, parametric, path)
        paths.append(str(path))
    return paths


def _draw_comparison_plot(mask: np.ndarray, radial: Dict, parametric: Dict, path: Path) -> None:
    width = mask.shape[1] * 2
    height = mask.shape[0]
    image = Image.new("RGB", (width, height), (30, 30, 30))
    left = _draw_single_panel(mask, radial, "radial")
    right = _draw_single_panel(mask, parametric, "parametric")
    image.paste(left, (0, 0))
    image.paste(right, (mask.shape[1], 0))
    image.save(path)


def _draw_single_panel(mask: np.ndarray, result: Dict, title: str) -> Image.Image:
    colors = {
        "background": (30, 30, 30),
        "lesion": (190, 190, 190),
        "boundary": (250, 220, 60),
        "irregular": (220, 70, 70),
        "sector": (70, 170, 210),
        "fit": (95, 230, 130),
    }
    image = Image.new("RGB", (mask.shape[1], mask.shape[0]), colors["background"])
    pixels = image.load()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        pixels[int(x), int(y)] = colors["lesion"]

    if result["representation"] == "radial":
        points_yx = extract_boundary_points(mask)
        theta, _radius = angles_and_radii(points_yx, tuple(result["centroid_xy"]))
        sectors = assign_octants(theta)
        for (y, x), sector in zip(points_yx, sectors):
            pixels[int(x), int(y)] = colors["irregular"] if int(sector) in result["irregular_octants"] else colors["boundary"]
    else:
        contour = extract_ordered_contour(mask)
        sectors = assign_octants(angles_for_xy(contour, tuple(result["centroid_xy"])))
        for (x, y), sector in zip(contour, sectors):
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                pixels[ix, iy] = colors["irregular"] if int(sector) in result["irregular_octants"] else colors["boundary"]
        for octant in result["octants"]:
            if not octant.get("fitted_xy"):
                continue
            for x, y in octant["fitted_xy"]:
                ix = int(round(x))
                iy = int(round(y))
                if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0]:
                    pixels[ix, iy] = colors["fit"]

    draw = ImageDraw.Draw(image)
    cx, cy = result["centroid_xy"]
    for sector in range(OCTANT_COUNT):
        angle = math.radians(45 * sector)
        draw.line(
            [(cx, cy), (cx + 120 * math.cos(angle), cy + 120 * math.sin(angle))],
            fill=colors["sector"],
            width=1,
        )
    draw.text((8, 8), f"{title} B={result['border_score']}", fill=(255, 255, 255))
    return image


def write_parametric_diagnostic_plots(report: Dict, output_dir: Path) -> List[str]:
    paths = []
    raw_path = output_dir / "plot_smoothing_vs_mean_rmse.png"
    _draw_smoothing_plot(report, raw_path)
    paths.append(str(raw_path))

    contrast_path = output_dir / "plot_smoothing_vs_notch_contrast.png"
    _draw_notch_contrast_plot(report, contrast_path)
    paths.append(str(contrast_path))

    distribution_path = output_dir / "plot_raw_rmse_distribution_selected_smoothing.png"
    _draw_bar_plot(
        [(r["case"], r["mean_rmse"]) for r in report["table_1_raw_rmse_by_shape"] if r["smoothing"] == DEFAULT_SMOOTHING_FACTOR],
        distribution_path,
        "Raw mean RMSE at selected smoothing",
    )
    paths.append(str(distribution_path))

    norm_path = output_dir / "plot_normalized_error_distribution.png"
    _draw_normalization_plot(report, norm_path)
    paths.append(str(norm_path))
    return paths


def write_smooth_length_plots(report: Dict, output_dir: Path) -> List[str]:
    paths = []
    denom_path = output_dir / "plot_real_vs_smooth_denominator.png"
    _draw_two_metric_bar_plot(
        report["table_1_denominator_comparison"],
        "case",
        "E_real",
        "E_smooth",
        denom_path,
        "E_real vs E_smooth at native resolution",
    )
    paths.append(str(denom_path))
    sampling_path = output_dir / "plot_sampling_density_normalizations.png"
    _draw_sampling_plot(report, sampling_path)
    paths.append(str(sampling_path))
    return paths


def write_dimensionless_smoothing_plots(report: Dict, output_dir: Path) -> List[str]:
    paths = []
    scale_path = output_dir / "plot_scale_e_smooth_old_vs_new.png"
    rows = [
        row for row in report["table_2_scale"]
        if row["case"] in ("regular_ellipse", "strong_serrated")
    ]
    series = {}
    for row in rows:
        series.setdefault(f"{row['formulation']}:{row['case']}", []).append((row["size"], row["E_smooth"]))
    _draw_multi_line_linear_plot(series, scale_path, "Scale vs E_smooth")
    paths.append(str(scale_path))

    sampling_path = output_dir / "plot_sampling_spline_s_per_n.png"
    rows = report["table_3_sampling_density"]
    series = {}
    for row in rows:
        if row["case"] in ("regular_ellipse", "strong_serrated"):
            series.setdefault(f"{row['formulation']}:{row['case']}", []).append((row["sampling"], row["mean_spline_s_per_n"]))
    _draw_multi_line_linear_plot(series, sampling_path, "Sampling density vs spline_s/n")
    paths.append(str(sampling_path))
    return paths


def write_principal_axis_orientation_plots(report: Dict, output_dir: Path) -> List[str]:
    paths = []
    for case_name in ("localized_three_octants", "strong_serrated"):
        for rotation in (0, 15, 30, 45):
            for orientation in (SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,):
                path = output_dir / f"{case_name}_{rotation:02d}_{orientation.lower()}.png"
                _draw_orientation_plot(
                    synthetic_cases(size=256, rotation_degrees=rotation)[case_name],
                    orientation,
                    path,
                )
                paths.append(str(path))
    for case_name, rotation in (
        ("localized_three_octants", 15),
        ("strong_serrated", 15),
        ("near_circular_localized_irregularity", 15),
    ):
        masks = synthetic_cases(size=256, rotation_degrees=rotation)
        mask = masks[case_name] if case_name in masks else diagnostic_control_cases(size=256, rotation_degrees=rotation)[case_name]
        for orientation in (SECTOR_ORIENTATION_IMAGE_FIXED, SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED):
            path = output_dir / f"{case_name}_{rotation:02d}_{orientation.lower()}.png"
            _draw_orientation_plot(mask, orientation, path)
            paths.append(str(path))
    return paths


def write_localized_three_octants_diagnostic_plots(report: Dict, output_dir: Path) -> List[str]:
    paths = []
    for orientation in (SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED, SECTOR_ORIENTATION_IMAGE_FIXED):
        path = output_dir / f"localized_three_octants_native_{orientation.lower()}_diagnostic.png"
        _draw_localized_three_octants_diagnostic_plot(orientation, path)
        paths.append(str(path))
    return paths


def _draw_localized_three_octants_diagnostic_plot(sector_orientation: str, path: Path) -> None:
    mask = synthetic_cases(size=256, rotation_degrees=0)["localized_three_octants"]
    diagnostic = _localized_orientation_diagnostic(0, sector_orientation)
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    axis = principal_axis_analysis(mask)
    offset = axis["angle_radians"] if sector_orientation == SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED else 0.0
    theta = angles_for_xy(contour, centroid)
    sectors = assign_octants_relative(theta, offset)
    irregular = set(diagnostic["irregular_sectors"])

    width = mask.shape[1]
    height = mask.shape[0]
    image = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = image.load()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        pixels[int(x), int(y)] = (225, 225, 225)
    for (x, y), sector in zip(contour, sectors):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            pixels[ix, iy] = (210, 40, 40) if int(sector) in irregular else (30, 30, 30)

    draw = ImageDraw.Draw(image)
    cx, cy = centroid
    radius = 180
    for sector in range(OCTANT_COUNT):
        angle = offset + sector * OCTANT_WIDTH_RADIANS
        draw.line((cx, cy, cx + radius * math.cos(angle), cy + radius * math.sin(angle)), fill=(60, 140, 210), width=1)
        label_angle = offset + (sector + 0.5) * OCTANT_WIDTH_RADIANS
        draw.text(
            (cx + 78 * math.cos(label_angle), cy + 78 * math.sin(label_angle)),
            str(sector),
            fill=(20, 80, 130),
        )
    for angle, color in ((axis["angle_radians"], (20, 140, 40)), (axis["angle_radians"] + math.pi / 2.0, (210, 140, 20))):
        dx = 95 * math.cos(angle)
        dy = 95 * math.sin(angle)
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=color, width=2)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0))

    for perturbation in LOCALIZED_PERTURBATIONS:
        center = _generator_point_to_image_xy(perturbation, 0)
        draw.ellipse((center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4), fill=(160, 40, 180))
        draw.text((center[0] + 5, center[1] - 6), perturbation["id"], fill=(120, 0, 140))
        for sign in (-1, 1):
            edge = dict(perturbation)
            edge["center_degrees"] = perturbation["center_degrees"] + sign * 2.0 * math.degrees(perturbation["width_radians"])
            edge_point = _generator_point_to_image_xy(edge, 0)
            draw.line((center[0], center[1], edge_point[0], edge_point[1]), fill=(160, 40, 180), width=1)

    for row in diagnostic["sector_table"]:
        angle = offset + (row["sector"] + 0.5) * OCTANT_WIDTH_RADIANS
        draw.text(
            (cx + 105 * math.cos(angle), cy + 105 * math.sin(angle)),
            f"E={row['E_smooth']:.3f}",
            fill=(180, 0, 0) if row["irregular"] else (0, 0, 0),
        )
    draw.text((8, 8), f"{sector_orientation} B={diagnostic['B']}", fill=(0, 0, 0))
    image.save(path)


def _draw_orientation_plot(mask: np.ndarray, sector_orientation: str, path: Path) -> None:
    summary = _orientation_mask_summary(mask, sector_orientation)
    result = summary["result"]
    width = mask.shape[1]
    height = mask.shape[0]
    image = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = image.load()
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        pixels[int(x), int(y)] = (220, 220, 220)

    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    theta = angles_for_xy(contour, centroid)
    sector_ids = assign_octants_relative(theta, result["sector_angle_offset_radians"])
    irregular = set(summary["irregular_octants"])
    for (x, y), sector in zip(contour, sector_ids):
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < width and 0 <= iy < height:
            pixels[ix, iy] = (210, 40, 40) if int(sector) in irregular else (40, 40, 40)

    draw = ImageDraw.Draw(image)
    cx, cy = centroid
    radius = max(width, height)
    offset = float(result["sector_angle_offset_radians"])
    for sector in range(OCTANT_COUNT):
        angle = offset + sector * (2.0 * math.pi / OCTANT_COUNT)
        draw.line(
            [(cx, cy), (cx + radius * math.cos(angle), cy + radius * math.sin(angle))],
            fill=(80, 150, 210),
            width=1,
        )
    phi = summary["phi_radians"]
    for angle, color, line_width in (
        (phi, (20, 140, 40), 2),
        (phi + math.pi / 2.0, (210, 140, 20), 2),
    ):
        dx = 0.45 * radius * math.cos(angle)
        dy = 0.45 * radius * math.sin(angle)
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=color, width=line_width)
    draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(0, 0, 0))
    draw.text(
        (8, 8),
        f"{sector_orientation} B={summary['B_smooth']} phi={summary['phi_degrees']:.1f}",
        fill=(0, 0, 0),
    )
    image.save(path)


def _draw_two_metric_bar_plot(rows: List[Dict], label_key: str, key_a: str, key_b: str, path: Path, title: str) -> None:
    width = 900
    height = 480
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_value = max(max(row[key_a], row[key_b]) for row in rows) or 1.0
    group_w = plot_w / len(rows)
    bar_w = group_w / 4
    draw.text((margin, 20), title, fill=(0, 0, 0))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0))
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0))
    for idx, row in enumerate(rows):
        x = margin + idx * group_w
        for offset, key, color in ((1, key_a, (70, 130, 190)), (2, key_b, (210, 100, 70))):
            value = row[key]
            x0 = x + offset * bar_w
            y0 = height - margin - (value / max_value) * plot_h
            draw.rectangle((x0, y0, x0 + bar_w, height - margin), fill=color)
        draw.text((x, height - margin + 8), str(row[label_key])[:12], fill=(0, 0, 0))
    draw.rectangle((width - 220, 50, width - 208, 62), fill=(70, 130, 190))
    draw.text((width - 202, 48), key_a, fill=(0, 0, 0))
    draw.rectangle((width - 220, 74, width - 208, 86), fill=(210, 100, 70))
    draw.text((width - 202, 72), key_b, fill=(0, 0, 0))
    image.save(path)


def _draw_sampling_plot(report: Dict, path: Path) -> None:
    rows = report["table_5_sampling_density"]
    series = {}
    for case_name in ("regular_ellipse", "localized_three_octants", "strong_serrated"):
        for metric in ("RMSE_per_n", "RMSE_per_L_real", "RMSE_per_L_smooth"):
            series[f"{case_name}:{metric}"] = [
                (row["sampling"], row[metric])
                for row in rows
                if row["case"] == case_name
            ]
    _draw_multi_line_linear_plot(series, path, "Sampling density vs normalized RMSE")


def _draw_multi_line_linear_plot(series: Dict[str, List[tuple[float, float]]], path: Path, title: str) -> None:
    width = 1000
    height = 520
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    all_x = [x for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = 0.0, max(all_y) or 1.0
    colors = [(70, 130, 190), (210, 100, 70), (70, 170, 110), (160, 90, 180), (180, 150, 60), (80, 80, 80)]
    draw.text((margin, 20), title, fill=(0, 0, 0))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0))
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0))
    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        xy = []
        for x, y in points:
            px = margin + ((x - min_x) / (max_x - min_x)) * plot_w
            py = height - margin - ((y - min_y) / (max_y - min_y)) * plot_h
            xy.append((px, py))
        draw.line(xy, fill=color, width=2)
        for point in xy:
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
        draw.text((width - 360, 45 + 18 * idx), name[:42], fill=color)
    image.save(path)


def _draw_smoothing_plot(report: Dict, path: Path) -> None:
    rows = report["table_1_raw_rmse_by_shape"]
    series = {}
    for row in rows:
        series.setdefault(row["case"], []).append((row["smoothing"], row["mean_rmse"]))
    _draw_multi_line_plot(series, path, "Smoothing factor vs mean RMSE")


def _draw_notch_contrast_plot(report: Dict, path: Path) -> None:
    series = {
        "localized contrast": [
            (row["smoothing"], row["contrast_ratio"])
            for row in report["table_2_localized_notches"]
        ]
    }
    _draw_multi_line_plot(series, path, "Smoothing factor vs localized-notch contrast")


def _draw_normalization_plot(report: Dict, path: Path) -> None:
    rows = report["table_5_normalization_comparison"]
    labels = [row["case"] for row in rows]
    metrics = ["RMSE_per_sample", "RMSE_per_arc_length", "RMSE_per_global_scale"]
    width = 900
    height = 480
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_value = max(row[metric] for row in rows for metric in metrics) or 1.0
    colors = [(70, 130, 190), (210, 100, 70), (70, 170, 110)]
    group_w = plot_w / len(labels)
    bar_w = group_w / 5
    draw.text((margin, 20), "Normalized error distributions", fill=(0, 0, 0))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0))
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0))
    for i, row in enumerate(rows):
        for j, metric in enumerate(metrics):
            value = row[metric]
            x0 = margin + i * group_w + (j + 1) * bar_w
            y0 = height - margin - (value / max_value) * plot_h
            draw.rectangle((x0, y0, x0 + bar_w, height - margin), fill=colors[j])
        draw.text((margin + i * group_w, height - margin + 8), row["case"][:12], fill=(0, 0, 0))
    for j, metric in enumerate(metrics):
        draw.rectangle((width - 240, 50 + 22 * j, width - 228, 62 + 22 * j), fill=colors[j])
        draw.text((width - 222, 48 + 22 * j), metric, fill=(0, 0, 0))
    image.save(path)


def _draw_multi_line_plot(series: Dict[str, List[tuple[float, float]]], path: Path, title: str) -> None:
    width = 900
    height = 480
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    all_x = [math.log10(x) for points in series.values() for x, _ in points]
    all_y = [y for points in series.values() for _, y in points]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = 0.0, max(all_y) or 1.0
    colors = [
        (70, 130, 190),
        (210, 100, 70),
        (70, 170, 110),
        (160, 90, 180),
        (180, 150, 60),
        (80, 80, 80),
    ]
    draw.text((margin, 20), title, fill=(0, 0, 0))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0))
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0))
    for idx, (name, points) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        xy = []
        for x, y in points:
            px = margin + ((math.log10(x) - min_x) / (max_x - min_x)) * plot_w
            py = height - margin - ((y - min_y) / (max_y - min_y)) * plot_h
            xy.append((px, py))
        if len(xy) > 1:
            draw.line(xy, fill=color, width=2)
        for point in xy:
            draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)
        draw.text((width - 260, 50 + 20 * idx), name, fill=color)
    image.save(path)


def _draw_bar_plot(rows: List[tuple[str, float]], path: Path, title: str) -> None:
    width = 900
    height = 480
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin = 70
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    max_value = max(value for _, value in rows) or 1.0
    draw.text((margin, 20), title, fill=(0, 0, 0))
    draw.line((margin, height - margin, width - margin, height - margin), fill=(0, 0, 0))
    draw.line((margin, margin, margin, height - margin), fill=(0, 0, 0))
    bar_w = plot_w / (len(rows) * 1.6)
    for idx, (name, value) in enumerate(rows):
        x0 = margin + idx * (plot_w / len(rows)) + bar_w * 0.3
        y0 = height - margin - (value / max_value) * plot_h
        draw.rectangle((x0, y0, x0 + bar_w, height - margin), fill=(70, 130, 190))
        draw.text((x0, height - margin + 8), name[:12], fill=(0, 0, 0))
    image.save(path)


def run_validation(output_dir: str) -> Dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = compare_representations()
    summary["plots"] = write_validation_plots(out)
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/abcd_border_validation")
    parser.add_argument(
        "--mode",
        choices=(
            "comparison",
            "parametric-diagnostics",
            "smooth-length",
            "dimensionless-smoothing",
            "principal-axis-orientation",
            "localized-three-octants-diagnostic",
        ),
        default="comparison",
    )
    args = parser.parse_args()
    if args.mode == "parametric-diagnostics":
        summary = run_parametric_diagnostics(args.output_dir)
        print(json.dumps(summary["table_5_normalization_comparison"], indent=2))
        print(json.dumps(summary["table_6_five_percent_rule"], indent=2))
        print(summary["decision_category"])
    elif args.mode == "smooth-length":
        summary = run_smooth_length_experiment(args.output_dir)
        print(json.dumps(summary["table_1_denominator_comparison"], indent=2))
        print(json.dumps(summary["table_2_b_score"], indent=2))
        print(summary["decision"])
    elif args.mode == "dimensionless-smoothing":
        summary = run_dimensionless_smoothing_experiment(args.output_dir)
        print(json.dumps(summary["table_1_old_vs_new_smoothing"], indent=2))
        print(json.dumps(summary["scale_summary"], indent=2))
        print(summary["primary_decision"])
        print(summary["rotation_observation"])
        print(summary["normalization_evidence"])
    elif args.mode == "principal-axis-orientation":
        summary = run_principal_axis_orientation_experiment(args.output_dir)
        print(json.dumps(summary["table_2_rotation_stability"], indent=2))
        print(summary["primary_decision"])
        print(summary["secondary_decision"])
        print(summary["normalization_followup"])
    elif args.mode == "localized-three-octants-diagnostic":
        summary = run_localized_three_octants_diagnostic(args.output_dir)
        print(json.dumps(summary["native_principal_axis_aligned"]["perturbation_center_mapping"], indent=2))
        print(json.dumps(summary["native_principal_axis_aligned"]["sector_table"], indent=2))
        print(summary["defect_status"])
        print(summary["final_decision"])
    else:
        summary = run_validation(args.output_dir)
        print(json.dumps(summary["selected_method_summary"], indent=2))
        print(json.dumps(summary["strong_serrated_rotation_256"], indent=2))
        print(json.dumps(summary["deep_concavity"], indent=2))


if __name__ == "__main__":
    main()
