import numpy as np
import pytest

from src.features.border import (
    DEFAULT_SMOOTHING_FACTOR,
    evaluate_border_octants_parametric,
    normalized_parametric_coordinates,
)
from src.features.border_validation import (
    build_dimensionless_smoothing_report,
    synthetic_cases,
)


@pytest.fixture(scope="module")
def dimensionless_report():
    return build_dimensionless_smoothing_report()


def test_normalized_arc_parameter_is_in_unit_interval():
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])

    normalized = normalized_parametric_coordinates(contour)

    assert normalized["u"][0] == 0.0
    assert normalized["u"][-1] == 1.0
    assert np.all(normalized["u"] >= 0.0)
    assert np.all(normalized["u"] <= 1.0)


def test_normalized_coordinates_are_finite_and_chord_positive_for_ordinary_octants():
    result = evaluate_border_octants_parametric(
        synthetic_cases(size=256)["regular_ellipse"],
        include_points=True,
    )

    for octant in result["octants"]:
        assert octant["local_chord_length"] > 0
        assert octant["degenerate_chord"] is False
        assert np.isfinite(np.array(octant["normalized_xy"])).all()


def test_degenerate_chord_handling_is_deterministic():
    contour = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

    normalized = normalized_parametric_coordinates(contour)

    assert normalized["degenerate_chord"] is True
    assert normalized["chord_fallback"] == "arc_length"
    assert normalized["local_scale"] > 0


def test_dimensionless_spline_s_equals_alpha_times_samples():
    result = evaluate_border_octants_parametric(synthetic_cases(size=256)["localized_three_octants"])

    for octant in result["octants"]:
        assert octant["spline_smoothing_parameter"] == DEFAULT_SMOOTHING_FACTOR * octant["boundary_point_count"]


def test_scaled_contours_have_equivalent_normalized_coordinates():
    contour = np.array([[2.0, 3.0], [4.0, 3.0], [5.0, 5.0], [8.0, 7.0]])
    scaled = contour * 3.0 + np.array([10.0, -4.0])

    a = normalized_parametric_coordinates(contour)
    b = normalized_parametric_coordinates(scaled)

    assert np.allclose(a["u"], b["u"])
    assert np.allclose(a["normalized_xy"], b["normalized_xy"])


def test_dimensionless_spline_geometry_is_scale_invariant_for_irregular_cases(dimensionless_report):
    report = dimensionless_report
    strong = [
        row for row in report["scale_summary"]
        if row["formulation"] == "dimensionless_chord" and row["case"] == "strong_serrated"
    ][0]

    assert strong["E_real_relative_variation"] < 0.02
    assert strong["E_smooth_relative_variation"] < 0.02


def test_raw_rmse_scales_with_spatial_scale(dimensionless_report):
    report = dimensionless_report
    rows = [
        row for row in report["table_2_scale"]
        if row["formulation"] == "dimensionless_chord" and row["case"] == "strong_serrated"
    ]

    assert rows[1]["RMSE"] / rows[0]["RMSE"] > 1.9
    assert rows[2]["RMSE"] / rows[0]["RMSE"] > 3.9


def test_sampling_density_keeps_dimensionless_s_per_n_constant(dimensionless_report):
    report = dimensionless_report
    rows = [
        row for row in report["table_3_sampling_density"]
        if row["formulation"] == "dimensionless_chord"
    ]

    assert all(row["mean_spline_s_per_n"] == 1.0 for row in rows)


def test_sampling_density_does_not_materially_change_geometric_normalized_error(dimensionless_report):
    report = dimensionless_report
    rows = [
        row for row in report["table_3_sampling_density"]
        if row["formulation"] == "dimensionless_chord" and row["case"] == "localized_three_octants"
    ]

    real_values = [row["E_real"] for row in rows]
    smooth_values = [row["E_smooth"] for row in rows]
    sample_values = [row["RMSE_per_n"] for row in rows]
    assert max(real_values) / min(real_values) < 1.1
    assert max(smooth_values) / min(smooth_values) < 1.1
    assert max(sample_values) / min(sample_values) > 7.0


def test_regular_low_error_and_irregular_raw_rmse_discrimination_remain(dimensionless_report):
    report = dimensionless_report
    rows = {row["case"]: row for row in report["table_4_raw_discrimination"]}

    assert rows["regular_ellipse"]["mean_RMSE"] < 0.25
    assert rows["localized_three_octants"]["mean_RMSE"] > rows["regular_ellipse"]["mean_RMSE"] * 8
    assert rows["strong_serrated"]["mean_RMSE"] > rows["regular_ellipse"]["mean_RMSE"] * 18


def test_localized_notch_affected_octants_retain_high_rmse_contrast(dimensionless_report):
    report = dimensionless_report

    assert report["table_5_localized_notches"]["contrast"] > 20


def test_dimensionless_b_remains_integer_in_clinical_range():
    result = evaluate_border_octants_parametric(synthetic_cases(size=256)["strong_serrated"])

    assert isinstance(result["border_score"], int)
    assert 0 <= result["border_score"] <= 8
