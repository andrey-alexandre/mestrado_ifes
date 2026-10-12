import math

import numpy as np

from src.features.border import (
    DEFAULT_SMOOTHING_FACTOR,
    assign_octants,
    cumulative_arc_length,
    euclidean_rmse,
    evaluate_border_octants,
    evaluate_border_octants_parametric,
    extract_abcd_border,
    extract_boundary_points,
    extract_ordered_contour,
    is_irregular_sector,
    normalize_angles,
    rmse,
    sector_threshold,
)
from src.features.border_validation import make_deep_concavity_mask, synthetic_cases
from src.features.schemas import EvidenceStatus


def test_boundary_extraction_returns_contour_pixels_only():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2:7, 2:7] = True

    points = extract_boundary_points(mask)

    assert points.shape == (16, 2)
    assert [4.0, 4.0] not in points.tolist()


def test_angle_normalization_and_octant_assignment():
    angles = normalize_angles(np.array([-math.pi / 2, 0, math.pi / 4, 2 * math.pi]))

    assert np.isclose(angles[0], 1.5 * math.pi)
    assert np.isclose(angles[3], 0.0)
    assert assign_octants(np.array([0, math.radians(44.9), math.radians(45), math.radians(359)])).tolist() == [0, 0, 1, 7]


def test_rmse_and_threshold_rule_are_strict():
    assert rmse(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 6.0])) == math.sqrt(3.0)
    assert sector_threshold(60) == 3.0
    assert is_irregular_sector(3.0, 60) is False
    assert is_irregular_sector(3.01, 60) is True
    assert is_irregular_sector(None, 60) is False


def test_nearly_empty_sector_is_explicitly_not_marked_irregular():
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True

    result = evaluate_border_octants(mask)

    assert result["border_score"] == 0
    assert any(octant["insufficient_points"] for octant in result["octants"])


def test_extract_abcd_border_returns_approved_limited_score():
    mask = synthetic_cases(size=256)["localized_three_octants"]

    result = extract_abcd_border(mask)

    assert result.status == EvidenceStatus.DETERMINISTIC_CLINICAL_WITH_LIMITATION
    assert result.value == 2
    assert result.evidence["representation"] == "parametric"
    assert result.evidence["sector_orientation"] == "PRINCIPAL_AXIS_ALIGNED"
    assert result.evidence["smoothing_factor"] == DEFAULT_SMOOTHING_FACTOR
    assert result.evidence["irregular_octants"] == [1, 4]


def test_ordered_contour_preserves_deep_concavity_geometry():
    mask = make_deep_concavity_mask()

    contour = extract_ordered_contour(mask)
    boundary = extract_boundary_points(mask)

    assert contour.shape[0] >= boundary.shape[0]
    assert contour[:, 0].min() < 90
    assert contour[:, 0].max() > 190


def test_contour_points_are_ordered_along_boundary():
    contour = extract_ordered_contour(synthetic_cases(size=256)["regular_ellipse"])
    closed = np.vstack([contour, contour[0]])
    steps = np.hypot(np.diff(closed[:, 0]), np.diff(closed[:, 1]))

    assert steps.max() <= 1.0


def test_every_ordered_contour_point_gets_exactly_one_octant():
    mask = synthetic_cases(size=256)["strong_serrated"]
    contour = extract_ordered_contour(mask)
    centroid = [float(np.nonzero(mask)[1].mean()), float(np.nonzero(mask)[0].mean())]
    angles = np.mod(np.arctan2(contour[:, 1] - centroid[1], contour[:, 0] - centroid[0]), 2 * np.pi)
    octants = assign_octants(angles)

    assert octants.shape[0] == contour.shape[0]
    assert octants.min() >= 0
    assert octants.max() <= 7


def test_parametric_path_does_not_average_by_theta_or_radius():
    mask = make_deep_concavity_mask()
    contour = extract_ordered_contour(mask)
    centroid = [float(np.nonzero(mask)[1].mean()), float(np.nonzero(mask)[0].mean())]
    rounded_angles = np.round(np.arctan2(contour[:, 1] - centroid[1], contour[:, 0] - centroid[0]), 2)
    duplicate_angle_bins = np.sum(np.bincount(np.unique(rounded_angles, return_inverse=True)[1]) > 1)
    result = evaluate_border_octants_parametric(mask, include_points=True)

    assert duplicate_angle_bins > 0
    assert sum(octant["boundary_point_count"] for octant in result["octants"]) == contour.shape[0]
    assert all(octant["unique_angle_count"] is None for octant in result["octants"])


def test_arc_length_is_monotonically_increasing_and_splines_are_valid():
    result = evaluate_border_octants_parametric(synthetic_cases(size=256)["localized_three_octants"], include_points=True)

    for octant in result["octants"]:
        if octant["insufficient_points"] or not octant["arc_length"]:
            continue
        arc = np.array(octant["arc_length"])
        fitted = np.array(octant["fitted_xy"])
        observed = np.array(octant["contour_xy"])
        assert np.all(np.diff(arc) >= 0)
        assert fitted.shape == observed.shape
        assert np.isfinite(fitted).all()


def test_euclidean_residual_is_computed_from_xy_distances():
    observed = np.array([[0.0, 0.0], [3.0, 4.0]])
    fitted = np.array([[0.0, 0.0], [0.0, 0.0]])

    assert euclidean_rmse(observed, fitted) == np.sqrt(12.5)


def test_parametric_b_score_is_integer_in_clinical_range():
    result = evaluate_border_octants_parametric(synthetic_cases(size=256)["strong_serrated"])

    assert isinstance(result["border_score"], int)
    assert 0 <= result["border_score"] <= 8
