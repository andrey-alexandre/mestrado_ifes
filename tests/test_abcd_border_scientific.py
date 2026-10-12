import numpy as np

from src.features.border import (
    DEFAULT_SMOOTHING_FACTOR,
    evaluate_border_octants_parametric,
    evaluate_border_octants_radial,
)
from src.features.border_validation import make_deep_concavity_mask, synthetic_cases


def _radial(case_name, size=256, rotation=0):
    return evaluate_border_octants_radial(synthetic_cases(size=size, rotation_degrees=rotation)[case_name])


def _parametric(case_name, size=256, rotation=0):
    return evaluate_border_octants_parametric(synthetic_cases(size=size, rotation_degrees=rotation)[case_name])


def test_regular_and_smooth_lobulated_lesions_are_not_flagged_by_either_representation():
    for case_name in ("regular_ellipse", "smooth_lobulated"):
        assert _radial(case_name)["border_score"] == 0
        assert _parametric(case_name)["border_score"] == 0


def test_final_parametric_pca_lsmooth_detects_localized_thresholded_sectors():
    radial = _radial("localized_three_octants")
    parametric = _parametric("localized_three_octants")

    assert radial["border_score"] == 3
    assert radial["irregular_octants"] == [1, 4, 6]
    assert parametric["sector_orientation"] == "PRINCIPAL_AXIS_ALIGNED"
    assert parametric["border_score"] == 2
    assert parametric["irregular_octants"] == [1, 4]


def test_principal_axis_parametric_stabilizes_moderate_irregularity():
    radial_scores = [_radial("moderate_irregular", rotation=rotation)["border_score"] for rotation in (0, 15, 30, 45)]
    parametric_scores = [_parametric("moderate_irregular", rotation=rotation)["border_score"] for rotation in (0, 15, 30, 45)]

    assert radial_scores == [1, 0, 1, 1]
    assert parametric_scores == [3, 3, 3, 3]


def test_principal_axis_parametric_stabilizes_strong_serrated_irregularity():
    radial_scores = [_radial("strong_serrated", rotation=rotation)["border_score"] for rotation in (0, 15, 30, 45)]
    parametric_scores = [_parametric("strong_serrated", rotation=rotation)["border_score"] for rotation in (0, 15, 30, 45)]

    assert radial_scores == [8, 2, 2, 6]
    assert parametric_scores == [8, 8, 8, 8]
    assert np.ptp(parametric_scores) < np.ptp(radial_scores)
    assert np.std(parametric_scores) < np.std(radial_scores)


def test_scale_behavior_keeps_threshold_rule_unchanged():
    radial_localized = [_radial("localized_three_octants", size=size)["border_score"] for size in (256, 512, 1024)]
    parametric_localized = [_parametric("localized_three_octants", size=size)["border_score"] for size in (256, 512, 1024)]
    radial_strong = [_radial("strong_serrated", size=size)["border_score"] for size in (256, 512, 1024)]
    parametric_strong = [_parametric("strong_serrated", size=size)["border_score"] for size in (256, 512, 1024)]

    assert radial_localized == [3, 3, 3]
    assert parametric_localized == [2, 2, 2]
    assert radial_strong == [8, 8, 8]
    assert parametric_strong == [8, 8, 8]


def test_interpolating_spline_remains_validation_only_for_both_representations():
    mask = synthetic_cases(size=256)["localized_three_octants"]
    radial = evaluate_border_octants_radial(mask, spline_kind="interpolating")
    parametric = evaluate_border_octants_parametric(mask, spline_kind="interpolating")

    assert radial["border_score"] == 0
    assert parametric["border_score"] == 0


def test_smoothing_factor_is_reused_for_controlled_representation_comparison():
    mask = synthetic_cases(size=256)["localized_three_octants"]
    selected = evaluate_border_octants_parametric(mask, smoothing_factor=DEFAULT_SMOOTHING_FACTOR)
    high_smoothing = evaluate_border_octants_parametric(mask, smoothing_factor=100.0)

    assert selected["border_score"] == high_smoothing["border_score"] == 2


def test_deep_concavity_confirms_radial_angle_collapse_risk_without_parametric_adoption():
    mask = make_deep_concavity_mask()
    radial = evaluate_border_octants_radial(mask)
    parametric = evaluate_border_octants_parametric(mask)

    assert radial["border_score"] == 1
    assert parametric["border_score"] == 1
    assert radial["irregular_octants"] == parametric["irregular_octants"] == [0]
