import math

import numpy as np
import pytest

from src.features.asymmetry import principal_axis as asymmetry_principal_axis
from src.features.border import (
    SECTOR_ORIENTATION_IMAGE_FIXED,
    SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    angles_for_xy,
    assign_octants,
    assign_octants_relative,
    evaluate_border_octants_parametric,
    evaluate_ordered_contour_parametric,
    extract_ordered_contour,
    lesion_centroid,
)
from src.features.border_validation import (
    build_principal_axis_orientation_report,
    diagnostic_control_cases,
    synthetic_cases,
)
from src.features.geometry import axial_angle_difference, principal_axis_analysis


@pytest.fixture(scope="module")
def orientation_report():
    return build_principal_axis_orientation_report()


def test_abcd_a_principal_axis_reuses_shared_second_moment_definition():
    mask = synthetic_cases(size=256)["regular_ellipse"]

    asymmetry_angle, asymmetry_centroid = asymmetry_principal_axis(mask)
    shared = principal_axis_analysis(mask)

    assert axial_angle_difference(asymmetry_angle, shared["angle_radians"]) < 1e-12
    assert asymmetry_centroid == shared["centroid_xy"]
    assert shared["mu20"] > shared["mu02"]


def test_principal_axis_orientation_tracks_elongated_lesion_rotation_modulo_pi():
    phi0 = principal_axis_analysis(diagnostic_control_cases(rotation_degrees=0)["clearly_elongated_ellipse"])["angle_radians"]

    for rotation in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45):
        phi = principal_axis_analysis(
            diagnostic_control_cases(rotation_degrees=rotation)["clearly_elongated_ellipse"]
        )["angle_radians"]

        assert math.degrees(axial_angle_difference(phi, phi0 + math.radians(rotation))) < 0.75


def test_sector_assignment_uses_theta_minus_phi():
    angles = np.array([math.radians(40), math.radians(50), math.radians(95)])
    phi = math.radians(10)

    assert assign_octants_relative(angles, phi).tolist() == assign_octants(angles - phi).tolist()


def test_phi_plus_pi_preserves_total_b_and_shifts_sector_ids_by_four(orientation_report):
    ambiguity = orientation_report["ambiguity_diagnostics"]

    assert ambiguity["phi_plus_180_preserves_B"] is True
    assert ambiguity["phi_plus_180_sector_shift"] == 4


def test_phi_plus_pi_over_two_preserves_total_b_with_axis_exchange_shift(orientation_report):
    ambiguity = orientation_report["ambiguity_diagnostics"]

    assert ambiguity["phi_plus_90_preserves_B"] is True
    assert ambiguity["phi_plus_90_sector_shift"] in (2, 6)


def test_principal_axis_alignment_does_not_alter_contour_coordinates_used_by_spline():
    mask = synthetic_cases(size=256, rotation_degrees=15)["localized_three_octants"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]

    image_fixed = evaluate_ordered_contour_parametric(
        contour,
        centroid,
        include_points=True,
        sector_orientation=SECTOR_ORIENTATION_IMAGE_FIXED,
        sector_angle_offset=0.0,
    )
    principal = evaluate_ordered_contour_parametric(
        contour,
        centroid,
        include_points=True,
        sector_orientation=SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
        sector_angle_offset=phi,
    )

    fixed_points = np.vstack([o["contour_xy"] for o in image_fixed["octants"] if o["contour_xy"]])
    principal_points = np.vstack([o["contour_xy"] for o in principal["octants"] if o["contour_xy"]])
    assert {tuple(point) for point in np.round(fixed_points, 6)} == {tuple(point) for point in np.round(principal_points, 6)}


def test_principal_axis_alignment_changes_only_sector_membership():
    mask = synthetic_cases(size=256, rotation_degrees=15)["localized_three_octants"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]
    theta = angles_for_xy(contour, centroid)

    assert not np.array_equal(assign_octants(theta), assign_octants_relative(theta, phi))
    assert contour.shape == extract_ordered_contour(mask).shape


def test_regular_ellipse_remains_b_zero_under_principal_axis_alignment():
    result = evaluate_border_octants_parametric(
        synthetic_cases(size=256)["regular_ellipse"],
        sector_orientation=SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    )

    octants = result["octants"]
    b_smooth = sum(
        1 for octant in octants
        if octant["smooth_arc_length"] and octant["rmse"] / octant["smooth_arc_length"] > 0.05
    )
    assert b_smooth == 0


def test_regular_circle_remains_b_zero_even_with_underdetermined_phi(orientation_report):
    rows = [
        row for row in orientation_report["table_5_axis_stability_controls"]
        if row["shape"] == "circle"
    ]

    assert all(row["orientation_status"] == "PRINCIPAL_AXIS_UNDERDETERMINED" for row in rows)
    assert {row["B_principal_axis"] for row in rows} == {0}


def test_localized_irregularity_retains_raw_rmse_discrimination_under_alignment():
    result = evaluate_border_octants_parametric(
        synthetic_cases(size=256)["localized_three_octants"],
        sector_orientation=SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    )
    affected = [
        octant["rmse"]
        for octant in result["octants"]
        if octant["smooth_arc_length"] and octant["rmse"] / octant["smooth_arc_length"] > 0.05
    ]
    unaffected = [
        octant["rmse"]
        for octant in result["octants"]
        if octant["smooth_arc_length"] and octant["rmse"] / octant["smooth_arc_length"] <= 0.05
    ]

    assert np.mean(affected) > np.median(unaffected) * 10


def test_principal_axis_scale_behavior_remains_acceptable(orientation_report):
    rows = orientation_report["table_6_scale"]
    by_shape = {shape: [row for row in rows if row["shape"] == shape] for shape in ("localized_three_octants", "strong_serrated")}

    assert {row["B_principal_axis"] for row in by_shape["localized_three_octants"]} == {2}
    assert {row["B_principal_axis"] for row in by_shape["strong_serrated"]} == {8}
    assert max(row["mean_E_smooth"] for row in by_shape["strong_serrated"]) / min(row["mean_E_smooth"] for row in by_shape["strong_serrated"]) < 1.03


def test_principal_axis_b_remains_integer_in_clinical_range(orientation_report):
    for row in orientation_report["table_1_rotation"]:
        assert isinstance(row["B_principal_axis"], int)
        assert 0 <= row["B_principal_axis"] <= 8
