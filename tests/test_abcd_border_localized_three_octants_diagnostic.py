import pytest

from src.features.border import (
    SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED,
    angles_for_xy,
    assign_octants_relative,
    extract_ordered_contour,
    lesion_centroid,
)
from src.features.border_validation import build_localized_three_octants_diagnostic, synthetic_cases
from src.features.geometry import principal_axis_analysis


@pytest.fixture(scope="module")
def localized_diagnostic():
    return build_localized_three_octants_diagnostic()


def test_every_contour_point_belongs_to_exactly_one_pca_sector():
    mask = synthetic_cases(size=256, rotation_degrees=0)["localized_three_octants"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]
    sectors = assign_octants_relative(angles_for_xy(contour, centroid), phi)

    assert sectors.shape[0] == contour.shape[0]
    assert sum((sectors == sector).sum() for sector in range(8)) == contour.shape[0]


def test_pca_sector_ids_are_in_clinical_range():
    mask = synthetic_cases(size=256, rotation_degrees=0)["localized_three_octants"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]
    sectors = assign_octants_relative(angles_for_xy(contour, centroid), phi)

    assert sectors.min() >= 0
    assert sectors.max() <= 7


def test_perturbation_mapping_is_rotation_equivariant_in_pca_frame(localized_diagnostic):
    rotations = localized_diagnostic["rotation_consistency"]
    baseline = rotations[0]["affected_sector_distribution"]

    for row in rotations[1:]:
        for perturbation in ("P1", "P2", "P3"):
            assert row["affected_sector_distribution"][perturbation]["sector_fractions"] == baseline[perturbation]["sector_fractions"]


def test_b_remains_stable_across_selected_rigid_rotations(localized_diagnostic):
    assert {row["B"] for row in localized_diagnostic["rotation_consistency"]} == {2}


def test_axial_ambiguity_only_permits_canonical_sector_permutation():
    mask = synthetic_cases(size=256, rotation_degrees=0)["localized_three_octants"]
    contour = extract_ordered_contour(mask)
    centroid = lesion_centroid(mask)
    phi = principal_axis_analysis(mask)["angle_radians"]

    base = assign_octants_relative(angles_for_xy(contour, centroid), phi)
    shifted = assign_octants_relative(angles_for_xy(contour, centroid), phi + 3.141592653589793)

    assert {int((s - b) % 8) for b, s in zip(base, shifted)} == {4}


def test_diagnostic_reports_no_implementation_defect(localized_diagnostic):
    assert localized_diagnostic["defect_status"] == "NO_IMPLEMENTATION_DEFECT_FOUND"
    assert localized_diagnostic["final_decision"] == "B2_THRESHOLD_EDGE_EFFECT"
    assert localized_diagnostic["native_principal_axis_aligned"]["sector_orientation"] == SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED
