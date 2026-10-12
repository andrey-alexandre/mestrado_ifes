import pytest

from src.features.border_validation import (
    SMOOTHING_GRID,
    build_parametric_diagnostic_report,
)


@pytest.fixture(scope="module")
def diagnostic_report():
    return build_parametric_diagnostic_report()


def test_parametric_diagnostic_report_uses_required_smoothing_grid(diagnostic_report):
    report = diagnostic_report

    assert report["smoothing_grid"] == list(SMOOTHING_GRID)
    assert report["smoothing_grid"] == [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def test_raw_rmse_discriminates_irregularity_at_selected_smoothing(diagnostic_report):
    report = diagnostic_report
    rows = {
        row["case"]: row
        for row in report["table_1_raw_rmse_by_shape"]
        if row["smoothing"] == 1.0
    }

    assert rows["regular_ellipse"]["mean_rmse"] < rows["localized_three_octants"]["mean_rmse"]
    assert rows["localized_three_octants"]["mean_rmse"] < rows["moderate_irregular"]["mean_rmse"]
    assert rows["moderate_irregular"]["mean_rmse"] < rows["strong_serrated"]["mean_rmse"]


def test_localized_notch_contrast_increases_with_smoothing(diagnostic_report):
    report = diagnostic_report
    contrasts = {row["smoothing"]: row["contrast_ratio"] for row in report["table_2_localized_notches"]}

    assert contrasts[1e-6] < 2.0
    assert contrasts[1.0] > 20.0


def test_normalization_and_five_percent_rule_outcomes_are_reported(diagnostic_report):
    report = diagnostic_report
    norm = {row["case"]: row for row in report["table_5_normalization_comparison"]}
    five = {row["case"]: row for row in report["table_6_five_percent_rule"]}

    assert norm["strong_serrated"]["RMSE_per_arc_length"] > norm["regular_ellipse"]["RMSE_per_arc_length"]
    assert five["regular_ellipse"] == {"case": "regular_ellipse", "sample_count_B": 0, "arc_length_B": 0, "global_scale_B": 0}
    assert five["strong_serrated"]["sample_count_B"] == 0
    assert five["strong_serrated"]["arc_length_B"] > 0
    assert five["strong_serrated"]["global_scale_B"] == 0


def test_decision_category_records_local_arc_length_candidate_support(diagnostic_report):
    report = diagnostic_report

    assert report["decision_category"] == "LOCAL_ARC_LENGTH_NORMALIZATION_SUPPORTED"
    assert report["raw_rmse_discriminates"] is True
