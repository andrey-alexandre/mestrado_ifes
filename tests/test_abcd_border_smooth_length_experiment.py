import pytest

from src.features.border_validation import build_smooth_length_report


@pytest.fixture(scope="module")
def smooth_length_report():
    return build_smooth_length_report()


def test_smooth_length_denominator_preserves_regular_sanity(smooth_length_report):
    b_rows = {row["case"]: row for row in smooth_length_report["table_2_b_score"]}

    assert b_rows["regular_ellipse"]["B_real"] == 0
    assert b_rows["regular_ellipse"]["B_smooth"] == 0
    assert b_rows["smooth_lobulated"]["B_real"] == 0
    assert b_rows["smooth_lobulated"]["B_smooth"] == 0


def test_smooth_length_recovers_localized_and_strong_irregularity_signal(smooth_length_report):
    b_rows = {row["case"]: row for row in smooth_length_report["table_2_b_score"]}

    assert b_rows["localized_three_octants"]["B_real"] == 1
    assert b_rows["localized_three_octants"]["B_smooth"] == 3
    assert b_rows["localized_three_octants"]["smooth_octants"] == [1, 4, 6]
    assert b_rows["strong_serrated"]["B_real"] == 2
    assert b_rows["strong_serrated"]["B_smooth"] == 8


def test_real_denominator_absorbs_more_signal_as_irregularity_increases(smooth_length_report):
    rows = {row["case"]: row for row in smooth_length_report["table_1_denominator_comparison"]}

    assert rows["regular_ellipse"]["L_real_over_L_smooth"] < rows["localized_three_octants"]["L_real_over_L_smooth"]
    assert rows["localized_three_octants"]["L_real_over_L_smooth"] < rows["moderate_irregular"]["L_real_over_L_smooth"]
    assert rows["moderate_irregular"]["L_real_over_L_smooth"] < rows["strong_serrated"]["L_real_over_L_smooth"]


def test_smooth_length_materially_worsens_rotation_sensitivity(smooth_length_report):
    summary = {row["case"]: row for row in smooth_length_report["rotation_summary"]}

    assert summary["localized_three_octants"]["B_smooth_range"] > summary["localized_three_octants"]["B_real_range"]
    assert summary["moderate_irregular"]["B_smooth_range"] > summary["moderate_irregular"]["B_real_range"]
    assert summary["strong_serrated"]["B_smooth_range"] > summary["strong_serrated"]["B_real_range"]


def test_arc_length_normalizations_are_stable_under_contour_resampling(smooth_length_report):
    rows = smooth_length_report["table_5_sampling_density"]
    for case_name in ("regular_ellipse", "localized_three_octants", "strong_serrated"):
        case_rows = [row for row in rows if row["case"] == case_name]
        sample_values = [row["RMSE_per_n"] for row in case_rows]
        real_values = [row["RMSE_per_L_real"] for row in case_rows]
        smooth_values = [row["RMSE_per_L_smooth"] for row in case_rows]

        assert max(sample_values) / min(sample_values) > 3.0
        assert max(real_values) / min(real_values) < 1.15
        assert max(smooth_values) / min(smooth_values) < 1.15


def test_final_smooth_length_decision_is_recorded(smooth_length_report):
    assert smooth_length_report["sample_count_dependence_confirmed"] is True
    assert "n_samples" in smooth_length_report["smoothing_sampling_interaction"]
    assert smooth_length_report["decision"] == "NEITHER_NORMALIZATION_IS_SUFFICIENT"
