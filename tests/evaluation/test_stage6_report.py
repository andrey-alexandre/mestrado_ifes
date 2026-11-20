from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_bootstrap_ci import CIResult  # noqa: E402
from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402
from src.evaluation.stage6_pairwise import compute_pairwise_comparison  # noqa: E402
from src.evaluation.stage6_report import (  # noqa: E402
    build_main_table_row,
    build_main_table_row_wilson,
    build_pairwise_row,
    main_table_markdown_row,
    main_table_markdown_row_wilson,
)
from src.evaluation.stage6_wilson import wilson_results_for_indicators  # noqa: E402

LESION_IDS = [f"L{i}" for i in range(1, 11)]
REAL = {
    "L1": "melanoma", "L2": "melanoma", "L3": "melanoma",
    "L4": "nao_melanoma", "L5": "nao_melanoma", "L6": "nao_melanoma",
    "L7": "nao_melanoma", "L8": "nao_melanoma", "L9": "nao_melanoma", "L10": "nao_melanoma",
}
STATUS = {
    "L1": "positivo", "L2": "positivo", "L3": "negativo",
    "L4": "negativo", "L5": "negativo", "L6": "negativo",
    "L7": "indeterminado", "L8": "ausente", "L9": "nao_reconhecido",
}


def _fake_ci(point=0.5):
    return CIResult(point_estimate=point, ci_low=0.1, ci_high=0.9, n_valid_replicas=5000, n_total_replicas=5000, ci_reported=True)


def test_v1_build_main_table_row_untouched_by_wilson_addition():
    """v1 (`build_main_table_row`) deve continuar funcionando exatamente
    como antes -- a adicao de Wilson nao pode alterar seu comportamento."""
    indicators = compute_stage6_indicators(LESION_IDS, STATUS, REAL)
    ci_by_metric = {name: _fake_ci() for name in ("accuracy", "precision", "sensitivity", "specificity", "f1")}
    row = build_main_table_row("Sistema X", 10, indicators, ci_by_metric, is_provisional=False)
    assert row["accuracy_point"] == 0.5
    md = main_table_markdown_row(row, ci_by_metric)
    assert "F1 [IC95%]" in md


def test_wilson_main_table_row_uses_wilson_for_four_metrics_and_bootstrap_for_f1():
    indicators = compute_stage6_indicators(LESION_IDS, STATUS, REAL)
    wilson_by_metric = wilson_results_for_indicators(indicators)
    f1_ci = _fake_ci(point=indicators.binary_metrics.f1)
    row = build_main_table_row_wilson("Sistema X", 10, indicators, wilson_by_metric, f1_ci, is_provisional=False)

    assert row["accuracy_method"] == "wilson_score_no_continuity_correction"
    assert row["precision_method"] == "wilson_score_no_continuity_correction"
    assert row["sensitivity_method"] == "wilson_score_no_continuity_correction"
    assert row["specificity_method"] == "wilson_score_no_continuity_correction"
    assert row["binarios_validos_pct_method"] == "wilson_score_no_continuity_correction"
    assert row["acertos_pct_sobre_total_method"] == "wilson_score_no_continuity_correction"
    assert row["f1_method"] == "bootstrap"
    # ponto nao mudou em relacao ao ja calculado por compute_stage6_indicators
    assert row["accuracy_point"] == indicators.binary_metrics.accuracy
    assert row["f1_point"] == f1_ci.point_estimate

    md = main_table_markdown_row_wilson(row, f1_ci)
    assert "(Wilson)" in md["Acurácia [IC95%]"]
    assert "F1 [IC95% bootstrap]" in md


def test_build_pairwise_row_backward_compatible_with_5_metric_dict():
    status_a = {lid: "positivo" for lid in LESION_IDS}
    status_b = {lid: "negativo" for lid in LESION_IDS}
    pairwise = compute_pairwise_comparison("A", "B", status_a, status_b, REAL, LESION_IDS)
    diff_ci = {name: _fake_ci() for name in ("accuracy", "precision", "sensitivity", "specificity", "f1")}
    row = build_pairwise_row("Analise X", pairwise, diff_ci)
    assert set(k for k in row if k.startswith("diff_") and k.endswith("_pp")) == {
        "diff_accuracy_pp", "diff_precision_pp", "diff_sensitivity_pp", "diff_specificity_pp", "diff_f1_pp",
    }


def test_build_pairwise_row_accepts_extended_7_metric_dict():
    status_a = {lid: "positivo" for lid in LESION_IDS}
    status_b = {lid: "negativo" for lid in LESION_IDS}
    pairwise = compute_pairwise_comparison("A", "B", status_a, status_b, REAL, LESION_IDS)
    diff_ci = {name: _fake_ci() for name in ("accuracy", "precision", "sensitivity", "specificity", "f1", "coverage_binary", "accuracy_over_total")}
    row = build_pairwise_row("Analise X", pairwise, diff_ci)
    assert "diff_coverage_binary_pp" in row
    assert "diff_accuracy_over_total_pp" in row
    assert row["diff_coverage_binary_method"] == "bootstrap"
