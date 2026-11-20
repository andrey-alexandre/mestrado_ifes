from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402
from src.evaluation.stage6_wilson import (  # noqa: E402
    METHOD_WILSON,
    wilson_results_for_indicators,
    wilson_score_interval,
)

LESION_IDS = [f"L{i}" for i in range(1, 11)]
REAL_BY_LESION = {
    "L1": "melanoma", "L2": "melanoma", "L3": "melanoma",
    "L4": "nao_melanoma", "L5": "nao_melanoma", "L6": "nao_melanoma",
    "L7": "nao_melanoma", "L8": "nao_melanoma", "L9": "nao_melanoma", "L10": "nao_melanoma",
}


def test_zero_denominator_is_none_not_zero():
    result = wilson_score_interval(0, 0)
    assert result.point_estimate is None
    assert result.ci_low is None
    assert result.ci_high is None
    assert result.numerator == 0 and result.denominator == 0


def test_zero_successes_computed_normally_and_stays_within_unit_interval():
    result = wilson_score_interval(0, 10)
    assert result.point_estimate == 0.0
    assert result.ci_low == 0.0  # formula naturalmente fica >= 0
    assert 0.0 < result.ci_high < 1.0


def test_all_successes_computed_normally_and_stays_within_unit_interval():
    result = wilson_score_interval(10, 10)
    assert result.point_estimate == 1.0
    assert result.ci_high == 1.0
    assert 0.0 < result.ci_low < 1.0


def test_point_estimate_is_observed_proportion_not_wilson_center():
    # Wilson center para phat=0.3 (n=10) e DIFERENTE de 0.3 -- confirma
    # que a estimativa pontual devolvida e a proporcao observada, nao o
    # centro deslocado do intervalo.
    result = wilson_score_interval(3, 10)
    assert result.point_estimate == 0.3
    midpoint = (result.ci_low + result.ci_high) / 2
    assert abs(midpoint - 0.3) > 1e-6  # centro de Wilson != phat


def test_reference_value_5_of_10_matches_known_textbook_interval():
    """Valor de referência bem conhecido (Wilson 1927 / Newcombe 1998):
    IC95% de Wilson para 5 sucessos em 10 tentativas ~= (0.2366, 0.7634)."""
    result = wilson_score_interval(5, 10)
    assert result.point_estimate == 0.5
    assert abs(result.ci_low - 0.2366) < 0.001
    assert abs(result.ci_high - 0.7634) < 0.001


def test_reference_value_asymmetric_case():
    """n=100, x=10 (phat=0.10): IC95% de Wilson bem documentado ~=
    (0.0554, 0.1737) (ver, ex., calculadoras/tabelas de referencia de
    Wilson score interval amplamente publicadas)."""
    result = wilson_score_interval(10, 100)
    assert result.point_estimate == 0.10
    assert abs(result.ci_low - 0.0554) < 0.003
    assert abs(result.ci_high - 0.1737) < 0.003


def test_method_label_is_wilson_not_wald_or_other():
    result = wilson_score_interval(5, 10)
    assert result.method == METHOD_WILSON
    assert "wald" not in result.method.lower()
    assert "agresti" not in result.method.lower()
    assert "clopper" not in result.method.lower()


def test_wilson_results_for_indicators_point_estimates_match_pre_existing_values():
    """Requisito de verificacao: as estimativas pontuais e os
    numeradores/denominadores usados por Wilson devem ser EXATAMENTE os
    mesmos ja calculados por `compute_stage6_indicators` -- nenhuma
    mudanca na contagem, so no metodo do intervalo."""
    status = {
        "L1": "positivo", "L2": "positivo", "L3": "negativo",
        "L4": "negativo", "L5": "negativo", "L6": "negativo",
        "L7": "indeterminado", "L8": "ausente", "L9": "nao_reconhecido",
    }
    indicators = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    bm = indicators.binary_metrics
    wilson = wilson_results_for_indicators(indicators)

    assert wilson["accuracy"].point_estimate == bm.accuracy
    assert wilson["precision"].point_estimate == bm.precision
    assert wilson["sensitivity"].point_estimate == bm.sensitivity
    assert wilson["specificity"].point_estimate == bm.specificity
    assert wilson["coverage_binary"].point_estimate == indicators.coverage_binary
    assert wilson["accuracy_over_total"].point_estimate == indicators.accuracy_over_total

    assert wilson["accuracy"].numerator == bm.tp + bm.tn
    assert wilson["accuracy"].denominator == bm.n_binary
    assert wilson["precision"].denominator == bm.tp + bm.fp
    assert wilson["sensitivity"].denominator == bm.tp + bm.fn
    assert wilson["specificity"].denominator == bm.tn + bm.fp
    assert wilson["coverage_binary"].denominator == indicators.n_total
    assert wilson["accuracy_over_total"].denominator == indicators.n_total


def test_wilson_results_for_indicators_undefined_when_denominator_zero():
    status = {lid: "indeterminado" for lid in LESION_IDS}  # nenhum binario -> todos denominadores de confusao zero
    indicators = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    wilson = wilson_results_for_indicators(indicators)
    assert wilson["accuracy"].point_estimate is None
    assert wilson["precision"].point_estimate is None
    assert wilson["sensitivity"].point_estimate is None
    assert wilson["specificity"].point_estimate is None
    # cobertura binaria e acertos sobre o total usam n_total (sempre > 0 aqui), entao ficam definidos (0.0)
    assert wilson["coverage_binary"].point_estimate == 0.0
    assert wilson["accuracy_over_total"].point_estimate == 0.0


def test_f1_is_not_part_of_wilson_table():
    status = {lid: "positivo" for lid in LESION_IDS}
    indicators = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    wilson = wilson_results_for_indicators(indicators)
    assert "f1" not in wilson
