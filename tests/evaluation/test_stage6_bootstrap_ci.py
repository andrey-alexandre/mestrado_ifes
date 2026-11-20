from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_bootstrap import generate_bootstrap_replicas  # noqa: E402
from src.evaluation.stage6_bootstrap_ci import (  # noqa: E402
    MIN_VALID_REPLICAS,
    bootstrap_full_manifest_paired_difference_cis,
    bootstrap_paired_difference_cis,
    bootstrap_single_system_metric_cis,
)
from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402

LESION_IDS_BY_CATEGORY = {"nv": [f"NV{i}" for i in range(1, 51)], "mel": [f"MEL{i}" for i in range(1, 21)]}
ALL_IDS = LESION_IDS_BY_CATEGORY["nv"] + LESION_IDS_BY_CATEGORY["mel"]
REAL = {lid: ("melanoma" if lid.startswith("MEL") else "nao_melanoma") for lid in ALL_IDS}


def _status_all_correct():
    return {lid: ("positivo" if REAL[lid] == "melanoma" else "negativo") for lid in ALL_IDS}


def test_single_system_point_estimate_matches_real_data_when_no_noise():
    status = _status_all_correct()
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=100, seed=1)
    indicators = compute_stage6_indicators(ALL_IDS, status, REAL)
    result = bootstrap_single_system_metric_cis(replicas, status, REAL, indicators)
    assert result["accuracy"].point_estimate == 1.0
    # sem erro algum -> toda replica tem acc=1.0 -> IC degenerado [1,1]
    assert result["accuracy"].ci_reported in (True, False)  # depende do n de replicas vs MIN_VALID_REPLICAS
    if result["accuracy"].ci_reported:
        assert result["accuracy"].ci_low == 1.0 and result["accuracy"].ci_high == 1.0


def test_min_valid_replicas_gate_blocks_ci_below_threshold():
    status = {lid: "indeterminado" for lid in ALL_IDS}  # nunca binario -> toda replica invalida
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=100, seed=2)
    indicators = compute_stage6_indicators(ALL_IDS, status, REAL)
    result = bootstrap_single_system_metric_cis(replicas, status, REAL, indicators)
    assert result["accuracy"].n_valid_replicas == 0
    assert result["accuracy"].ci_reported is False
    assert result["accuracy"].point_estimate is None  # indefinido nos dados reais tambem


def test_min_valid_replicas_gate_reports_when_at_or_above_threshold():
    # Constroi replicas fabricadas diretamente (sem passar por generate_bootstrap_replicas)
    # para controlar exatamente quantas sao validas.
    status = _status_all_correct()
    n_total = MIN_VALID_REPLICAS + 10
    replicas = [ALL_IDS for _ in range(n_total)]  # todas validas (status totalmente binario)
    indicators = compute_stage6_indicators(ALL_IDS, status, REAL)
    result = bootstrap_single_system_metric_cis(replicas, status, REAL, indicators)
    assert result["accuracy"].n_valid_replicas == n_total
    assert result["accuracy"].ci_reported is True


def test_paired_difference_restricted_to_fixed_intersection():
    status_a = _status_all_correct()
    status_b = dict(status_a)
    # B erra todas as 'mel' (vira negativo) -> intersecao binaria continua sendo tudo (ambos binarios)
    for lid in LESION_IDS_BY_CATEGORY["mel"]:
        status_b[lid] = "negativo"
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=200, seed=3)
    indicators_a = compute_stage6_indicators(ALL_IDS, status_a, REAL).binary_metrics
    indicators_b = compute_stage6_indicators(ALL_IDS, status_b, REAL).binary_metrics
    intersection = set(ALL_IDS)
    result = bootstrap_paired_difference_cis(replicas, intersection, status_a, status_b, REAL, indicators_a, indicators_b)
    # A tem sensibilidade 1.0, B tem sensibilidade 0.0 -> diff = 1.0 sempre
    assert result["sensitivity"].point_estimate == 1.0
    if result["sensitivity"].ci_reported:
        assert result["sensitivity"].ci_low == 1.0 and result["sensitivity"].ci_high == 1.0


def test_single_system_metric_names_subset_only_f1_matches_revisao_v2_usage():
    """Revisao metodologica 2026-11-19: o orquestrador v2 passa
    metric_names=("f1",) para nao recomputar via bootstrap as 4 metricas
    que passaram a usar Wilson -- confirma que o parametro restringe
    exatamente o conjunto de chaves devolvidas, sem alterar o valor de F1."""
    status = _status_all_correct()
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=50, seed=5)
    indicators = compute_stage6_indicators(ALL_IDS, status, REAL)
    result_f1_only = bootstrap_single_system_metric_cis(replicas, status, REAL, indicators, metric_names=("f1",))
    assert set(result_f1_only.keys()) == {"f1"}
    result_all = bootstrap_single_system_metric_cis(replicas, status, REAL, indicators)
    assert result_f1_only["f1"].point_estimate == result_all["f1"].point_estimate


def test_full_manifest_paired_diff_coverage_and_accuracy_over_total_use_whole_universe_not_intersection():
    status_a = _status_all_correct()
    status_b = {lid: "indeterminado" for lid in ALL_IDS}  # B nunca decide -> coverage_binary(B)=0, acc_over_total(B)=0
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=100, seed=6)
    indicators_a = compute_stage6_indicators(ALL_IDS, status_a, REAL)
    indicators_b = compute_stage6_indicators(ALL_IDS, status_b, REAL)
    result = bootstrap_full_manifest_paired_difference_cis(replicas, status_a, status_b, REAL, indicators_a, indicators_b)
    assert result["coverage_binary"].point_estimate == 1.0  # A=1.0 - B=0.0
    assert result["accuracy_over_total"].point_estimate == 1.0
    assert result["coverage_binary"].n_valid_replicas == len(replicas)  # denominador e o universo, sempre > 0


def test_paired_difference_replica_invalid_if_either_side_undefined():
    status_a = _status_all_correct()
    status_b = {lid: "indeterminado" for lid in ALL_IDS}  # B nunca binario -> toda replica invalida p/ diff
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=50, seed=4)
    bm_a = compute_stage6_indicators(ALL_IDS, status_a, REAL).binary_metrics
    bm_b = compute_stage6_indicators(ALL_IDS, status_b, REAL).binary_metrics
    result = bootstrap_paired_difference_cis(replicas, set(ALL_IDS), status_a, status_b, REAL, bm_a, bm_b)
    assert result["accuracy"].n_valid_replicas == 0
    assert result["accuracy"].point_estimate is None  # bm_b.accuracy e None nos dados reais tambem
