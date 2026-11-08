from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402

LESION_IDS = [f"L{i}" for i in range(1, 11)]  # 10 lesoes
REAL_BY_LESION = {
    "L1": "melanoma", "L2": "melanoma", "L3": "melanoma",
    "L4": "nao_melanoma", "L5": "nao_melanoma", "L6": "nao_melanoma",
    "L7": "nao_melanoma", "L8": "nao_melanoma", "L9": "nao_melanoma", "L10": "nao_melanoma",
}


def test_indeterminado_excluded_from_binary_metrics_but_counts_as_valid_output():
    status = {
        "L1": "positivo", "L2": "positivo", "L3": "negativo",  # 1 FN (L3 real melanoma, pred negativo)
        "L4": "negativo", "L5": "negativo",
        "L6": "indeterminado", "L7": "indeterminado",
        "L8": "ausente", "L9": "nao_reconhecido",
        # L10 missing entirely from status dict -> treated as ausente
    }
    ind = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    assert ind.n_total == 10
    assert ind.n_positivo == 2
    assert ind.n_negativo == 3
    assert ind.n_indeterminado == 2
    assert ind.n_ausente == 2  # L8 explicito + L10 implicito (ausente do status dict)
    assert ind.n_nao_reconhecido == 1

    bm = ind.binary_metrics
    assert bm.n_binary == 5  # L1,L2,L3,L4,L5 -- indeterminado/ausente/nao_reconhecido fora
    assert bm.tp == 2  # L1, L2
    assert bm.fn == 1  # L3
    assert bm.tn == 2  # L4, L5
    assert bm.fp == 0

    # Cobertura binaria: (2+3)/10 -- indeterminado NAO conta no numerador.
    assert ind.coverage_binary == 0.5
    # Taxa de saida valida: (2+3+2)/10 -- indeterminado CONTA aqui.
    assert ind.valid_output_rate == 0.7
    # Acertos sobre o total: (tp+tn)/10 -- indeterminado/ausente/nao_reconhecido contam como erro.
    assert ind.accuracy_over_total == 0.4


def test_zero_denominator_metrics_are_none_not_zero():
    # Nenhum caso binario em lugar nenhum.
    status = {lid: "indeterminado" for lid in LESION_IDS}
    ind = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    bm = ind.binary_metrics
    assert bm.n_binary == 0
    assert bm.accuracy is None
    assert bm.precision is None
    assert bm.sensitivity is None
    assert bm.specificity is None
    assert bm.f1 is None
    # Cobertura binaria tambem fica 0.0 (denominador=n_total=10 existe, numerador=0) -- nao None.
    assert ind.coverage_binary == 0.0


def test_sensitivity_none_when_no_real_positives_in_binary_subset():
    # So casos reais nao-melanoma entram no binario -- nenhum TP/FN possivel.
    status = {lid: "negativo" for lid in LESION_IDS if REAL_BY_LESION[lid] == "nao_melanoma"}
    for lid in LESION_IDS:
        status.setdefault(lid, "ausente")
    ind = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    assert ind.binary_metrics.tp == 0
    assert ind.binary_metrics.fn == 0
    assert ind.binary_metrics.sensitivity is None  # tp+fn == 0 -> indefinida, nao 0.0
    assert ind.binary_metrics.specificity == 1.0  # tn=7 (L4..L10), fp=0


def test_two_systems_with_different_missing_cases_do_not_contaminate():
    status_system_a = {lid: "negativo" for lid in LESION_IDS if REAL_BY_LESION[lid] == "nao_melanoma"}
    status_system_a.update({"L1": "positivo", "L2": "positivo", "L3": "positivo"})
    # sistema A perde L4 (omitido do dict -> ausente automatico)
    del status_system_a["L4"]

    status_system_b = {lid: "negativo" for lid in LESION_IDS if REAL_BY_LESION[lid] == "nao_melanoma"}
    status_system_b.update({"L1": "positivo", "L2": "positivo", "L3": "positivo"})
    # sistema B perde L9, nao L4
    del status_system_b["L9"]

    ind_a = compute_stage6_indicators(LESION_IDS, status_system_a, REAL_BY_LESION)
    ind_b = compute_stage6_indicators(LESION_IDS, status_system_b, REAL_BY_LESION)

    assert ind_a.n_ausente == 1
    assert ind_b.n_ausente == 1
    # Cada sistema tem 9 casos binarios (10 - 1 ausente cada), mas sao lesoes DIFERENTES.
    assert ind_a.binary_metrics.n_binary == 9
    assert ind_b.binary_metrics.n_binary == 9
    # tn difere: A tem L4..L10 exceto L4 ausente -> 6 negativos reais presentes (L5-L10);
    # B tem L4..L10 exceto L9 ausente -> 6 negativos reais presentes (L4-L8,L10).
    assert ind_a.binary_metrics.tn == 6
    assert ind_b.binary_metrics.tn == 6
    # Mas os indefinidos (ausentes) sao lesoes diferentes -- confirma sem contaminacao.
    assert ind_a.n_total == ind_b.n_total == 10


def test_indeterminado_breakdown_by_real_class():
    status = {lid: "negativo" for lid in LESION_IDS if REAL_BY_LESION[lid] == "nao_melanoma"}
    status.update({"L1": "indeterminado", "L2": "positivo", "L3": "indeterminado"})
    ind = compute_stage6_indicators(LESION_IDS, status, REAL_BY_LESION)
    breakdown = ind.indeterminado_breakdown.by_real_class
    assert breakdown["melanoma"]["count"] == 2  # L1, L3
    assert breakdown["melanoma"]["total_in_class"] == 3  # L1, L2, L3
    assert breakdown["nao_melanoma"]["count"] == 0


def test_repeated_lesion_ids_count_independently_bootstrap_style():
    # Simula uma replica bootstrap: L1 sorteado duas vezes, L4 tres vezes.
    replica = ["L1", "L1", "L2", "L3", "L4", "L4", "L4"]
    status = {"L1": "positivo", "L2": "positivo", "L3": "negativo", "L4": "negativo"}
    ind = compute_stage6_indicators(replica, status, REAL_BY_LESION)
    assert ind.n_total == 7
    assert ind.n_positivo == 3  # L1 (x2) + L2
    assert ind.n_negativo == 4  # L3 + L4 (x3)
    assert ind.binary_metrics.tp == 3  # L1 x2 (real melanoma, pred positivo) + L2
    assert ind.binary_metrics.fn == 1  # L3 (real melanoma, pred negativo)
    assert ind.binary_metrics.tn == 3  # L4 x3 (real nao_melanoma, pred negativo)
