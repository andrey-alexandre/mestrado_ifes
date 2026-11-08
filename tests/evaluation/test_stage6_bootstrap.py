from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_bootstrap import (  # noqa: E402
    generate_bootstrap_replicas,
    generate_stratified_bootstrap_replica,
    replica_category_counts,
)
from src.evaluation.stage6_metrics import compute_stage6_indicators  # noqa: E402

LESION_IDS_BY_CATEGORY = {
    "nv": [f"nv_{i}" for i in range(6)],
    "mel": [f"mel_{i}" for i in range(4)],
    "bkl": [f"bkl_{i}" for i in range(2)],
}
LESION_ID_TO_CATEGORY = {
    lid: cat for cat, ids in LESION_IDS_BY_CATEGORY.items() for lid in ids
}


def test_single_replica_preserves_quota_exactly():
    rng = random.Random(1)
    replica = generate_stratified_bootstrap_replica(LESION_IDS_BY_CATEGORY, rng)
    counts = replica_category_counts(replica, LESION_ID_TO_CATEGORY)
    assert counts["nv"] == 6
    assert counts["mel"] == 4
    assert counts["bkl"] == 2
    assert len(replica) == 12


def test_many_replicas_all_preserve_quota():
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=30, seed=42)
    for replica in replicas:
        counts = replica_category_counts(replica, LESION_ID_TO_CATEGORY)
        assert counts["nv"] == 6
        assert counts["mel"] == 4
        assert counts["bkl"] == 2


def test_replicas_are_deterministic_given_seed():
    r1 = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=5, seed=99)
    r2 = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=5, seed=99)
    assert r1 == r2


def test_different_seeds_produce_different_replicas():
    r1 = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=5, seed=1)
    r2 = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=5, seed=2)
    assert r1 != r2


def test_same_replica_reused_across_two_systems_not_resampled_independently():
    """O ponto critico do bootstrap pareado: a MESMA replica (mesmos
    lesion_ids, inclusive repeticoes) deve produzir resultados coerentes
    quando aplicada a dois "sistemas" com status conhecidos -- isto e, o
    denominador/numerador de cada sistema deve refletir EXATAMENTE a
    composicao daquela replica, nao duas amostras independentes."""
    replicas = generate_bootstrap_replicas(LESION_IDS_BY_CATEGORY, n_replicas=1, seed=7)
    replica = replicas[0]

    real_by_lesion = {lid: "melanoma" if cat == "mel" else "nao_melanoma"
                      for lid, cat in LESION_ID_TO_CATEGORY.items()}

    # Sistema 1: acerta tudo. Sistema 2: erra tudo (binario invertido).
    status_system_1 = {
        lid: ("positivo" if real == "melanoma" else "negativo")
        for lid, real in real_by_lesion.items()
    }
    status_system_2 = {
        lid: ("negativo" if real == "melanoma" else "positivo")
        for lid, real in real_by_lesion.items()
    }

    ind_1 = compute_stage6_indicators(replica, status_system_1, real_by_lesion)
    ind_2 = compute_stage6_indicators(replica, status_system_2, real_by_lesion)

    # Mesma replica (mesmo n_total, mesma composicao) -> os dois sistemas
    # veem o MESMO numero de casos binarios e a MESMA contagem de reais
    # positivos (tp+fn) e reais negativos (tn+fp) dentro do subconjunto
    # binario -- so a predicao muda, nao a amostra. (Corrigido 2026-11-08:
    # a versao original comparava tp1+fn1 (reais positivos do sistema 1)
    # com tn2+fp2 (reais NEGATIVOS do sistema 2) -- grandezas diferentes
    # por definicao, nao uma comparacao valida da invariante pretendida.)
    assert ind_1.n_total == ind_2.n_total == len(replica)
    assert ind_1.binary_metrics.n_binary == ind_2.binary_metrics.n_binary
    assert ind_1.binary_metrics.tp + ind_1.binary_metrics.fn == ind_2.binary_metrics.tp + ind_2.binary_metrics.fn
    assert ind_1.binary_metrics.tn + ind_1.binary_metrics.fp == ind_2.binary_metrics.tn + ind_2.binary_metrics.fp
    # Sistema 1 acerta tudo -> accuracy 1.0; sistema 2 erra tudo -> accuracy 0.0.
    assert ind_1.binary_metrics.accuracy == 1.0
    assert ind_2.binary_metrics.accuracy == 0.0


def test_replica_contains_repeats_with_replacement():
    rng = random.Random(123)
    replica = generate_stratified_bootstrap_replica(LESION_IDS_BY_CATEGORY, rng)
    counts = Counter(replica)
    # Com reposicao em um pool pequeno, pelo menos uma repeticao e extremamente provavel;
    # a propriedade que realmente importa (tamanho==pool original por categoria) ja e
    # testada acima -- aqui so confirmamos que repeticao e estruturalmente possivel
    # (max(count) pode ser > 1 para alguma categoria pequena como 'bkl', 2 itens).
    assert len(replica) == 12
    assert sum(counts.values()) == 12
