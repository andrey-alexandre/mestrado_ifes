"""Etapa 6 (continuação) -- intervalos de confiança por bootstrap
(percentis 2,5/97,5), para uma métrica individual OU para uma diferença
pareada entre dois sistemas/condições, sobre as réplicas estratificadas já
geradas por ``src.evaluation.stage6_bootstrap.generate_bootstrap_replicas``
(reusado sem modificação -- este módulo não gera réplicas, só as consome).

Regra aprovada (protocolo, seção 4), implementada aqui e em nenhum outro
lugar:
  - cada réplica é reutilizada, sem modificação, para TODOS os sistemas/
    condições do mesmo conjunto (3.000 ou 300) -- o chamador passa a MESMA
    lista de réplicas para cada sistema, nunca gera réplicas por sistema;
  - para uma métrica individual, a seleção de casos binários é aplicada
    DENTRO de cada réplica (reusa
    ``stage6_metrics.compute_stage6_indicators`` sobre a réplica inteira,
    que já faz essa seleção);
  - para uma diferença pareada, a interseção binária do par (fixa, calculada
    uma vez sobre os dados reais -- nunca recalculada por réplica) é
    aplicada DENTRO de cada réplica antes de subtrair;
  - uma réplica só é válida para uma MÉTRICA/DIFERENÇA se o valor
    correspondente estiver definido (denominador > 0) nessa réplica --
    nunca substituído por 0;
  - IC só é reportado com pelo menos 4.750 réplicas válidas (de 5.000);
    abaixo disso, mantém a estimativa pontual (se definida) e marca
    ``ci_reported=False`` com a contagem real de réplicas válidas --
    nunca gera réplicas adicionais nem substitui indefinidos por zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluation.stage6_metrics import BinaryMetrics, Stage6Indicators, compute_stage6_indicators

MIN_VALID_REPLICAS = 4750
N_REPLICAS = 5000
BOOTSTRAP_SEED = 42

METRIC_NAMES = ("accuracy", "precision", "sensitivity", "specificity", "f1")


@dataclass
class CIResult:
    point_estimate: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_valid_replicas: int
    n_total_replicas: int
    ci_reported: bool


def _percentile_ci(values: list[float]) -> tuple[float, float]:
    """Percentis 2,5 e 97,5 por interpolação linear (mesma convenção de
    ``numpy.percentile`` com ``method="linear"``), sobre ``values`` já
    ordenados internamente. Implementado aqui sem depender de numpy para
    manter este módulo sem dependências externas novas."""
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        rank = p / 100 * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return ordered[lo] + (ordered[hi] - ordered[lo]) * frac

    return pct(2.5), pct(97.5)


def _ci_from_values(point_estimate: Optional[float], values: list[float], n_total: int) -> CIResult:
    n_valid = len(values)
    if n_valid < MIN_VALID_REPLICAS:
        return CIResult(point_estimate, None, None, n_valid, n_total, ci_reported=False)
    ci_low, ci_high = _percentile_ci(values)
    return CIResult(point_estimate, ci_low, ci_high, n_valid, n_total, ci_reported=True)


def bootstrap_single_system_metric_cis(
    replicas: Sequence[Sequence[str]],
    status_by_lesion: dict[str, str],
    real_by_lesion: dict[str, str],
    point_estimate_indicators: Stage6Indicators,
    metric_names: Sequence[str] = METRIC_NAMES,
) -> dict[str, CIResult]:
    """IC95% por bootstrap das métricas binárias de UM sistema/condição,
    sobre as réplicas dadas. ``point_estimate_indicators`` deve ter sido
    calculado sobre os dados reais (não-reamostrados) pelo chamador, com
    os MESMOS ``status_by_lesion``/``real_by_lesion`` -- reusado aqui só
    para extrair a estimativa pontual, nunca recalculado.

    ``metric_names`` (default: as 5 métricas, comportamento HISTÓRICO
    inalterado para qualquer chamador existente que não passe este
    argumento) -- a partir da revisão metodológica 2026-11-19 (decisão:
    Wilson para acurácia/precisão/sensibilidade/especificidade/cobertura/
    acertos-sobre-total, bootstrap só para F1), o chamador oficial da
    análise quantitativa final passa ``metric_names=("f1",)``, evitando
    recomputar via bootstrap as 4 métricas que agora usam Wilson."""
    per_metric_values: dict[str, list[float]] = {name: [] for name in metric_names}
    for replica_ids in replicas:
        indicators = compute_stage6_indicators(replica_ids, status_by_lesion, real_by_lesion)
        bm = indicators.binary_metrics
        for name in metric_names:
            value = getattr(bm, name)
            if value is not None:
                per_metric_values[name].append(value)

    point = point_estimate_indicators.binary_metrics
    return {
        name: _ci_from_values(getattr(point, name), per_metric_values[name], len(replicas))
        for name in metric_names
    }


def bootstrap_paired_difference_cis(
    replicas: Sequence[Sequence[str]],
    intersection_lesion_ids: set[str],
    status_a: dict[str, str],
    status_b: dict[str, str],
    real_by_lesion: dict[str, str],
    point_estimate_a: BinaryMetrics,
    point_estimate_b: BinaryMetrics,
) -> dict[str, CIResult]:
    """IC95% da diferença (A - B) das 5 métricas, restrita à interseção
    binária FIXA do par (``intersection_lesion_ids``, calculada uma vez
    sobre os dados reais pelo chamador -- ver
    ``src.evaluation.stage6_pairwise.compute_pairwise_intersection``).

    Dentro de cada réplica, filtra a lista reamostrada (preservando
    repetições) para só os ids que pertencem à interseção, antes de
    recomputar as métricas dos dois lados -- nunca recalcula a interseção
    por réplica. Uma réplica só é válida para uma métrica se AMBOS os
    lados tiverem essa métrica definida nela."""
    per_metric_diffs: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for replica_ids in replicas:
        filtered = [lid for lid in replica_ids if lid in intersection_lesion_ids]
        indicators_a = compute_stage6_indicators(filtered, status_a, real_by_lesion)
        indicators_b = compute_stage6_indicators(filtered, status_b, real_by_lesion)
        bm_a, bm_b = indicators_a.binary_metrics, indicators_b.binary_metrics
        for name in METRIC_NAMES:
            va, vb = getattr(bm_a, name), getattr(bm_b, name)
            if va is not None and vb is not None:
                per_metric_diffs[name].append(va - vb)

    point_diffs = {
        name: (
            (getattr(point_estimate_a, name) - getattr(point_estimate_b, name))
            if getattr(point_estimate_a, name) is not None and getattr(point_estimate_b, name) is not None
            else None
        )
        for name in METRIC_NAMES
    }
    return {
        name: _ci_from_values(point_diffs[name], per_metric_diffs[name], len(replicas))
        for name in METRIC_NAMES
    }


# Indicadores em `Stage6Indicators` (não em `BinaryMetrics`) cujo
# denominador é o manifesto INTEIRO de cada sistema (nunca a interseção
# do par) -- revisão metodológica 2026-11-19: "Compare cobertura binária
# e acertos sobre o total usando todas as lesões do manifesto
# correspondente" (protocolo original, seção 3) agora também recebe IC
# bootstrap da DIFERENÇA pareada, usando a MESMA réplica (mesmos ids
# reamostrados) para os dois lados -- nunca a interseção binária do par,
# que não se aplica a estes dois indicadores.
FULL_MANIFEST_DIFF_METRIC_NAMES = ("coverage_binary", "accuracy_over_total")


def bootstrap_full_manifest_paired_difference_cis(
    replicas: Sequence[Sequence[str]],
    status_a: dict[str, str],
    status_b: dict[str, str],
    real_by_lesion: dict[str, str],
    point_estimate_a: Stage6Indicators,
    point_estimate_b: Stage6Indicators,
    metric_names: Sequence[str] = FULL_MANIFEST_DIFF_METRIC_NAMES,
) -> dict[str, CIResult]:
    """IC95% da diferença (A - B) de indicadores calculados sobre o
    manifesto INTEIRO de cada réplica (``coverage_binary``/
    ``accuracy_over_total``), nunca sobre uma interseção -- distinto de
    ``bootstrap_paired_difference_cis`` (que é só para as 5 métricas
    binárias, sobre a interseção fixa do par). Mesma réplica reusada para
    os dois lados; réplica só válida para uma métrica se ambos os lados
    tiverem o valor definido nela (sempre o caso aqui, já que o
    denominador é o tamanho da própria réplica, nunca zero -- mas a
    checagem é mantida, sem caso especial, pela mesma regra geral de
    nunca substituir indefinido por zero)."""
    per_metric_diffs: dict[str, list[float]] = {name: [] for name in metric_names}
    for replica_ids in replicas:
        indicators_a = compute_stage6_indicators(replica_ids, status_a, real_by_lesion)
        indicators_b = compute_stage6_indicators(replica_ids, status_b, real_by_lesion)
        for name in metric_names:
            va, vb = getattr(indicators_a, name), getattr(indicators_b, name)
            if va is not None and vb is not None:
                per_metric_diffs[name].append(va - vb)

    point_diffs = {
        name: (
            (getattr(point_estimate_a, name) - getattr(point_estimate_b, name))
            if getattr(point_estimate_a, name) is not None and getattr(point_estimate_b, name) is not None
            else None
        )
        for name in metric_names
    }
    return {
        name: _ci_from_values(point_diffs[name], per_metric_diffs[name], len(replicas))
        for name in metric_names
    }
