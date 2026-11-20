"""Etapa 6 (continuação) -- comparação pareada de duas configurações
(sistema/condição) sobre a INTERSEÇÃO dos casos com decisão binária de
ambas (protocolo, seção 3). Nunca subtrai métricas calculadas sobre
subconjuntos diferentes.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.case_status import NEGATIVO, POSITIVO
from src.evaluation.stage6_metrics import BinaryMetrics, compute_stage6_indicators

MELANOMA = "melanoma"
NAO_MELANOMA = "nao_melanoma"


@dataclass
class PairwiseIntersection:
    first_label: str
    second_label: str
    intersection_lesion_ids: list[str]  # ordem estável (ordenada)
    n_intersection: int
    n_melanoma: int
    n_nao_melanoma: int
    first_metrics: BinaryMetrics
    second_metrics: BinaryMetrics
    point_diff_first_minus_second: dict[str, float | None]


def _binary_lesion_ids(status_by_lesion: dict[str, str], universe: list[str]) -> set[str]:
    return {lid for lid in universe if status_by_lesion.get(lid) in (POSITIVO, NEGATIVO)}


def compute_pairwise_comparison(
    first_label: str,
    second_label: str,
    status_first: dict[str, str],
    status_second: dict[str, str],
    real_by_lesion: dict[str, str],
    universe_lesion_ids: list[str],
) -> PairwiseIntersection:
    """Recalcula as 5 métricas de ``first``/``second`` SOBRE A MESMA
    interseção de lesões com decisão binária válida nos dois lados,
    dentro de ``universe_lesion_ids`` (o manifesto -- 3.000 ou 300 --
    nunca um universo maior). A diferença reportada é SEMPRE
    ``first - second`` (primeiro menos segundo, ordem fixada pelo
    chamador, nunca decidida aqui)."""
    binary_first = _binary_lesion_ids(status_first, universe_lesion_ids)
    binary_second = _binary_lesion_ids(status_second, universe_lesion_ids)
    intersection = sorted(binary_first & binary_second)

    n_melanoma = sum(1 for lid in intersection if real_by_lesion.get(lid) == MELANOMA)
    n_nao_melanoma = sum(1 for lid in intersection if real_by_lesion.get(lid) == NAO_MELANOMA)

    first_indicators = compute_stage6_indicators(intersection, status_first, real_by_lesion)
    second_indicators = compute_stage6_indicators(intersection, status_second, real_by_lesion)
    first_metrics, second_metrics = first_indicators.binary_metrics, second_indicators.binary_metrics

    diffs: dict[str, float | None] = {}
    for name in ("accuracy", "precision", "sensitivity", "specificity", "f1"):
        a, b = getattr(first_metrics, name), getattr(second_metrics, name)
        diffs[name] = (a - b) if (a is not None and b is not None) else None

    return PairwiseIntersection(
        first_label=first_label, second_label=second_label,
        intersection_lesion_ids=intersection, n_intersection=len(intersection),
        n_melanoma=n_melanoma, n_nao_melanoma=n_nao_melanoma,
        first_metrics=first_metrics, second_metrics=second_metrics,
        point_diff_first_minus_second=diffs,
    )
