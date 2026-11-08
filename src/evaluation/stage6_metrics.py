"""Stage 6 — the four denominator-distinct indicators computed per
system/condition, over the frozen 3000-lesion HAM10000 sample (or, for
bootstrap replicas, over a resampled MULTISET of that sample's lesion_ids,
with repetition -- see ``src.evaluation.stage6_bootstrap``). Pure functions,
no I/O -- callers supply the lesion-id universe (a list, possibly with
repeats for a bootstrap replica), the per-lesion case status
(``src.evaluation.case_status.classify_case_status``) and the per-lesion
ground truth (from the frozen manifest).

Researcher decision (2026-11-08), implemented exactly as specified:

  | Indicador                        | Tratamento de "indeterminado"          |
  |-----------------------------------|-----------------------------------------|
  | 5 métricas binárias (acc/prec/   | EXCLUÍDO da matriz de confusão.        |
  | sens/espec/F1)                   |                                          |
  | Cobertura diagnóstica binária    | FORA do numerador (positivo+negativo). |
  | Taxa de saída válida             | DENTRO do numerador (é saída válida).  |
  | Acertos sobre o total            | Conta como erro (como ausente/malform).|

Any metric whose denominator is zero (e.g. zero binary decisions, or zero
real positives for sensitivity) is reported as ``None`` ("indefinida") --
NEVER substituted by 0.0.

"Diferenças entre métricas calculadas sobre subconjuntos válidos distintos
(ex. binário-apenas vs. acertos-sobre-total) continuam pendentes de decisão
metodológica do pesquisador" -- this module computes and reports BOTH views
side by side (``accuracy`` inside :class:`BinaryMetrics` vs.
``accuracy_over_total`` on :class:`Stage6Indicators`), without picking one
as "the" final approach.

A lesion_id present in ``expected_lesion_ids`` but absent from
``status_by_lesion`` is treated as ``"ausente"`` automatically -- never
silently dropped from any denominator. ``expected_lesion_ids`` MAY contain
repeated ids (a bootstrap replica resamples with replacement); every
occurrence is counted independently, so a lesion drawn twice contributes
twice to every count/metric -- this is standard bootstrap behaviour, not a
bug to deduplicate away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.evaluation.case_status import (
    AUSENTE,
    INDETERMINADO,
    NAO_RECONHECIDO,
    NEGATIVO,
    POSITIVO,
)

MELANOMA = "melanoma"
NAO_MELANOMA = "nao_melanoma"


@dataclass
class BinaryMetrics:
    n_binary: int
    accuracy: float | None
    precision: float | None
    sensitivity: float | None
    specificity: float | None
    f1: float | None
    tp: int
    tn: int
    fp: int
    fn: int


@dataclass
class IndeterminadoBreakdown:
    """Count/rate of "indeterminado" cases split by real (ground-truth) class."""
    by_real_class: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class Stage6Indicators:
    n_total: int
    n_positivo: int
    n_negativo: int
    n_indeterminado: int
    n_ausente: int
    n_nao_reconhecido: int
    binary_metrics: BinaryMetrics
    coverage_binary: float | None
    valid_output_rate: float | None
    accuracy_over_total: float | None
    indeterminado_breakdown: IndeterminadoBreakdown


def _safe_div(numerator: float, denominator: float) -> float | None:
    return (numerator / denominator) if denominator > 0 else None


def compute_stage6_indicators(
    expected_lesion_ids: Sequence[str],
    status_by_lesion: dict[str, str],
    real_by_lesion: dict[str, str],
) -> Stage6Indicators:
    """Full indicator set for one system/condition over
    ``expected_lesion_ids`` (the full manifest for a real run, or one
    bootstrap replica's resampled, possibly-repeating list).

    Args:
        expected_lesion_ids: the lesion universe for every denominator here
            (coverage, valid-output rate, "acertos sobre o total"). May
            contain repeated ids (bootstrap replica) -- each occurrence is
            counted independently.
        status_by_lesion: ``lesion_id -> case status``, only for lesions
            this system/condition actually produced an event for (from
            pairing -- ``src.evaluation.stage6_pairing``). Any id in
            ``expected_lesion_ids`` missing here is treated as
            ``"ausente"``.
        real_by_lesion: ``lesion_id -> "melanoma"/"nao_melanoma"`` (the
            manifest's ``binary_target``).
    """
    n_total = len(expected_lesion_ids)
    n_positivo = n_negativo = n_indeterminado = n_ausente = n_nao_reconhecido = 0
    tp = tn = fp = fn = 0
    totals_by_real_class: dict[str, int] = {}
    indeterminado_by_real_class: dict[str, int] = {}

    for lesion_id in expected_lesion_ids:
        status = status_by_lesion.get(lesion_id, AUSENTE)
        real = real_by_lesion.get(lesion_id)

        if status == POSITIVO:
            n_positivo += 1
        elif status == NEGATIVO:
            n_negativo += 1
        elif status == INDETERMINADO:
            n_indeterminado += 1
        elif status == AUSENTE:
            n_ausente += 1
        elif status == NAO_RECONHECIDO:
            n_nao_reconhecido += 1

        if status in (POSITIVO, NEGATIVO):
            real_positive = real == MELANOMA
            pred_positive = status == POSITIVO
            if real_positive and pred_positive:
                tp += 1
            elif (not real_positive) and (not pred_positive):
                tn += 1
            elif (not real_positive) and pred_positive:
                fp += 1
            elif real_positive and (not pred_positive):
                fn += 1

        if real is not None:
            totals_by_real_class[real] = totals_by_real_class.get(real, 0) + 1
            if status == INDETERMINADO:
                indeterminado_by_real_class[real] = (
                    indeterminado_by_real_class.get(real, 0) + 1
                )

    n_binary = tp + tn + fp + fn
    accuracy = _safe_div(tp + tn, n_binary)
    precision = _safe_div(tp, tp + fp)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = (
        _safe_div(2 * precision * sensitivity, precision + sensitivity)
        if precision is not None and sensitivity is not None
        else None
    )
    binary_metrics = BinaryMetrics(
        n_binary=n_binary, accuracy=accuracy, precision=precision,
        sensitivity=sensitivity, specificity=specificity, f1=f1,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )

    coverage_binary = _safe_div(n_positivo + n_negativo, n_total)
    valid_output_rate = _safe_div(n_positivo + n_negativo + n_indeterminado, n_total)
    accuracy_over_total = _safe_div(tp + tn, n_total)

    by_real_class: dict[str, dict[str, Any]] = {}
    for real_class, total in totals_by_real_class.items():
        count = indeterminado_by_real_class.get(real_class, 0)
        by_real_class[real_class] = {
            "count": count,
            "total_in_class": total,
            "rate": _safe_div(count, total),
        }
    indeterminado_breakdown = IndeterminadoBreakdown(by_real_class=by_real_class)

    return Stage6Indicators(
        n_total=n_total,
        n_positivo=n_positivo,
        n_negativo=n_negativo,
        n_indeterminado=n_indeterminado,
        n_ausente=n_ausente,
        n_nao_reconhecido=n_nao_reconhecido,
        binary_metrics=binary_metrics,
        coverage_binary=coverage_binary,
        valid_output_rate=valid_output_rate,
        accuracy_over_total=accuracy_over_total,
        indeterminado_breakdown=indeterminado_breakdown,
    )
