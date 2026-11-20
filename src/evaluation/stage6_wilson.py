"""Etapa 6 (revisão metodológica 2026-11-19) -- intervalo de confiança de
Wilson (score interval, bilateral, SEM correção de continuidade) para as
proporções individuais listadas na decisão do pesquisador: acurácia,
precisão, sensibilidade, especificidade, cobertura binária e acertos
sobre o total.

Implementação do formato fechado padrão (Wilson, 1927), usando
``scipy.stats.norm.ppf`` só para o quantil normal -- ``scipy`` já é
dependência do projeto (`src/data_audit/phash.py` usa `scipy.fft.dctn`),
então nenhuma dependência nova é adicionada só para este cálculo
(decisão explícita do pesquisador: "evite adicionar uma dependência
pesada apenas para esse cálculo" -- `statsmodels` foi considerado e
descartado por esse motivo).

NÃO é Wald (que usa ``phat ± z*sqrt(phat(1-phat)/n)``, sem o termo de
correção ``z²/n`` que desloca o centro), NÃO é Agresti-Coull (que usa um
``n`` ajustado, ``n + z²``, com o centro recalculado a partir dele) e
NÃO é Clopper-Pearson (exato, baseado na distribuição Beta/F, sem forma
fechada). As três são descartadas explicitamente pelo pesquisador.

A estimativa pontual devolvida é SEMPRE a proporção observada
(``successes / n``), nunca o centro do intervalo de Wilson (que é
deslocado em relação a ela por construção) -- distinção explícita
pedida pelo pesquisador.

Estes intervalos são rotulados, em toda saída (tabela/relatório/JSON),
como "intervalos binomiais aproximados para as proporções observadas":
eles tratam cada proporção como uma única amostra binomial independente
sobre o denominador aprovado, SEM o condicionamento por categoria
diagnóstica que o bootstrap estratificado aplica, e SEM qualquer ajuste
ao desenho amostral da amostra congelada -- não devem ser lidos como
equivalentes aos IC bootstrap que permanecem em uso para F1 e para
diferenças pareadas (ver `src.evaluation.stage6_bootstrap_ci`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scipy.stats import norm

from src.evaluation.stage6_metrics import Stage6Indicators

METHOD_WILSON = "wilson_score_no_continuity_correction"

DEFAULT_CONFIDENCE = 0.95


@dataclass
class WilsonResult:
    point_estimate: Optional[float]  # proporcao observada (successes/n), NUNCA o centro de Wilson
    ci_low: Optional[float]
    ci_high: Optional[float]
    numerator: int
    denominator: int
    confidence: float
    method: str


def wilson_score_interval(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE) -> WilsonResult:
    """Intervalo de Wilson bilateral, sem correção de continuidade.

    Args:
        successes: numerador (ex.: TP+TN, TP, decisões binárias válidas).
        n: denominador aprovado para este indicador (ex.: TP+TN+FP+FN,
            TP+FP, total do manifesto).
        confidence: nível de confiança (default 0,95 -- IC95%).

    Se ``n == 0``, denominador indefinido: estimativa e IC são ``None``
    (NUNCA 0.0) -- ``numerator``/``denominator`` continuam preservados
    (0/0) para auditoria. ``successes == 0`` e ``successes == n`` são
    calculados normalmente, sem caso especial (o formato fechado de
    Wilson é bem definido nos dois extremos e naturalmente permanece
    dentro de [0, 1])."""
    if n == 0:
        return WilsonResult(None, None, None, successes, n, confidence, METHOD_WILSON)
    if successes < 0 or successes > n:
        raise ValueError(f"successes ({successes}) fora do intervalo [0, n={n}]")

    z = norm.ppf(1 - (1 - confidence) / 2)
    z2 = z * z
    phat = successes / n

    denom = 1 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z / denom) * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)

    ci_low = max(0.0, center - margin)
    ci_high = min(1.0, center + margin)

    return WilsonResult(phat, ci_low, ci_high, successes, n, confidence, METHOD_WILSON)


# Tabela fechada numerador/denominador aprovada (decisão do pesquisador,
# seção 1) -- a ÚNICA fonte desta correspondência; nenhum outro lugar do
# código decide de novo quais contagens alimentam Wilson para cada
# indicador.
def wilson_results_for_indicators(indicators: Stage6Indicators, confidence: float = DEFAULT_CONFIDENCE) -> dict[str, WilsonResult]:
    """``accuracy``/``precision``/``sensitivity``/``specificity``/
    ``coverage_binary``/``accuracy_over_total`` -- todos via Wilson, a
    partir das mesmas contagens já computadas por
    ``stage6_metrics.compute_stage6_indicators`` (nunca recontadas aqui).
    F1 NÃO está nesta tabela -- permanece bootstrap (ver
    ``src.evaluation.stage6_bootstrap_ci``), decisão explícita do
    pesquisador (F1 não é uma proporção simples de sucessos/denominador)."""
    bm = indicators.binary_metrics
    return {
        "accuracy": wilson_score_interval(bm.tp + bm.tn, bm.tp + bm.tn + bm.fp + bm.fn, confidence),
        "precision": wilson_score_interval(bm.tp, bm.tp + bm.fp, confidence),
        "sensitivity": wilson_score_interval(bm.tp, bm.tp + bm.fn, confidence),
        "specificity": wilson_score_interval(bm.tn, bm.tn + bm.fp, confidence),
        "coverage_binary": wilson_score_interval(bm.n_binary, indicators.n_total, confidence),
        "accuracy_over_total": wilson_score_interval(bm.tp + bm.tn, indicators.n_total, confidence),
    }
