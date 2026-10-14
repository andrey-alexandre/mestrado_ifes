"""Métricas diagnósticas binárias com intervalos de confiança (métodos explícitos).

- Sensibilidade, especificidade, precisão e acurácia: proporções binomiais -> IC de
  Wilson (escore).
- F1 e acurácia balanceada: razões/combinações de proporções -> IC por bootstrap
  percentil ESTRATIFICADO por classe (preserva n por classe), semente fixa.
- Diferença de proporções entre datasets: IC de Newcombe (híbrido de escore, método 10)
  e teste exato de Fisher (exploratório).
- ROC-AUC NÃO é calculada: a saída é categórica, sem escores contínuos comparáveis.
"""
import math
from statistics import NormalDist
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import fisher_exact

NAN = float("nan")


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    t, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    return {"tp": int(((t == 1) & (p == 1)).sum()), "tn": int(((t == 0) & (p == 0)).sum()),
            "fp": int(((t == 0) & (p == 1)).sum()), "fn": int(((t == 1) & (p == 0)).sum())}


def _div(a: float, b: float) -> float:
    return a / b if b > 0 else NAN


def point_metrics(c: Dict[str, int]) -> Dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    sens, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    bal = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else NAN
    return {"sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1,
            "balanced_accuracy": bal, "accuracy": _div(tp + tn, tp + tn + fp + fn)}


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        return NAN, NAN
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe_diff_ci(k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Diferença p1 - p2 e IC de Newcombe (híbrido de escore de Wilson)."""
    if n1 <= 0 or n2 <= 0:
        return NAN, NAN, NAN
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_ci(k1, n1, confidence)
    l2, u2 = wilson_ci(k2, n2, confidence)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def _stratified_bootstrap(y: np.ndarray, p: np.ndarray, n_resamples: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = {k: np.full(n_resamples, np.nan) for k in ("f1", "balanced_accuracy", "precision", "accuracy")}
    for b in range(n_resamples):
        idx = np.concatenate([rng.choice(pos, len(pos)) if len(pos) else pos,
                              rng.choice(neg, len(neg)) if len(neg) else neg])
        m = point_metrics(confusion(y[idx], p[idx]))
        for k in out:
            out[k][b] = m[k]
    return out


def _pct(dist: np.ndarray, confidence: float) -> Tuple[float, float, int]:
    d = dist[~np.isnan(dist)]
    if len(d) == 0:
        return NAN, NAN, 0
    a = (1 - confidence) / 2
    return float(np.quantile(d, a)), float(np.quantile(d, 1 - a)), int(len(d))


def metrics_with_ci(y_true: Sequence[int], y_pred: Sequence[int], confidence: float = 0.95,
                    n_resamples: int = 10000, seed: int = 42) -> Dict:
    """Matriz de confusão, métricas e ICs (Wilson / bootstrap estratificado)."""
    y, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    c = confusion(y, p)
    m = point_metrics(c)
    n_pos, n_neg = c["tp"] + c["fn"], c["tn"] + c["fp"]
    out: Dict = {
        "n": int(len(y)), "n_positive": n_pos, "n_negative": n_neg, **c,
        "confusion_matrix": [[c["tn"], c["fp"]], [c["fn"], c["tp"]]],
        "confusion_matrix_axes": "linhas = real [nevo, melanoma]; colunas = previsto [nevo, melanoma]",
        "ci_method": {"sensitivity": "Wilson", "specificity": "Wilson", "precision": "Wilson",
                      "accuracy": "Wilson",
                      "f1": f"bootstrap percentil estratificado (B={n_resamples}, seed={seed})",
                      "balanced_accuracy": f"bootstrap percentil estratificado (B={n_resamples}, seed={seed})"},
        "confidence": confidence,
    }
    dens = {"sensitivity": (c["tp"], n_pos), "specificity": (c["tn"], n_neg),
            "precision": (c["tp"], c["tp"] + c["fp"]), "accuracy": (c["tp"] + c["tn"], len(y))}
    for k, (num, den) in dens.items():
        lo, hi = wilson_ci(num, den, confidence)
        out[k] = {"estimate": m[k], "ci_low": lo, "ci_high": hi, "numerator": num, "denominator": den}
    if len(y) and n_pos and n_neg:
        boot = _stratified_bootstrap(y, p, n_resamples, seed)
    else:
        boot = {k: np.full(1, np.nan) for k in ("f1", "balanced_accuracy")}
    for k in ("f1", "balanced_accuracy"):
        lo, hi, nv = _pct(boot[k], confidence)
        out[k] = {"estimate": m[k], "ci_low": lo, "ci_high": hi, "n_resamples_valid": nv}
    out["roc_auc"] = {"estimate": None, "note": "não calculada: saída categórica, sem escores contínuos comparáveis"}
    return out


def compare_proportions(k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95) -> Dict:
    d, lo, hi = newcombe_diff_ci(k1, n1, k2, n2, confidence)
    p = NAN
    if n1 > 0 and n2 > 0:
        p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    return {"p1": _div(k1, n1), "n1": n1, "p2": _div(k2, n2), "n2": n2, "difference": d,
            "ci_low": lo, "ci_high": hi, "ci_method": "Newcombe (escore híbrido)",
            "fisher_exact_p_two_sided": p, "exploratory": True}


def holm_adjust(pvalues: Sequence[float]) -> list:
    """Correção de Holm-Bonferroni (NaN preservado)."""
    p = np.asarray(pvalues, float)
    out = np.full(len(p), np.nan)
    valid = np.where(~np.isnan(p))[0]
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out.tolist()


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return NAN
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def safe_float(x: Optional[float]) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
