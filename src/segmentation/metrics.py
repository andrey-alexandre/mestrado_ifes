"""Métricas de segmentação e estatística pareada. Independente de arquitetura/torch.

Casos degenerados (convenção explícita, registrada em `degenerate`):
  - ``pred_empty``: predição vazia, GT não vazia -> Dice = IoU = 0; HD95 indefinido (NaN).
  - ``gt_empty``:   GT vazia, predição não vazia -> Dice = IoU = 0; HD95 indefinido (NaN).
  - ``both_empty``: ambas vazias -> todas as métricas indefinidas (NaN).
Valores NaN são excluídos dos agregados e contados em ``n_undefined``.
"""
import math
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy import ndimage

METRIC_NAMES = ("dice", "iou", "hd95")
HIGHER_IS_BETTER = {"dice": True, "iou": True, "hd95": False}


def _as_bool(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"Máscara deve ser 2D, recebido shape {m.shape}")
    return m.astype(bool)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Distância de Hausdorff (percentil 95, em pixels) entre superfícies.

    Distâncias dirigidas pred->gt e gt->pred são agrupadas antes do percentil
    (convenção usual de MedPy). NaN se alguma das máscaras for vazia.
    """
    pred, gt = _as_bool(pred), _as_bool(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes diferentes: {pred.shape} vs {gt.shape}")
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_gt = ndimage.distance_transform_edt(~sg)
    d_to_pred = ndimage.distance_transform_edt(~sp)
    dists = np.concatenate([d_to_gt[sp], d_to_pred[sg]])
    return float(np.percentile(dists, 95))


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray, compute_hd95: bool = True) -> Dict:
    """Dice, IoU e HD95 para um par (predição, referência) já na mesma resolução."""
    pred, gt = _as_bool(pred), _as_bool(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"Shapes diferentes: {pred.shape} vs {gt.shape}")
    p, g = int(pred.sum()), int(gt.sum())
    if p == 0 and g == 0:
        return {"dice": float("nan"), "iou": float("nan"), "hd95": float("nan"),
                "degenerate": "both_empty", "pred_area": 0, "gt_area": 0}
    inter = int((pred & gt).sum())
    union = p + g - inter
    out = {
        "dice": 2.0 * inter / (p + g),
        "iou": inter / union,
        "hd95": hd95(pred, gt) if compute_hd95 else float("nan"),
        "degenerate": "pred_empty" if p == 0 else ("gt_empty" if g == 0 else "ok"),
        "pred_area": p,
        "gt_area": g,
    }
    return out


def summarize(values: Sequence[float]) -> Dict:
    """Média, mediana, desvio-padrão (ddof=1) e contagens, ignorando NaN."""
    arr = np.asarray(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    n = int(valid.size)
    return {
        "n": int(arr.size),
        "n_valid": n,
        "n_undefined": int(arr.size - n),
        "mean": float(valid.mean()) if n else float("nan"),
        "median": float(np.median(valid)) if n else float("nan"),
        "std": float(valid.std(ddof=1)) if n > 1 else float("nan"),
        "min": float(valid.min()) if n else float("nan"),
        "max": float(valid.max()) if n else float("nan"),
    }


def _percentile_ci(dist: np.ndarray, confidence: float) -> Dict:
    d = dist[~np.isnan(dist)]
    if d.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_resamples_valid": 0}
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.percentile(d, [100 * alpha, 100 * (1 - alpha)])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_resamples_valid": int(d.size)}


def bootstrap_statistic(
    arrays: Sequence[np.ndarray],
    statistic: Callable[..., float],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap pareado genérico: reamostra os mesmos índices em todos os arrays.

    Args:
        arrays: sequências de mesmo comprimento (uma linha por imagem).
        statistic: função (*arrays_reamostrados) -> escalar (NaN permitido).
    """
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    if any(len(a) != n for a in arrs):
        raise ValueError("Arrays do bootstrap pareado devem ter o mesmo comprimento.")
    if n == 0:
        raise ValueError("Bootstrap requer ao menos uma amostra.")
    rng = np.random.default_rng(seed)
    point = float(statistic(*arrs))
    dist = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        dist[b] = statistic(*[a[idx] for a in arrs])
    out = {"estimate": point, "n_pairs": n, "n_resamples": int(n_resamples),
           "confidence": confidence, "seed": seed}
    out.update(_percentile_ci(dist, confidence))
    return out


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    statistic: str = "mean",
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """IC por bootstrap pareado de (b - a). Pares com NaN em qualquer lado são excluídos.

    Vetorizado: não chama `bootstrap_statistic` em laço para manter 10k reamostras rápidas.
    """
    if statistic not in ("mean", "median"):
        raise ValueError("statistic deve ser 'mean' ou 'median'")
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Vetores pareados devem ter o mesmo comprimento.")
    ok = ~(np.isnan(a_arr) | np.isnan(b_arr))
    diff = (b_arr - a_arr)[ok]
    n = int(diff.size)
    base = {"statistic": statistic, "n_pairs": n, "n_excluded_undefined": int(a_arr.size - n),
            "n_resamples": int(n_resamples), "confidence": confidence, "seed": seed}
    if n == 0:
        return {**base, "estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples_valid": 0, "frac_positive": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    sample = diff[idx]
    dist = sample.mean(axis=1) if statistic == "mean" else np.median(sample, axis=1)
    est = float(diff.mean() if statistic == "mean" else np.median(diff))
    return {**base, "estimate": est, **_percentile_ci(dist, confidence),
            "frac_positive": float((diff > 0).mean())}


def nan_safe(value: Optional[float]) -> Optional[float]:
    """Converte NaN/inf em None para serialização JSON estrita."""
    if value is None:
        return None
    return None if (isinstance(value, float) and (math.isnan(value) or math.isinf(value))) else value
