"""Youden-index threshold selection for the BCN20000 EfficientNetV2-S
baseline (stage 3). Consumes ONLY ``val_scores.csv`` (raw, non-binarized
sigmoid scores from the internal BCN20000 validation split) -- HAM10000
is never read here, and never participates in fixing this threshold.

This is deliberately the "separate, later task" referenced in
``src/cnn_baseline/efficientnet_training.py``'s module docstring.
"""
from __future__ import annotations

from typing import Any, Sequence


def select_youden_threshold(
    true_labels: Sequence[int], predicted_scores: Sequence[float]
) -> dict[str, Any]:
    """Pick the ROC threshold maximizing the Youden index
    ``J = sensitivity + specificity - 1`` (equivalently ``tpr - fpr``).

    Uses ``sklearn.metrics.roc_curve``, which always prepends a
    threshold of +inf (representing "classify nothing as positive") to
    guarantee the curve starts at (0, 0) -- that point has
    ``tpr=fpr=0`` so ``J=0``, never the argmax unless every real
    threshold has ``J<=0``, but it is explicitly excluded from the
    selection below so a degenerate run can never select a
    non-actionable threshold.

    Ties (multiple thresholds sharing the max J) are broken by the
    smallest threshold value -- a deterministic, documented choice,
    not arbitrary (prefers the more permissive/sensitive operating
    point among equally-good ones).

    Raises ``ValueError`` if ``true_labels`` has fewer than 2 distinct
    classes (Youden's index is undefined without both classes present).
    """
    from sklearn.metrics import roc_curve

    if len(set(true_labels)) < 2:
        raise ValueError(
            "e necessario ter as duas classes presentes em true_labels para "
            "calcular o indice de Youden."
        )

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)

    finite_mask = thresholds != float("inf")
    fpr_f = fpr[finite_mask]
    tpr_f = tpr[finite_mask]
    thresholds_f = thresholds[finite_mask]

    youden_j = tpr_f - fpr_f
    best_j = youden_j.max()
    candidate_idxs = [i for i, j in enumerate(youden_j) if j == best_j]
    best_idx = min(candidate_idxs, key=lambda i: thresholds_f[i])

    sensitivity = float(tpr_f[best_idx])
    specificity = float(1.0 - fpr_f[best_idx])

    return {
        "threshold": float(thresholds_f[best_idx]),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_j": float(best_j),
        "rule": (
            "maior indice de Youden (sensibilidade + especificidade - 1) na "
            "curva ROC da validacao interna do BCN20000; empate: menor limiar"
        ),
    }


__all__ = ["select_youden_threshold"]
