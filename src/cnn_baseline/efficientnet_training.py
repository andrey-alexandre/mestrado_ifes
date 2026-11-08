"""Pure training-support logic for the BCN20000 EfficientNetV2-S baseline:
model construction, the checkpoint-selection rule, and validation-score
formatting. Kept separate from scripts/train_bcn20000_efficientnet.py so
the decision logic (which epoch wins, how scores are shaped) is testable
with fake numbers, without running real training or touching the GPU.

Threshold selection (indice de Youden) is explicitly NOT done anywhere in
this module -- it only ever produces/consumes raw prediction scores in
[0, 1]. That is a separate, later task.
"""
from __future__ import annotations

from typing import Any, Sequence

from src.cnn_baseline.bcn20000_dataset import BINARY_TARGET_TO_LABEL, LABEL_TO_BINARY_TARGET

CHECKPOINT_SELECTION_RULE = (
    "melhor AUC-ROC de validacao (maior valor); em caso de empate, a epoca mais cedo"
)


def select_best_epoch(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pick the epoch with the highest ``val_auc`` in ``history`` (each
    entry: ``{'epoch': int, 'train_loss': float, 'val_loss': float,
    'val_auc': float, ...}``).

    AUC-ROC (threshold-independent) was chosen over a fixed-0.5-threshold
    metric (e.g. accuracy) specifically to avoid any temporary threshold
    choice before the Youden-index task runs -- this selection never
    binarizes a prediction. Ties are broken by earliest epoch (prefers an
    earlier-converged, presumably simpler model over a later one with
    identical AUC) -- a documented, deterministic choice, not arbitrary.
    Raises ``ValueError`` on empty history rather than returning something
    silently wrong.
    """
    if not history:
        raise ValueError("historico de epocas vazio -- nada para selecionar.")
    best = max(history, key=lambda h: (h["val_auc"], -h["epoch"]))
    return {
        "selected_epoch": best["epoch"],
        "selected_val_auc": best["val_auc"],
        "rule": CHECKPOINT_SELECTION_RULE,
    }


def build_model(pretrained: bool = True):
    """Build EfficientNetV2-S with a single-logit binary classification
    head. Sigmoid is applied at inference/scoring time, never inside the
    model -- ``BCEWithLogitsLoss`` expects raw logits for numerical
    stability during training.
    """
    import torch.nn as nn
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model, weights


def build_val_scores_rows(
    lesion_ids: Sequence[str],
    bcn_filenames: Sequence[str],
    true_labels: Sequence[int],
    predicted_scores: Sequence[float],
) -> list[dict[str, Any]]:
    """Assemble the ``val_scores.csv`` rows: one row per validation image,
    with the RAW (sigmoid) ``predicted_score`` in [0, 1] -- never
    binarized here. Threshold selection (Youden index) is a separate,
    later task that consumes this file; this function's only job is to
    preserve the score faithfully and traceably.
    """
    if not (len(lesion_ids) == len(bcn_filenames) == len(true_labels) == len(predicted_scores)):
        raise ValueError("listas de tamanhos diferentes -- nao e possivel montar as linhas.")

    rows: list[dict[str, Any]] = []
    for lid, fname, label, score in zip(lesion_ids, bcn_filenames, true_labels, predicted_scores):
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"predicted_score fora de [0,1]: {score} para {fname}")
        if label not in LABEL_TO_BINARY_TARGET:
            raise ValueError(f"true_label inesperado: {label!r} para {fname}")
        rows.append(
            {
                "lesion_id": lid,
                "bcn_filename": fname,
                "binary_target": LABEL_TO_BINARY_TARGET[label],
                "true_label": label,
                "predicted_score": score,
            }
        )
    return rows


__all__ = [
    "BINARY_TARGET_TO_LABEL",
    "LABEL_TO_BINARY_TARGET",
    "CHECKPOINT_SELECTION_RULE",
    "select_best_epoch",
    "build_model",
    "build_val_scores_rows",
]
