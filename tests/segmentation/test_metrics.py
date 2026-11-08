"""Synthetic (fabricated arrays, no real I/O) tests for the approved
segmentation metrics (stage 4, architecture-comparison pilot).

STATUS: implemented, awaiting researcher execution (pytest). Not run by
the assistant/fork that wrote them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_image_metrics,
    dice_per_image,
    hd95_symmetric,
    iou_jaccard,
    is_isic_failure,
    restore_probability_to_original_resolution,
    thresholded_jaccard,
)


def _square_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """box = (row_start, row_end, col_start, col_end), exclusive ends."""
    mask = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = box
    mask[r0:r1, c0:c1] = True
    return mask


# ──────────────────────────────────────────────
# Mascara perfeita
# ──────────────────────────────────────────────


def test_perfect_mask_dice_iou_one_no_failure():
    mask = _square_mask((20, 20), (4, 12, 4, 12))
    assert dice_per_image(mask, mask) == 1.0
    assert iou_jaccard(mask, mask) == 1.0
    assert is_isic_failure(iou_jaccard(mask, mask)) is False


def test_perfect_mask_hd95_zero():
    mask = _square_mask((20, 20), (4, 12, 4, 12))
    hd95 = hd95_symmetric(mask, mask)
    assert hd95 is not None
    assert hd95 == 0.0


def test_compute_image_metrics_perfect_mask_status_ok():
    ref = _square_mask((20, 20), (4, 12, 4, 12))
    pred_prob = ref.astype(np.float32)  # 0.0/1.0 "probabilities"
    result = compute_image_metrics(pred_prob, ref, threshold=0.5)
    assert result["status"] == "ok"
    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["hd95"] == 0.0
    assert result["hd95_normalizado_diagonal"] == 0.0  # 0.0 / diagonal == 0.0
    assert result["is_isic_failure"] is False
    assert result["thresholded_jaccard"] == 1.0


# ──────────────────────────────────────────────
# Predicao vazia, referencia valida
# ──────────────────────────────────────────────


def test_empty_prediction_valid_reference():
    ref = _square_mask((20, 20), (4, 12, 4, 12))
    pred_prob = np.zeros((20, 20), dtype=np.float32)  # all below any reasonable threshold
    result = compute_image_metrics(pred_prob, ref, threshold=0.5)
    assert result["status"] == "ok"
    assert result["dice"] == 0.0
    assert result["iou"] == 0.0
    assert result["hd95"] is None  # undefined -- never 0.0
    assert result["hd95_normalizado_diagonal"] is None  # undefined follows hd95 -- never 0.0
    assert result["is_isic_failure"] is True
    assert result["thresholded_jaccard"] == 0.0


# ──────────────────────────────────────────────
# Referencia invalida (vazia)
# ──────────────────────────────────────────────


def test_invalid_empty_reference():
    ref = np.zeros((20, 20), dtype=bool)
    pred_prob = np.full((20, 20), 0.9, dtype=np.float32)
    result = compute_image_metrics(pred_prob, ref, threshold=0.5)
    assert result["status"] == "referencia_invalida"
    assert result["dice"] is None
    assert result["iou"] is None
    assert result["hd95"] is None
    assert result["hd95_normalizado_diagonal"] is None
    assert result["is_isic_failure"] is None
    assert result["thresholded_jaccard"] is None


# ──────────────────────────────────────────────
# Mascara incompativel (formas diferentes)
# ──────────────────────────────────────────────


def test_incompatible_shapes():
    ref = _square_mask((20, 20), (4, 12, 4, 12))
    pred_prob = np.zeros((10, 10), dtype=np.float32)
    result = compute_image_metrics(pred_prob, ref, threshold=0.5)
    assert result["status"] == "mascara_incompativel"
    assert result["dice"] is None
    assert result["hd95_normalizado_diagonal"] is None


# ──────────────────────────────────────────────
# Ponto de corte do Jaccard (Thresholded Jaccard)
# ──────────────────────────────────────────────


def test_thresholded_jaccard_below_cutoff_is_zero():
    assert thresholded_jaccard(0.5) == 0.0


def test_thresholded_jaccard_exactly_at_cutoff_is_preserved():
    assert thresholded_jaccard(0.65) == 0.65


def test_thresholded_jaccard_above_cutoff_is_preserved():
    assert thresholded_jaccard(0.70) == 0.70


def test_is_isic_failure_strict_cutoff():
    assert is_isic_failure(0.64999) is True
    assert is_isic_failure(0.65) is False
    assert is_isic_failure(0.66) is False


# ──────────────────────────────────────────────
# Restauracao a resolucao original (ANTES de binarizar)
# ──────────────────────────────────────────────


def test_restore_before_binarize_differs_from_wrong_order():
    # Values chosen so the bilinear-interpolated (align_corners=False) row at
    # output index 1 lands on the OPPOSITE side of the 0.5 threshold from the
    # naive "binarize-small-then-nearest-repeat" result -- hand-verified:
    # torch's align_corners=False maps output row i=1 (of 4) for a 2->4
    # upsample to input coordinate 0.25, i.e. value = 0.75*v0 + 0.25*v1.
    # With v0=0.45, v1=1.0: 0.75*0.45 + 0.25*1.0 = 0.5875 > 0.5 (True), while
    # the naive method just repeats v0's own classification (0.45 > 0.5 is
    # False) for that row. This is a deliberate, verified divergence, not
    # the two orders coincidentally agreeing.
    small_prob = np.array([[0.45, 0.45], [1.0, 1.0]], dtype=np.float32)
    target_shape = (4, 4)

    restored = restore_probability_to_original_resolution(small_prob, target_shape)
    # Continuous (bilinear) interpolation must produce intermediate values,
    # not just the original 0.45/1.0 repeated -- proves real interpolation.
    unique_values = {round(v, 3) for v in restored.flatten().tolist()}
    assert len(unique_values) > 2

    correct_binary = restored > 0.5

    # Wrong order: binarize FIRST at the small resolution, then upsample the
    # already-binary mask with nearest-neighbor repetition.
    small_binary = small_prob > 0.5
    wrong_binary = np.repeat(np.repeat(small_binary, 2, axis=0), 2, axis=1)

    assert correct_binary.shape == wrong_binary.shape == target_shape
    assert not np.array_equal(correct_binary, wrong_binary)
    # Specifically: output row 1 flips classification between the two orders.
    assert bool(correct_binary[1, 0]) is True
    assert bool(wrong_binary[1, 0]) is False


def test_compute_image_metrics_uses_restored_shape_directly():
    ref = _square_mask((8, 8), (2, 6, 2, 6))
    small_prob = np.full((4, 4), 0.9, dtype=np.float32)
    restored = restore_probability_to_original_resolution(small_prob, ref.shape)
    result = compute_image_metrics(restored, ref, threshold=0.5)
    assert result["status"] == "ok"
    assert result["dice"] is not None


# ──────────────────────────────────────────────
# aggregate_metrics
# ──────────────────────────────────────────────


def test_aggregate_metrics_separates_status_and_hd95_undefined():
    ref = _square_mask((20, 20), (4, 12, 4, 12))
    ok_perfect = compute_image_metrics(ref.astype(np.float32), ref, threshold=0.5)
    ok_empty_pred = compute_image_metrics(np.zeros((20, 20), dtype=np.float32), ref, threshold=0.5)
    invalid_ref = compute_image_metrics(
        np.full((20, 20), 0.9, dtype=np.float32), np.zeros((20, 20), dtype=bool), threshold=0.5
    )

    agg = aggregate_metrics([ok_perfect, ok_empty_pred, invalid_ref])
    assert agg["n_total"] == 3
    assert agg["n_ok"] == 2
    assert agg["counts_by_status"]["ok"] == 2
    assert agg["counts_by_status"]["referencia_invalida"] == 1
    # hd95: one finite (0.0, perfect case) + one undefined (empty-pred case) among the 2 "ok" cases.
    assert agg["hd95_n_undefined"] == 1
    assert agg["hd95_median_finite"] == 0.0
    # hd95_normalizado_diagonal follows the same finite/undefined split as hd95.
    assert agg["hd95_normalizado_diagonal_median_finite"] == 0.0
    assert agg["mean_dice"] == 0.5  # (1.0 + 0.0) / 2
