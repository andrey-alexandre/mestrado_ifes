"""Synthetic (fabricated small dicts, no real I/O) tests for
src/segmentation/nested_cv.py -- the nested cross-validation executor
logic (stage 4, mechanical rehearsal of one external round).

STATUS: implemented, awaiting researcher execution (pytest). Not run by
the assistant/fork that wrote them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.segmentation.nested_cv import (  # noqa: E402
    THRESHOLD_GRID,
    build_external_fold_ids,
    build_internal_rounds,
    build_retrain_ids,
    early_stopping_check,
    select_epochs_via_median,
    select_threshold_from_internal_dice,
)


# ──────────────────────────────────────────────
# early_stopping_check
# ──────────────────────────────────────────────


def test_early_stopping_continues_when_neither_condition_met():
    assert early_stopping_check(epoch=3, max_epochs=20, epochs_without_improvement=1, patience=5) is None


def test_early_stopping_fires_on_patience_exhausted():
    assert early_stopping_check(epoch=6, max_epochs=20, epochs_without_improvement=5, patience=5) == "paciencia_esgotada"


def test_early_stopping_fires_on_ceiling_reached():
    assert early_stopping_check(epoch=20, max_epochs=20, epochs_without_improvement=2, patience=5) == "teto_atingido"


def test_early_stopping_patience_takes_precedence_over_ceiling():
    # Both conditions true simultaneously -- patience is checked first, so
    # this must be reported as "paciencia_esgotada", never "teto_atingido"
    # (a round that happens to exhaust patience on its last possible epoch
    # did NOT converge just because it also hit the ceiling).
    assert early_stopping_check(epoch=20, max_epochs=20, epochs_without_improvement=5, patience=5) == "paciencia_esgotada"


# ──────────────────────────────────────────────
# select_epochs_via_median
# ──────────────────────────────────────────────


def test_select_epochs_via_median_no_rounding_needed():
    assert select_epochs_via_median([2, 2, 2, 2]) == 2
    assert select_epochs_via_median([10, 20, 30, 40]) == 25  # median exactly 25.0


def test_select_epochs_via_median_forces_half_up_rounding():
    # median([1, 2, 3, 4]) == 2.5 -- round-half-up gives 3, NOT Python's
    # round() banker's rounding (which would give 2).
    assert select_epochs_via_median([1, 2, 3, 4]) == 3
    # Same halfway case via a different set of values, confirming it is
    # not a coincidence of this specific list.
    assert select_epochs_via_median([1, 1, 4, 4]) == 3


# ──────────────────────────────────────────────
# select_threshold_from_internal_dice
# ──────────────────────────────────────────────


def test_select_threshold_clear_winner():
    dice_by_threshold = {0.30: [0.5], 0.50: [0.9], 0.70: [0.4]}
    result = select_threshold_from_internal_dice(dice_by_threshold)
    assert result["chosen_threshold"] == 0.50
    assert result["empate"] is False
    assert result["empate_simetrico"] is False
    assert result["tied_thresholds"] == []


def test_select_threshold_tie_resolved_by_proximity_to_half():
    # 0.45 and 0.65 tie at mean Dice 0.8; 0.45 is closer to 0.5 (0.05 vs 0.15).
    dice_by_threshold = {0.45: [0.8, 0.8], 0.65: [0.8, 0.8], 0.30: [0.1]}
    result = select_threshold_from_internal_dice(dice_by_threshold)
    assert result["empate"] is True
    assert sorted(result["tied_thresholds"]) == [0.45, 0.65]
    assert result["chosen_threshold"] == 0.45
    assert result["empate_simetrico"] is False


def test_select_threshold_symmetric_tie_is_flagged():
    # 0.45 and 0.55 tie at mean Dice 0.8 AND are equally close to 0.5 (0.05 each).
    dice_by_threshold = {0.45: [0.8], 0.55: [0.8], 0.30: [0.1]}
    result = select_threshold_from_internal_dice(dice_by_threshold)
    assert result["empate"] is True
    assert result["empate_simetrico"] is True
    # Deterministic fallback (documented): the smaller of the residual-tied
    # thresholds -- but the important assertion is that the ambiguity is
    # flagged, not that this specific value is "correct".
    assert result["chosen_threshold"] == 0.45


def test_select_threshold_skips_empty_threshold_lists():
    dice_by_threshold = {0.30: [], 0.50: [0.9], 0.70: []}
    result = select_threshold_from_internal_dice(dice_by_threshold)
    assert result["chosen_threshold"] == 0.50
    assert 0.30 not in result["mean_dice_by_threshold"]
    assert 0.70 not in result["mean_dice_by_threshold"]


def test_select_threshold_raises_when_all_empty():
    with pytest.raises(ValueError):
        select_threshold_from_internal_dice({0.30: [], 0.50: []})


def test_threshold_grid_matches_protocol_spec():
    assert THRESHOLD_GRID == (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


# ──────────────────────────────────────────────
# Construcao de listas por fase + checagem de vazamento
# ──────────────────────────────────────────────


def _fake_folds_data() -> dict:
    # 5 folds, 2 images each, 10 images total.
    image_to_fold = {}
    for fold in range(5):
        for i in range(2):
            image_to_fold[f"img_f{fold}_{i}"] = fold
    return {"n_folds": 5, "image_to_fold": image_to_fold}


def test_build_external_fold_ids():
    folds_data = _fake_folds_data()
    external_ids = build_external_fold_ids(folds_data, external_fold=0)
    assert sorted(external_ids) == ["img_f0_0", "img_f0_1"]


def test_build_internal_rounds_structure():
    folds_data = _fake_folds_data()
    rounds = build_internal_rounds(folds_data, external_fold=0)
    assert len(rounds) == 4
    val_folds = sorted(r["internal_val_fold"] for r in rounds)
    assert val_folds == [1, 2, 3, 4]

    round_for_val1 = next(r for r in rounds if r["internal_val_fold"] == 1)
    assert sorted(round_for_val1["train_folds"]) == [2, 3, 4]
    assert sorted(round_for_val1["val_ids"]) == ["img_f1_0", "img_f1_1"]
    assert sorted(round_for_val1["train_ids"]) == sorted(
        ["img_f2_0", "img_f2_1", "img_f3_0", "img_f3_1", "img_f4_0", "img_f4_1"]
    )


def test_build_retrain_ids_is_union_of_development_folds():
    folds_data = _fake_folds_data()
    retrain_ids = build_retrain_ids(folds_data, external_fold=0)
    expected = sorted(
        f"img_f{fold}_{i}" for fold in (1, 2, 3, 4) for i in range(2)
    )
    assert sorted(retrain_ids) == expected


def test_no_leakage_of_external_fold_into_any_phase():
    folds_data = _fake_folds_data()
    external_ids = set(build_external_fold_ids(folds_data, external_fold=0))
    rounds = build_internal_rounds(folds_data, external_fold=0)
    retrain_ids = set(build_retrain_ids(folds_data, external_fold=0))

    assert external_ids.isdisjoint(retrain_ids)
    for r in rounds:
        assert external_ids.isdisjoint(set(r["train_ids"]))
        assert external_ids.isdisjoint(set(r["val_ids"]))
