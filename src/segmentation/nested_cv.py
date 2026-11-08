"""Pure, testable logic for the nested cross-validation protocol (stage 4,
segmentation). No I/O, no model, no torch -- separated from training so it
can be unit-tested with small fabricated fold dicts.

This module implements the EXECUTOR logic for the scientific protocol
already approved by the researcher (quoted verbatim, not reopened here):

    "A execucao cientifica tera cinco folds externos e quatro internos por
    rodada externa. Hiperparametros, checkpoint e limiar serao escolhidos
    somente nos dados internos. Em cada rodada, o retreino nos quatro
    folds de desenvolvimento usara a mediana das melhores epocas internas,
    sem early stopping no fold externo. O limiar sera escolhido pelo Dice
    medio das predicoes internas na grade de 0,30 a 0,70, em passos de
    0,05; em empate, use o valor mais proximo de 0,5 e sinalize se ainda
    houver empate simetrico."

This module does NOT run the 5 external rounds (only one external fold is
exercised, by the caller script, as a mechanical rehearsal) and does NOT
select a final architecture/configuration -- see
``scripts/run_nested_cv_rehearsal.py`` and
``docs/reconstruction/13_segmentation_deeplabv3plus_pilot.md`` for the
scope boundary.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Grid built via round(..., 2) to avoid float-arithmetic drift (0.30 + 3*0.05
# would otherwise be 0.44999999999999996, not 0.45) -- both this module's
# tests and scripts/run_nested_cv_rehearsal.py must reuse THIS constant
# rather than regenerating the grid independently.
THRESHOLD_GRID: tuple[float, ...] = tuple(round(0.30 + i * 0.05, 2) for i in range(9))


def early_stopping_check(
    epoch: int, max_epochs: int, epochs_without_improvement: int, patience: int
) -> str | None:
    """Decides whether an internal round should stop AFTER the given
    epoch, and why -- pure function, no I/O, so the stopping RULE is
    testable independently of real training.

    Researcher decision (2026-11-08, consolidation doc section 5, fixed
    BEFORE any new execution so the rule could not be adjusted after
    seeing results): a round stops when 5 consecutive epochs pass without
    a strict improvement of the internal monitoring Dice ("paciencia
    esgotada"), or when it reaches ``max_epochs`` without that happening
    ("teto atingido") -- ``max_epochs`` is an operational ceiling, NEVER a
    claim of convergence, even if the Dice was still improving when the
    ceiling was hit.

    Returns:
        ``"paciencia_esgotada"`` if ``epochs_without_improvement >= patience``.
        ``"teto_atingido"`` if ``epoch >= max_epochs`` (checked only after
            the patience condition, so a round that exhausts patience
            exactly on its last possible epoch is still reported as
            ``"paciencia_esgotada"``, not ``"teto_atingido"``).
        ``None`` if the round should keep training.
    """
    if epochs_without_improvement >= patience:
        return "paciencia_esgotada"
    if epoch >= max_epochs:
        return "teto_atingido"
    return None


def select_epochs_via_median(best_epochs: list[int]) -> int:
    """Median of the best-epoch values (one per internal fold), rounded to
    an integer epoch count.

    Rounding rule (deliberately NOT Python's ``round()``, which uses
    round-half-to-even / "banker's rounding" and would silently give
    different results for e.g. 2.5 vs. 3.5): round-HALF-UP,
    ``floor(median + 0.5)``. Chosen for being the more intuitive,
    unsurprising convention when a human reads "median of [1, 2, 3, 4] is
    2.5, rounds to 3" -- documented here precisely because an even-length
    list (4 internal folds) always produces a median that is either an
    exact integer or exactly-`.5`, so the halfway case is not a rare edge
    case here, it is the common one whenever the two middle values are not
    equal.
    """
    median = float(np.median(best_epochs))
    return math.floor(median + 0.5)


def select_threshold_from_internal_dice(dice_by_threshold: dict[float, list[float]]) -> dict[str, Any]:
    """Selects the mask threshold from the grid using pooled internal Dice.

    Args:
        dice_by_threshold: for each threshold in the grid, the pooled list
            of per-image Dice values (from the 4 internal folds' held-out
            predictions, using each fold's best-epoch checkpoint).

    Returns a dict with:
      - ``chosen_threshold``: the selected value.
      - ``mean_dice_by_threshold``: mean Dice per threshold (thresholds
        with an empty list are skipped, not treated as 0.0).
      - ``empate``: True if more than one threshold tied for the highest
        mean Dice.
      - ``tied_thresholds``: the thresholds that tied (empty list if
        ``empate`` is False).
      - ``empate_simetrico``: True if, AFTER resolving ``empate`` by
        proximity to 0.5, more than one threshold remains equally close to
        0.5 (e.g. 0.45 and 0.55) -- this is flagged explicitly, never
        silently resolved. The function still returns one deterministic
        ``chosen_threshold`` even when this flag is True (the smallest of
        the residual-tied thresholds, a documented, reproducible
        tie-break) -- the caller MUST surface ``empate_simetrico`` in the
        report; this function does not hide the ambiguity.

    Float-equality comparisons (both for the Dice-mean tie and the
    distance-to-0.5 tie) use an absolute tolerance of 1e-9 to avoid
    floating-point noise manufacturing or hiding a tie.
    """
    mean_by_threshold: dict[float, float] = {}
    for threshold, dices in dice_by_threshold.items():
        if dices:
            mean_by_threshold[threshold] = float(np.mean(dices))

    if not mean_by_threshold:
        raise ValueError("dice_by_threshold has no threshold with any Dice values")

    best_mean = max(mean_by_threshold.values())
    tied = sorted(t for t, m in mean_by_threshold.items() if abs(m - best_mean) < 1e-9)

    empate = len(tied) > 1
    empate_simetrico = False

    if not empate:
        chosen = tied[0]
    else:
        distances = {t: abs(t - 0.5) for t in tied}
        min_distance = min(distances.values())
        closest = sorted(t for t, d in distances.items() if abs(d - min_distance) < 1e-9)
        if len(closest) == 1:
            chosen = closest[0]
        else:
            empate_simetrico = True
            chosen = min(closest)  # documented, deterministic fallback -- flagged, not hidden

    return {
        "chosen_threshold": chosen,
        "mean_dice_by_threshold": mean_by_threshold,
        "empate": empate,
        "tied_thresholds": tied if empate else [],
        "empate_simetrico": empate_simetrico,
    }


def _ids_by_fold(folds_data: dict[str, Any]) -> dict[int, list[str]]:
    """Groups image ids by fold index, sorted -- same grouping already
    used by ``scripts/run_segmentation_arch_pilot.py``'s
    ``_build_fold_id_lists``, reimplemented here (not imported) to keep
    this module free of any script-level import and fully I/O-free."""
    n_folds = folds_data["n_folds"]
    ids_by_fold: dict[int, list[str]] = {i: [] for i in range(n_folds)}
    for image_id, fold in folds_data["image_to_fold"].items():
        ids_by_fold[fold].append(image_id)
    for fold in ids_by_fold:
        ids_by_fold[fold].sort()
    return ids_by_fold


def build_external_fold_ids(folds_data: dict[str, Any], external_fold: int) -> list[str]:
    """Image ids belonging to the external fold -- NEVER used for training,
    internal validation, or threshold selection; only for the single final
    external evaluation."""
    return _ids_by_fold(folds_data)[external_fold]


def build_internal_rounds(folds_data: dict[str, Any], external_fold: int) -> list[dict[str, Any]]:
    """Builds the 4 internal rounds (one development fold held out as
    internal validation per round, the other 3 development folds as
    internal training) for the given ``external_fold`` -- which never
    appears in any round's ``train_ids``/``val_ids``.

    Returns a list of dicts: ``{"internal_val_fold", "train_folds",
    "train_ids", "val_ids"}``, one per development fold.
    """
    ids_by_fold = _ids_by_fold(folds_data)
    dev_folds = [i for i in range(folds_data["n_folds"]) if i != external_fold]

    rounds = []
    for val_fold in dev_folds:
        train_folds = [f for f in dev_folds if f != val_fold]
        train_ids = sorted(iid for f in train_folds for iid in ids_by_fold[f])
        rounds.append({
            "internal_val_fold": val_fold,
            "train_folds": train_folds,
            "train_ids": train_ids,
            "val_ids": ids_by_fold[val_fold],
        })
    return rounds


def build_retrain_ids(folds_data: dict[str, Any], external_fold: int) -> list[str]:
    """Union of all 4 development folds (never the external fold) -- the
    training set for the final retrain-with-chosen-epochs step."""
    ids_by_fold = _ids_by_fold(folds_data)
    dev_folds = [i for i in range(folds_data["n_folds"]) if i != external_fold]
    return sorted(iid for f in dev_folds for iid in ids_by_fold[f])
