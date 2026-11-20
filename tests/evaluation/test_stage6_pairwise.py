from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stage6_pairwise import compute_pairwise_comparison  # noqa: E402

UNIVERSE = [f"L{i}" for i in range(1, 9)]
REAL = {
    "L1": "melanoma", "L2": "melanoma", "L3": "melanoma", "L4": "melanoma",
    "L5": "nao_melanoma", "L6": "nao_melanoma", "L7": "nao_melanoma", "L8": "nao_melanoma",
}


def test_intersection_excludes_non_binary_on_either_side():
    # A: L1..L6 binario, L7/L8 indeterminado
    status_a = {"L1": "positivo", "L2": "positivo", "L3": "negativo", "L4": "negativo",
                "L5": "negativo", "L6": "negativo", "L7": "indeterminado", "L8": "indeterminado"}
    # B: L3..L8 binario, L1/L2 ausente
    status_b = {"L1": "ausente", "L2": "ausente", "L3": "positivo", "L4": "negativo",
                "L5": "negativo", "L6": "negativo", "L7": "negativo", "L8": "negativo"}
    result = compute_pairwise_comparison("A", "B", status_a, status_b, REAL, UNIVERSE)
    assert result.intersection_lesion_ids == ["L3", "L4", "L5", "L6"]
    assert result.n_intersection == 4
    assert result.n_melanoma == 2  # L3,L4
    assert result.n_nao_melanoma == 2  # L5,L6


def test_metrics_computed_on_same_intersection_for_both_sides():
    status_a = {"L1": "positivo", "L2": "positivo", "L3": "positivo", "L4": "positivo",
                "L5": "positivo", "L6": "positivo", "L7": "positivo", "L8": "positivo"}
    status_b = {"L1": "negativo", "L2": "negativo", "L3": "negativo", "L4": "negativo",
                "L5": "negativo", "L6": "negativo", "L7": "negativo", "L8": "negativo"}
    result = compute_pairwise_comparison("A", "B", status_a, status_b, REAL, UNIVERSE)
    assert result.n_intersection == 8
    assert result.first_metrics.tp == 4 and result.first_metrics.fp == 4
    assert result.second_metrics.fn == 4 and result.second_metrics.tn == 4
    assert result.point_diff_first_minus_second["accuracy"] == result.first_metrics.accuracy - result.second_metrics.accuracy


def test_diff_order_is_first_minus_second():
    status_a = {lid: "positivo" for lid in UNIVERSE}
    status_b = {lid: "negativo" for lid in UNIVERSE}
    result = compute_pairwise_comparison("first", "second", status_a, status_b, REAL, UNIVERSE)
    # accuracy_a = 4/8=0.5 (TP4,FP4); accuracy_b = 4/8=0.5 (TN4,FN4) -> diff 0
    assert result.point_diff_first_minus_second["accuracy"] == 0.0
    assert result.first_metrics.sensitivity == 1.0
    assert result.second_metrics.sensitivity == 0.0
    assert result.point_diff_first_minus_second["sensitivity"] == 1.0


def test_empty_intersection_yields_none_metrics_not_zero():
    status_a = {lid: "indeterminado" for lid in UNIVERSE}
    status_b = {lid: "positivo" for lid in UNIVERSE}
    result = compute_pairwise_comparison("A", "B", status_a, status_b, REAL, UNIVERSE)
    assert result.n_intersection == 0
    assert result.first_metrics.accuracy is None
    assert result.point_diff_first_minus_second["accuracy"] is None
