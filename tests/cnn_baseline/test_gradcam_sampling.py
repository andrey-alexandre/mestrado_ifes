from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cnn_baseline.gradcam_sampling import ALL_GROUPS, select_gradcam_sample  # noqa: E402


def _row(lesion_id, true_target, pred_target, status="ok"):
    return {"lesion_id": lesion_id, "image_id": f"IMG_{lesion_id}", "dx": "nv",
            "true_binary_target": true_target, "predicted_binary_target": pred_target,
            "predicted_score": 0.5, "status": status}


def _make_rows(n_per_group=15):
    rows = []
    specs = {"TP": ("melanoma", "melanoma"), "TN": ("nao_melanoma", "nao_melanoma"),
             "FP": ("nao_melanoma", "melanoma"), "FN": ("melanoma", "nao_melanoma")}
    for group, (real, pred) in specs.items():
        for i in range(n_per_group):
            rows.append(_row(f"{group}{i:03d}", real, pred))
    return rows


def test_selects_10_per_cell_when_enough_eligible():
    rows = _make_rows(15)
    result = select_gradcam_sample(rows, seed=42, n_per_cell=10)
    for group in ALL_GROUPS:
        assert result.n_selected_by_group[group] == 10
        assert result.n_eligible_by_group[group] == 15
    assert not result.shortfall_groups


def test_deterministic_given_same_seed():
    rows = _make_rows(15)
    r1 = select_gradcam_sample(rows, seed=42)
    r2 = select_gradcam_sample(rows, seed=42)
    for group in ALL_GROUPS:
        ids1 = [r["lesion_id"] for r in r1.selected_by_group[group]]
        ids2 = [r["lesion_id"] for r in r2.selected_by_group[group]]
        assert ids1 == ids2


def test_different_seed_can_change_selection():
    rows = _make_rows(15)
    r1 = select_gradcam_sample(rows, seed=42)
    r2 = select_gradcam_sample(rows, seed=43)
    ids1 = [r["lesion_id"] for r in r1.selected_by_group["TP"]]
    ids2 = [r["lesion_id"] for r in r2.selected_by_group["TP"]]
    assert ids1 != ids2  # nao e garantia estatistica geral, mas verdade para estes dados fabricados


def test_non_ok_status_rows_excluded_from_eligibility():
    rows = _make_rows(15)
    rows.append(_row("TPextra", "melanoma", "melanoma", status="falha"))
    result = select_gradcam_sample(rows, seed=42)
    assert result.n_eligible_by_group["TP"] == 15  # a linha com falha NAO conta
    selected_ids = [r["lesion_id"] for r in result.selected_by_group["TP"]]
    assert "TPextra" not in selected_ids


def test_shortfall_registered_when_fewer_than_n_per_cell():
    rows = _make_rows(15)
    rows = [r for r in rows if not (r["lesion_id"].startswith("FN") and r["lesion_id"] != "FN000")]  # so 1 FN sobra
    result = select_gradcam_sample(rows, seed=42, n_per_cell=10)
    assert result.n_eligible_by_group["FN"] == 1
    assert result.n_selected_by_group["FN"] == 1
    assert "FN" in result.shortfall_groups
    assert result.n_selected_by_group["TP"] == 10  # outras celulas nao afetadas


def test_candidates_sorted_by_lesion_id_before_sampling():
    """Mesmo conjunto de candidatos em ordem diferente de leitura deve
    produzir a MESMA selecao (ordenacao antes do sorteio elimina
    dependencia da ordem do CSV)."""
    rows_a = _make_rows(15)
    rows_b = list(reversed(rows_a))
    result_a = select_gradcam_sample(rows_a, seed=42)
    result_b = select_gradcam_sample(rows_b, seed=42)
    for group in ALL_GROUPS:
        ids_a = [r["lesion_id"] for r in result_a.selected_by_group[group]]
        ids_b = [r["lesion_id"] for r in result_b.selected_by_group[group]]
        assert ids_a == ids_b


def test_all_groups_present_even_when_fully_empty():
    result = select_gradcam_sample([], seed=42)
    for group in ALL_GROUPS:
        assert result.n_eligible_by_group[group] == 0
        assert result.n_selected_by_group[group] == 0
        assert group in result.shortfall_groups
