from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.qualitative_review_schema import (  # noqa: E402
    CRITERIA,
    MODEL_REAL_LABELS,
    NAO_AVALIAVEL,
    PENDENTE,
    PROBLEM_CRITERIA,
    build_neutral_model_id_mapping,
    build_pending_review_rows,
)


def test_exactly_10_problem_criteria_plus_1_observation_marker():
    assert len(PROBLEM_CRITERIA) == 10
    assert len(CRITERIA) == 11
    non_problem = [c for c in CRITERIA if not c.is_problem]
    assert len(non_problem) == 1
    assert non_problem[0].criterion_id == "observacao_hipotese_propria_identificada"


def test_neutral_mapping_is_bijection_over_all_models():
    mapping = build_neutral_model_id_mapping()
    assert set(mapping.keys()) == set(MODEL_REAL_LABELS)
    assert set(mapping.values()) == {"Modelo A", "Modelo B", "Modelo C", "Modelo D"}
    assert len(set(mapping.values())) == 4


def test_neutral_mapping_deterministic_by_seed():
    m1 = build_neutral_model_id_mapping(seed=123)
    m2 = build_neutral_model_id_mapping(seed=123)
    assert m1 == m2


def test_pending_rows_text_unavailable_defaults_to_nao_avaliavel_with_auto_comment():
    rows = build_pending_review_rows("L1", "Modelo A", text_available=False)
    assert len(rows) == 11
    for row in rows:
        assert row["status"] == NAO_AVALIAVEL
        assert "auto" in row["reviewer_comment"]
        assert row["reviewer_id"] == ""
        assert row["review_date"] == ""


def test_pending_rows_text_available_leaves_status_empty_for_human_judgment():
    rows = build_pending_review_rows("L1", "Modelo A", text_available=True)
    for row in rows:
        assert row["status"] == PENDENTE
        assert row["reviewer_comment"] == ""
