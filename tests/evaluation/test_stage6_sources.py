from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.explicit_decision_contract import PARSER_VERSION  # noqa: E402
from src.evaluation.stage6_sources import (  # noqa: E402
    build_lesion_events_for_role_condition,
    load_cnn_s0_records_as_synthetic_call_records,
    merge_gemini_reprocessed_decisao_final,
    select_official_call_records,
)

MANIFEST_ROWS = [
    {"lesion_id": "L1", "image_id": "IMG1", "dx": "nv", "binary_target": "nao_melanoma"},
    {"lesion_id": "L2", "image_id": "IMG2", "dx": "mel", "binary_target": "melanoma"},
]


def _row(lesion_id, image_id, role="summary", condition="s0", call_status="ok",
         decision_status="ok", decisao_final="indeterminado", ts="2026-01-01T00:00:00Z",
         model_name="gemini-3.1-flash-lite", decision_contract_parser_version=PARSER_VERSION):
    return {
        "model_name": model_name, "role": role, "condition": condition,
        "lesion_id": lesion_id, "image_id": image_id, "attempt_timestamp_utc": ts,
        "call_status": call_status, "failure_cause": "", "decision_status": decision_status,
        "decisao_final": decisao_final, "decision_contract_parser_version": decision_contract_parser_version,
        "call_mechanism_version": None,
    }


def test_select_official_picks_latest_timestamp_among_multiple_real_attempts():
    rows = [
        _row("L1", "IMG1", ts="2026-01-01T00:00:00Z", decisao_final="melanoma"),
        _row("L1", "IMG1", ts="2026-01-02T00:00:00Z", decisao_final="nao_melanoma"),  # mais recente -> oficial
    ]
    selection = select_official_call_records(rows, "gemini-3.1-flash-lite")
    key = ("summary", "s0", "L1", "IMG1")
    assert selection.record_by_key[key]["decisao_final"] == "nao_melanoma"
    assert key in selection.multi_attempt_keys
    assert len(selection.multi_attempt_keys[key]) == 2


def test_select_official_filters_by_model_name_exact_match():
    rows = [_row("L1", "IMG1", model_name="gpt-5.4-nano")]
    selection = select_official_call_records(rows, "gemini-3.1-flash-lite")
    assert selection.record_by_key == {}


def test_select_official_excludes_nao_executado():
    rows = [_row("L1", "IMG1", call_status="nao_executado", decision_status=None, decisao_final=None,
                 decision_contract_parser_version=None)]
    selection = select_official_call_records(rows, "gemini-3.1-flash-lite")
    assert selection.record_by_key == {}


def test_build_lesion_events_includes_none_for_lesions_without_record():
    rows = [_row("L1", "IMG1")]
    selection = select_official_call_records(rows, "gemini-3.1-flash-lite")
    events = build_lesion_events_for_role_condition(selection, "summary", "s0", MANIFEST_ROWS)
    assert events["L1"] is not None
    assert events["L2"] is None


def test_merge_gemini_reprocessed_overrides_only_decision_fields(tmp_path):
    original = [_row("L1", "IMG1", decision_status="field_absent", decisao_final=None)]
    reprocessed = [{
        "role": "summary", "condition": "s0", "lesion_id": "L1", "image_id": "IMG1",
        "attempt_timestamp_utc": "2026-01-01T00:00:00Z",
        "reprocessed_decision_status": "ok", "reprocessed_decisao_final": "indeterminado",
        "reprocessed_decision_contract_parser_version": "v2",
    }]
    reprocessed_path = tmp_path / "reprocessed.jsonl"
    reprocessed_path.write_text(json.dumps(reprocessed[0]) + "\n", encoding="utf-8")

    merged = merge_gemini_reprocessed_decisao_final(original, reprocessed_path)
    assert merged[0]["decision_status"] == "ok"
    assert merged[0]["decisao_final"] == "indeterminado"
    assert merged[0]["decision_contract_parser_version"] == "v2"
    assert merged[0]["call_status"] == "ok"  # nunca alterado pelo reprocessamento


def test_cnn_synthetic_records_map_status_and_failure(tmp_path):
    csv_path = tmp_path / "s0_cnn_scores.csv"
    csv_path.write_text(
        "lesion_id,image_id,dx,true_binary_target,status,predicted_score,predicted_binary_target,error\n"
        "L1,IMG1,nv,nao_melanoma,ok,0.1,nao_melanoma,\n"
        "L2,IMG2,mel,melanoma,ok,0.9,melanoma,\n"
        "L3,IMG3,nv,nao_melanoma,erro,,,imagem_corrompida\n",
        encoding="utf-8",
    )
    records = load_cnn_s0_records_as_synthetic_call_records(csv_path)
    assert records["L1"]["decisao_final"] == "nao_melanoma"
    assert records["L2"]["decisao_final"] == "melanoma"
    assert records["L3"]["call_status"] == "falha_tecnica"
    assert records["L3"]["decisao_final"] is None
