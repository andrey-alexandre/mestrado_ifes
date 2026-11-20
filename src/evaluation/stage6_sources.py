"""Etapa 6 (continuação) -- carregamento dos registros REAIS já produzidos
pelas execuções da matriz de 3.000/300 lesões, convertidos para o evento
normalizado consumido por `src.evaluation.stage6_operational_status`.

Nenhuma inferência é feita aqui: este módulo só LÊ artefatos já existentes
em disco (`call_log_main*.jsonl`, `s0_cnn_scores.csv`,
`s0_s1_s2_preparation_manifest.csv`, o reprocessamento oficial do Gemini) e
aplica a MESMA regra de seleção de tentativa oficial já usada pelo
executor da matriz (`src.multiagent.vlm_direct.is_call_record_reusable`),
nunca uma regra nova e paralela.

Seleção da tentativa oficial quando mais de um registro REAL (reaproveitável)
existe para a mesma chave (role, condition, lesion_id, image_id): a de
``attempt_timestamp_utc`` mais recente -- cega ao resultado (nunca por
diagnóstico favorável), documentada em ``multi_attempt_keys`` do relatório
de carregamento para rastreabilidade (protocolo, seção 1: "Não escolha
tentativas por resultado favorável").
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from src.evaluation.explicit_decision_contract import (
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISION_STATUS_OK,
)
from src.multiagent.vlm_direct import (
    STATUS_OK,
    STATUS_TECHNICAL_FAILURE,
    is_call_record_reusable,
)

CallKey = tuple[str, str, str, str]  # (role, condition, lesion_id, image_id)


def load_manifest_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def lesion_ids_by_category_from_manifest(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(row["dx"], []).append(row["lesion_id"])
    return by_category


def real_by_lesion_from_manifest(rows: list[dict[str, str]]) -> dict[str, str]:
    return {row["lesion_id"]: row["binary_target"] for row in rows}


# ---------------------------------------------------------------------------
# S0/S1/S2 structural availability (reason_s2 etc.) -- fonte única de
# verdade para "máscara vazia" / qualquer outra indisponibilidade
# estrutural de S2, nunca re-derivada de outro lugar.
# ---------------------------------------------------------------------------
@dataclass
class S2Availability:
    status_by_lesion: dict[str, str]  # lesion_id -> "disponivel" / outro
    reason_by_lesion: dict[str, str]  # lesion_id -> reason_s2 (pode ser "")


def load_s2_availability(s0_s1_s2_manifest_path: str | Path) -> S2Availability:
    rows = load_manifest_rows(s0_s1_s2_manifest_path)
    return S2Availability(
        status_by_lesion={row["lesion_id"]: row.get("status_s2", "") for row in rows},
        reason_by_lesion={row["lesion_id"]: row.get("reason_s2", "") for row in rows},
    )


# ---------------------------------------------------------------------------
# Call logs (Gemini/GPT/Qwen3-VL/MedGemma) -- JSONL, um registro por
# tentativa (histórico pode conter mais de uma tentativa por chave).
# ---------------------------------------------------------------------------
def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_call_log_rows(call_log_path: str | Path) -> list[dict[str, Any]]:
    return _read_jsonl(call_log_path)


def merge_gemini_reprocessed_decisao_final(
    original_rows: list[dict[str, Any]], reprocessing_jsonl_path: str | Path,
) -> list[dict[str, Any]]:
    """Sobrepõe, em cada linha de ``original_rows``, os campos
    ``decision_status``/``decisao_final``/``decision_contract_parser_version``
    pela versão REPROCESSADA (fonte oficial para Gemini, protocolo seção 1:
    "utilize a fonte oficial reprocessada de decisao_final, não o texto
    contaminado do log original"). ``call_status`` nunca é alterado pelo
    reprocessamento (reprocessing só recupera conteúdo textual de chamadas
    que já tiveram ``call_status == STATUS_OK``, nunca reclassifica uma
    falha como sucesso).

    Casamento feito pela chave completa (role, condition, lesion_id,
    image_id, attempt_timestamp_utc) -- a mesma granularidade de uma linha
    individual do log, não só da chave lógica (role,condition,lesion_id,
    image_id), para que reprocessamento de uma tentativa específica nunca
    seja aplicado por engano a outra tentativa da mesma chave lógica.
    """
    reprocessed_rows = _read_jsonl(reprocessing_jsonl_path)
    reprocessed_by_full_key: dict[tuple, dict[str, Any]] = {
        (r["role"], r["condition"], r["lesion_id"], r["image_id"], r["attempt_timestamp_utc"]): r
        for r in reprocessed_rows
    }

    merged: list[dict[str, Any]] = []
    for row in original_rows:
        full_key = (row["role"], row["condition"], row["lesion_id"], row["image_id"], row["attempt_timestamp_utc"])
        reprocessed = reprocessed_by_full_key.get(full_key)
        new_row = dict(row)
        if reprocessed is not None:
            if reprocessed.get("reprocessed_decision_status") is not None:
                new_row["decision_status"] = reprocessed["reprocessed_decision_status"]
            if reprocessed.get("reprocessed_decisao_final") is not None:
                new_row["decisao_final"] = reprocessed["reprocessed_decisao_final"]
            if reprocessed.get("reprocessed_decision_contract_parser_version") is not None:
                new_row["decision_contract_parser_version"] = reprocessed["reprocessed_decision_contract_parser_version"]
        merged.append(new_row)
    return merged


def load_gemini_recovered_text_by_full_key(reprocessing_jsonl_path: str | Path) -> dict[tuple, Optional[str]]:
    """``(role, condition, lesion_id, image_id, attempt_timestamp_utc) ->
    recovered_text`` a partir do reprocessamento oficial do Gemini --
    usado pela revisão qualitativa (``scripts/prepare_qualitative_review_
    materials.py``) para exibir o texto já recuperado da representação
    contaminada (lista de blocos em repr Python), em vez do
    ``raw_response_text`` bruto. ADITIVO: não é usado pela análise
    quantitativa (que só lê os campos ``reprocessed_decision_status``/
    ``reprocessed_decisao_final`` via ``merge_gemini_reprocessed_
    decisao_final``, nunca o texto em si)."""
    rows = _read_jsonl(reprocessing_jsonl_path)
    return {
        (r["role"], r["condition"], r["lesion_id"], r["image_id"], r["attempt_timestamp_utc"]): r.get("recovered_text")
        for r in rows
    }


@dataclass
class OfficialRecordSelection:
    record_by_key: dict[CallKey, dict[str, Any]]
    multi_attempt_keys: dict[CallKey, list[str]] = field(default_factory=dict)  # key -> timestamps (all reusable)


def select_official_call_records(rows: list[dict[str, Any]], model_name: str) -> OfficialRecordSelection:
    """Agrupa ``rows`` (já carregadas, 1 ou mais arquivos concatenados) por
    chave lógica e seleciona UMA tentativa oficial por chave, reusando
    ``is_call_record_reusable`` (mesma regra de elegibilidade do executor
    da matriz) -- nunca uma heurística nova. Entre múltiplas tentativas
    reaproveitáveis para a mesma chave, a de timestamp mais recente vence
    (cego ao resultado); todas as chaves com mais de uma tentativa real
    reaproveitável são registradas em ``multi_attempt_keys`` para auditoria."""
    by_key: dict[CallKey, list[dict[str, Any]]] = {}
    for row in rows:
        if not is_call_record_reusable(row, model_name):
            continue
        key: CallKey = (row["role"], row["condition"], row["lesion_id"], row["image_id"])
        by_key.setdefault(key, []).append(row)

    record_by_key: dict[CallKey, dict[str, Any]] = {}
    multi_attempt_keys: dict[CallKey, list[str]] = {}
    for key, candidates in by_key.items():
        candidates_sorted = sorted(candidates, key=lambda r: r["attempt_timestamp_utc"])
        record_by_key[key] = candidates_sorted[-1]
        if len(candidates_sorted) > 1:
            multi_attempt_keys[key] = [c["attempt_timestamp_utc"] for c in candidates_sorted]

    return OfficialRecordSelection(record_by_key=record_by_key, multi_attempt_keys=multi_attempt_keys)


# ---------------------------------------------------------------------------
# CNN (s0_cnn_scores.csv) -- fonte não-call-log, adaptada para o mesmo
# formato de evento consumido por `stage6_operational_status
# .classify_operational_status` (call_status/decision_status/decisao_final
# sintéticos, reusando o MESMO vocabulário, nunca um caminho de
# classificação paralelo).
# ---------------------------------------------------------------------------
def load_cnn_s0_records_as_synthetic_call_records(csv_path: str | Path) -> dict[str, dict[str, Any]]:
    """``lesion_id -> evento sintético`` a partir de
    ``experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv``
    (colunas: lesion_id,image_id,dx,true_binary_target,status,
    predicted_score,predicted_binary_target,error). ``status != "ok"``
    (erro de inferência da CNN para aquela imagem) mapeia para
    STATUS_TECHNICAL_FAILURE -- nunca descartado da contagem, nunca
    convertido em não-melanoma."""
    records: dict[str, dict[str, Any]] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            lesion_id = row["lesion_id"]
            if row["status"] == "ok":
                predicted = row["predicted_binary_target"]
                decisao_final = DECISAO_MELANOMA if predicted == "melanoma" else DECISAO_NAO_MELANOMA
                records[lesion_id] = {
                    "call_status": STATUS_OK,
                    "decision_status": DECISION_STATUS_OK,
                    "decisao_final": decisao_final,
                    "failure_cause": "",
                }
            else:
                records[lesion_id] = {
                    "call_status": STATUS_TECHNICAL_FAILURE,
                    "decision_status": None,
                    "decisao_final": None,
                    "failure_cause": row.get("error", row["status"]),
                }
    return records


# ---------------------------------------------------------------------------
# Montagem final: para um sistema/condição/papel, devolve
# lesion_id -> evento normalizado (ou ausente), combinando os três sinais
# (registro oficial, disponibilidade estrutural de S2, universo de
# lesões do manifesto) -- usado por
# `src.evaluation.stage6_operational_status.classify_operational_status`.
# ---------------------------------------------------------------------------
def build_lesion_events_for_role_condition(
    official_selection: OfficialRecordSelection,
    role: str,
    condition: str,
    manifest_rows: list[dict[str, str]],
) -> dict[str, Optional[dict[str, Any]]]:
    """``lesion_id -> registro oficial (ou None)`` para TODAS as lesões do
    manifesto, no (role, condition) dado -- nunca restrito só às lesões
    com registro (lesões sem registro aparecem com valor ``None``, para
    que o chamador sempre decida explicitamente entre "ausência de
    execução" e "máscara vazia" via `s2_availability`)."""
    events: dict[str, Optional[dict[str, Any]]] = {}
    for row in manifest_rows:
        lesion_id, image_id = row["lesion_id"], row["image_id"]
        key: CallKey = (role, condition, lesion_id, image_id)
        events[lesion_id] = official_selection.record_by_key.get(key)
    return events
