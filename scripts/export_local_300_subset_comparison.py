"""Exportação pareada, por `lesion_id`, da comparação complementar sobre
o subconjunto de 300 lesões (ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §15) entre:
  - CNN em S0 (`experiments/cnn_baseline/s0_ham10000_eval/s0_cnn_scores.csv`,
    já avaliada -- NUNCA rechamada);
  - Gemini e GPT, multiagente em S2 (fonte oficial do Gemini é o arquivo
    REPROCESSADO de `decisao_final`,
    `call_log_main.decisao_final_reprocessed.jsonl` -- NUNCA o log
    original, que mantém `decision_status: "field_absent"` para
    `vlm_direto`/`summary`; GPT lido do `call_log_main.jsonl` do escopo
    `full` quando existir);
  - Qwen3-VL e MedGemma 1.5 4B, multiagente em S2 (lidos da UNIÃO do log
    do escopo `local_s2_subset` com o log pré-existente do escopo `full`
    -- mesma política de reaproveitamento do executor).

Este script NÃO gera nenhuma chamada nova a modelo/API -- é uma
exportação pura de leitura. Para QUALQUER sistema sem resultado
registrado para uma lesão, a célula correspondente é marcada
explicitamente `"pendente"` -- nunca omitida, nunca inventada, nunca
convertida silenciosamente em "falha" ou "negativo".

IMPORTANTE (decisão do pesquisador): esta comparação restringe TODOS os
5 sistemas ao MESMO subconjunto de 300 lesões -- nunca mistura métricas
locais de 300 com métricas de outros sistemas calculadas nas 3.000.
Quem quiser a comparação principal nas 3.000 (Gemini/GPT/CNN) usa outra
fonte -- este script não produz isso.

Preserva a distinção entre: resposta válida (`call_status == "ok"`),
decisão binária (`melanoma`/`nao_melanoma`), indeterminação
(`indeterminado` -- NUNCA tratado como negativo), falha
(`falha_tecnica`/`recusa_explicita`/`sem_resposta`), e ausência de
execução (`nao_executado`/nenhum registro -- "pendente").

A regra pendente para comparar as 5 métricas binárias quando os
sistemas produzem decisões sobre subconjuntos diferentes (indeterminado/
falha num sistema, binário no outro) NÃO é resolvida aqui -- este
script só exporta os dados pareados com a classificação completa
preservada; qualquer cálculo de métrica posterior precisa dessa decisão
do pesquisador, não um default de código.

STATUS: implementado, NÃO executado pelo assistente que o escreveu --
aguardando execução do pesquisador.

Uso:
    python scripts/export_local_300_subset_comparison.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SUBSET_MANIFEST = str(REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset" / "local_300_s2_subset_manifest.csv")
DEFAULT_CNN_S0_SCORES_CSV = str(REPO_ROOT / "experiments" / "cnn_baseline" / "s0_ham10000_eval" / "s0_cnn_scores.csv")
DEFAULT_FULL_SCALE_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix")
DEFAULT_LOCAL_300_MATRIX_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_matrix")
DEFAULT_GEMINI_REPROCESSED_LOG = str(
    REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix" / "gemini-3.1-flash-lite"
    / "reprocessing" / "call_log_main.decisao_final_reprocessed.jsonl"
)
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_300_s2_subset_comparison")

GPT_MODEL_NAME = "gpt-5.4-nano"
QWEN_MODEL_NAME = "qwen3-vl:8b-instruct-q4_K_M"
MEDGEMMA_MODEL_NAME = "medgemma1.5:4b-it-q4_K_M"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"

PENDING = "pendente"
NOT_APPLICABLE = "nao_aplicavel"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset-manifest", default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--cnn-s0-scores-csv", default=DEFAULT_CNN_S0_SCORES_CSV)
    parser.add_argument("--full-scale-dir", default=DEFAULT_FULL_SCALE_DIR)
    parser.add_argument("--local-300-matrix-dir", default=DEFAULT_LOCAL_300_MATRIX_DIR)
    parser.add_argument("--gemini-reprocessed-log", default=DEFAULT_GEMINI_REPROCESSED_LOG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pick_final_record_per_key(
    records: list[dict[str, Any]], role: str, condition: str, lesion_ids: set[str]
) -> dict[str, dict[str, Any]]:
    """Mesma regra de seleção de `scripts/reprocess_decisao_final.py::
    reconcile_historical_vs_final` (nunca reimplementada com lógica
    diferente): dentre as tentativas reais (`call_status !=
    "nao_executado"`) de uma mesma chave, prefere a última `ok`
    (comprovadamente única por chave -- nenhuma chamada bem-sucedida é
    repetida neste pipeline); na ausência de `ok`, a última tentativa
    real por ordem de `attempt_timestamp_utc`; sem nenhuma tentativa
    real, a chave simplesmente não aparece no resultado (pendente)."""
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("role") != role or r.get("condition") != condition:
            continue
        lid = r.get("lesion_id")
        if lid not in lesion_ids:
            continue
        by_key.setdefault(lid, []).append(r)

    final_by_lesion: dict[str, dict[str, Any]] = {}
    for lid, recs in by_key.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("attempt_timestamp_utc") or "")
        real_attempts = [r for r in recs_sorted if r.get("call_status") != "nao_executado"]
        if not real_attempts:
            continue
        ok_attempts = [r for r in real_attempts if r.get("call_status") == "ok"]
        final_by_lesion[lid] = ok_attempts[-1] if ok_attempts else real_attempts[-1]
    return final_by_lesion


def _classify_multiagent_cell(record: Optional[dict[str, Any]], decisao_final_field: str, decision_status_field: str) -> dict[str, Any]:
    if record is None:
        return {"status": PENDING, "call_status": None, "decisao_final": None}
    call_status = record.get("call_status")
    if call_status != "ok":
        return {"status": call_status, "call_status": call_status, "decisao_final": None}
    decisao_final = record.get(decisao_final_field)
    decision_status = record.get(decision_status_field)
    if decision_status != "ok" or decisao_final is None:
        return {"status": "resposta_valida_sem_decisao_contrato", "call_status": call_status, "decisao_final": None, "decision_status": decision_status}
    return {"status": decisao_final, "call_status": call_status, "decisao_final": decisao_final}


def _classify_cnn_cell(row: Optional[dict[str, str]]) -> dict[str, Any]:
    if row is None:
        return {"status": PENDING, "predicted_binary_target": None, "predicted_score": None}
    if row.get("status") != "ok":
        return {"status": row.get("status") or "falha", "predicted_binary_target": None, "predicted_score": None, "error": row.get("error")}
    return {
        "status": row.get("predicted_binary_target"),
        "predicted_binary_target": row.get("predicted_binary_target"),
        "predicted_score": row.get("predicted_score"),
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset_manifest_path = Path(args.subset_manifest)
    if not subset_manifest_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {subset_manifest_path} não encontrado -- rode a seleção do subconjunto de 300 primeiro.")
    with open(subset_manifest_path, encoding="utf-8") as f:
        subset_rows = list(csv.DictReader(f))
    if len(subset_rows) != 300:
        raise SystemExit(f"ERRO OPERACIONAL: {subset_manifest_path} tem {len(subset_rows)} linhas, esperado exatamente 300.")
    lesion_ids = {r["lesion_id"] for r in subset_rows}
    lesion_meta = {r["lesion_id"]: r for r in subset_rows}
    print(f"Subconjunto de 300 lesões carregado: {len(lesion_ids)} lesion_id distintos.")

    # --- CNN S0 (já avaliada, nunca rechamada -- só filtra pelo subconjunto) ---
    cnn_csv_path = Path(args.cnn_s0_scores_csv)
    if not cnn_csv_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {cnn_csv_path} não encontrado.")
    with open(cnn_csv_path, encoding="utf-8") as f:
        cnn_rows_all = list(csv.DictReader(f))
    cnn_by_lesion = {r["lesion_id"]: r for r in cnn_rows_all if r["lesion_id"] in lesion_ids}
    print(f"CNN S0: {len(cnn_by_lesion)}/300 lesões encontradas no CSV já existente.")

    # --- Gemini (fonte OFICIAL: arquivo reprocessado, nunca o log original) ---
    gemini_reprocessed_path = Path(args.gemini_reprocessed_log)
    gemini_records = _load_jsonl(gemini_reprocessed_path)
    if not gemini_records:
        print(f"AVISO: {gemini_reprocessed_path} vazio/ausente -- Gemini ficará 'pendente' em todas as 300 lesões.")
    # No arquivo reprocessado, `original_call_status` é o status real da
    # chamada; `reprocessed_decision_status`/`reprocessed_decisao_final`
    # são a fonte oficial do veredito -- adaptamos os nomes de campo
    # esperados por `_pick_final_record_per_key`/`_classify_multiagent_cell`
    # criando uma visão com os nomes padronizados, sem alterar o arquivo.
    gemini_records_normalized = [
        {**r, "call_status": r.get("original_call_status"), "decisao_final": r.get("reprocessed_decisao_final"), "decision_status": r.get("reprocessed_decision_status")}
        for r in gemini_records
    ]
    gemini_final = _pick_final_record_per_key(gemini_records_normalized, role="summary", condition="s2", lesion_ids=lesion_ids)
    print(f"Gemini (multiagente S2, fonte reprocessada): {len(gemini_final)}/300 lesões com tentativa real registrada.")

    # --- GPT (escopo full, se já tiver rodado) ---
    gpt_log_path = Path(args.full_scale_dir) / GPT_MODEL_NAME.replace(":", "_").replace("/", "_") / "call_log_main.jsonl"
    gpt_records = _load_jsonl(gpt_log_path)
    gpt_final = _pick_final_record_per_key(gpt_records, role="summary", condition="s2", lesion_ids=lesion_ids)
    print(f"GPT (multiagente S2): {len(gpt_final)}/300 lesões com tentativa real registrada ({gpt_log_path} {'encontrado' if gpt_records else 'ausente/vazio'}).")

    # --- Qwen3-VL / MedGemma (união do escopo local_s2_subset com o full, read-only) ---
    def _local_model_final(model_name: str) -> dict[str, dict[str, Any]]:
        slug = model_name.replace(":", "_").replace("/", "_")
        local_log = Path(args.local_300_matrix_dir) / slug / "call_log_main.jsonl"
        full_log = Path(args.full_scale_dir) / slug / "call_log_main.jsonl"
        records = _load_jsonl(local_log) + _load_jsonl(full_log)
        return _pick_final_record_per_key(records, role="summary", condition="s2", lesion_ids=lesion_ids)

    qwen_final = _local_model_final(QWEN_MODEL_NAME)
    print(f"Qwen3-VL (multiagente S2): {len(qwen_final)}/300 lesões com tentativa real registrada (união local_s2_subset + full).")
    medgemma_final = _local_model_final(MEDGEMMA_MODEL_NAME)
    print(f"MedGemma 1.5 4B (multiagente S2): {len(medgemma_final)}/300 lesões com tentativa real registrada (união local_s2_subset + full).")

    # --- Monta a exportação pareada, uma linha por lesão ---
    export_rows: list[dict[str, Any]] = []
    for lid in sorted(lesion_ids):
        meta = lesion_meta[lid]
        cnn_cell = _classify_cnn_cell(cnn_by_lesion.get(lid))
        gemini_cell = _classify_multiagent_cell(gemini_final.get(lid), "decisao_final", "decision_status")
        gpt_cell = _classify_multiagent_cell(gpt_final.get(lid), "decisao_final", "decision_status")
        qwen_cell = _classify_multiagent_cell(qwen_final.get(lid), "decisao_final", "decision_status")
        medgemma_cell = _classify_multiagent_cell(medgemma_final.get(lid), "decisao_final", "decision_status")

        export_rows.append({
            "lesion_id": lid,
            "image_id": meta["image_id"],
            "dx": meta["dx"],
            "binary_target": meta["binary_target"],
            "cnn_s0_status": cnn_cell["status"],
            "cnn_s0_predicted_score": cnn_cell.get("predicted_score"),
            "gemini_multiagent_s2_status": gemini_cell["status"],
            "gpt_multiagent_s2_status": gpt_cell["status"],
            "qwen3vl_multiagent_s2_status": qwen_cell["status"],
            "medgemma_multiagent_s2_status": medgemma_cell["status"],
        })

    export_csv_path = output_dir / "local_300_s2_subset_comparison.csv"
    fieldnames = list(export_rows[0].keys())
    with open(export_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)
    print(f"\nExportação pareada (300 linhas) gravada em: {export_csv_path}")

    def _status_counts(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in export_rows:
            v = str(row[key])
            counts[v] = counts.get(v, 0) + 1
        return counts

    summary = {
        "n_lesions": len(export_rows),
        "status_counts_by_system": {
            "cnn_s0": _status_counts("cnn_s0_status"),
            "gemini_multiagent_s2": _status_counts("gemini_multiagent_s2_status"),
            "gpt_multiagent_s2": _status_counts("gpt_multiagent_s2_status"),
            "qwen3vl_multiagent_s2": _status_counts("qwen3vl_multiagent_s2_status"),
            "medgemma_multiagent_s2": _status_counts("medgemma_multiagent_s2_status"),
        },
        "note": (
            "status possiveis por sistema: 'melanoma'/'nao_melanoma' (decisao binaria valida), "
            "'indeterminado' (decisao valida, FORA das 5 metricas binarias), 'falha_tecnica'/"
            "'recusa_explicita'/'sem_resposta' (tentativa real sem decisao), 'pendente' (nenhuma "
            "tentativa real registrada ainda -- NUNCA tratado como negativo/falha). Para CNN, "
            "status e o predicted_binary_target quando status=='ok' no CSV original, ou o status "
            "de erro do CSV quando diferente de 'ok'. A regra para comparar as 5 metricas quando "
            "os sistemas tem subconjuntos binarios diferentes NAO foi resolvida aqui -- decisao "
            "pendente do pesquisador."
        ),
    }
    summary_path = output_dir / "local_300_s2_subset_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Resumo por status/sistema gravado em: {summary_path}")
    for system, counts in summary["status_counts_by_system"].items():
        print(f"  {system}: {counts}")


if __name__ == "__main__":
    main()
