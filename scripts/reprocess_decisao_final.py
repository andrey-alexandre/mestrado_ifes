"""Reprocessamento, sem nenhuma chamada nova a API/modelo, das decisões
`decisao_final` de um `call_log_main.jsonl` já salvo -- corrige a
REPRESENTAÇÃO do texto (bug real do Gemini, 2026-11-18: `.content`
devolvido como lista de blocos, corrompido por `str(lista)` em algum
ponto anterior à correção), reaplicando o extrator oficial
`src.evaluation.explicit_decision_contract.extract_decisao_final` SEM
NENHUMA mudança de regra -- isto é uma correção de como o texto chega
ao extrator, nunca uma mudança do contrato diagnóstico.

Preserva o arquivo original intocado -- grava um arquivo DERIVADO
separado (nunca sobrescreve `call_log_main.jsonl`). Cada registro
derivado inclui: identificação da chamada (role/condition/lesion_id/
image_id/attempt_timestamp_utc/model_name), status/decisão ANTERIOR,
status de recuperação do texto, texto recuperado (quando aplicável), e
status/decisão REPROCESSADOS -- nunca um dos dois escondido ou
sobrescrito pelo outro.

Além da recuperação, reconcilia duas contagens que NÃO são a mesma
coisa e não devem ser somadas ingenuamente: (1) TENTATIVAS HISTÓRICAS
(toda linha do `.jsonl`, incluindo tentativas supersedidas por uma
invocação posterior) e (2) RESULTADO FINAL POR CHAVE PLANEJADA (um
único veredito por `(role, condition, lesion_id, image_id)` -- deve
somar exatamente ao número de chamadas planejadas). Chaves com MAIS DE
UMA tentativa real (não `nao_executado`) são reportadas explicitamente,
nunca silenciosamente colapsadas -- isso inclui as tentativas do
mecanismo antigo que foram corretamente re-tentadas sob o mecanismo de
cancelamento corrigido (ver `is_call_record_reusable` ponto 4,
docs/reconstruction/30_real_lesion_evaluation_protocol.md §10): essa é
uma EXCEÇÃO OPERACIONAL documentada à regra de uma tentativa, não um
erro a esconder.

STATUS: implementado, NÃO executado pelo assistente que o escreveu --
aguardando execução do pesquisador.

Uso:
    python scripts/reprocess_decisao_final.py \\
        --call-log experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/call_log_main.jsonl \\
        --output-dir experiments/multiagent/full_scale_matrix/gemini-3.1-flash-lite/reprocessing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.content_recovery import (  # noqa: E402
    CONTENT_RECOVERY_VERSION,
    RECOVERED_FROM_LIST_REPR,
    RECOVERY_NOT_APPLICABLE,
    RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT,
    recover_raw_response_text,
)
from src.evaluation.explicit_decision_contract import (  # noqa: E402
    DECISION_STATUS_OK,
    PARSER_VERSION as DECISION_CONTRACT_PARSER_VERSION,
    extract_decisao_final,
)

REPROCESSING_VERSION = "decisao_final_reprocess_v1_2026_11_18"
ROLES_WITH_DECISION_CONTRACT = ("vlm_direto", "summary")
STATUS_NOT_EXECUTED = "nao_executado"
STATUS_OK = "ok"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--call-log", required=True, help="call_log_main.jsonl original -- NUNCA modificado por este script.")
    parser.add_argument("--output-dir", required=True, help="Diretório onde gravar os arquivos derivados (criado se não existir).")
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Parte 1: recuperação de decisao_final para registros vlm_direto/summary
# com call_status == "ok" (mesmo gate do extrator original -- nunca
# estendido para falha_tecnica/sem_resposta/recusa_explicita, que nunca
# tiveram decisao_final calculado por desenho).
# ---------------------------------------------------------------------------
def _identification(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": record.get("model_name"),
        "role": record.get("role"),
        "condition": record.get("condition"),
        "lesion_id": record.get("lesion_id"),
        "image_id": record.get("image_id"),
        "attempt_timestamp_utc": record.get("attempt_timestamp_utc"),
    }


def reprocess_record(record: dict[str, Any]) -> dict[str, Any]:
    out = _identification(record)
    out["original_call_status"] = record.get("call_status")
    out["original_decision_status"] = record.get("decision_status")
    out["original_decisao_final"] = record.get("decisao_final")
    out["original_decision_contract_parser_version"] = record.get("decision_contract_parser_version")
    out["reprocessing_version"] = REPROCESSING_VERSION
    out["content_recovery_version"] = CONTENT_RECOVERY_VERSION

    applicable = record.get("role") in ROLES_WITH_DECISION_CONTRACT and record.get("call_status") == STATUS_OK
    if not applicable:
        out.update({
            "recovery_status": RECOVERY_NOT_APPLICABLE,
            "recovered_text": None,
            "reextraction_attempted": False,
            "reprocessed_decision_status": record.get("decision_status"),
            "reprocessed_decisao_final": record.get("decisao_final"),
            "reprocessed_decision_contract_parser_version": record.get("decision_contract_parser_version"),
        })
        return out

    recovery = recover_raw_response_text(record.get("raw_response_text"))
    out["recovery_status"] = recovery.recovery_status
    out["recovered_text"] = recovery.recovered_text
    out["recovery_parse_error_detail"] = recovery.parse_error_detail

    if recovery.recovery_status in (RECOVERED_FROM_LIST_REPR, RECOVERY_NOT_NEEDED_ALREADY_PLAIN_TEXT):
        decision = extract_decisao_final(recovery.recovered_text)
        out["reextraction_attempted"] = True
        out["reprocessed_decision_status"] = decision.decision_status
        out["reprocessed_decisao_final"] = decision.decisao_final
        out["reprocessed_decision_contract_parser_version"] = (
            DECISION_CONTRACT_PARSER_VERSION if decision.decision_status == DECISION_STATUS_OK else None
        )
    else:
        # IRRECOVERABLE_* -- nunca reextrai sobre um valor None/parcial;
        # mantém o status original, explicitamente sinalizado como
        # irrecuperável no campo `recovery_status` (nunca escondido).
        out["reextraction_attempted"] = False
        out["reprocessed_decision_status"] = record.get("decision_status")
        out["reprocessed_decisao_final"] = record.get("decisao_final")
        out["reprocessed_decision_contract_parser_version"] = record.get("decision_contract_parser_version")

    return out


# ---------------------------------------------------------------------------
# Parte 2: reconciliação tentativas históricas x resultado final por chave
# planejada (role, condition, lesion_id, image_id) -- nunca soma linhas
# históricas como se fossem contagem de resultado final.
# ---------------------------------------------------------------------------
def _call_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (record.get("role"), record.get("condition"), record.get("lesion_id"), record.get("image_id"))


def reconcile_historical_vs_final(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for r in records:
        by_key.setdefault(_call_key(r), []).append(r)

    final_status_counts: dict[str, int] = {}
    multi_real_attempt_keys: list[dict[str, Any]] = []
    exception_old_mechanism_retried: list[dict[str, Any]] = []
    historical_duplicate_bug_pattern: list[dict[str, Any]] = []
    unexplained_duplicate_same_mechanism: list[dict[str, Any]] = []

    for key, recs in by_key.items():
        recs_sorted = sorted(recs, key=lambda r: r.get("attempt_timestamp_utc") or "")
        real_attempts = [r for r in recs_sorted if r.get("call_status") != STATUS_NOT_EXECUTED]

        if real_attempts:
            ok_attempts = [r for r in real_attempts if r.get("call_status") == STATUS_OK]
            final = ok_attempts[-1] if ok_attempts else real_attempts[-1]
        else:
            final = None

        final_status = final.get("call_status") if final is not None else "pending_no_real_attempt_yet"
        final_status_counts[final_status] = final_status_counts.get(final_status, 0) + 1

        if len(real_attempts) > 1:
            entry = {
                "role": key[0], "condition": key[1], "lesion_id": key[2], "image_id": key[3],
                "n_real_attempts": len(real_attempts),
                "statuses_in_order": [r.get("call_status") for r in real_attempts],
                "mechanism_versions_in_order": [r.get("call_mechanism_version") for r in real_attempts],
            }
            multi_real_attempt_keys.append(entry)

            mechanisms = [r.get("call_mechanism_version") for r in real_attempts]
            if any(m is not None for m in mechanisms) and any(m is None for m in mechanisms):
                exception_old_mechanism_retried.append(entry)
            elif all(m is None for m in mechanisms):
                historical_duplicate_bug_pattern.append(entry)
            else:
                # Todas as tentativas reais sob o MESMO mecanismo
                # (não-nulo) -- NÃO se encaixa em nenhuma das duas
                # categorias explicadas (nem bug histórico antigo, nem
                # retomada legítima pós-correção). Isto seria um retry
                # genuíno não documentado -- nunca descartado/ignorado
                # silenciosamente; sinalizado para investigação manual.
                unexplained_duplicate_same_mechanism.append(entry)

    n_historical_lines = len(records)
    n_unique_planned_keys = len(by_key)

    return {
        "n_historical_log_lines": n_historical_lines,
        "n_unique_planned_keys": n_unique_planned_keys,
        "final_result_per_key_counts": final_status_counts,
        "final_result_counts_sum_check": sum(final_status_counts.values()),
        "n_keys_with_multiple_real_attempts": len(multi_real_attempt_keys),
        "note_on_multiple_attempts": (
            "Chaves com mais de uma tentativa REAL (nao_executado nao conta) nao sao erro "
            "por padrao -- duas categorias distintas, NUNCA colapsadas: "
            "(a) exception_old_mechanism_retried: falha do mecanismo antigo de "
            "cancelamento (call_mechanism_version ausente) seguida de uma tentativa real "
            "sob o mecanismo corrigido -- EXCECAO OPERACIONAL DOCUMENTADA a regra de uma "
            "tentativa (ver doc 30 secao 10, ponto 4 de is_call_record_reusable), nunca um "
            "retry disfarcado da MESMA tentativa; "
            "(b) historical_duplicate_bug_pattern: as duas tentativas sao do mecanismo "
            "antigo (ambas sem call_mechanism_version) -- produzidas pelo bug, ja corrigido, "
            "de decision_contract_parser_version exigido mesmo para falhas (ver doc 30 secao 9); "
            "(c) unexplained_duplicate_same_mechanism: todas as tentativas reais sob o MESMO "
            "mecanismo (nao-nulo) -- NAO se encaixa em (a) nem (b), seria um retry genuino nao "
            "documentado; reportado para investigacao manual, nunca descartado silenciosamente."
        ),
        "exception_old_mechanism_retried": exception_old_mechanism_retried,
        "n_exception_old_mechanism_retried": len(exception_old_mechanism_retried),
        "historical_duplicate_bug_pattern": historical_duplicate_bug_pattern,
        "n_historical_duplicate_bug_pattern": len(historical_duplicate_bug_pattern),
        "unexplained_duplicate_same_mechanism": unexplained_duplicate_same_mechanism,
        "n_unexplained_duplicate_same_mechanism": len(unexplained_duplicate_same_mechanism),
    }


# ---------------------------------------------------------------------------
# Relatório único consolidado
# ---------------------------------------------------------------------------
def build_report(
    reprocessed_records: list[dict[str, Any]],
    reconciliation: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    applicable = [r for r in reprocessed_records if r["recovery_status"] != RECOVERY_NOT_APPLICABLE]

    by_recovery_status: dict[str, int] = {}
    for r in applicable:
        by_recovery_status[r["recovery_status"]] = by_recovery_status.get(r["recovery_status"], 0) + 1

    by_reprocessed_decision_status: dict[str, int] = {}
    by_reprocessed_decisao_final: dict[str, int] = {}
    by_condition: dict[str, dict[str, int]] = {}
    by_role: dict[str, dict[str, int]] = {}
    n_changed_from_absent_to_ok = 0

    for r in applicable:
        ds = r["reprocessed_decision_status"]
        by_reprocessed_decision_status[ds] = by_reprocessed_decision_status.get(ds, 0) + 1
        if ds == DECISION_STATUS_OK:
            df = r["reprocessed_decisao_final"]
            by_reprocessed_decisao_final[df] = by_reprocessed_decisao_final.get(df, 0) + 1

        cond = r.get("condition") or "desconhecida"
        by_condition.setdefault(cond, {}).setdefault(ds, 0)
        by_condition[cond][ds] += 1

        role = r.get("role") or "desconhecido"
        by_role.setdefault(role, {}).setdefault(ds, 0)
        by_role[role][ds] += 1

        if r["original_decision_status"] != DECISION_STATUS_OK and ds == DECISION_STATUS_OK:
            n_changed_from_absent_to_ok += 1

    return {
        "model_name": model_name,
        "reprocessing_version": REPROCESSING_VERSION,
        "content_recovery_version": CONTENT_RECOVERY_VERSION,
        "decision_contract_parser_version": DECISION_CONTRACT_PARSER_VERSION,
        "n_records_applicable_vlm_direto_or_summary_ok": len(applicable),
        "recovery_status_breakdown": by_recovery_status,
        "reprocessed_decision_status_breakdown": by_reprocessed_decision_status,
        "reprocessed_decisao_final_breakdown_among_status_ok": by_reprocessed_decisao_final,
        "n_records_decision_newly_recovered_from_absent_to_ok": n_changed_from_absent_to_ok,
        "breakdown_by_condition": by_condition,
        "breakdown_by_role": by_role,
        "historical_vs_final_reconciliation": reconciliation,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    call_log_path = Path(args.call_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Lendo (somente leitura): {call_log_path}")
    records = _load_jsonl(call_log_path)
    print(f"{len(records)} linhas carregadas.")

    model_names = {r.get("model_name") for r in records}
    if len(model_names) != 1:
        raise SystemExit(f"ERRO: esperado 1 model_name único no arquivo, encontrados: {model_names}")
    model_name = next(iter(model_names))

    print("Recuperando/reextraindo decisao_final para registros vlm_direto/summary com call_status=ok...")
    reprocessed_records = [reprocess_record(r) for r in records]

    derived_path = output_dir / "call_log_main.decisao_final_reprocessed.jsonl"
    with open(derived_path, "w", encoding="utf-8") as f:
        for r in reprocessed_records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"Arquivo derivado (original intocado) gravado em: {derived_path}")

    print("Reconciliando tentativas históricas x resultado final por chave planejada...")
    reconciliation = reconcile_historical_vs_final(records)

    report = build_report(reprocessed_records, reconciliation, model_name)
    report_path = output_dir / "decisao_final_reprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n=== Relatório consolidado -- {model_name} ===")
    print(f"registros aplicáveis (vlm_direto/summary, ok): {report['n_records_applicable_vlm_direto_or_summary_ok']}")
    print(f"por status de recuperação: {report['recovery_status_breakdown']}")
    print(f"por decision_status reprocessado: {report['reprocessed_decision_status_breakdown']}")
    print(f"por decisao_final reprocessada (status=ok): {report['reprocessed_decisao_final_breakdown_among_status_ok']}")
    print(f"registros que passaram de ausente para recuperado com sucesso: {report['n_records_decision_newly_recovered_from_absent_to_ok']}")
    print(f"\nlinhas históricas no log: {reconciliation['n_historical_log_lines']}")
    print(f"chaves únicas planejadas: {reconciliation['n_unique_planned_keys']}")
    print(f"resultado final por chave: {reconciliation['final_result_per_key_counts']} (soma={reconciliation['final_result_counts_sum_check']})")
    print(f"chaves com múltiplas tentativas reais: {reconciliation['n_keys_with_multiple_real_attempts']}")
    print(f"  -- exceção documentada (mecanismo antigo retried): {reconciliation['n_exception_old_mechanism_retried']}")
    print(f"  -- padrão de bug histórico (ambas antigas): {reconciliation['n_historical_duplicate_bug_pattern']}")
    if reconciliation["n_unexplained_duplicate_same_mechanism"] > 0:
        print(
            f"  *** AVISO: {reconciliation['n_unexplained_duplicate_same_mechanism']} chave(s) com múltiplas "
            "tentativas reais sob o MESMO mecanismo -- não se encaixam em nenhuma categoria explicada, "
            "investigar manualmente (ver 'unexplained_duplicate_same_mechanism' no relatório). ***"
        )
    print(f"\nrelatório completo gravado em: {report_path}")


if __name__ == "__main__":
    main()
