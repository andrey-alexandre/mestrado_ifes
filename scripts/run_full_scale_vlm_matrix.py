"""Etapa 5 — executor da matriz APROVADA de VLMs sobre as 3.000 lesões
congeladas do HAM10000 (`docs/reconstruction/10_frozen_sample_registry.md`),
separado do piloto de 12 lesões (`scripts/run_local_vlm_pilot.py`, que
continua existindo e não é pré-requisito desta execução).

Preserva, sem reinterpretar: o contrato `decisao_final` (vlm_direto e
sumarizador oficial, via `src.multiagent.vlm_direct.run_vlm_direct_call`/
`run_summary_role_call_official_decision_contract`, reusados sem
modificação de lógica), sem retry, sem fallback para `classified_status`/
`classified_status_v2` (v1/v2 continuam computados em paralelo, só
rastreabilidade), e a política de máscara vazia/indisponibilidade
estrutural de S2 já definida em `src.multiagent.presentation` e aplicada
por `scripts/run_s0_s1_s2_preparation.py` (uma condição indisponível é
OMITIDA do plano, nunca substituída por outra). Não exige aprovação
qualitativa nem execução prévia do piloto de 12 -- ver
`docs/reconstruction/30_real_lesion_evaluation_protocol.md`.

Matriz planejada (`src.multiagent.vlm_direct.build_full_scale_planned_calls`):
  - VLM-direto em S0 (comparação VLM-direto S0 × multiagente S0);
  - multiagente (ABCD/Menzies/SPCL/sumarizador) em S0, S1 e S2, cada
    condição só quando disponível para aquela lesão.
A comparação CNN S0 × multiagente S2 NÃO dispara nenhuma chamada à CNN --
reusa diretamente `experiments/cnn_baseline/s0_ham10000_eval/
s0_cnn_scores.csv` (já real, checkpoint congelado, ver docs 28/29),
conferindo o pareamento por (`lesion_id`, `image_id`) contra o manifesto
S0/S1/S2 antes de aceitar a execução (checagem OPERACIONAL).

Dois modos:
  --mode plan     Não faz NENHUMA chamada a modelo (nem checagem de
                   config/credencial). Só lê o manifesto S0/S1/S2 e os
                   call_log*.jsonl já existentes para o modelo, aplica a
                   checagem de reaproveitamento
                   (`src.multiagent.vlm_direct.check_reuse`) e imprime/
                   grava, por papel e condição: quantas chamadas estão
                   planejadas, quantas já estão cobertas por saída
                   existente, quantas faltam.
  --mode execute  Roda as checagens operacionais (manifesto, imagens,
                   pareamento da CNN, config/credencial do modelo,
                   diretório de saída gravável), calcula o mesmo plano de
                   reaproveitamento, e executa SÓ as chamadas que faltam.
                   Cada tentativa (sucesso, falha técnica, sem-resposta ou
                   recusa) é gravada no log IMEDIATAMENTE após acontecer
                   (append + flush + fsync) -- uma interrupção (Ctrl+C,
                   queda de energia, etc.) nunca perde o que já foi
                   tentado, e uma nova invocação no mesmo --output-dir
                   retoma exatamente de onde parou, sem repetir nenhuma
                   tentativa já registrada (inclusive falhas -- não há
                   retry em lugar nenhum deste pipeline).

Problema OPERACIONAL (único tipo que interrompe a execução, ver doc 30):
manifesto S0/S1/S2 ausente ou com contagem errada, SHA-256 do manifesto
congelado de 3.000 não confere, CSV da CNN ausente ou pareamento
lesão/imagem não bate com o manifesto S0/S1/S2, modelo/credencial
incompatível, ou diretório de saída não gravável. Acertar/errar
quadrante, parecer distorcido, decisão inválida, ou taxa de validade
abaixo de qualquer limiar são RESULTADO, nunca motivo de parada.

STATUS: implementado, aguardando execução do pesquisador (requer Ollama
local com os modelos já baixados para Qwen3-VL/MedGemma, ou
`OPENAI_API_KEY`/`GOOGLE_API_KEY` definidos para GPT/Gemini). Não
executado pelo assistente que o escreveu (`feedback_no_autonomous_execution`).

Uso:
    # 1. Planejamento -- nenhuma chamada a modelo, mostra reaproveitamento/faltantes:
    python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag gemini-3.1-flash-lite
    python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag gpt-5.4-nano
    python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag qwen3-vl:8b-instruct-q4_K_M
    python scripts/run_full_scale_vlm_matrix.py --mode plan --model-tag medgemma1.5:4b-it-q4_K_M

    # 2. Execução -- roda só o que falta; pode ser interrompida (Ctrl+C) e
    #    retomada reinvocando o MESMO comando (mesmo --output-dir):
    python scripts/run_full_scale_vlm_matrix.py --mode execute --model-tag gemini-3.1-flash-lite

    # 3. Retomada após interrupção -- comando IDÊNTICO ao de execução; o
    #    script relê o call_log_main.jsonl já gravado e pula tudo que já
    #    foi tentado:
    python scripts/run_full_scale_vlm_matrix.py --mode execute --model-tag gemini-3.1-flash-lite
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from src.multiagent import presentation as pres  # noqa: E402
from src.multiagent import vlm_direct as vd  # noqa: E402

DEFAULT_SAMPLE_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_S0_S1_S2_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "s0_s1_s2_preparation_manifest.csv")
DEFAULT_CNN_S0_SCORES_CSV = str(REPO_ROOT / "experiments" / "cnn_baseline" / "s0_ham10000_eval" / "s0_cnn_scores.csv")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "full_scale_matrix")

CALL_LOG_FILENAME = "call_log_main.jsonl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("plan", "execute"))
    parser.add_argument("--model-tag", required=True, help="Ex.: gemini-3.1-flash-lite, gpt-5.4-nano, qwen3-vl:8b-instruct-q4_K_M, medgemma1.5:4b-it-q4_K_M")
    parser.add_argument("--sample-manifest", default=DEFAULT_SAMPLE_MANIFEST_PATH)
    parser.add_argument("--s0-s1-s2-manifest", default=DEFAULT_S0_S1_S2_MANIFEST_PATH)
    parser.add_argument("--cnn-s0-scores-csv", default=DEFAULT_CNN_S0_SCORES_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    return parser


# ---------------------------------------------------------------------------
# Checagens OPERACIONAIS -- qualquer falha aqui interrompe com SystemExit
# (ou exceção propagada), antes de qualquer chamada a modelo.
# ---------------------------------------------------------------------------
def _load_s0_s1_s2_manifest(sample_manifest_path: Path, s0_s1_s2_manifest_path: Path) -> list[dict[str, str]]:
    pres.verify_manifest_or_die(sample_manifest_path)
    frozen_rows = pres.load_manifest_rows(sample_manifest_path)
    frozen_keys = {(r["lesion_id"], r["image_id"]) for r in frozen_rows}

    if not s0_s1_s2_manifest_path.exists():
        raise SystemExit(
            f"ERRO OPERACIONAL: {s0_s1_s2_manifest_path} não encontrado -- rode "
            "scripts/run_s0_s1_s2_preparation.py primeiro (não refeito por este "
            "script, que só reusa a saída real já gerada)."
        )
    with open(s0_s1_s2_manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != pres.EXPECTED_N_CASES:
        raise SystemExit(
            f"ERRO OPERACIONAL: {s0_s1_s2_manifest_path} tem {len(rows)} linhas, "
            f"esperado exatamente {pres.EXPECTED_N_CASES}."
        )
    manifest_keys = {(r["lesion_id"], r["image_id"]) for r in rows}
    if manifest_keys != frozen_keys:
        only_in_manifest = manifest_keys - frozen_keys
        only_in_frozen = frozen_keys - manifest_keys
        raise SystemExit(
            "ERRO OPERACIONAL: conjunto de (lesion_id, image_id) de "
            f"{s0_s1_s2_manifest_path} não é idêntico ao do manifesto congelado "
            f"verificado por SHA-256 -- {len(only_in_manifest)} só na preparação "
            f"S0/S1/S2, {len(only_in_frozen)} só no manifesto congelado. A "
            "preparação S0/S1/S2 precisa ser derivada exatamente do manifesto "
            "congelado atual."
        )
    return rows


def _load_cnn_s0_scores(cnn_csv_path: Path) -> list[dict[str, str]]:
    if not cnn_csv_path.exists():
        raise SystemExit(
            f"ERRO OPERACIONAL: {cnn_csv_path} não encontrado -- este executor "
            "reusa a saída real já existente da CNN em S0 (não a recalcula; ver "
            "docs/reconstruction/28_cnn_stage3_threshold_closure.md e "
            "29_cnn_stage3_full_closure_s0_eval.md)."
        )
    with open(cnn_csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_cnn_s0_pairing(cnn_rows: list[dict[str, str]], s0_s1_s2_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Confere o pareamento CNN S0 × manifesto S0/S1/S2 por
    (``lesion_id``, ``image_id``) -- ambos devem ser derivados do MESMO
    manifesto congelado de 3.000 lesões, então os dois conjuntos devem ser
    EXATAMENTE iguais. Qualquer divergência é um problema OPERACIONAL real
    (os dois arquivos não correspondem à mesma amostra) -- interrompe,
    nunca é tratado como "resultado"."""
    cnn_keys = {(r["lesion_id"], r["image_id"]) for r in cnn_rows}
    manifest_keys = {(r["lesion_id"], r["image_id"]) for r in s0_s1_s2_rows}
    only_in_cnn = cnn_keys - manifest_keys
    only_in_manifest = manifest_keys - cnn_keys
    if only_in_cnn or only_in_manifest:
        raise SystemExit(
            "ERRO OPERACIONAL: pareamento CNN S0 x manifesto S0/S1/S2 não bate -- "
            f"{len(only_in_cnn)} lesão(ões) só no CSV da CNN, {len(only_in_manifest)} "
            "só no manifesto S0/S1/S2. Os dois arquivos precisam vir exatamente da "
            "mesma amostra congelada de 3.000 lesões."
        )
    return {"n_matched": len(cnn_keys), "n_only_in_cnn": 0, "n_only_in_manifest": 0}


def _ensure_output_dir_writable(output_dir_root: Path, model_name: str) -> Path:
    output_dir = output_dir_root / model_name.replace(":", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    probe_path = output_dir / ".write_check"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        raise SystemExit(f"ERRO OPERACIONAL: não foi possível gravar em {output_dir}: {exc}")
    return output_dir


def _load_existing_call_log(output_dir: Path) -> list[dict[str, Any]]:
    log_path = output_dir / CALL_LOG_FILENAME
    if not log_path.exists():
        return []
    records = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Resumo por papel/condição -- usado tanto em --mode plan quanto ao final
# de --mode execute (relendo o log já ATUALIZADO).
# ---------------------------------------------------------------------------
_DIAG_STATE_KEY_BY_ROLE = {
    vd.ROLE_ABCD: "diagnosis_abcd",
    vd.ROLE_MENZIES: "diagnosis_menzies",
    vd.ROLE_SPCL: "diagnosis_spcl",
}


def _index_existing_diagnoses(existing_records: list[dict[str, Any]], model_name: str) -> dict[tuple[str, str, str], dict[str, str]]:
    """``{(lesion_id, image_id, condition): {"diagnosis_abcd": texto, ...}}``
    a partir de registros ABCD/Menzies/SPCL JÁ bem-sucedidos
    (``call_status == STATUS_OK``) do ``model_name`` em questão -- usado
    para RE-ALIMENTAR o estado em memória de um grupo multiagente cujo
    sumarizador (ou um papel restante) ainda falta rodar nesta invocação,
    sem precisar RECHAMAR ABCD/Menzies/SPCL que já têm resultado real
    registrado (isso seria um retry disfarçado, que este pipeline nunca
    faz). O texto alimentado é `raw_response_text` -- para papéis
    Ollama já era uma string (idêntico a uma chamada fresca); para
    Gemini/GPT é a representação textual do objeto Pydantic original
    (não json.dumps idêntico a uma chamada fresca via
    `_serialize_for_prompt`, mas o MESMO conteúdo de campos, nunca um
    valor inventado/adivinhado) -- aceito pelo sumarizador porque
    `GraphState.diagnosis_*` já declara `Optional[Union[str, Answer]]`:
    uma string não é "um tipo novo"."""
    index: dict[tuple[str, str, str], dict[str, str]] = {}
    for rec in existing_records:
        if rec.get("model_name") != model_name or rec.get("call_status") != vd.STATUS_OK:
            continue
        diag_key = _DIAG_STATE_KEY_BY_ROLE.get(rec.get("role"))
        if diag_key is None or rec.get("raw_response_text") is None:
            continue
        key = (rec["lesion_id"], rec["image_id"], rec["condition"])
        index.setdefault(key, {})[diag_key] = rec["raw_response_text"]
    return index


def _summarize_plan_by_role_condition(reuse_result: dict[str, Any]) -> dict[str, dict[str, int]]:
    by_key: dict[str, dict[str, int]] = {}

    def _bump(call: dict[str, str], field: str) -> None:
        key = f"{call['role']}__{call['condition']}"
        d = by_key.setdefault(key, {"planejadas": 0, "ja_cobertas": 0, "faltam": 0})
        d["planejadas"] += 1
        d[field] += 1

    for call in reuse_result["already_covered"]:
        _bump(call, "ja_cobertas")
    for call in reuse_result["missing"]:
        _bump(call, "faltam")
    return by_key


def _build_plan(args: argparse.Namespace, model_name: str) -> tuple[dict[str, Any], Path, list[dict[str, str]]]:
    s0_s1_s2_rows = _load_s0_s1_s2_manifest(Path(args.sample_manifest), Path(args.s0_s1_s2_manifest))
    cnn_rows = _load_cnn_s0_scores(Path(args.cnn_s0_scores_csv))
    cnn_pairing = _check_cnn_s0_pairing(cnn_rows, s0_s1_s2_rows)

    output_dir = Path(args.output_dir) / model_name.replace(":", "_").replace("/", "_")
    existing_records = _load_existing_call_log(output_dir)

    planned_calls = vd.build_full_scale_planned_calls(s0_s1_s2_rows)
    reuse_result = vd.check_reuse(model_name, existing_records, planned_calls)

    plan = {
        "model_name": model_name,
        "n_lesions_in_manifest": len(s0_s1_s2_rows),
        "cnn_s0_pairing": cnn_pairing,
        "reuse": reuse_result,
        "by_role_condition": _summarize_plan_by_role_condition(reuse_result),
    }
    return plan, output_dir, s0_s1_s2_rows


def _run_plan_mode(args: argparse.Namespace) -> None:
    model_name = args.model_tag
    print(f"=== Planejamento (sem chamadas a modelo) -- modelo: {model_name} ===")
    plan, output_dir, _rows = _build_plan(args, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    plan_path = output_dir / "plan_report.json"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2, default=str)

    print(f"lesões no manifesto S0/S1/S2: {plan['n_lesions_in_manifest']}")
    print(f"pareamento CNN S0 x manifesto: {plan['cnn_s0_pairing']}")
    print(f"total planejado: {plan['reuse']['n_planned']}  já cobertas: {plan['reuse']['n_already_covered_by_existing_output']}  faltam: {plan['reuse']['n_missing_must_run']}")
    print("\npor papel/condição:")
    for key in sorted(plan["by_role_condition"]):
        d = plan["by_role_condition"][key]
        print(f"  {key:40s} planejadas={d['planejadas']:6d}  já_cobertas={d['ja_cobertas']:6d}  faltam={d['faltam']:6d}")
    print(f"\nplano completo gravado em: {plan_path}")


# ---------------------------------------------------------------------------
# Execução -- grava cada tentativa IMEDIATAMENTE (append + flush + fsync),
# para que uma interrupção nunca perca/repita uma tentativa já registrada.
# ---------------------------------------------------------------------------
def _append_record(log_file, record: "vd.CallRecord") -> None:
    log_file.write(json.dumps(record.to_flat_dict(), ensure_ascii=False, default=str) + "\n")
    log_file.flush()
    os.fsync(log_file.fileno())


def _load_condition_image(s0_s1_s2_rows_by_key: dict[tuple[str, str], dict[str, str]], image_id: str, condition: str, lesion_id: str) -> Image.Image:
    row = s0_s1_s2_rows_by_key[(lesion_id, image_id)]
    path_str = row[f"path_{condition}"]
    if not path_str:
        raise FileNotFoundError(
            f"ERRO OPERACIONAL: linha do manifesto S0/S1/S2 para lesion_id={lesion_id}/"
            f"image_id={image_id} não tem path_{condition} preenchido, mas o plano "
            "indicou esta condição como disponível -- inconsistência interna do "
            "manifesto, não um problema de desempenho do modelo."
        )
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"ERRO OPERACIONAL: imagem da condição {condition} ausente em disco para "
            f"lesion_id={lesion_id}/image_id={image_id}: {path}"
        )
    return Image.open(path).convert("RGB")


def _run_execute_mode(args: argparse.Namespace) -> None:
    model_name = args.model_tag
    print(f"=== Execução da matriz de 3.000 lesões -- modelo: {model_name} ===")

    plan, output_dir, s0_s1_s2_rows = _build_plan(args, model_name)
    output_dir = _ensure_output_dir_writable(Path(args.output_dir), model_name)
    s0_s1_s2_rows_by_key = {(r["lesion_id"], r["image_id"]): r for r in s0_s1_s2_rows}

    print("--- Checagem operacional de modelo/configuração ---")
    model_info = vd.get_model_info(model_name)  # levanta RuntimeError se incompatível
    print(f"  modelo OK: {model_info}")

    missing_calls = plan["reuse"]["missing"]
    print(
        f"planejadas: {plan['reuse']['n_planned']}  já cobertas por saída existente: "
        f"{plan['reuse']['n_already_covered_by_existing_output']}  a executar agora: {len(missing_calls)}"
    )
    if not missing_calls:
        print("Nada a executar -- todas as chamadas planejadas já estão cobertas por saída existente.")
        return

    # Índice de diagnósticos ABCD/Menzies/SPCL já reais (bem-sucedidos em
    # invocação(ões) anterior(es)) -- usado para re-alimentar o estado em
    # memória de um grupo cujo sumarizador (ou outro papel do mesmo
    # grupo) ainda falta rodar, SEM rechamar o que já tem resultado real
    # (ver `_index_existing_diagnoses`).
    existing_records_for_seeding = _load_existing_call_log(output_dir)
    existing_diagnoses_by_key = _index_existing_diagnoses(existing_records_for_seeding, model_name)

    log_path = output_dir / CALL_LOG_FILENAME
    tracker = vd.AbstentionTracker()
    accumulated_state: dict[tuple[str, str], dict[str, Any]] = {}
    image_cache: dict[tuple[str, str, str], Image.Image] = {}

    def _get_image(lesion_id: str, image_id: str, condition: str) -> Image.Image:
        key = (lesion_id, image_id, condition)
        if key not in image_cache:
            image_cache[key] = _load_condition_image(s0_s1_s2_rows_by_key, image_id, condition, lesion_id)
        return image_cache[key]

    def _init_state_if_needed(state_key: tuple[str, str], lesion_id: str, image_id: str, condition: str) -> None:
        if state_key in accumulated_state:
            return
        image = _get_image(lesion_id, image_id, condition)
        base = vd.build_graph_state(image, f"{image_id}_{condition}").model_dump()
        base.update(existing_diagnoses_by_key.get((lesion_id, image_id, condition), {}))
        accumulated_state[state_key] = base

    n_done = 0
    with open(log_path, "a", encoding="utf-8") as log_file:
        for call in missing_calls:
            role, condition, lesion_id, image_id = call["role"], call["condition"], call["lesion_id"], call["image_id"]

            if tracker.is_paused(role):
                record = vd.not_executed_record(model_name, role, condition, lesion_id, image_id, tracker.by_role[role].pause_reason)
                _append_record(log_file, record)
                n_done += 1
                continue

            if role == vd.ROLE_VLM_DIRETO:
                image = _get_image(lesion_id, image_id, condition)
                record = vd.run_vlm_direct_call(model_name, image, lesion_id, image_id, timeout_s=args.timeout_s)
            elif role == vd.ROLE_SUMMARY:
                state_key = (lesion_id, condition)
                _init_state_if_needed(state_key, lesion_id, image_id, condition)
                state = vd.GraphState(**accumulated_state[state_key])
                record, state_update = vd.run_summary_role_call_official_decision_contract(
                    model_name, state, lesion_id, image_id, condition, timeout_s=args.timeout_s
                )
                for k, v in state_update.items():
                    accumulated_state[state_key][k] = v
            else:
                state_key = (lesion_id, condition)
                _init_state_if_needed(state_key, lesion_id, image_id, condition)
                state = vd.GraphState(**accumulated_state[state_key])
                record, state_update = vd.run_multiagent_role_call(
                    model_name, role, state, lesion_id, image_id, condition, timeout_s=args.timeout_s
                )
                for k, v in state_update.items():
                    accumulated_state[state_key][k] = v

            tracker.register(role, record.call_status)
            _append_record(log_file, record)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(missing_calls):
                print(f"  [{n_done}/{len(missing_calls)}] último: {role}/{condition} lesion={lesion_id} status={record.call_status}")

    print(f"\nExecução concluída (ou interrompida) após {n_done}/{len(missing_calls)} chamadas desta invocação.")
    print(f"Log completo (acumulado de todas as invocações): {log_path}")

    # Reconstrói o plano lendo o log JÁ ATUALIZADO -- mostra quanto da
    # matriz de 3.000 lesões está coberto no total, não só nesta invocação.
    final_plan, _out, _rows = _build_plan(args, model_name)
    final_plan_path = output_dir / "plan_report_after_execution.json"
    with open(final_plan_path, "w", encoding="utf-8") as f:
        json.dump(final_plan, f, ensure_ascii=False, indent=2, default=str)
    print(
        f"cobertura total após esta invocação: {final_plan['reuse']['n_already_covered_by_existing_output']}/"
        f"{final_plan['reuse']['n_planned']} (faltam {final_plan['reuse']['n_missing_must_run']})"
    )
    print(f"plano pós-execução gravado em: {final_plan_path}")
    if final_plan["reuse"]["n_missing_must_run"] > 0:
        print(
            "Ainda falta rodar parte da matriz -- reinvoque o MESMO comando "
            "(--mode execute) para retomar; nenhuma tentativa já registrada será repetida."
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "plan":
        _run_plan_mode(args)
    else:
        _run_execute_mode(args)


if __name__ == "__main__":
    main()
