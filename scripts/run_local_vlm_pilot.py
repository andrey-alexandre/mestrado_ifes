"""Etapa 5, segunda entrega — driver único do piloto local de VLMs via
Ollama sobre as 12 lesões fora da amostra congelada (ver
`docs/reconstruction/20_local_vlm_pilot.md` e
`src/multiagent/vlm_direct.py`, que concentra toda a lógica real reusada
aqui).

Roda UMA vez por tag de modelo Ollama (argumento `--model-tag`). Ordem
imposta por este script, por modelo:

  1. `ollama.show(model_tag)` uma única vez (metadados do modelo).
  2. Controles de envio de imagem (3 imagens sintéticas) -- SE FALHAR,
     PARA aqui, não tenta os controles de síntese nem a matriz principal.
  3. Controles de síntese (4 cenários fixos) -- SE FALHAR, PARA aqui, não
     tenta a matriz principal de 92 chamadas.
  4. Matriz principal: exatamente 92 chamadas planejadas
     (`src.multiagent.vlm_direct.build_planned_calls`), respeitando a
     lógica de pausa por abstenção sistemática (3 contadores por papel,
     nunca retry).

Modelos a testar (Passo 7 da entrega): `qwen3-vl:8b-instruct-q4_K_M` e
`medgemma1.5:4b-it-q4_K_M`. Rodar este script uma vez para CADA um
(`--model-tag` diferente), independentemente -- este script NÃO encadeia
os dois automaticamente. Se, ao concluir os dois, apenas UM passou nos
controles de pré-execução (ver `summary.json["controls"]["image_sending"]
["overall_pass"]` e cada `summary.json["controls"]["synthesis"][i]
["passed"]`, ou `stopped_before_main_matrix` se já tiver parado antes),
o pesquisador deve rodar este mesmo script
uma TERCEIRA vez com `--model-tag qwen2.5vl:7b-q4_K_M` (modelo de
contingência) -- esta decisão é indicada no relatório Markdown final de
cada execução, não decidida/encadeada automaticamente aqui (este
assistente não pode executar nada para descobrir ao vivo qual passa).

STATUS: implementado, aguardando execução do pesquisador (requer Ollama
rodando localmente com os modelos já baixados, `pip install psutil
ollama` se ainda não instalados). Não executado pelo assistente que o
escreveu (`feedback_no_autonomous_execution`).

Uso (uma vez por modelo):
    python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M
    python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M
    # Só se necessário (ver acima):
    python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image  # noqa: E402

from src.multiagent import vlm_direct as vd  # noqa: E402

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_SAMPLING_RESULTS_JSON = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_sampling_results.json")
DEFAULT_IMAGES_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_images")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", required=True, help="Tag Ollama exata, ex. qwen3-vl:8b-instruct-q4_K_M")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sampling-results-json", default=DEFAULT_SAMPLING_RESULTS_JSON)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    return parser


def _load_pilot12_rows(manifest_path: Path) -> list[dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 12:
        raise SystemExit(f"ERRO: {manifest_path} tem {len(rows)} linhas, esperado exatamente 12.")
    return rows


def _resolve_designated_image_ids(rows: list[dict[str, str]], sampling_results_json: Path) -> list[str]:
    if sampling_results_json.exists():
        with open(sampling_results_json, encoding="utf-8") as f:
            data = json.load(f)
        designated_lesion_ids = data.get("designated_s0_s1_lesion_ids")
        if designated_lesion_ids:
            by_lesion = {r["lesion_id"]: r["image_id"] for r in rows}
            return [by_lesion[lid] for lid in designated_lesion_ids if lid in by_lesion]
    # Fallback determinístico idêntico à regra de run_pilot12_sample.py: os
    # 4 primeiros lesion_id em ordem alfabética.
    rows_sorted = sorted(rows, key=lambda r: r["lesion_id"])
    return [r["image_id"] for r in rows_sorted[:4]]


def _load_condition_image(images_dir: Path, image_id: str, condition: str) -> Image.Image:
    path = images_dir / condition / f"{image_id}_{condition}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"imagem da condição {condition} não encontrada para image_id={image_id}: {path} -- "
            "rode scripts/run_pilot12_image_preparation.py primeiro."
        )
    return Image.open(path).convert("RGB")


def _memory_samples_from_records(records: list["vd.CallRecord"]) -> list[int]:
    samples: list[int] = []
    for r in records:
        if r.mem_before is not None:
            samples.append(r.mem_before.system_swap_used_bytes)
        if r.mem_after is not None:
            samples.append(r.mem_after.system_swap_used_bytes)
    return samples


def main() -> None:
    args = build_arg_parser().parse_args()
    model_name = args.model_tag
    manifest_path = Path(args.manifest_path)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir) / model_name.replace(":", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Piloto local de VLM via Ollama -- modelo: {model_name} ===")
    rows = _load_pilot12_rows(manifest_path)
    designated_image_ids = _resolve_designated_image_ids(rows, Path(args.sampling_results_json))
    print(f"4 image_id designados para multiagente S0/S1: {designated_image_ids}")

    print("--- ollama.show(modelo) ---")
    try:
        model_info = vd.get_model_info(model_name)
    except Exception as exc:  # noqa: BLE001 -- registrado, não interrompe a escrita do relatório parcial
        model_info = {"error": str(exc)}
        print(f"AVISO: falha ao obter informações do modelo via ollama.show: {exc}")

    all_records: list[vd.CallRecord] = []

    # --- 1. Controles de envio de imagem ---
    print("--- Controles de envio de imagem (3 imagens sintéticas) ---")
    image_control_results = []
    for spec in vd.IMAGE_CONTROL_SPECS:
        image = vd.build_image_sending_control_image(spec)
        message = vd._image_message(vd.convert_to_base64(image), vd.IMAGE_CONTROL_PROMPT)  # type: ignore[attr-defined]
        llm = vd.ChatOllama(model=model_name)
        outcome = vd.call_with_timeout(lambda m=message, l=llm: l.invoke([m]).content, timeout_s=args.timeout_s)
        status, cause, raw_text = vd.classify_call_outcome(outcome)
        record = vd.CallRecord(
            model_name=model_name, role="controle_imagem", condition=spec["name"],
            lesion_id="controle_sintetico", image_id=spec["name"],
            attempt_timestamp_utc=vd.datetime.now(vd.timezone.utc).isoformat(),
            elapsed_s=outcome.elapsed_s, mem_before=None, mem_after=None,
            raw_response_text=raw_text, call_status=status, failure_cause=cause, classified_status=None,
        )
        all_records.append(record)
        image_control_results.append({"spec": spec, "raw_text": raw_text})

    image_controls_eval = vd.evaluate_image_sending_controls(image_control_results)
    print(f"controles de envio de imagem: overall_pass={image_controls_eval['overall_pass']}")

    if not image_controls_eval["overall_pass"]:
        _write_stop_report(
            output_dir, model_name, model_info, all_records,
            stage="controle_de_envio_de_imagem",
            controls={"image_sending": image_controls_eval},
        )
        print("PARADO: controle de envio de imagem reprovado -- matriz principal de 92 chamadas NÃO tentada.")
        return

    # --- 2. Controles de síntese ---
    print("--- Controles de síntese (4 cenários fixos) ---")
    synthesis_results = []
    for scenario_name in vd.FIXED_SYNTHESIS_SCENARIOS:
        result = vd.run_synthesis_control(model_name, scenario_name, timeout_s=args.timeout_s)
        all_records.append(result["call_record"])
        synthesis_results.append({k: v for k, v in result.items() if k != "call_record"})
        print(f"  cenário={scenario_name} passed={result['passed']} classificado={result['classified_status']}")

    synthesis_overall_pass = all(r["passed"] for r in synthesis_results)
    print(f"controles de síntese: overall_pass={synthesis_overall_pass}")

    if not synthesis_overall_pass:
        _write_stop_report(
            output_dir, model_name, model_info, all_records,
            stage="controle_de_sintese",
            controls={"image_sending": image_controls_eval, "synthesis": synthesis_results},
        )
        print("PARADO: controle de síntese reprovado -- matriz principal de 92 chamadas NÃO tentada.")
        return

    # --- 3. Matriz principal (92 chamadas planejadas) ---
    print("--- Matriz principal: 92 chamadas planejadas ---")
    planned_calls = vd.build_planned_calls(rows, designated_image_ids)
    print(f"chamadas planejadas: {len(planned_calls)} (deve ser 92)")

    tracker = vd.AbstentionTracker()
    # Estado acumulado por (lesion_id, condition) para alimentar o
    # sumarizador com os diagnósticos ABCD/Menzies/SPCL reais já obtidos
    # neste mesmo run -- nunca com dublês fixos (isso é só nos controles).
    accumulated_state: dict[tuple[str, str], dict[str, Any]] = {}
    matrix_records: list[vd.CallRecord] = []

    image_cache: dict[tuple[str, str], Image.Image] = {}

    def _get_image(image_id: str, condition: str) -> Image.Image:
        key = (image_id, condition)
        if key not in image_cache:
            image_cache[key] = _load_condition_image(images_dir, image_id, condition)
        return image_cache[key]

    for call in planned_calls:
        role, condition, lesion_id, image_id = call["role"], call["condition"], call["lesion_id"], call["image_id"]

        if tracker.is_paused(role):
            record = vd.not_executed_record(model_name, role, condition, lesion_id, image_id, tracker.by_role[role].pause_reason)
            matrix_records.append(record)
            continue

        if role == vd.ROLE_VLM_DIRETO:
            image = _get_image(image_id, condition)
            record = vd.run_vlm_direct_call(model_name, image, lesion_id, image_id, timeout_s=args.timeout_s)
        else:
            key = (lesion_id, condition)
            if key not in accumulated_state:
                image = _get_image(image_id, condition)
                accumulated_state[key] = vd.build_graph_state(image, f"{image_id}_{condition}").model_dump()
            state = vd.GraphState(**accumulated_state[key])
            record, state_update = vd.run_multiagent_role_call(
                model_name, role, state, lesion_id, image_id, condition, timeout_s=args.timeout_s
            )
            for k, v in state_update.items():
                accumulated_state[key][k] = v

        tracker.register(role, record.call_status)
        matrix_records.append(record)
        print(
            f"  [{role}/{condition}] lesion={lesion_id} image={image_id} "
            f"status={record.call_status} classificado={record.classified_status}"
        )

    all_records.extend(matrix_records)

    _write_full_report(
        output_dir, model_name, model_info, all_records, planned_calls, matrix_records, tracker,
        controls={"image_sending": image_controls_eval, "synthesis": synthesis_results},
    )
    print(f"\nRelatórios/logs gravados em: {output_dir}")


def _write_call_log(output_dir: Path, records: list["vd.CallRecord"]) -> Path:
    log_path = output_dir / "call_log.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_flat_dict(), ensure_ascii=False, default=str) + "\n")
    return log_path


def _write_stop_report(output_dir: Path, model_name: str, model_info: dict, records: list, stage: str, controls: dict) -> None:
    log_path = _write_call_log(output_dir, records)
    summary = {
        "model_name": model_name,
        "model_info": model_info,
        "stopped_before_main_matrix": True,
        "stopped_at_stage": stage,
        "controls": _serialize_controls(controls),
        "n_calls_executed_total": len(records),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    report_lines = [
        f"# Piloto local de VLM — {model_name} — PARADO ANTES DA MATRIZ PRINCIPAL",
        "",
        f"- Etapa em que parou: `{stage}`",
        "- Matriz principal de 92 chamadas: NÃO tentada.",
        f"- Log de chamadas (controles): `{log_path}`",
        "",
        "## Controles",
        "",
        f"```json\n{json.dumps(_serialize_controls(controls), ensure_ascii=False, indent=2, default=str)}\n```",
        "",
    ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def _serialize_controls(controls: dict) -> dict:
    return json.loads(json.dumps(controls, default=str, ensure_ascii=False))


def _write_full_report(
    output_dir: Path, model_name: str, model_info: dict, all_records: list, planned_calls: list,
    matrix_records: list, tracker, controls: dict,
) -> None:
    log_path = _write_call_log(output_dir, all_records)

    n_planned = len(planned_calls)
    n_executed = sum(1 for r in matrix_records if r.call_status != vd.STATUS_NOT_EXECUTED)
    n_not_executed = n_planned - n_executed

    by_role_condition: dict[str, dict[str, int]] = {}
    for r in matrix_records:
        key = f"{r.role}__{r.condition}"
        d = by_role_condition.setdefault(key, {"planejadas": 0, "executadas": 0, "nao_executadas": 0, "ok": 0, "sem_conclusao": 0})
        d["planejadas"] += 1
        if r.call_status == vd.STATUS_NOT_EXECUTED:
            d["nao_executadas"] += 1
        else:
            d["executadas"] += 1
            if r.call_status == vd.STATUS_OK:
                d["ok"] += 1

    # NOTA (interpretação de design): classify_case_status só se aplica a um
    # parecer com vocabulário "Suspeita de Melanoma/Lesão Benigna/Lesão
    # Indeterminada" -- isso existe para VLM-direto e para o sumarizador
    # (campo "validation"/"diagnostico"), NÃO para as saídas estruturadas de
    # ABCD/Menzies/SPCL (sem esse campo). Por isso esta checagem de "zero
    # parecer conclusivo" só é aplicada aos papéis vlm_direto/summary -- para
    # ABCD/Menzies/SPCL ela seria sempre "zero" por construção e não
    # informativa; flagueado aqui para não confundir o pesquisador com um
    # falso alarme.
    _ROLES_WITH_CLASSIFIABLE_OUTPUT = {vd.ROLE_VLM_DIRETO, vd.ROLE_SUMMARY}
    roles_with_zero_conclusive: list[str] = []
    for key, d in by_role_condition.items():
        role = key.split("__")[0]
        if role not in _ROLES_WITH_CLASSIFIABLE_OUTPUT:
            continue
        has_conclusive = any(
            r.classified_status in ("positivo", "negativo", "indeterminado")
            for r in matrix_records
            if f"{r.role}__{r.condition}" == key
        )
        if not has_conclusive and key not in roles_with_zero_conclusive:
            roles_with_zero_conclusive.append(key)

    n_valid_first_attempt = sum(1 for r in matrix_records if r.call_status == vd.STATUS_OK)
    valid_rate = n_valid_first_attempt / n_executed if n_executed else 0.0

    binary_statuses = [vd.classify_binary_coverage(r.classified_status) for r in matrix_records]
    n_positivo = sum(1 for b in binary_statuses if b is True)
    n_negativo = sum(1 for b in binary_statuses if b is False)

    swap_samples = _memory_samples_from_records(matrix_records)
    swap_growth = vd.detect_sustained_swap_growth(swap_samples) if swap_samples else {
        "note": "psutil indisponível ou nenhuma amostra coletada -- não computado.",
        "definition_needs_researcher_signoff": True,
    }

    summary = {
        "model_name": model_name,
        "model_info": model_info,
        "stopped_before_main_matrix": False,
        "n_calls_planned": n_planned,
        "n_calls_executed": n_executed,
        "n_calls_not_executed_due_to_pause": n_not_executed,
        "valid_on_first_attempt_rate": valid_rate,
        "binary_coverage": {"positivo": n_positivo, "negativo": n_negativo},
        "by_role_condition": by_role_condition,
        "roles_with_zero_conclusive_opinion": roles_with_zero_conclusive,
        "abstention_counters_trace": tracker.trace(),
        "controls": _serialize_controls(controls),
        "sustained_swap_growth": swap_growth,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    report_lines = [
        f"# Piloto local de VLM — {model_name}",
        "",
        f"- Log de chamadas (todas, incl. controles): `{log_path}`",
        f"- Chamadas planejadas: {n_planned} (deve ser 92)",
        f"- Chamadas efetivamente realizadas: {n_executed}",
        f"- Chamadas NÃO executadas (pausa por abstenção): {n_not_executed}",
        f"- Taxa de validade na primeira tentativa: {valid_rate:.2%}",
        f"- Cobertura binária: positivo={n_positivo}, negativo={n_negativo}",
        "",
        "## Papéis x condições sem nenhum parecer conclusivo (positivo/negativo/indeterminado)",
        "",
        f"{roles_with_zero_conclusive if roles_with_zero_conclusive else 'nenhum'}",
        "",
        "## Rastro dos contadores de abstenção (por papel)",
        "",
        f"```json\n{json.dumps(tracker.trace(), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Crescimento sustentado de swap (definição carece de aval do pesquisador)",
        "",
        f"```json\n{json.dumps(swap_growth, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Controles",
        "",
        f"```json\n{json.dumps(_serialize_controls(controls), ensure_ascii=False, indent=2, default=str)}\n```",
        "",
    ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
