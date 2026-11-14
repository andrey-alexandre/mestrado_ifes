"""Etapa 5, segunda entrega (REWORK) — driver único do piloto local de
VLMs via Ollama sobre as 12 lesões fora da amostra congelada (ver
`docs/reconstruction/20_local_vlm_pilot.md` e
`src/multiagent/vlm_direct.py`, que concentra toda a lógica real reusada
aqui).

Fluxo de DUAS INVOCAÇÕES por tag de modelo Ollama (`--model-tag`) --
mudança de arquitetura desta entrega (decisão fechada #4, "achado
clínico novo" deixou de ser pass/fail automático):

  ESTÁGIO 1 (invocação normal, sem `--synthesis-manual-review-file`):
    1. `ollama.show(model_tag)` uma única vez (metadados do modelo).
    2. Controles de envio de imagem -- POR PAPEL VISUAL (vlm_direto,
       diagnostic_abcd, diagnostic_menzies, diagnostic_spcl): 2 controles
       de imagem (cor+quadrante diferentes entre si) + 1 controle
       "sem imagem" (texto puro) por papel = 12 chamadas. SE QUALQUER
       PAPEL reprovar, PARA aqui -- não tenta controles de síntese nem a
       matriz principal.
    3. Controles de síntese (4 cenários fixos) -- SE a saída não for
       sequer PARSEÁVEL pela A34, PARA aqui. Se for parseável, este
       script grava `synthesis_manual_review_template.json` e PARA DE
       QUALQUER FORMA -- a heurística lexical de "achado novo" NUNCA mais
       decide pass/fail por conta própria; o pesquisador revisa os 4
       cenários manualmente e preenche o veredito antes de prosseguir.

  ESTÁGIO 2 (segunda invocação, com `--synthesis-manual-review-file
  <caminho>`): lê o template preenchido; se os 4 cenários tiverem
  `manual_review_verdict == "sem_achado_novo"` E os controles de envio de
  imagem da invocação anterior (mesmo `--output-dir`) já tiverem passado,
  prossegue para a matriz principal (92 chamadas,
  `src.multiagent.vlm_direct.build_planned_calls`), respeitando a lógica
  de pausa por abstenção sistemática (3 contadores por papel, nunca
  retry). Qualquer veredito ausente ou `"achado_novo_encontrado"` PARA a
  execução com mensagem clara -- nunca assume/adivinha.

Total de chamadas planejadas por modelo = 108 = 92 (matriz) + 16
(controles: 12 de envio de imagem + 4 de síntese) -- 108 é um TETO
OPERACIONAL de 4h para a execução COMPLETA de um modelo (estágio 1 +
estágio 2), não uma meta de desempenho; passar do teto não estabelece por
si só viabilidade para os 3.000 reais (ver projeção de duração no
`report.md` do estágio 2).

Modelos a testar (Passo 7 da entrega): `qwen3-vl:8b-instruct-q4_K_M` e
`medgemma1.5:4b-it-q4_K_M`. Rodar este script (os dois estágios) uma vez
para CADA um (`--model-tag` diferente), independentemente -- este script
NÃO encadeia os dois modelos automaticamente. Se, ao concluir os dois,
apenas UM passou nos controles de pré-execução, o pesquisador deve rodar
este mesmo script (os dois estágios) uma TERCEIRA vez com
`--model-tag qwen2.5vl:7b-q4_K_M` (modelo de contingência) -- esta decisão
é indicada no relatório Markdown final de cada execução, não decidida/
encadeada automaticamente aqui (este assistente não pode executar nada
para descobrir ao vivo qual passa).

STATUS: implementado, aguardando execução do pesquisador (requer Ollama
rodando localmente com os modelos já baixados, `pip install psutil
ollama` se ainda não instalados). Não executado pelo assistente que o
escreveu (`feedback_no_autonomous_execution`).

Uso (dois estágios, por modelo):
    # Estágio 1: controles + template de revisão manual
    python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M
    # (pesquisador revisa e preenche experiments/multiagent/local_vlm_pilot/
    #  qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json)
    # Estágio 2: matriz principal
    python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M \\
        --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_manual_review_template.json

    python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M
    python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M \\
        --synthesis-manual-review-file experiments/multiagent/local_vlm_pilot/medgemma1.5_4b-it-q4_K_M/synthesis_manual_review_template.json

    # Só se necessário (ver docstring acima): modelo de contingência, mesmo fluxo de 2 estágios
    python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M
    python scripts/run_local_vlm_pilot.py --model-tag qwen2.5vl:7b-q4_K_M --synthesis-manual-review-file ...
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

from PIL import Image  # noqa: E402

from src.evaluation import case_status_v2  # noqa: E402
from src.multiagent import vlm_direct as vd  # noqa: E402

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_SAMPLING_RESULTS_JSON = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_sampling_results.json")
DEFAULT_IMAGES_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_images")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot")

# Projeção de duração (item 10 da entrega) -- N hipotético usado só como
# referência no report.md, não uma meta real do piloto.
HYPOTHETICAL_FULL_SCALE_N = 3000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", required=True, help="Tag Ollama exata, ex. qwen3-vl:8b-instruct-q4_K_M")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sampling-results-json", default=DEFAULT_SAMPLING_RESULTS_JSON)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    parser.add_argument(
        "--synthesis-manual-review-file",
        default=None,
        help=(
            "ESTÁGIO 2: caminho do synthesis_manual_review_template.json já "
            "preenchido pelo pesquisador (gerado pelo ESTÁGIO 1, invocação "
            "sem este argumento). Quando fornecido, pula direto para a "
            "matriz principal de 92 chamadas, reusando o resultado dos "
            "controles de envio de imagem já gravado em --output-dir."
        ),
    )
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


def _resolve_output_dir(output_dir_root: Path, model_name: str) -> Path:
    output_dir = output_dir_root / model_name.replace(":", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main() -> None:
    args = build_arg_parser().parse_args()
    model_name = args.model_tag
    output_dir = _resolve_output_dir(Path(args.output_dir), model_name)

    rows = _load_pilot12_rows(Path(args.manifest_path))
    designated_image_ids = _resolve_designated_image_ids(rows, Path(args.sampling_results_json))

    if args.synthesis_manual_review_file:
        _run_stage2_matrix(args, output_dir, model_name, rows, designated_image_ids)
    else:
        _run_stage1_controls(args, output_dir, model_name)


# ---------------------------------------------------------------------------
# ESTÁGIO 1 — controles de envio de imagem (por papel) + controles de
# síntese + template de revisão manual. PARA sempre ao final (nunca
# prossegue para a matriz na mesma invocação).
# ---------------------------------------------------------------------------
def _run_stage1_controls(args: argparse.Namespace, output_dir: Path, model_name: str) -> None:
    print(f"=== Piloto local de VLM via Ollama -- modelo: {model_name} -- ESTÁGIO 1 (controles) ===")

    print("--- ollama.show(modelo) ---")
    try:
        model_info = vd.get_model_info(model_name)
    except Exception as exc:  # noqa: BLE001 -- registrado, não interrompe a escrita do relatório parcial
        model_info = {"error": str(exc)}
        print(f"AVISO: falha ao obter informações do modelo via ollama.show: {exc}")

    all_records: list[vd.CallRecord] = []

    # --- 1. Controles de envio de imagem, POR PAPEL VISUAL ---
    print("--- Controles de envio de imagem (4 papéis x [2 imagem + 1 sem imagem] = 12 chamadas) ---")
    results_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in vd.VISUAL_ROLES:
        role_results = []
        for spec in vd.IMAGE_CONTROL_SPECS_BY_ROLE[role]:
            result = vd.run_image_sending_control_call(model_name, role, "imagem", spec, timeout_s=args.timeout_s)
            all_records.append(result["call_record"])
            role_results.append(result)
            print(f"  [{role}] controle imagem '{result['name']}' status={result['call_record'].call_status} payload_correto={result['payload_correct']}")
        no_image_result = vd.run_image_sending_control_call(model_name, role, "sem_imagem", None, timeout_s=args.timeout_s)
        all_records.append(no_image_result["call_record"])
        role_results.append(no_image_result)
        print(f"  [{role}] controle sem imagem status={no_image_result['call_record'].call_status} payload_sem_imagem_confirmado={no_image_result['payload_correct']}")
        results_by_role[role] = role_results

    image_controls_eval = vd.evaluate_image_sending_controls(results_by_role)
    print(f"controles de envio de imagem: overall_pass={image_controls_eval['overall_pass']} (por papel: {image_controls_eval['n_roles_passed']}/{image_controls_eval['n_roles_evaluated']})")

    if not image_controls_eval["overall_pass"]:
        _write_stop_report(
            output_dir, model_name, model_info, all_records,
            stage="controle_de_envio_de_imagem",
            controls={"image_sending": image_controls_eval},
        )
        print("PARADO: controle de envio de imagem reprovado (ao menos um papel visual) -- matriz principal NÃO tentada.")
        return

    # --- 2. Controles de síntese ---
    print("--- Controles de síntese (4 cenários fixos) ---")
    synthesis_results = []
    for scenario_name in vd.FIXED_SYNTHESIS_SCENARIOS:
        result = vd.run_synthesis_control(model_name, scenario_name, timeout_s=args.timeout_s)
        all_records.append(result["call_record"])
        synthesis_results.append({k: v for k, v in result.items() if k != "call_record"})
        print(
            f"  cenário={scenario_name} parseavel(passed)={result['passed']} "
            f"classificado_v2_oficial={result['classified_status_v2']} "
            f"classificado_v1_historico={result['classified_status_v1_historical']} "
            f"achado_novo_candidatos={result['novel_clinical_terms_candidates']}"
        )

    synthesis_overall_pass = all(r["passed"] for r in synthesis_results)
    print(
        f"controles de síntese (gate oficial 2026-11-14 -- parseável pela A34 via parser "
        f"v2 '{case_status_v2.PARSER_VERSION}', v1 só rastreabilidade): overall_pass={synthesis_overall_pass}"
    )

    if not synthesis_overall_pass:
        _write_stop_report(
            output_dir, model_name, model_info, all_records,
            stage="controle_de_sintese",
            controls={"image_sending": image_controls_eval, "synthesis": synthesis_results},
        )
        print("PARADO: ao menos um cenário de síntese não produziu saída parseável pela A34 -- matriz principal NÃO tentada.")
        return

    # --- 3. Template de revisão manual -- SEMPRE escrito e SEMPRE para aqui
    #     (decisão fechada #4: achado clínico novo nunca decide pass/fail
    #     por conta própria; é sempre o pesquisador que decide). ---
    manual_review_template = vd.build_synthesis_manual_review_template(synthesis_results)
    template_path = output_dir / "synthesis_manual_review_template.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(manual_review_template, f, ensure_ascii=False, indent=2, default=str)

    _write_stop_report(
        output_dir, model_name, model_info, all_records,
        stage="aguardando_revisao_manual_sintese",
        controls={"image_sending": image_controls_eval, "synthesis": synthesis_results},
        extra_summary_fields={"synthesis_manual_review_template_path": str(template_path)},
    )
    print(
        "\nESTÁGIO 1 CONCLUÍDO -- controles de envio de imagem e de síntese "
        "(parsing) passaram. Revise MANUALMENTE os 4 cenários de síntese e "
        f"preencha o veredito em:\n  {template_path}\n"
        "Depois rode novamente este script com "
        f"--synthesis-manual-review-file {template_path} para prosseguir à "
        "matriz principal (92 chamadas). A matriz NÃO é tentada "
        "automaticamente nesta invocação."
    )


# ---------------------------------------------------------------------------
# ESTÁGIO 2 — matriz principal (92 chamadas), só prossegue se o veredito
# manual de síntese estiver todo "sem_achado_novo" E os controles de envio
# de imagem da invocação anterior já tiverem passado.
# ---------------------------------------------------------------------------
def _run_stage2_matrix(
    args: argparse.Namespace,
    output_dir: Path,
    model_name: str,
    rows: list[dict[str, str]],
    designated_image_ids: list[str],
) -> None:
    print(f"=== Piloto local de VLM via Ollama -- modelo: {model_name} -- ESTÁGIO 2 (matriz principal) ===")

    stage1_summary_path = output_dir / "summary.json"
    if not stage1_summary_path.exists():
        raise SystemExit(
            f"ERRO: {stage1_summary_path} não encontrado -- rode este script SEM "
            "--synthesis-manual-review-file primeiro (ESTÁGIO 1) neste mesmo --output-dir."
        )
    with open(stage1_summary_path, encoding="utf-8") as f:
        stage1_summary = json.load(f)

    if stage1_summary.get("stopped_at_stage") != "aguardando_revisao_manual_sintese":
        raise SystemExit(
            f"ERRO: o ESTÁGIO 1 gravado em {stage1_summary_path} parou em "
            f"'{stage1_summary.get('stopped_at_stage')}', não em "
            "'aguardando_revisao_manual_sintese' -- os controles de envio de "
            "imagem e/ou de síntese (parsing) não passaram. Corrija o ESTÁGIO 1 "
            "antes de tentar a matriz principal."
        )
    image_controls_eval = stage1_summary["controls"]["image_sending"]
    if not image_controls_eval.get("overall_pass"):
        raise SystemExit("ERRO: controles de envio de imagem do ESTÁGIO 1 não passaram -- matriz principal bloqueada.")

    review_path = Path(args.synthesis_manual_review_file)
    if not review_path.exists():
        raise SystemExit(f"ERRO: arquivo de revisão manual não encontrado: {review_path}")
    with open(review_path, encoding="utf-8") as f:
        manual_review_data = json.load(f)
    review_eval = vd.evaluate_manual_review_file(manual_review_data)
    print(f"revisão manual de síntese: {review_eval}")
    if not review_eval["all_scenarios_cleared"]:
        print(
            "PARADO: revisão manual de síntese incompleta/reprovada -- matriz "
            "principal NÃO tentada. Detalhe:\n"
            f"  cenários sem veredito: {review_eval['missing_verdict_scenarios']}\n"
            f"  cenários com achado novo encontrado: {review_eval['novel_finding_scenarios']}\n"
            f"  vereditos inválidos (fora do vocabulário aceito): {review_eval['invalid_verdict_scenarios']}"
        )
        return

    print("--- Matriz principal: 92 chamadas planejadas ---")
    planned_calls = vd.build_planned_calls(rows, designated_image_ids)
    print(f"chamadas planejadas: {len(planned_calls)} (deve ser 92; total da execução do modelo incluindo controles = {vd.N_TOTAL_PLANNED_CALLS_PER_MODEL})")

    # Baseline de swap: capturado AGORA, explicitamente -- modelo já
    # carregado (ESTÁGIO 1 já rodou controles com este model_name) e
    # controles já completos. Nunca derivado da própria lista de amostras
    # da matriz (decisão fechada #6).
    baseline_snapshot = vd.capture_memory_snapshot() if vd.psutil is not None else None
    baseline_swap_bytes = baseline_snapshot.system_swap_used_bytes if baseline_snapshot is not None else None
    print(f"baseline de swap pós-controles (pré-matriz): {baseline_swap_bytes}")

    tracker = vd.AbstentionTracker()
    accumulated_state: dict[tuple[str, str], dict[str, Any]] = {}
    matrix_records: list[vd.CallRecord] = []

    images_dir = Path(args.images_dir)
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
            f"status={record.call_status} classificado_v1={record.classified_status} "
            f"classificado_v2={record.classified_status_v2}"
        )

    _write_full_report(
        output_dir, model_name, stage1_summary, matrix_records, planned_calls, tracker,
        image_controls_eval=image_controls_eval,
        synthesis_controls=stage1_summary["controls"].get("synthesis", []),
        manual_review_eval=review_eval,
        baseline_swap_bytes=baseline_swap_bytes,
    )
    print(f"\nRelatórios/logs gravados em: {output_dir}")


# ---------------------------------------------------------------------------
# Escrita de relatórios/logs
# ---------------------------------------------------------------------------
def _write_call_log(output_dir: Path, records: list["vd.CallRecord"], filename: str = "call_log.jsonl") -> Path:
    log_path = output_dir / filename
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_flat_dict(), ensure_ascii=False, default=str) + "\n")
    return log_path


def _serialize_controls(controls: dict) -> dict:
    return json.loads(json.dumps(controls, default=str, ensure_ascii=False))


def _write_stop_report(
    output_dir: Path, model_name: str, model_info: dict, records: list, stage: str, controls: dict,
    extra_summary_fields: Optional[dict[str, Any]] = None,
) -> None:
    log_path = _write_call_log(output_dir, records)
    summary = {
        "model_name": model_name,
        "model_info": model_info,
        "stopped_before_main_matrix": True,
        "stopped_at_stage": stage,
        # GATE OFICIAL registrado explicitamente (decisão do pesquisador,
        # 2026-11-14, ver docs/reconstruction/20_local_vlm_pilot.md §9c):
        # versão EXATA do parser de classificação que decidiu "parseable"/
        # "passed" dos controles de síntese (quando esta etapa os
        # alcançou) -- v1 (classify_case_status) é mantido só para
        # rastreabilidade, nunca como fallback/desempate.
        "official_classification_parser_version": case_status_v2.PARSER_VERSION,
        "controls": _serialize_controls(controls),
        "n_calls_executed_total": len(records),
        "n_total_planned_calls_per_model": vd.N_TOTAL_PLANNED_CALLS_PER_MODEL,
    }
    if extra_summary_fields:
        summary.update(extra_summary_fields)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    report_lines = [
        f"# Piloto local de VLM — {model_name} — ESTÁGIO 1 PARADO (stage=`{stage}`)",
        "",
        f"- Etapa em que parou: `{stage}`",
        "- Matriz principal de 92 chamadas: NÃO tentada.",
        f"- Log de chamadas (controles): `{log_path}`",
        f"- Total de chamadas planejadas por modelo (controles + matriz, teto operacional de 4h): {vd.N_TOTAL_PLANNED_CALLS_PER_MODEL}",
        f"- Parser OFICIAL de classificação/gate (decisão 2026-11-14): `{case_status_v2.PARSER_VERSION}` "
        "(v1/`classify_case_status` mantido só para rastreabilidade -- nunca fallback/desempate)",
        "",
    ]
    if extra_summary_fields and "synthesis_manual_review_template_path" in extra_summary_fields:
        report_lines += [
            "## Próximo passo: revisão manual obrigatória",
            "",
            "A heurística lexical de achado clínico novo NUNCA decide pass/fail "
            "por conta própria (decisão fechada #4) -- revise manualmente:",
            f"`{extra_summary_fields['synthesis_manual_review_template_path']}`",
            "",
            "Depois rode novamente com `--synthesis-manual-review-file <esse caminho>`.",
            "",
        ]
    report_lines += [
        "## Controles",
        "",
        f"```json\n{json.dumps(_serialize_controls(controls), ensure_ascii=False, indent=2, default=str)}\n```",
        "",
    ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def _write_full_report(
    output_dir: Path,
    model_name: str,
    stage1_summary: dict,
    matrix_records: list["vd.CallRecord"],
    planned_calls: list,
    tracker: "vd.AbstentionTracker",
    image_controls_eval: dict,
    synthesis_controls: list,
    manual_review_eval: dict,
    baseline_swap_bytes: Optional[int],
) -> None:
    matrix_log_path = _write_call_log(output_dir, matrix_records, filename="call_log_matriz_principal.jsonl")

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

    # DECISÃO FECHADA #9 (extensão): "zero parecer conclusivo" só se aplica
    # a vlm_direto/summary -- únicos papéis com campo de diagnóstico
    # categorizável por classify_case_status[_v2].
    #
    # GATE/COMPUTAÇÃO OFICIAL trocado para v2 (2026-11-14, decisão verbatim
    # do pesquisador, ver docs/reconstruction/20_local_vlm_pilot.md §9c):
    # esta contagem agora lê `classified_status_v2`, não `classified_status`
    # (v1) -- aplicado IDENTICAMENTE para vlm_direto e sumarizador, sem caso
    # especial por papel, mesma forma que o restante da troca de gate.
    _ROLES_WITH_CLASSIFIABLE_OUTPUT = {vd.ROLE_VLM_DIRETO, vd.ROLE_SUMMARY}
    roles_with_zero_conclusive: list[str] = []
    for key, d in by_role_condition.items():
        role = key.split("__")[0]
        if role not in _ROLES_WITH_CLASSIFIABLE_OUTPUT:
            continue
        has_conclusive = any(
            r.classified_status_v2 in ("positivo", "negativo", "indeterminado")
            for r in matrix_records
            if f"{r.role}__{r.condition}" == key
        )
        if not has_conclusive and key not in roles_with_zero_conclusive:
            roles_with_zero_conclusive.append(key)

    # Relatório INFORMATIVO (nunca pass/fail) para ABCD/Menzies/SPCL.
    diagnostic_role_reporting = {
        role: vd.summarize_diagnostic_role_reporting(matrix_records, role)
        for role in (vd.ROLE_ABCD, vd.ROLE_MENZIES, vd.ROLE_SPCL)
    }

    n_valid_first_attempt = sum(1 for r in matrix_records if r.call_status == vd.STATUS_OK)
    valid_rate = n_valid_first_attempt / n_executed if n_executed else 0.0

    # OFICIAL (2026-11-14): cobertura binária reportada agora vem do v2 --
    # mesma troca de gate, "da mesma forma" para vlm_direto e sumarizador
    # (não há ramificação por papel em classify_binary_coverage).
    binary_statuses_v2 = [vd.classify_binary_coverage(r.classified_status_v2) for r in matrix_records]
    n_positivo = sum(1 for b in binary_statuses_v2 if b is True)
    n_negativo = sum(1 for b in binary_statuses_v2 if b is False)

    # Rastreabilidade SÓ (nunca mais a fonte oficial dos números do
    # relatório): mesma contagem a partir de `classified_status` (v1,
    # intocado) -- mantido para comparação histórica, nunca lido por
    # nenhum gate/critério de aprovação.
    binary_statuses_v1_historical = [vd.classify_binary_coverage(r.classified_status) for r in matrix_records]
    n_positivo_v1_historical = sum(1 for b in binary_statuses_v1_historical if b is True)
    n_negativo_v1_historical = sum(1 for b in binary_statuses_v1_historical if b is False)

    # Candidatos a recusa -- surfaced SEMPRE que houver qualquer
    # STATUS_REFUSAL na matriz, não só na 3ª consecutiva que dispara pausa
    # (decisão fechada #5).
    refusal_candidates = [
        {
            "role": r.role, "condition": r.condition, "lesion_id": r.lesion_id, "image_id": r.image_id,
            "raw_response_text": r.raw_response_text, "failure_cause": r.failure_cause,
        }
        for r in matrix_records if r.call_status == vd.STATUS_REFUSAL
    ]

    # Swap: janelas de 5 sobre a MATRIZ PRINCIPAL, com baseline explícito
    # pós-controles (decisão fechada #6).
    matrix_swap_samples: list[int] = []
    for r in matrix_records:
        if r.mem_before is not None:
            matrix_swap_samples.append(r.mem_before.system_swap_used_bytes)
        if r.mem_after is not None:
            matrix_swap_samples.append(r.mem_after.system_swap_used_bytes)
    if matrix_swap_samples and baseline_swap_bytes is not None:
        swap_growth = vd.detect_sustained_swap_growth(matrix_swap_samples, baseline_swap_bytes=baseline_swap_bytes)
    else:
        swap_growth = {
            "note": "psutil indisponível, baseline ausente, ou nenhuma amostra coletada -- não computado.",
            "baseline_swap_bytes": baseline_swap_bytes,
        }

    # Projeção de duração (item 10 -- extensão: soma sequencial dos 4
    # papéis por lesão+condição, projeção para HYPOTHETICAL_FULL_SCALE_N).
    per_lesion_condition_totals = vd.compute_per_lesion_condition_elapsed_totals(matrix_records)
    duration_projection = vd.project_full_scale_duration(
        list(per_lesion_condition_totals.values()), HYPOTHETICAL_FULL_SCALE_N
    )

    summary = {
        "model_name": model_name,
        "model_info": stage1_summary.get("model_info"),
        "stopped_before_main_matrix": False,
        "n_calls_planned_main_matrix": n_planned,
        "n_calls_executed_main_matrix": n_executed,
        "n_calls_not_executed_due_to_pause": n_not_executed,
        "n_total_planned_calls_per_model": vd.N_TOTAL_PLANNED_CALLS_PER_MODEL,
        "valid_on_first_attempt_rate": valid_rate,
        # GATE OFICIAL registrado explicitamente (decisão do pesquisador,
        # 2026-11-14, ver docs/reconstruction/20_local_vlm_pilot.md §9c):
        # a versão EXATA do parser de classificação usado para o
        # gate/classificação oficial deste relatório -- nunca mantida
        # silenciosamente quando o parser mudar (PARSER_VERSION é trocado
        # junto, ver case_status_v2.py).
        "official_classification_parser_version": case_status_v2.PARSER_VERSION,
        "official_classification_parser_note": (
            "classify_case_status_v2 (v2, 'conclusão final') é o parser "
            "OFICIAL de todo gate/classificação reportada aqui -- "
            "binary_coverage, roles_with_zero_conclusive_opinion, e o gate "
            "de parseabilidade dos controles de síntese do Estágio 1. v1 "
            "(classify_case_status) é mantido só para RASTREABILIDADE (ver "
            "binary_coverage_v1_historical_informational abaixo e "
            "classified_status em call_log_matriz_principal.jsonl) -- "
            "nunca usado como fallback/desempate."
        ),
        "binary_coverage": {
            "positivo": n_positivo,
            "negativo": n_negativo,
            "parser_version": case_status_v2.PARSER_VERSION,
            "nota_proxy": (
                "'negativo' aqui é o status 'negativo' de "
                "classify_case_status_v2 (variantes de 'lesão benigna' na "
                "região de conclusão final -- ver case_status_v2.py), usado "
                "como PROXY de binary_target='não_melanoma' -- convenção "
                "existente do projeto, não um diagnóstico literal de "
                "benignidade."
            ),
        },
        # RASTREABILIDADE SÓ (2026-11-14): mesma cobertura binária, mas a
        # partir do classificador v1 histórico (classify_case_status) --
        # nunca mais a fonte oficial deste relatório (trocado para v2, ver
        # "official_classification_parser_version" acima e
        # docs/reconstruction/20_local_vlm_pilot.md §9c). Mantido só para
        # comparação/auditoria entre versões do parser.
        "binary_coverage_v1_historical_informational": {
            "positivo": n_positivo_v1_historical,
            "negativo": n_negativo_v1_historical,
        },
        "by_role_condition": by_role_condition,
        "roles_with_zero_conclusive_opinion": roles_with_zero_conclusive,
        "diagnostic_role_reporting_informational": diagnostic_role_reporting,
        "abstention_counters_trace": tracker.trace(),
        "controls": {
            "image_sending": _serialize_controls(image_controls_eval),
            "synthesis": _serialize_controls(synthesis_controls),
            "manual_review_evaluation": _serialize_controls(manual_review_eval),
        },
        "sustained_swap_growth": _serialize_controls(swap_growth),
        "duration_projection_full_scale": _serialize_controls(duration_projection),
        "refusal_candidates_for_manual_review": refusal_candidates,
        "seed_note": (
            "A semente 20270203 (e a sub-semente nova de redistribuição de "
            "grupo, ver scripts/run_pilot12_sample.py) governam só a "
            "SELEÇÃO das lesões -- não garantem determinismo das respostas "
            "do VLM (inferência local via Ollama não é seedada por este valor)."
        ),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    report_lines = [
        f"# Piloto local de VLM — {model_name} — ESTÁGIO 2 (matriz principal concluída)",
        "",
        f"- Log de chamadas da matriz principal: `{matrix_log_path}`",
        f"- Log de chamadas dos controles (ESTÁGIO 1): ver `{output_dir / 'call_log.jsonl'}`",
        f"- Chamadas planejadas na matriz: {n_planned} (deve ser 92)",
        f"- Total de chamadas planejadas por modelo (controles + matriz): {vd.N_TOTAL_PLANNED_CALLS_PER_MODEL}",
        f"- Chamadas efetivamente realizadas na matriz: {n_executed}",
        f"- Chamadas NÃO executadas (pausa por abstenção): {n_not_executed}",
        f"- Taxa de validade na primeira tentativa: {valid_rate:.2%}",
        f"- Parser OFICIAL de classificação/gate (decisão 2026-11-14): `{case_status_v2.PARSER_VERSION}` "
        "(v1/`classify_case_status` mantido só para rastreabilidade -- nunca fallback/desempate)",
        f"- Cobertura binária (OFICIAL, via classificador v2): positivo={n_positivo}, negativo={n_negativo} "
        "('negativo' é proxy de não_melanoma via classify_case_status_v2, não diagnóstico literal de benignidade)",
        f"- Cobertura binária (v1 histórico, só rastreabilidade): positivo={n_positivo_v1_historical}, negativo={n_negativo_v1_historical}",
        "",
        "## Papéis x condições sem nenhum parecer conclusivo (positivo/negativo/indeterminado via classified_status_v2 oficial) -- só vlm_direto/summary",
        "",
        f"{roles_with_zero_conclusive if roles_with_zero_conclusive else 'nenhum'}",
        "",
        "## Relatório informativo (NÃO é pass/fail) -- ABCD/Menzies/SPCL",
        "",
        f"```json\n{json.dumps(diagnostic_role_reporting, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Candidatos a recusa para revisão manual (TODA chamada classificada STATUS_REFUSAL, não só a 3ª consecutiva)",
        "",
        (
            f"```json\n{json.dumps(refusal_candidates, ensure_ascii=False, indent=2, default=str)}\n```"
            if refusal_candidates else "nenhuma chamada classificada como recusa explícita nesta execução."
        ),
        "",
        "## Rastro dos contadores de abstenção (por papel)",
        "",
        f"```json\n{json.dumps(tracker.trace(), ensure_ascii=False, indent=2)}\n```",
        "",
        "## Crescimento sustentado de swap (decisão fechada #6 -- janelas de 5 sobre a matriz principal, baseline pós-controles)",
        "",
        f"```json\n{json.dumps(swap_growth, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Projeção de duração em escala real (informativa -- NÃO é promessa de tempo real)",
        "",
        f"```json\n{json.dumps(duration_projection, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Controles (ESTÁGIO 1, reusados aqui)",
        "",
        f"```json\n{json.dumps(_serialize_controls(image_controls_eval), ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        f"```json\n{json.dumps(_serialize_controls(synthesis_controls), ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Avaliação da revisão manual de síntese",
        "",
        f"```json\n{json.dumps(manual_review_eval, ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        "## Nota sobre a semente",
        "",
        summary["seed_note"],
        "",
    ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
