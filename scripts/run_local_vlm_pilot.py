"""Etapa 5 — driver único do piloto de VLMs sobre as 12 lesões fora da
amostra congelada (ver `docs/reconstruction/20_local_vlm_pilot.md`,
`docs/reconstruction/30_real_lesion_evaluation_protocol.md` e
`src/multiagent/vlm_direct.py`, que concentra toda a lógica real reusada
aqui).

MUDANÇA DE PROTOCOLO (2026-11-16, decisão do pesquisador, ver doc 30):
este script deixou de ter duas invocações separadas por um gate manual.
Numa ÚNICA invocação por `--model-tag`, ele agora:

  1. Roda checagens OPERACIONAIS (manifesto, imagens, config/credenciais
     do modelo, diretório de saída gravável) -- PARA aqui, com
     `SystemExit`, se qualquer uma falhar. Isto é o único tipo de
     problema que interrompe o lote.
  2. Roda os 12 controles de envio de imagem + os 4 controles de síntese
     -- sempre executados e sempre registrados no relatório, mas como
     RESULTADO (desempenho/fidelidade), nunca mais como gate. Em
     particular: acertar quadrante/cor numa imagem de controle, a saída
     de um controle de síntese ser parseável pela A34, ou o veredito de
     fidelidade da síntese (manual, opcional) -- nenhum desses bloqueia
     a matriz principal. O ÚNICO aspecto desses controles que seria
     operacional (o payload multimodal enviado estar correto -- ver
     `payload_correct`/`operational_pass` em
     `src.multiagent.vlm_direct.evaluate_image_sending_controls`) ainda é
     checado, e uma falha aí indica um bug real deste pipeline, não do
     modelo.
  3. Prossegue DIRETO para a matriz principal (92 chamadas,
     `src.multiagent.vlm_direct.build_planned_calls`), respeitando a
     lógica de pausa por abstenção sistemática (3 contadores por papel,
     nunca retry) -- essa pausa é uma salvaguarda operacional (evita
     gastar o restante do orçamento de chamadas quando um papel está
     sistematicamente sem resposta/com falha técnica), não um critério
     de qualidade de conteúdo.

Histórico preservado, não removido: `--synthesis-fidelity-review-file`
continua aceito como argumento OPCIONAL -- se fornecido (um
`synthesis_fidelity_review_template.json` já preenchido pelo
pesquisador, ou reconstruído por
`scripts/rebuild_synthesis_fidelity_review_from_stage1.py`), seu
conteúdo é anexado ao relatório final como documentação da revisão de
fidelidade; se omitido, a matriz roda e o relatório final indica
explicitamente que a revisão de fidelidade ainda não foi anexada --
nunca bloqueia.

Preservado sem alteração: o contrato `decisao_final` (vlm_direto e
sumarizador oficial), a extração de template do medgemma (fora deste
script -- ver `src/evaluation/response_template_extraction.py`), a
distinção indeterminado/resposta-inválida/falha-técnica, "sem retry,
sem fallback para v1/v2", e a política de máscara vazia de S2 (definida
em `src/multiagent/presentation.py`, não tocada por este script).

Modelos candidatos (ver `src.multiagent.vlm_direct.MAIN_EVALUATION_MODELS`):
Gemini, GPT, Qwen3-VL e MedGemma 1.5 4B -- todos sob a MESMA regra.
Qwen2.5-VL está registrado como NÃO avaliado por indisponibilidade de
download (sem substituto) -- ver doc 30.

STATUS: implementado, aguardando execução do pesquisador (requer Ollama
rodando localmente com os modelos Ollama já baixados para
Qwen3-VL/MedGemma, ou `OPENAI_API_KEY`/`GOOGLE_API_KEY` definidos para
GPT/Gemini; `pip install psutil ollama` se ainda não instalados). Não
executado pelo assistente que o escreveu (`feedback_no_autonomous_execution`).

Uso (uma única invocação por modelo):
    python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M
    python scripts/run_local_vlm_pilot.py --model-tag medgemma1.5:4b-it-q4_K_M
    python scripts/run_local_vlm_pilot.py --model-tag gemini-3.1-flash-lite
    python scripts/run_local_vlm_pilot.py --model-tag gpt-5.4-nano

    # Opcional, pode ser rodado antes OU depois da matriz, e anexado numa
    # nova invocação (reaproveita o output-dir, não repete nenhuma
    # chamada): revisão de fidelidade da síntese já preenchida.
    python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M \\
        --synthesis-fidelity-review-file experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/synthesis_fidelity_review_template.json
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
from src.evaluation.explicit_decision_contract import (  # noqa: E402
    DECISAO_INDETERMINADO,
    DECISAO_MELANOMA,
    DECISAO_NAO_MELANOMA,
    DECISION_STATUS_OK as DECISION_CONTRACT_STATUS_OK,
    PARSER_VERSION as DECISION_CONTRACT_PARSER_VERSION,
)
from src.multiagent import vlm_direct as vd  # noqa: E402

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_SAMPLING_RESULTS_JSON = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_sampling_results.json")
DEFAULT_IMAGES_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_images")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot")

# Projeção de duração -- N hipotético usado só como referência no
# report.md, não uma meta real do piloto.
HYPOTHETICAL_FULL_SCALE_N = 3000


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-tag", required=True, help="Ex.: qwen3-vl:8b-instruct-q4_K_M, medgemma1.5:4b-it-q4_K_M, gemini-3.1-flash-lite, gpt-5.4-nano")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--sampling-results-json", default=DEFAULT_SAMPLING_RESULTS_JSON)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    parser.add_argument(
        "--synthesis-fidelity-review-file",
        default=None,
        help=(
            "OPCIONAL (deixou de ser pré-requisito em 2026-11-16, ver doc "
            "30): caminho do synthesis_fidelity_review_template.json já "
            "preenchido pelo pesquisador. Se fornecido, seu conteúdo é "
            "anexado ao relatório final como documentação da revisão de "
            "fidelidade -- nunca condiciona a execução da matriz principal, "
            "que já roda na mesma invocação independentemente deste "
            "argumento."
        ),
    )
    return parser


def _load_pilot12_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Checagem OPERACIONAL: manifesto com o número errado de linhas é um
    problema de configuração, não de desempenho -- interrompe aqui."""
    if not manifest_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: manifesto não encontrado: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 12:
        raise SystemExit(f"ERRO OPERACIONAL: {manifest_path} tem {len(rows)} linhas, esperado exatamente 12.")
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
    """Checagem OPERACIONAL: imagem da condição ausente/trocada é um
    problema de preparação de dados, não de desempenho do modelo --
    interrompe o lote (propagado como FileNotFoundError, nunca
    engolido)."""
    path = images_dir / condition / f"{image_id}_{condition}.png"
    if not path.exists():
        raise FileNotFoundError(
            f"ERRO OPERACIONAL: imagem da condição {condition} não encontrada para "
            f"image_id={image_id}: {path} -- rode scripts/run_pilot12_image_preparation.py primeiro."
        )
    return Image.open(path).convert("RGB")


def _resolve_output_dir(output_dir_root: Path, model_name: str) -> Path:
    output_dir = output_dir_root / model_name.replace(":", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Checagem OPERACIONAL: diretório de saída precisa ser gravável ANTES
    # de qualquer chamada real -- descobrir isso só depois de gastar o
    # orçamento de chamadas do piloto seria o pior momento possível.
    probe_path = output_dir / ".write_check"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        raise SystemExit(f"ERRO OPERACIONAL: não foi possível gravar em {output_dir}: {exc}")
    return output_dir


def main() -> None:
    args = build_arg_parser().parse_args()
    model_name = args.model_tag
    output_dir = _resolve_output_dir(Path(args.output_dir), model_name)

    rows = _load_pilot12_rows(Path(args.manifest_path))
    designated_image_ids = _resolve_designated_image_ids(rows, Path(args.sampling_results_json))

    fidelity_review_data = None
    if args.synthesis_fidelity_review_file:
        review_path = Path(args.synthesis_fidelity_review_file)
        if not review_path.exists():
            raise SystemExit(f"ERRO OPERACIONAL: arquivo de revisão de fidelidade não encontrado: {review_path}")
        with open(review_path, encoding="utf-8") as f:
            fidelity_review_data = json.load(f)

    _run_single_stage(args, output_dir, model_name, rows, designated_image_ids, fidelity_review_data)


# ---------------------------------------------------------------------------
# Fluxo único: checagens operacionais -> controles (resultado, não gate)
# -> matriz principal, tudo na mesma invocação.
# ---------------------------------------------------------------------------
def _run_single_stage(
    args: argparse.Namespace,
    output_dir: Path,
    model_name: str,
    rows: list[dict[str, str]],
    designated_image_ids: list[str],
    fidelity_review_data: Optional[dict[str, Any]],
) -> None:
    print(f"=== Piloto de VLM -- modelo: {model_name} ===")

    print("--- Checagem operacional de modelo/configuração ---")
    # get_model_info levanta (SystemExit indireto via RuntimeError não
    # capturado aqui) se o modelo Ollama não existir localmente, ou se a
    # variável de ambiente de API esperada (GOOGLE_API_KEY/OPENAI_API_KEY)
    # não estiver definida para Gemini/GPT -- problema OPERACIONAL real
    # (configuração incompatível), nunca um aviso que deixa a execução
    # continuar silenciosamente.
    model_info = vd.get_model_info(model_name)
    print(f"  modelo OK: {model_info}")

    all_records: list[vd.CallRecord] = []

    # --- 1. Controles de envio de imagem, POR PAPEL VISUAL (RESULTADO) ---
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
    print(
        f"controles de envio de imagem: operational_pass={image_controls_eval['operational_pass']} "
        f"(payload correto em todos os papéis) -- overall_pass (inclui acerto de cor/quadrante, "
        f"RESULTADO, não gate)={image_controls_eval['overall_pass']} "
        f"({image_controls_eval['n_roles_passed']}/{image_controls_eval['n_roles_evaluated']} papéis)"
    )
    if not image_controls_eval["operational_pass"]:
        # ÚNICO motivo operacional real para parar aqui: o payload enviado
        # não era o esperado -- bug deste pipeline (mensagem/imagem
        # trocada), não desempenho do modelo.
        _write_report(
            output_dir, model_name, model_info, all_records,
            image_controls_eval=image_controls_eval, synthesis_results=[], fidelity_review_data=fidelity_review_data,
            matrix_records=[], planned_calls=[], tracker=vd.AbstentionTracker(), baseline_swap_bytes=None,
            stopped_operationally=True,
            stop_reason=(
                "payload multimodal incorreto em ao menos um controle de envio de imagem -- "
                "ver controls.image_sending.per_role para o papel/controle específico."
            ),
        )
        print("PARADO (problema OPERACIONAL): payload de controle de imagem incorreto -- matriz principal NÃO tentada.")
        return

    # --- 2. Controles de síntese (RESULTADO -- parseabilidade da A34/v2 é
    #        desempenho, não gate a partir de 2026-11-16) ---
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
        f"controles de síntese -- RESULTADO (não gate a partir de 2026-11-16): "
        f"parseável pela A34 em todos os 4 cenários={synthesis_overall_pass} "
        f"(parser v2 '{case_status_v2.PARSER_VERSION}', v1 só rastreabilidade)"
    )

    # Template de revisão de FIDELIDADE -- sempre gerado para o pesquisador
    # poder revisar depois (análise qualitativa), nunca mais usado para
    # bloquear a matriz principal.
    fidelity_review_template = vd.build_synthesis_fidelity_review_template(synthesis_results)
    template_path = output_dir / "synthesis_fidelity_review_template.json"
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(fidelity_review_template, f, ensure_ascii=False, indent=2, default=str)
    print(f"template de revisão de fidelidade (OPCIONAL, não bloqueante) gravado em: {template_path}")

    # --- 3. Matriz principal -- roda SEMPRE na mesma invocação, nenhum
    #        gate manual intermediário. ---
    print("--- Matriz principal: 92 chamadas planejadas ---")
    planned_calls = vd.build_planned_calls(rows, designated_image_ids)
    print(f"chamadas planejadas: {len(planned_calls)} (deve ser 92; total da execução do modelo incluindo controles = {vd.N_TOTAL_PLANNED_CALLS_PER_MODEL})")

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
        elif role == vd.ROLE_SUMMARY:
            key = (lesion_id, condition)
            if key not in accumulated_state:
                image = _get_image(image_id, condition)
                accumulated_state[key] = vd.build_graph_state(image, f"{image_id}_{condition}").model_dump()
            state = vd.GraphState(**accumulated_state[key])
            record, state_update = vd.run_summary_role_call_official_decision_contract(
                model_name, state, lesion_id, image_id, condition, timeout_s=args.timeout_s
            )
            for k, v in state_update.items():
                accumulated_state[key][k] = v
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
            f"status={record.call_status} decisao_final={record.decisao_final} "
            f"(v1_historico={record.classified_status} v2_historico={record.classified_status_v2})"
        )

    _write_report(
        output_dir, model_name, model_info, all_records,
        image_controls_eval=image_controls_eval, synthesis_results=synthesis_results,
        fidelity_review_data=fidelity_review_data,
        matrix_records=matrix_records, planned_calls=planned_calls, tracker=tracker,
        baseline_swap_bytes=baseline_swap_bytes, stopped_operationally=False, stop_reason=None,
    )
    print(f"\nRelatórios/logs gravados em: {output_dir}")


# ---------------------------------------------------------------------------
# Escrita de relatório único (controles + matriz, ou só controles se um
# problema OPERACIONAL interrompeu antes da matriz).
# ---------------------------------------------------------------------------
def _write_call_log(output_dir: Path, records: list["vd.CallRecord"], filename: str) -> Path:
    log_path = output_dir / filename
    with open(log_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_flat_dict(), ensure_ascii=False, default=str) + "\n")
    return log_path


def _serialize(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str, ensure_ascii=False))


def _write_report(
    output_dir: Path,
    model_name: str,
    model_info: dict,
    control_records: list,
    image_controls_eval: dict,
    synthesis_results: list,
    fidelity_review_data: Optional[dict[str, Any]],
    matrix_records: list,
    planned_calls: list,
    tracker: "vd.AbstentionTracker",
    baseline_swap_bytes: Optional[int],
    stopped_operationally: bool,
    stop_reason: Optional[str],
) -> None:
    control_log_path = _write_call_log(output_dir, control_records, "call_log.jsonl")
    matrix_log_path = None
    if matrix_records:
        matrix_log_path = _write_call_log(output_dir, matrix_records, "call_log_matriz_principal.jsonl")

    fidelity_review_eval = vd.evaluate_synthesis_fidelity_review(fidelity_review_data) if fidelity_review_data else None

    summary: dict[str, Any] = {
        "model_name": model_name,
        "model_info": model_info,
        "stopped_operationally": stopped_operationally,
        "stop_reason": stop_reason,
        "protocol_note": (
            "Protocolo alterado em 2026-11-16 (ver docs/reconstruction/"
            "30_real_lesion_evaluation_protocol.md): controles de envio de "
            "imagem e de síntese são sempre executados e registrados como "
            "RESULTADO (desempenho/fidelidade), nunca como gate -- só um "
            "problema OPERACIONAL real (payload incorreto, manifesto "
            "errado, imagem ausente/trocada, configuração/credencial "
            "incompatível, diretório de saída não gravável) interrompe a "
            "execução. Revisão manual de fidelidade da síntese e o limiar "
            "de >=95% de respostas válidas deixaram de ser pré-requisito "
            "de avanço."
        ),
        "official_classification_parser_version": DECISION_CONTRACT_PARSER_VERSION,
        "n_control_calls_executed": len(control_records),
        "n_total_planned_calls_per_model": vd.N_TOTAL_PLANNED_CALLS_PER_MODEL,
        "controls": {
            "image_sending": _serialize(image_controls_eval),
            "synthesis": _serialize(synthesis_results),
            "synthesis_fidelity_review": _serialize(fidelity_review_eval) if fidelity_review_eval else {
                "note": "nenhum arquivo de revisão de fidelidade anexado nesta execução -- não bloqueante.",
            },
        },
    }

    if not stopped_operationally:
        n_planned = len(planned_calls)
        n_executed = sum(1 for r in matrix_records if r.call_status != vd.STATUS_NOT_EXECUTED)
        n_not_executed = n_planned - n_executed

        by_role_condition: dict[str, dict[str, int]] = {}
        for r in matrix_records:
            key = f"{r.role}__{r.condition}"
            d = by_role_condition.setdefault(key, {"planejadas": 0, "executadas": 0, "nao_executadas": 0, "ok": 0})
            d["planejadas"] += 1
            if r.call_status == vd.STATUS_NOT_EXECUTED:
                d["nao_executadas"] += 1
            else:
                d["executadas"] += 1
                if r.call_status == vd.STATUS_OK:
                    d["ok"] += 1

        _ROLES_WITH_CLASSIFIABLE_OUTPUT = {vd.ROLE_VLM_DIRETO, vd.ROLE_SUMMARY}
        roles_with_zero_conclusive: list[str] = []
        for key in by_role_condition:
            role = key.split("__")[0]
            if role not in _ROLES_WITH_CLASSIFIABLE_OUTPUT:
                continue
            has_conclusive = any(
                r.decisao_final in (DECISAO_MELANOMA, DECISAO_NAO_MELANOMA, DECISAO_INDETERMINADO)
                for r in matrix_records
                if f"{r.role}__{r.condition}" == key
            )
            if not has_conclusive:
                roles_with_zero_conclusive.append(key)

        diagnostic_role_reporting = {
            role: vd.summarize_diagnostic_role_reporting(matrix_records, role)
            for role in (vd.ROLE_ABCD, vd.ROLE_MENZIES, vd.ROLE_SPCL)
        }

        n_valid_first_attempt = sum(1 for r in matrix_records if r.call_status == vd.STATUS_OK)
        valid_rate = n_valid_first_attempt / n_executed if n_executed else 0.0

        classifiable_records = [r for r in matrix_records if r.role in _ROLES_WITH_CLASSIFIABLE_OUTPUT and r.call_status != vd.STATUS_NOT_EXECUTED]
        n_decision_contract_valid = sum(1 for r in classifiable_records if r.decision_status == DECISION_CONTRACT_STATUS_OK)
        n_decision_contract_applicable = len(classifiable_records)
        decision_contract_valid_rate = (
            n_decision_contract_valid / n_decision_contract_applicable if n_decision_contract_applicable else 0.0
        )

        decision_contract_failures_detail = [
            {
                "role": r.role, "condition": r.condition, "lesion_id": r.lesion_id, "image_id": r.image_id,
                "call_status": r.call_status, "failure_cause": r.failure_cause,
                "decision_status": r.decision_status, "decisao_final": r.decisao_final,
            }
            for r in classifiable_records
            if not (r.call_status == vd.STATUS_OK and r.decision_status == DECISION_CONTRACT_STATUS_OK)
        ]

        binary_statuses_decision_contract = [vd.classify_binary_coverage_from_decisao_final(r.decisao_final) for r in matrix_records]
        n_positivo = sum(1 for b in binary_statuses_decision_contract if b is True)
        n_negativo = sum(1 for b in binary_statuses_decision_contract if b is False)

        binary_statuses_v1_historical = [vd.classify_binary_coverage(r.classified_status) for r in matrix_records]
        binary_statuses_v2_historical = [vd.classify_binary_coverage(r.classified_status_v2) for r in matrix_records]

        refusal_candidates = [
            {
                "role": r.role, "condition": r.condition, "lesion_id": r.lesion_id, "image_id": r.image_id,
                "raw_response_text": r.raw_response_text, "failure_cause": r.failure_cause,
            }
            for r in matrix_records if r.call_status == vd.STATUS_REFUSAL
        ]

        matrix_swap_samples: list[int] = []
        for r in matrix_records:
            if r.mem_before is not None:
                matrix_swap_samples.append(r.mem_before.system_swap_used_bytes)
            if r.mem_after is not None:
                matrix_swap_samples.append(r.mem_after.system_swap_used_bytes)
        if matrix_swap_samples and baseline_swap_bytes is not None:
            swap_growth = vd.detect_sustained_swap_growth(matrix_swap_samples, baseline_swap_bytes=baseline_swap_bytes)
        else:
            swap_growth = {"note": "psutil indisponível, baseline ausente, ou nenhuma amostra coletada -- não computado.", "baseline_swap_bytes": baseline_swap_bytes}

        per_lesion_condition_totals = vd.compute_per_lesion_condition_elapsed_totals(matrix_records)
        duration_projection = vd.project_full_scale_duration(list(per_lesion_condition_totals.values()), HYPOTHETICAL_FULL_SCALE_N)

        summary.update({
            "n_calls_planned_main_matrix": n_planned,
            "n_calls_executed_main_matrix": n_executed,
            "n_calls_not_executed_due_to_pause": n_not_executed,
            "valid_on_first_attempt_rate": valid_rate,
            "decision_contract_valid_rate_informational": decision_contract_valid_rate,
            "decision_contract_valid_count": n_decision_contract_valid,
            "decision_contract_applicable_count": n_decision_contract_applicable,
            "decision_contract_failures_detail": decision_contract_failures_detail,
            "decision_contract_valid_rate_note": (
                "Fração de chamadas vlm_direto/summary com decision_status=='ok' -- "
                "indeterminado conta como válido. Reportada como RESULTADO/cobertura "
                "do lote; não é mais usada para bloquear/reprovar a execução (ver "
                "protocol_note)."
            ),
            "binary_coverage": {
                "positivo": n_positivo, "negativo": n_negativo,
                "parser_version": DECISION_CONTRACT_PARSER_VERSION,
            },
            "binary_coverage_v1_historical_informational": {
                "positivo": sum(1 for b in binary_statuses_v1_historical if b is True),
                "negativo": sum(1 for b in binary_statuses_v1_historical if b is False),
            },
            "binary_coverage_v2_historical_informational": {
                "positivo": sum(1 for b in binary_statuses_v2_historical if b is True),
                "negativo": sum(1 for b in binary_statuses_v2_historical if b is False),
                "parser_version": case_status_v2.PARSER_VERSION,
            },
            "by_role_condition": by_role_condition,
            "roles_with_zero_conclusive_opinion": roles_with_zero_conclusive,
            "diagnostic_role_reporting_informational": diagnostic_role_reporting,
            "abstention_counters_trace": tracker.trace(),
            "sustained_swap_growth": _serialize(swap_growth),
            "duration_projection_full_scale": _serialize(duration_projection),
            "refusal_candidates_for_manual_review": refusal_candidates,
        })

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    report_lines = [
        f"# Piloto de VLM — {model_name}",
        "",
        f"- Parou por problema operacional: {stopped_operationally}" + (f" ({stop_reason})" if stop_reason else ""),
        f"- Log de chamadas de controle: `{control_log_path}`",
    ]
    if matrix_log_path:
        report_lines.append(f"- Log de chamadas da matriz principal: `{matrix_log_path}`")
    report_lines += [
        "",
        summary["protocol_note"],
        "",
        "## Controles (resultado, não gate)",
        "",
        f"```json\n{json.dumps(summary['controls'], ensure_ascii=False, indent=2, default=str)}\n```",
        "",
    ]
    if not stopped_operationally:
        report_lines += [
            "## Resumo da matriz principal",
            "",
            f"```json\n{json.dumps({k: v for k, v in summary.items() if k not in ('controls', 'model_info')}, ensure_ascii=False, indent=2, default=str)}\n```",
            "",
        ]
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


if __name__ == "__main__":
    main()
