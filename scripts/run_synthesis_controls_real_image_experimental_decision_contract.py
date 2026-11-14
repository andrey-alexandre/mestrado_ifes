"""Etapa 5 -- continuação do doc 23 (protótipo de contrato de decisão
final explícito): repete os 4 cenários fixos de controle de síntese
(`src.multiagent.vlm_direct.FIXED_SYNTHESIS_SCENARIOS`) COM A MESMA
IMAGEM REAL já selecionada/verificada por
`scripts/run_synthesis_control_real_image_selection.py` (mesmo registry
lido por `scripts/run_synthesis_controls_real_image.py` -- NENHUMA nova
seleção de imagem é feita aqui), para os 2 modelos do piloto local
(`qwen3-vl:8b-instruct-q4_K_M`, `medgemma1.5:4b-it-q4_K_M`) -- 4 x 2 =
**8 chamadas**, uma por (cenário, modelo), SEM repetição/retry.

Diferença em relação a `scripts/run_synthesis_controls_real_image.py`
(NÃO MODIFICADO por esta tarefa, continua produzindo exatamente o mesmo
resultado de antes se reexecutado): aqui o sumarizador recebe a
instrução de tarefa EXPERIMENTAL
(`src.multiagent.experimental_decision_contract
.build_experimental_summary_task_instruction`, contrato de decisão
`decisao_final: melanoma|nao_melanoma|indeterminado`) em vez do prompt
oficial, e a resposta é classificada SÓ pelo parser do contrato
(`src.evaluation.explicit_decision_contract.extract_decisao_final`) --
`classify_case_status`/`classify_case_status_v2` NÃO são chamados por
este script.

DECISÃO DO PESQUISADOR (verbatim, ver
`docs/reconstruction/23_explicit_decision_contract_prototype.md`):
"repetir os quatro controles de síntese com imagem real, uma chamada
por cenário para Qwen3-VL e MedGemma, sem retry". Esta rodada é uma
decisão SEPARADA e NÃO aprova automaticamente a troca do critério de
avanço do Estágio 1 -- `scripts/run_local_vlm_pilot.py` (gate oficial
via `classify_case_status_v2`) NÃO é modificado nem lido por este
script.

Escreve em
`experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/`
-- NUNCA em `experiments/multiagent/synthesis_controls_real_image/`
(preserva os artefatos históricos já lá, incluindo os produzidos por
`run_synthesis_controls_real_image.py`).

STATUS: implementado, aguardando execução do pesquisador. Nada deste
script foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) -- nenhum `python`/`pytest`/`git`/
`ollama` foi chamado.

Uso (depois de confirmar que o registry da imagem real já existe --
mesmo registry usado por `run_synthesis_controls_real_image.py`; este
script roda a MESMA pré-checagem sem inferência antes de qualquer
chamada):

    python scripts/run_synthesis_controls_real_image_experimental_decision_contract.py \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.multiagent import vlm_direct as vd  # noqa: E402
from src.multiagent.presentation import load_original_image  # noqa: E402
from src.multiagent.experimental_synthesis_control_real_image import (  # noqa: E402
    ExperimentalDecisionCallRecord,
    run_experimental_synthesis_control_real_image,
)

import run_synthesis_control_real_image_selection as selection_mod  # noqa: E402

DEFAULT_REGISTRY_PATH = str(REPO_ROOT / "experiments" / "multiagent" / selection_mod.REGISTRY_FILENAME)
DEFAULT_FROZEN_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_OUTPUT_DIR_ROOT = str(
    REPO_ROOT / "experiments" / "multiagent" / "synthesis_controls_real_image_experimental_decision_contract"
)

MODEL_TAGS: tuple[str, ...] = ("qwen3-vl:8b-instruct-q4_K_M", "medgemma1.5:4b-it-q4_K_M")
SCENARIO_ORDER: tuple[str, ...] = tuple(vd.FIXED_SYNTHESIS_SCENARIOS.keys())


def _write_call_log(path: Path, records: list[ExperimentalDecisionCallRecord]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_flat_dict(), ensure_ascii=False, default=str) + "\n")


def _build_report_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Controles de síntese com imagem REAL -- contrato de decisão experimental (8 chamadas)",
        "",
        "Execução SEPARADA e NARROW -- repete os 4 controles de síntese com a MESMA "
        "imagem real de `run_synthesis_controls_real_image.py`, mas com a instrução de "
        "tarefa experimental do sumarizador e classificação só pelo contrato de decisão "
        "explícito (`decisao_final`). `src/evaluation/case_status_v2.py`, o gate do "
        "Estágio 1 (`scripts/run_local_vlm_pilot.py`) e os prompts oficiais de "
        "`src/agents/agents.py` NÃO foram tocados nem lidos para decidir nada aqui.",
        "",
        f"- Modelos: {results['model_tags']}",
        f"- Cenários (ordem): {results['scenario_order']}",
        f"- Total de chamadas: {results['n_calls_total']} (esperado 8, sem repetição/retry)",
        "",
        "## Imagem real usada (registry, ver script de seleção -- mesma imagem da "
        "execução histórica com o sumarizador oficial)",
        "",
        f"```json\n{json.dumps(results['registry_used'], ensure_ascii=False, indent=2, default=str)}\n```",
        "",
        results["note"],
        "",
        "## Chamadas (uma seção por cenário x modelo)",
        "",
    ]
    for entry in results["calls"]:
        lines += [
            f"### modelo=`{entry['model_name']}` cenário=`{entry['scenario']}`",
            "",
            f"- `call_status`: `{entry['call_status']}` (causa: `{entry['failure_cause']}`)",
            f"- `template_extraction_status`: `{entry['template_extraction_status']}`",
            f"- `decision_status`: `{entry['decision_status']}`",
            f"- `decisao_final`: `{entry['decisao_final']}`",
            f"- `matched_raw_values`: {entry['matched_raw_values']}",
            "",
            "**Texto bruto (verbatim):**",
            "",
            f"```\n{entry['raw_response_text']}\n```",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--frozen-manifest", default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--output-dir-root", default=DEFAULT_OUTPUT_DIR_ROOT)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    parser.add_argument(
        "--model-tags", nargs="+", default=None,
        help=f"default (não sobrescrever sem motivo): {MODEL_TAGS}",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    registry_path = Path(args.registry_path)
    frozen_manifest_path = Path(args.frozen_manifest)
    images_dir = Path(args.ham10000_images_dir)

    print("--- Pré-checagem SEM INFERÊNCIA (hash do manifesto, exclusão por ambas as chaves, imagem carrega) ---")
    verify_result = selection_mod.verify_existing_registry(registry_path, frozen_manifest_path, images_dir=images_dir)
    print(json.dumps({k: v for k, v in verify_result.items() if k != "registry"}, ensure_ascii=False, indent=2, default=str))
    if not verify_result["ok"]:
        print(
            "ABORTADO: pré-checagem falhou -- NENHUMA chamada a modelo será feita. "
            "Corrija a seleção/registry (scripts/run_synthesis_control_real_image_selection.py) "
            "antes de rodar este script."
        )
        return
    registry = verify_result["registry"]

    image = load_original_image(Path(registry["image_path"]))
    if image is None:
        print(
            "ABORTADO: a imagem carregou na pré-checagem mas falhou ao recarregar "
            "aqui -- inconsistência inesperada, não prosseguindo silenciosamente."
        )
        return

    model_tags = tuple(args.model_tags) if args.model_tags else MODEL_TAGS
    output_root = Path(args.output_dir_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_entries: list[dict[str, Any]] = []
    for model_tag in model_tags:
        model_dir = output_root / model_tag.replace(":", "_").replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        call_records: list[ExperimentalDecisionCallRecord] = []
        for scenario_name in SCENARIO_ORDER:
            print(f"--- modelo={model_tag} cenário={scenario_name} (1 chamada, sem repetição) ---")
            record = run_experimental_synthesis_control_real_image(
                model_tag, scenario_name, image, timeout_s=args.timeout_s
            )
            call_records.append(record)
            all_entries.append(record.to_flat_dict())
            print(
                f"  status={record.call_status} decision_status={record.decision_status} "
                f"decisao_final={record.decisao_final}"
            )
        _write_call_log(model_dir / "call_log.jsonl", call_records)
        print(f"  log de chamadas gravado: {model_dir / 'call_log.jsonl'}")

    results = {
        "registry_used": registry,
        "model_tags": list(model_tags),
        "scenario_order": list(SCENARIO_ORDER),
        "n_calls_total": len(all_entries),
        "calls": all_entries,
        "note": (
            "Cada chamada é ÚNICA (1 cenário x 1 modelo, sem repetição/retry). "
            "decisao_final/decision_status vêm SÓ do contrato de decisão explícito "
            "experimental (src.evaluation.explicit_decision_contract) -- "
            "classify_case_status/classify_case_status_v2 NÃO foram chamados neste "
            "script. Esta rodada é uma decisão SEPARADA (ver doc 23) e NÃO aprova "
            "automaticamente a troca do critério de avanço do Estágio 1."
        ),
    }
    results_path = output_root / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    report_path = output_root / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_build_report_markdown(results))

    print(f"\nresultados (json) gravados: {results_path}")
    print(f"relatório (md) gravado: {report_path}")
    print(f"logs de chamada por modelo em: {output_root}/<model_tag>/call_log.jsonl")


if __name__ == "__main__":
    main()
