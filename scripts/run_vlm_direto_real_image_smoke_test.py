"""Etapa 5 -- verificação real de INTEGRAÇÃO do contrato de decisão
explícito no papel VLM-direto (ver
`docs/reconstruction/26_decision_contract_official_pilot_criterion.md`).

Decisão do pesquisador: antes de liberar o piloto das 12 lesões com o
contrato congelado, confirmar com execução real (uma chamada por
modelo, SEM retry) que:
  1. a imagem chega corretamente ao papel VLM-direto (payload
     multimodal construído com a imagem certa);
  2. a saída bruta do modelo é preservada (texto não vazio, gravado
     verbatim);
  3. o campo `decisao_final` é lido corretamente pelo parser oficial
     (`src.evaluation.explicit_decision_contract.extract_decisao_final`,
     já wireado como instrução PADRÃO de `vd.run_vlm_direct_call` via
     `vd.VLM_DIRECT_TASK_INSTRUCTION`, ver doc 26).

Esta verificação NÃO exige que o modelo acerte o diagnóstico -- não há
"categoria esperada" aqui, nenhuma comparação com rótulo-gabarito.
`decision_status != "ok"` (campo ausente/repetido/inválido) é reportado
como FALHA desta verificação (o objetivo é confirmar que a leitura do
campo funciona na integração real); o CONTEÚDO da decisão
(melanoma/nao_melanoma/indeterminado) nunca é avaliado como certo/
errado.

NÃO repete os 8 controles de síntese reais (doc 24/25) -- o papel
sumarizador já foi validado lá, com a mesma rota/configuração agora
integrada ao executor do piloto (`vd.run_summary_role_call_official_decision_contract`).
Esta verificação cobre SÓ o papel VLM-direto, que ainda não tinha sido
testado com imagem real.

Imagem usada: a MESMA imagem real já selecionada/verificada por
`scripts/run_synthesis_control_real_image_selection.py` (registry em
`experiments/multiagent/synthesis_control_real_image_registry.json`,
já confirmada externa à amostra congelada de 3.000 lesões por
`lesion_id` E `image_id`, ver doc 24). Este script ADICIONALMENTE
verifica exclusão do manifesto das 12 lesões do piloto
(`experiments/multiagent/pilot12_manifest.csv`) -- se esse arquivo
ainda não existir (não gerado até agora), registra isso explicitamente
("nada para sobrepor ainda", não assumido como aprovado silenciosamente).

Modelos (mesmos 2 candidatos do piloto): `qwen3-vl:8b-instruct-q4_K_M`,
`medgemma1.5:4b-it-q4_K_M`. 1 chamada por modelo = 2 chamadas totais,
sem repetição/retry.

STATUS: implementado, aguardando execução do pesquisador. Nada deste
script foi executado por este agente (`feedback_no_autonomous_execution`)
-- nenhum `python`/`pytest`/`git`/`ollama` foi chamado.

Uso:

    python scripts/run_vlm_direto_real_image_smoke_test.py \\
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

import run_synthesis_control_real_image_selection as selection_mod  # noqa: E402

DEFAULT_REGISTRY_PATH = str(REPO_ROOT / "experiments" / "multiagent" / selection_mod.REGISTRY_FILENAME)
DEFAULT_FROZEN_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_PILOT12_MANIFEST = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot_vlm_direto_smoke_test")

MODEL_TAGS: tuple[str, ...] = ("qwen3-vl:8b-instruct-q4_K_M", "medgemma1.5:4b-it-q4_K_M")


def _check_pilot12_exclusion(pilot12_manifest_path: Path, chosen_lesion_id: str, chosen_image_id: str) -> dict[str, Any]:
    """Verifica, SEM INFERÊNCIA, se a imagem escolhida está fora do
    manifesto das 12 lesões do piloto. Se o manifesto ainda não existir
    (não gerado até agora), registra isso explicitamente -- NUNCA
    assume exclusão silenciosamente."""
    if not pilot12_manifest_path.exists():
        return {
            "pilot12_manifest_exists": False,
            "overlap_lesion_id": None,
            "overlap_image_id": None,
            "note": (
                f"{pilot12_manifest_path} ainda não existe (piloto das 12 lesões "
                "não foi gerado até o momento desta verificação) -- nada para "
                "sobrepor; checagem considerada satisfeita por ausência de "
                "conjunto a excluir, não assumida silenciosamente."
            ),
        }
    import csv

    with open(pilot12_manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lesion_ids = {r["lesion_id"] for r in rows}
    image_ids = {r["image_id"] for r in rows}
    overlap_lesion = chosen_lesion_id in lesion_ids
    overlap_image = chosen_image_id in image_ids
    return {
        "pilot12_manifest_exists": True,
        "overlap_lesion_id": overlap_lesion,
        "overlap_image_id": overlap_image,
        "note": "verificado contra o manifesto das 12 lesões já gerado.",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-path", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--frozen-manifest", default=DEFAULT_FROZEN_MANIFEST)
    parser.add_argument("--pilot12-manifest", default=DEFAULT_PILOT12_MANIFEST)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout-s", type=float, default=vd.TIMEOUT_S)
    parser.add_argument("--model-tags", nargs="+", default=None, help=f"default: {MODEL_TAGS}")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    registry_path = Path(args.registry_path)
    frozen_manifest_path = Path(args.frozen_manifest)
    pilot12_manifest_path = Path(args.pilot12_manifest)
    images_dir = Path(args.ham10000_images_dir)

    print("--- Pré-checagem SEM INFERÊNCIA (hash do manifesto congelado, exclusão por ambas as chaves, imagem carrega) ---")
    verify_result = selection_mod.verify_existing_registry(registry_path, frozen_manifest_path, images_dir=images_dir)
    print(json.dumps({k: v for k, v in verify_result.items() if k != "registry"}, ensure_ascii=False, indent=2, default=str))
    if not verify_result["ok"]:
        print("ABORTADO: pré-checagem contra a amostra congelada falhou -- NENHUMA chamada a modelo será feita.")
        return
    registry = verify_result["registry"]

    print("--- Checagem ADICIONAL SEM INFERÊNCIA: exclusão do manifesto das 12 lesões do piloto ---")
    pilot12_check = _check_pilot12_exclusion(
        pilot12_manifest_path, registry["chosen_lesion_id"], registry["chosen_image_id"]
    )
    print(json.dumps(pilot12_check, ensure_ascii=False, indent=2))
    if pilot12_check["overlap_lesion_id"] or pilot12_check["overlap_image_id"]:
        print("ABORTADO: a imagem escolhida está no manifesto das 12 lesões do piloto -- NENHUMA chamada a modelo será feita.")
        return

    image = load_original_image(Path(registry["image_path"]))
    if image is None:
        print("ABORTADO: a imagem carregou na pré-checagem mas falhou ao recarregar aqui.")
        return

    model_tags = tuple(args.model_tags) if args.model_tags else MODEL_TAGS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    message_for_payload_check = vd.build_vlm_direct_message(image)
    payload_correct = vd.verify_image_control_payload_sent_correct_image(message_for_payload_check, image)
    print(f"payload multimodal construído com a imagem certa: {payload_correct}")

    results: list[dict[str, Any]] = []
    for model_tag in model_tags:
        print(f"--- modelo={model_tag} (1 chamada, sem repetição) ---")
        record = vd.run_vlm_direct_call(
            model_tag, image, "smoke_test_vlm_direto", registry["chosen_image_id"], timeout_s=args.timeout_s
        )
        raw_text_preserved = isinstance(record.raw_response_text, str) and len(record.raw_response_text.strip()) > 0
        field_read_correctly = record.decision_status == vd.DECISION_CONTRACT_STATUS_OK
        entry = {
            "model_name": model_tag,
            "call_status": record.call_status,
            "failure_cause": record.failure_cause,
            "payload_correct": payload_correct,
            "raw_response_text_preserved_nonempty": raw_text_preserved,
            "raw_response_text": record.raw_response_text,
            "decision_status": record.decision_status,
            "decisao_final": record.decisao_final,
            "field_read_correctly": field_read_correctly,
            "note": (
                "field_read_correctly=True significa que decision_status == 'ok' "
                "(o campo decisao_final foi lido com sucesso) -- NÃO avalia se a "
                "decisão em si (melanoma/nao_melanoma/indeterminado) está certa; "
                "não há categoria esperada nesta verificação."
            ),
            "classified_status_v1_historical": record.classified_status,
            "classified_status_v2_historical": record.classified_status_v2,
        }
        results.append(entry)
        print(
            f"  call_status={record.call_status} raw_text_preserved={raw_text_preserved} "
            f"decision_status={record.decision_status} decisao_final={record.decisao_final}"
        )

    all_passed = all(
        r["call_status"] == vd.STATUS_OK and r["payload_correct"] and r["raw_response_text_preserved_nonempty"] and r["field_read_correctly"]
        for r in results
    )

    summary = {
        "registry_used": registry,
        "pilot12_exclusion_check": pilot12_check,
        "model_tags": list(model_tags),
        "n_calls_total": len(results),
        "all_integration_checks_passed": all_passed,
        "results": results,
        "note": (
            "Verificação de INTEGRAÇÃO (imagem chega, saída bruta preservada, "
            "campo lido corretamente), não de acerto diagnóstico. 1 chamada por "
            "modelo, sem retry. Se all_integration_checks_passed for True, a "
            "integração do contrato de decisão ao papel vlm_direto do piloto "
            "está confirmada com execução real -- decisão de liberar o piloto "
            "das 12 lesões continua sendo do pesquisador."
        ),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nall_integration_checks_passed: {all_passed}")
    print(f"resultados (json) gravados: {results_path}")


if __name__ == "__main__":
    main()
