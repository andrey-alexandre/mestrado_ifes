"""Etapa 5 -- tarefa adicional (2026-11-14): executa os 4 cenários fixos
de controle de síntese (`src.multiagent.vlm_direct.FIXED_SYNTHESIS_SCENARIOS`)
COM A IMAGEM REAL selecionada por
`scripts/run_synthesis_control_real_image_selection.py`, para os 2
modelos do piloto local (`qwen3-vl:8b-instruct-q4_K_M`,
`medgemma1.5:4b-it-q4_K_M`) -- 4 x 2 = **8 chamadas**, uma por
(cenário, modelo), SEM repetição/retry.

Isto é uma execução SEPARADA e NARROW -- não é o Estágio 1 completo de
`scripts/run_local_vlm_pilot.py` (NÃO MODIFICADO por esta tarefa), não
inclui os 12 controles de envio de imagem por papel visual, e não toca a
matriz principal de 92 chamadas nem a matriz de 12 lesões
(`experiments/multiagent/pilot12_manifest.csv`, fora de escopo). Os
pareceres fixos de cada cenário são usados EXATAMENTE como em
`FIXED_SYNTHESIS_SCENARIOS` -- nunca alterados por este script; só a
imagem muda (ver `vd.build_synthesis_control_state_real_image`/
`vd.run_synthesis_control_real_image`, ADITIVOS, que não tocam
`vd.build_synthesis_control_state`/`vd.run_synthesis_control` originais).

Para cada uma das 8 chamadas, grava SEPARADAMENTE (nunca misturado):
  - o texto bruto da resposta (verbatim);
  - a "categoria esperada" por cenário (conceito NOVO desta tarefa --
    ver `EXPECTED_CATEGORY_BY_SCENARIO`/`EXPECTED_CATEGORY_RATIONALE_BY_SCENARIO`
    abaixo; para `conflitante`, é deliberadamente `None` -- ver
    justificativa no dict, não há "gabarito" único por desenho desse
    cenário);
  - a classificação do parser v2 (`classify_case_status_v2`, REUSADO SEM
    MODIFICAÇÃO -- `src/evaluation/case_status_v2.py` NÃO é tocado por
    esta tarefa, nem mesmo para adicionar um caso; se o v2 não reconhecer
    uma resposta, isto é registrado tal como está, sem ampliar vocabulário
    nem recorrer ao v1 como desempate);
  - um TEMPLATE DE REVISÃO DE FIDELIDADE manual (placeholders `None`/`""`,
    nunca preenchidos automaticamente) cobrindo as 3 checagens pedidas:
    autoria dos achados/divergências preservada, distinção "não avaliado"
    vs "ausente" correta, e ausência de um 3º parecer fictício -- mais um
    campo para a observação própria do sumarizador (se houver, se está
    marcada como dele) e um campo `faithful_but_unrecognized_by_v2` para o
    caso de uma conclusão fiel que o v2 não reconheça.

NÃO modifica: `src/evaluation/case_status_v2.py`, o gate/lógica do
Estágio 1 de `scripts/run_local_vlm_pilot.py`, nem nenhum prompt real em
`src/agents/agents.py`. Escreve em
`experiments/multiagent/synthesis_controls_real_image/` -- NUNCA em
`experiments/multiagent/local_vlm_pilot/<model_tag>/` (preserva os 3
artefatos históricos reais já lá).

STATUS: implementado, aguardando execução do pesquisador. Nada deste
script foi executado pelo assistente que o escreveu
(`feedback_no_autonomous_execution`) -- nenhum `python`/`pytest`/`git`/
`ollama` foi chamado.

Uso (depois de rodar a seleção E a verificação sem inferência -- ver
`scripts/run_synthesis_control_real_image_selection.py`):

    # 0. Pré-checagem sem inferência (recomendado antes de autorizar as 8
    #    chamadas reais abaixo -- repete a mesma checagem internamente,
    #    mas é bom o pesquisador conferir o relatório à parte também):
    python scripts/run_synthesis_control_real_image_selection.py --verify-only \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000

    # 1. As 8 chamadas reais (Ollama já rodando, os 2 modelos já baixados):
    python scripts/run_synthesis_controls_real_image.py \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
# Mesmo diretório deste script -- para importar
# run_synthesis_control_real_image_selection.py como módulo plano, sem
# depender de `scripts` ser um pacote Python (não tem __init__.py, e
# nenhum outro script deste projeto importa de outro script).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.multiagent import vlm_direct as vd  # noqa: E402
from src.multiagent.presentation import load_original_image  # noqa: E402

import run_synthesis_control_real_image_selection as selection_mod  # noqa: E402

DEFAULT_REGISTRY_PATH = str(REPO_ROOT / "experiments" / "multiagent" / selection_mod.REGISTRY_FILENAME)
DEFAULT_FROZEN_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_OUTPUT_DIR_ROOT = str(REPO_ROOT / "experiments" / "multiagent" / "synthesis_controls_real_image")

# Os 2 modelos do piloto local (ver docs/reconstruction/20_local_vlm_pilot.md
# §7) -- fixos por instrução explícita do pesquisador para esta tarefa
# ("qwen3-vl:8b-instruct-q4_K_M e medgemma1.5:4b-it-q4_K_M"); --model-tags
# permite sobrescrever só para conveniência de depuração, nunca o default.
MODEL_TAGS: tuple[str, ...] = ("qwen3-vl:8b-instruct-q4_K_M", "medgemma1.5:4b-it-q4_K_M")
SCENARIO_ORDER: tuple[str, ...] = tuple(vd.FIXED_SYNTHESIS_SCENARIOS.keys())

# ---------------------------------------------------------------------------
# "Categoria esperada" por cenário -- conceito NOVO desta tarefa (não
# existia em run_synthesis_control original). Ver justificativa de cada
# valor no dict de rationale abaixo -- em especial `conflitante`, que é
# deliberadamente None (não há gabarito único por desenho do cenário).
# ---------------------------------------------------------------------------
EXPECTED_CATEGORY_BY_SCENARIO: dict[str, Optional[str]] = {
    "unanime_positivo": "positivo",
    "unanime_negativo": "negativo",
    "conflitante": None,
    "um_parecer_ausente": "negativo",
}

EXPECTED_CATEGORY_RATIONALE_BY_SCENARIO: dict[str, str] = {
    "unanime_positivo": (
        "Os 3 pareceres fixos (ABCD, Menzies, SPCL) concordam unanimemente "
        "em suspeita de melanoma (ver FIXED_SYNTHESIS_SCENARIOS['unanime_positivo'] "
        "em src/multiagent/vlm_direct.py) -- um sumarizador fiel que apenas "
        "integra os 3 pareceres sem inventar nada deveria concluir "
        "'positivo'."
    ),
    "unanime_negativo": (
        "Os 3 pareceres fixos concordam unanimemente em lesão benigna -- "
        "um sumarizador fiel deveria concluir 'negativo'."
    ),
    "conflitante": (
        "NÃO HÁ categoria 'correta' única por desenho deste cenário: ABCD "
        "indica benigno, Menzies indica melanoma, e SPCL fica abaixo do "
        "limiar de suspeita (pontuação 1 de 3), com o critério "
        "vasos_atipicos explicitamente marcado como 'não avaliado' (nunca "
        "'avaliado e ausente'). O propósito deste cenário é testar se o "
        "sumarizador PRESERVA a divergência real entre os 3 algoritmos e a "
        "distinção 'não avaliado' vs 'avaliado e ausente' -- não se ele "
        "chega a uma conclusão específica. Por isso 'expected_category' é "
        "deliberadamente None aqui: a revisão de fidelidade deste cenário "
        "deve avaliar COMO o sumarizador resolveu/relatou o conflito e se "
        "preservou a autoria/divergência dos 3 pareceres, nunca comparar "
        "contra um rótulo esperado único. Registro histórico (não tratado "
        "como gabarito): em execuções reais anteriores deste piloto com "
        "imagem placeholder, o modelo concluiu algo como 'Suspeita de "
        "Melanoma segundo critério Menzies' -- uma resolução fiel possível, "
        "mas não a única logicamente válida para este cenário."
    ),
    "um_parecer_ausente": (
        "O parecer SPCL está ausente (diagnosis_spcl=None, "
        "status_spcl=AgentStatus(status='sem_resposta', "
        "causa='controle_sintetico_parecer_ausente')) -- reusa o mecanismo "
        "real de ausência A05 (src/agents/schemas.py::AgentStatus), não "
        "um caso especial deste piloto. Os 2 pareceres DISPONÍVEIS (ABCD e "
        "Menzies) concordam em lesão benigna; um sumarizador fiel que "
        "integra só o que de fato recebeu (sem inventar um 3º parecer "
        "SPCL) deveria concluir 'negativo' a partir dos 2 disponíveis."
    ),
}


def _build_fidelity_review_template_fields() -> dict[str, Any]:
    """Placeholders para preenchimento manual do pesquisador -- MESMO
    estilo/convenção de `vd.build_synthesis_manual_review_template`
    (`None`/`""` até revisão humana, nunca preenchido automaticamente).
    As 3 checagens pedidas pelo pesquisador + o campo de observação
    própria do sumarizador + o campo de "fiel mas não reconhecido pelo
    v2"."""
    return {
        # (a) autoria dos achados/divergências preservada -- cada achado
        # atribuído ao algoritmo correto (ABCD/Menzies/SPCL), não trocado
        # nem atribuído a uma invenção do sumarizador sem marcação.
        "fidelity_authorship_preserved": None,
        "fidelity_authorship_notes": "",
        # (b) distingue "não avaliado" de "ausente" corretamente -- usa as
        # 2 heurísticas lexicais de triagem já existentes em vlm_direct.py
        # (not_evaluated_vs_evaluated_absent_check/
        # absence_vs_lesion_lacks_feature_check, ver campos abaixo no
        # mesmo registro) como PONTEIRO para o revisor, nunca como
        # veredito -- confirmar/refutar manualmente lendo o texto bruto.
        "fidelity_not_evaluated_vs_absent_correct": None,
        "fidelity_not_evaluated_vs_absent_notes": "",
        # (c) não inventa um 3º/fictício parecer.
        "fidelity_no_fictional_third_parecer": None,
        "fidelity_no_fictional_third_parecer_notes": "",
        # Observação própria do sumarizador baseada na imagem -- permitida
        # pelo pesquisador, DESDE QUE claramente marcada como dele (não
        # atribuída a ABCD/Menzies/SPCL).
        "summarizer_added_own_observation": None,
        "summarizer_own_observation_clearly_marked_as_summarizer_own": None,
        "summarizer_own_observation_notes": "",
        # Caso explícito pedido pelo pesquisador: uma conclusão FIEL que o
        # v2 não reconheça -- registrado aqui, nunca resolvido
        # automaticamente, nunca aciona mudança de vocabulário/fallback v1.
        "faithful_but_unrecognized_by_v2": None,
        "reviewer_notes": "",
    }


def run_one_call(model_tag: str, scenario_name: str, image, timeout_s: float) -> tuple[dict[str, Any], "vd.CallRecord"]:
    result = vd.run_synthesis_control_real_image(model_tag, scenario_name, image, timeout_s=timeout_s)
    record = result["call_record"]
    entry: dict[str, Any] = {
        "model_name": model_tag,
        "scenario": scenario_name,
        "attempt_timestamp_utc": record.attempt_timestamp_utc,
        "elapsed_s": record.elapsed_s,
        "call_status": record.call_status,
        "failure_cause": record.failure_cause,
        "raw_response_text": record.raw_response_text,
        "expected_category": EXPECTED_CATEGORY_BY_SCENARIO[scenario_name],
        "expected_category_rationale": EXPECTED_CATEGORY_RATIONALE_BY_SCENARIO[scenario_name],
        "classified_status_v2": result["classified_status_v2"],
        "official_classification_parser_version": result["official_classification_parser_version"],
        "classified_status_v1_historical": result["classified_status_v1_historical"],
        "parseable": result["parseable"],
        "passed": result["passed"],
        "novel_clinical_terms_candidates": result["novel_clinical_terms_candidates"],
        "no_novel_terms_detected": result["no_novel_terms_detected"],
        "not_evaluated_vs_evaluated_absent_check": result.get("not_evaluated_vs_evaluated_absent_check"),
        "absence_vs_lesion_lacks_feature_check": result.get("absence_vs_lesion_lacks_feature_check"),
        "fidelity_review_template": _build_fidelity_review_template_fields(),
    }
    return entry, record


def _write_call_log(path: Path, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_flat_dict(), ensure_ascii=False, default=str) + "\n")


def _build_report_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Controles de síntese com imagem REAL -- 8 chamadas (4 cenários x 2 modelos)",
        "",
        "Execução SEPARADA e NARROW -- só os 4 controles de síntese corrigidos "
        "(imagem real em vez de placeholder cinza), não o Estágio 1 completo nem "
        "a matriz de 12/92/3.000. `src/evaluation/case_status_v2.py`, o gate do "
        "Estágio 1 (`scripts/run_local_vlm_pilot.py`) e os prompts reais de "
        "`src/agents/agents.py` NÃO foram tocados.",
        "",
        f"- Modelos: {results['model_tags']}",
        f"- Cenários (ordem): {results['scenario_order']}",
        f"- Total de chamadas: {results['n_calls_total']} (esperado 8, sem repetição/retry)",
        "",
        "## Imagem real usada (registry, ver script de seleção)",
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
            f"- `expected_category`: `{entry['expected_category']}`",
            f"- Justificativa da categoria esperada: {entry['expected_category_rationale']}",
            f"- `classified_status_v2` (gate oficial, parser `{entry['official_classification_parser_version']}`): `{entry['classified_status_v2']}`",
            f"- `classified_status_v1_historical` (rastreabilidade só, nunca fallback): `{entry['classified_status_v1_historical']}`",
            f"- `parseable`/`passed` (só parsing A34 via v2): {entry['parseable']} / {entry['passed']}",
            f"- `novel_clinical_terms_candidates` (triagem lexical, nunca gate): {entry['novel_clinical_terms_candidates']}",
            "",
            "**Texto bruto (verbatim):**",
            "",
            f"```\n{entry['raw_response_text']}\n```",
            "",
        ]
        if entry["not_evaluated_vs_evaluated_absent_check"] is not None:
            lines += [
                f"- Checagem de triagem 'não avaliado vs avaliado-e-ausente' (SÓ ponteiro, não veredito): "
                f"{entry['not_evaluated_vs_evaluated_absent_check']}",
                "",
            ]
        if entry["absence_vs_lesion_lacks_feature_check"] is not None:
            lines += [
                f"- Checagem de triagem 'ausência de parecer vs lesão sem característica' (SÓ ponteiro, não veredito): "
                f"{entry['absence_vs_lesion_lacks_feature_check']}",
                "",
            ]
        lines += [
            "**Template de revisão de fidelidade -- PREENCHER MANUALMENTE (nunca automatizado):**",
            "",
            f"```json\n{json.dumps(entry['fidelity_review_template'], ensure_ascii=False, indent=2)}\n```",
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
        call_records = []
        for scenario_name in SCENARIO_ORDER:
            print(f"--- modelo={model_tag} cenário={scenario_name} (1 chamada, sem repetição) ---")
            entry, record = run_one_call(model_tag, scenario_name, image, args.timeout_s)
            all_entries.append(entry)
            call_records.append(record)
            print(
                f"  status={record.call_status} classificado_v2={entry['classified_status_v2']} "
                f"esperado={entry['expected_category']} passed={entry['passed']}"
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
            "'expected_category' e 'fidelity_review_template' são conceitos NOVOS "
            "desta tarefa -- não existiam em run_synthesis_control original. "
            "classified_status_v2 é só TRIAGEM automatizada (gate v2, "
            "src/evaluation/case_status_v2.py, NÃO modificado nesta tarefa); a "
            "fidelidade real exige leitura humana de 'raw_response_text' e "
            "preenchimento manual de 'fidelity_review_template'."
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
