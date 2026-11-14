"""Reprocessamento OFFLINE dos 8 controles de síntese reais já salvos --
ZERO chamadas novas de modelo, ZERO rede, ZERO inferência.

Lê `experiments/multiagent/synthesis_controls_real_image/results.json`
(as 8 chamadas já executadas/já revisadas, ver
`docs/reconstruction/21_synthesis_control_fidelity_review_closure.md`) e,
para cada uma:

  1. Aplica `src.evaluation.response_template_extraction.extract_final_answer`
     ao `raw_response_text` salvo -- extrai a resposta final, removendo
     SÓ blocos de template com delimitadores reconhecidos
     (`<unused94>thought` / `<unused95>`, confirmados nos 4 casos reais
     do medgemma1.5 neste piloto).
  2. Classifica DUAS VEZES com
     `src.evaluation.case_status_v2.classify_case_status_v2_detailed`
     (NÃO MODIFICADO por esta tarefa):
       - "ANTES": sobre o `raw_response_text` ORIGINAL (igual ao que já
         está salvo em `classified_status_v2` em `results.json` -- este
         script VERIFICA essa consistência explicitamente, não assume).
       - "DEPOIS": sobre o `final_answer_text` extraído -- SÓ quando
         `extraction.status` não é um dos status de problema
         (`incomplete_block_no_closing_delimiter`,
         `empty_after_extraction`). Quando é um desses dois, o script
         NUNCA tenta classificar -- registra o problema explicitamente no
         relatório, por instrução explícita do pesquisador.
  3. Grava DOIS arquivos NOVOS (nunca sobrescreve
     `results.json`/`report.md`, que são o registro histórico já
     revisado por fidelidade):
       - `experiments/multiagent/synthesis_controls_real_image/
         final_answer_extraction_reprocessing_results.json`
       - `experiments/multiagent/synthesis_controls_real_image/
         final_answer_extraction_reprocessing_report.md`

NÃO modifica `src/evaluation/case_status_v2.py`, `src/multiagent/
vlm_direct.py` (nenhuma das funções `run_vlm_direct_call`/
`run_multiagent_role_call`/`run_synthesis_control_real_image`), nem
`src/agents/agents.py`. Nenhuma dessas é importada aqui exceto
`classify_case_status_v2_detailed`, usada em modo SOMENTE LEITURA sobre
texto já salvo em disco.

Caso especial `medgemma1.5/conflitante`: `expected_category` é `None`
neste caso POR DESENHO (não há gabarito único -- ver
`results.json`/`FIXED_SYNTHESIS_SCENARIOS['conflitante']`). Este script
NUNCA compara a classificação desse caso contra um rótulo esperado --
só reporta, lado a lado, a região de conclusão extraída e o rótulo
casado em cada passada (ANTES/DEPOIS), para leitura humana.

Uso (pelo PESQUISADOR -- este script não é executado por nenhum agente
desta tarefa):

    python scripts/run_reprocess_synthesis_controls_final_answer_extraction.py

Saída: os dois arquivos acima, mais um resumo no stdout.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.case_status_v2 import (  # noqa: E402
    CaseStatusV2Result,
    classify_case_status_v2_detailed,
)
from src.evaluation.response_template_extraction import (  # noqa: E402
    PROBLEM_STATUSES,
    ExtractionResult,
    extract_final_answer,
)

RESULTS_JSON_PATH = REPO_ROOT / "experiments" / "multiagent" / "synthesis_controls_real_image" / "results.json"
OUTPUT_JSON_PATH = REPO_ROOT / "experiments" / "multiagent" / "synthesis_controls_real_image" / "final_answer_extraction_reprocessing_results.json"
OUTPUT_MD_PATH = REPO_ROOT / "experiments" / "multiagent" / "synthesis_controls_real_image" / "final_answer_extraction_reprocessing_report.md"


def _v2_result_to_dict(result: Optional[CaseStatusV2Result]) -> dict[str, Any]:
    if result is None:
        return {
            "status": None,
            "parser_version": None,
            "matched_variant": None,
            "matched_category": None,
            "extraction_region_text": None,
            "extraction_fallback_used": None,
        }
    return {
        "status": result.status,
        "parser_version": result.parser_version,
        "matched_variant": result.matched_variant,
        "matched_category": result.matched_category,
        "extraction_region_text": result.extraction.region_text if result.extraction else None,
        "extraction_fallback_used": result.extraction.extraction_fallback_used if result.extraction else None,
    }


def _extraction_result_to_dict(result: ExtractionResult) -> dict[str, Any]:
    d = asdict(result)
    return d


def process_one_call(call: dict[str, Any]) -> dict[str, Any]:
    model_name = call["model_name"]
    scenario = call["scenario"]
    raw_text = call.get("raw_response_text")
    stored_v2_status = call.get("classified_status_v2")

    extraction = extract_final_answer(raw_text)

    before = classify_case_status_v2_detailed({"validation": raw_text})
    before_matches_stored = before.status == stored_v2_status

    after: Optional[CaseStatusV2Result] = None
    after_skipped_reason: Optional[str] = None
    if extraction.status in PROBLEM_STATUSES:
        after_skipped_reason = (
            f"extracao_com_problema:{extraction.status} -- nao classificado, "
            "por instrucao explicita do pesquisador (nunca classificar texto "
            "bruto/parcial quando a extracao falha)."
        )
    else:
        after = classify_case_status_v2_detailed({"validation": extraction.final_answer_text})

    agree: Optional[bool] = None
    if after is not None:
        agree = before.status == after.status

    return {
        "model_name": model_name,
        "scenario": scenario,
        "expected_category": call.get("expected_category"),
        "expected_category_rationale": call.get("expected_category_rationale"),
        "extraction": _extraction_result_to_dict(extraction),
        "before_stored_classified_status_v2": stored_v2_status,
        "before_recomputed": _v2_result_to_dict(before),
        "before_recomputed_matches_stored": before_matches_stored,
        "after": _v2_result_to_dict(after) if after is not None else None,
        "after_skipped_reason": after_skipped_reason,
        "before_after_agree": agree,
    }


def build_markdown_report(cases: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Reprocessamento offline -- extração de resposta final antes do v2 (Etapa 5)")
    lines.append("")
    lines.append(
        "Gerado por `scripts/run_reprocess_synthesis_controls_final_answer_extraction.py`. "
        "ZERO chamadas novas de modelo -- reprocessa os 8 `raw_response_text` já salvos em "
        "`experiments/multiagent/synthesis_controls_real_image/results.json`. "
        "`src/evaluation/case_status_v2.py` e os prompts reais não foram modificados."
    )
    lines.append("")
    lines.append(
        "| Modelo | Cenário | Extração: status | ANTES (v2 sobre bruto) | "
        "ANTES bate c/ salvo? | DEPOIS (v2 sobre resposta final) | ANTES==DEPOIS? |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for c in cases:
        after_status = c["after"]["status"] if c["after"] is not None else f"(não classificado: {c['extraction']['status']})"
        lines.append(
            f"| {c['model_name']} | {c['scenario']} | {c['extraction']['status']} | "
            f"{c['before_recomputed']['status']} | "
            f"{'sim' if c['before_recomputed_matches_stored'] else 'NAO -- DIVERGENCIA'} | "
            f"{after_status} | "
            f"{'sim' if c['before_after_agree'] else ('n/a' if c['before_after_agree'] is None else 'NAO')} |"
        )
    lines.append("")
    lines.append("## Detalhe por caso")
    lines.append("")
    for c in cases:
        lines.append(f"### {c['model_name']} / {c['scenario']}")
        lines.append("")
        lines.append(f"- `expected_category`: `{c['expected_category']}`")
        if c["expected_category"] is None:
            lines.append(
                "  - Sem gabarito único por desenho deste cenário -- NÃO usado para julgar "
                "a classificação abaixo (ver rationale em results.json)."
            )
        lines.append(f"- Extração: `status={c['extraction']['status']}`, "
                      f"`template_removed={c['extraction']['template_removed']}`, "
                      f"`delimiter_pair_used={c['extraction']['delimiter_pair_used']}`")
        if c["extraction"]["status"] in ("incomplete_block_no_closing_delimiter", "empty_after_extraction"):
            lines.append(f"  - **PROBLEMA REGISTRADO**: {c['after_skipped_reason']}")
        lines.append("")
        lines.append(
            f"- ANTES (v2 sobre texto bruto completo): `status={c['before_recomputed']['status']}`, "
            f"`matched_variant={c['before_recomputed']['matched_variant']}`, "
            f"`extraction_fallback_used={c['before_recomputed']['extraction_fallback_used']}`"
        )
        lines.append(
            f"  - Confere com `classified_status_v2` já salvo em results.json "
            f"(`{c['before_stored_classified_status_v2']}`)? "
            f"{'SIM' if c['before_recomputed_matches_stored'] else 'NAO -- INVESTIGAR'}"
        )
        if c["after"] is not None:
            lines.append(
                f"- DEPOIS (v2 sobre resposta final extraída): `status={c['after']['status']}`, "
                f"`matched_variant={c['after']['matched_variant']}`, "
                f"`extraction_fallback_used={c['after']['extraction_fallback_used']}`"
            )
            lines.append(
                f"  - ANTES == DEPOIS? {'SIM' if c['before_after_agree'] else 'NAO'}"
            )
        else:
            lines.append("- DEPOIS: não classificado (ver problema registrado acima).")
        lines.append("")
        lines.append("<details><summary>Região de conclusão final extraída pelo v2 -- ANTES</summary>")
        lines.append("")
        lines.append("```")
        lines.append(str(c["before_recomputed"]["extraction_region_text"]))
        lines.append("```")
        lines.append("</details>")
        lines.append("")
        if c["after"] is not None:
            lines.append("<details><summary>Região de conclusão final extraída pelo v2 -- DEPOIS</summary>")
            lines.append("")
            lines.append("```")
            lines.append(str(c["after"]["extraction_region_text"]))
            lines.append("```")
            lines.append("</details>")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = [process_one_call(call) for call in data["calls"]]

    mismatches = [c for c in cases if not c["before_recomputed_matches_stored"]]
    disagreements = [c for c in cases if c["before_after_agree"] is False]

    output = {
        "source_results_json": str(RESULTS_JSON_PATH.relative_to(REPO_ROOT)),
        "n_cases_total": len(cases),
        "n_before_vs_stored_mismatches": len(mismatches),
        "n_before_after_disagreements": len(disagreements),
        "cases": cases,
        "note": (
            "Gerado por reprocessamento OFFLINE -- zero chamadas novas de modelo. "
            "'before_*' usa classify_case_status_v2_detailed sobre o raw_response_text "
            "ORIGINAL (deve bater com classified_status_v2 já salvo em results.json -- "
            "ver n_before_vs_stored_mismatches). 'after' usa o mesmo classificador sobre "
            "a resposta final extraida por extract_final_answer, SOMENTE quando a extracao "
            "nao reportou um status de problema. src/evaluation/case_status_v2.py e os "
            "prompts reais (src/agents/agents.py) NAO foram modificados nesta tarefa."
        ),
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(build_markdown_report(cases))

    print(f"OK -- {len(cases)} casos reprocessados.")
    print(f"Mismatches ANTES vs. classified_status_v2 já salvo: {len(mismatches)}")
    print(f"Divergências ANTES vs. DEPOIS: {len(disagreements)}")
    print(f"JSON: {OUTPUT_JSON_PATH}")
    print(f"MD:   {OUTPUT_MD_PATH}")
    if mismatches:
        print("ATENÇÃO -- mismatches ANTES vs. salvo (investigar antes de confiar no restante):")
        for c in mismatches:
            print(f"  - {c['model_name']}/{c['scenario']}: salvo={c['before_stored_classified_status_v2']} "
                  f"recomputado={c['before_recomputed']['status']}")


if __name__ == "__main__":
    main()
