"""Reprocessamento OFFLINE das 8 chamadas reais já salvas pela 1ª
execução dos controles de síntese com imagem real via contrato de
decisão experimental -- ZERO chamadas novas de modelo, ZERO rede, ZERO
inferência.

Contexto (ver `docs/reconstruction/25_accent_variant_and_reprocessing.md`):
a 1ª execução real (`scripts/run_synthesis_controls_real_image_experimental_decision_contract.py`,
resultado em
`experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`)
teve 2 das 8 chamadas (ambas `medgemma1.5:4b-it-q4_K_M`, cenários
`unanime_negativo` e `conflitante`) classificadas como
`decision_status: "invalid_value"` porque o modelo escreveu
`decisao_final: não_melanoma` (grafia correta em português, com "ã")
em vez do token ASCII `nao_melanoma` pedido literalmente na instrução.
Decisão do pesquisador: aceitar `"não_melanoma"` como variante literal
FECHADA de `"nao_melanoma"`, só dentro do campo `decisao_final`, nos
dois papéis (VLM-direto e sumarizador, que usam o mesmo parser) --
implementada em `src.evaluation.explicit_decision_contract
._VALUE_VARIANTS` (ver módulo, não modificado por este script).

Este script NÃO faz nenhuma chamada nova a modelo -- lê o
`raw_response_text` já salvo das 8 chamadas reais e reaplica
`extract_decisao_final` (já com a variante aceita) para cada uma,
registrando ANTES (valor já salvo em `results.json`, re-verificado
explicitamente, não assumido) e DEPOIS (recomputado agora) lado a
lado.

Grava um arquivo NOVO (nunca sobrescreve `results.json`/`report.md`
originais, que preservam o registro histórico da 1ª execução real):

  - `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/
    accent_variant_reprocessing_results.json`
  - `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/
    accent_variant_reprocessing_report.md`

NÃO modifica `src/evaluation/explicit_decision_contract.py`,
`src/multiagent/experimental_synthesis_control_real_image.py`,
`src/evaluation/case_status.py`/`case_status_v2.py`,
`src/agents/agents.py` nem `scripts/run_local_vlm_pilot.py` (critério
de avanço do Estágio 1, fora de escopo). Nenhum destes é importado
aqui exceto `extract_decisao_final`, usado em modo SOMENTE LEITURA
sobre texto já salvo em disco.

Uso (pelo PESQUISADOR -- este script não é executado por nenhum
agente desta tarefa):

    python scripts/run_reprocess_experimental_decision_contract_accent_variant.py

Saída: os dois arquivos acima, mais um resumo no stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.explicit_decision_contract import extract_decisao_final  # noqa: E402

INPUT_RESULTS_PATH = (
    REPO_ROOT
    / "experiments"
    / "multiagent"
    / "synthesis_controls_real_image_experimental_decision_contract"
    / "results.json"
)
OUTPUT_DIR = INPUT_RESULTS_PATH.parent
OUTPUT_RESULTS_PATH = OUTPUT_DIR / "accent_variant_reprocessing_results.json"
OUTPUT_REPORT_PATH = OUTPUT_DIR / "accent_variant_reprocessing_report.md"


def reprocess_one_call(saved_entry: dict[str, Any]) -> dict[str, Any]:
    """Recomputa `extract_decisao_final` sobre o `raw_response_text` JÁ
    SALVO de uma chamada -- nenhuma chamada nova a modelo. Compara
    explicitamente o resultado "antes" (já salvo em `results.json`) com
    o "depois" (recomputado agora, já com a variante acentuada
    aceita)."""
    raw_text = saved_entry["raw_response_text"]
    before = {
        "decision_status": saved_entry["decision_status"],
        "decisao_final": saved_entry["decisao_final"],
        "matched_raw_values": saved_entry["matched_raw_values"],
        "template_extraction_status": saved_entry["template_extraction_status"],
    }
    after_result = extract_decisao_final(raw_text)
    after = {
        "decision_status": after_result.decision_status,
        "decisao_final": after_result.decisao_final,
        "matched_raw_values": list(after_result.matched_raw_values),
        "template_extraction_status": after_result.template_extraction_status,
    }
    changed = before != after
    return {
        "model_name": saved_entry["model_name"],
        "scenario": saved_entry["scenario"],
        "call_status": saved_entry["call_status"],
        "raw_response_text": raw_text,
        "before": before,
        "after": after,
        "changed": changed,
    }


def build_report_markdown(reprocessed: list[dict[str, Any]]) -> str:
    n_changed = sum(1 for r in reprocessed if r["changed"])
    lines = [
        "# Reprocessamento offline -- variante acentuada de `nao_melanoma`",
        "",
        "ZERO chamadas novas de modelo -- só reaplica `extract_decisao_final` "
        "(já com `não_melanoma` aceito como variante de `nao_melanoma`) sobre o "
        "`raw_response_text` já salvo das 8 chamadas reais da 1ª execução "
        "(ver `experiments/multiagent/synthesis_controls_real_image_experimental_decision_contract/results.json`, "
        "não modificado por este script).",
        "",
        f"- Total de chamadas reprocessadas: {len(reprocessed)} (esperado 8)",
        f"- Chamadas cujo `decision_status`/`decisao_final` MUDOU: {n_changed}",
        "",
        "## Chamadas (uma seção por cenário x modelo)",
        "",
    ]
    for r in reprocessed:
        lines += [
            f"### modelo=`{r['model_name']}` cenário=`{r['scenario']}`",
            "",
            f"- `changed`: **{r['changed']}**",
            f"- ANTES: `decision_status={r['before']['decision_status']!r}` "
            f"`decisao_final={r['before']['decisao_final']!r}` "
            f"`matched_raw_values={r['before']['matched_raw_values']!r}`",
            f"- DEPOIS: `decision_status={r['after']['decision_status']!r}` "
            f"`decisao_final={r['after']['decisao_final']!r}` "
            f"`matched_raw_values={r['after']['matched_raw_values']!r}`",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    if not INPUT_RESULTS_PATH.exists():
        print(f"ABORTADO: não encontrado: {INPUT_RESULTS_PATH}")
        return

    with open(INPUT_RESULTS_PATH, encoding="utf-8") as f:
        saved_results = json.load(f)

    saved_calls = saved_results["calls"]
    if len(saved_calls) != 8:
        print(
            f"AVISO: esperado 8 chamadas salvas, encontrado {len(saved_calls)} -- "
            "prosseguindo mesmo assim, reprocessando o que existe."
        )

    reprocessed = [reprocess_one_call(entry) for entry in saved_calls]

    n_changed = sum(1 for r in reprocessed if r["changed"])
    print(f"Reprocessadas {len(reprocessed)} chamadas (zero chamadas novas de modelo).")
    print(f"Mudaram de decision_status/decisao_final: {n_changed}")
    for r in reprocessed:
        if r["changed"]:
            print(
                f"  - {r['model_name']} / {r['scenario']}: "
                f"antes={r['before']['decision_status']}/{r['before']['decisao_final']!r} -> "
                f"depois={r['after']['decision_status']}/{r['after']['decisao_final']!r}"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"source_results_path": str(INPUT_RESULTS_PATH), "calls": reprocessed},
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(build_report_markdown(reprocessed))

    print(f"\nresultados (json) gravados: {OUTPUT_RESULTS_PATH}")
    print(f"relatório (md) gravado: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()
