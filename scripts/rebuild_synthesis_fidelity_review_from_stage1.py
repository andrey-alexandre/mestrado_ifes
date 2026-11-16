"""Reconstrói o template de revisão de FIDELIDADE da síntese (ver
`src/multiagent/vlm_direct.py`, seção 12c, critério fechado pelo
pesquisador em 2026-11-16) a partir do Estágio 1 de um modelo JÁ
EXECUTADO E SALVO (`summary.json`), SEM NENHUMA NOVA INFERÊNCIA: usa
literalmente as 4 justificativas e os sinais heurísticos já computados e
salvos naquela execução real (`summary.json::controls.synthesis`), só
reorganizados no formato do novo critério -- a mesma função
(`vd.build_synthesis_fidelity_review_template`) que
`scripts/run_local_vlm_pilot.py` chamaria se o Estágio 1 fosse executado
hoje.

Serve para os modelos cujo Estágio 1 já rodou sob o gate ANTIGO (achado
novo) antes desta mudança -- hoje, só `qwen3-vl:8b-instruct-q4_K_M`. Um
Estágio 1 executado DEPOIS desta mudança já grava
`synthesis_fidelity_review_template.json` diretamente; este script não é
necessário nesse caso.

NÃO lê, não converte, nem apaga nenhum `synthesis_manual_review_template.json`
antigo (vocabulário "achado novo") -- esse arquivo fica onde está,
congelado, só para rastreabilidade. NÃO executa nenhuma chamada a
modelo/Ollama -- só lê um JSON já salvo e escreve outro JSON.

STATUS: implementado, aguardando execução do pesquisador
(`feedback_no_autonomous_execution`) -- não executado pelo assistente
que o escreveu.

Uso:
    python scripts/rebuild_synthesis_fidelity_review_from_stage1.py \\
        --model-dir experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent import vlm_direct as vd  # noqa: E402

DEFAULT_MODEL_DIR = str(
    REPO_ROOT / "experiments" / "multiagent" / "local_vlm_pilot" / "qwen3-vl_8b-instruct-q4_K_M"
)
EXPECTED_STAGE = "aguardando_revisao_manual_sintese"
EXPECTED_N_SCENARIOS = 4


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Pasta com o summary.json do Estágio 1 já executado.")
    parser.add_argument(
        "--output-filename", default="synthesis_fidelity_review_template.json",
        help="Nome do arquivo de saída dentro de --model-dir.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    model_dir = Path(args.model_dir)
    summary_path = model_dir / "summary.json"

    if not summary_path.exists():
        print(f"ERRO: {summary_path} não encontrado.", file=sys.stderr)
        sys.exit(1)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    stage = summary.get("stopped_at_stage")
    if stage != EXPECTED_STAGE:
        print(
            f"ERRO: {summary_path} tem stopped_at_stage='{stage}', esperado "
            f"'{EXPECTED_STAGE}' -- este Estágio 1 não chegou ao ponto de "
            "controles de síntese concluídos, ou já passou disso. Nada foi "
            "gravado.",
            file=sys.stderr,
        )
        sys.exit(1)

    synthesis_results = summary.get("controls", {}).get("synthesis")
    if not synthesis_results:
        print(f"ERRO: {summary_path}::controls.synthesis está vazio/ausente.", file=sys.stderr)
        sys.exit(1)
    if len(synthesis_results) != EXPECTED_N_SCENARIOS:
        print(
            f"ERRO: {summary_path}::controls.synthesis tem {len(synthesis_results)} "
            f"cenários, esperado exatamente {EXPECTED_N_SCENARIOS}. Nada foi gravado.",
            file=sys.stderr,
        )
        sys.exit(1)

    template = vd.build_synthesis_fidelity_review_template(synthesis_results)

    output_path = model_dir / args.output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2, default=str)

    print(f"template de revisão de fidelidade gravado em: {output_path}")
    print(f"regra: {template['rule_version']}")
    print(f"cenários: {[s['scenario'] for s in template['scenarios']]}")
    print(
        "\nPróximo passo: preencher 'fidelity_verdict' e 'reviewer_notes' de "
        "cada cenário em", output_path,
        "\nDepois rodar:\n"
        f"  python scripts/run_local_vlm_pilot.py --model-tag qwen3-vl:8b-instruct-q4_K_M "
        f"--synthesis-fidelity-review-file {output_path}"
    )


if __name__ == "__main__":
    main()
