"""Etapa 5 — checagem MANUAL/standalone de reaproveitamento de chamadas já
executadas, antes de rodar (ou re-rodar) a matriz de VLMs sobre a amostra
congelada ou o piloto de 12 lesões (ver
docs/reconstruction/30_real_lesion_evaluation_protocol.md §3).

A lógica de equivalência (``is_call_record_reusable``/``check_reuse``) foi
movida para `src.multiagent.vlm_direct` (seção 13) nesta entrega, para ser
a ÚNICA implementação -- reusada tanto por este script (checagem manual,
externa, sob demanda) quanto por `scripts/run_full_scale_vlm_matrix.py`
(que precisa da mesma checagem acoplada ao próprio executor, não só como
um passo manual). Este script agora é um wrapper fino de linha de
comando sobre essa função; nenhuma regra de equivalência está duplicada
aqui.

Lê um ou mais arquivos `call_log*.jsonl` (formato produzido por
`src.multiagent.vlm_direct.CallRecord.to_flat_dict`) e compara contra uma
lista de chamadas planejadas (`role`, `condition`, `lesion_id`,
`image_id`). Ver `src.multiagent.vlm_direct.is_call_record_reusable` para
os critérios exatos de equivalência (lesão+imagem, condição, `model_name`
exato, e -- corrigido nesta entrega -- versão do contrato de decisão
REALMENTE comparada via o novo campo `CallRecord.decision_contract_
parser_version`, não só afirmada no docstring).

STATUS: implementado, aguardando execução do pesquisador
(`feedback_no_autonomous_execution`) -- não executado pelo assistente
que o escreveu.

Uso:
    python scripts/check_vlm_matrix_reuse.py \\
        --model-name qwen3-vl:8b-instruct-q4_K_M \\
        --call-log experiments/multiagent/local_vlm_pilot/qwen3-vl_8b-instruct-q4_K_M/call_log_matriz_principal.jsonl \\
        --planned-calls-json experiments/multiagent/pilot12_planned_calls.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent.vlm_direct import check_reuse  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--call-log", action="append", required=True, dest="call_logs",
        help="Caminho de um call_log*.jsonl já gravado. Pode ser passado várias vezes (ex.: controles + matriz principal).",
    )
    parser.add_argument(
        "--planned-calls-json", required=True,
        help="JSON com uma lista de objetos {role, condition, lesion_id, image_id} -- a matriz que se pretende rodar.",
    )
    parser.add_argument("--output-path", default=None, help="Default: ao lado de --planned-calls-json, sufixo _reuse_check.json")
    return parser


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: call log não encontrado: {path}")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = build_arg_parser().parse_args()

    existing_records: list[dict[str, Any]] = []
    for call_log_path in args.call_logs:
        existing_records.extend(_load_jsonl(Path(call_log_path)))

    planned_calls_path = Path(args.planned_calls_json)
    if not planned_calls_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {planned_calls_path} não encontrado.")
    with open(planned_calls_path, encoding="utf-8") as f:
        planned_calls = json.load(f)

    result = check_reuse(args.model_name, existing_records, planned_calls)

    output_path = Path(args.output_path) if args.output_path else planned_calls_path.with_name(
        planned_calls_path.stem + "_reuse_check.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print(f"modelo: {args.model_name}")
    print(f"planejadas: {result['n_planned']}  já cobertas por saída existente: {result['n_already_covered_by_existing_output']}  faltam rodar: {result['n_missing_must_run']}")
    print(f"detalhe gravado em: {output_path}")


if __name__ == "__main__":
    main()
