"""Remove, de um `call_log_main.jsonl`, só as linhas cuja `failure_cause`
contém uma substring exata dada -- usado para o caso real encontrado em
2026-11-18: 339 tentativas do GPT (`gpt-5.4-nano`) falharam 100% por
`OpenAIAuthenticationError: ... expired_secret_key` (chave de API
expirada) -- um problema OPERACIONAL de credencial, nunca uma tentativa
real contra o modelo (a API nunca foi alcançada de verdade). Sob a
política de não-retry, essas 339 ficariam permanentemente marcadas como
"já tentadas" mesmo depois da chave renovada -- este script as remove
do log para que voltem a ser tentadas de verdade na próxima execução,
sem contar como resultado real.

NUNCA remove silenciosamente: sempre grava um backup do arquivo original
ANTES de qualquer alteração (`<arquivo>.bak_<timestamp>.jsonl`, nunca
sobrescrito), imprime exatamente quantas linhas seriam removidas e quais
causas distintas correspondem ao filtro antes de perguntar confirmação
via `--confirm` explícito (sem `--confirm`, só mostra o que FARIA, não
grava nada).

STATUS: implementado, NÃO executado pelo assistente que o escreveu --
aguardando execução do pesquisador.

Uso:
    # 1. Dry-run (padrão) -- mostra o que seria removido, não altera nada:
    python scripts/purge_call_log_failures.py \\
        --call-log experiments/multiagent/full_scale_matrix/gpt-5.4-nano/call_log_main.jsonl \\
        --failure-cause-contains expired_secret_key

    # 2. Confirmado -- grava backup, remove as linhas, reescreve o arquivo:
    python scripts/purge_call_log_failures.py \\
        --call-log experiments/multiagent/full_scale_matrix/gpt-5.4-nano/call_log_main.jsonl \\
        --failure-cause-contains expired_secret_key --confirm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--call-log", required=True, help="call_log_main.jsonl a limpar.")
    parser.add_argument(
        "--failure-cause-contains", required=True,
        help="Substring EXATA que failure_cause precisa conter para a linha ser removida (ex.: expired_secret_key).",
    )
    parser.add_argument("--model-name", default=None, help="Se fornecido, só remove linhas também com este model_name exato (checagem extra de segurança).")
    parser.add_argument("--confirm", action="store_true", help="Sem isto, só mostra o que seria removido (dry-run) -- nunca altera o arquivo.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    log_path = Path(args.call_log)
    if not log_path.exists():
        raise SystemExit(f"ERRO OPERACIONAL: {log_path} não encontrado.")

    with open(log_path, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    kept_lines: list[str] = []
    removed_lines: list[str] = []
    removed_causes: dict[str, int] = {}
    for line in lines:
        record = json.loads(line)
        cause = record.get("failure_cause") or ""
        matches_cause = args.failure_cause_contains in cause
        matches_model = args.model_name is None or record.get("model_name") == args.model_name
        if matches_cause and matches_model:
            removed_lines.append(line)
            removed_causes[cause] = removed_causes.get(cause, 0) + 1
        else:
            kept_lines.append(line)

    print(f"Total de linhas no arquivo: {len(lines)}")
    print(f"Linhas que seriam removidas: {len(removed_lines)}")
    print(f"Linhas que permaneceriam: {len(kept_lines)}")
    print("Causas distintas entre as removidas:")
    for cause, count in sorted(removed_causes.items(), key=lambda kv: -kv[1]):
        print(f"  {count}x  {cause[:160]}")

    if not removed_lines:
        print("\nNenhuma linha corresponde ao filtro -- nada a fazer.")
        return

    if not args.confirm:
        print("\nDRY-RUN (padrão) -- nenhum arquivo foi alterado. Reinvoque com --confirm para aplicar.")
        return

    backup_path = log_path.with_name(
        f"{log_path.stem}.bak_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    with open(backup_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"\nBackup do arquivo original (intocado) gravado em: {backup_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    print(f"{len(removed_lines)} linha(s) removida(s) de {log_path} -- {len(kept_lines)} linha(s) restante(s).")
    print(
        "As chamadas correspondentes voltam a ser tentadas de verdade na próxima "
        "invocação de --mode execute (não contam mais como 'já tentadas')."
    )


if __name__ == "__main__":
    main()
