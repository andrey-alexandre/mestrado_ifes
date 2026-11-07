"""Register the SHA-256 of the APPROVED, frozen HAM10000 sample manifest.

This is a one-time formalization step, not a regenerable audit report: it
records the exact byte-identity (hash) of the manifest file the researcher
approved as the definitive protocol sample, alongside the seed used to
generate it and a reference to its verification report
(scripts/verify_sample_manifest.py's output). The hash is the real
identifier of the approved version -- the seed alone does not identify the
file's content (a code change, a different HAM10000 metadata snapshot, or
a different random-number-generator version could all produce a different
file from the same seed).

STATUS: implementada, aguardando execução do pesquisador. This script was
not run by the assistant that wrote it.

This script does not alter the manifest, does not re-run sampling, and
does not re-verify the 4 criteria (scripts/verify_sample_manifest.py
already did that) -- it only computes a hash and writes the registry
entry. Requires --confirm-freeze to run, so it cannot be invoked by
accident and silently produce a new "frozen sample" registration.

Usage:
    python scripts/register_frozen_sample.py --confirm-freeze
    python scripts/register_frozen_sample.py --confirm-freeze \
        --manifest experiments/sampling/sample_manifest.csv \
        --seed 20261106 \
        --verification-report experiments/sampling/verify_sample_manifest_report.md \
        --verification-results experiments/sampling/verify_sample_manifest_results.json \
        --output docs/reconstruction/10_frozen_sample_registry.md
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import core  # noqa: E402

DEFAULT_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_SEED = 20261106
DEFAULT_VERIFICATION_REPORT = str(
    REPO_ROOT / "experiments" / "sampling" / "verify_sample_manifest_report.md"
)
DEFAULT_VERIFICATION_RESULTS = str(
    REPO_ROOT / "experiments" / "sampling" / "verify_sample_manifest_results.json"
)
DEFAULT_OUTPUT = str(REPO_ROOT / "docs" / "reconstruction" / "10_frozen_sample_registry.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-freeze",
        action="store_true",
        help="Obrigatório. Confirma que esta execução está registrando o "
        "congelamento aprovado pelo pesquisador, não uma amostra de teste.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--verification-report", default=DEFAULT_VERIFICATION_REPORT)
    parser.add_argument("--verification-results", default=DEFAULT_VERIFICATION_RESULTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.confirm_freeze:
        print(
            "ERRO: --confirm-freeze não foi passado. Nada foi escrito. "
            "Rode de novo com --confirm-freeze só depois de ter a aprovação "
            "explícita do pesquisador para este arquivo específico."
        )
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"ERRO: manifesto não encontrado em {manifest_path}")
        sys.exit(1)

    print(f"lendo e hasheando: {manifest_path}")
    manifest_sha256 = core.sha256_file(manifest_path)
    manifest_size = manifest_path.stat().st_size
    print(f"SHA-256: {manifest_sha256}")
    print(f"tamanho: {manifest_size} bytes")

    verification_results_path = Path(args.verification_results)
    verification_summary = None
    if verification_results_path.is_file():
        with open(verification_results_path, encoding="utf-8") as f:
            verification_summary = json.load(f)
        print(
            f"relatório de verificação lido: overall_ok="
            f"{verification_summary.get('overall_ok')}"
        )
    else:
        print(
            f"AVISO: {verification_results_path} não encontrado -- o registro "
            "será gravado sem o resumo da verificação embutido."
        )

    registry_md = _build_registry_markdown(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        manifest_size=manifest_size,
        seed=args.seed,
        verification_report_path=args.verification_report,
        verification_summary=verification_summary,
        approval_date=date.today().isoformat(),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(registry_md)
    print(f"registro de congelamento gravado: {output_path}")
    print(
        "LEMBRETE: preserve experiments/sampling/sample_manifest.csv exatamente "
        "como está -- o SHA-256 acima só é válido para este conteúdo exato. "
        "Qualquer alteração do arquivo invalida este registro."
    )


def _build_registry_markdown(
    *,
    manifest_path: str,
    manifest_sha256: str,
    manifest_size: int,
    seed: int,
    verification_report_path: str,
    verification_summary: dict | None,
    approval_date: str,
) -> str:
    lines = [
        "# Registro de Congelamento — Amostra Definitiva do Protocolo (HAM10000, 3.000 lesões)",
        "",
        "**Este arquivo registra, de forma permanente, a identidade exata da "
        "amostra aprovada pelo pesquisador como entrada fixa de todas as "
        "condições diagnósticas (S0/S1/S2) e de todos os sistemas/modelos "
        "avaliados no protocolo experimental.**",
        "",
        f"- Data de aprovação: {approval_date}",
        f"- Arquivo: `{manifest_path}`",
        f"- Tamanho: {manifest_size} bytes",
        f"- **SHA-256: `{manifest_sha256}`**",
        f"- Semente usada na geração: `{seed}`",
        "",
        "## Por que o SHA-256, não só a semente",
        "",
        "A semente sozinha não identifica o conteúdo do arquivo: uma mudança "
        "de código na amostragem, uma versão diferente do instantâneo de "
        "metadados do HAM10000, ou uma versão diferente do gerador de números "
        "aleatórios do Python poderiam produzir um arquivo diferente a partir "
        "da mesma semente. O SHA-256 acima é o identificador real da versão "
        "aprovada — qualquer cópia deste manifesto usada em qualquer etapa "
        "futura do protocolo deve ter exatamente este hash.",
        "",
        "## Verificação que fundamentou a aprovação",
        "",
        f"- Relatório: `{verification_report_path}`",
    ]

    if verification_summary is not None:
        lines += [
            f"- Resultado geral: {'TODOS OS 4 CRITÉRIOS PASSARAM' if verification_summary.get('overall_ok') else 'FALHOU EM AO MENOS 1 CRITÉRIO'}",
            f"- `lesion_id` distintos: {verification_summary.get('uniqueness', {}).get('n_distinct_lesion_ids')}",
            f"- Divergências de cota: {verification_summary.get('quotas', {}).get('mismatches') or 'nenhuma'}",
            f"- Máscaras ausentes (S1): {verification_summary.get('masks', {}).get('n_missing_mask')}",
        ]
    else:
        lines.append(
            "- (resumo não disponível no momento deste registro -- ver o "
            "relatório Markdown diretamente)"
        )

    lines += [
        "",
        "## Decisão associada",
        "",
        "Tarefa 11 (auditoria de duplicata visual, Tarefas 7–11) foi encerrada "
        "pelo pesquisador sem nenhuma exclusão aprovada por duplicidade — os 2 "
        "pares `indeterminado` permanecem no registro como indeterminados, não "
        "excluídos e não confirmados. Nenhuma outra exclusão foi aplicada a "
        "esta amostra.",
        "",
        "## Regra de uso",
        "",
        "Qualquer execução experimental (sistema multiagente, CNN baseline, "
        "qualquer condição visual) que use \"a amostra do HAM10000\" deve usar "
        "exatamente este arquivo, identificado pelo SHA-256 acima — não "
        "regenerar, não usar outra semente, não usar uma cópia não verificada "
        "contra este hash.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
