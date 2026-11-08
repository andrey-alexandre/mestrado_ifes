"""Aggregates results across all external-round rehearsal/execution files
already on disk (stage 4, segmentation), written by
``scripts/run_nested_cv_rehearsal.py``.

Scans ``experiments/segmentation/`` for every
``nested_cv_rehearsal_externalfold*_results.json`` present AT THE MOMENT
IT RUNS -- works incrementally, does not require all 5 external rounds to
exist yet. Re-executable at any time; always reflects whatever is on disk
when it runs (no caching, no state carried between runs).

For each architecture, across ALL internal rounds of ALL external-round
files found (up to 4 x n_found):
  - Counts ``stopped_reason == "teto_atingido"`` vs. ``"paciencia_esgotada"``.
  - SEPARATELY counts how many rounds had ``best_epoch == max_epochs`` --
    a stronger, more specific signal than "hit the ceiling" alone (a round
    can hit the ceiling while its best epoch was earlier, e.g. epoch 17 of
    20 -- that round counts as "teto_atingido" but NOT as
    "best_epoch_igual_a_max_epochs"). These two counts are reported
    separately and must never be conflated: "teto_atingido" only means the
    round reached the operational epoch ceiling, not that it was still
    improving when it did.

Per external round found: ``chosen_epochs``, ``chosen_threshold`` (with
``empate``/``empate_simetrico``), and the external evaluation's aggregated
metrics (Dice/IoU/ISIC failure rate/HD95) -- for RECORD-KEEPING AND
TRACKING ONLY. This script does NOT apply the final architecture-selection
rule from the original protocol (Dice difference < 0.005 is a practical
tie, etc.) -- that happens only once all 5 external rounds are complete
and is explicitly requested, not before, and not implemented here.

HAM10000 is never read or referenced here.

STATUS: implementada, aguardando execucao do pesquisador. Not run by the
assistant/fork that wrote it.

Usage:
    python scripts/aggregate_nested_cv_rounds.py
    python scripts/aggregate_nested_cv_rounds.py --output-dir experiments/segmentation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
EXPECTED_EXTERNAL_FOLDS = (0, 1, 2, 3, 4)
RESULT_GLOB_PATTERN = "nested_cv_rehearsal_externalfold*_results.json"


def _discover_external_round_files(output_dir: Path) -> dict[int, Path]:
    """Maps external_fold index -> results.json path, for every matching
    file currently on disk. The external_fold index is read from the
    JSON's own ``"external_fold"`` field (not parsed from the filename),
    so a renamed/copied file is still attributed correctly."""
    found: dict[int, Path] = {}
    for path in sorted(output_dir.glob(RESULT_GLOB_PATTERN)):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        external_fold = data.get("external_fold")
        if external_fold is None:
            continue
        found[external_fold] = path
    return found


def _aggregate_architecture(arch_results_by_fold: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """``arch_results_by_fold``: external_fold -> that architecture's
    result dict (``report["architectures"][arch]`` from one rehearsal
    file)."""
    stopped_reason_counts: dict[str, int] = {}
    best_epoch_equals_max_epochs_count = 0
    n_internal_rounds_total = 0
    per_external_round = []

    for external_fold, res in sorted(arch_results_by_fold.items()):
        for round_result in res["internal_rounds"]:
            n_internal_rounds_total += 1
            reason = round_result.get("stopped_reason")
            stopped_reason_counts[reason] = stopped_reason_counts.get(reason, 0) + 1
            # Separate, more specific signal -- see module docstring.
            if round_result.get("best_epoch") == res.get("max_epochs_used_for_this_round"):
                best_epoch_equals_max_epochs_count += 1

        ts = res["threshold_selection"]
        agg = res["external_evaluation"]["aggregate"]
        per_external_round.append({
            "external_fold": external_fold,
            "chosen_epochs": res["chosen_epochs"],
            "chosen_threshold": ts["chosen_threshold"],
            "empate": ts["empate"],
            "empate_simetrico": ts["empate_simetrico"],
            "external_eval_mean_dice": agg["mean_dice"],
            "external_eval_mean_iou": agg["mean_iou"],
            "external_eval_isic_failure_rate": agg["isic_failure_rate"],
            "external_eval_hd95_median_finite": agg["hd95_median_finite"],
            "external_eval_hd95_n_undefined": agg["hd95_n_undefined"],
            "external_eval_hd95_normalizado_diagonal_median_finite": agg[
                "hd95_normalizado_diagonal_median_finite"
            ],
        })

    return {
        "n_internal_rounds_aggregated": n_internal_rounds_total,
        "stopped_reason_counts": stopped_reason_counts,
        "best_epoch_equals_max_epochs_count": best_epoch_equals_max_epochs_count,
        "note": (
            "A contagem de 'teto_atingido' indica apenas que o treino bateu "
            "no limite operacional de epocas, nao necessariamente que ainda "
            "estava melhorando -- ver a contagem separada de "
            "best_epoch_equals_max_epochs_count para esse sinal mais "
            "especifico (ex.: uma rodada pode bater no teto com a melhor "
            "epoca tendo sido uma epoca anterior, nao a ultima)."
        ),
        "per_external_round": per_external_round,
    }


def run_aggregation(output_dir: Path) -> dict[str, Any]:
    found = _discover_external_round_files(output_dir)
    missing = sorted(set(EXPECTED_EXTERNAL_FOLDS) - set(found.keys()))

    reports_by_fold: dict[int, dict[str, Any]] = {}
    for external_fold, path in found.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # max_epochs used in THAT run, attached per-round so later
        # comparisons never assume a single global ceiling across runs
        # made with different --max-epochs values.
        for arch_res in data["architectures"].values():
            arch_res["max_epochs_used_for_this_round"] = data.get("max_epochs_rehearsal")
        reports_by_fold[external_fold] = data

    all_archs = sorted({arch for data in reports_by_fold.values() for arch in data["architectures"]})

    architectures_summary = {}
    for arch in all_archs:
        arch_results_by_fold = {
            ef: data["architectures"][arch]
            for ef, data in reports_by_fold.items()
            if arch in data["architectures"]
        }
        architectures_summary[arch] = _aggregate_architecture(arch_results_by_fold)

    return {
        "external_folds_found": sorted(found.keys()),
        "external_folds_missing": missing,
        "n_found": len(found),
        "n_expected": len(EXPECTED_EXTERNAL_FOLDS),
        "source_files": {ef: str(path) for ef, path in sorted(found.items())},
        "architectures": architectures_summary,
        "ham10000_touched": False,
        "aviso": (
            "Agregacao incremental dos resultados ja existentes em disco. NAO "
            "aplica a regra final de selecao de arquitetura do protocolo "
            "(isso fica para quando as 5 rodadas externas estiverem completas "
            "e for pedido explicitamente). HAM10000 nao foi tocado."
        ),
    }


def _render_report_md(report: dict[str, Any]) -> str:
    lines = [
        "# Agregação entre rodadas externas da validação cruzada aninhada",
        "",
        f"**{report['aviso']}**",
        "",
        f"- Rodadas externas encontradas: {report['external_folds_found']} "
        f"({report['n_found']}/{report['n_expected']})",
        f"- Rodadas externas faltantes: {report['external_folds_missing']}",
        f"- HAM10000 tocado: {report['ham10000_touched']}",
        "",
    ]
    for arch, summary in report["architectures"].items():
        lines.append(f"## Arquitetura: `{arch}`")
        lines.append("")
        lines.append(f"- Rodadas internas agregadas (até 4 × rodadas externas encontradas): {summary['n_internal_rounds_aggregated']}")
        lines.append(f"- Motivo da parada (contagem): {summary['stopped_reason_counts']}")
        lines.append(f"- **best_epoch == max_epochs** (sinal mais específico, separado): {summary['best_epoch_equals_max_epochs_count']}")
        lines.append(f"- _{summary['note']}_")
        lines.append("")
        lines.append(
            "| fold_externo | chosen_epochs | chosen_threshold | empate | empate_simetrico | "
            "dice_externo | iou_externo | falha_isic_externa | hd95_mediana | hd95_indefinidos | hd95_norm_diag_mediana |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in summary["per_external_round"]:
            lines.append(
                f"| {r['external_fold']} | {r['chosen_epochs']} | {r['chosen_threshold']} | "
                f"{r['empate']} | {r['empate_simetrico']} | {r['external_eval_mean_dice']} | "
                f"{r['external_eval_mean_iou']} | {r['external_eval_isic_failure_rate']} | "
                f"{r['external_eval_hd95_median_finite']} | {r['external_eval_hd95_n_undefined']} | "
                f"{r['external_eval_hd95_normalizado_diagonal_median_finite']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report = run_aggregation(output_dir)

    print("=== AGREGACAO entre rodadas externas ja concluidas ===")
    print(f"encontradas: {report['external_folds_found']} ({report['n_found']}/{report['n_expected']})")
    print(f"faltando: {report['external_folds_missing']}")
    for arch, summary in report["architectures"].items():
        print(
            f"  [{arch}] rodadas_internas={summary['n_internal_rounds_aggregated']} "
            f"motivo_parada={summary['stopped_reason_counts']} "
            f"best_epoch_eq_max={summary['best_epoch_equals_max_epochs_count']}"
        )

    results_json_path = output_dir / "nested_cv_aggregate_report.json"
    report_md_path = output_dir / "nested_cv_aggregate_report.md"

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(_render_report_md(report))

    print(f"\nresultados (json) gravados: {results_json_path}")
    print(f"relatorio (md) gravado: {report_md_path}")
    print(
        "\nLEMBRETE: agregacao incremental, nao aplica a regra final de "
        "selecao de arquitetura. HAM10000 nao foi tocado."
    )


if __name__ == "__main__":
    main()
