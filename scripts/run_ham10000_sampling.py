"""Task 12: build the HAM10000 experimental sample manifest (3,000 lesions,
quotas proportional to the 7 dx categories via Hamilton's largest-remainder
method, one image per sampled lesion).

STATUS: implementada, aguardando execução do pesquisador. This script was
not run by the assistant that wrote it -- see
docs/reconstruction/09_sampling_infrastructure_report.md for the exact
commands to run and what to return for analysis.

**Esta execução NÃO congela a amostra definitiva do protocolo** -- é
preparação/teste; o congelamento definitivo depende do fechamento de
quaisquer exclusões aprovadas (decisão fora do escopo desta tarefa).

Population eligibility: all 7 HAM10000 dx categories (akiec, bcc, bkl, df,
mel, nv, vasc), no category excluded. Unit of eligibility/sampling is the
lesion (lesion_id), not the image.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import sampling  # noqa: E402

DEFAULT_METADATA_CSV = "D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata"
DEFAULT_SEED = 20261106


def _load_exclusions(path: Path) -> list[str]:
    """Accepts a CSV (one lesion_id per line, optionally with a header named
    'lesion_id') or a JSON list of lesion_id strings."""
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: esperado uma lista JSON de lesion_id, recebido {type(data)}")
        return [str(x) for x in data]

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]
    if rows and rows[0] and rows[0][0].strip().lower() == "lesion_id":
        rows = rows[1:]
    return [r[0].strip() for r in rows if r and r[0].strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--target-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--exclusions-file",
        default=None,
        help="CSV (coluna/linha 'lesion_id') ou JSON (lista de lesion_id) a excluir ANTES da amostragem. "
        "Nenhuma exclusão aprovada existe hoje (Tarefa 11 encerrada sem confirmar duplicata).",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments" / "sampling"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"lendo metadados: {args.metadata_csv}")
    rows = sampling.load_ham10000_metadata(args.metadata_csv)
    print(f"linhas de metadados lidas: {len(rows)}")

    population_check = sampling.check_lesion_population(rows)
    if not population_check["ok"]:
        result = {
            "status": "populacao_invalida",
            "population_check": population_check,
        }
        _write_json(output_dir / "sampling_results.json", result)
        print(
            "POPULAÇÃO INVÁLIDA (lesion_id ausente, dx conflitante, ou dx fora das "
            "7 categorias esperadas) -- amostragem NÃO executada. Ver "
            f"{output_dir / 'sampling_results.json'}"
        )
        return

    lesion_table = sampling.build_lesion_table(rows)
    print(f"lesões distintas (antes de exclusões): {len(lesion_table)}")

    exclusions_summary: dict[str, Any] | None = None
    if args.exclusions_file:
        excluded_ids = _load_exclusions(Path(args.exclusions_file))
        exclusion_result = sampling.apply_exclusions(lesion_table, excluded_ids)
        lesion_table = exclusion_result["filtered_lesion_table"]
        exclusions_summary = {
            "n_requested": exclusion_result["n_requested"],
            "n_actually_excluded": exclusion_result["n_actually_excluded"],
            "unknown_ids": exclusion_result["unknown_ids"],
        }
        print(
            f"exclusões aplicadas: {exclusion_result['n_actually_excluded']}/"
            f"{exclusion_result['n_requested']} solicitadas efetivamente excluídas "
            f"({len(exclusion_result['unknown_ids'])} id(s) desconhecido(s))"
        )

    counts_by_category = {
        cat: sum(1 for v in lesion_table.values() if v["dx"] == cat)
        for cat in sampling.ELIGIBLE_DX_CATEGORIES
    }
    n_eligible_total = sum(counts_by_category.values())
    print(f"lesões elegíveis por categoria: {counts_by_category}")
    print(f"total elegível (após exclusões, se houver): {n_eligible_total}")

    if n_eligible_total < args.target_n:
        result = {
            "status": "populacao_insuficiente",
            "target_n": args.target_n,
            "n_eligible_total": n_eligible_total,
            "counts_by_category": counts_by_category,
            "population_check": population_check,
            "exclusions": exclusions_summary,
            "note": (
                f"População elegível ({n_eligible_total}) é menor que o alvo "
                f"({args.target_n}). Amostragem NÃO executada. Alvo não foi reduzido "
                "silenciosamente nem completado com outras imagens das mesmas lesões."
            ),
        }
        _write_json(output_dir / "sampling_results.json", result)
        print(
            f"POPULAÇÃO INSUFICIENTE: {n_eligible_total} < alvo {args.target_n} -- "
            f"amostragem NÃO executada. Ver {output_dir / 'sampling_results.json'}"
        )
        return

    quotas = sampling.compute_quotas_largest_remainder(counts_by_category, args.target_n)
    print(f"cotas finais por categoria: {quotas['final_quotas']}")

    sample_result = sampling.sample_lesions(lesion_table, quotas["final_quotas"], args.seed)
    sampled_rows = sample_result["sampled_rows"]
    print(f"lesões sorteadas: {len(sampled_rows)}")

    manifest_path = output_dir / "sample_manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lesion_id", "image_id", "dx", "binary_target"])
        writer.writeheader()
        writer.writerows(sampled_rows)
    print(f"manifesto gravado: {manifest_path}")

    summary_md = sampling.build_sampling_summary_markdown(
        metadata_csv=args.metadata_csv,
        target_n=args.target_n,
        seed=args.seed,
        population_check=population_check,
        counts_by_category=counts_by_category,
        quotas=quotas,
        exclusions=exclusions_summary,
        n_sampled_rows=len(sampled_rows),
        n_eligible_total=n_eligible_total,
    )
    summary_path = output_dir / "sampling_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"resumo gravado: {summary_path}")

    results_json = {
        "status": "concluido_teste_nao_definitivo",
        "metadata_csv": args.metadata_csv,
        "target_n": args.target_n,
        "seed": args.seed,
        "population_check": population_check,
        "exclusions": exclusions_summary,
        "counts_by_category": counts_by_category,
        "n_eligible_total": n_eligible_total,
        "quotas": quotas,
        "n_sampled_rows": len(sampled_rows),
        "manifest_path": _relative_or_absolute(manifest_path, REPO_ROOT),
    }
    results_path = output_dir / "sampling_results.json"
    _write_json(results_path, results_json)
    print(f"resultados (json completo) gravado: {results_path}")
    print(
        "LEMBRETE: esta execução é de preparação/teste, NÃO congela a amostra "
        "definitiva do protocolo."
    )


def _relative_or_absolute(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` when it's actually inside it (the
    common case, ``--output-dir`` under the repo), else the absolute path
    as-is (``--output-dir`` pointed elsewhere, e.g. outside the repo)."""
    try:
        return str(path.relative_to(base)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
