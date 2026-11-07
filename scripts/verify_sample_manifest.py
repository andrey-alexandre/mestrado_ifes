"""Verify a HAM10000 sample_manifest.csv (already produced by
scripts/run_ham10000_sampling.py) against the 4 objective criteria the
researcher requires before treating it as a candidate for the frozen
protocol sample:

  (a) exactly --expected-n distinct lesion_id values
  (b) one image per lesion (no duplicate lesion_id or image_id)
  (c) reproducible per-category quotas (recomputed from the real metadata
      population via the same functions the sampling run used, compared
      against the manifest's actual per-dx counts)
  (d) every sampled image has a corresponding ground-truth mask file
      (required to render condition S1: original image + reference-mask
      contour)

STATUS: implementada, aguardando execução do pesquisador. This script was
not run by the assistant/fork that wrote it.

This script ONLY verifies and reports. It does not alter the manifest,
does not decide exclusions, and does not freeze the definitive protocol
sample -- that remains a separate, later decision.

Usage:
    python scripts/verify_sample_manifest.py
    python scripts/verify_sample_manifest.py --manifest experiments/sampling/sample_manifest.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import sampling  # noqa: E402

DEFAULT_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_METADATA_CSV = "D:/Andrey/Mestrado/dataverse_files/HAM10000_metadata"
DEFAULT_MASKS_DIR = (
    "D:/Andrey/Mestrado/dataverse_files/HAM10000_segmentations_lesion_tschandl/"
    "HAM10000_segmentations_lesion_tschandl"
)
DEFAULT_EXPECTED_N = 3000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata-csv", default=DEFAULT_METADATA_CSV)
    parser.add_argument("--masks-dir", default=DEFAULT_MASKS_DIR)
    parser.add_argument("--expected-n", type=int, default=DEFAULT_EXPECTED_N)
    parser.add_argument(
        "--exclusions-file",
        default=None,
        help="Mesmo arquivo (se houver) passado a run_ham10000_sampling.py --exclusions-file "
        "ao gerar o manifesto, para que as cotas recomputadas usem a mesma população "
        "pós-exclusão. Hoje nenhuma exclusão aprovada existe (Tarefa 11 encerrada sem "
        "confirmar duplicata).",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "experiments" / "sampling"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"lendo manifesto: {args.manifest}")
    manifest_rows = sampling.read_sample_manifest(args.manifest)
    print(f"linhas no manifesto: {len(manifest_rows)}")

    print("--- critério 1/2: unicidade de lesion_id/image_id ---")
    uniqueness = sampling.verify_lesion_and_image_uniqueness(manifest_rows, args.expected_n)
    print(
        f"lesion_id distintos: {uniqueness['n_distinct_lesion_ids']} "
        f"(esperado {uniqueness['expected_n']}) -- ok={uniqueness['lesion_count_matches_expected']}"
    )
    print(f"uma imagem por lesão: {uniqueness['one_image_per_lesion']}")

    print(f"lendo metadados reais: {args.metadata_csv}")
    metadata_rows = sampling.load_ham10000_metadata(args.metadata_csv)
    excluded_ids: list[str] = []
    if args.exclusions_file:
        excluded_ids = _load_exclusions(Path(args.exclusions_file))

    print("--- critério 3: cotas por categoria reproduzíveis ---")
    quotas = sampling.verify_quota_reproducibility(
        manifest_rows, metadata_rows, args.expected_n, excluded_ids
    )
    print(f"cotas esperadas: {quotas['expected_quotas']}")
    print(f"contagem real no manifesto: {quotas['actual_counts_by_category']}")
    print(f"divergências: {quotas['mismatches'] if quotas['mismatches'] else 'nenhuma'}")

    print(f"--- critério 4: correspondência de máscara (S1), dir={args.masks_dir} ---")
    masks = sampling.verify_mask_correspondence(manifest_rows, args.masks_dir)
    print(
        f"com máscara: {masks['n_with_mask']}/{masks['n_checked']} -- "
        f"sem máscara: {masks['n_missing_mask']}"
    )

    overall_ok = uniqueness["ok"] and quotas["ok"] and masks["ok"]
    print(f"=== RESULTADO GERAL: {'TODOS OS 4 CRITÉRIOS PASSARAM' if overall_ok else 'FALHOU EM AO MENOS 1 CRITÉRIO'} ===")

    report_md = sampling.build_verification_report_markdown(
        manifest_path=args.manifest,
        uniqueness=uniqueness,
        quotas=quotas,
        masks=masks,
        overall_ok=overall_ok,
    )
    report_path = output_dir / "verify_sample_manifest_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"relatório gravado: {report_path}")

    results: dict[str, Any] = {
        "manifest": args.manifest,
        "expected_n": args.expected_n,
        "uniqueness": uniqueness,
        "quotas": quotas,
        "masks": masks,
        "overall_ok": overall_ok,
    }
    results_path = output_dir / "verify_sample_manifest_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"resultados (json) gravados: {results_path}")
    print(
        "LEMBRETE: isto é verificação, não congelamento da amostra definitiva "
        "do protocolo."
    )


def _load_exclusions(path: Path) -> list[str]:
    """Same accepted formats as run_ham10000_sampling.py._load_exclusions
    (CSV with optional 'lesion_id' header, or JSON list)."""
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path}: esperado uma lista JSON de lesion_id, recebido {type(data)}")
        return [str(x) for x in data]

    import csv

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]
    if rows and rows[0] and rows[0][0].strip().lower() == "lesion_id":
        rows = rows[1:]
    return [r[0].strip() for r in rows if r and r[0].strip()]


if __name__ == "__main__":
    main()
