"""Content-hash (sha256) audit for the two zip-only datasets prioritized by
the researcher for cross-dataset overlap checking: ISIC2018 Task1 and
BCN20000 (task 6 of the reconstruction).

Hashes every image entry's bytes directly: from the zip for ISIC2018 Task1
(no extraction), and from BCN20000's inner per-split zips after a one-time,
idempotent materialization to D: (random access into a zip nested inside
another zip's stream is not otherwise possible). Individual images are
never extracted to disk in either case.

This is DETECTION input only — it builds manifests. The actual overlap
comparison against HAM10000 happens in ``scripts/run_overlap_audit.py``,
which reads the JSON manifests written here.

Usage:
    python scripts/run_content_hash_audit.py
    python scripts/run_content_hash_audit.py --config config/dataset_inventory.yaml
    python scripts/run_content_hash_audit.py --extracted-dir D:/Andrey/Mestrado/_extracted
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import content_hash, core  # noqa: E402

DEFAULT_EXTRACTED_DIR = "D:/Andrey/Mestrado/_extracted"


def _csv_row_count(outer_zip: Path, csv_entry: str) -> int:
    text = core.read_zip_entry_text(outer_zip, csv_entry)
    return len(core.parse_csv_text(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"),
        help="Caminho do YAML de configuração do inventário (mesmo do task 2).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "data_audit"),
        help="Diretório onde os manifestos JSON de conteúdo serão gravados.",
    )
    parser.add_argument(
        "--extracted-dir",
        default=DEFAULT_EXTRACTED_DIR,
        help=(
            "Onde materializar os zips internos do BCN20000 (deve estar em "
            "D:, nunca em C:; persistente, reaproveitado por tarefas futuras)."
        ),
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== ISIC2018 Task1 (hash de conteúdo, direto do zip, sem extração) ===")
    isic_result = content_hash.hash_isic2018_task1_images(
        images_zip=config["isic2018_task1"]["images_zip"],
    )
    print(
        f"expected={isic_result['expected']} processed={isic_result['processed']} "
        f"missing={isic_result['missing']} unreadable={len(isic_result['unreadable'])}"
    )
    isic_path = output_dir / "isic2018_task1_content_manifest.json"
    with open(isic_path, "w", encoding="utf-8") as f:
        json.dump(isic_result, f, ensure_ascii=False, indent=2)
    print(f"manifesto gravado: {isic_path}")

    print("=== BCN20000 (materialização em D: + hash de conteúdo) ===")
    bcn_config = config["bcn20000"]
    outer_zip = Path(bcn_config["outer_zip"])
    train_csv_count = _csv_row_count(outer_zip, bcn_config["train_csv_entry"])
    test_csv_count = _csv_row_count(outer_zip, bcn_config["test_csv_entry"])
    bcn_result = content_hash.hash_bcn20000_images(
        outer_zip=outer_zip,
        train_zip_entry=bcn_config["train_zip_entry"],
        test_zip_entry=bcn_config["test_zip_entry"],
        extracted_dir=args.extracted_dir,
        train_csv_row_count=train_csv_count,
        test_csv_row_count=test_csv_count,
    )
    for split in ("train", "test"):
        s = bcn_result[split]
        m = s["materialization"]
        print(
            f"[{split}] expected={s['expected']} processed={s['processed']} "
            f"missing={s['missing']} unreadable={len(s['unreadable'])} "
            f"csv_row_count={s['csv_row_count']} "
            f"csv_vs_zip_count_matches={s['csv_vs_zip_count_matches']} "
            f"materialized={m['materialized']} reused_existing={m['reused_existing']} "
            f"dest_path={m['dest_path']} size={m['size']}"
        )
    bcn_path = output_dir / "bcn20000_content_manifest.json"
    with open(bcn_path, "w", encoding="utf-8") as f:
        json.dump(bcn_result, f, ensure_ascii=False, indent=2)
    print(f"manifesto gravado: {bcn_path}")


if __name__ == "__main__":
    main()
