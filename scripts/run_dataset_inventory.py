"""Structural inventory/validation of the 4 experimental-protocol datasets.

Read-only: no sampling, no training, no LLM/VLM calls, no extraction of any
large (>100MB) zip. Writes a raw JSON manifest per dataset under
experiments/data_audit/ and a consolidated Markdown report under
docs/reconstruction/02_dataset_inventory_report.md.

Usage:
    python scripts/run_dataset_inventory.py
    python scripts/run_dataset_inventory.py --config config/dataset_inventory.yaml
    python scripts/run_dataset_inventory.py --set ham10000.images_dir=D:/other/path
    python scripts/run_dataset_inventory.py --skip-ham10000-hash
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import adapters, report  # noqa: E402


def _apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply --set dataset.key=value overrides onto the loaded YAML config."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set espera dataset.key=value, recebido: {item!r}")
        dotted_key, value = item.split("=", 1)
        dataset, _, key = dotted_key.partition(".")
        if dataset not in config or key not in config[dataset]:
            raise ValueError(f"--set alvo desconhecido: {dotted_key!r}")
        config[dataset][key] = value
    return config


def _resolve_ph2_path(repo_relative: str) -> Path:
    p = Path(repo_relative)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_previous_ham10000_manifest(output_dir: Path) -> dict | None:
    """Load a previously written ham10000_manifest.json if it carries real hashes.

    Returns None (not an empty dict) when the file is missing, unreadable, or
    has empty image_manifest/mask_manifest — callers must not mistake "no
    reusable previous hashes" for "reuse an empty manifest".
    """
    manifest_path = output_dir / "ham10000_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as f:
            previous = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if previous.get("image_manifest") and previous.get("mask_manifest"):
        return previous
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "config" / "dataset_inventory.yaml"),
        help="Caminho do YAML de configuração do inventário.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments" / "data_audit"),
        help="Diretório onde os manifestos JSON brutos serão gravados.",
    )
    parser.add_argument(
        "--report-path",
        default=str(REPO_ROOT / "docs" / "reconstruction" / "02_dataset_inventory_report.md"),
        help="Caminho do relatório Markdown consolidado.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dataset.key=value",
        help="Sobrescreve um caminho do YAML, ex.: --set ham10000.images_dir=D:/x",
    )
    parser.add_argument(
        "--skip-ham10000-hash",
        action="store_true",
        help="Pula o cálculo de sha256 das imagens/máscaras do HAM10000 (mais rápido, menos completo).",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = _apply_overrides(config, args.overrides)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}

    print("=== PH2 ===")
    summaries["ph2"] = adapters.audit_ph2(
        root_dir=_resolve_ph2_path(config["ph2"]["root"]),
        metadata_txt=_resolve_ph2_path(config["ph2"]["metadata_txt"]),
    )
    print(f"status={summaries['ph2']['status']}")

    print("=== HAM10000 (hashing pode levar minutos) ===")
    previous_ham10000_manifest = None
    if args.skip_ham10000_hash:
        previous_ham10000_manifest = _load_previous_ham10000_manifest(output_dir)
        if previous_ham10000_manifest is not None:
            print(
                "--skip-ham10000-hash: manifesto anterior com hashes reais encontrado "
                f"em {output_dir / 'ham10000_manifest.json'} "
                f"({len(previous_ham10000_manifest['image_manifest'])} imagens, "
                f"{len(previous_ham10000_manifest['mask_manifest'])} máscaras) — "
                "reaproveitando em vez de apagar."
            )
        else:
            print(
                "--skip-ham10000-hash: nenhum manifesto anterior com hashes reais em "
                f"{output_dir / 'ham10000_manifest.json'} — image_manifest/mask_manifest "
                "ficarão vazios nesta rodada (nada para reaproveitar, não é perda de dados)."
            )
    summaries["ham10000"] = adapters.audit_ham10000(
        images_dir=config["ham10000"]["images_dir"],
        masks_dir=config["ham10000"]["masks_dir"],
        metadata_csv=config["ham10000"]["metadata_csv"],
        hash_files=not args.skip_ham10000_hash,
        previous_manifest=previous_ham10000_manifest,
    )
    print(f"status={summaries['ham10000']['status']}")

    print("=== ISIC2018 Task1 ===")
    summaries["isic2018_task1"] = adapters.audit_isic2018_task1(
        images_zip=config["isic2018_task1"]["images_zip"],
        masks_zip=config["isic2018_task1"]["masks_zip"],
    )
    print(f"status={summaries['isic2018_task1']['status']}")

    print("=== BCN20000 ===")
    summaries["bcn20000"] = adapters.audit_bcn20000(
        outer_zip=config["bcn20000"]["outer_zip"],
        train_csv_entry=config["bcn20000"]["train_csv_entry"],
        test_csv_entry=config["bcn20000"]["test_csv_entry"],
        train_zip_entry=config["bcn20000"]["train_zip_entry"],
        test_zip_entry=config["bcn20000"]["test_zip_entry"],
    )
    print(f"status={summaries['bcn20000']['status']}")

    for name, summary in summaries.items():
        manifest_path = output_dir / f"{name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"manifesto gravado: {manifest_path}")

    report_text = report.render_report(summaries)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"relatório gravado: {report_path}")


if __name__ == "__main__":
    main()
