"""BCN20000 train/validation split preparation for the CNN baseline
(EfficientNetV2-S, stage 3 of the reconstruction). Data preparation only --
no training, no threshold selection, no Grad-CAM, no HAM10000 evaluation.

Two modes:

  --mode diagnostico
      Inspects both official partitions (train/test): available images,
      usable labels, lesion_id coverage/conflicts, categories, a heuristic
      (unconfirmed) cross-partition lesion-id comparison, and reproduces
      the missing-image-vs-CSV check. Generates NO split. Does not require
      --partition-policy/--val-fraction/--seed.

  --mode geracao
      Generates the actual train/val split manifest. REQUIRES
      --partition-policy, --val-fraction, and --seed explicitly -- there
      are no defaults, so this cannot run with an unintended configuration.

STATUS: implementada, aguardando execução do pesquisador. Not run by the
assistant/fork that wrote it.

HAM10000 is never read by this script.

Usage:
    python scripts/run_bcn20000_split.py --mode diagnostico
    python scripts/run_bcn20000_split.py --mode geracao \
        --partition-policy apenas_train --val-fraction 0.2 --seed 20261107
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import bcn_materialize, core  # noqa: E402
from src.cnn_baseline import bcn20000_diagnostics as diag  # noqa: E402
from src.cnn_baseline import bcn20000_split as split_mod  # noqa: E402

DEFAULT_CONFIG = str(REPO_ROOT / "config" / "dataset_inventory.yaml")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "cnn_baseline")


def _load_bcn_config(config_path: str) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["bcn20000"]


def _materialize_and_list(bcn_cfg: dict[str, Any], zip_entry_key: str, extracted_dir: Path) -> tuple[Path, set[str]]:
    materialization = bcn_materialize.materialize_inner_zip(
        bcn_cfg["outer_zip"], bcn_cfg[zip_entry_key], extracted_dir
    )
    inner_zip_path = Path(materialization["dest_path"])
    filenames = diag.list_zip_image_filenames(inner_zip_path)
    return inner_zip_path, filenames


def _run_diagnostico(args: argparse.Namespace) -> None:
    bcn_cfg = _load_bcn_config(args.config)
    extracted_dir = Path(bcn_cfg["outer_zip"]).resolve().parent / "_extracted"

    print("=== lendo CSVs oficiais do BCN20000 ===")
    train_rows = diag.load_bcn20000_csv(bcn_cfg["outer_zip"], bcn_cfg["train_csv_entry"])
    test_rows = diag.load_bcn20000_csv(bcn_cfg["outer_zip"], bcn_cfg["test_csv_entry"])
    print(f"train: {len(train_rows)} linhas / test: {len(test_rows)} linhas")

    print("=== listando imagens materializadas (reaproveitando zips já extraídos) ===")
    _, train_filenames = _materialize_and_list(bcn_cfg, "train_zip_entry", extracted_dir)
    _, test_filenames = _materialize_and_list(bcn_cfg, "test_zip_entry", extracted_dir)
    print(f"train: {len(train_filenames)} imagens no zip / test: {len(test_filenames)} imagens no zip")

    print("=== diagnóstico por partição ===")
    train_diag = diag.diagnose_partition(train_rows, train_filenames, "train")
    test_diag = diag.diagnose_partition(test_rows, test_filenames, "test")
    print(
        f"train: {train_diag['n_usable_label_rows']} rótulos utilizáveis, "
        f"{train_diag['n_distinct_lesion_ids']} lesões distintas, "
        f"{train_diag['n_missing_from_zip']} imagens ausentes do zip"
    )
    print(
        f"test: {test_diag['n_usable_label_rows']} rótulos utilizáveis, "
        f"{test_diag['n_distinct_lesion_ids']} lesões distintas, "
        f"{test_diag['n_missing_from_zip']} imagens ausentes do zip "
        "(relatado anteriormente como 32 -- confira se este número bate)"
    )

    print("=== comparação heurística (NÃO confirmada) de lesion_id entre partições ===")
    train_lesion_ids = {r["lesion_id"] for r in train_rows if r.get("lesion_id")}
    test_lesion_ids = {r["lesion_id"] for r in test_rows if r.get("lesion_id")}
    overlap_heuristic = diag.compare_lesion_ids_heuristic(train_lesion_ids, test_lesion_ids)
    print(f"colisões normalizadas (heurística, não confirmada): {overlap_heuristic['n_normalized_collisions']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = diag.build_diagnostic_report_markdown(
        train_diag=train_diag, test_diag=test_diag, overlap_heuristic=overlap_heuristic
    )
    report_path = output_dir / "bcn20000_diagnostics_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"relatório gravado: {report_path}")

    results = {"train": train_diag, "test": test_diag, "overlap_heuristic": overlap_heuristic}
    results_path = output_dir / "bcn20000_diagnostics_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"resultados (json) gravados: {results_path}")
    print(
        "LEMBRETE: isto é diagnóstico. Nenhuma divisão foi gerada. A "
        "recomendação de política de partição ainda depende da sua leitura "
        "destes números -- não decidida automaticamente."
    )


def _run_geracao(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, val in (
            ("--partition-policy", args.partition_policy),
            ("--val-fraction", args.val_fraction),
            ("--seed", args.seed),
        )
        if val is None
    ]
    if missing:
        print(
            f"ERRO: --mode geracao exige {missing} explicitamente, sem "
            "default. Nada foi gerado."
        )
        sys.exit(1)

    bcn_cfg = _load_bcn_config(args.config)

    print("=== lendo CSVs oficiais do BCN20000 ===")
    train_rows = diag.load_bcn20000_csv(bcn_cfg["outer_zip"], bcn_cfg["train_csv_entry"])
    test_rows = diag.load_bcn20000_csv(bcn_cfg["outer_zip"], bcn_cfg["test_csv_entry"])

    print(f"=== aplicando política de partição: {args.partition_policy} ===")
    selected_rows = split_mod.select_partition_rows(train_rows, test_rows, args.partition_policy)
    print(f"linhas selecionadas: {len(selected_rows)}")

    print("=== construindo tabela de lesões elegíveis ===")
    table_result = split_mod.build_eligible_lesion_table(selected_rows)
    lesion_table = table_result["lesion_table"]
    print(
        f"lesões elegíveis: {len(lesion_table)} / registros excluídos: "
        f"{len(table_result['excluded_rows'])}"
    )

    print(f"=== dividindo (val_fraction={args.val_fraction}, seed={args.seed}) ===")
    split_result = split_mod.split_lesions_stratified(lesion_table, args.val_fraction, args.seed)

    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    summary = split_mod.summarize_split(manifest_rows)
    overlap_check = split_mod.verify_no_lesion_overlap(manifest_rows)
    print(f"sobreposição de lesão entre train/val: ok={overlap_check['ok']}")
    print(
        f"train: {summary['train']['n_images']} imagens / {summary['train']['n_lesions']} lesões -- "
        f"val: {summary['val']['n_images']} imagens / {summary['val']['n_lesions']} lesões"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "bcn20000_split_manifest.csv"
    _write_manifest_csv(manifest_path, manifest_rows)
    print(f"manifesto gravado: {manifest_path}")

    report_md = split_mod.build_split_report_markdown(
        partition_policy=args.partition_policy,
        val_fraction=args.val_fraction,
        seed=args.seed,
        excluded_rows_count=len(table_result["excluded_rows"]),
        split_result=split_result,
        summary=summary,
        overlap_check=overlap_check,
    )
    report_path = output_dir / "bcn20000_split_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"relatório gravado: {report_path}")

    results = {
        "partition_policy": args.partition_policy,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "excluded_rows": table_result["excluded_rows"],
        "split_result": split_result,
        "summary": summary,
        "overlap_check": overlap_check,
    }
    results_path = output_dir / "bcn20000_split_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"resultados (json) gravados: {results_path}")
    print(
        "LEMBRETE: isto é preparação de dados. Seleção de limiar (Youden) "
        "fica para a validação interna, depois. HAM10000 não foi tocado."
    )


def _write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = ["lesion_id", "bcn_filename", "source_partition", "diagnosis", "binary_target", "split"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["diagnostico", "geracao"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--partition-policy", choices=list(split_mod.PARTITION_POLICIES), default=None,
        help="Obrigatório em --mode geracao. Sem default.",
    )
    parser.add_argument("--val-fraction", type=float, default=None, help="Obrigatório em --mode geracao. Sem default.")
    parser.add_argument("--seed", type=int, default=None, help="Obrigatório em --mode geracao. Sem default.")
    args = parser.parse_args()

    if args.mode == "diagnostico":
        _run_diagnostico(args)
    else:
        _run_geracao(args)


if __name__ == "__main__":
    main()
