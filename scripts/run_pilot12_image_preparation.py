"""Etapa 5, segunda entrega — constrói as imagens de apresentação S0/S1/S2
para as 12 lesões do piloto local de VLM
(`experiments/multiagent/pilot12_manifest.csv`), reusando INTEGRALMENTE
as funções de `src/multiagent/presentation.py` (`draw_contour`,
`load_binary_mask`, `load_original_image`, `build_pairing_report`,
`index_s2_cases_by_image_id`, helpers de resolução de caminho) —
mesma lógica que `scripts/run_s0_s1_s2_preparation.py` usa para as 3.000
lesões congeladas, aqui parametrizada para um manifesto e diretórios de
saída SEPARADOS e MENORES.

Saída: `experiments/multiagent/pilot12_images/{s0,s1,s2}/` — NUNCA grava
nem lê `experiments/multiagent/s0_s1_s2_images/` (saída da amostra de
3.000) para não misturar as duas execuções.

S1 usa a máscara de referência REAL (Tschandl) de cada lesão, mesma
convenção de nome (`{image_id}_segmentation.png`) e mesmo diretório real
do HAM10000 usado pelas 3.000. S2 usa as máscaras preditas geradas por
`scripts/run_s2_mask_generation_pilot12.py` (deve ser executado antes).

STATUS: implementado, aguardando execução do pesquisador. Não executado
pelo assistente que o escreveu.

Uso:
    python scripts/run_pilot12_image_preparation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent.presentation import (  # noqa: E402
    build_pairing_report,
    draw_contour,
    index_s2_cases_by_image_id,
    load_binary_mask,
    load_manifest_rows,
    load_original_image,
    resolve_s2_mask_path_locally,
)

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_HAM10000_REF_MASKS_DIR = (
    "D:/Andrey/Mestrado/dataverse_files/"
    "HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl"
)
DEFAULT_S2_MASKS_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_s2_masks")
DEFAULT_S2_RESULTS_JSON = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_s2_mask_generation_results.json")
DEFAULT_OUTPUT_IMAGES_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_images")
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "experiments" / "multiagent")
EXPECTED_N_CASES = 12


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--ham10000-ref-masks-dir", default=DEFAULT_HAM10000_REF_MASKS_DIR)
    parser.add_argument("--s2-masks-dir", default=DEFAULT_S2_MASKS_DIR)
    parser.add_argument("--s2-results-json", default=DEFAULT_S2_RESULTS_JSON)
    parser.add_argument("--output-images-dir", default=DEFAULT_OUTPUT_IMAGES_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    return parser


def _save_condition_image(image, out_dir: Path, image_id: str, suffix: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}_{suffix}.png"
    image.save(out_path)
    return out_path


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    images_dir = Path(args.ham10000_images_dir)
    ref_masks_dir = Path(args.ham10000_ref_masks_dir)
    s2_masks_dir = Path(args.s2_masks_dir)
    s2_results_json_path = Path(args.s2_results_json)
    output_images_dir = Path(args.output_images_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Preparação S0/S1/S2 — PILOTO de 12 lesões ===")
    if not manifest_path.exists():
        raise SystemExit(f"ERRO: {manifest_path} não encontrado -- rode scripts/run_pilot12_sample.py primeiro.")
    rows = load_manifest_rows(manifest_path)
    if len(rows) != EXPECTED_N_CASES:
        raise RuntimeError(f"Manifesto do piloto tem {len(rows)} linhas, esperado {EXPECTED_N_CASES}.")
    print(f"{len(rows)} casos carregados do manifesto do piloto.")

    if not s2_results_json_path.exists():
        raise FileNotFoundError(
            f"{s2_results_json_path} não encontrado -- rode "
            "scripts/run_s2_mask_generation_pilot12.py primeiro."
        )
    with open(s2_results_json_path, encoding="utf-8") as f:
        s2_results_json = json.load(f)
    s2_cases_by_image_id = index_s2_cases_by_image_id(s2_results_json)
    print(f"{len(s2_cases_by_image_id)} entradas carregadas do JSON de geração de S2 do piloto.")

    pairings, extras = build_pairing_report(
        rows, s2_cases_by_image_id, images_dir, ref_masks_dir, s2_masks_dir
    )

    output_rows: list[dict] = []
    s0_dir = output_images_dir / "s0"
    s1_dir = output_images_dir / "s1"
    s2_dir = output_images_dir / "s2"

    for pairing in pairings:
        row_out = {
            "lesion_id": pairing.lesion_id,
            "image_id": pairing.image_id,
            "dx": pairing.dx,
            "binary_target": pairing.binary_target,
            "status_s0": "indisponivel", "reason_s0": pairing.reason_s0_unavailable, "path_s0": "",
            "status_s1": "indisponivel", "reason_s1": pairing.reason_s1_unavailable, "path_s1": "",
            "status_s2": "indisponivel", "reason_s2": pairing.reason_s2_unavailable, "path_s2": "",
        }

        image = None
        if pairing.original_image_available:
            image = load_original_image(pairing.original_image_path)
        if image is not None:
            path_s0 = _save_condition_image(image, s0_dir, pairing.image_id, "s0")
            row_out["status_s0"] = "disponivel"
            row_out["path_s0"] = str(path_s0)

            if pairing.reference_mask_available:
                ref_mask = load_binary_mask(pairing.reference_mask_path)
                s1_image = draw_contour(image, ref_mask)
                path_s1 = _save_condition_image(s1_image, s1_dir, pairing.image_id, "s1")
                row_out["status_s1"] = "disponivel"
                row_out["path_s1"] = str(path_s1)

            if pairing.s2_mask_available_locally:
                local_s2_path = resolve_s2_mask_path_locally(s2_masks_dir, pairing.image_id)
                s2_mask = load_binary_mask(local_s2_path)
                s2_image = draw_contour(image, s2_mask)
                path_s2 = _save_condition_image(s2_image, s2_dir, pairing.image_id, "s2")
                row_out["status_s2"] = "disponivel"
                row_out["path_s2"] = str(path_s2)

        output_rows.append(row_out)

    manifest_csv_path = results_dir / "pilot12_image_preparation_manifest.csv"
    fieldnames = list(output_rows[0].keys())
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    def _count(status_key: str) -> dict[str, int]:
        avail = sum(1 for r in output_rows if r[status_key] == "disponivel")
        return {"disponivel": avail, "indisponivel": len(output_rows) - avail}

    counts = {"s0": _count("status_s0"), "s1": _count("status_s1"), "s2": _count("status_s2")}
    summary = {
        "n_cases_total": len(output_rows),
        "counts_by_condition": counts,
        "duplicate_lesion_ids": extras["duplicate_lesion_ids"],
        "duplicate_image_ids": extras["duplicate_image_ids"],
        "image_ids_without_s2_entry": extras["image_ids_without_s2_entry"],
        "s2_entries_without_manifest_row": extras["s2_entries_without_manifest_row"],
    }
    summary_json_path = results_dir / "pilot12_image_preparation_results.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Contagens por condição:")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"Relatório por caso: {manifest_csv_path}")
    print(f"Resumo: {summary_json_path}")
    print(f"Imagens S0/S1/S2 em: {output_images_dir}")
    if any(counts[c]["indisponivel"] > 0 for c in counts):
        print(
            "AVISO: há condições indisponíveis para algum caso -- ver coluna "
            "reason_s0/s1/s2 no CSV antes de rodar o piloto de VLM."
        )


if __name__ == "__main__":
    main()
