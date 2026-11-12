"""Primeira entrega da Etapa 5: prepara as tres condicoes de apresentacao
de imagem (S0/S1/S2) sobre as 3.000 lesoes do manifesto HAM10000
congelado (`docs/reconstruction/10_frozen_sample_registry.md`), SEM
nenhuma chamada a VLM ou a qualquer outro modelo.

Para cada caso do manifesto:
  - S0 = imagem original HAM10000, sem nenhuma alteracao alem de
    abrir/resalvar no formato de saida.
  - S1 = imagem original + contorno da mascara de referencia
    (Tschandl/HAM10000).
  - S2 = imagem original + contorno da mascara predita pelo segmentador
    final (DeepLabV3+-ResNet34, checkpoint SHA-256
    d15b712895fafac665329d138b03e9de6bd8837676bcc8f4c675042c5d206f31,
    limiar 0.5), gerada por `scripts/run_s2_mask_generation.py`.

S1 e S2 usam a MESMA funcao de contorno
(`src.multiagent.presentation.draw_contour`), com os mesmos parametros
de cor/espessura/retrieval -- a unica diferenca entre as duas condicoes
e qual mascara entra.

Casos em que S2 fica estruturalmente indisponivel (mascara predita vazia,
mascara "ok" no JSON mas ausente/ilegivel localmente, falha registrada no
JSON, ou shape incompativel com a imagem) NAO geram arquivo de S2 e NAO
caem de volta para S0/S1 -- essa indisponibilidade e registrada
explicitamente por caso, nunca omitida.

Nenhuma linha do manifesto e descartada ou substituida em nenhum dos
relatorios produzidos aqui.

STATUS: implementada, aguardando execucao do pesquisador. Este script
NAO foi executado pelo assistente que o escreveu (`feedback_
no_autonomous_execution`) -- apenas revisado estaticamente.

Uso (todos os defaults ja apontam para os caminhos reais do projeto):
    python scripts/run_s0_s1_s2_preparation.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.multiagent.presentation import (  # noqa: E402
    EXPECTED_N_CASES,
    build_pairing_report,
    build_vlm_payload_stub,
    draw_contour,
    index_s2_cases_by_image_id,
    load_binary_mask,
    load_manifest_rows,
    load_original_image,
    resolve_s2_mask_path_locally,
    verify_manifest_or_die,
    verify_payload_has_no_labels,
)

DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_HAM10000_REF_MASKS_DIR = (
    "D:/Andrey/Mestrado/dataverse_files/"
    "HAM10000_segmentations_lesion_tschandl/HAM10000_segmentations_lesion_tschandl"
)
DEFAULT_S2_MASKS_DIR = str(REPO_ROOT / "experiments" / "segmentation" / "s2_masks")
DEFAULT_S2_RESULTS_JSON = str(REPO_ROOT / "experiments" / "segmentation" / "s2_mask_generation_results.json")
DEFAULT_OUTPUT_IMAGES_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "s0_s1_s2_images")
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "experiments" / "multiagent")
N_EXAMPLE_LESIONS = 5


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

    print("=== Preparacao S0/S1/S2 (Etapa 5, primeira entrega -- sem chamada a VLM) ===")
    verify_manifest_or_die(manifest_path)
    print("Manifesto verificado: SHA-256 corresponde ao registro congelado.")

    rows = load_manifest_rows(manifest_path)
    if len(rows) != EXPECTED_N_CASES:
        raise RuntimeError(
            f"Manifesto tem {len(rows)} linhas, esperado {EXPECTED_N_CASES} -- "
            "SHA-256 ja bateu, isso seria uma inconsistencia interna do arquivo."
        )
    print(f"{len(rows)} casos carregados do manifesto.")

    if not s2_results_json_path.exists():
        raise FileNotFoundError(
            f"{s2_results_json_path} nao encontrado -- copie o resultado real de "
            "scripts/run_s2_mask_generation.py antes de rodar esta preparacao."
        )
    with open(s2_results_json_path, encoding="utf-8") as f:
        s2_results_json = json.load(f)
    s2_cases_by_image_id = index_s2_cases_by_image_id(s2_results_json)
    print(f"{len(s2_cases_by_image_id)} entradas carregadas do JSON de geracao de S2.")

    pairings, extras = build_pairing_report(
        rows, s2_cases_by_image_id, images_dir, ref_masks_dir, s2_masks_dir
    )

    # --- geracao das imagens das 3 condicoes para TODOS os 3.000 casos ---
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
            "duplicate_lesion_id": pairing.duplicate_lesion_id,
            "duplicate_image_id": pairing.duplicate_image_id,
            "status_s0": "indisponivel",
            "reason_s0": pairing.reason_s0_unavailable,
            "path_s0": "",
            "status_s1": "indisponivel",
            "reason_s1": pairing.reason_s1_unavailable,
            "path_s1": "",
            "status_s2": "indisponivel",
            "reason_s2": pairing.reason_s2_unavailable,
            "path_s2": "",
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

    # --- relatorio CSV por caso ---
    manifest_csv_path = results_dir / "s0_s1_s2_preparation_manifest.csv"
    fieldnames = list(output_rows[0].keys())
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    # --- contagens por condicao ---
    def _count(status_key: str) -> dict[str, int]:
        avail = sum(1 for r in output_rows if r[status_key] == "disponivel")
        return {"disponivel": avail, "indisponivel": len(output_rows) - avail}

    counts = {"s0": _count("status_s0"), "s1": _count("status_s1"), "s2": _count("status_s2")}
    reason_counts_s1: dict[str, int] = {}
    reason_counts_s2: dict[str, int] = {}
    for r in output_rows:
        if r["status_s1"] == "indisponivel" and r["reason_s1"]:
            reason_counts_s1[r["reason_s1"]] = reason_counts_s1.get(r["reason_s1"], 0) + 1
        if r["status_s2"] == "indisponivel" and r["reason_s2"]:
            reason_counts_s2[r["reason_s2"]] = reason_counts_s2.get(r["reason_s2"], 0) + 1

    summary = {
        "n_cases_total": len(output_rows),
        "manifest_sha256_verified": True,
        "counts_by_condition": counts,
        "reason_counts_s1_unavailable": reason_counts_s1,
        "reason_counts_s2_unavailable": reason_counts_s2,
        "duplicate_lesion_ids": extras["duplicate_lesion_ids"],
        "duplicate_image_ids": extras["duplicate_image_ids"],
        "image_ids_without_s2_entry": extras["image_ids_without_s2_entry"],
        "s2_entries_without_manifest_row": extras["s2_entries_without_manifest_row"],
    }
    summary_json_path = results_dir / "s0_s1_s2_preparation_results.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Contagens por condicao:")
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"Relatorio por caso: {manifest_csv_path}")
    print(f"Resumo: {summary_json_path}")

    # --- exemplos lado a lado para inspecao visual ---
    examples_dir = output_images_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    example_rows = [r for r in output_rows if r["status_s0"] == "disponivel"][:N_EXAMPLE_LESIONS - 1]
    empty_mask_example = next(
        (r for r in output_rows if r["status_s0"] == "disponivel" and r["reason_s2"] == "mascara_predita_vazia"),
        None,
    )
    if empty_mask_example is not None and empty_mask_example not in example_rows:
        example_rows.append(empty_mask_example)

    for r in example_rows:
        panels = []
        for key, label in (("path_s0", "S0"), ("path_s1", "S1"), ("path_s2", "S2")):
            if r[key]:
                panels.append((PILImage.open(r[key]), label))
        if not panels:
            continue
        pad = 8
        label_h = 24
        max_h = max(p.height for p, _ in panels)
        total_w = sum(p.width for p, _ in panels) + pad * (len(panels) + 1)
        composite = PILImage.new("RGB", (total_w, max_h + label_h + 2 * pad), "white")
        draw = ImageDraw.Draw(composite)
        x = pad
        for panel, label in panels:
            composite.paste(panel, (x, label_h + pad))
            draw.text((x, 4), f"{label} -- {r['image_id']}", fill="black")
            x += panel.width + pad
        composite.save(examples_dir / f"{r['image_id']}_s0_s1_s2_example.png")

    print(f"Exemplos lado a lado salvos em: {examples_dir}")

    # --- stub de payload dublê (apenas demonstracao, nenhuma chamada real) ---
    demo_row = next((r for r in output_rows if r["status_s2"] == "disponivel"), None)
    if demo_row is not None:
        payload = build_vlm_payload_stub(Path(demo_row["path_s2"]), condition="S2")
        assert verify_payload_has_no_labels(payload), "payload dublê vazou rotulo -- defeito, nao prosseguir"
        print("Payload dublê de exemplo (sem rotulos):")
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
