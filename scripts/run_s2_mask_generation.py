"""Generates the predicted segmentation masks for condition S2 of the
multiagent experimental matrix (``docs/reconstruction/
15_multiagent_experimental_matrix.md``) -- one mask per image in the
FROZEN HAM10000 sample manifest (3,000 lesions, seed 20261106, SHA-256
registered in ``docs/reconstruction/10_frozen_sample_registry.md``),
using the final segmentation checkpoint + frozen threshold produced by
``scripts/run_segmentation_final_retrain.py``.

Hard requirements from the researcher, enforced here (not left implicit):
  - The manifest's SHA-256 is verified against the registered hash
    BEFORE anything else runs -- a mismatch is a hard, loud failure
    (``SystemExit``), never a silent continue with a different file.
  - Exactly one row is written per manifest case, including empty
    predicted masks and failures -- a failed case is NEVER dropped,
    NEVER retried, and NEVER substituted with the reference (Tschandl)
    mask. S2 is strictly the SEGMENTER'S OWN prediction; if the model
    fails on a case, that case's row records the failure, it does not
    silently fall back to ground truth.
  - The reference/ground-truth HAM10000 mask is never read anywhere in
    this script -- S2 must not be contaminated by the thing S1 already
    uses as the reference contour.

Inference pipeline per image (frozen by
``scripts/run_segmentation_final_retrain.py``, not decided here):
  resize to 256x256 (bilinear) -> model -> sigmoid -> restore to the
  image's OWN native resolution (bilinear) -> binarize at the frozen
  threshold. Restoration always happens before binarization, never after
  (same rule as ``src.segmentation.metrics.compute_image_metrics``).

STATUS: implementada, aguardando execucao do pesquisador (Mac/MPS). Nao
executado pelo assistente que escreveu este script.

Usage:
    python scripts/run_s2_mask_generation.py --device mps \\
        --ham10000-images-dir /Volumes/Andrey/Mestrado/dataverse_files/HAM10000
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.segmentation.isic2018_dataset import build_image_transform  # noqa: E402
from src.segmentation.metrics import restore_probability_to_original_resolution  # noqa: E402
from src.segmentation.model_factory import build_segmentation_model  # noqa: E402
from run_segmentation_arch_pilot import _resolve_device  # noqa: E402

ARCHITECTURE = "deeplabv3plus"
DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
REGISTERED_MANIFEST_SHA256 = "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_FROZEN_CONFIG = str(REPO_ROOT / "experiments" / "segmentation" / "final_model_frozen_config.json")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation" / "s2_masks")
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "experiments" / "segmentation")
EXPECTED_N_CASES = 3000


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_manifest_or_die(manifest_path: Path) -> None:
    if not manifest_path.exists():
        print(f"ERRO: manifesto nao encontrado em {manifest_path}", file=sys.stderr)
        raise SystemExit(1)
    actual = _sha256_of_file(manifest_path)
    if actual != REGISTERED_MANIFEST_SHA256:
        print(
            f"ERRO FATAL: SHA-256 do manifesto NAO corresponde ao registrado.\n"
            f"  arquivo: {manifest_path}\n"
            f"  esperado: {REGISTERED_MANIFEST_SHA256}\n"
            f"  encontrado: {actual}\n"
            "Esta execucao foi abortada -- NAO use um manifesto nao verificado para "
            "gerar mascaras de S2 (ver docs/reconstruction/10_frozen_sample_registry.md).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Manifesto verificado: SHA-256 corresponde ao registro congelado ({actual}).")


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_N_CASES:
        raise ValueError(
            f"Manifesto tem {len(rows)} linhas, esperado exatamente {EXPECTED_N_CASES} -- "
            "nao prosseguir sem entender a divergencia (o SHA-256 ja foi verificado, "
            "isso seria uma inconsistencia interna do arquivo)."
        )
    return rows


def _load_model(frozen_config: dict, device) -> torch.nn.Module:
    model = build_segmentation_model(frozen_config["architecture"], img_ch=3, output_ch=1)
    ckpt = torch.load(frozen_config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _predict_mask(model, image_path: Path, device, threshold: float, transform) -> dict:
    image = Image.open(image_path).convert("RGB")
    native_w, native_h = image.size  # PIL gives (W, H)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        prob_256 = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    prob_native = restore_probability_to_original_resolution(prob_256, (native_h, native_w))
    pred_binary = prob_native > threshold
    n_positive = int(pred_binary.sum())
    return {
        "pred_binary": pred_binary,
        "n_positive_pixels": n_positive,
        "n_total_pixels": int(pred_binary.size),
        "native_shape": [native_h, native_w],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--frozen-config", default=DEFAULT_FROZEN_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrao: auto-detecta cuda>mps>cpu)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.ham10000_images_dir)

    print("=== Geracao de mascaras S2 (segmentador final sobre o manifesto HAM10000 congelado) ===")
    _verify_manifest_or_die(manifest_path)
    rows = _load_manifest(manifest_path)
    print(f"{len(rows)} casos carregados do manifesto.")

    frozen_config_path = Path(args.frozen_config)
    if not frozen_config_path.exists():
        raise FileNotFoundError(
            f"{frozen_config_path} nao encontrado -- rode "
            "scripts/run_segmentation_final_retrain.py primeiro."
        )
    with open(frozen_config_path, encoding="utf-8") as f:
        frozen_config = json.load(f)
    if frozen_config["architecture"] != ARCHITECTURE:
        raise ValueError(f"{frozen_config_path} e de '{frozen_config['architecture']}', esperado '{ARCHITECTURE}'.")
    threshold = frozen_config["threshold"]
    print(f"Checkpoint: {frozen_config['checkpoint_path']} (SHA-256 {frozen_config['checkpoint_sha256']})")
    print(f"Limiar congelado: {threshold}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    device = _resolve_device(args.device)
    print(f"Dispositivo: {device}")

    model = _load_model(frozen_config, device)
    transform = build_image_transform()

    case_rows: list[dict] = []
    t0 = time.time()
    for i, row in enumerate(rows):
        image_id = row["image_id"]
        lesion_id = row["lesion_id"]
        image_path = images_dir / f"{image_id}.jpg"
        case = {
            "lesion_id": lesion_id,
            "image_id": image_id,
            "dx": row.get("dx", ""),
            "binary_target": row.get("binary_target", ""),
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"imagem nao encontrada: {image_path}")
            pred = _predict_mask(model, image_path, device, threshold, transform)
            mask_path = output_dir / f"{image_id}_s2_mask.png"
            mask_uint8 = (pred["pred_binary"].astype(np.uint8)) * 255
            Image.fromarray(mask_uint8, mode="L").save(mask_path)
            case.update({
                "status": "ok",
                "mask_path": str(mask_path),
                "mask_empty": pred["n_positive_pixels"] == 0,
                "n_positive_pixels": pred["n_positive_pixels"],
                "n_total_pixels": pred["n_total_pixels"],
                "native_shape": pred["native_shape"],
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001 -- registered per-case, never propagated/dropped
            case.update({
                "status": "falha",
                "mask_path": "",
                "mask_empty": None,
                "n_positive_pixels": None,
                "n_total_pixels": None,
                "native_shape": None,
                "error": str(exc),
            })
        case_rows.append(case)
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"  [{i + 1}/{len(rows)}] processados...")

    elapsed_s = time.time() - t0

    if len(case_rows) != EXPECTED_N_CASES:
        raise RuntimeError(
            f"Gerou {len(case_rows)} linhas, esperado {EXPECTED_N_CASES} -- "
            "nenhuma linha deve ser perdida; isso e um defeito, nao um aviso."
        )

    n_ok = sum(1 for c in case_rows if c["status"] == "ok")
    n_empty = sum(1 for c in case_rows if c["status"] == "ok" and c["mask_empty"])
    n_failure = sum(1 for c in case_rows if c["status"] == "falha")

    csv_path = results_dir / "s2_mask_generation_manifest.csv"
    fieldnames = [
        "lesion_id", "image_id", "dx", "binary_target", "status", "mask_path",
        "mask_empty", "n_positive_pixels", "n_total_pixels", "native_shape", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in case_rows:
            row_out = dict(c)
            row_out["native_shape"] = json.dumps(row_out["native_shape"]) if row_out["native_shape"] else ""
            writer.writerow(row_out)

    results_json_path = results_dir / "s2_mask_generation_results.json"
    summary = {
        "manifest_path": str(manifest_path),
        "manifest_sha256_verified": REGISTERED_MANIFEST_SHA256,
        "frozen_config_path": str(frozen_config_path),
        "checkpoint_sha256": frozen_config["checkpoint_sha256"],
        "threshold": threshold,
        "device": str(device),
        "n_cases_total": len(case_rows),
        "n_ok": n_ok,
        "n_mask_empty": n_empty,
        "n_failure": n_failure,
        "elapsed_s": elapsed_s,
        "reference_mask_used_as_fallback": False,
        "cases_dropped_or_substituted": 0,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": case_rows}, f, ensure_ascii=False, indent=2, default=str)

    report_path = results_dir / "s2_mask_generation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# Geracao de mascaras S2 -- resumo\n\n"
            f"- Casos totais: {summary['n_cases_total']} (esperado {EXPECTED_N_CASES})\n"
            f"- OK: {n_ok}\n"
            f"- Mascara vazia (OK, mas sem pixel positivo): {n_empty}\n"
            f"- Falha: {n_failure}\n"
            f"- Limiar: {threshold}\n"
            f"- Checkpoint SHA-256: {frozen_config['checkpoint_sha256']}\n"
            f"- Mascara de referencia usada como fallback: Nao (nunca)\n"
            f"- Casos descartados ou substituidos: 0\n"
            f"- Tempo total: {elapsed_s:.1f}s\n"
        )

    print(f"\nmanifesto de saida (csv): {csv_path}")
    print(f"resultados (json): {results_json_path}")
    print(f"relatorio (md): {report_path}")
    print(f"mascaras: {output_dir}")
    print(f"\nOK={n_ok} vazias={n_empty} falhas={n_failure} de {len(case_rows)} casos.")
    if n_failure > 0:
        print(f"AVISO: {n_failure} caso(s) falharam -- revise o campo 'error' no CSV antes de usar S2.")


if __name__ == "__main__":
    main()
