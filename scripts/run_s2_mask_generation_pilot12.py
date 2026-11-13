"""Etapa 5, segunda entrega — gera máscaras S2 (segmentador final congelado,
DeepLabV3+-ResNet34, checkpoint SHA-256
`d15b712895fafac665329d138b03e9de6bd8837676bcc8f4c675042c5d206f31`, limiar
0.5) para o manifesto de 12 lesões do PILOTO LOCAL DE VLM
(`experiments/multiagent/pilot12_manifest.csv`, produzido por
`scripts/run_pilot12_sample.py`), NÃO o manifesto congelado de 3.000.

Variante fina de `scripts/run_s2_mask_generation.py`: mesmo pipeline de
inferência (resize 256x256 -> modelo -> sigmoid -> restaurar à resolução
nativa -> binarizar no limiar congelado), mesmo checkpoint/config
congelados, única diferença real é o manifesto de entrada (parametrizado
via `--manifest-path`, não hardcoded) e os diretórios de saída, que são
SEPARADOS dos da amostra de 3.000:

  - máscaras: `experiments/multiagent/pilot12_s2_masks/`
  - manifesto/relatório/resultados: `experiments/multiagent/
    pilot12_s2_mask_generation_*`

**As máscaras e os resultados de S2 da amostra de 3.000 lesões
(`experiments/segmentation/s2_masks/`,
`experiments/segmentation/s2_mask_generation_results.json`) NUNCA são
lidos nem usados por este script para escolher ou influenciar nada deste
piloto de 12 lesões -- esta é uma execução estruturalmente separada e
menor, usando apenas o MESMO modelo/limiar.**

STATUS: implementado, aguardando execução do pesquisador (Mac/MPS). Não
executado pelo assistente que o escreveu.

Uso:
    python scripts/run_s2_mask_generation_pilot12.py --device mps \\
        --ham10000-images-dir /Volumes/Andrey/Mestrado/dataverse_files/HAM10000
"""
from __future__ import annotations

import argparse
import csv
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
DEFAULT_MANIFEST_PATH = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_manifest.csv")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_FROZEN_CONFIG = str(REPO_ROOT / "experiments" / "segmentation" / "final_model_frozen_config.json")
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "multiagent" / "pilot12_s2_masks")
DEFAULT_RESULTS_DIR = str(REPO_ROOT / "experiments" / "multiagent")
EXPECTED_N_CASES = 12
RESULT_FILE_PREFIX = "pilot12_s2_mask_generation"


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise SystemExit(
            f"ERRO: manifesto do piloto de 12 lesões não encontrado em {manifest_path} -- "
            "rode scripts/run_pilot12_sample.py primeiro."
        )
    with open(manifest_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_N_CASES:
        raise ValueError(
            f"Manifesto do piloto tem {len(rows)} linhas, esperado exatamente "
            f"{EXPECTED_N_CASES} -- este script é só para o piloto de 12 lesões, "
            "não para a amostra congelada de 3.000 (use run_s2_mask_generation.py para essa)."
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
    native_w, native_h = image.size
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
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrão: auto-detecta cuda>mps>cpu)")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.ham10000_images_dir)

    print("=== Geração de máscaras S2 — PILOTO de 12 lesões (fora da amostra congelada de 3.000) ===")
    print(
        "AVISO: este script nunca lê experiments/segmentation/s2_masks/ nem "
        "s2_mask_generation_results.json (amostra de 3.000) -- execução estruturalmente separada."
    )
    rows = _load_manifest(manifest_path)
    print(f"{len(rows)} casos carregados do manifesto do piloto: {manifest_path}")

    frozen_config_path = Path(args.frozen_config)
    if not frozen_config_path.exists():
        raise FileNotFoundError(
            f"{frozen_config_path} não encontrado -- rode scripts/run_segmentation_final_retrain.py primeiro "
            "(mesmo checkpoint congelado usado pela amostra de 3.000, reaproveitado aqui, não retreinado)."
        )
    with open(frozen_config_path, encoding="utf-8") as f:
        frozen_config = json.load(f)
    if frozen_config["architecture"] != ARCHITECTURE:
        raise ValueError(f"{frozen_config_path} é de '{frozen_config['architecture']}', esperado '{ARCHITECTURE}'.")
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
                raise FileNotFoundError(f"imagem não encontrada: {image_path}")
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
        except Exception as exc:  # noqa: BLE001 -- registrado por caso, nunca propagado/descartado
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
        print(f"  [{i + 1}/{len(rows)}] {image_id}: {case['status']}")

    elapsed_s = time.time() - t0

    if len(case_rows) != EXPECTED_N_CASES:
        raise RuntimeError(
            f"Gerou {len(case_rows)} linhas, esperado {EXPECTED_N_CASES} -- defeito, não aviso."
        )

    n_ok = sum(1 for c in case_rows if c["status"] == "ok")
    n_empty = sum(1 for c in case_rows if c["status"] == "ok" and c["mask_empty"])
    n_failure = sum(1 for c in case_rows if c["status"] == "falha")

    csv_path = results_dir / f"{RESULT_FILE_PREFIX}_manifest.csv"
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

    results_json_path = results_dir / f"{RESULT_FILE_PREFIX}_results.json"
    summary = {
        "manifest_path": str(manifest_path),
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
        "frozen_3000_s2_masks_or_results_read": False,
        "note": (
            "Execução separada do pilot de 12 lesões; não lê nem usa máscaras/resultados "
            "de S2 da amostra congelada de 3.000 lesões em nenhum ponto."
        ),
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": case_rows}, f, ensure_ascii=False, indent=2, default=str)

    report_path = results_dir / f"{RESULT_FILE_PREFIX}_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# Geração de máscaras S2 — piloto de 12 lesões (fora da amostra congelada)\n\n"
            f"- Casos totais: {summary['n_cases_total']} (esperado {EXPECTED_N_CASES})\n"
            f"- OK: {n_ok}\n- Máscara vazia: {n_empty}\n- Falha: {n_failure}\n"
            f"- Limiar: {threshold}\n- Checkpoint SHA-256: {frozen_config['checkpoint_sha256']}\n"
            f"- Tempo total: {elapsed_s:.1f}s\n"
            "- Máscaras/resultados de S2 da amostra de 3.000 lidos nesta execução: Não (nunca)\n"
        )

    print(f"\nmanifesto (csv): {csv_path}")
    print(f"resultados (json): {results_json_path}")
    print(f"relatório (md): {report_path}")
    print(f"máscaras: {output_dir}")
    print(f"OK={n_ok} vazias={n_empty} falhas={n_failure} de {len(case_rows)} casos.")


if __name__ == "__main__":
    main()
