"""Evaluate the fixed BCN20000 EfficientNetV2-S baseline (stage 3) on
condition S0 of the multiagent experimental matrix: one score + decision
per case in the FROZEN HAM10000 sample manifest (3,000 lesions, seed
20261106, SHA-256 registered in
``docs/reconstruction/10_frozen_sample_registry.md``), using the
ORIGINAL (unsegmented) HAM10000 images.

Hard requirements, enforced here (not left implicit):
  - The manifest's SHA-256 is verified against the registered hash
    BEFORE anything else runs -- a mismatch is a hard, loud failure
    (``SystemExit``), never a silent continue with a different file.
  - The decision threshold is READ from
    ``experiments/cnn_baseline/training/threshold_selection.json``
    (fixed on the BCN20000 internal validation split, by Youden index)
    -- it is NEVER recomputed, re-tuned, or touched using any HAM10000
    score. This script only applies an already-fixed threshold.
  - Exactly one row is written per manifest case, including failures
    (e.g. missing image file) -- a failed case is NEVER dropped, NEVER
    retried, and NEVER assigned a fabricated score/decision.
  - The true label (``dx``/``binary_target``) is recorded alongside the
    prediction for the later paired comparison, but is NEVER used here
    to pick or adjust the threshold.

STATUS: implementada, aguardando execucao do pesquisador (Mac/MPS, onde
``best_checkpoint.pt`` e as imagens originais do HAM10000 estao
disponiveis). Nao executado pelo assistente que escreveu este script.

Usage:
    python scripts/run_cnn_s0_ham10000_eval.py --device mps \\
        --checkpoint-path experiments/cnn_baseline/training/best_checkpoint.pt \\
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cnn_baseline.bcn20000_dataset import BINARY_TARGET_TO_LABEL  # noqa: E402
from src.cnn_baseline.efficientnet_training import build_model  # noqa: E402

DEFAULT_SAMPLE_MANIFEST = str(REPO_ROOT / "experiments" / "sampling" / "sample_manifest.csv")
REGISTERED_SAMPLE_MANIFEST_SHA256 = (
    "32bbd1b318dccca048abeb25de64e6b1f2db6184d589286606a73271d6392799"
)
DEFAULT_THRESHOLD_SELECTION = str(
    REPO_ROOT / "experiments" / "cnn_baseline" / "training" / "threshold_selection.json"
)
DEFAULT_CHECKPOINT_PATH = str(
    REPO_ROOT / "experiments" / "cnn_baseline" / "training" / "best_checkpoint.pt"
)
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "cnn_baseline" / "s0_ham10000_eval")
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
    if actual != REGISTERED_SAMPLE_MANIFEST_SHA256:
        print(
            f"ERRO FATAL: SHA-256 do manifesto NAO corresponde ao registrado.\n"
            f"  arquivo: {manifest_path}\n"
            f"  esperado: {REGISTERED_SAMPLE_MANIFEST_SHA256}\n"
            f"  encontrado: {actual}\n"
            "Esta execucao foi abortada -- NAO use um manifesto nao verificado para "
            "avaliar S0 (ver docs/reconstruction/10_frozen_sample_registry.md).",
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


def _load_threshold(threshold_selection_path: Path) -> dict:
    if not threshold_selection_path.exists():
        raise FileNotFoundError(
            f"{threshold_selection_path} nao encontrado -- rode "
            "scripts/select_bcn20000_threshold.py primeiro (o limiar e fixado "
            "SOMENTE na validacao interna do BCN20000, nunca no HAM10000)."
        )
    with open(threshold_selection_path, encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(name: str):
    import torch

    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-manifest", default=DEFAULT_SAMPLE_MANIFEST)
    parser.add_argument("--threshold-selection", default=DEFAULT_THRESHOLD_SELECTION)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--checkpoint-sha256", default=None, help="Registrado no relatorio para auditoria; nao verificado contra um valor fixo (ainda nao congelado em doc).")
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrao: auto-detecta cuda>mps>cpu)")
    args = parser.parse_args()

    import torch
    from torchvision.models import EfficientNet_V2_S_Weights

    manifest_path = Path(args.sample_manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.ham10000_images_dir)

    print("=== Avaliacao S0 (CNN BCN20000 -> HAM10000 congelado, imagens originais) ===")
    _verify_manifest_or_die(manifest_path)
    rows = _load_manifest(manifest_path)
    print(f"{len(rows)} casos carregados do manifesto.")

    threshold_selection = _load_threshold(Path(args.threshold_selection))
    threshold = threshold_selection["threshold"]
    print(f"Limiar fixado na validacao interna do BCN20000: {threshold} (Youden J={threshold_selection['youden_j']})")

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} nao encontrado -- aponte --checkpoint-path para onde "
            "best_checkpoint.pt esta nesta maquina."
        )
    checkpoint_sha256 = args.checkpoint_sha256 or _sha256_of_file(checkpoint_path)
    print(f"Checkpoint: {checkpoint_path} (SHA-256 {checkpoint_sha256})")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    device = _resolve_device(args.device)
    print(f"Dispositivo: {device}")

    model, _weights = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()

    transform = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()

    from PIL import Image

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
            "true_binary_target": row.get("binary_target", ""),
        }
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"imagem nao encontrada: {image_path}")
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(image_tensor).squeeze()
                score = float(torch.sigmoid(logit).cpu().item())
            decision_label = 1 if score >= threshold else 0
            case.update({
                "status": "ok",
                "predicted_score": score,
                "predicted_binary_target": "melanoma" if decision_label == 1 else "nao_melanoma",
                "error": "",
            })
        except Exception as exc:  # noqa: BLE001 -- registered per-case, never propagated/dropped
            case.update({
                "status": "falha",
                "predicted_score": None,
                "predicted_binary_target": "",
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
    n_failure = sum(1 for c in case_rows if c["status"] == "falha")

    csv_path = output_dir / "s0_cnn_scores.csv"
    fieldnames = [
        "lesion_id", "image_id", "dx", "true_binary_target", "status",
        "predicted_score", "predicted_binary_target", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    results_json_path = output_dir / "s0_cnn_eval_results.json"
    summary = {
        "sample_manifest_path": str(manifest_path),
        "sample_manifest_sha256_verified": REGISTERED_SAMPLE_MANIFEST_SHA256,
        "threshold_selection_path": str(Path(args.threshold_selection).resolve()),
        "threshold": threshold,
        "threshold_source": "val_scores.csv (validacao interna BCN20000), indice de Youden",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "device": str(device),
        "n_cases_total": len(case_rows),
        "n_ok": n_ok,
        "n_failure": n_failure,
        "elapsed_s": elapsed_s,
        "ham10000_used_to_tune_threshold": False,
        "cases_dropped_or_substituted": 0,
    }
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cases": case_rows}, f, ensure_ascii=False, indent=2, default=str)

    report_path = output_dir / "s0_cnn_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            "# Avaliacao S0 -- CNN BCN20000 sobre HAM10000 congelado (imagens originais)\n\n"
            f"- Casos totais: {summary['n_cases_total']} (esperado {EXPECTED_N_CASES})\n"
            f"- OK: {n_ok}\n"
            f"- Falha: {n_failure}\n"
            f"- Limiar (fixado no BCN20000, Youden): {threshold}\n"
            f"- Checkpoint SHA-256: {checkpoint_sha256}\n"
            f"- HAM10000 usado para ajustar o limiar: Nao (nunca)\n"
            f"- Casos descartados ou substituidos: 0\n"
            f"- Tempo total: {elapsed_s:.1f}s\n"
        )

    print(f"\nescores e decisoes (csv): {csv_path}")
    print(f"resultados (json): {results_json_path}")
    print(f"relatorio (md): {report_path}")
    print(f"\nOK={n_ok} falhas={n_failure} de {len(case_rows)} casos.")
    if n_failure > 0:
        print(f"AVISO: {n_failure} caso(s) falharam -- revise o campo 'error' no CSV antes de usar S0.")


if __name__ == "__main__":
    main()
