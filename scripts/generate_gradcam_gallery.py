"""Gera os mapas Grad-CAM (classe-alvo: melanoma, sempre) para as 40
lesões selecionadas por ``scripts/select_gradcam_sample.py``, usando a
CNN EfficientNetV2-S congelada (checkpoint verificado contra o SHA-256
registrado) e as imagens ORIGINAIS S0, com o mesmo pré-processamento da
avaliação final (``EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()``,
igual a ``scripts/run_cnn_s0_ham10000_eval.py``).

NÃO altera pesos, limiar ou pré-processamento. NÃO ajusta nada com base
no mapa. Qualitativo -- não comprova raciocínio clínico nem causalidade;
mapas vazios/constantes/com falha aparecem com status explícito, nunca
substituídos silenciosamente.

Usage:
    python scripts/generate_gradcam_gallery.py \\
        --checkpoint-path experiments/cnn_baseline/training/best_checkpoint.pt \\
        --ham10000-images-dir D:/Andrey/Mestrado/dataverse_files/HAM10000 \\
        --device mps
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cnn_baseline.efficientnet_training import build_model  # noqa: E402
from src.cnn_baseline.gradcam import (  # noqa: E402
    NORMALIZATION_METHOD,
    STATUS_FAILURE,
    STATUS_MAP_CONSTANT_NONZERO,
    STATUS_MAP_EMPTY_ALL_ZERO,
    STATUS_OK,
    TARGET_LAYER_ATTR_PATH,
    compute_gradcam,
    overlay_heatmap_on_image,
)

SAMPLE_MANIFEST = REPO_ROOT / "experiments" / "evaluation" / "gradcam" / "gradcam_sample_manifest.csv"
REGISTERED_CHECKPOINT_SHA256 = "42c385a36845bdfeb33c308862973dc55b1d92c0a1779bdf3193ebd9e6ede0c3"
DEFAULT_CHECKPOINT_PATH = str(REPO_ROOT / "experiments" / "cnn_baseline" / "training" / "best_checkpoint.pt")
DEFAULT_HAM10000_IMAGES_DIR = "D:/Andrey/Mestrado/dataverse_files/HAM10000"
OUTPUT_DIR = REPO_ROOT / "experiments" / "evaluation" / "gradcam"
IMAGES_OUTPUT_DIR = OUTPUT_DIR / "images"
GALLERY_HTML = OUTPUT_DIR / "gradcam_gallery.html"
MANIFEST_OUTPUT = OUTPUT_DIR / "gradcam_execution_log.csv"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checkpoint_or_die(checkpoint_path: Path) -> str:
    if not checkpoint_path.exists():
        print(f"ERRO: checkpoint não encontrado em {checkpoint_path}", file=sys.stderr)
        raise SystemExit(1)
    actual = _sha256_of_file(checkpoint_path)
    if actual != REGISTERED_CHECKPOINT_SHA256:
        print(
            f"ERRO FATAL: SHA-256 do checkpoint não corresponde ao registrado.\n"
            f"  arquivo: {checkpoint_path}\n  esperado: {REGISTERED_CHECKPOINT_SHA256}\n  encontrado: {actual}\n"
            "Esta execução foi abortada -- não gere Grad-CAM a partir de um checkpoint não verificado "
            "(ver docs/reconstruction/29_cnn_stage3_full_closure_s0_eval.md).",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"Checkpoint verificado: SHA-256 corresponde ao registrado ({actual}).")
    return actual


def _denormalize_to_uint8_rgb(tensor_chw, mean, std):
    import numpy as np

    arr = tensor_chw.detach().cpu().numpy()
    mean_arr = np.array(mean).reshape(3, 1, 1)
    std_arr = np.array(std).reshape(3, 1, 1)
    arr = arr * std_arr + mean_arr
    arr = (arr.transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--ham10000-images-dir", default=DEFAULT_HAM10000_IMAGES_DIR)
    parser.add_argument("--sample-manifest", default=str(SAMPLE_MANIFEST))
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrão: auto-detecta cuda>mps>cpu)")
    args = parser.parse_args()

    import torch
    from PIL import Image
    from torchvision.models import EfficientNet_V2_S_Weights

    sample_manifest_path = Path(args.sample_manifest)
    if not sample_manifest_path.exists():
        raise FileNotFoundError(f"{sample_manifest_path} não encontrado -- rode scripts/select_gradcam_sample.py primeiro.")
    with open(sample_manifest_path, encoding="utf-8", newline="") as f:
        sample_rows = list(csv.DictReader(f))
    print(f"{len(sample_rows)} lesões selecionadas carregadas de {sample_manifest_path}.")
    if len(sample_rows) > 40:
        print(f"AVISO: manifesto tem {len(sample_rows)} linhas, mais que o lote inicial de 40 -- "
              "nenhuma amostra foi ampliada automaticamente por este script; confira a origem do manifesto.")

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_sha256 = _verify_checkpoint_or_die(checkpoint_path)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(args.device)
    print(f"Dispositivo: {device}")

    model, _weights = build_model(pretrained=False)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()  # BatchNorm/Dropout em modo de avaliacao -- gradientes seguem habilitados via compute_gradcam

    transform = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    mean, std = transform.mean, transform.std

    IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.ham10000_images_dir)

    log_rows: list[dict] = []
    t_total0 = time.time()
    for i, row in enumerate(sample_rows):
        lesion_id, image_id, group = row["lesion_id"], row["image_id"], row["group"]
        image_path = images_dir / f"{image_id}.jpg"
        log_row = {
            "lesion_id": lesion_id, "image_id": image_id, "group": group,
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha256,
            "target_layer": TARGET_LAYER_ATTR_PATH, "target_class": "melanoma",
            "normalization_method": NORMALIZATION_METHOD, "device": str(device),
            "raw_logit": None, "sigmoid_score": None, "status": None, "error": None, "elapsed_s": None,
            "original_image_path": str(image_path), "overlay_image_path": "", "cam_npy_path": "",
        }
        t0 = time.time()
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"imagem não encontrada: {image_path}")
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            target_hw = (image_tensor.shape[2], image_tensor.shape[3])

            result = compute_gradcam(model, image_tensor, target_image_size=target_hw)
            log_row["raw_logit"] = result.raw_logit
            log_row["sigmoid_score"] = result.sigmoid_score
            log_row["status"] = result.status
            log_row["error"] = result.error

            base_name = f"{group}_{lesion_id}_{image_id}"
            if result.status == STATUS_OK:
                import numpy as np

                preprocessed_rgb = _denormalize_to_uint8_rgb(image_tensor[0], mean, std)
                overlay_rgb = overlay_heatmap_on_image(preprocessed_rgb, result.cam_normalized_upsampled)

                cam_npy_path = IMAGES_OUTPUT_DIR / f"{base_name}_cam.npy"
                np.save(cam_npy_path, result.cam_normalized_upsampled)
                Image.fromarray(preprocessed_rgb).save(IMAGES_OUTPUT_DIR / f"{base_name}_input.png")
                Image.fromarray(overlay_rgb).save(IMAGES_OUTPUT_DIR / f"{base_name}_overlay.png")
                image.save(IMAGES_OUTPUT_DIR / f"{base_name}_original.png")

                log_row["cam_npy_path"] = str(cam_npy_path)
                log_row["overlay_image_path"] = str(IMAGES_OUTPUT_DIR / f"{base_name}_overlay.png")
            elif result.status in (STATUS_MAP_EMPTY_ALL_ZERO, STATUS_MAP_CONSTANT_NONZERO):
                image.save(IMAGES_OUTPUT_DIR / f"{base_name}_original.png")
                print(f"  [{lesion_id}] status={result.status} (sem sobreposição -- mapa degenerado, nunca substituído)")
            else:
                print(f"  [{lesion_id}] FALHA: {result.error}")
        except Exception as exc:  # noqa: BLE001 -- registrado por imagem, galeria continua
            log_row["status"] = STATUS_FAILURE
            log_row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  [{lesion_id}] FALHA (fora do compute_gradcam): {exc}")
        log_row["elapsed_s"] = time.time() - t0
        log_rows.append(log_row)
        print(f"  [{i + 1}/{len(sample_rows)}] {lesion_id} ({group}) -- status={log_row['status']} ({log_row['elapsed_s']:.2f}s)")

    elapsed_total_s = time.time() - t_total0
    n_ok = sum(1 for r in log_rows if r["status"] == STATUS_OK)
    n_degenerate = sum(1 for r in log_rows if r["status"] in (STATUS_MAP_EMPTY_ALL_ZERO, STATUS_MAP_CONSTANT_NONZERO))
    n_failure = sum(1 for r in log_rows if r["status"] == STATUS_FAILURE)

    with open(MANIFEST_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else [])
        writer.writeheader()
        writer.writerows(log_rows)

    _write_gallery_html(log_rows, elapsed_total_s)

    print(f"\nConcluído: {n_ok} ok, {n_degenerate} mapas degenerados, {n_failure} falhas de {len(log_rows)}.")
    print(f"Tempo total: {elapsed_total_s:.1f}s ({elapsed_total_s / max(len(log_rows), 1):.2f}s/imagem em média).")
    print(f"Log de execução: {MANIFEST_OUTPUT}")
    print(f"Galeria: {GALLERY_HTML}")
    print("Lote inicial de até 40 imagens -- nenhuma amostra adicional foi gerada automaticamente.")


def _write_gallery_html(log_rows: list[dict], elapsed_total_s: float) -> None:
    by_group: dict[str, list[dict]] = {"TP": [], "TN": [], "FP": [], "FN": []}
    for row in log_rows:
        by_group.setdefault(row["group"], []).append(row)

    parts = [
        "<!doctype html><html lang='pt-br'><head><meta charset='utf-8'>",
        "<title>Galeria Grad-CAM -- CNN EfficientNetV2-S (classe-alvo: melanoma)</title>",
        "<style>body{font-family:sans-serif;margin:2em;}",
        ".card{display:inline-block;width:260px;vertical-align:top;margin:0.5em;border:1px solid #ccc;padding:0.5em;}",
        ".card img{width:100%;}",
        ".status-ok{color:#2a7;} .status-degenerate{color:#c80;} .status-falha{color:#c33;font-weight:bold;}",
        "</style></head><body>",
        "<h1>Galeria Grad-CAM qualitativa -- classe-alvo: melanoma (sempre)</h1>",
        "<p><strong>Aviso:</strong> análise qualitativa. Não comprova raciocínio clínico correto nem causalidade. "
        "Normalização min-max por imagem, independente entre imagens -- não compare magnitude absoluta entre mapas.</p>",
        f"<p>Tempo total de geração: {elapsed_total_s:.1f}s para {len(log_rows)} imagens "
        f"({elapsed_total_s / max(len(log_rows), 1):.2f}s/imagem em média). Lote inicial (até 40 imagens).</p>",
    ]
    for group, rows in by_group.items():
        parts.append(f"<h2>{group} ({len(rows)} casos)</h2>")
        for row in sorted(rows, key=lambda r: r["lesion_id"]):
            status = row["status"]
            status_class = "status-ok" if status == STATUS_OK else ("status-falha" if status == STATUS_FAILURE else "status-degenerate")
            parts.append("<div class='card'>")
            parts.append(f"<p><strong>{row['lesion_id']}</strong> ({row['image_id']})</p>")
            if status == STATUS_OK and row["overlay_image_path"]:
                rel = Path(row["overlay_image_path"]).name
                parts.append(f"<img src='images/{rel}' alt='Grad-CAM'>")
            parts.append(f"<p class='{status_class}'>status: {status}</p>")
            parts.append(f"<p>score (sigmoide, melanoma): {row['sigmoid_score']}</p>")
            if row["error"]:
                parts.append(f"<p>erro: {row['error']}</p>")
            parts.append("</div>")
    parts.append("</body></html>")

    with open(GALLERY_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    main()
