"""Reinitializes and trains the SINGLE final segmentation model (stage 4)
-- DeepLabV3+-ResNet34, approved by the researcher as the final protocol
segmenter -- on ALL 2,594 eligible ISIC2018 Task1 images, using the epoch
count and threshold already fixed by
``scripts/run_segmentation_final_config_fit.py``.

Freezes, before anything touches HAM10000:
  - the checkpoint (path + SHA-256 + byte size),
  - the mask threshold (read from ``final_config_fit_results.json``,
    never recomputed here),
  - the preprocessing (resize 256x256, bilinear for the image, ToTensor
    to [0,1] -- same as every other stage-4 script, no normalization
    stats applied, nothing new introduced here),
  - the postprocessing (sigmoid -> bilinear restore to the target
    image's native resolution -> binarize at the frozen threshold --
    same order already enforced by
    ``src.segmentation.metrics.compute_image_metrics``/
    ``restore_probability_to_original_resolution``: restore BEFORE
    binarizing, never after).

This script does NOT read, import, or reference HAM10000 in any way --
confirmed by inspection, not just by convention. Mask generation over the
frozen HAM10000 manifest is a separate, later script
(``scripts/run_s2_mask_generation.py``) that depends on THIS script's
output checkpoint + frozen config.

STATUS: implementada, aguardando execucao do pesquisador (Mac/MPS). Nao
executado pelo assistente que escreveu este script.

Usage:
    python scripts/run_segmentation_final_retrain.py --device mps \\
        --images-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip \\
        --masks-zip /Volumes/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from src.segmentation import folds as folds_mod  # noqa: E402
from src.segmentation.isic2018_dataset import ISIC2018Task1Dataset, verify_image_mask_correspondence  # noqa: E402
from src.segmentation.model_factory import build_segmentation_model  # noqa: E402

from run_isic2018_unet_pilot import _ensure_folds  # noqa: E402
from run_segmentation_arch_pilot import _resolve_device  # noqa: E402
from run_nested_cv_rehearsal import _filter_folds_data_to_matched, _train_one_epoch  # noqa: E402

ARCHITECTURE = "deeplabv3plus"
DEFAULT_IMAGES_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1-2_Training_Input.zip"
DEFAULT_MASKS_ZIP = "D:/Andrey/Mestrado/ISIC2018_Task1_Training_GroundTruth.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "segmentation")
DEFAULT_FOLDS_PATH = str(REPO_ROOT / "experiments" / "segmentation" / "isic2018_task1_folds.json")
DEFAULT_CONFIG_FIT_RESULTS = str(REPO_ROOT / "experiments" / "segmentation" / "final_config_fit_results.json")
DEFAULT_LR = 1e-4
DEFAULT_TRAIN_SEED = 42


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-zip", default=DEFAULT_IMAGES_ZIP)
    parser.add_argument("--masks-zip", default=DEFAULT_MASKS_ZIP)
    parser.add_argument("--folds-path", default=DEFAULT_FOLDS_PATH)
    parser.add_argument("--config-fit-results", default=DEFAULT_CONFIG_FIT_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None, help="cpu|cuda|mps (padrao: auto-detecta cuda>mps>cpu)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--seed", type=int, default=folds_mod.DEFAULT_SEED, help="semente do arquivo de folds")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
    device = _resolve_device(args.device)

    config_fit_path = Path(args.config_fit_results)
    if not config_fit_path.exists():
        raise FileNotFoundError(
            f"{config_fit_path} nao encontrado -- rode "
            "scripts/run_segmentation_final_config_fit.py primeiro."
        )
    with open(config_fit_path, encoding="utf-8") as f:
        config_fit = json.load(f)
    if config_fit["architecture"] != ARCHITECTURE:
        raise ValueError(
            f"{config_fit_path} foi gerado para '{config_fit['architecture']}', "
            f"esperado '{ARCHITECTURE}'."
        )
    chosen_epochs = config_fit["chosen_epochs"]
    chosen_threshold = config_fit["threshold_selection"]["chosen_threshold"]

    print(f"=== Retreino final -- {ARCHITECTURE} === dispositivo={device}")
    print(f"epocas fixadas (de {config_fit_path.name}): {chosen_epochs}")
    print(f"limiar fixado (de {config_fit_path.name}): {chosen_threshold}")

    print("[1/3] verificando correspondencia imagem<->mascara e carregando folds congelados...")
    correspondence = verify_image_mask_correspondence(args.images_zip, args.masks_zip)
    if not correspondence["ok"]:
        raise ValueError(f"Correspondencia imagem<->mascara incompleta: {correspondence}")
    folds_data = _ensure_folds(args.images_zip, args.folds_path, folds_mod.N_FOLDS, args.seed)
    matched_ids = set(folds_data["image_to_fold"].keys()) - set(correspondence["only_in_images"])
    folds_data = _filter_folds_data_to_matched(folds_data, matched_ids)
    all_ids = sorted(folds_data["image_to_fold"].keys())
    print(f"      {len(all_ids)} imagens elegiveis (esperado 2594)")
    if len(all_ids) != 2594:
        raise ValueError(
            f"Esperado exatamente 2594 imagens elegiveis do ISIC2018 Task1, encontrado {len(all_ids)} -- "
            "nao prosseguir sem entender a divergencia."
        )

    print(f"[2/3] treinando (reinicializado) em TODAS as {len(all_ids)} imagens, {chosen_epochs} epocas fixas...")
    torch.manual_seed(args.train_seed)
    model = build_segmentation_model(ARCHITECTURE, img_ch=3, output_ch=1)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset_train = ISIC2018Task1Dataset(all_ids, args.images_zip, args.masks_zip)
    loader = DataLoader(dataset_train, batch_size=min(4, len(all_ids)), shuffle=True, num_workers=0, drop_last=True)

    t0 = time.time()
    for epoch in range(1, chosen_epochs + 1):
        _train_one_epoch(model, loader, optimizer, criterion, device)
        print(f"      epoca {epoch}/{chosen_epochs} concluida")
    training_time_s = time.time() - t0

    ckpt_path = output_dir / f"final_model_{ARCHITECTURE}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": ARCHITECTURE,
            "epochs": chosen_epochs,
            "chosen_threshold": chosen_threshold,
            "train_ids_count": len(all_ids),
        },
        ckpt_path,
    )
    print(f"[3/3] checkpoint salvo: {ckpt_path}")
    sha256 = _sha256_of_file(ckpt_path)
    size_bytes = ckpt_path.stat().st_size
    print(f"      SHA-256: {sha256}")
    print(f"      tamanho: {size_bytes} bytes")

    frozen_config = {
        "architecture": ARCHITECTURE,
        "epochs": chosen_epochs,
        "epochs_source": str(config_fit_path),
        "threshold": chosen_threshold,
        "threshold_source": str(config_fit_path),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256,
        "checkpoint_size_bytes": size_bytes,
        "training_time_s": training_time_s,
        "n_training_images": len(all_ids),
        "training_lr": args.lr,
        "training_optimizer": "AdamW",
        "training_loss": "BCEWithLogitsLoss",
        "preprocessing": {
            "resize": "256x256, bilinear (imagem), ToTensor para [0,1] -- sem normalizacao adicional",
            "mask_resize_during_training": "256x256, nearest (apenas para o treino; nao usado na inferencia de S2)",
        },
        "postprocessing": {
            "order": "sigmoid -> restaurar para a resolucao nativa da imagem de entrada (bilinear) -> binarizar no limiar congelado acima",
            "note": "restaurar ANTES de binarizar, nunca depois -- mesma regra de src.segmentation.metrics.compute_image_metrics",
        },
        "device_used": str(device),
        "ham10000_touched": False,
        "note": (
            "Modelo FINAL unico, treinado do zero em todas as 2594 imagens elegiveis "
            "do ISIC2018 Task1 -- nao e nenhum dos checkpoints intermediarios/OOF de "
            "run_segmentation_final_config_fit.py. Limiar congelado aqui NAO deve ser "
            "reaberto sem nova decisao do pesquisador."
        ),
    }

    frozen_config_path = output_dir / "final_model_frozen_config.json"
    with open(frozen_config_path, "w", encoding="utf-8") as f:
        json.dump(frozen_config, f, ensure_ascii=False, indent=2, default=str)

    report_path = output_dir / "final_model_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"# Modelo final congelado -- {ARCHITECTURE}\n\n"
            f"- Epocas: {chosen_epochs} (fonte: {config_fit_path.name})\n"
            f"- Limiar: {chosen_threshold} (fonte: {config_fit_path.name})\n"
            f"- Checkpoint: `{ckpt_path}`\n"
            f"- SHA-256: `{sha256}`\n"
            f"- Tamanho: {size_bytes} bytes\n"
            f"- Imagens de treino: {len(all_ids)} (todas as elegiveis do ISIC2018 Task1)\n"
            f"- Tempo de treino: {training_time_s:.1f}s\n"
            f"- HAM10000 tocado: False\n"
        )

    print(f"\nconfiguracao congelada (json): {frozen_config_path}")
    print(f"relatorio (md): {report_path}")
    print("\nPROXIMO PASSO: scripts/run_s2_mask_generation.py (gera as mascaras S2 sobre o manifesto HAM10000).")


if __name__ == "__main__":
    main()
