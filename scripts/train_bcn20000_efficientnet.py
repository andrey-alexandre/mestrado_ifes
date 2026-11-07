"""Train EfficientNetV2-S (ImageNet init) on the APPROVED BCN20000
train/val split for the binary melanoma vs. non-melanoma CNN baseline
(stage 3). Threshold selection (indice de Youden) is a SEPARATE, later
task that consumes val_scores.csv -- this script never binarizes
predictions.

HAM10000 is never read or referenced here: it is reserved exclusively
for the final external evaluation.

Two modes:

  --smoke-test
      Fast (seconds), real-pipeline check on a tiny real subset (not
      synthetic): image loading from the real zip, label loading, a
      weight update, checkpoint save/reload. Does NOT require
      --epochs/--batch-size/--lr/--seed (uses small fixed internal
      values for this short diagnostic only). Prints PASS/FAIL per
      check. Intended to be run by the coordinator before the
      researcher commits to the full training run.

  (default, full training)
      REQUIRES --epochs, --batch-size, --lr, --seed explicitly -- no
      silent defaults for any of these. Trains, logs per-epoch history,
      selects the best checkpoint by validation AUC-ROC, and writes
      val_scores.csv (raw, non-binarized predicted scores) for the
      selected checkpoint.

STATUS: implementada, aguardando execucao (smoke test: coordenador;
treino completo: pesquisador). Not run by the assistant/fork that wrote
it.

Usage:
    python scripts/train_bcn20000_efficientnet.py --smoke-test
    python scripts/train_bcn20000_efficientnet.py \
        --epochs 20 --batch-size 32 --lr 0.0001 --seed 777
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_audit import core  # noqa: E402
from src.cnn_baseline.bcn20000_dataset import (  # noqa: E402
    BCN20000ZipDataset,
    filter_rows_by_split,
    read_split_manifest,
)
from src.cnn_baseline.efficientnet_training import (  # noqa: E402
    BINARY_TARGET_TO_LABEL,
    build_model,
    build_val_scores_rows,
    select_best_epoch,
)

DEFAULT_MANIFEST = str(REPO_ROOT / "experiments" / "cnn_baseline" / "bcn20000_split_manifest.csv")
DEFAULT_ZIP_PATH = "D:/Andrey/Mestrado/_extracted/BCN_20k_train.zip"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "cnn_baseline" / "training")
SMOKE_TEST_OUTPUT_DIR = str(REPO_ROOT / "experiments" / "cnn_baseline" / "smoke_test")
SMOKE_TEST_N_IMAGES = 12
SMOKE_TEST_STEPS = 2

# Data-split seed (20261107) used by scripts/run_bcn20000_split.py to
# generate DEFAULT_MANIFEST -- recorded here only for the config.json
# audit trail, never re-used as a training seed (those are independent
# and must be passed explicitly via --seed).
DATA_SPLIT_SEED = 20261107
DATA_SPLIT_PARTITION_POLICY = "apenas_train"
DATA_SPLIT_VAL_FRACTION = 0.20


def _best_available_device() -> str:
    """Pick the best device PyTorch can actually use on this machine,
    preferring CUDA (NVIDIA) > MPS (Apple Silicon GPU, e.g. M-series Macs)
    > CPU. ``--device`` can always override this explicitly.

    The original version of this function only checked
    ``torch.cuda.is_available()``, which is always False on a Mac with no
    NVIDIA GPU -- it would silently fall back to CPU on Apple Silicon
    hardware that actually has a usable GPU via the MPS backend. Fixed
    here rather than left as a trap for the first non-CUDA machine to run
    this script.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_eval_transform():
    from torchvision.models import EfficientNet_V2_S_Weights

    return EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()


def _build_augment():
    """Train-only augmentation: random horizontal flip (p=0.5), applied
    to the PIL image BEFORE the ImageNet eval transform (resize/
    normalize). Never applied to the validation split. This is the only
    augmentation used -- documented here, not left implicit."""
    from torchvision import transforms as T

    return T.RandomHorizontalFlip(p=0.5)


def _run_epoch(model, loader, criterion, optimizer, device, train: bool) -> dict[str, Any]:
    import torch

    model.train(mode=train)
    total_loss = 0.0
    n = 0
    all_labels: list[int] = []
    all_scores: list[float] = []
    all_lesion_ids: list[str] = []
    all_filenames: list[str] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, lesion_ids, filenames in loader:
            images = images.to(device)
            labels_float = labels.float().to(device)

            if train:
                optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_float)
            if train:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n += batch_size

            scores = torch.sigmoid(logits).detach().cpu().tolist()
            all_scores.extend(scores)
            all_labels.extend(labels.tolist())
            all_lesion_ids.extend(lesion_ids)
            all_filenames.extend(filenames)

    avg_loss = total_loss / n if n else float("nan")
    return {
        "avg_loss": avg_loss,
        "labels": all_labels,
        "scores": all_scores,
        "lesion_ids": all_lesion_ids,
        "filenames": all_filenames,
    }


def _compute_auc(labels: list[int], scores: list[float]) -> float:
    """AUC-ROC is undefined with a single class present; returns NaN in
    that case rather than raising, so a degenerate epoch does not crash
    the whole run -- NaN then naturally loses any AUC-based selection.
    Not expected in practice at the full-validation-set level (715
    lesions, both classes well represented per the approved split), but
    kept defensive."""
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def run_smoke_test(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    output_dir = Path(SMOKE_TEST_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(output_dir / "smoke_test.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )
    results: dict[str, str] = {}

    print("=== SMOKE TEST: pipeline real do treino BCN20000 ===")
    manifest_rows = read_split_manifest(args.manifest)
    train_rows = filter_rows_by_split(manifest_rows, "train")[:SMOKE_TEST_N_IMAGES]
    print(f"[1/4] carregando {len(train_rows)} imagens reais do manifesto...")

    eval_transform = _build_eval_transform()
    augment = _build_augment()
    dataset = BCN20000ZipDataset(train_rows, args.zip_path, transform=eval_transform, augment=augment)

    try:
        image, label, lesion_id, filename = dataset[0]
        results["image_load"] = "PASSOU"
        print(f"      imagem carregada: shape={tuple(image.shape)}, lesion_id={lesion_id}, label={label}")
    except Exception as exc:  # noqa: BLE001 - smoke test must report, not crash
        results["image_load"] = f"FALHOU: {exc}"
        label = None
        print(f"      FALHOU: {exc}")

    if label not in (0, 1):
        results["label_load"] = f"FALHOU: label inesperado {label!r}"
    else:
        results["label_load"] = "PASSOU"
    print(f"[2/4] rotulo: {results['label_load']}")

    loader = DataLoader(dataset, batch_size=min(4, len(train_rows)), shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model, _weights = build_model(pretrained=True)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    before = model.classifier[-1].weight.detach().clone()
    try:
        for step, (images, labels, _lesion_ids, _filenames) in enumerate(loader):
            if step >= SMOKE_TEST_STEPS:
                break
            images = images.to(device)
            labels_float = labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_float)
            loss.backward()
            optimizer.step()
        after = model.classifier[-1].weight.detach().clone()
        changed = not torch.equal(before, after)
        results["weight_update"] = "PASSOU" if changed else "FALHOU: pesos nao mudaram"
    except Exception as exc:  # noqa: BLE001
        results["weight_update"] = f"FALHOU: {exc}"
    print(f"[3/4] atualizacao de peso: {results['weight_update']}")

    ckpt_path = output_dir / "smoke_test_checkpoint.pt"
    try:
        torch.save({"model_state_dict": model.state_dict(), "smoke_test": True}, ckpt_path)
        reloaded_model, _ = build_model(pretrained=False)
        state = torch.load(ckpt_path, map_location="cpu")
        reloaded_model.load_state_dict(state["model_state_dict"])
        results["checkpoint_save_reload"] = "PASSOU"
    except Exception as exc:  # noqa: BLE001
        results["checkpoint_save_reload"] = f"FALHOU: {exc}"
    print(f"[4/4] checkpoint (salvar+recarregar): {results['checkpoint_save_reload']}")

    all_passed = all(v == "PASSOU" for v in results.values())
    print(f"\n=== RESULTADO: {'TODAS AS VERIFICACOES PASSARAM' if all_passed else 'AO MENOS UMA FALHOU'} ===")
    with open(output_dir / "smoke_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"resultados gravados: {output_dir / 'smoke_test_results.json'}")
    if not all_passed:
        sys.exit(1)


def run_training(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    missing = [
        name
        for name, val in (
            ("--epochs", args.epochs),
            ("--batch-size", args.batch_size),
            ("--lr", args.lr),
            ("--seed", args.seed),
        )
        if val is None
    ]
    if missing:
        print(f"ERRO: treino completo exige {missing} explicitamente, sem default. Nada foi feito.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(output_dir / "training.log"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    _set_seed(args.seed)
    device = torch.device(args.device)
    print(f"dispositivo: {device}")

    manifest_rows = read_split_manifest(args.manifest)
    manifest_sha256 = core.sha256_file(args.manifest)
    train_rows = filter_rows_by_split(manifest_rows, "train")
    val_rows = filter_rows_by_split(manifest_rows, "val")
    print(f"treino: {len(train_rows)} imagens / validacao: {len(val_rows)} imagens")

    eval_transform = _build_eval_transform()
    augment = _build_augment()
    train_dataset = BCN20000ZipDataset(train_rows, args.zip_path, transform=eval_transform, augment=augment)
    val_dataset = BCN20000ZipDataset(val_rows, args.zip_path, transform=eval_transform, augment=None)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model, _weights = build_model(pretrained=True)
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config = {
        "architecture": "efficientnet_v2_s",
        "pretrained_weights": "IMAGENET1K_V1",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed_treino": args.seed,
        "device": str(device),
        "num_workers": args.num_workers,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "augmentation": "RandomHorizontalFlip(p=0.5), somente no split de treino",
        "binary_target_to_label": BINARY_TARGET_TO_LABEL,
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": manifest_sha256,
        "data_split_seed": DATA_SPLIT_SEED,
        "data_split_partition_policy": DATA_SPLIT_PARTITION_POLICY,
        "data_split_val_fraction_requested": DATA_SPLIT_VAL_FRACTION,
        "checkpoint_selection_rule": (
            "melhor AUC-ROC de validacao (maior valor; empate: epoca mais cedo)"
        ),
        "n_train_images": len(train_rows),
        "n_val_images": len(val_rows),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"config gravado: {output_dir / 'config.json'}")

    history: list[dict[str, Any]] = []
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_result = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_result = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        val_auc = _compute_auc(val_result["labels"], val_result["scores"])
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_result["avg_loss"],
            "val_loss": val_result["avg_loss"],
            "val_auc": val_auc,
            "elapsed_seconds": elapsed,
        }
        history.append(entry)
        msg = (
            f"epoca {epoch}/{args.epochs} - train_loss={train_result['avg_loss']:.4f} "
            f"val_loss={val_result['avg_loss']:.4f} val_auc={val_auc:.4f} ({elapsed:.1f}s)"
        )
        print(msg)
        logging.info(msg)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
            },
            checkpoints_dir / f"epoch_{epoch:03d}.pt",
        )

    with open(output_dir / "training_history.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_auc", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows(history)
    print(f"historico gravado: {output_dir / 'training_history.csv'}")

    selection = select_best_epoch(history)
    print(
        f"checkpoint selecionado: epoca {selection['selected_epoch']} "
        f"(val_auc={selection['selected_val_auc']:.4f}) -- regra: {selection['rule']}"
    )

    best_epoch = selection["selected_epoch"]
    best_ckpt_src = checkpoints_dir / f"epoch_{best_epoch:03d}.pt"
    best_ckpt_dst = output_dir / "best_checkpoint.pt"
    shutil.copy(best_ckpt_src, best_ckpt_dst)
    print(f"melhor checkpoint copiado para: {best_ckpt_dst}")

    best_state = torch.load(best_ckpt_dst, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    final_val_result = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

    val_scores_rows = build_val_scores_rows(
        final_val_result["lesion_ids"],
        final_val_result["filenames"],
        final_val_result["labels"],
        final_val_result["scores"],
    )
    val_scores_path = output_dir / "val_scores.csv"
    with open(val_scores_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["lesion_id", "bcn_filename", "binary_target", "true_label", "predicted_score"]
        )
        writer.writeheader()
        writer.writerows(val_scores_rows)
    print(f"escores de validacao (checkpoint selecionado) gravados: {val_scores_path}")

    with open(output_dir / "checkpoint_selection.json", "w", encoding="utf-8") as f:
        json.dump({**selection, "config": config}, f, ensure_ascii=False, indent=2)

    print(
        "LEMBRETE: val_scores.csv contem escore bruto (sigmoid em [0,1]), "
        "nao binarizado. Selecao de limiar (Youden) e a proxima tarefa. "
        "HAM10000 nao foi lido nem tocado nesta execucao."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--zip-path", default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=None, help="Obrigatorio fora de --smoke-test.")
    parser.add_argument("--batch-size", type=int, default=None, help="Obrigatorio fora de --smoke-test.")
    parser.add_argument("--lr", type=float, default=None, help="Obrigatorio fora de --smoke-test.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Obrigatorio fora de --smoke-test. Seed de TREINO, distinto do "
        f"seed {DATA_SPLIT_SEED} da divisao de dados -- nao confundir os dois.",
    )
    parser.add_argument(
        "--device",
        default=_best_available_device(),
        help="Auto-detecta cuda > mps (GPU Apple Silicon) > cpu, nessa ordem de "
        "preferencia. Use para forcar um valor especifico, ex. --device mps.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
