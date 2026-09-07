"""
Treinamento da ResNet-50 como baseline binário no dataset PH2.

Classificação: Melanoma = 1, demais casos (nevus comum, nevus displásico) = 0.
Split estratificado 70/30 com semente fixada para reprodutibilidade.

Todos os hiperparâmetros são lidos de config/config.yaml (seção `training`).
Valores individuais podem ser sobrescritos via CLI sem alterar o YAML.

Uso:
    python scripts/train_resnet50_ph2.py
    python scripts/train_resnet50_ph2.py --config config/config.yaml
    python scripts/train_resnet50_ph2.py --epochs 50 --learning_rate 1e-2
    python scripts/train_resnet50_ph2.py --freeze_backbone --output_dir runs/linear_probe
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("resnet50_ph2")

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DA CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config(config_path: Path) -> dict:
    """Lê o YAML e retorna o dicionário completo."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Sobrescreve valores do YAML com os argumentos CLI fornecidos explicitamente."""
    t = cfg["training"]
    p = cfg["paths"]

    overrides = {
        "output_dir":      args.output_dir,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "learning_rate":   args.learning_rate,
        "freeze_backbone": args.freeze_backbone if args.freeze_backbone else None,
    }
    for key, val in overrides.items():
        if val is not None:
            t[key] = val

    if args.dataset_dir:
        p["ph2_dataset_dir"] = args.dataset_dir
    if args.metadata_txt:
        p["ph2_metadata_txt"] = args.metadata_txt

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PYTORCH
# ─────────────────────────────────────────────────────────────────────────────

class PH2TorchDataset(Dataset):
    """Dataset PyTorch que lê imagens BMP do PH2 e aplica transformações."""

    def __init__(self, image_paths: List[str], labels: List[int], transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def build_transforms(t: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """Constrói transforms de treino e avaliação a partir da seção `training` do YAML."""
    size = t["input_size"]

    aug_steps = [
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
    ]

    if t.get("aug_horizontal_flip"):
        aug_steps.append(transforms.RandomHorizontalFlip())

    if t.get("aug_vertical_flip"):
        aug_steps.append(transforms.RandomVerticalFlip())

    if t.get("aug_rotation_degrees", 0) > 0:
        aug_steps.append(
            transforms.RandomRotation(
                degrees=t["aug_rotation_degrees"],
                interpolation=InterpolationMode.BICUBIC,
                fill=0,
            )
        )

    if t.get("aug_color_jitter"):
        aug_steps.append(
            transforms.ColorJitter(
                brightness=t.get("aug_color_brightness", 0.2),
                contrast=t.get("aug_color_contrast", 0.2),
                saturation=t.get("aug_color_saturation", 0.2),
                hue=t.get("aug_color_hue", 0.05),
            )
        )

    aug_steps += [
        transforms.ToTensor(),
        transforms.Normalize(t["normalize_mean"], t["normalize_std"]),
    ]

    if t.get("aug_random_erasing"):
        aug_steps.append(transforms.RandomErasing(p=t.get("aug_erasing_prob", 0.2)))

    train_tf = transforms.Compose(aug_steps)

    eval_tf = transforms.Compose([
        transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(t["normalize_mean"], t["normalize_std"]),
    ])

    return train_tf, eval_tf


# ─────────────────────────────────────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────────────────────────────────────

def build_model(t: dict) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if t.get("pretrained", True) else None
    model = models.resnet50(weights=weights)

    if t.get("freeze_backbone", False):
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# OTIMIZADOR E SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, t: dict):
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = t.get("optimizer", "sgd").lower()
    lr   = t["learning_rate"]
    wd   = t.get("weight_decay", 1e-4)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=t.get("momentum", 0.9), weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Otimizador desconhecido: {name}")


def build_scheduler(optimizer, t: dict):
    name = t.get("scheduler", "none").lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t["epochs"])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=t.get("step_size", 10), gamma=t.get("step_gamma", 0.1)
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LOOP DE TREINO / AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    training: bool,
) -> Tuple[float, float, List[float], List[int]]:
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    all_probs: List[float] = []
    all_labels: List[int] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long()
            correct += (preds == labels.squeeze(1).long()).sum().item()
            n += len(labels)
            total_loss += loss.item() * len(labels)

            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(labels.squeeze(1).cpu().long().tolist())

    return total_loss / n, correct / n, all_probs, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(probs: List[float], labels: List[int]) -> Dict[str, float]:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    return {
        "accuracy":          accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "f1":                f1_score(labels, preds, zero_division=0),
        "auc":               auc,
        "sensitivity":       tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "specificity":       tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DOS DADOS PH2
# ─────────────────────────────────────────────────────────────────────────────

def load_ph2_entries(dataset_dir: str, metadata_txt: str) -> Tuple[List[str], List[int]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.dataset import PH2Dataset

    ds = PH2Dataset(dataset_dir=dataset_dir, metadata_txt=metadata_txt)
    entries = ds.get_image_paths()
    paths  = [e[0] for e in entries if e[2] != -1]
    labels = [e[2] for e in entries if e[2] != -1]
    logger.info(
        "PH2: %d amostras válidas — Melanoma=%d, Não-melanoma=%d",
        len(labels), sum(labels), len(labels) - sum(labels),
    )
    return paths, labels


# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict) -> Dict:
    t = cfg["training"]
    p = cfg["paths"]

    output_path = Path(t["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(t["random_seed"])
    np.random.seed(t["random_seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    # ── Dados ────────────────────────────────────────────────────────────────
    paths, labels = load_ph2_entries(p["ph2_dataset_dir"], p["ph2_metadata_txt"])

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels,
        test_size=t["test_size"],
        random_state=t["random_seed"],
        stratify=labels,
    )
    logger.info(
        "Split estratificado — Treino: %d (mel=%d) | Teste: %d (mel=%d)",
        len(train_labels), sum(train_labels),
        len(test_labels),  sum(test_labels),
    )

    train_tf, eval_tf = build_transforms(t)

    train_loader = DataLoader(
        PH2TorchDataset(train_paths, train_labels, train_tf),
        batch_size=t["batch_size"], shuffle=True,
        num_workers=t.get("num_workers", 4), pin_memory=True,
    )
    test_loader = DataLoader(
        PH2TorchDataset(test_paths, test_labels, eval_tf),
        batch_size=t["batch_size"], shuffle=False,
        num_workers=t.get("num_workers", 4), pin_memory=True,
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    model = build_model(t).to(device)

    # ── Perda com pos_weight para desbalanceamento ───────────────────────────
    pw = t.get("pos_weight") or (train_labels.count(0) / max(train_labels.count(1), 1))
    logger.info("pos_weight: %.4f", pw)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], dtype=torch.float32).to(device)
    )

    optimizer = build_optimizer(model, t)
    scheduler = build_scheduler(optimizer, t)

    # ── Loop de épocas ────────────────────────────────────────────────────────
    history: List[Dict] = []
    best_state: Dict    = {}
    best_metric_val     = -float("inf")
    epochs_no_improve   = 0
    best_epoch          = 0

    monitor_key_map = {"val_loss": "loss", "val_auc": "auc", "val_f1": "f1"}

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_probs, tr_labs = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        va_loss, va_acc, va_probs, va_labs = run_epoch(
            model, test_loader, criterion, None, device, training=False
        )

        if scheduler is not None:
            scheduler.step()

        tr_m = compute_metrics(tr_probs, tr_labs)
        va_m = compute_metrics(va_probs, va_labs)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch":            epoch,
            "train_loss":       round(tr_loss, 4),
            "train_acc":        round(tr_acc, 4),
            "train_auc":        round(tr_m["auc"], 4) if not np.isnan(tr_m["auc"]) else None,
            "train_f1":         round(tr_m["f1"], 4),
            "val_loss":         round(va_loss, 4),
            "val_acc":          round(va_acc, 4),
            "val_auc":          round(va_m["auc"], 4) if not np.isnan(va_m["auc"]) else None,
            "val_f1":           round(va_m["f1"], 4),
            "val_sensitivity":  round(va_m["sensitivity"], 4),
            "val_specificity":  round(va_m["specificity"], 4),
            "lr":               round(current_lr, 8),
            "epoch_time_s":     round(time.time() - t0, 1),
        }
        history.append(row)

        logger.info(
            "Época %02d/%02d | loss %.4f→%.4f | acc %.3f→%.3f | "
            "AUC %.3f→%.3f | F1 %.3f→%.3f | sen %.3f | esp %.3f | lr %.2e | %.1fs",
            epoch, t["epochs"],
            tr_loss, va_loss, tr_acc, va_acc,
            tr_m["auc"], va_m["auc"], tr_m["f1"], va_m["f1"],
            va_m["sensitivity"], va_m["specificity"],
            current_lr, time.time() - t0,
        )

        # Early stopping
        monitor_key = monitor_key_map.get(t.get("monitor_metric", "val_auc"), "auc")
        current_val = va_m.get(monitor_key, va_loss)
        if t.get("monitor_metric") == "val_loss":
            current_val = -va_loss

        if current_val > best_metric_val:
            best_metric_val   = current_val
            best_epoch        = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if t.get("early_stopping") and epochs_no_improve >= t.get("patience", 10):
            logger.info(
                "Early stopping na época %d (sem melhora em %d épocas). Melhor: época %d",
                epoch, t["patience"], best_epoch,
            )
            break

    # ── Avaliação final com os melhores pesos ─────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)

    _, _, test_probs, test_labs = run_epoch(
        model, test_loader, criterion, None, device, training=False
    )
    final_metrics = compute_metrics(test_probs, test_labs)
    final_preds   = [1 if p >= 0.5 else 0 for p in test_probs]

    logger.info("═" * 60)
    logger.info("RESULTADO FINAL (melhor época: %d)", best_epoch)
    for k, v in final_metrics.items():
        logger.info("  %-22s %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    print("\nRelatório de classificação (teste):")
    print(classification_report(test_labs, final_preds, target_names=["Não-melanoma", "Melanoma"]))
    print(f"Matriz de confusão:\n{confusion_matrix(test_labs, final_preds)}")

    # ── Persistência ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output_path / "best_model.pt")

    results = {
        "config":        t,
        "best_epoch":    best_epoch,
        "final_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in final_metrics.items()},
        "history":       history,
        "split": {
            "train_total":    len(train_labels),
            "train_melanoma": sum(train_labels),
            "test_total":     len(test_labels),
            "test_melanoma":  sum(test_labels),
        },
    }
    with open(output_path / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Artefatos salvos em: %s", output_path)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline ResNet-50 — PH2 Dataset")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Caminho para config.yaml (padrão: config/config.yaml)")
    # Caminhos — sobrescrevem paths.ph2_* do YAML
    p.add_argument("--dataset_dir",  default=None,
                   help='Raiz do PH2 (sobrescreve paths.ph2_dataset_dir do YAML)')
    p.add_argument("--metadata_txt", default=None,
                   help="PH2_dataset.txt (sobrescreve paths.ph2_metadata_txt do YAML)")
    # Hiperparâmetros — sobrescrevem training.* do YAML
    p.add_argument("--output_dir",     default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch_size",     type=int,   default=None)
    p.add_argument("--learning_rate",  type=float, default=None)
    p.add_argument("--freeze_backbone", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(Path(args.config))
    cfg  = apply_cli_overrides(cfg, args)
    train(cfg)
