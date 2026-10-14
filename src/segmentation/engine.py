"""Laço de treino/validação genérico (independente de arquitetura).

A estratégia fornece o `nn.Module` (logits de saída) e a política de congelamento;
este módulo apenas otimiza e valida. Usado tanto no treino no ISIC quanto no
fine-tuning, para que ambos compartilhem exatamente a mesma lógica.
"""
import copy
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' solicitado, mas CUDA não está disponível.")
    return torch.device(name)


# ─────────────────────────────────────────────────────────────────────────────
# Perdas e métrica de validação (diferenciáveis / em lote)
# ─────────────────────────────────────────────────────────────────────────────

def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def build_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    bce = nn.BCEWithLogitsLoss()
    if name == "bce":
        return lambda lg, tg: bce(lg, tg)
    if name == "dice":
        return soft_dice_loss
    if name == "bce_dice":
        return lambda lg, tg: bce(lg, tg) + soft_dice_loss(lg, tg)
    raise ValueError(f"Loss desconhecida: {name}")


def batch_dice(logits: torch.Tensor, target: torch.Tensor, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    """Dice duro por imagem (limiar fixo). Convenção: ambas vazias -> 1."""
    pred = (torch.sigmoid(logits) > threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    denom = pred.sum(dims) + target.sum(dims)
    return (2.0 * inter + eps) / (denom + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Otimizador / scheduler
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float, momentum: float):
    # Todos os parâmetros entram no otimizador; os congelados têm grad None e são
    # ignorados, o que permite descongelar durante o treino sem recriá-lo.
    params = list(model.parameters())
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Otimizador desconhecido: {name}")


def build_scheduler(optimizer, sched_cfg: Dict, max_epochs: int):
    name = sched_cfg.get("name", "none")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs, eta_min=float(sched_cfg.get("min_lr", 0.0)))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sched_cfg.get("step_size", 10)),
            gamma=float(sched_cfg.get("gamma", 0.1)))
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(sched_cfg.get("gamma", 0.1)),
            patience=int(sched_cfg.get("step_size", 10)))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Treino
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FitConfig:
    max_epochs: int
    learning_rate: float
    batch_size: int
    optimizer: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    scheduler: Dict = field(default_factory=lambda: {"name": "none"})
    loss: str = "bce_dice"
    threshold: float = 0.5
    early_stopping: Dict = field(default_factory=lambda: {"enabled": True, "patience": 10,
                                                          "min_delta": 0.0, "monitor": "val_dice"})
    seed: int = 42
    num_workers: int = 0
    mixed_precision: bool = False


@dataclass
class FitResult:
    best_epoch: int
    best_monitor: float
    epochs_run: int
    stopped_early: bool
    history: List[Dict]
    selected_by: str  # "validation" | "last_epoch" (sem validação)


def _loader(ds: Dataset, cfg: FitConfig, shuffle: bool, seed_offset: int = 0) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(cfg.seed + seed_offset)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                      generator=gen, drop_last=False)


def evaluate_loader(model: nn.Module, loader: DataLoader, loss_fn, threshold: float,
                    device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    dices: List[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            total_loss += float(loss_fn(logits, masks)) * images.size(0)
            n += images.size(0)
            dices.extend(batch_dice(logits, masks, threshold).cpu().tolist())
    return {"val_loss": total_loss / max(n, 1), "val_dice": float(np.mean(dices)) if dices else 0.0}


def fit(
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    cfg: FitConfig,
    device: torch.device,
    on_epoch_start: Optional[Callable[[nn.Module, int], None]] = None,
    on_train_mode: Optional[Callable[[nn.Module], None]] = None,
) -> FitResult:
    """Treina `model` e restaura os pesos da melhor época (por validação).

    Sem `val_ds`, treina `max_epochs` épocas fixas e mantém a última (sem seleção).

    Args:
        on_epoch_start: gancho chamado antes de cada época (ex.: descongelamento gradual).
        on_train_mode: gancho chamado após `model.train()` (ex.: BN de blocos congelados em eval).
    """
    model.to(device)
    loss_fn = build_loss(cfg.loss)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.learning_rate, cfg.weight_decay, cfg.momentum)
    scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.max_epochs)
    use_amp = bool(cfg.mixed_precision and device.type == "cuda")
    if cfg.mixed_precision and not use_amp:
        logger.warning("mixed_precision ignorado: só é suportado em CUDA (device=%s).", device.type)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_loader = _loader(train_ds, cfg, shuffle=True)
    val_loader = _loader(val_ds, cfg, shuffle=False) if val_ds is not None else None

    es = cfg.early_stopping
    monitor = es.get("monitor", "val_dice")
    maximize = monitor == "val_dice"
    best_val = -float("inf") if maximize else float("inf")
    best_epoch, best_state, bad_epochs = 0, None, 0
    history: List[Dict] = []
    stopped_early = False

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        if on_epoch_start is not None:
            on_epoch_start(model, epoch)
        model.train()
        if on_train_mode is not None:
            on_train_mode(model)
        run_loss, n = 0.0, 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * images.size(0)
            n += images.size(0)
        row = {"epoch": epoch, "train_loss": run_loss / max(n, 1),
               "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": None}

        if val_loader is not None:
            row.update(evaluate_loader(model, val_loader, loss_fn, cfg.threshold, device))
            current = row[monitor]
            improved = (current > best_val + es.get("min_delta", 0.0)) if maximize \
                else (current < best_val - es.get("min_delta", 0.0))
            if improved:
                best_val, best_epoch, bad_epochs = current, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad_epochs += 1
        row["epoch_time_s"] = round(time.time() - t0, 3)
        history.append(row)
        logger.info("Época %03d/%03d | %s", epoch, cfg.max_epochs,
                    " | ".join(f"{k}={v:.4f}" for k, v in row.items()
                               if isinstance(v, float) and k != "epoch_time_s"))

        if scheduler is not None:
            if cfg.scheduler.get("name") == "plateau" and val_loader is not None:
                scheduler.step(row["val_loss"])
            else:
                scheduler.step()

        if val_loader is not None and es.get("enabled", True) and bad_epochs >= es.get("patience", 10):
            stopped_early = True
            logger.info("Early stopping na época %d (melhor: %d).", epoch, best_epoch)
            break

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        return FitResult(best_epoch, float(best_val), len(history), stopped_early, history, "validation")
    return FitResult(len(history), float("nan"), len(history), False, history, "last_epoch")
