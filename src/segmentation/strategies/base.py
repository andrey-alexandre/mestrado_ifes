"""Interface abstrata das estratégias de segmentação (Strategy Pattern).

Uma estratégia encapsula apenas o que é específico da arquitetura:
  - `build_model()`: constrói o `nn.Module` (saída = logits, 1 canal);
  - `block_groups()`: separa os blocos nomeados em encoder/decoder (usado para congelar).
O restante (treino, fine-tuning, predição, checkpoints) é implementado aqui uma
única vez e pode ser sobrescrito por arquiteturas que precisem de outro
comportamento. O pipeline experimental (folds, avaliação, persistência) só conhece
esta interface.
"""
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import Dataset

from src.segmentation.data import JointTransform
from src.segmentation.engine import (
    FitConfig, FitResult, fit, resolve_device, seed_everything,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1


class CheckpointError(RuntimeError):
    """Checkpoint ausente, corrompido ou incompatível com a estratégia."""


def state_dict_sha256(model: nn.Module) -> str:
    """Hash do conteúdo dos pesos (independe de metadados do arquivo)."""
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class SegmentationStrategy(ABC):
    """Contrato mínimo de uma arquitetura de segmentação."""

    #: nome usado no config.yaml (`segmentation.strategy`)
    name: str = ""
    #: o lado da imagem de entrada precisa ser múltiplo deste valor
    size_multiple: int = 1

    def __init__(self, seg_cfg: Dict):
        self.seg_cfg = seg_cfg
        self.params: Dict = dict(seg_cfg.get("strategy_params") or {})
        self.device = resolve_device(seg_cfg["device"])
        size = int(seg_cfg["image_size"])
        if size % self.size_multiple != 0:
            raise ValueError(
                f"image_size={size} deve ser múltiplo de {self.size_multiple} para a estratégia '{self.name}'."
            )
        self.model: Optional[nn.Module] = None

    # ── específico da arquitetura ───────────────────────────────────────────
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Instancia o módulo PyTorch (pesos aleatórios; saída em logits)."""

    @abstractmethod
    def block_groups(self, model: nn.Module) -> Dict[str, List[str]]:
        """Nomes dos blocos de primeiro nível: {'encoder': [...], 'decoder': [...]}."""

    # ── infraestrutura compartilhada ────────────────────────────────────────
    def ensure_model(self) -> nn.Module:
        if self.model is None:
            self.model = self.build_model().to(self.device)
        return self.model

    def transform(self, train: bool) -> JointTransform:
        return JointTransform(
            size=int(self.seg_cfg["image_size"]),
            train=train,
            augmentation=self.seg_cfg.get("augmentation") if train else None,
            normalize=self.seg_cfg.get("normalize"),
        )

    def _fit_config(self, **overrides) -> FitConfig:
        s = self.seg_cfg
        base = dict(
            max_epochs=s["max_epochs"], learning_rate=s["learning_rate"], batch_size=s["batch_size"],
            optimizer=s["optimizer"], weight_decay=s["weight_decay"], momentum=s["momentum"],
            scheduler=s["scheduler"], loss=s["loss"], threshold=s["threshold"],
            early_stopping=s["early_stopping"], seed=s["seed"], num_workers=s["num_workers"],
            mixed_precision=s["mixed_precision"],
        )
        base.update(overrides)
        return FitConfig(**base)

    # ── treino do zero (ISIC 2018) ──────────────────────────────────────────
    def train(self, train_ds: Dataset, val_ds: Optional[Dataset]) -> FitResult:
        """Treina a partir da inicialização configurada (aleatória ou checkpoint)."""
        seed_everything(self.seg_cfg["seed"])
        self.model = None
        model = self.ensure_model()
        if self.seg_cfg["weights_init"] == "checkpoint":
            self.load_checkpoint(self.seg_cfg["init_checkpoint"])
            model = self.model
        for p in model.parameters():
            p.requires_grad = True
        return fit(model, train_ds, val_ds, self._fit_config(), self.device)

    # ── fine-tuning (PH²) ───────────────────────────────────────────────────
    def apply_freeze_policy(self, tl_cfg: Dict) -> List[str]:
        """Congela/descongela blocos conforme `transfer_learning`. Retorna os congelados."""
        model = self.ensure_model()
        groups = self.block_groups(model)
        enc, dec = groups["encoder"], groups["decoder"]
        frozen_enc = list(tl_cfg.get("frozen_encoder_blocks") or [])
        train_dec = tl_cfg.get("trainable_decoder_blocks")
        train_dec = list(dec) if train_dec is None else list(train_dec)
        for names, universe, label in ((frozen_enc, enc, "encoder"), (train_dec, dec, "decoder"),
                                       (tl_cfg.get("unfreeze_order") or [], enc, "encoder (unfreeze_order)")):
            bad = [n for n in names if n not in universe]
            if bad:
                raise ValueError(
                    f"Blocos inexistentes no {label}: {bad}. Disponíveis: {universe}. "
                    "Ajuste transfer_learning.* no config.yaml."
                )
        frozen: List[str] = []
        children = dict(model.named_children())
        for name in enc:
            trainable = name not in frozen_enc
            for p in children[name].parameters():
                p.requires_grad = trainable
            if not trainable:
                frozen.append(name)
        for name in dec:
            trainable = name in train_dec
            for p in children[name].parameters():
                p.requires_grad = trainable
            if not trainable:
                frozen.append(name)
        return frozen

    def _frozen_names(self, model: nn.Module) -> List[str]:
        out = []
        for name, child in model.named_children():
            params = list(child.parameters())
            if params and not any(p.requires_grad for p in params):
                out.append(name)
        return out

    def fine_tune(self, train_ds: Dataset, val_ds: Optional[Dataset], tl_cfg: Dict,
                  seed: Optional[int] = None) -> FitResult:
        """Adapta um modelo já carregado (`load_checkpoint`) aos dados de treino do PH².

        Não recarrega nem reinicia pesos: o chamador é responsável por instanciar uma
        estratégia nova e carregar o checkpoint original antes de cada rodada.
        """
        if self.model is None:
            raise CheckpointError("fine_tune requer um modelo carregado (chame load_checkpoint antes).")
        seed = self.seg_cfg["seed"] if seed is None else seed
        seed_everything(seed)
        model = self.model
        for p in model.parameters():
            p.requires_grad = True
        self.apply_freeze_policy(tl_cfg)

        policy = tl_cfg.get("unfreeze_policy", "fixed")
        order = list(tl_cfg.get("unfreeze_order") or [])
        every = int(tl_cfg.get("unfreeze_every_n_epochs", 10))
        children = dict(model.named_children())

        def on_epoch_start(m: nn.Module, epoch: int) -> None:
            if policy != "gradual" or epoch == 1 or (epoch - 1) % every != 0:
                return
            for name in order:
                if name in self._frozen_names(m):
                    for p in children[name].parameters():
                        p.requires_grad = True
                    logger.info("Época %d: descongelando bloco %s", epoch, name)
                    return

        def on_train_mode(m: nn.Module) -> None:
            if not tl_cfg.get("freeze_bn_stats", True):
                return
            for name in self._frozen_names(m):
                for mod in children[name].modules():
                    if isinstance(mod, _BatchNorm):
                        mod.eval()

        cfg = self._fit_config(
            max_epochs=tl_cfg["max_epochs"], learning_rate=tl_cfg["learning_rate"],
            early_stopping=tl_cfg["early_stopping"], seed=seed,
        )
        return fit(model, train_ds, val_ds, cfg, self.device,
                   on_epoch_start=on_epoch_start, on_train_mode=on_train_mode)

    # ── predição ────────────────────────────────────────────────────────────
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """Probabilidade por pixel (H, W) na resolução original da imagem. Determinística."""
        model = self.ensure_model()
        model.eval()
        rgb = image.convert("RGB")
        tensor, _ = self.transform(train=False)(rgb, None)
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0).to(self.device))
            prob = torch.sigmoid(logits)
            prob = F.interpolate(prob, size=(rgb.height, rgb.width), mode="bilinear",
                                 align_corners=False)
        return prob[0, 0].cpu().numpy().astype(np.float32)

    def predict_mask(self, image: Image.Image, threshold: Optional[float] = None) -> np.ndarray:
        """Máscara binária (H, W) bool, com limiar fixo da configuração."""
        thr = self.seg_cfg["threshold"] if threshold is None else threshold
        return self.predict_proba(image) > thr

    # ── checkpoints ─────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> str:
        """Salva pesos + metadados (apenas tensores e strings, compatível com weights_only).

        Returns:
            SHA-256 do arquivo salvo.
        """
        model = self.ensure_model()
        payload = {
            "format_version": CHECKPOINT_FORMAT,
            "strategy": self.name,
            "strategy_params_json": json.dumps(self.params, sort_keys=True),
            "image_size": int(self.seg_cfg["image_size"]),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "weights_sha256": state_dict_sha256(model),
            "metadata_json": json.dumps(metadata or {}, default=str, sort_keys=True),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)
        return file_sha256(str(p))

    def load_checkpoint(self, path: str) -> Dict:
        """Carrega pesos. Aceita o formato novo e o `state_dict` cru legado (.pkl)."""
        p = Path(path)
        if not p.is_file():
            raise CheckpointError(f"Checkpoint não encontrado: {p}")
        try:
            obj = torch.load(p, map_location=self.device, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - mensagem única e acionável
            raise CheckpointError(f"Falha ao ler checkpoint {p}: {exc}") from exc

        if isinstance(obj, dict) and "state_dict" in obj and "format_version" in obj:
            if obj.get("strategy") != self.name:
                raise CheckpointError(
                    f"Checkpoint de estratégia '{obj.get('strategy')}' incompatível com '{self.name}'.")
            if json.loads(obj["strategy_params_json"]) != self.params:
                raise CheckpointError(
                    "strategy_params do checkpoint difere da configuração atual: "
                    f"{obj['strategy_params_json']} vs {json.dumps(self.params, sort_keys=True)}")
            state, meta, legacy = obj["state_dict"], json.loads(obj["metadata_json"]), False
        elif isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
            state, meta, legacy = obj, {}, True
            logger.info("Checkpoint legado (state_dict cru) carregado de %s", p)
        else:
            raise CheckpointError(f"Formato de checkpoint não reconhecido em {p}")

        model = self.ensure_model()
        try:
            model.load_state_dict(state)
        except RuntimeError as exc:
            raise CheckpointError(f"state_dict incompatível com a arquitetura: {exc}") from exc
        model.to(self.device)
        return {"legacy": legacy, "metadata": meta, "file_sha256": file_sha256(str(p)),
                "weights_sha256": state_dict_sha256(model)}
