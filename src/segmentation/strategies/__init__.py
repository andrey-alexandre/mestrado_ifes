"""Registro de estratégias e Factory Function selecionada pelo config.yaml."""
from typing import Dict, Type

from src.segmentation.strategies.base import (  # noqa: F401
    CheckpointError, SegmentationStrategy, file_sha256, state_dict_sha256,
)
from src.segmentation.strategies.unet import UNetStrategy

_REGISTRY: Dict[str, Type[SegmentationStrategy]] = {UNetStrategy.name: UNetStrategy}


def register_strategy(cls: Type[SegmentationStrategy]) -> Type[SegmentationStrategy]:
    """Registra uma nova arquitetura sem alterar treino, avaliação ou inferência."""
    if not cls.name:
        raise ValueError("A estratégia precisa definir `name`.")
    _REGISTRY[cls.name] = cls
    return cls


def available_strategies():
    return sorted(_REGISTRY)


def build_strategy(cfg: Dict) -> SegmentationStrategy:
    """Factory: instancia a estratégia de `segmentation_experiment.segmentation.strategy`."""
    seg_cfg = cfg["segmentation"]
    name = seg_cfg["strategy"]
    if name not in _REGISTRY:
        raise ValueError(f"Estratégia de segmentação desconhecida: '{name}'. "
                         f"Disponíveis: {available_strategies()}")
    return _REGISTRY[name](seg_cfg)
