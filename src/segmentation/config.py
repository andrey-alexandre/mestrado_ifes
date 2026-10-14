"""Configuração do experimento de segmentação (seção `segmentation_experiment`).

A seção legada `segmentation` do config.yaml (inferência com U-Net) permanece
intocada. Os parâmetros do experimento pós-qualificação vivem em
`segmentation_experiment`, mesclados sobre os defaults abaixo e validados antes
de qualquer etapa. Caminhos `null` só falham quando a etapa que os usa é executada.
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)

SECTION = "segmentation_experiment"


class ConfigError(ValueError):
    """Configuração inválida ou campo obrigatório ausente."""


DEFAULTS: Dict[str, Any] = {
    "datasets": {
        "isic2018": {
            "image_dir": None,          # ISIC2018_Task1-2_Training_Input
            "mask_dir": None,           # ISIC2018_Task1_Training_GroundTruth
            "mask_suffix": "_segmentation",
            "image_extensions": [".jpg", ".jpeg", ".png"],
            "mask_extensions": [".png"],
            "allow_unpaired": False,    # False: imagem sem máscara (ou vice-versa) é erro
            "lesion_groups_csv": None,  # CSV opcional: image_id,lesion_id (evita vazamento)
            "overlap_ids_file": None,   # ids a excluir do treino (sobreposição entre datasets)
            "val_fraction": 0.15,
        },
        "ph2": {
            "dataset_dir": None,        # null -> paths.ph2_dataset_dir
            "mask_dir": None,           # null -> estrutura nativa {NAME}/{NAME}_lesion/
            "metadata_txt": None,       # null -> paths.ph2_metadata_txt
            "label_source": "clinical_diagnosis",  # clinical_diagnosis | histological_diagnosis
        },
    },
    "segmentation": {
        "strategy": "unet",
        "strategy_params": {"model_name": "U_Net", "img_ch": 3, "output_ch": 1},
        "weights_init": "random",       # random | checkpoint
        "init_checkpoint": None,
        "image_size": 256,              # múltiplo de 16 para a U-Net
        "normalize": None,              # null (apenas /255, como o modelo legado) | {mean, std}
        "batch_size": 8,
        "learning_rate": 5.0e-4,
        "weight_decay": 0.0,
        "momentum": 0.9,                # apenas SGD
        "max_epochs": 100,
        "optimizer": "adam",            # adam | adamw | sgd
        "scheduler": {"name": "none", "step_size": 10, "gamma": 0.1, "min_lr": 1.0e-6},
        "loss": "bce_dice",             # bce | dice | bce_dice
        "early_stopping": {"enabled": True, "patience": 15, "min_delta": 0.0,
                           "monitor": "val_dice"},   # val_dice | val_loss
        "threshold": 0.5,               # fixado a priori; nunca ajustado no PH2
        "seed": 42,
        "device": "cpu",                # cpu | cuda | mps | auto
        "mixed_precision": False,
        "num_workers": 0,
        "augmentation": {
            "hflip": True, "vflip": True,
            "rotation_degrees": 30, "color_jitter": 0.1,
        },
    },
    "transfer_learning": {
        "enabled": True,
        "frozen_encoder_blocks": ["Conv1", "Conv2", "Conv3"],
        "trainable_decoder_blocks": None,   # null -> todo o decoder treinável
        "freeze_bn_stats": True,
        "unfreeze_policy": "fixed",         # fixed | gradual
        "unfreeze_order": ["Conv3", "Conv2", "Conv1"],
        "unfreeze_every_n_epochs": 10,
        "learning_rate": 1.0e-4,
        "max_epochs": 40,
        "inner_val_fraction": 0.15,         # validação interna aos 4 folds de treino
        "early_stopping": {"enabled": True, "patience": 10, "min_delta": 0.0,
                           "monitor": "val_dice"},
        "final_epochs": "median_cv_best_epoch",  # ou inteiro
    },
    "evaluation": {
        "n_folds": 5,
        "stratify_by": "binary_melanoma",   # binary_melanoma | clinical_diagnosis
        "primary_metric": "dice",           # dice | iou | hd95
        "secondary_metrics": ["iou", "hd95"],
        "selection_rule": "higher_mean",    # higher_mean | mean_with_ci
        "bootstrap": {"n_resamples": 10000, "confidence": 0.95, "seed": 42},
        "n_examples": 4,                    # melhores/piores por estratégia
    },
    "multiagent": {
        "run_vlm": False,                   # False: apenas avaliação de segmentação
        "empty_mask_policy": "flag_and_skip",  # flag_and_skip | full_image
        "feature_extractor": "src.features.extractor:build_feature_package",
        "vlm_results_prefix": "vlm_result",
    },
    "output": {
        "root_dir": "experiments/segmentation",
        "checkpoints_dir": None,            # null -> {root_dir}/checkpoints
        "predictions_dir": None,            # null -> {root_dir}/predictions
        "results_dir": None,                # null -> {root_dir}/results
        "reports_dir": None,                # null -> {root_dir}/reports
        "folds_manifest": None,             # null -> {results_dir}/folds_manifest.json
    },
}

_CHOICES = {
    ("segmentation", "weights_init"): {"random", "checkpoint"},
    ("segmentation", "optimizer"): {"adam", "adamw", "sgd"},
    ("segmentation", "loss"): {"bce", "dice", "bce_dice"},
    ("segmentation", "device"): {"cpu", "cuda", "mps", "auto"},
    ("transfer_learning", "unfreeze_policy"): {"fixed", "gradual"},
    ("evaluation", "stratify_by"): {"binary_melanoma", "clinical_diagnosis"},
    ("evaluation", "primary_metric"): {"dice", "iou", "hd95"},
    ("evaluation", "selection_rule"): {"higher_mean", "mean_with_ci"},
    ("datasets", "ph2", "label_source"): {"clinical_diagnosis", "histological_diagnosis"},
    ("multiagent", "empty_mask_policy"): {"flag_and_skip", "full_image"},
}


def _deep_merge(base: Dict, override: Optional[Dict]) -> Dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _get(cfg: Dict, dotted: str) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(full_cfg: Dict) -> Dict:
    """Extrai a seção do experimento, mescla defaults, resolve fallbacks e valida.

    Args:
        full_cfg: dicionário completo do config.yaml.

    Returns:
        Dicionário resolvido. `cfg["paths"]` contém o `paths` legado para consulta.
    """
    if not isinstance(full_cfg, dict):
        raise ConfigError("config.yaml inválido: esperado um mapeamento no topo.")
    unknown = set(full_cfg.get(SECTION, {}) or {}) - set(DEFAULTS)
    if unknown:
        raise ConfigError(f"Chaves desconhecidas em '{SECTION}': {sorted(unknown)}")

    cfg = _deep_merge(DEFAULTS, full_cfg.get(SECTION))
    legacy_paths = full_cfg.get("paths", {}) or {}

    ph2 = cfg["datasets"]["ph2"]
    if ph2["dataset_dir"] is None:
        ph2["dataset_dir"] = legacy_paths.get("ph2_dataset_dir")
    if ph2["metadata_txt"] is None:
        ph2["metadata_txt"] = legacy_paths.get("ph2_metadata_txt")

    out = cfg["output"]
    root = out["root_dir"]
    for key, sub in (("checkpoints_dir", "checkpoints"), ("predictions_dir", "predictions"),
                     ("results_dir", "results"), ("reports_dir", "reports")):
        if out[key] is None:
            out[key] = str(Path(root) / sub)
    if out["folds_manifest"] is None:
        out["folds_manifest"] = str(Path(out["results_dir"]) / "folds_manifest.json")

    cfg["paths"] = legacy_paths
    cfg["_llm"] = full_cfg.get("llm", {}) or {}
    cfg["_output_legacy"] = full_cfg.get("output", {}) or {}
    cfg["_feature_extraction"] = full_cfg.get("feature_extraction", {}) or {}
    cfg["_pipeline"] = full_cfg.get("pipeline", {}) or {}
    cfg["_rag"] = full_cfg.get("rag", {}) or {}
    validate(cfg)
    return cfg


def validate(cfg: Dict) -> None:
    """Valida tipos, intervalos e escolhas. Levanta ConfigError com todas as falhas."""
    errors: List[str] = []

    for path, allowed in _CHOICES.items():
        val = _get(cfg, ".".join(path))
        if val not in allowed:
            errors.append(f"{SECTION}.{'.'.join(path)}={val!r} inválido; use um de {sorted(allowed)}")

    seg = cfg["segmentation"]
    size = seg["image_size"]
    if not isinstance(size, int) or size <= 0:
        errors.append(f"{SECTION}.segmentation.image_size deve ser inteiro positivo (recebido {size!r})")
    for key in ("batch_size", "max_epochs"):
        if not isinstance(seg[key], int) or seg[key] <= 0:
            errors.append(f"{SECTION}.segmentation.{key} deve ser inteiro positivo")
    if not (0.0 < float(seg["threshold"]) < 1.0):
        errors.append(f"{SECTION}.segmentation.threshold deve estar em (0, 1)")
    if float(seg["learning_rate"]) <= 0:
        errors.append(f"{SECTION}.segmentation.learning_rate deve ser > 0")
    if seg["weights_init"] == "checkpoint" and not seg["init_checkpoint"]:
        errors.append(f"{SECTION}.segmentation.init_checkpoint é obrigatório com weights_init='checkpoint'")
    if seg["scheduler"].get("name") not in {"none", "cosine", "step", "plateau"}:
        errors.append(f"{SECTION}.segmentation.scheduler.name inválido: {seg['scheduler'].get('name')!r}")
    if seg["early_stopping"].get("monitor") not in {"val_dice", "val_loss"}:
        errors.append(f"{SECTION}.segmentation.early_stopping.monitor deve ser val_dice|val_loss")

    isic = cfg["datasets"]["isic2018"]
    for key in ("val_fraction",):
        if not (0.0 < float(isic[key]) < 0.9):
            errors.append(f"{SECTION}.datasets.isic2018.{key} deve estar em (0, 0.9)")

    tl = cfg["transfer_learning"]
    if not (0.0 < float(tl["inner_val_fraction"]) < 0.9):
        errors.append(f"{SECTION}.transfer_learning.inner_val_fraction deve estar em (0, 0.9)")
    if tl["max_epochs"] <= 0 or float(tl["learning_rate"]) <= 0:
        errors.append(f"{SECTION}.transfer_learning.max_epochs e learning_rate devem ser > 0")
    fe = tl["final_epochs"]
    if not (fe == "median_cv_best_epoch" or (isinstance(fe, int) and fe > 0)):
        errors.append(f"{SECTION}.transfer_learning.final_epochs deve ser 'median_cv_best_epoch' ou inteiro > 0")

    ev = cfg["evaluation"]
    if ev["n_folds"] < 2:
        errors.append(f"{SECTION}.evaluation.n_folds deve ser >= 2")
    if ev["primary_metric"] in ev["secondary_metrics"]:
        errors.append(f"{SECTION}.evaluation.secondary_metrics não deve repetir a métrica primária")
    boot = ev["bootstrap"]
    if boot["n_resamples"] < 100 or not (0.5 < boot["confidence"] < 1.0):
        errors.append(f"{SECTION}.evaluation.bootstrap: n_resamples >= 100 e confidence em (0.5, 1)")

    if errors:
        raise ConfigError("Configuração inválida:\n  - " + "\n  - ".join(errors))


def require(cfg: Dict, keys: Iterable[str], stage: str) -> None:
    """Garante que campos obrigatórios da etapa estejam preenchidos.

    Args:
        cfg: configuração resolvida (saída de build_config).
        keys: caminhos pontuados relativos à seção do experimento
              (ex.: "datasets.isic2018.image_dir").
        stage: nome da etapa, usado na mensagem.
    """
    missing = [k for k in keys if _get(cfg, k) in (None, "", [])]
    if missing:
        listed = "\n  - ".join(f"{SECTION}.{k}" for k in missing)
        raise ConfigError(
            f"Etapa '{stage}' requer campos ausentes no config.yaml:\n  - {listed}\n"
            "Preencha-os (ou, para o PH², defina paths.ph2_dataset_dir / paths.ph2_metadata_txt)."
        )


def require_existing_dir(path: Optional[str], label: str) -> Path:
    if not path:
        raise ConfigError(f"{label} não configurado.")
    p = Path(path)
    if not p.is_dir():
        raise ConfigError(f"{label} não encontrado ou não é diretório: {p}")
    return p
