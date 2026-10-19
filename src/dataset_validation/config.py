"""Configuração do experimento (seção `dataset_validation` de config/config.yaml).

Os valores da seção são mesclados sobre os defaults abaixo. Caminhos relativos são
resolvidos a partir da raiz do repositório. `null` em caminhos de dataset só falha
quando a etapa que o utiliza é executada.
"""
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

SECTION = "dataset_validation"
ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Configuração inválida ou campo obrigatório ausente."""


DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "sample": {"n_per_class": 20},
    "datasets": {
        "ph2": {
            "dataset_dir": None,        # null -> paths.ph2_dataset_dir
            "metadata_txt": None,       # null -> paths.ph2_metadata_txt
            "melanoma_clinical_codes": [2],
            "nevus_clinical_codes": [0, 1],   # 0 = nevo comum, 1 = nevo atípico/displásico
        },
        "ham10000": {
            "image_dir": None,          # {image_id}.jpg (HAM10000 ou ISIC 2019 Training Input)
            "metadata_csv": None,       # HAM10000_metadata (lesion_id,image_id,dx,dx_type,...)
            "mask_dir": None,           # {image_id}_segmentation.png já extraídas (opcional)
            "mask_zip": None,           # zip com as máscaras (extrai somente as amostradas)
            "mask_zip_prefix": "HAM10000_segmentations_lesion_tschandl/",
            "mask_suffix": "_segmentation",
            "melanoma_dx": ["mel"],
            "nevus_dx": ["nv"],
            "one_image_per_lesion": True,
        },
        "isic2019": {                   # apenas auditoria documental (sem inferência)
            "image_dir": None,          # ISIC_2019_Training_Input
            "isic2018_task1_zip": None, # ISIC2018_Task1-2_Training_Input.zip
            "ground_truth_csv": None,   # ISIC_2019_Training_GroundTruth.csv (ausente = registrado)
            "metadata_csv": None,       # ISIC_2019_Training_Metadata.csv (ausente = registrado)
        },
    },
    "quality": {
        "lesion_interior_erosion_px": 5,
        "boundary_band_px": 4,
        "skin_ring_dilation_px": 15,
        "hair_blackhat_size": 9,
        "hair_threshold": 15,
        "reference_width_px": 512,
        "corner_fraction": 0.1,
    },
    "inference": {
        "models": ["gpt-5.4-nano", "gemini-3.1-flash-lite"],
        "temperature": None,            # null -> llm.temperature do config
        "segmentation_mode": "precomputed",
        "results_prefix": "vlm_validation",
        "calls_per_image": 4,           # ABCD + Menzies + SPCL + sumarizador
        "assumptions": {                # estimativa de tokens (premissas, não medições)
            "input_tokens_per_diagnostic_call": 1700,
            "input_tokens_summary_call": 3500,
            "output_tokens_per_diagnostic_call": 350,
            "output_tokens_summary_call": 450,
            "price_per_1m_input_usd": None,    # preencher com o preço vigente do provedor
            "price_per_1m_output_usd": None,
        },
    },
    "evaluation": {
        "confidence": 0.95,
        "bootstrap": {"n_resamples": 10000, "seed": 42},
        "primary_policy": "indeterminate_as_positive",
        "positive_class": "melanoma",
    },
    "legacy": {
        "ph2_results_json": {
            "gpt-5.4-nano": "llm_result_gpt-5.4-nano_20260609.json",
            "gemini-3.1-flash-lite": "llm_result_gemini-3.1-flash-lite_20260609.json",
        },
        "ph2_estimated_csv": {
            "gpt-5.4-nano": "llm_result_gpt-5.4-nano_20260606.csv",
            "gemini-3.1-flash-lite": "llm_result_gemini-3.1-flash-lite_20260606.csv",
        },
    },
    "output": {"root_dir": "experiments/dataset_validation"},
    "allclasses": {                     # extensão HAM10000: melanoma × todas as classes
        "n_per_class": 20,
        "output_root": "experiments/dataset_validation_allclasses",
        "reuse_results_dir": "experiments/dataset_validation/inference",   # piloto mel/nv já inferido
    },
}


def _merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: Optional[str], base: Optional[Path] = None) -> Optional[str]:
    """Resolve caminhos relativos a partir da raiz do repositório."""
    if p is None:
        return None
    q = Path(p)
    if not q.is_absolute():
        q = (base or ROOT) / q
    return str(q)


def build_config(full: Dict, overrides: Optional[Dict] = None) -> Dict:
    """Mescla a seção sobre os defaults, aplica overrides e resolve caminhos."""
    cfg = _merge(DEFAULTS, full.get(SECTION) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    paths = full.get("paths", {}) or {}
    ph2 = cfg["datasets"]["ph2"]
    ph2["dataset_dir"] = ph2["dataset_dir"] or paths.get("ph2_dataset_dir")
    ph2["metadata_txt"] = ph2["metadata_txt"] or paths.get("ph2_metadata_txt")
    for key in ("dataset_dir", "metadata_txt"):
        ph2[key] = resolve_path(ph2[key])
    ham = cfg["datasets"]["ham10000"]
    for key in ("image_dir", "metadata_csv", "mask_dir", "mask_zip"):
        ham[key] = resolve_path(ham[key])
    isic = cfg["datasets"]["isic2019"]
    for key in ("image_dir", "isic2018_task1_zip", "ground_truth_csv", "metadata_csv"):
        isic[key] = resolve_path(isic[key])
    cfg["output"]["root_dir"] = resolve_path(cfg["output"]["root_dir"])
    ac = cfg["allclasses"]
    ac["output_root"] = resolve_path(ac["output_root"])
    ac["reuse_results_dir"] = resolve_path(ac["reuse_results_dir"])
    llm = full.get("llm", {}) or {}
    if cfg["inference"]["temperature"] is None:
        cfg["inference"]["temperature"] = llm.get("temperature", 0)
    cfg["_llm_default_model"] = llm.get("model_name")
    cfg["_config_sections"] = {"llm": llm, "paths": paths}
    validate(cfg)
    return cfg


def load_config(path: str = "config/config.yaml", overrides: Optional[Dict] = None) -> Dict:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / path
    with open(p, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    cfg = build_config(full, overrides)
    cfg["_config_path"] = str(p)
    return cfg


def validate(cfg: Dict) -> None:
    n = cfg["sample"]["n_per_class"]
    if not isinstance(n, int) or n <= 0:
        raise ConfigError("sample.n_per_class deve ser inteiro positivo.")
    if not isinstance(cfg["seed"], int):
        raise ConfigError("seed deve ser inteiro (semente fixa obrigatória).")
    if cfg["evaluation"]["primary_policy"] not in {"indeterminate_as_positive", "indeterminate_as_negative"}:
        raise ConfigError("evaluation.primary_policy inválida.")
    if cfg["inference"]["segmentation_mode"] != "precomputed":
        raise ConfigError("O piloto usa segmentação 'precomputed' (máscaras de referência) nos dois datasets.")
    ph2 = cfg["datasets"]["ph2"]
    if set(ph2["melanoma_clinical_codes"]) & set(ph2["nevus_clinical_codes"]):
        raise ConfigError("Códigos clínicos de melanoma e nevo do PH² não podem se sobrepor.")
    ham = cfg["datasets"]["ham10000"]
    if set(ham["melanoma_dx"]) & set(ham["nevus_dx"]):
        raise ConfigError("Classes dx de melanoma e nevo do HAM10000 não podem se sobrepor.")


def output_dirs(cfg: Dict) -> Dict[str, Path]:
    root = Path(cfg["output"]["root_dir"])
    return {
        "root": root,
        "manifest": root / "manifest",
        "audit": root / "audit",
        "quality": root / "quality",
        "inference": root / "inference",
        "evaluation": root / "evaluation",
        "reports": root / "reports",
        "inputs": root / "inputs",
    }
