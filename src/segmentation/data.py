"""Registros imagem/máscara, pareamento por identificador, splits e transformações.

Não reutiliza `PH2Dataset` por dois motivos: (i) ele acopla máscara e imagem à
estrutura nativa de diretórios, e (ii) seu rótulo de melanoma usa a coluna
*Histological Diagnosis*, vazia em 159 das 200 linhas de PH2_dataset.txt
(28 positivos em vez dos 40 do protocolo). Aqui o rótulo vem de
*Clinical Diagnosis* e a discrepância é registrada explicitamente.
"""
import csv
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

PH2_CLINICAL_NAMES = {0: "common_nevus", 1: "atypical_nevus", 2: "melanoma"}


class DataError(ValueError):
    """Problema de integridade nos dados (pareamento, rótulos, vazamento)."""


@dataclass(frozen=True)
class Record:
    image_id: str
    image_path: str
    mask_path: Optional[str] = None
    label: Optional[int] = None        # 1 = melanoma, 0 = demais (PH2)
    diagnosis: Optional[str] = None    # classe diagnóstica nominal (PH2)
    group: Optional[str] = None        # identificador da lesão (ISIC), se conhecido

    def to_dict(self) -> Dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Leitura de imagens / máscaras
# ─────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def load_mask_bool(path: str, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Carrega máscara binária (H, W) bool. `size` = (W, H); usa nearest-neighbor."""
    with Image.open(path) as im:
        m = im.convert("L")
    if size is not None and m.size != tuple(size):
        m = m.resize(tuple(size), Image.NEAREST)
    return np.asarray(m) > 127


def save_mask_png(mask: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8)) * 255, mode="L").save(path)


# ─────────────────────────────────────────────────────────────────────────────
# ISIC 2018 Task 1: pareamento seguro
# ─────────────────────────────────────────────────────────────────────────────

def _index_dir(directory: Path, extensions: Sequence[str], strip_suffix: str = "") -> Dict[str, str]:
    exts = {e.lower() for e in extensions}
    out: Dict[str, str] = {}
    for p in sorted(directory.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if strip_suffix:
            if not stem.endswith(strip_suffix):
                continue
            stem = stem[: -len(strip_suffix)]
        if stem in out:
            raise DataError(f"Identificador duplicado '{stem}' em {directory}: {out[stem]} e {p}")
        out[stem] = str(p)
    return out


def pair_images_and_masks(
    image_dir: str,
    mask_dir: str,
    mask_suffix: str = "_segmentation",
    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
    mask_extensions: Sequence[str] = (".png",),
    allow_unpaired: bool = False,
) -> List[Record]:
    """Pareia imagens e máscaras estritamente por identificador.

    Args:
        image_dir: diretório das imagens (`ISIC_0000000.jpg`).
        mask_dir: diretório das máscaras (`ISIC_0000000_segmentation.png`).
        mask_suffix: sufixo removido do nome da máscara para obter o identificador.
        allow_unpaired: se False, qualquer imagem sem máscara (ou o inverso) é erro.

    Returns:
        Registros ordenados por identificador.
    """
    image_dir_p, mask_dir_p = Path(image_dir), Path(mask_dir)
    for label, d in (("image_dir", image_dir_p), ("mask_dir", mask_dir_p)):
        if not d.is_dir():
            raise DataError(f"{label} não encontrado: {d}")

    images = _index_dir(image_dir_p, image_extensions)
    masks = _index_dir(mask_dir_p, mask_extensions, strip_suffix=mask_suffix)
    only_img = sorted(set(images) - set(masks))
    only_mask = sorted(set(masks) - set(images))
    if (only_img or only_mask) and not allow_unpaired:
        raise DataError(
            f"Pareamento imagem/máscara inconsistente: {len(only_img)} imagens sem máscara "
            f"(ex.: {only_img[:3]}) e {len(only_mask)} máscaras sem imagem (ex.: {only_mask[:3]}). "
            "Corrija os diretórios ou defina datasets.isic2018.allow_unpaired: true."
        )
    if only_img or only_mask:
        logger.warning("Descartando %d imagens e %d máscaras sem par.", len(only_img), len(only_mask))
    ids = sorted(set(images) & set(masks))
    if not ids:
        raise DataError(f"Nenhum par imagem/máscara encontrado em {image_dir} / {mask_dir}.")
    return [Record(image_id=i, image_path=images[i], mask_path=masks[i]) for i in ids]


def read_id_list(path: str) -> Set[str]:
    """Lê identificadores (um por linha, ou primeira coluna de CSV). Ignora extensão e #."""
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip().split(",")[0].strip()
            if not tok or tok.startswith("#") or tok.lower() in {"image", "image_id", "id"}:
                continue
            ids.add(Path(tok).stem)
    return ids


def read_lesion_groups(path: str) -> Dict[str, str]:
    """CSV com colunas `image_id` e `lesion_id` -> {image_id: lesion_id}."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if not {"image_id", "lesion_id"} <= cols:
            raise DataError(f"{path} deve conter as colunas image_id e lesion_id (encontradas: {sorted(cols)})")
        return {Path(r["image_id"]).stem: r["lesion_id"] for r in reader}


def apply_groups_and_exclusions(
    records: List[Record],
    groups: Optional[Dict[str, str]] = None,
    exclude_ids: Optional[Set[str]] = None,
) -> Tuple[List[Record], Dict]:
    """Anexa `group` (lesion_id) e remove ids excluídos. Retorna (registros, resumo)."""
    exclude_ids = exclude_ids or set()
    kept, dropped = [], []
    for r in records:
        if r.image_id in exclude_ids:
            dropped.append(r.image_id)
            continue
        kept.append(Record(**{**r.to_dict(), "group": (groups or {}).get(r.image_id)}))
    summary = {
        "n_input": len(records),
        "n_excluded_overlap": len(dropped),
        "n_with_group": sum(1 for r in kept if r.group is not None),
        "group_info_available": groups is not None,
    }
    if groups is None:
        logger.warning(
            "Sem metadados de lesão (lesion_groups_csv): não é possível garantir que imagens da "
            "mesma lesão fiquem do mesmo lado do split treino/validação."
        )
    return kept, summary


def split_train_val(
    records: Sequence[Record], val_fraction: float, seed: int
) -> Tuple[List[Record], List[Record]]:
    """Split treino/validação sem vazamento entre imagens da mesma lesão.

    Grupos = `group` quando conhecido; caso contrário, cada imagem é seu próprio grupo.
    """
    if len(records) < 2:
        raise DataError("São necessários ao menos 2 registros para separar treino/validação.")
    group_ids = np.array([r.group if r.group is not None else f"__img__{r.image_id}" for r in records])
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(records)), groups=group_ids))
    train = [records[i] for i in sorted(tr_idx)]
    val = [records[i] for i in sorted(va_idx)]
    assert_no_group_leak(train, val)
    return train, val


def assert_no_group_leak(a: Sequence[Record], b: Sequence[Record]) -> None:
    ids_a, ids_b = {r.image_id for r in a}, {r.image_id for r in b}
    if ids_a & ids_b:
        raise DataError(f"Imagens presentes em ambos os conjuntos: {sorted(ids_a & ids_b)[:5]}")
    g_a = {r.group for r in a if r.group is not None}
    g_b = {r.group for r in b if r.group is not None}
    if g_a & g_b:
        raise DataError(f"Lesões presentes em ambos os conjuntos: {sorted(g_a & g_b)[:5]}")


# ─────────────────────────────────────────────────────────────────────────────
# PH2: descoberta e rótulos
# ─────────────────────────────────────────────────────────────────────────────

_PH2_ID = re.compile(r"^IMD\d+$")


def parse_ph2_metadata(txt_path: str) -> Dict[str, Dict]:
    """Lê PH2_dataset.txt (delimitado por '||') -> {id: {clinical: int, histological: str}}."""
    out: Dict[str, Dict] = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.split("||")]
            if len(parts) < 4 or not _PH2_ID.match(parts[1]):
                continue
            try:
                clinical = int(parts[3])
            except ValueError:
                continue
            out[parts[1]] = {"clinical": clinical, "histological": parts[2]}
    if not out:
        raise DataError(f"Nenhuma linha IMD* válida em {txt_path}")
    return out


def discover_ph2(
    dataset_dir: str,
    metadata_txt: str,
    mask_dir: Optional[str] = None,
    label_source: str = "clinical_diagnosis",
) -> Tuple[List[Record], Dict]:
    """Descobre as imagens do PH² e associa máscara de referência e rótulo.

    Returns:
        (registros ordenados por id, resumo com contagens e discrepância de rótulos).
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise DataError(f"Diretório do PH² não encontrado: {root}")
    meta = parse_ph2_metadata(metadata_txt)
    mask_root = Path(mask_dir) if mask_dir else None
    if mask_root is not None and not mask_root.is_dir():
        raise DataError(f"Diretório de máscaras do PH² não encontrado: {mask_root}")

    records: List[Record] = []
    problems: List[str] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        name = d.name
        image = d / f"{name}_Dermoscopic_Image" / f"{name}.bmp"
        if mask_root is None:
            candidates = [d / f"{name}_lesion" / f"{name}_lesion.bmp"]
        else:
            candidates = [mask_root / f"{name}_lesion.bmp", mask_root / f"{name}.bmp",
                          mask_root / name / f"{name}_lesion" / f"{name}_lesion.bmp"]
        mask = next((c for c in candidates if c.exists()), None)
        if not image.exists() or mask is None:
            problems.append(name)
            continue
        if name not in meta:
            problems.append(f"{name} (sem metadados)")
            continue
        clinical = meta[name]["clinical"]
        if label_source == "clinical_diagnosis":
            label = int(clinical == 2)
        else:
            label = int("melanoma" in meta[name]["histological"].lower())
        records.append(Record(
            image_id=name, image_path=str(image), mask_path=str(mask), label=label,
            diagnosis=PH2_CLINICAL_NAMES.get(clinical, f"clinical_{clinical}"),
        ))
    if problems:
        raise DataError(f"PH²: {len(problems)} casos incompletos/sem metadados: {problems[:5]}")
    if not records:
        raise DataError(f"Nenhum caso do PH² encontrado em {root}")

    legacy_pos = {n for n, m in meta.items() if "melanoma" in m["histological"].lower()}
    clinical_pos = {n for n, m in meta.items() if m["clinical"] == 2}
    summary = {
        "n_images": len(records),
        "n_melanoma": sum(r.label for r in records),
        "label_source": label_source,
        "class_counts": {k: sum(1 for r in records if r.diagnosis == k)
                         for k in sorted({r.diagnosis for r in records})},
        "legacy_histological_rule_positives": len(legacy_pos & {r.image_id for r in records}),
        "clinical_rule_positives": len(clinical_pos & {r.image_id for r in records}),
        "label_discrepancy_ids": sorted((legacy_pos ^ clinical_pos) & {r.image_id for r in records}),
    }
    return records, summary


# ─────────────────────────────────────────────────────────────────────────────
# Transformações sincronizadas imagem/máscara + Dataset
# ─────────────────────────────────────────────────────────────────────────────

class JointTransform:
    """Transformação conjunta: interpolação contínua na imagem, nearest na máscara.

    Parâmetros aleatórios são sorteados uma única vez e aplicados aos dois. Augmentations
    só atuam com `train=True`. O sorteio usa o RNG do torch (semeado pelo DataLoader).
    """

    def __init__(self, size: int, train: bool = False, augmentation: Optional[Dict] = None,
                 normalize: Optional[Dict] = None):
        self.size = int(size)
        self.train = train
        self.aug = augmentation or {}
        self.normalize = normalize

    def _flip_and_rotate(self, image, mask):
        if self.aug.get("hflip") and torch.rand(1).item() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.aug.get("vflip") and torch.rand(1).item() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)
        deg = float(self.aug.get("rotation_degrees", 0) or 0)
        if deg > 0:
            angle = (torch.rand(1).item() * 2 - 1) * deg
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return image, mask

    def __call__(self, image: Image.Image, mask: Optional[Image.Image] = None):
        size = [self.size, self.size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        if mask is not None:
            mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        if self.train and mask is not None:
            image, mask = self._flip_and_rotate(image, mask)
            jitter = float(self.aug.get("color_jitter", 0) or 0)
            if jitter > 0:
                b = 1 + (torch.rand(1).item() * 2 - 1) * jitter
                c = 1 + (torch.rand(1).item() * 2 - 1) * jitter
                image = TF.adjust_contrast(TF.adjust_brightness(image, b), c)
        img_t = TF.to_tensor(image)
        if self.normalize:
            img_t = TF.normalize(img_t, self.normalize["mean"], self.normalize["std"])
        if mask is None:
            return img_t, None
        mask_t = (TF.to_tensor(mask) > 0.5).float()
        return img_t, mask_t


class SegmentationDataset(Dataset):
    """Dataset PyTorch de pares (imagem, máscara) a partir de `Record`s."""

    def __init__(self, records: Sequence[Record], transform: JointTransform):
        missing = [r.image_id for r in records if r.mask_path is None]
        if missing:
            raise DataError(f"Registros sem máscara em dataset supervisionado: {missing[:5]}")
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        with Image.open(r.image_path) as im:
            image = im.convert("RGB")
        with Image.open(r.mask_path) as m:
            mask = m.convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        return self.transform(image, mask)
