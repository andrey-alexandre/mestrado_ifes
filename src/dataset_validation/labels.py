"""Rótulos originais e classificação binária melanoma × nevo (PH² e HAM10000).

PH²: `Clinical Diagnosis` (0 = nevo comum, 1 = nevo atípico, 2 = melanoma) é a fonte
usada aqui. A regra legada de `src/data/dataset.py` (substring "melanoma" em
`Histological Diagnosis`) é reproduzida apenas para auditoria, sem alterar o módulo.
"""
import csv
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MELANOMA, NEVUS = "melanoma", "nevus"
BINARY = {MELANOMA: 1, NEVUS: 0}


class LabelError(ValueError):
    """Rótulo ausente, inconsistente ou fora do mapeamento esperado."""


# ── PH² ────────────────────────────────────────────────────────────────────────

def parse_ph2_metadata(path: str) -> Dict[str, Dict]:
    """Lê PH2_dataset.txt (delimitador `||`) preservando todas as colunas."""
    records: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = [p.strip() for p in line.split("||")]
            if len(parts) < 6 or not parts[1]:
                continue
            ann = [a.strip() for a in parts[4].split("|")]
            if len(ann) != 6:
                raise LabelError(f"Linha PH² com anotações inesperadas: {line!r}")
            name = parts[1]
            if name in records:
                raise LabelError(f"Caso PH² duplicado no metadado: {name}")
            clin = parts[3].strip()
            records[name] = {
                "name": name,
                "histological_diagnosis": parts[2],
                "clinical_diagnosis": int(clin) if clin.lstrip("-").isdigit() else None,
                "asymmetry": ann[0], "pigment_network": ann[1], "dots_globules": ann[2],
                "streaks": ann[3], "regression_areas": ann[4], "blue_whitish_veil": ann[5],
                "colors": parts[5],
            }
    return records


def ph2_class(rec: Dict, cfg_ph2: Dict) -> Optional[str]:
    """Classe binária pela fonte clínica; None se o código está fora de ambos os grupos."""
    code = rec.get("clinical_diagnosis")
    if code in cfg_ph2["melanoma_clinical_codes"]:
        return MELANOMA
    if code in cfg_ph2["nevus_clinical_codes"]:
        return NEVUS
    return None


def ph2_legacy_label(rec: Dict) -> int:
    """Regra legada (src/data/dataset.py::PH2Dataset._load_metadata): 'melanoma' na histologia."""
    return 1 if "melanoma" in rec.get("histological_diagnosis", "").lower() else 0


def ph2_label_audit(records: Dict[str, Dict], cfg_ph2: Dict) -> Dict:
    """Tabula diagnóstico clínico × histológico e a divergência da regra legada."""
    table: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    legacy_neg_but_clinical_mel: List[Dict] = []
    for name, r in sorted(records.items()):
        hist = r["histological_diagnosis"] or "(vazio)"
        table[str(r["clinical_diagnosis"])][hist] += 1
        cls = ph2_class(r, cfg_ph2)
        if cls == MELANOMA and ph2_legacy_label(r) == 0:
            legacy_neg_but_clinical_mel.append({"name": name, "histological": r["histological_diagnosis"]})
    legacy_pos = sum(ph2_legacy_label(r) for r in records.values())
    clinical_pos = sum(1 for r in records.values() if ph2_class(r, cfg_ph2) == MELANOMA)
    hist_nonempty = sum(1 for r in records.values() if r["histological_diagnosis"])
    return {
        "n_cases": len(records),
        "clinical_counts": dict(collections.Counter(str(r["clinical_diagnosis"]) for r in records.values())),
        "clinical_x_histological": {k: dict(v) for k, v in sorted(table.items())},
        "histological_nonempty": hist_nonempty,
        "legacy_positive": legacy_pos,
        "clinical_melanoma": clinical_pos,
        "melanomas_labelled_negative_by_legacy_rule": legacy_neg_but_clinical_mel,
        "n_melanomas_labelled_negative_by_legacy_rule": len(legacy_neg_but_clinical_mel),
        "legacy_positive_not_clinical_melanoma": sorted(
            n for n, r in records.items() if ph2_legacy_label(r) == 1 and ph2_class(r, cfg_ph2) != MELANOMA
        ),
    }


def ph2_paths(dataset_dir: str, name: str) -> Tuple[str, str]:
    """(imagem, máscara) na estrutura nativa do PH²."""
    base = Path(dataset_dir) / name
    return (str(base / f"{name}_Dermoscopic_Image" / f"{name}.bmp"),
            str(base / f"{name}_lesion" / f"{name}_lesion.bmp"))


# ── HAM10000 ───────────────────────────────────────────────────────────────────

def parse_ham_metadata(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    need = {"lesion_id", "image_id", "dx", "dx_type", "dataset"}
    if not rows or not need <= set(rows[0]):
        raise LabelError(f"Metadado HAM10000 sem colunas {sorted(need)}: {path}")
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        dup = [k for k, c in collections.Counter(ids).items() if c > 1][:5]
        raise LabelError(f"image_id duplicado no metadado HAM10000: {dup}")
    return rows


def ham_class(row: Dict, cfg_ham: Dict) -> Optional[str]:
    if row["dx"] in cfg_ham["melanoma_dx"]:
        return MELANOMA
    if row["dx"] in cfg_ham["nevus_dx"]:
        return NEVUS
    return None


def ham_paths(cfg_ham: Dict, image_id: str, mask_cache_dir: Optional[Path] = None) -> Tuple[str, Optional[str]]:
    """(imagem, máscara-ou-None). A máscara pode estar em mask_dir, no cache ou apenas no zip."""
    image = str(Path(cfg_ham["image_dir"]) / f"{image_id}.jpg")
    name = f"{image_id}{cfg_ham['mask_suffix']}.png"
    for d in (cfg_ham.get("mask_dir"), mask_cache_dir):
        if d and (Path(d) / name).exists():
            return image, str(Path(d) / name)
    return image, None
