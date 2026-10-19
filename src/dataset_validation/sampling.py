"""Amostragem determinística e manifesto do piloto (sem seleção manual).

Cada fluxo (dataset × classe) usa `random.Random(f"{seed}:{stream}")`, cuja semente de
string é determinística entre execuções (não depende de PYTHONHASHSEED). O sorteio é feito
sobre o conjunto de unidades ordenado lexicograficamente.
"""
import collections
import random
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from src.dataset_validation import labels as L
from src.dataset_validation.io_utils import fingerprint, sha256_file

MANIFEST_FIELDS = [
    "sample_id", "dataset", "class_label", "binary_label", "original_label", "label_source",
    "image_id", "lesion_id", "image_path", "mask_path", "seed", "stream", "pool_size",
    "draw_rank", "dx_type", "acquisition_source", "ph2_nevus_subtype", "age", "sex",
    "localization", "image_sha256", "mask_sha256",
]


class SamplingError(RuntimeError):
    """Pool insuficiente ou manifesto inconsistente."""


def draw(units: Sequence[str], n: int, seed: int, stream: str) -> List[str]:
    """Sorteia `n` unidades distintas, na ordem do sorteio, de forma reprodutível."""
    pool = sorted(set(units))
    if len(pool) < n:
        raise SamplingError(f"Pool '{stream}' tem {len(pool)} unidades (< {n} solicitadas).")
    return random.Random(f"{seed}:{stream}").sample(pool, n)


def _ph2_rows(records: Dict[str, Dict], cfg: Dict) -> List[Dict]:
    ph2, n, seed = cfg["datasets"]["ph2"], cfg["sample"]["n_per_class"], cfg["seed"]
    rows: List[Dict] = []
    for cls in (L.MELANOMA, L.NEVUS):
        pool = [k for k, r in records.items() if L.ph2_class(r, ph2) == cls]
        stream = f"ph2:{cls}"
        for rank, name in enumerate(draw(pool, n, seed, stream), start=1):
            r = records[name]
            img, msk = L.ph2_paths(ph2["dataset_dir"], name)
            code = r["clinical_diagnosis"]
            rows.append({
                "sample_id": f"ph2:{name}", "dataset": "ph2", "class_label": cls,
                "binary_label": L.BINARY[cls], "original_label": f"clinical={code}",
                "label_source": "PH2_dataset.txt:Clinical Diagnosis",
                "image_id": name, "lesion_id": name, "image_path": img, "mask_path": msk,
                "seed": seed, "stream": stream, "pool_size": len(pool), "draw_rank": rank,
                "dx_type": "", "acquisition_source": "pedro_hispano_hospital",
                "ph2_nevus_subtype": {0: "common", 1: "atypical"}.get(code, "") if cls == L.NEVUS else "",
                "age": "", "sex": "", "localization": "",
            })
    return rows


def _ham_rows(meta: List[Dict], cfg: Dict, mask_resolver: Optional[Callable]) -> List[Dict]:
    ham, n, seed = cfg["datasets"]["ham10000"], cfg["sample"]["n_per_class"], cfg["seed"]
    by_id = {r["image_id"]: r for r in meta}
    rows: List[Dict] = []
    lesion_class: Dict[str, set] = collections.defaultdict(set)
    for r in meta:
        c = L.ham_class(r, ham)
        if c:
            lesion_class[r["lesion_id"]].add(c)
    mixed = [k for k, v in lesion_class.items() if len(v) > 1]
    if mixed:
        raise SamplingError(f"lesion_id com imagens em ambas as classes: {mixed[:5]}")
    for cls in (L.MELANOMA, L.NEVUS):
        pool_imgs = [r for r in meta if L.ham_class(r, ham) == cls]
        stream = f"ham10000:{cls}"
        if ham["one_image_per_lesion"]:
            by_lesion: Dict[str, List[str]] = collections.defaultdict(list)
            for r in pool_imgs:
                by_lesion[r["lesion_id"]].append(r["image_id"])
            lesions = draw(list(by_lesion), n, seed, stream)
            chosen = []
            for les in lesions:
                imgs = sorted(by_lesion[les])
                chosen.append(random.Random(f"{seed}:{stream}:{les}").choice(imgs))
            pool_size = len(by_lesion)
        else:
            chosen = draw([r["image_id"] for r in pool_imgs], n, seed, stream)
            pool_size = len(pool_imgs)
        for rank, iid in enumerate(chosen, start=1):
            r = by_id[iid]
            img, msk = L.ham_paths(ham, iid)
            if msk is None and mask_resolver:
                msk = mask_resolver(iid)
            rows.append({
                "sample_id": f"ham10000:{iid}", "dataset": "ham10000", "class_label": cls,
                "binary_label": L.BINARY[cls], "original_label": f"dx={r['dx']}",
                "label_source": "HAM10000_metadata:dx", "image_id": iid, "lesion_id": r["lesion_id"],
                "image_path": img, "mask_path": msk, "seed": seed, "stream": stream,
                "pool_size": pool_size, "draw_rank": rank, "dx_type": r["dx_type"],
                "acquisition_source": r["dataset"], "ph2_nevus_subtype": "",
                "age": r.get("age", ""), "sex": r.get("sex", ""),
                "localization": r.get("localization", ""),
            })
    return rows


def extract_ham_masks(cfg: Dict, image_ids: Sequence[str], out_dir: Path) -> Dict[str, str]:
    """Extrai do zip local SOMENTE as máscaras amostradas (sem download)."""
    ham = cfg["datasets"]["ham10000"]
    if not ham.get("mask_zip"):
        md = ham.get("mask_dir")
        expected = [f"{i}{ham['mask_suffix']}.png" for i in list(image_ids)[:3]]
        found = sorted(p.name for p in Path(md).iterdir())[:5] if md and Path(md).is_dir() else None
        raise SamplingError(
            f"{len(list(image_ids))} máscara(s) não encontrada(s) em mask_dir={md!r} "
            f"(existe: {bool(found is not None)}). Nome esperado, ex.: {expected}. "
            f"Primeiros arquivos na pasta: {found}. Verifique sufixo/extensão/subpasta ou informe mask_zip.")
    out_dir.mkdir(parents=True, exist_ok=True)
    found: Dict[str, str] = {}
    with zipfile.ZipFile(ham["mask_zip"]) as z:
        names = set(z.namelist())
        for iid in image_ids:
            member = f"{ham['mask_zip_prefix']}{iid}{ham['mask_suffix']}.png"
            if member not in names:
                raise SamplingError(f"Máscara ausente no zip: {member}")
            dest = out_dir / f"{iid}{ham['mask_suffix']}.png"
            if not dest.exists():
                dest.write_bytes(z.read(member))
            found[iid] = str(dest)
    return found


def build_manifest(cfg: Dict, ph2_records: Dict[str, Dict], ham_meta: List[Dict],
                   mask_cache_dir: Optional[Path] = None, add_hashes: bool = True) -> List[Dict]:
    """Constrói o manifesto (PH² e HAM10000) e valida a consistência."""
    rows = _ph2_rows(ph2_records, cfg)
    ham_rows = _ham_rows(ham_meta, cfg, mask_resolver=None)
    ham = cfg["datasets"]["ham10000"]
    missing = [r["image_id"] for r in ham_rows if r["mask_path"] is None]
    if missing:
        cache = mask_cache_dir or Path(cfg["output"]["root_dir"]) / "inputs" / "ham10000_masks"
        extracted = extract_ham_masks(cfg, missing, cache)
        for r in ham_rows:
            if r["mask_path"] is None:
                r["mask_path"] = extracted[r["image_id"]]
    rows += ham_rows
    if add_hashes:
        for r in rows:
            for key, col in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
                p = r[key]
                r[col] = sha256_file(p) if p and Path(p).exists() else ""
    validate_manifest(rows, cfg)
    return rows


def validate_manifest(rows: List[Dict], cfg: Dict, check_files: bool = True) -> None:
    """Falha (SamplingError) em duplicatas, contagens erradas ou rótulos inconsistentes."""
    n = cfg["sample"]["n_per_class"]
    ph2, ham = cfg["datasets"]["ph2"], cfg["datasets"]["ham10000"]
    errors: List[str] = []
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append("sample_id duplicado")
    for ds in ("ph2", "ham10000"):
        sub = [r for r in rows if r["dataset"] == ds]
        img = [r["image_id"] for r in sub]
        les = [r["lesion_id"] for r in sub]
        if len(img) != len(set(img)):
            errors.append(f"{ds}: image_id duplicado")
        if len(les) != len(set(les)):
            errors.append(f"{ds}: lesion_id duplicado (mesma lesão amostrada mais de uma vez)")
        for cls in (L.MELANOMA, L.NEVUS):
            c = sum(1 for r in sub if r["class_label"] == cls)
            if c != n:
                errors.append(f"{ds}/{cls}: {c} amostras (esperado {n})")
    for r in rows:
        if r["binary_label"] != L.BINARY.get(r["class_label"]):
            errors.append(f"{r['sample_id']}: binary_label incompatível com class_label")
        key, _, val = r["original_label"].partition("=")
        if r["dataset"] == "ph2":
            allowed = ph2["melanoma_clinical_codes"] if r["class_label"] == L.MELANOMA else ph2["nevus_clinical_codes"]
            if key != "clinical" or int(val) not in allowed:
                errors.append(f"{r['sample_id']}: rótulo original {r['original_label']} fora do mapeamento")
        else:
            allowed = ham["melanoma_dx"] if r["class_label"] == L.MELANOMA else ham["nevus_dx"]
            if key != "dx" or val not in allowed:
                errors.append(f"{r['sample_id']}: rótulo original {r['original_label']} fora do mapeamento")
        if check_files:
            for k in ("image_path", "mask_path"):
                if not r.get(k) or not Path(r[k]).exists():
                    errors.append(f"{r['sample_id']}: arquivo ausente ({k}={r.get(k)})")
    if errors:
        raise SamplingError("Manifesto inválido:\n  - " + "\n  - ".join(errors[:40]))


def manifest_fingerprint(rows: List[Dict]) -> str:
    return fingerprint([(r["sample_id"], r["class_label"], r["draw_rank"]) for r in rows])
