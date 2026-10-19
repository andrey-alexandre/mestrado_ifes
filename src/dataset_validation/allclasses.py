"""Extensão "todas as classes × melanoma" (somente HAM10000; PH² fora).

Reaproveita EXATAMENTE as 20 + 20 imagens `mel`/`nv` do piloto (mesmo sorteio, semente 42) e
sorteia, por classe, 20 lesões das demais classes (akiec, bcc, bkl, df, vasc) — uma imagem por
`lesion_id`, `random.Random(f"{seed}:ham10000:{dx}")`. Positivo = melanoma; todo o resto = 0.
Nenhuma alteração nos agentes, prompts ou regra de leitura; experimento em diretório próprio.
"""
import collections
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.dataset_validation import labels as L
from src.dataset_validation import sampling as S
from src.dataset_validation.io_utils import fingerprint, read_json, sha256_file, write_json
from src.dataset_validation.metrics import metrics_with_ci, wilson_ci

OTHER_CLASSES = ("akiec", "bcc", "bkl", "df", "vasc")
CLASS_ORDER = ("melanoma", "nevus") + OTHER_CLASSES
EXTRA_FIELDS = ["dx"]


def _row_dx(r: Dict) -> str:
    return r["original_label"].partition("=")[2]


def build_manifest(cfg: Dict, ham_meta: List[Dict], mask_cache_dir: Optional[Path] = None,
                   check_files: bool = True, add_hashes: bool = True) -> List[Dict]:
    ham, seed = cfg["datasets"]["ham10000"], cfg["seed"]
    n = cfg["allclasses"]["n_per_class"]
    sub = dict(cfg, sample=dict(cfg["sample"], n_per_class=n))
    base = S._ham_rows(ham_meta, sub, mask_resolver=None)   # mel + nv: idêntico ao piloto
    by_id = {r["image_id"]: r for r in ham_meta}
    mel_lesions = {r["lesion_id"] for r in ham_meta if r["dx"] == "mel"}
    nv_lesions = {r["lesion_id"] for r in ham_meta if r["dx"] == "nv"}
    rows = list(base)
    for dx in OTHER_CLASSES:
        pool = [r for r in ham_meta if r["dx"] == dx]
        by_lesion: Dict[str, List[str]] = collections.defaultdict(list)
        for r in pool:
            if r["lesion_id"] in mel_lesions | nv_lesions:
                raise S.SamplingError(f"lesion_id {r['lesion_id']} aparece em {dx} e em mel/nv")
            by_lesion[r["lesion_id"]].append(r["image_id"])
        stream = f"ham10000:{dx}"
        for rank, les in enumerate(S.draw(list(by_lesion), n, seed, stream), start=1):
            iid = random.Random(f"{seed}:{stream}:{les}").choice(sorted(by_lesion[les]))
            r = by_id[iid]
            img, msk = L.ham_paths(ham, iid)
            rows.append({
                "sample_id": f"ham10000:{iid}", "dataset": "ham10000", "class_label": dx, "binary_label": 0,
                "original_label": f"dx={dx}", "label_source": "HAM10000_metadata:dx", "image_id": iid,
                "lesion_id": les, "image_path": img, "mask_path": msk, "seed": seed, "stream": stream,
                "pool_size": len(by_lesion), "draw_rank": rank, "dx_type": r["dx_type"],
                "acquisition_source": r["dataset"], "ph2_nevus_subtype": "", "age": r.get("age", ""),
                "sex": r.get("sex", ""), "localization": r.get("localization", ""),
            })
    missing = [r["image_id"] for r in rows if r["mask_path"] is None]
    if missing:
        cache = mask_cache_dir or Path(cfg["output"]["root_dir"]) / "inputs" / "ham10000_masks"
        found = S.extract_ham_masks(cfg, missing, cache)
        for r in rows:
            if r["mask_path"] is None:
                r["mask_path"] = found[r["image_id"]]
    if add_hashes:
        for r in rows:
            for key, col in (("image_path", "image_sha256"), ("mask_path", "mask_sha256")):
                p = r[key]
                r[col] = sha256_file(p) if p and Path(p).exists() else ""
    validate_manifest(rows, cfg, check_files=check_files)
    return rows


def validate_manifest(rows: Sequence[Dict], cfg: Dict, check_files: bool = True) -> None:
    n = cfg["allclasses"]["n_per_class"]
    errors: List[str] = []
    for col in ("sample_id", "image_id", "lesion_id"):
        vals = [r[col] for r in rows]
        if len(vals) != len(set(vals)):
            errors.append(f"{col} duplicado")
    for cls in CLASS_ORDER:
        c = sum(1 for r in rows if r["class_label"] == cls)
        if c != n:
            errors.append(f"{cls}: {c} amostras (esperado {n})")
    allowed = {"melanoma": "mel", "nevus": "nv", **{d: d for d in OTHER_CLASSES}}
    for r in rows:
        if r["dataset"] != "ham10000" or allowed.get(r["class_label"]) != _row_dx(r):
            errors.append(f"{r['sample_id']}: rótulo {r['original_label']} incompatível com {r['class_label']}")
        if r["binary_label"] != (1 if r["class_label"] == "melanoma" else 0):
            errors.append(f"{r['sample_id']}: binary_label incompatível")
        if check_files:
            for k in ("image_path", "mask_path"):
                if not r.get(k) or not Path(r[k]).exists():
                    errors.append(f"{r['sample_id']}: arquivo ausente ({k})")
    if errors:
        raise S.SamplingError("Manifesto all-classes inválido:\n  - " + "\n  - ".join(errors[:40]))


def manifest_fingerprint(rows: Sequence[Dict]) -> str:
    return fingerprint([(r["sample_id"], r["class_label"], r["draw_rank"]) for r in rows])


def seed_reused_results(reuse_dir: Path, model: str, raw_dir: Path, manifest: Sequence[Dict],
                        agents_py: Path, schemas_py: Path, allow_mismatch: bool = False) -> List[str]:
    """Pré-popula `raw/<modelo>/` com respostas do piloto para as imagens em comum.

    Só reaproveita se agents.py/schemas.py (normalizados para LF) forem idênticos aos do piloto.
    """
    from src.dataset_validation.inference import safe_name
    safe = safe_name(model)
    res_p, meta_p = reuse_dir / f"results_{safe}.json", reuse_dir / f"results_{safe}.meta.json"
    if not res_p.exists() or not meta_p.exists():
        return []
    meta = read_json(meta_p)
    for key, path in (("agents_sha256", agents_py), ("schemas_sha256", schemas_py)):
        cur = _sha_lf(path)
        if meta.get(key) not in (cur, sha256_file(path)) and not allow_mismatch:
            raise S.SamplingError(f"Reuso recusado: {key} do piloto difere do código atual ({path.name}).")
    if meta.get("temperature") is not None and meta["model_name"] != model:
        raise S.SamplingError("Reuso recusado: modelo do piloto difere.")
    by_sample = {r["sample_id"]: r for r in manifest}
    raw_dir.mkdir(parents=True, exist_ok=True)
    reused = []
    for ev in read_json(res_p).values():
        sid = (ev.get("_meta") or {}).get("sample_id")
        if sid in by_sample and by_sample[sid]["dataset"] == "ham10000":
            r = by_sample[sid]
            clean = {k: v for k, v in ev.items() if k != "_meta"}
            part = raw_dir / f"{r['dataset']}__{r['image_id']}.json"
            if not part.exists():
                write_json(part, {r["image_path"]: clean})
            reused.append(sid)
    return sorted(reused)


def _sha_lf(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── Avaliação ─────────────────────────────────────────────────────────────────

def evaluate_allclasses(rows: Sequence[Dict], pool_counts: Dict[str, int], eval_cfg: Dict) -> Dict:
    """Por classe e agregada, para as duas políticas de leitura."""
    from src.dataset_validation import interpretation as I
    conf, bo = eval_cfg["confidence"], eval_cfg["bootstrap"]
    out: Dict = {}
    for pol in I.POLICIES:
        pred = lambda r: r[f"pred_{pol}"]  # noqa: E731
        per_class: Dict = {}
        for cls in CLASS_ORDER:
            sub = [r for r in rows if r["class_label"] == cls]
            ok = [r for r in sub if pred(r) is not None]
            pos = sum(int(pred(r)) for r in ok)          # previstos positivos (melanoma)
            n = len(ok)
            lo, hi = wilson_ci(pos, n, conf)
            key = "sensitivity" if cls == "melanoma" else "false_positive_rate"
            spec_lo, spec_hi = wilson_ci(n - pos, n, conf)
            per_class[cls] = {
                "n": len(sub), "n_evaluable": n, "predicted_melanoma": pos,
                "categories": dict(collections.Counter(r["category"] for r in sub)),
                key: pos / n if n else None, f"{key}_ci": [lo, hi],
                **({"specificity": (n - pos) / n if n else None, "specificity_ci": [spec_lo, spec_hi]}
                   if cls != "melanoma" else {}),
            }
        usable = [r for r in rows if pred(r) is not None]
        y = [int(r["binary_label"]) for r in usable]
        p = [int(pred(r)) for r in usable]
        pooled = metrics_with_ci(y, p, conf, bo["n_resamples"], bo["seed"])
        # especificidade macro (classes não-melanoma com peso igual) e por prevalência natural do HAM10000
        specs = {c: per_class[c]["specificity"] for c in CLASS_ORDER if c != "melanoma"
                 and per_class[c].get("specificity") is not None}
        macro = float(np.mean(list(specs.values()))) if specs else None
        tot = sum(pool_counts[c] for c in specs)
        weighted = sum(specs[c] * pool_counts[c] / tot for c in specs) if tot else None
        rng = np.random.default_rng(bo["seed"])
        boots = []
        for _ in range(bo["n_resamples"]):
            acc = 0.0
            for c in specs:
                rs = [r for r in usable if r["class_label"] == c]
                v = np.mean([int(pred(rs[i]) == 0) for i in rng.integers(0, len(rs), len(rs))])
                acc += v * pool_counts[c] / tot
            boots.append(acc)
        out[pol] = {
            "per_class": per_class,
            "melanoma_vs_all_others_pooled": pooled,
            "melanoma_vs_nevus_only": metrics_with_ci(
                [int(r["binary_label"]) for r in usable if r["class_label"] in ("melanoma", "nevus")],
                [int(pred(r)) for r in usable if r["class_label"] in ("melanoma", "nevus")], conf,
                bo["n_resamples"], bo["seed"]),
            "macro_specificity_non_melanoma": macro,
            "natural_prevalence_weighted_specificity": {
                "estimate": weighted, "ci_low": float(np.quantile(boots, 0.025)),
                "ci_high": float(np.quantile(boots, 0.975)),
                "weights": {c: pool_counts[c] / tot for c in specs} if tot else {},
                "note": "pesos = frequência de cada classe não-melanoma no HAM10000 completo (imagens); "
                        "IC por bootstrap estratificado por classe",
            },
        }
    return out


def pool_counts(ham_meta: Sequence[Dict]) -> Dict[str, int]:
    names = {"mel": "melanoma", "nv": "nevus"}
    c = collections.Counter(names.get(r["dx"], r["dx"]) for r in ham_meta)
    return {k: c[k] for k in CLASS_ORDER}
