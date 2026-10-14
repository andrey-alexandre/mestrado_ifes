"""Recuperação das evidências experimentais legadas do PH² (somente leitura).

Reavalia os arquivos existentes SEM reconstruir resultados: preserva o texto original
das VLMs, reproduz a matriz da tabela da dissertação a partir de `estimated_value` e
mostra o efeito de (i) rótulos legados × clínicos e (ii) regra programática × leitura
manual. Nada aqui chama VLMs.
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset_validation import interpretation as I
from src.dataset_validation import labels as L
from src.dataset_validation.config import ROOT
from src.dataset_validation.io_utils import read_csv, read_json, sha256_file
from src.dataset_validation.metrics import metrics_with_ci

_NAME = re.compile(r"(IMD\d+)")


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def load_legacy_rows(cfg: Dict, ph2_records: Dict[str, Dict]) -> Dict[str, Dict]:
    """{modelo: {"rows": [...], "files": {...}, "alignment": {...}}} para os resultados do PH²."""
    out: Dict[str, Dict] = {}
    ph2 = cfg["datasets"]["ph2"]
    for model, json_rel in cfg["legacy"]["ph2_results_json"].items():
        jp = _abs(json_rel)
        if not jp.exists():
            out[model] = {"rows": [], "files": {"json": str(jp), "present": False}}
            continue
        data = read_json(jp)
        csv_rel = cfg["legacy"]["ph2_estimated_csv"].get(model)
        csv_rows = read_csv(_abs(csv_rel)) if csv_rel and _abs(csv_rel).exists() else []
        rows: List[Dict] = []
        text_match = 0
        rv_match = 0
        cat_agree = 0
        compared = 0
        for i, (path, ev) in enumerate(data.items()):
            m = _NAME.search(path)
            if not m:
                continue
            name = m.group(1)
            rec = ph2_records.get(name)
            text = I.diagnosis_text(ev) or ""
            cat, tier = I.classify_diagnosis(text)
            cls = L.ph2_class(rec, ph2) if rec else None
            thesis_est: Optional[int] = None
            csv_cat = None
            if i < len(csv_rows):
                c = csv_rows[i]
                if (c.get("validation.diagnostico") or "").strip() == text.strip():
                    text_match += 1
                if str(c.get("real_value")) == str(int(ev.get("real_value", -1))):
                    rv_match += 1
                csv_cat = I.classify_diagnosis(c.get("validation.diagnostico"))[0]
                compared += 1
                cat_agree += int(csv_cat == cat)
                est = c.get("estimated_value")
                thesis_est = int(est) if est not in (None, "") else None
            rows.append({
                "model": model, "image_id": name, "legacy_real_value": int(ev.get("real_value", -1)),
                "corrected_label": L.BINARY.get(cls) if cls else None,
                "clinical_code": rec["clinical_diagnosis"] if rec else None,
                "histological": rec["histological_diagnosis"] if rec else None,
                "diagnostico": text, "category": cat, "tier": tier,
                "pred_indeterminate_as_positive": I.binarize(cat, "indeterminate_as_positive"),
                "pred_indeterminate_as_negative": I.binarize(cat, "indeterminate_as_negative"),
                "thesis_estimated_value": thesis_est, "csv_category": csv_cat,
                "has_seg_path": bool(ev.get("seg_path")), "lesion_size": ev.get("lesion_size"),
                "event": ev,
            })
        out[model] = {
            "rows": rows,
            "files": {"json": str(jp), "json_sha256": sha256_file(jp), "present": True,
                      "csv": str(_abs(csv_rel)) if csv_rel else None},
            "alignment": {
                "n_json": len(rows), "n_csv": len(csv_rows), "diagnostico_text_match": text_match,
                "real_value_match_by_row_order": rv_match,
                "category_agreement_json_vs_csv_run": cat_agree, "n_compared": compared,
                "note": "CSV (06/06) e JSON (09/06) são execuções distintas quando os textos diferem; o "
                        "CSV não tem id de imagem: o pareamento por ordem é uma inferência.",
            },
        }
    return out


def _pairs(rows: List[Dict], y_key: str, p_key: str):
    ys, ps = [], []
    for r in rows:
        y, p = r.get(y_key), r.get(p_key)
        if y is None or p is None or y == -1:
            continue
        ys.append(int(y))
        ps.append(int(p))
    return ys, ps


def summarize_legacy(legacy: Dict[str, Dict], cfg: Dict) -> Dict:
    """Matrizes/métricas do PH² legado sob rótulos legados × corrigidos e regras de leitura."""
    ev = cfg["evaluation"]
    boot = ev["bootstrap"]
    out: Dict = {}
    for model, blob in legacy.items():
        rows = blob["rows"]
        if not rows:
            out[model] = {"present": False}
            continue
        scenarios = {
            "thesis_estimated_value__legacy_labels": ("legacy_real_value", "thesis_estimated_value"),
            "thesis_estimated_value__clinical_labels": ("corrected_label", "thesis_estimated_value"),
            "programmatic_indet_positive__legacy_labels": ("legacy_real_value", "pred_indeterminate_as_positive"),
            "programmatic_indet_positive__clinical_labels": ("corrected_label", "pred_indeterminate_as_positive"),
            "programmatic_indet_negative__clinical_labels": ("corrected_label", "pred_indeterminate_as_negative"),
        }
        res: Dict = {"present": True, "n_images": len(rows), "scenarios": {}}
        for name, (yk, pk) in scenarios.items():
            y, p = _pairs(rows, yk, pk)
            res["scenarios"][name] = metrics_with_ci(y, p, ev["confidence"], boot["n_resamples"], boot["seed"]) \
                if y else {"n": 0}
        res["label_disagreement"] = [
            {"image_id": r["image_id"], "legacy_real_value": r["legacy_real_value"],
             "clinical_code": r["clinical_code"], "corrected_label": r["corrected_label"]}
            for r in rows if r["corrected_label"] is not None and r["legacy_real_value"] != r["corrected_label"]
        ]
        res["n_positive_legacy"] = sum(1 for r in rows if r["legacy_real_value"] == 1)
        res["n_positive_clinical"] = sum(1 for r in rows if r["corrected_label"] == 1)
        res["category_counts"] = _count(r["category"] for r in rows)
        res["tier_counts"] = _count(r["tier"] for r in rows)
        res["alignment"] = blob.get("alignment")
        res["files"] = blob["files"]
        out[model] = res
    return out


def _count(items) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for it in items:
        d[it] = d.get(it, 0) + 1
    return d
