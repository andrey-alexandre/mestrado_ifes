"""Análises exploratórias: qualidade, avaliação diagnóstica e relação com as hipóteses.

Tudo é descritivo/exploratório (n pequeno, sem desenho causal). Valores-p servem para
ordenar sinais, não como prova; comparações múltiplas usam Holm.
"""
import collections
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import mannwhitneyu

from src.dataset_validation import interpretation as I
from src.dataset_validation import labels as L
from src.dataset_validation.metrics import (
    cliffs_delta, compare_proportions, holm_adjust, metrics_with_ci, safe_float,
)
from src.dataset_validation.quality import INDICATORS, INDICATOR_GROUPS


# ── Qualidade ──────────────────────────────────────────────────────────────────

def _vals(rows: Iterable[Dict], key: str) -> np.ndarray:
    out = []
    for r in rows:
        v = r.get(key)
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        out.append(f)
    a = np.asarray(out, float)
    return a[~np.isnan(a)]


def compare_quality(rows_a: Sequence[Dict], rows_b: Sequence[Dict], label_a: str, label_b: str,
                    indicators: Sequence[str] = INDICATORS) -> List[Dict]:
    """Mediana/IQR por grupo, Mann–Whitney bilateral, delta de Cliff e p ajustado (Holm)."""
    res: List[Dict] = []
    for k in indicators:
        a, b = _vals(rows_a, k), _vals(rows_b, k)
        row = {"indicator": k, "group": INDICATOR_GROUPS.get(k, ""), "n_a": len(a), "n_b": len(b)}
        for tag, v in ((label_a, a), (label_b, b)):
            if len(v):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                row[f"median_{tag}"], row[f"q1_{tag}"], row[f"q3_{tag}"] = float(med), float(q1), float(q3)
            else:
                row[f"median_{tag}"] = row[f"q1_{tag}"] = row[f"q3_{tag}"] = None
        if len(a) >= 3 and len(b) >= 3 and (np.ptp(np.concatenate([a, b])) > 0):
            row["mannwhitney_p"] = float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
            row["cliffs_delta"] = cliffs_delta(a, b)
        else:
            row["mannwhitney_p"], row["cliffs_delta"] = None, None
        res.append(row)
    adj = holm_adjust([np.nan if r["mannwhitney_p"] is None else r["mannwhitney_p"] for r in res])
    for r, p in zip(res, adj):
        r["holm_p"] = None if p is None or np.isnan(p) else float(p)
        r["direction"] = _direction(r["cliffs_delta"], label_a, label_b)
    return res


def _direction(delta: Optional[float], a: str, b: str) -> str:
    if delta is None:
        return "indefinido"
    mag = abs(delta)
    size = "desprezível" if mag < 0.147 else "pequeno" if mag < 0.33 else "médio" if mag < 0.474 else "grande"
    if size == "desprezível":
        return "sem diferença apreciável"
    return f"{a} > {b} ({size})" if delta > 0 else f"{b} > {a} ({size})"


def quality_summary(rows: Sequence[Dict]) -> Dict:
    """Comparações PH² × HAM10000 total e dentro de cada classe + efeito de classe por dataset."""
    out: Dict = {}
    sel = lambda ds, cls=None: [r for r in rows if r["dataset"] == ds and (cls is None or r["class_label"] == cls)]  # noqa: E731
    out["ph2_vs_ham10000_all"] = compare_quality(sel("ph2"), sel("ham10000"), "ph2", "ham10000")
    for cls in (L.MELANOMA, L.NEVUS):
        out[f"ph2_vs_ham10000_{cls}"] = compare_quality(sel("ph2", cls), sel("ham10000", cls), "ph2", "ham10000")
    for ds in ("ph2", "ham10000"):
        out[f"{ds}_melanoma_vs_nevus"] = compare_quality(
            sel(ds, L.MELANOMA), sel(ds, L.NEVUS), "melanoma", "nevus")
    return out


def source_composition(manifest: Sequence[Dict]) -> Dict:
    """Composição da amostra por fonte de aquisição / tipo de confirmação (confundidores de H1/H3)."""
    out: Dict = {}
    for ds in ("ph2", "ham10000"):
        sub = [r for r in manifest if r["dataset"] == ds]
        out[ds] = {
            "acquisition_source_x_class": _crosstab(sub, "acquisition_source", "class_label"),
            "dx_type_x_class": _crosstab(sub, "dx_type", "class_label"),
            "ph2_nevus_subtype": dict(collections.Counter(r["ph2_nevus_subtype"] for r in sub
                                                          if r["ph2_nevus_subtype"])),
            "localization_x_class": _crosstab(sub, "localization", "class_label"),
        }
    return out


def _crosstab(rows: Sequence[Dict], a: str, b: str) -> Dict[str, Dict[str, int]]:
    t: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        if r.get(a) in (None, ""):
            continue
        t[str(r[a])][str(r[b])] += 1
    return {k: dict(v) for k, v in sorted(t.items())}


# ── Avaliação diagnóstica ──────────────────────────────────────────────────────

def interpret_results(manifest: Sequence[Dict], results: Dict[str, Dict]) -> List[Dict]:
    """Une manifesto e respostas originais; preserva o texto e registra categoria/tier."""
    by_sample = {ev["_meta"]["sample_id"]: ev for ev in results.values()
                 if isinstance(ev, dict) and isinstance(ev.get("_meta"), dict)}
    rows = []
    for m in manifest:
        ev = by_sample.get(m["sample_id"]) or results.get(m["image_path"])
        row = {k: m[k] for k in ("sample_id", "dataset", "class_label", "binary_label", "image_id",
                                 "acquisition_source", "dx_type", "ph2_nevus_subtype")}
        if ev is None:
            row.update(status="sem_resposta", diagnostico=None, category=I.NAO_AVALIAVEL, tier="missing")
        else:
            text = I.diagnosis_text(ev)
            cat, tier = I.classify_diagnosis(text)
            row.update(status="ok", diagnostico=text, category=cat, tier=tier)
            row.update(I.agent_features(ev))
        for pol in I.POLICIES:
            row[f"pred_{pol}"] = I.binarize(row["category"], pol)
        rows.append(row)
    return rows


def evaluate_rows(rows: Sequence[Dict], policy: str, confidence: float, n_resamples: int, seed: int) -> Dict:
    ok = [r for r in rows if r[f"pred_{policy}"] is not None]
    y = [int(r["binary_label"]) for r in ok]
    p = [int(r[f"pred_{policy}"]) for r in ok]
    m = metrics_with_ci(y, p, confidence, n_resamples, seed)
    m["n_total"] = len(rows)
    m["n_evaluable"] = len(ok)
    m["n_not_evaluable"] = len(rows) - len(ok)
    # sensibilidade: respostas não avaliáveis contadas como erro
    y2 = [int(r["binary_label"]) for r in rows]
    p2 = [(int(r[f"pred_{policy}"]) if r[f"pred_{policy}"] is not None else 1 - int(r["binary_label"]))
          for r in rows]
    alt = metrics_with_ci(y2, p2, confidence, n_resamples, seed)
    m["non_evaluable_as_error"] = {k: alt[k] for k in ("sensitivity", "specificity", "precision", "f1",
                                                      "balanced_accuracy", "accuracy")}
    m["category_counts"] = dict(collections.Counter(r["category"] for r in rows))
    m["tier_counts"] = dict(collections.Counter(r["tier"] for r in rows))
    m["category_by_true_class"] = {
        cls: dict(collections.Counter(r["category"] for r in rows if r["class_label"] == cls))
        for cls in (L.MELANOMA, L.NEVUS)
    }
    return m


def evaluate_model(rows: Sequence[Dict], eval_cfg: Dict) -> Dict:
    conf, bo = eval_cfg["confidence"], eval_cfg["bootstrap"]
    out: Dict = {"by_dataset": {}, "policies": list(I.POLICIES),
                 "primary_policy": eval_cfg["primary_policy"]}
    for ds in ("ph2", "ham10000"):
        sub = [r for r in rows if r["dataset"] == ds]
        out["by_dataset"][ds] = {pol: evaluate_rows(sub, pol, conf, bo["n_resamples"], bo["seed"])
                                 for pol in I.POLICIES}
    prim = eval_cfg["primary_policy"]
    cmp: Dict = {}
    for metric, key in (("sensitivity", "tp"), ("specificity", "tn")):
        a = out["by_dataset"]["ph2"][prim]
        b = out["by_dataset"]["ham10000"][prim]
        cmp[metric] = compare_proportions(a[metric]["numerator"], a[metric]["denominator"],
                                          b[metric]["numerator"], b[metric]["denominator"], conf)
    out["ph2_vs_ham10000_primary_policy"] = cmp
    return out


# ── Relações com as hipóteses (exploratórias) ─────────────────────────────────

def correctness(rows: Sequence[Dict], policy: str) -> Dict[str, Optional[bool]]:
    return {r["sample_id"]: (None if r[f"pred_{policy}"] is None
                             else int(r[f"pred_{policy}"]) == int(r["binary_label"])) for r in rows}


def quality_vs_error(interp: Sequence[Dict], quality: Sequence[Dict], policy: str) -> Dict:
    """Indicadores de qualidade: acertos × erros (por dataset), Mann–Whitney exploratório."""
    qmap = {q["sample_id"]: q for q in quality}
    ok = correctness(interp, policy)
    out: Dict = {}
    for ds in ("ph2", "ham10000", "pooled"):
        sub = [dict(qmap[r["sample_id"]], correct=ok[r["sample_id"]]) for r in interp
               if r["sample_id"] in qmap and ok[r["sample_id"]] is not None
               and (ds == "pooled" or r["dataset"] == ds)]
        right = [r for r in sub if r["correct"]]
        wrong = [r for r in sub if not r["correct"]]
        out[ds] = {"n_correct": len(right), "n_incorrect": len(wrong),
                   "table": compare_quality(wrong, right, "incorrect", "correct") if len(wrong) >= 3 and len(right) >= 3
                   else []}
    return out


def error_by_group(interp: Sequence[Dict], policy: str, key: str) -> Dict:
    """Taxa de erro (e n) por valor de `key` (fonte, dx_type, subtipo do nevo, ...)."""
    ok = correctness(interp, policy)
    t: Dict[str, Dict] = {}
    for r in interp:
        g = r.get(key)
        if g in (None, "") or ok[r["sample_id"]] is None:
            continue
        cell = t.setdefault(f"{r['dataset']}|{r['class_label']}|{g}", {"n": 0, "errors": 0})
        cell["n"] += 1
        cell["errors"] += int(not ok[r["sample_id"]])
    for c in t.values():
        c["error_rate"] = c["errors"] / c["n"]
    return dict(sorted(t.items()))


def ph2_annotation_errors(interp: Sequence[Dict], ph2_records: Dict[str, Dict], policy: str) -> Dict:
    """Taxa de erro por anotação dermatoscópica do PH² (somente PH² tem essas anotações)."""
    ok = correctness(interp, policy)
    fields = ("asymmetry", "pigment_network", "dots_globules", "streaks", "regression_areas",
              "blue_whitish_veil")
    out: Dict = {}
    for f in fields:
        t: Dict[str, Dict] = {}
        for r in interp:
            if r["dataset"] != "ph2" or ok[r["sample_id"]] is None:
                continue
            rec = ph2_records.get(r["image_id"])
            if not rec:
                continue
            cell = t.setdefault(f"{r['class_label']}|{rec[f]}", {"n": 0, "errors": 0})
            cell["n"] += 1
            cell["errors"] += int(not ok[r["sample_id"]])
        for c in t.values():
            c["error_rate"] = c["errors"] / c["n"]
        out[f] = dict(sorted(t.items()))
    return out


AGENT_NUMERIC = ("abcd_assimetria", "abcd_bordas", "abcd_cor", "abcd_estruturas", "abcd_total",
                 "menzies_positive_count", "menzies_negative_count", "spcl_score", "n_methods_high_risk")


def agent_profiles(interp: Sequence[Dict]) -> Dict:
    """Perfil dos critérios dos agentes por dataset × classe (média, DP, n) e decisões."""
    out: Dict = {}
    for ds in ("ph2", "ham10000"):
        for cls in (L.MELANOMA, L.NEVUS):
            sub = [r for r in interp if r["dataset"] == ds and r["class_label"] == cls]
            cell: Dict = {"n": len(sub)}
            for k in AGENT_NUMERIC:
                v = _vals(sub, k)
                cell[k] = {"mean": safe_float(float(v.mean())) if len(v) else None,
                           "sd": safe_float(float(v.std(ddof=1))) if len(v) > 1 else None, "n": int(len(v))}
            for k in ("abcd_class", "menzies_decision", "spcl_decision"):
                cell[k] = dict(collections.Counter(r.get(k) for r in sub if r.get(k)))
            crit = [c for c in sub[0].keys() if c.startswith(("menzies_", "spcl_"))] if sub else []
            cell["binary_criteria_rate"] = {
                c: float(np.mean(_vals(sub, c))) for c in crit
                if c not in ("menzies_decision", "spcl_decision", "menzies_positive_count",
                             "menzies_negative_count", "spcl_score") and len(_vals(sub, c))
            }
            out[f"{ds}|{cls}"] = cell
    return out
