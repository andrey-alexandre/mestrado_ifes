"""Etapa 5 — comparação pareada entre transferência direta e fine-tuning (mesmas imagens)."""
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.segmentation.artifacts import StageStore, fingerprint, read_csv, read_json, write_csv, write_json
from src.segmentation.data import DataError
from src.segmentation.metrics import (
    HIGHER_IS_BETTER, bootstrap_statistic, paired_bootstrap_diff, summarize,
)
from src.segmentation.stages import Layout

logger = logging.getLogger(__name__)

SELECTION_BIAS_NOTE = (
    "A estratégia vencedora foi escolhida com base no próprio PH². A estimativa de desempenho do "
    "modelo selecionado nesse mesmo conjunto está sujeita a viés de seleção (otimismo) e deve ser "
    "interpretada como limite superior; não constitui avaliação independente."
)


def _num(v) -> float:
    return float("nan") if v in (None, "", "None") else float(v)


def _load(path: Path) -> List[Dict]:
    if not path.exists():
        raise DataError(f"Resultados ausentes: {path}. Execute a etapa correspondente antes de 'compare'.")
    return read_csv(str(path))


def _col(rows: List[Dict], key: str) -> np.ndarray:
    return np.array([_num(r[key]) for r in rows], dtype=float)


def _mean_ci(values: np.ndarray, boot: Dict) -> Dict:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    res = bootstrap_statistic([valid], lambda x: float(np.mean(x)), boot["n_resamples"],
                              boot["confidence"], boot["seed"])
    return {k: res[k] for k in ("estimate", "ci_low", "ci_high")}


def _group_stats(direct: List[Dict], tuned: List[Dict], metrics: List[str], boot: Dict) -> Dict:
    out: Dict = {"n_images": len(direct)}
    for m in metrics:
        a, b = _col(direct, m), _col(tuned, m)
        out[m] = {
            "direct_transfer": summarize(a), "finetune": summarize(b),
            "paired_diff_mean": paired_bootstrap_diff(a, b, "mean", boot["n_resamples"],
                                                      boot["confidence"], boot["seed"]),
            "paired_diff_median": paired_bootstrap_diff(a, b, "median", boot["n_resamples"],
                                                        boot["confidence"], boot["seed"]),
        }
    return out


def _rank(rows: List[Dict], metric: str, n: int, best: bool) -> List[str]:
    vals = [(r["image_id"], _num(r[metric])) for r in rows if not np.isnan(_num(r[metric]))]
    higher = HIGHER_IS_BETTER[metric]
    vals.sort(key=lambda t: t[1], reverse=(higher == best))
    return [i for i, _ in vals[:n]]


def decide_selection(primary: str, rule: str, direct_mean: float, tuned_mean: float,
                     paired_mean_diff: Dict) -> Dict:
    """Regra de seleção pré-declarada (config). Empate favorece a transferência direta."""
    if np.isnan(direct_mean) or np.isnan(tuned_mean):
        raise DataError(f"Média indefinida para a métrica primária '{primary}'; não há como selecionar.")
    sign = 1.0 if HIGHER_IS_BETTER[primary] else -1.0
    improvement = sign * paired_mean_diff["estimate"]
    lo, hi = paired_mean_diff["ci_low"], paired_mean_diff["ci_high"]
    ci_improvement = (sign * lo, sign * hi) if sign > 0 else (sign * hi, sign * lo)
    if rule == "higher_mean":
        tuned_wins = improvement > 0
    else:  # mean_with_ci: exige que o IC do ganho exclua zero a favor do fine-tuning
        tuned_wins = improvement > 0 and ci_improvement[0] > 0
    return {"selected": "finetune" if tuned_wins else "direct_transfer",
            "improvement_of_finetune_over_direct": improvement,
            "improvement_ci": list(ci_improvement)}


def run_comparison(cfg: Dict, store: StageStore) -> Dict:
    lay = Layout(cfg)
    ev = cfg["evaluation"]
    d_csv, f_csv = lay.results / "direct_transfer" / "per_image.csv", lay.results / "finetune" / "per_image.csv"
    direct, tuned = _load(d_csv), _load(f_csv)
    if [r["image_id"] for r in direct] != [r["image_id"] for r in tuned]:
        raise DataError("As duas estratégias não cobrem exatamente as mesmas imagens, na mesma ordem.")
    for a, b in zip(direct, tuned):
        if a["fold"] != b["fold"] or a["diagnosis"] != b["diagnosis"]:
            raise DataError(f"Fold/diagnóstico divergem para {a['image_id']} entre as estratégias.")

    from src.segmentation.artifacts import sha256_file
    fp = fingerprint(sha256_file(str(d_csv)), sha256_file(str(f_csv)), ev)
    out = lay.results / "comparison"
    if store.is_complete("comparison", fp) and (out / "selection.json").exists():
        return read_json(str(out / "summary.json"))

    boot = ev["bootstrap"]
    metrics = [ev["primary_metric"]] + list(ev["secondary_metrics"])
    n = len(direct)

    overall = _group_stats(direct, tuned, metrics, boot)
    for m in metrics:
        overall[m]["direct_transfer"]["mean_ci"] = _mean_ci(_col(direct, m), boot)
        overall[m]["finetune"]["mean_ci"] = _mean_ci(_col(tuned, m), boot)

    def subset(rows, key, value):
        return [i for i, r in enumerate(rows) if r[key] == value]

    per_fold, per_class, per_label = {}, {}, {}
    for key, store_dict in (("fold", per_fold), ("diagnosis", per_class), ("label", per_label)):
        for v in sorted({r[key] for r in direct}):
            idx = subset(direct, key, v)
            store_dict[str(v)] = _group_stats([direct[i] for i in idx], [tuned[i] for i in idx],
                                              metrics, boot)

    diffs = []
    for a, b in zip(direct, tuned):
        row = {"image_id": a["image_id"], "fold": a["fold"], "diagnosis": a["diagnosis"], "label": a["label"]}
        for m in metrics:
            row[f"{m}_direct"], row[f"{m}_finetune"] = a[m], b[m]
            row[f"{m}_diff"] = _num(b[m]) - _num(a[m])
        diffs.append(row)
    write_csv(str(out / "paired_differences.csv"), diffs)

    primary = ev["primary_metric"]
    decision = decide_selection(primary, ev["selection_rule"],
                                overall[primary]["direct_transfer"]["mean"],
                                overall[primary]["finetune"]["mean"],
                                overall[primary]["paired_diff_mean"])
    selection = {
        "primary_metric": primary, "rule": ev["selection_rule"], "n_images": n,
        **decision,
        "direct_transfer_mean": overall[primary]["direct_transfer"]["mean"],
        "finetune_mean": overall[primary]["finetune"]["mean"],
        "selection_bias_note": SELECTION_BIAS_NOTE,
        "preserved_artifacts": {"direct_transfer": str(d_csv), "finetune": str(f_csv)},
    }

    n_ex = int(ev["n_examples"])
    examples = {"primary_metric": primary}
    for tag, rows in (("direct_transfer", direct), ("finetune", tuned)):
        examples[tag] = {"best": _rank(rows, primary, n_ex, True), "worst": _rank(rows, primary, n_ex, False)}
    delta = [(r["image_id"], r[f"{primary}_diff"]) for r in diffs if r[f"{primary}_diff"] is not None
             and not np.isnan(float(r[f"{primary}_diff"]))]
    sign = 1.0 if HIGHER_IS_BETTER[primary] else -1.0
    delta.sort(key=lambda t: sign * float(t[1]), reverse=True)
    examples["largest_improvement"] = [i for i, _ in delta[:n_ex]]
    examples["largest_degradation"] = [i for i, _ in delta[::-1][:n_ex]]

    summary = {
        "same_images": True, "n_images": n, "metrics": metrics,
        "diff_convention": "finetune - direct_transfer (para HD95, menor é melhor)",
        "overall": overall, "per_fold": per_fold, "per_diagnosis": per_class,
        "per_binary_label": per_label,
        "direct_transfer_threshold": read_json(str(lay.results / "direct_transfer" / "summary.json"))["threshold"],
        "finetune_threshold": read_json(str(lay.results / "finetune" / "summary.json"))["threshold"],
    }
    write_json(str(out / "summary.json"), summary)
    write_json(str(out / "selection.json"), selection)
    write_json(str(out / "examples.json"), examples)
    store.mark_complete("comparison", fp)
    logger.info("Seleção: %s (métrica %s)", selection["selected"], primary)
    return summary
