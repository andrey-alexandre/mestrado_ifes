"""Etapa 6 — impacto da segmentação no sistema multiagente (C0 = GT, C1 = predita).

Interface de adaptação, sem tocar nos agentes: as duas condições usam o mesmo
grafo em `segmentation_mode="precomputed"` e diferem somente em `seg_path`. A
extração de features do pipeline (que consome `state.seg_path`) é, portanto,
recalculada a partir da máscara de cada condição.

O modo padrão NÃO chama VLMs: calcula/compara apenas as features dependentes da
máscara e sua relação com o erro de segmentação. As chamadas aos VLMs exigem
`run_vlm=True` explícito.
"""
import hashlib
import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.segmentation.artifacts import (
    ArtifactMismatch, StageStore, clean, fingerprint, read_csv, read_json, sha256_file,
    write_csv, write_json,
)
from src.segmentation.data import DataError, load_mask_bool, save_mask_png
from src.segmentation.metrics import bootstrap_statistic
from src.segmentation.stages import Layout, _load_folds_and_records

logger = logging.getLogger(__name__)

CONDITIONS = ("C0", "C1")
AGENT_FIELDS = ("diagnosis_abcd", "diagnosis_menzies", "diagnosis_spcl", "validation")

# Escore ordinal do parecer final do SummaryAgent (benigno < indeterminado < suspeito).
_ORDINAL_SCORE = {"lesão benigna": 0.0, "lesão indeterminada": 0.5, "suspeita de melanoma": 1.0}

ImageEntry = Tuple[str, float, int, Optional[str]]


# ─────────────────────────────────────────────────────────────────────────────
# Construção das condições
# ─────────────────────────────────────────────────────────────────────────────

def build_condition_entries(
    image_ids: Sequence[str], by_id: Dict, pred_dir: str, empty_policy: str = "flag_and_skip",
    fallback_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Monta as entradas `ImageEntry` de C0 e C1.

    Somente `seg_path` difere entre as condições. Máscaras previstas vazias são
    tratadas conforme `empty_policy` (declarada previamente no config).

    Returns:
        {"C0": [...], "C1": [...], "flagged": {image_id: motivo}}
    """
    c0: List[ImageEntry] = []
    c1: List[ImageEntry] = []
    flagged: Dict[str, str] = {}
    for iid in image_ids:
        rec = by_id[iid]
        target = -1 if rec.label is None else int(rec.label)
        pred_path = str(Path(pred_dir) / f"{iid}.png")
        if not Path(pred_path).exists():
            raise DataError(f"Máscara prevista ausente para {iid}: {pred_path}")
        c0.append((rec.image_path, 0.0, target, rec.mask_path))
        seg_path = pred_path
        if not load_mask_bool(pred_path).any():
            if empty_policy == "full_image":
                if fallback_dir is None:
                    raise DataError("full_image requer fallback_dir.")
                full = np.ones(load_mask_bool(rec.mask_path).shape, dtype=bool)
                seg_path = str(Path(fallback_dir) / f"{iid}.png")
                save_mask_png(full, seg_path)
                flagged[iid] = "empty_prediction_replaced_by_full_image"
            else:
                flagged[iid] = "empty_prediction_skipped"
                continue
        c1.append((rec.image_path, 0.0, target, seg_path))
    return {"C0": c0, "C1": c1, "flagged": flagged}


def condition_invariants(cfg: Dict) -> Dict:
    """Tudo que deve ser idêntico entre C0 e C1 (exceto a origem da máscara)."""
    root = Path(__file__).resolve().parents[1]

    def digest(rel: str) -> Optional[str]:
        p = root / rel
        if p.is_dir():
            h = hashlib.sha256()
            for f in sorted(p.rglob("*.py")):
                h.update(f.name.encode())
                h.update(f.read_bytes())
            return h.hexdigest()
        return sha256_file(str(p)) if p.exists() else None

    return {
        "llm": cfg.get("_llm", {}),
        "feature_extraction_config": cfg.get("_feature_extraction", {}),
        "agents_source_sha256": digest("agents/agents.py"),
        "schemas_source_sha256": digest("agents/schemas.py"),
        "features_source_sha256": digest("features"),
        "pipeline_source_sha256": digest("pipeline/pipeline.py"),
        "segmentation_mode_in_graph": "precomputed",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Features dependentes da máscara
# ─────────────────────────────────────────────────────────────────────────────

def resolve_callable(spec: str) -> Tuple[Optional[Callable], Optional[str]]:
    """Resolve 'pacote.modulo:funcao'. Retorna (callable, motivo_da_indisponibilidade)."""
    try:
        mod_name, attr = spec.split(":")
        return getattr(importlib.import_module(mod_name), attr), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def flatten_numeric(obj: Any, prefix: str = "") -> Dict[str, float]:
    """Achata um dict/lista aninhado, mantendo apenas folhas numéricas (sem bool)."""
    out: Dict[str, float] = {}
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="python")
    obj = clean(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_numeric(v, f"{prefix}{k}."))
    elif isinstance(obj, list) and len(obj) <= 32:
        for i, v in enumerate(obj):
            out.update(flatten_numeric(v, f"{prefix}{i}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def extract_features(extractor: Callable, entries: Sequence[ImageEntry], fe_cfg: Dict,
                     condition: str) -> Dict[str, Dict]:
    """Recalcula as features para cada entrada usando a máscara da condição (sem cache)."""
    out: Dict[str, Dict] = {}
    for image_path, _, _, mask_path in entries:
        key = Path(image_path).stem
        try:
            pkg = extractor(image_path=image_path, mask_path=mask_path, config=fe_cfg)
            out[key] = {"condition": condition, "mask_path": mask_path,
                        "features": flatten_numeric(pkg), "error": None}
        except Exception as exc:  # noqa: BLE001 - registrar e seguir
            out[key] = {"condition": condition, "mask_path": mask_path, "features": {},
                        "error": f"{type(exc).__name__}: {exc}"}
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    from scipy.stats import spearmanr
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan"), int(ok.sum())
    return float(spearmanr(x[ok], y[ok]).statistic), int(ok.sum())


def compare_features(f0: Dict[str, Dict], f1: Dict[str, Dict], seg_rows: Dict[str, Dict]) -> List[Dict]:
    """Diferença C1-C0 por feature e correlação com o erro de segmentação (1-Dice, HD95)."""
    common = sorted(set(f0) & set(f1))
    names = sorted({n for i in common for n in f0[i]["features"]} &
                   {n for i in common for n in f1[i]["features"]})
    err_dice = np.array([1.0 - float(seg_rows[i]["dice"]) if seg_rows.get(i) and
                         seg_rows[i]["dice"] not in (None, "") else np.nan for i in common])
    err_hd = np.array([float(seg_rows[i]["hd95"]) if seg_rows.get(i) and
                       seg_rows[i]["hd95"] not in (None, "") else np.nan for i in common])
    rows = []
    for n in names:
        a = np.array([f0[i]["features"].get(n, np.nan) for i in common], dtype=float)
        b = np.array([f1[i]["features"].get(n, np.nan) for i in common], dtype=float)
        absdiff = np.abs(b - a)
        r_dice, n_d = _spearman(absdiff, err_dice)
        r_hd, n_h = _spearman(absdiff, err_hd)
        ok = ~(np.isnan(a) | np.isnan(b))
        rows.append({
            "feature": n, "n_pairs": int(ok.sum()),
            "mean_c0": float(np.mean(a[ok])) if ok.any() else None,
            "mean_c1": float(np.mean(b[ok])) if ok.any() else None,
            "mean_diff_c1_minus_c0": float(np.mean((b - a)[ok])) if ok.any() else None,
            "mean_abs_diff": float(np.mean(absdiff[ok])) if ok.any() else None,
            "spearman_absdiff_vs_1_minus_dice": r_dice,
            "spearman_absdiff_vs_hd95": r_hd,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Métricas diagnósticas
# ─────────────────────────────────────────────────────────────────────────────

def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    sens, prec = _ratio(tp, tp + fn), _ratio(tp, tp + fp)
    f1 = _ratio(2 * prec * sens, prec + sens) if not (np.isnan(prec) or np.isnan(sens)) else float("nan")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "sensitivity": sens,
            "specificity": _ratio(tn, tn + fp), "precision": prec, "f1": f1,
            "accuracy": _ratio(tp + tn, tp + tn + fp + fn)}


def decision_from_event(event: Dict) -> Optional[Tuple[int, float, str]]:
    """(rótulo binário, escore ordinal, texto) do parecer final, com o mapeamento legado."""
    from src.evaluation.evaluate import _DIAGNOSTICO_TO_LABEL  # lógica diagnóstica existente
    val = event.get("validation")
    if not val:
        return None
    text = (val.get("diagnostico", "") if isinstance(val, dict) else str(val)).lower()
    for key, label in _DIAGNOSTICO_TO_LABEL.items():
        if key in text:
            return label, _ORDINAL_SCORE.get(key, float(label)), key
    return None


def diagnostic_comparison(y: np.ndarray, p0: np.ndarray, s0: np.ndarray, p1: np.ndarray,
                          s1: np.ndarray, boot: Dict) -> Dict:
    """Métricas por condição, kappa, taxa de mudança e IC pareado das diferenças (C1-C0)."""
    from sklearn.metrics import cohen_kappa_score, roc_auc_score

    def auc(score):
        return float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else float("nan")

    out: Dict[str, Any] = {"n_images": int(len(y))}
    out["C0"] = {**binary_metrics(y, p0), "roc_auc_ordinal": auc(s0)}
    out["C1"] = {**binary_metrics(y, p1), "roc_auc_ordinal": auc(s1)}
    for c, p in (("C0", p0), ("C1", p1)):
        out[c]["confusion_matrix"] = [[out[c]["tn"], out[c]["fp"]], [out[c]["fn"], out[c]["tp"]]]
    out["change_rate"] = float(np.mean(p0 != p1))
    out["n_changed"] = int(np.sum(p0 != p1))
    if len(set(p0.tolist())) < 2 and len(set(p1.tolist())) < 2 and set(p0.tolist()) == set(p1.tolist()):
        out["cohen_kappa"] = float("nan")  # indefinido: ambas as condições constantes e iguais
    else:
        out["cohen_kappa"] = float(cohen_kappa_score(p0, p1))
    out["paired_diff_c1_minus_c0"] = {}
    for m in ("sensitivity", "specificity", "precision", "f1", "accuracy"):
        res = bootstrap_statistic(
            [y, p0, p1],
            lambda yy, a, b, m=m: binary_metrics(yy, b)[m] - binary_metrics(yy, a)[m],
            boot["n_resamples"], boot["confidence"], boot["seed"])
        out["paired_diff_c1_minus_c0"][m] = {k: res[k] for k in
                                             ("estimate", "ci_low", "ci_high", "n_resamples_valid")}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Execução dos VLMs (opcional) com retomada por imagem
# ─────────────────────────────────────────────────────────────────────────────

def build_default_graph(cfg: Dict):
    """Constrói o grafo existente em modo 'precomputed', espelhando scripts/run_pipeline.py."""
    from src.agents.agents import build_graph
    params = inspect.signature(build_graph).parameters
    llm = cfg.get("_llm", {})
    kwargs = dict(model_name=llm["model_name"], temperature=llm.get("temperature", 0),
                  seg_threshold=cfg["segmentation"]["threshold"],
                  segmentation_mode="precomputed", seg_model=None)
    retriever = None
    if "enable_critical_reviewer" in params:   # versão com Feature Extraction (feature/feat_extraction)
        fe = cfg.get("_feature_extraction", {})
        kwargs.update(experiment_condition=fe.get("experiment_condition", "image_only"),
                      feature_extraction_config=fe,
                      enable_critical_reviewer=bool(fe.get("enable_critical_reviewer", False)))
        need_rag = kwargs["enable_critical_reviewer"]
    else:                                       # versão legada: o crítico RAG é incondicional
        need_rag = True
    if need_rag:
        from src.data.preprocess import build_rag
        r, p = cfg["_rag"], cfg["paths"]
        retriever = build_rag(p["medical_texts"], r["embedding_model"], r["chunk_size"], r["chunk_overlap"])
    return build_graph(retriever=retriever, **kwargs)


def _read_condition_file(path: Path, inv_fp: str, force: bool) -> Dict:
    if not path.exists():
        return {"invariants_fp": inv_fp, "results": {}, "mask_sha256": {}}
    data = read_json(str(path))
    if data.get("invariants_fp") != inv_fp and not force:
        raise ArtifactMismatch(
            f"{path} foi gerado com agentes/prompts/config diferentes dos atuais. "
            "Use --force para refazer as chamadas aos VLMs.")
    if data.get("invariants_fp") != inv_fp:
        return {"invariants_fp": inv_fp, "results": {}, "mask_sha256": {}}
    return data


def run_vlm_condition(cfg: Dict, condition: str, entries: Sequence[ImageEntry], out_dir: Path,
                      graph_factory: Callable, inv_fp: str, force: bool = False,
                      pipeline_runner: Optional[Callable] = None) -> Dict:
    """Executa (ou retoma) uma condição; só chama o grafo para imagens pendentes."""
    path = out_dir / f"vlm_{condition}.json"
    data = _read_condition_file(path, inv_fp, force)
    pending = []
    for e in entries:
        sha = sha256_file(e[3])
        done = e[0] in data["results"] and data["mask_sha256"].get(e[0]) == sha
        if not done:
            pending.append(e)
    if not pending:
        logger.info("[%s] todas as %d imagens já processadas; nenhuma chamada a VLM.", condition, len(entries))
        return data
    logger.info("[%s] %d imagens pendentes de %d.", condition, len(pending), len(entries))

    if pipeline_runner is None:
        from src.pipeline.pipeline import run_pipeline
        graph = graph_factory()
        tmp_dir = out_dir / f"_run_{condition}"
        llm = cfg["_llm"]
        legacy_out = cfg["_output_legacy"]

        def pipeline_runner(entries_):  # noqa: E306
            prefix = f"{cfg['multiagent']['vlm_results_prefix']}_{condition}"
            res_path = run_pipeline(graph, list(entries_), str(tmp_dir), llm["model_name"],
                                    filename_prefix=prefix,
                                    date_format=legacy_out.get("date_format", "%Y%m%d"))
            return read_json(res_path)

    new = pipeline_runner(pending)
    data["results"].update(new)
    for e in pending:
        if e[0] in new:
            data["mask_sha256"][e[0]] = sha256_file(e[3])
    write_json(str(path), data)
    return data


def agent_rows(condition: str, results: Dict[str, Dict]) -> List[Dict]:
    rows = []
    for image_path, event in results.items():
        for agent in AGENT_FIELDS:
            if agent in event:
                rows.append({"image_id": Path(image_path).stem, "condition": condition, "agent": agent,
                             "output_json": json_dumps(event[agent])})
    return rows


def json_dumps(obj) -> str:
    import json
    return json.dumps(clean(obj), ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Orquestração da etapa
# ─────────────────────────────────────────────────────────────────────────────

def run_multiagent(
    cfg: Dict, store: StageStore, selected: Optional[str] = None, run_vlm: Optional[bool] = None,
    graph_factory: Optional[Callable] = None, feature_extractor: Optional[Callable] = None,
    pipeline_runner: Optional[Callable] = None,
) -> Dict:
    lay = Layout(cfg)
    ma = cfg["multiagent"]
    run_vlm = ma["run_vlm"] if run_vlm is None else run_vlm
    if selected is None:
        sel = lay.results / "comparison" / "selection.json"
        if not sel.exists():
            raise DataError("Seleção ausente. Execute 'compare' ou informe --strategy.")
        selected = read_json(str(sel))["selected"]
    per_image = lay.results / selected / "per_image.csv"
    if not per_image.exists():
        raise DataError(f"Resultados da estratégia '{selected}' ausentes: {per_image}")
    seg_rows = {r["image_id"]: r for r in read_csv(str(per_image))}

    manifest, by_id = _load_folds_and_records(cfg, lay)
    ids = [im["image_id"] for im in manifest["images"]]
    out = lay.results / "multiagent"
    conds = build_condition_entries(ids, by_id, str(lay.predictions / selected),
                                    ma["empty_mask_policy"], str(out / "fallback_masks"))
    inv = condition_invariants(cfg)
    inv_fp = fingerprint(inv)
    write_json(str(out / "invariants.json"), {
        "invariants": inv, "fingerprint": inv_fp, "selected_segmenter": selected,
        "flagged_images": conds["flagged"], "empty_mask_policy": ma["empty_mask_policy"],
        "varying_factor": "origem da máscara (seg_path): C0=ground truth, C1=predição fora da amostra"})

    result: Dict[str, Any] = {"selected_segmenter": selected, "n_images": len(ids),
                              "n_flagged": len(conds["flagged"]), "flagged": conds["flagged"],
                              "vlm_executed": False}

    # 1) features dependentes da máscara (sem VLM, sempre recalculadas)
    extractor, why = (feature_extractor, None) if feature_extractor else resolve_callable(ma["feature_extractor"])
    if extractor is None:
        result["features"] = {"available": False, "reason": why}
    else:
        fe_cfg = cfg.get("_feature_extraction", {})
        f0 = extract_features(extractor, conds["C0"], fe_cfg, "C0")
        f1 = extract_features(extractor, conds["C1"], fe_cfg, "C1")
        write_json(str(out / "features_C0.json"), f0)
        write_json(str(out / "features_C1.json"), f1)
        comp = compare_features(f0, f1, seg_rows)
        write_csv(str(out / "feature_comparison.csv"), comp)
        result["features"] = {"available": True, "n_features": len(comp),
                              "n_errors_c0": sum(1 for v in f0.values() if v["error"]),
                              "n_errors_c1": sum(1 for v in f1.values() if v["error"])}

    # 2) VLMs (explícito)
    diag: Optional[Dict] = None
    if run_vlm:
        factory = graph_factory or (lambda: build_default_graph(cfg))
        data = {c: run_vlm_condition(cfg, c, conds[c], out, factory, inv_fp, store.force, pipeline_runner)
                for c in CONDITIONS}
        result["vlm_executed"] = True
        write_csv(str(out / "agent_outputs.csv"),
                  agent_rows("C0", data["C0"]["results"]) + agent_rows("C1", data["C1"]["results"]))
        diag = _diagnostics(cfg, conds, data, seg_rows, out)
    else:
        existing = [out / f"vlm_{c}.json" for c in CONDITIONS]
        if all(p.exists() for p in existing):
            data = {c: read_json(str(p)) for c, p in zip(CONDITIONS, existing)}
            diag = _diagnostics(cfg, conds, data, seg_rows, out)
    result["diagnostics"] = diag if diag else {
        "available": False, "reason": "VLMs não executados (use --run-vlm) e sem resultados prévios."}
    write_json(str(out / "summary.json"), result)
    return result


def _diagnostics(cfg: Dict, conds: Dict, data: Dict, seg_rows: Dict, out: Path) -> Dict:
    ma_boot = cfg["evaluation"]["bootstrap"]
    dec: Dict[str, Dict[str, Tuple[int, float, str]]] = {}
    for c in CONDITIONS:
        dec[c] = {}
        for path, event in data[c]["results"].items():
            d = decision_from_event(event)
            if d is not None:
                dec[c][Path(path).stem] = (int(event["real_value"]), d[1], d[0])
    common = sorted(set(dec["C0"]) & set(dec["C1"]))
    valid = [i for i in common if dec["C0"][i][0] in (0, 1)]
    if not valid:
        return {"available": False, "reason": "Nenhuma imagem com decisão reconhecível em C0 e C1."}
    y = np.array([dec["C0"][i][0] for i in valid])
    s0, s1 = (np.array([dec[c][i][1] for i in valid]) for c in CONDITIONS)
    p0, p1 = (np.array([dec[c][i][2] for i in valid], dtype=int) for c in CONDITIONS)
    res = diagnostic_comparison(y, p0, s0, p1, s1, ma_boot)
    res["available"] = True
    res["n_images_excluded"] = len(conds["C0"]) - len(valid)
    res["decision_scope"] = ("parecer final do SummaryAgent (validation.diagnostico); os agentes "
                             "ABCD/Menzies/SPCL não possuem decisão binária no esquema atual e são "
                             "persistidos como saídas brutas em agent_outputs.csv.")

    changed = np.array([i for i, (a, b) in zip(valid, zip(p0, p1)) if a != b])
    kept = np.array([i for i, (a, b) in zip(valid, zip(p0, p1)) if a == b])
    def dice_of(ids_):
        return [float(seg_rows[i]["dice"]) for i in ids_ if seg_rows.get(i) and seg_rows[i]["dice"] not in (None, "")]
    res["dice_by_decision_change"] = {
        "changed": {"n": int(len(changed)), "mean_dice": float(np.mean(dice_of(changed))) if len(changed) and dice_of(changed) else None},
        "unchanged": {"n": int(len(kept)), "mean_dice": float(np.mean(dice_of(kept))) if len(kept) and dice_of(kept) else None},
    }
    write_json(str(out / "diagnostic_metrics.json"), res)
    write_csv(str(out / "decisions_by_image.csv"), [
        {"image_id": i, "label": int(dec["C0"][i][0]), "decision_C0": dec["C0"][i][2],
         "decision_C1": dec["C1"][i][2], "changed": int(dec["C0"][i][2] != dec["C1"][i][2])}
        for i in valid])
    return res
