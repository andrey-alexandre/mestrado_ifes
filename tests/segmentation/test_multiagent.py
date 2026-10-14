"""Avaliação multiagente C0 (GT) vs C1 (predita): invariantes, features, VLM injetado, retomada."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.segmentation import artifacts as A
from src.segmentation import data as D
from src.segmentation import multiagent as MA
from src.segmentation import stages


def _selected(cfg):
    return A.read_json(str(stages.Layout(cfg).results / "comparison" / "selection.json"))["selected"]


def _fake_extractor(calls):
    """Feature dependente da máscara: área e centróide; registra cada chamada."""
    def extractor(image_path, mask_path, config=None):
        calls.append((Path(image_path).stem, mask_path, json.dumps(config, sort_keys=True)))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        ys, xs = np.nonzero(mask)
        return {"abcd": {"area": float(mask.sum()),
                         "centroid_x": float(xs.mean()) if xs.size else 0.0,
                         "flag": True, "label": "text"}}
    return extractor


def _runner(decisions, calls):
    """Executor de VLM falso: `decisions[(condition_marker, stem)] -> diagnóstico`."""
    def run(entries):
        calls.append(list(entries))
        out = {}
        for image_path, _, target, seg_path in entries:
            stem = Path(image_path).stem
            cond = "C0" if seg_path.endswith("_lesion.bmp") else "C1"
            diag = decisions(cond, stem, target)
            out[image_path] = {
                "real_value": target, "diagnosis_abcd": {"assimetria": "1"},
                "diagnosis_menzies": {"positivas": {}}, "diagnosis_spcl": {"vasos_atipicos": "não"},
                "validation": {"diagnostico": diag, "justificativa": "j", "recomendacoes": "r"},
            }
        return out
    return run


# ── construção das condições ────────────────────────────────────────────────────

def test_conditions_differ_only_by_mask_source(full_run):
    cfg, _ = full_run
    lay = stages.Layout(cfg)
    manifest, by_id = stages._load_folds_and_records(cfg, lay)
    ids = [im["image_id"] for im in manifest["images"]]
    conds = MA.build_condition_entries(ids, by_id, str(lay.predictions / _selected(cfg)))
    c0, c1 = conds["C0"], conds["C1"]
    assert len(c0) == len(c1) == 30 and not conds["flagged"]
    for a, b in zip(c0, c1):
        assert a[:3] == b[:3]                       # imagem, tamanho da lesão e alvo idênticos
        assert a[3].endswith("_lesion.bmp") and b[3].endswith(".png") and a[3] != b[3]
    # invariantes (agentes, prompts, config de FE, LLM) são únicos e estáveis
    assert A.fingerprint(MA.condition_invariants(cfg)) == A.fingerprint(MA.condition_invariants(cfg))


def test_empty_prediction_policies(full_run):
    cfg, _ = full_run
    lay = stages.Layout(cfg)
    manifest, by_id = stages._load_folds_and_records(cfg, lay)
    ids = [im["image_id"] for im in manifest["images"]]
    pred_dir = lay.predictions / _selected(cfg)
    D.save_mask_png(np.zeros((40, 56), bool), str(pred_dir / f"{ids[0]}.png"))

    skip = MA.build_condition_entries(ids, by_id, str(pred_dir), "flag_and_skip")
    assert ids[0] in skip["flagged"] and len(skip["C0"]) == 30 and len(skip["C1"]) == 29

    full = MA.build_condition_entries(ids, by_id, str(pred_dir), "full_image",
                                      str(lay.results / "fb"))
    assert len(full["C1"]) == 30 and full["flagged"][ids[0]].endswith("full_image")
    assert D.load_mask_bool(full["C1"][0][3]).all()


# ── apenas segmentação: sem VLM, features recalculadas ─────────────────────────

def test_segmentation_only_mode_never_calls_vlm_and_recomputes_features(full_run):
    cfg, store = full_run
    calls = []

    def graph_factory():
        raise AssertionError("o grafo/VLM não deveria ser construído")

    res = MA.run_multiagent(cfg, store, run_vlm=False, graph_factory=graph_factory,
                            feature_extractor=_fake_extractor(calls))
    assert res["vlm_executed"] is False and res["diagnostics"]["available"] is False
    assert len(calls) == 60                         # 30 em C0 + 30 em C1, nada reaproveitado
    c0_masks = {c[1] for c in calls[:30]}
    c1_masks = {c[1] for c in calls[30:]}
    assert all(m.endswith("_lesion.bmp") for m in c0_masks)
    assert all(m.endswith(".png") for m in c1_masks) and not c0_masks & c1_masks
    assert len({c[2] for c in calls}) == 1          # mesma configuração de extração nas duas condições

    MA.run_multiagent(cfg, store, run_vlm=False, feature_extractor=_fake_extractor(calls))
    assert len(calls) == 120                        # nova execução recalcula (sem cache cruzado)

    out = stages.Layout(cfg).results / "multiagent"
    comp = A.read_csv(str(out / "feature_comparison.csv"))
    names = {r["feature"] for r in comp}
    assert names == {"abcd.area", "abcd.centroid_x"}   # bool/str descartados
    area = next(r for r in comp if r["feature"] == "abcd.area")
    assert float(area["mean_abs_diff"]) > 0          # máscara prevista ≠ GT altera a feature
    assert (out / "features_C0.json").exists() and (out / "features_C1.json").exists()
    assert (out / "invariants.json").exists()


def test_features_unavailable_is_reported_not_fabricated(full_run):
    cfg, store = full_run
    cfg = json.loads(json.dumps(cfg))
    cfg["multiagent"]["feature_extractor"] = "modulo.que.nao.existe:fn"
    res = MA.run_multiagent(cfg, store, run_vlm=False)
    assert res["features"]["available"] is False and "ModuleNotFoundError" in res["features"]["reason"]


# ── VLM injetado: persistência, métricas e retomada ────────────────────────────

def test_vlm_flow_metrics_persistence_and_resume(full_run):
    cfg, store = full_run
    calls = []
    flips = set()

    def decisions(cond, stem, target):
        truth = "Suspeita de Melanoma" if target == 1 else "Lesão Benigna"
        if cond == "C1" and stem in flips:
            return "Lesão Benigna" if target == 1 else "Suspeita de Melanoma"
        return truth

    manifest = A.read_json(str(stages.Layout(cfg).folds_manifest))
    ids = [im["image_id"] for im in manifest["images"]]
    flips.update(ids[:3])

    res = MA.run_multiagent(cfg, store, run_vlm=True, graph_factory=lambda: None,
                            feature_extractor=_fake_extractor([]), pipeline_runner=_runner(decisions, calls))
    d = res["diagnostics"]
    assert d["available"] and d["n_images"] == 30
    assert d["C0"]["accuracy"] == 1.0 and d["C0"]["sensitivity"] == 1.0
    assert d["n_changed"] == 3 and d["change_rate"] == pytest.approx(0.1)
    assert d["C1"]["accuracy"] == pytest.approx(27 / 30)
    assert d["cohen_kappa"] < 1.0
    lo, hi = d["paired_diff_c1_minus_c0"]["accuracy"]["ci_low"], d["paired_diff_c1_minus_c0"]["accuracy"]["ci_high"]
    assert lo <= -0.1 + 1e-9 <= hi + 1e-9 or lo <= d["paired_diff_c1_minus_c0"]["accuracy"]["estimate"] <= hi

    # C0 e C1: mesmas imagens, só a máscara muda
    assert len(calls) == 2
    c0, c1 = calls
    assert [e[0] for e in c0] == [e[0] for e in c1]
    assert all(a[3] != b[3] for a, b in zip(c0, c1))

    out = stages.Layout(cfg).results / "multiagent"
    rows = A.read_csv(str(out / "agent_outputs.csv"))
    assert len(rows) == 30 * 2 * 4                   # imagem × condição × agente
    assert {r["condition"] for r in rows} == {"C0", "C1"}
    assert {r["agent"] for r in rows} == {"diagnosis_abcd", "diagnosis_menzies", "diagnosis_spcl",
                                          "validation"}
    assert len(A.read_csv(str(out / "decisions_by_image.csv"))) == 30

    # retomada: nenhuma nova chamada a VLM
    MA.run_multiagent(cfg, store, run_vlm=True, graph_factory=lambda: None,
                      feature_extractor=_fake_extractor([]), pipeline_runner=_runner(decisions, calls))
    assert len(calls) == 2

    # máscara C1 alterada para uma imagem -> apenas ela é reprocessada em C1
    pred = stages.Layout(cfg).predictions / _selected(cfg) / f"{ids[5]}.png"
    D.save_mask_png(D.load_mask_bool(str(pred)) ^ np.eye(40, 56, dtype=bool), str(pred))
    MA.run_multiagent(cfg, store, run_vlm=True, graph_factory=lambda: None,
                      feature_extractor=_fake_extractor([]), pipeline_runner=_runner(decisions, calls))
    assert len(calls) == 3 and len(calls[2]) == 1 and Path(calls[2][0][0]).stem == ids[5]


def test_changed_invariants_block_resume(full_run, tmp_path):
    cfg, store = full_run
    calls = []
    run = _runner(lambda c, s, t: "Lesão Benigna", calls)
    MA.run_multiagent(cfg, store, run_vlm=True, graph_factory=lambda: None,
                      feature_extractor=_fake_extractor([]), pipeline_runner=run)
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["_llm"]["model_name"] = "outro-modelo"
    with pytest.raises(A.ArtifactMismatch, match="--force"):
        MA.run_multiagent(cfg2, store, run_vlm=True, graph_factory=lambda: None,
                          feature_extractor=_fake_extractor([]), pipeline_runner=run)


# ── métricas diagnósticas (unidade) ─────────────────────────────────────────────

def test_binary_metrics_and_undefined_values():
    y = np.array([1, 1, 0, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 0, 1])
    m = MA.binary_metrics(y, p)
    assert (m["tp"], m["tn"], m["fp"], m["fn"]) == (2, 2, 1, 1)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    undefined = MA.binary_metrics(np.array([0, 0]), np.array([0, 0]))
    assert np.isnan(undefined["sensitivity"]) and undefined["specificity"] == 1.0


def test_diagnostic_comparison_kappa_and_degenerate():
    boot = {"n_resamples": 200, "confidence": 0.95, "seed": 0}
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    p0 = y.copy()
    p1 = y.copy()
    p1[0] = 0
    s0, s1 = p0 * 1.0, p1 * 1.0
    r = MA.diagnostic_comparison(y, p0, s0, p1, s1, boot)
    assert r["change_rate"] == pytest.approx(1 / 8)
    assert 0 < r["cohen_kappa"] < 1
    assert r["C0"]["roc_auc_ordinal"] == 1.0
    const = np.zeros(8, int)
    r2 = MA.diagnostic_comparison(y, const, const * 1.0, const, const * 1.0, boot)
    assert np.isnan(r2["cohen_kappa"])               # kappa indefinido, tratado explicitamente


def test_decision_mapping_reuses_existing_diagnostic_logic():
    ev = {"validation": {"diagnostico": "Lesão Indeterminada"}}
    assert MA.decision_from_event(ev)[0] == 1        # conservador, como em evaluate.py
    assert MA.decision_from_event({"validation": {"diagnostico": "Lesão Benigna"}})[0] == 0
    assert MA.decision_from_event({"validation": {"diagnostico": "???"}}) is None
    assert MA.decision_from_event({}) is None


def test_flatten_numeric_keeps_only_numbers():
    flat = MA.flatten_numeric({"a": {"b": 1, "c": True, "d": "x", "e": [1.5, 2]}, "f": None})
    assert flat == {"a.b": 1.0, "a.e.0": 1.5, "a.e.1": 2.0}
