"""Checkpoints, fine-tuning, etapas ponta a ponta com dados sintéticos, retomada e relatório."""
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.segmentation import artifacts as A
from src.segmentation import comparison, report, stages
from src.segmentation import data as D
from src.segmentation.strategies import CheckpointError, build_strategy, state_dict_sha256


# ── helpers ─────────────────────────────────────────────────────────────────────

def _train_strategy(cfg, isic, n=16):
    recs = D.pair_images_and_masks(str(isic[0]), str(isic[1]))[:n]
    s = build_strategy(cfg)
    ds = D.SegmentationDataset(recs, s.transform(train=True))
    val = D.SegmentationDataset(recs[:4], s.transform(train=False))
    s.train(ds, val)
    return s, recs


# ── checkpoints ─────────────────────────────────────────────────────────────────

def test_checkpoint_roundtrip_preserves_weights(cfg_factory, make_isic, tmp_path):
    isic = make_isic(8)
    cfg = cfg_factory(isic=isic)
    s, _ = _train_strategy(cfg, isic, 8)
    path = str(tmp_path / "ck.pt")
    sha = s.save_checkpoint(path, metadata={"k": 1})
    s2 = build_strategy(cfg)
    info = s2.load_checkpoint(path)
    assert info["file_sha256"] == sha and not info["legacy"] and info["metadata"] == {"k": 1}
    assert state_dict_sha256(s2.model) == state_dict_sha256(s.model)


def test_legacy_raw_state_dict_checkpoint_loads(cfg_factory, tmp_path):
    cfg = cfg_factory()
    s = build_strategy(cfg)
    model = s.ensure_model()
    legacy = tmp_path / "U_Net-legacy.pkl"
    torch.save(model.state_dict(), legacy)     # formato do checkpoint existente do projeto
    s2 = build_strategy(cfg)
    assert s2.load_checkpoint(str(legacy))["legacy"] is True
    assert state_dict_sha256(s2.model) == state_dict_sha256(model)


def test_checkpoint_errors_are_explicit(cfg_factory, tmp_path):
    cfg = cfg_factory()
    s = build_strategy(cfg)
    with pytest.raises(CheckpointError, match="não encontrado"):
        s.load_checkpoint(str(tmp_path / "x.pt"))
    s.ensure_model()
    p = str(tmp_path / "ck.pt")
    s.save_checkpoint(p)
    other = build_strategy(cfg)
    other.params = {**other.params, "model_name": "R2U_Net"}
    with pytest.raises(CheckpointError, match="strategy_params"):
        other.load_checkpoint(p)
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {"w": torch.zeros(1)}, "format_version": 1, "strategy": "unet",
                "strategy_params_json": json.dumps(s.params, sort_keys=True), "metadata_json": "{}"}, bad)
    with pytest.raises(CheckpointError, match="incompatível"):
        build_strategy(cfg).load_checkpoint(str(bad))


# ── política de congelamento ────────────────────────────────────────────────────

def test_frozen_blocks_do_not_change_and_unfrozen_do(cfg_factory, make_isic, tmp_path):
    isic = make_isic(12)
    cfg = cfg_factory(isic=isic)
    s, recs = _train_strategy(cfg, isic, 12)
    ck = str(tmp_path / "base.pt")
    s.save_checkpoint(ck)

    s2 = build_strategy(cfg)
    s2.load_checkpoint(ck)
    before = {k: v.clone() for k, v in s2.model.state_dict().items()}
    tl = dict(cfg["transfer_learning"], frozen_encoder_blocks=["Conv1", "Conv2"], max_epochs=2)
    ds = D.SegmentationDataset(recs, s2.transform(train=True))
    s2.fine_tune(ds, None, tl, seed=1)
    after = s2.model.state_dict()
    for k in before:
        if k.startswith(("Conv1.", "Conv2.")) and "num_batches_tracked" not in k:
            assert torch.equal(before[k], after[k]), f"{k} deveria estar congelado (inclui estatísticas do BN)"
    assert any(not torch.equal(before[k], after[k]) for k in before if k.startswith("Up_conv2."))


def test_gradual_unfreeze_and_invalid_block_names(cfg_factory, make_isic, tmp_path):
    isic = make_isic(8)
    cfg = cfg_factory(isic=isic)
    s, recs = _train_strategy(cfg, isic, 8)
    ck = str(tmp_path / "b.pt")
    s.save_checkpoint(ck)
    s2 = build_strategy(cfg)
    s2.load_checkpoint(ck)
    with pytest.raises(ValueError, match="Blocos inexistentes.*Disponíveis"):
        s2.apply_freeze_policy({"frozen_encoder_blocks": ["Conv9"]})

    tl = dict(cfg["transfer_learning"], frozen_encoder_blocks=["Conv1", "Conv2", "Conv3"],
              unfreeze_policy="gradual", unfreeze_order=["Conv3", "Conv2"], unfreeze_every_n_epochs=1,
              max_epochs=3)
    s2.fine_tune(D.SegmentationDataset(recs, s2.transform(train=True)), None, tl, seed=0)
    still_frozen = s2._frozen_names(s2.model)
    assert "Conv3" not in still_frozen and "Conv2" not in still_frozen and "Conv1" in still_frozen


# ── etapa 1 ─────────────────────────────────────────────────────────────────────

def test_train_isic_does_not_touch_ph2(world, monkeypatch):
    cfg, *_ = world

    def boom(*a, **k):
        raise AssertionError("PH² acessado no treino inicial")

    monkeypatch.setattr(stages, "discover_ph2", boom)
    store = A.StageStore(cfg["output"]["results_dir"])
    summary = stages.run_train_isic(cfg, store)
    lay = stages.Layout(cfg)
    assert lay.isic_checkpoint.exists() and summary["n_train"] + summary["n_val"] == 24
    split = A.read_json(str(lay.results / "isic_training" / "split.json"))
    assert not set(split["train_ids"]) & set(split["val_ids"])
    assert (lay.results / "isic_training" / "history.json").exists()


def test_stage_resume_skips_and_detects_incompatible_config(world, monkeypatch):
    cfg, *_ = world
    store = A.StageStore(cfg["output"]["results_dir"])
    stages.run_train_isic(cfg, store)

    calls = []
    real = stages.build_strategy
    monkeypatch.setattr(stages, "build_strategy", lambda c: calls.append(1) or real(c))
    stages.run_train_isic(cfg, store)           # mesmo fingerprint -> pula
    assert calls == []

    changed = json.loads(json.dumps(cfg))
    changed["segmentation"]["learning_rate"] = 0.123
    with pytest.raises(A.ArtifactMismatch, match="--force"):
        stages.run_train_isic(changed, store)
    # --force refaz
    stages.run_train_isic(changed, A.StageStore(cfg["output"]["results_dir"], force=True))
    assert calls


# ── etapas 2–5 ponta a ponta ────────────────────────────────────────────────────

def test_end_to_end_same_folds_same_images_and_out_of_sample(full_run):
    cfg, _ = full_run
    lay = stages.Layout(cfg)
    manifest = A.read_json(str(lay.folds_manifest))
    ids = [im["image_id"] for im in manifest["images"]]
    direct = A.read_csv(str(lay.results / "direct_transfer" / "per_image.csv"))
    tuned = A.read_csv(str(lay.results / "finetune" / "per_image.csv"))

    # mesmas imagens, mesma ordem, mesmos folds (reutilização do manifesto)
    assert [r["image_id"] for r in direct] == ids == [r["image_id"] for r in tuned]
    fmap = {im["image_id"]: str(im["fold"]) for im in manifest["images"]}
    assert all(r["fold"] == fmap[r["image_id"]] for r in direct + tuned)

    # predição ↔ identificador: um PNG por imagem, na resolução original
    for sub, rows in (("direct_transfer", direct), ("finetune", tuned)):
        for r in rows:
            png = lay.predictions / sub / f"{r['image_id']}.png"
            assert png.exists() and r["pred_mask_path"] == str(png)
            with Image.open(r["image_path"]) as im, Image.open(png) as pm:
                assert pm.size == im.size
            assert Path(r["gt_mask_path"]).name == f"{r['image_id']}_lesion.bmp"

    ds = A.read_json(str(lay.results / "direct_transfer" / "summary.json"))
    assert ds["weights_unchanged"] is True
    assert "não são 5 treinos" in ds["note"]
    ft = A.read_json(str(lay.results / "finetune" / "summary.json"))
    assert ft["all_rounds_started_from_same_weights"] is True
    base = A.sha256_file(str(lay.isic_checkpoint))
    assert ft["checkpoint_sha256"] == base


def test_finetune_never_sees_test_fold(world, monkeypatch):
    cfg, *_ = world
    store = A.StageStore(cfg["output"]["results_dir"])
    stages.run_train_isic(cfg, store)
    manifest = stages.run_make_folds(cfg, store)

    seen = []
    orig = stages.SegmentationDataset

    class Spy(orig):
        def __init__(self, records, transform):
            seen.append({r.image_id for r in records})
            super().__init__(records, transform)

    monkeypatch.setattr(stages, "SegmentationDataset", Spy)
    stages.run_finetune(cfg, store)
    tests = [set(r["test_ids"]) for r in manifest["rounds"]]
    assert len(seen) == 2 * len(tests)         # (treino, validação interna) por rodada
    for k, test in enumerate(tests):
        for used in seen[2 * k:2 * k + 2]:
            assert not used & test
        assert (seen[2 * k] | seen[2 * k + 1]) == set(manifest["rounds"][k]["train_ids"])


def test_weights_reset_before_each_finetune_round(world):
    cfg, *_ = world
    store = A.StageStore(cfg["output"]["results_dir"])
    stages.run_train_isic(cfg, store)
    manifest = stages.run_make_folds(cfg, store)
    lay = stages.Layout(cfg)
    manifest, by_id = stages._load_folds_and_records(cfg, lay)
    fmap = {im["image_id"]: im["fold"] for im in manifest["images"]}
    base_hash = state_dict_sha256(_loaded(cfg, str(lay.isic_checkpoint)).model)

    r0 = stages.run_finetune_round(cfg, str(lay.isic_checkpoint), manifest["rounds"][0], by_id, fmap, lay)
    r1 = stages.run_finetune_round(cfg, str(lay.isic_checkpoint), manifest["rounds"][1], by_id, fmap, lay)
    assert r0["base_weights_sha256"] == base_hash == r1["base_weights_sha256"]
    # e o fine-tuning de fato alterou os pesos da rodada 0 (sem contaminar a rodada 1)
    tuned0 = state_dict_sha256(_loaded(cfg, str(lay.finetune_checkpoint(0))).model)
    assert tuned0 != base_hash
    assert state_dict_sha256(_loaded(cfg, str(lay.isic_checkpoint)).model) == base_hash


def _loaded(cfg, path):
    s = build_strategy(cfg)
    s.load_checkpoint(path)
    return s


def test_comparison_pairs_and_selection_artifacts(full_run):
    cfg, _ = full_run
    lay = stages.Layout(cfg)
    summ = A.read_json(str(lay.results / "comparison" / "summary.json"))
    sel = A.read_json(str(lay.results / "comparison" / "selection.json"))
    assert summ["same_images"] and summ["n_images"] == 30
    dice = summ["overall"]["dice"]
    assert {"direct_transfer", "finetune", "paired_diff_mean", "paired_diff_median"} <= set(dice)
    assert set(summ["per_fold"]) == {"0", "1", "2", "3", "4"}
    assert set(summ["per_diagnosis"]) == {"common_nevus", "atypical_nevus", "melanoma"}
    assert sel["selected"] in ("direct_transfer", "finetune")
    assert "viés de seleção" in sel["selection_bias_note"]
    assert (lay.results / "comparison" / "paired_differences.csv").exists()
    # resultados das duas estratégias preservados
    assert (lay.results / "direct_transfer" / "per_image.csv").exists()
    assert (lay.results / "finetune" / "per_image.csv").exists()


def test_selection_rule_unit():
    ok = {"estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03}
    wide = {"estimate": 0.02, "ci_low": -0.01, "ci_high": 0.05}
    assert comparison.decide_selection("dice", "higher_mean", 0.8, 0.82, wide)["selected"] == "finetune"
    assert comparison.decide_selection("dice", "mean_with_ci", 0.8, 0.82, wide)["selected"] == "direct_transfer"
    assert comparison.decide_selection("dice", "mean_with_ci", 0.8, 0.82, ok)["selected"] == "finetune"
    # HD95: menor é melhor
    neg = {"estimate": -2.0, "ci_low": -3.0, "ci_high": -1.0}
    assert comparison.decide_selection("hd95", "mean_with_ci", 10.0, 8.0, neg)["selected"] == "finetune"


def test_comparison_rejects_mismatched_images(full_run):
    cfg, store = full_run
    lay = stages.Layout(cfg)
    p = lay.results / "finetune" / "per_image.csv"
    rows = A.read_csv(str(p))[:-1]
    A.write_csv(str(p), rows, stages.PER_IMAGE_FIELDS)
    with pytest.raises(D.DataError, match="mesmas imagens"):
        comparison.run_comparison(cfg, A.StageStore(cfg["output"]["results_dir"], force=True))


# ── modelo final ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("choice", ["direct_transfer", "finetune"])
def test_final_model_does_not_change_out_of_sample_predictions(full_run, choice):
    cfg, store = full_run
    lay = stages.Layout(cfg)
    before = {p.name: A.sha256_file(str(p)) for p in (lay.predictions / "finetune").glob("*.png")}
    csv_before = A.sha256_file(str(lay.results / "finetune" / "per_image.csv"))
    prov = stages.prepare_final_model(cfg, store, choice)
    assert lay.final_checkpoint.exists() and prov["selected_strategy"] == choice
    assert prov["checkpoint_sha256"] == A.sha256_file(str(lay.final_checkpoint))
    assert prov["seed"] == cfg["segmentation"]["seed"] and prov["training_history"] is not None
    assert "torch" in prov["provenance"]["versions"]
    assert "ISIC 2019" in prov["isic2019_inference"] and "não executada" in prov["isic2019_inference"]
    after = {p.name: A.sha256_file(str(p)) for p in (lay.predictions / "finetune").glob("*.png")}
    assert before == after and csv_before == A.sha256_file(str(lay.results / "finetune" / "per_image.csv"))
    if choice == "direct_transfer":
        assert prov["checkpoint_sha256"] == prov["base_checkpoint_sha256"]
        assert prov["data_provenance"]["ph2_used_for_training"] is False
    else:
        assert prov["data_provenance"]["ph2_used_for_training"] is True
        assert "em-amostra" in prov["data_provenance"]["in_sample_warning"]
        assert len(prov["training_history"]) == prov["data_provenance"]["fixed_epochs"]


# ── relatório ───────────────────────────────────────────────────────────────────

def test_report_from_real_artifacts(full_run):
    cfg, store = full_run
    stages.prepare_final_model(cfg, store, "finetune")
    path = Path(report.build_report(cfg))
    text = path.read_text(encoding="utf-8")
    for n in range(1, 15):
        assert f"## {n}. " in text
    assert "viés de seleção" in text
    assert "indisponíveis" in text and "multiagent" in text      # etapa 6 não foi executada
    assert "nan" not in text.lower().replace("finance", "")      # nada de NaN literal
    fig_dir = path.parent / "figures"
    assert any(fig_dir.glob("dist_dice.png")) and any(fig_dir.glob("diff_dice.png"))
    assert (path.parent / "report_index.json").exists()
    # números do relatório vêm dos artefatos
    summ = A.read_json(str(stages.Layout(cfg).results / "comparison" / "summary.json"))
    assert f"{summ['overall']['dice']['direct_transfer']['mean']:.4f}" in text


def test_report_with_no_artifacts_states_unavailability(cfg_factory):
    cfg = cfg_factory()
    text = Path(report.build_report(cfg)).read_text(encoding="utf-8")
    assert text.count("indisponíveis") >= 6
    assert "Sem conclusão" in text
    assert "0.0000" not in text    # sem números fabricados
