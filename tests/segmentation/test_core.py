"""Factory, configuração, pareamento, transformações, folds, métricas e bootstrap."""
import copy

import numpy as np
import pytest
import torch
from PIL import Image

from src.segmentation import config as cfgmod
from src.segmentation import data as D
from src.segmentation import folds as F
from src.segmentation import metrics as M
from src.segmentation.strategies import (
    SegmentationStrategy, available_strategies, build_strategy, register_strategy,
)
from src.segmentation.strategies.unet import UNetStrategy


# ── Factory / estratégias ───────────────────────────────────────────────────────

def test_factory_instantiates_unet(cfg_factory):
    s = build_strategy(cfg_factory())
    assert isinstance(s, UNetStrategy)
    model = s.build_model()
    assert isinstance(model, torch.nn.Module)
    assert set(s.block_groups(model)) == {"encoder", "decoder"}


def test_factory_unknown_strategy_has_clear_message(cfg_factory):
    cfg = cfg_factory()
    cfg["segmentation"]["strategy"] = "nao_existe"
    with pytest.raises(ValueError, match="desconhecida.*unet"):
        build_strategy(cfg)


def test_new_strategy_plugs_in_without_touching_pipeline(cfg_factory):
    @register_strategy
    class Dummy(UNetStrategy):
        name = "dummy_unet"

    assert "dummy_unet" in available_strategies()
    cfg = cfg_factory()
    cfg["segmentation"]["strategy"] = "dummy_unet"
    assert isinstance(build_strategy(cfg), SegmentationStrategy)


def test_image_size_must_match_architecture_multiple(cfg_factory):
    cfg = cfg_factory()
    cfg["segmentation"]["image_size"] = 30
    with pytest.raises(ValueError, match="múltiplo de 16"):
        build_strategy(cfg)


# ── Configuração ────────────────────────────────────────────────────────────────

def test_config_defaults_and_ph2_fallback_to_legacy_paths():
    full = {"paths": {"ph2_dataset_dir": "/x/PH2", "ph2_metadata_txt": "/x/m.txt"}}
    cfg = cfgmod.build_config(full)
    assert cfg["datasets"]["ph2"]["dataset_dir"] == "/x/PH2"
    assert cfg["datasets"]["ph2"]["metadata_txt"] == "/x/m.txt"
    assert cfg["evaluation"]["n_folds"] == 5 and cfg["evaluation"]["primary_metric"] == "dice"
    assert cfg["output"]["predictions_dir"].endswith("predictions")


@pytest.mark.parametrize("section,patch", [
    ("segmentation", {"optimizer": "rmsprop"}),
    ("segmentation", {"threshold": 1.5}),
    ("segmentation", {"weights_init": "checkpoint"}),          # sem init_checkpoint
    ("evaluation", {"primary_metric": "dice", "secondary_metrics": ["dice"]}),
    ("evaluation", {"n_folds": 1}),
    ("transfer_learning", {"final_epochs": "abc"}),
])
def test_invalid_configs_are_rejected(section, patch):
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.build_config({cfgmod.SECTION: {section: patch}})


def test_unknown_section_key_rejected():
    with pytest.raises(cfgmod.ConfigError, match="desconhecidas"):
        cfgmod.build_config({cfgmod.SECTION: {"dataset": {}}})


def test_missing_required_fields_reported_by_name():
    cfg = cfgmod.build_config({})
    with pytest.raises(cfgmod.ConfigError) as e:
        cfgmod.require(cfg, ["datasets.isic2018.image_dir", "datasets.isic2018.mask_dir"], "treino")
    assert "datasets.isic2018.image_dir" in str(e.value) and "treino" in str(e.value)


# ── Pareamento imagem-máscara ───────────────────────────────────────────────────

def test_pairing_ok(make_isic):
    img, mask, _ = make_isic(6)
    recs = D.pair_images_and_masks(str(img), str(mask))
    assert [r.image_id for r in recs] == [f"ISIC_{i:07d}" for i in range(6)]
    assert all(r.image_path.endswith(".jpg") and r.mask_path.endswith("_segmentation.png") for r in recs)


def test_pairing_rejects_unpaired_and_can_drop(make_isic):
    img, mask, _ = make_isic(5)
    (mask / "ISIC_0000002_segmentation.png").unlink()
    with pytest.raises(D.DataError, match="sem máscara"):
        D.pair_images_and_masks(str(img), str(mask))
    recs = D.pair_images_and_masks(str(img), str(mask), allow_unpaired=True)
    assert len(recs) == 4 and "ISIC_0000002" not in {r.image_id for r in recs}


def test_pairing_rejects_duplicate_ids(make_isic):
    img, mask, _ = make_isic(3)
    Image.new("RGB", (8, 8)).save(img / "ISIC_0000001.png")  # mesmo id, outra extensão
    with pytest.raises(D.DataError, match="duplicado"):
        D.pair_images_and_masks(str(img), str(mask))


def test_split_has_no_lesion_leak_and_exclusions(make_isic):
    img, mask, groups = make_isic(20, with_groups=True)
    recs = D.pair_images_and_masks(str(img), str(mask))
    g = D.read_lesion_groups(str(groups))
    recs, summary = D.apply_groups_and_exclusions(recs, g, {"ISIC_0000000"})
    assert summary["n_excluded_overlap"] == 1 and summary["group_info_available"]
    tr, va = D.split_train_val(recs, 0.3, seed=3)
    assert {r.group for r in tr}.isdisjoint({r.group for r in va})
    assert "ISIC_0000000" not in {r.image_id for r in tr + va}
    assert len(tr) + len(va) == 19


def test_leak_assertion_detects_shared_lesion():
    a = [D.Record("a", "a.jpg", "a.png", group="L1")]
    b = [D.Record("b", "b.jpg", "b.png", group="L1")]
    with pytest.raises(D.DataError, match="Lesões"):
        D.assert_no_group_leak(a, b)


# ── PH²: rótulos ────────────────────────────────────────────────────────────────

def test_ph2_labels_use_clinical_diagnosis_and_report_discrepancy(make_ph2):
    root, meta = make_ph2()
    recs, summary = D.discover_ph2(str(root), str(meta))
    assert len(recs) == 30 and summary["n_melanoma"] == 10
    assert summary["clinical_rule_positives"] == 10
    assert summary["legacy_histological_rule_positives"] < 10  # histologia parcialmente vazia
    assert len(summary["label_discrepancy_ids"]) > 0
    assert {r.diagnosis for r in recs} == {"common_nevus", "atypical_nevus", "melanoma"}


# ── Transformações sincronizadas ────────────────────────────────────────────────

def _shape_pair(size=64):
    yy, xx = np.mgrid[:size, :size]
    m = ((yy - 40) ** 2 / 400 + (xx - 22) ** 2 / 100) <= 1  # elipse assimétrica
    img = np.zeros((size, size, 3), np.uint8)
    img[m] = 255
    return Image.fromarray(img), Image.fromarray(m.astype(np.uint8) * 255)


def test_flips_are_synchronized_exactly():
    img, mask = _shape_pair()
    aug = {"hflip": True, "vflip": True, "rotation_degrees": 0, "color_jitter": 0}
    tf = D.JointTransform(64, train=True, augmentation=aug)
    for seed in range(12):
        torch.manual_seed(seed)
        it, mt = tf(img, mask)
        assert torch.equal((it[0] > 0.5).float(), mt[0]), f"seed {seed}"


def test_rotation_is_synchronized_and_mask_stays_binary():
    img, mask = _shape_pair()
    aug = {"hflip": False, "vflip": False, "rotation_degrees": 45, "color_jitter": 0}
    tf = D.JointTransform(64, train=True, augmentation=aug)
    for seed in range(8):
        torch.manual_seed(seed)
        it, mt = tf(img, mask)
        assert set(mt.unique().tolist()) <= {0.0, 1.0}   # nearest: sem valores intermediários
        a, b = (it[0] > 0.5), (mt[0] > 0.5)
        iou = (a & b).sum().item() / max((a | b).sum().item(), 1)
        assert iou > 0.9


def test_eval_transform_has_no_augmentation():
    img, mask = _shape_pair()
    tf = D.JointTransform(32, train=False, augmentation={"hflip": True, "vflip": True,
                                                         "rotation_degrees": 40, "color_jitter": 0.3})
    torch.manual_seed(0)
    a = tf(img, mask)
    torch.manual_seed(99)
    b = tf(img, mask)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_train_transform_never_modifies_mask_interpolation_values():
    img, mask = _shape_pair()
    tf = D.JointTransform(32, train=True, augmentation={"hflip": True, "rotation_degrees": 30,
                                                        "color_jitter": 0.2})
    _, mt = tf(img, mask)
    assert set(mt.unique().tolist()) <= {0.0, 1.0}


# ── Folds ───────────────────────────────────────────────────────────────────────

def _recs(n_pos=40, n_neg=160):
    out = [D.Record(f"P{i:03d}", f"p{i}", f"m{i}", label=1, diagnosis="melanoma") for i in range(n_pos)]
    half = n_neg // 2
    out += [D.Record(f"N{i:03d}", f"n{i}", f"m{i}", label=0,
                     diagnosis="common_nevus" if i < half else "atypical_nevus") for i in range(n_neg)]
    return out


def test_five_stratified_folds_cover_each_image_once():
    m = F.make_folds(_recs(), n_folds=5, seed=42)
    assert m["n_images"] == 200 and m["n_positive"] == 40
    tests = [i for r in m["rounds"] for i in r["test_ids"]]
    assert len(tests) == 200 and len(set(tests)) == 200
    for r in m["rounds"]:
        pos = sum(1 for i in r["test_ids"] if i.startswith("P"))
        assert pos == 8 and len(r["test_ids"]) == 40
        assert not set(r["train_ids"]) & set(r["test_ids"])      # sem vazamento


def test_folds_are_deterministic_and_seed_sensitive():
    recs = _recs()
    a = F.make_folds(recs, 5, 42)
    b = F.make_folds(list(reversed(recs)), 5, 42)   # ordem de entrada não importa
    c = F.make_folds(recs, 5, 43)
    assert a["images"] == b["images"]
    assert a["images"] != c["images"]


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    m = F.make_folds(_recs(10, 40), 5, 1)
    p = tmp_path / "folds.json"
    F.save_manifest(m, str(p))
    assert F.load_manifest(str(p))["ids_sha256"] == m["ids_sha256"]
    bad = copy.deepcopy(m)
    bad["rounds"][0]["train_ids"].append(bad["rounds"][0]["test_ids"][0])
    with pytest.raises(D.DataError, match="sobreposição"):
        F.validate_manifest(bad)


def test_folds_reject_too_few_samples():
    with pytest.raises(D.DataError, match="menor classe"):
        F.make_folds(_recs(3, 40), 5, 1)


# ── Métricas ────────────────────────────────────────────────────────────────────

def _square(n=40, y=10, x=10, s=10):
    m = np.zeros((n, n), bool)
    m[y:y + s, x:x + s] = True
    return m


def test_dice_iou_known_values():
    a, b = _square(), _square(x=15)       # sobreposição de 5x10 = 50 de 100
    m = M.segmentation_metrics(a, b)
    assert m["dice"] == pytest.approx(0.5) and m["iou"] == pytest.approx(1 / 3)
    assert m["degenerate"] == "ok"
    p = M.segmentation_metrics(a, a)
    assert p["dice"] == 1.0 and p["iou"] == 1.0 and p["hd95"] == 0.0


def test_hd95_of_shifted_squares():
    a, b = _square(), _square(x=13)       # deslocamento de 3 px
    assert M.hd95(a, b) == pytest.approx(3.0, abs=1.0)
    assert M.hd95(a, b) == M.hd95(b, a)   # simétrica


def test_degenerate_cases_are_explicit():
    z, a = np.zeros((20, 20), bool), _square(20, 5, 5, 5)
    both = M.segmentation_metrics(z, z)
    assert both["degenerate"] == "both_empty" and np.isnan(both["dice"]) and np.isnan(both["hd95"])
    pe = M.segmentation_metrics(z, a)
    assert pe["degenerate"] == "pred_empty" and pe["dice"] == 0.0 and np.isnan(pe["hd95"])
    ge = M.segmentation_metrics(a, z)
    assert ge["degenerate"] == "gt_empty" and ge["iou"] == 0.0


def test_summarize_ignores_nan_and_counts_them():
    s = M.summarize([0.5, float("nan"), 1.0])
    assert s["n"] == 3 and s["n_valid"] == 2 and s["n_undefined"] == 1 and s["mean"] == pytest.approx(0.75)


# ── Bootstrap pareado ───────────────────────────────────────────────────────────

def test_paired_bootstrap_recovers_constant_shift():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 0.9, 200)
    r = M.paired_bootstrap_diff(a, a + 0.05, "mean", n_resamples=500, seed=1)
    assert r["estimate"] == pytest.approx(0.05)
    assert r["ci_low"] == pytest.approx(0.05) and r["ci_high"] == pytest.approx(0.05)
    assert r["frac_positive"] == 1.0


def test_paired_bootstrap_is_paired_not_independent():
    # Linhas muito heterogêneas, diferença constante: IC pareado é estreito; o independente seria largo.
    a = np.linspace(0, 1, 100)
    r = M.paired_bootstrap_diff(a, a + 0.1, "median", n_resamples=500, seed=2)
    assert r["ci_high"] - r["ci_low"] < 1e-9


def test_paired_bootstrap_excludes_undefined_pairs_and_is_seeded():
    a = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
    b = np.array([0.2, 0.1, 0.3, np.nan, 0.7, 0.5])
    r1 = M.paired_bootstrap_diff(a, b, "mean", 400, 0.9, seed=5)
    r2 = M.paired_bootstrap_diff(a, b, "mean", 400, 0.9, seed=5)
    assert r1 == r2 and r1["n_pairs"] == 4 and r1["n_excluded_undefined"] == 2
    assert r1["ci_low"] <= r1["estimate"] <= r1["ci_high"]


def test_bootstrap_with_zero_difference_includes_zero():
    rng = np.random.default_rng(3)
    a = rng.normal(0.8, 0.1, 100)
    b = a + rng.normal(0, 0.05, 100)
    r = M.paired_bootstrap_diff(a, b, "mean", 1000, seed=4)
    assert r["ci_low"] < 0 < r["ci_high"]


def test_generic_bootstrap_statistic_resamples_same_indices():
    x = np.arange(50, dtype=float)
    r = M.bootstrap_statistic([x, x + 1], lambda p, q: float(np.mean(q - p)), 200, seed=1)
    assert r["estimate"] == 1.0 and r["ci_low"] == 1.0 == r["ci_high"]
