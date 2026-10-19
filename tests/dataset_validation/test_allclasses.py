import collections

import pytest

from src.dataset_validation import allclasses as A
from src.dataset_validation import analysis
from src.dataset_validation import sampling as S
from src.dataset_validation.io_utils import read_json, write_json


@pytest.fixture
def ham_all(cfg, tmp_path):
    """HAM sintético com 7 classes, 2 imagens por lesão, 6 lesões por classe."""
    from conftest import _img, _mask
    rows, n = [], 0
    for dx in ("mel", "nv", "akiec", "bcc", "bkl", "df", "vasc"):
        for les in range(6):
            for j in range(2):
                iid = f"ISIC_{n:07d}"
                rows.append({"lesion_id": f"HAM_{dx}{les}", "image_id": iid, "dx": dx, "dx_type": "histo",
                             "age": "50", "sex": "male", "localization": "back", "dataset": "rosendahl"})
                _img(tmp_path / "ham" / f"{iid}.jpg")
                _mask(tmp_path / "hammask" / f"{iid}_segmentation.png")
                n += 1
    cfg["allclasses"]["n_per_class"] = 3
    cfg["sample"]["n_per_class"] = 3
    return rows


def test_all_classes_manifest(cfg, ham_all):
    rows = A.build_manifest(cfg, ham_all, add_hashes=False)
    assert collections.Counter(r["class_label"] for r in rows) == {c: 3 for c in A.CLASS_ORDER}
    assert len({r["lesion_id"] for r in rows}) == len(rows)            # sem lesão repetida
    assert all(r["binary_label"] == (r["class_label"] == "melanoma") for r in rows)
    assert A.manifest_fingerprint(rows) == A.manifest_fingerprint(A.build_manifest(cfg, ham_all, add_hashes=False))


def test_mel_nv_identical_to_pilot_ham_sample(cfg, ham_all, ph2_records):
    pilot = S._ham_rows(ham_all, cfg, None)
    rows = A.build_manifest(cfg, ham_all, add_hashes=False)
    assert {r["sample_id"] for r in pilot} <= {r["sample_id"] for r in rows}


def test_validation_detects_wrong_counts_and_labels(cfg, ham_all):
    rows = A.build_manifest(cfg, ham_all, add_hashes=False)
    with pytest.raises(S.SamplingError, match="esperado"):
        A.validate_manifest(rows[:-1], cfg)
    bad = [dict(r) for r in rows]
    bad[-1]["original_label"] = "dx=mel"
    with pytest.raises(S.SamplingError, match="incompatível"):
        A.validate_manifest(bad, cfg)


def test_reuse_seeds_only_matching_images_and_checks_code_hash(cfg, ham_all, tmp_path):
    rows = A.build_manifest(cfg, ham_all, add_hashes=False)
    reuse = tmp_path / "reuse"
    agents, schemas = tmp_path / "agents.py", tmp_path / "schemas.py"
    agents.write_bytes(b"a\r\nb")
    schemas.write_bytes(b"s")
    events = {f"/old/{r['image_id']}": {"image_path": "x", "real_value": 1, "validation": {"diagnostico": "Lesão Benigna"},
                                         "_meta": {"sample_id": r["sample_id"]}}
              for r in rows if r["class_label"] in ("melanoma", "nevus")}
    events["/old/ISIC_9999999"] = {"_meta": {"sample_id": "ham10000:ISIC_9999999"}}
    write_json(reuse / "results_gpt-5.4-nano.json", events)
    write_json(reuse / "results_gpt-5.4-nano.meta.json", {
        "model_name": "gpt-5.4-nano", "temperature": 0,
        "agents_sha256": A._sha_lf(agents), "schemas_sha256": A._sha_lf(schemas)})
    raw = tmp_path / "out" / "raw" / "gpt-5.4-nano"
    got = A.seed_reused_results(reuse, "gpt-5.4-nano", raw, rows, agents, schemas)
    assert len(got) == 6 and len(list(raw.glob("*.json"))) == 6          # mel + nv; imagem fora do manifesto ignorada
    part = read_json(next(raw.glob("*.json")))
    assert "_meta" not in next(iter(part.values()))
    schemas.write_bytes(b"changed")
    with pytest.raises(S.SamplingError, match="Reuso recusado"):
        A.seed_reused_results(reuse, "gpt-5.4-nano", tmp_path / "o2", rows, agents, schemas)
    assert A.seed_reused_results(tmp_path / "vazio", "gpt-5.4-nano", tmp_path / "o3", rows, agents, schemas) == []


def test_evaluate_allclasses_per_class_and_weighted(cfg):
    cats = {"melanoma": ["SUSPEITA"] * 8 + ["BENIGNA"] * 2, "nevus": ["BENIGNA"] * 10,
            "bkl": ["SUSPEITA"] * 5 + ["BENIGNA"] * 5, "bcc": ["INDETERMINADA"] * 10,
            "akiec": ["BENIGNA"] * 10, "df": ["BENIGNA"] * 10, "vasc": ["BENIGNA"] * 10}
    rows = []
    for cls, cs in cats.items():
        for i, c in enumerate(cs):
            rows.append({"sample_id": f"{cls}{i}", "class_label": cls, "binary_label": int(cls == "melanoma"),
                         "category": c, "pred_indeterminate_as_positive": 0 if c == "BENIGNA" else 1,
                         "pred_indeterminate_as_negative": 1 if c == "SUSPEITA" else 0})
    pool = {"melanoma": 100, "nevus": 600, "akiec": 30, "bcc": 50, "bkl": 100, "df": 10, "vasc": 10}
    ev = A.evaluate_allclasses(rows, pool, {"confidence": 0.95, "bootstrap": {"n_resamples": 200, "seed": 1}})
    p = ev["indeterminate_as_positive"]
    assert p["per_class"]["melanoma"]["sensitivity"] == 0.8
    assert p["per_class"]["bkl"]["specificity"] == 0.5 and p["per_class"]["bcc"]["specificity"] == 0.0
    assert p["per_class"]["nevus"]["specificity"] == 1.0
    assert p["macro_specificity_non_melanoma"] == pytest.approx((1 + 0.5 + 0 + 1 + 1 + 1) / 6)
    w = p["natural_prevalence_weighted_specificity"]["estimate"]
    assert w == pytest.approx((600 * 1 + 100 * 0.5 + 50 * 0 + 30 + 10 + 10) / 800)
    assert p["melanoma_vs_all_others_pooled"]["n"] == 70
    assert p["melanoma_vs_nevus_only"]["n"] == 20
