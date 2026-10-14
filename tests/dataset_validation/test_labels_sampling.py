import pytest

from src.dataset_validation import config as C
from src.dataset_validation import labels as L
from src.dataset_validation import sampling as S


def test_ph2_parse_real_format(tmp_path):
    p = tmp_path / "ph2.txt"
    p.write_text(
        "||   Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry | Pigment Network | "
        "Dots/Globules | Streaks | Regression Areas | Blue-Whitish Veil ||           Colors ||\n"
        "|| IMD003 ||                        ||                  0 ||         0 |               T |"
        "             A |       A |                A |                 A ||                4 ||\n"
        "|| IMD435 ||        Lentigo Maligna ||                  2 ||         2 |              AT |"
        "             A |       A |                A |                 P ||         3  4  5  6 ||\n",
        encoding="utf-8")
    r = L.parse_ph2_metadata(str(p))
    assert r["IMD003"]["clinical_diagnosis"] == 0 and r["IMD003"]["pigment_network"] == "T"
    assert r["IMD435"]["blue_whitish_veil"] == "P" and r["IMD435"]["colors"] == "3  4  5  6"


def test_legacy_rule_misses_lentigo_maligna_and_empty_histology(cfg, ph2_records):
    audit = L.ph2_label_audit(ph2_records, cfg["datasets"]["ph2"])
    assert audit["clinical_melanoma"] == 5
    assert audit["legacy_positive"] == 2
    assert audit["n_melanomas_labelled_negative_by_legacy_rule"] == 3
    assert audit["legacy_positive_not_clinical_melanoma"] == []


def test_binary_mapping(cfg, ph2_records):
    ph2 = cfg["datasets"]["ph2"]
    assert L.ph2_class(ph2_records["IMD000"], ph2) == L.MELANOMA
    assert L.ph2_class(ph2_records["IMD006"], ph2) == L.NEVUS
    assert L.ph2_class({"clinical_diagnosis": 7}, ph2) is None
    assert L.BINARY == {"melanoma": 1, "nevus": 0}


def test_draw_is_deterministic_and_seed_sensitive():
    pool = [f"x{i}" for i in range(50)]
    a = S.draw(pool, 5, 42, "s")
    assert a == S.draw(list(reversed(pool)), 5, 42, "s")          # independe da ordem de entrada
    assert a != S.draw(pool, 5, 43, "s") and a != S.draw(pool, 5, 42, "other")
    assert len(set(a)) == 5
    with pytest.raises(S.SamplingError):
        S.draw(pool, 51, 42, "s")


def test_manifest_balanced_no_duplicates_and_labels(cfg, ph2_records, ham_meta):
    rows = S.build_manifest(cfg, ph2_records, ham_meta, add_hashes=False)
    assert len(rows) == 12
    for ds in ("ph2", "ham10000"):
        for cls in ("melanoma", "nevus"):
            assert sum(r["dataset"] == ds and r["class_label"] == cls for r in rows) == 3
    assert len({r["sample_id"] for r in rows}) == len(rows)
    ham = [r for r in rows if r["dataset"] == "ham10000"]
    assert len({r["lesion_id"] for r in ham}) == len(ham)          # 1 imagem por lesão
    assert all(r["original_label"] == "dx=mel" for r in ham if r["class_label"] == "melanoma")
    assert all(r["original_label"] == "dx=nv" for r in ham if r["class_label"] == "nevus")
    assert not any(r["original_label"] == "dx=bkl" for r in ham)   # outras classes nunca entram
    ph2 = [r for r in rows if r["dataset"] == "ph2"]
    assert all(r["original_label"] == "clinical=2" for r in ph2 if r["class_label"] == "melanoma")
    assert all(r["seed"] == 42 for r in rows)
    again = S.build_manifest(cfg, ph2_records, ham_meta, add_hashes=False)
    assert S.manifest_fingerprint(rows) == S.manifest_fingerprint(again)


def test_manifest_validation_detects_problems(cfg, ph2_records, ham_meta):
    rows = S.build_manifest(cfg, ph2_records, ham_meta, add_hashes=False)
    dup = [dict(r) for r in rows] + [dict(rows[0])]
    with pytest.raises(S.SamplingError, match="duplicado"):
        S.validate_manifest(dup, cfg)
    bad = [dict(r) for r in rows]
    bad[0]["binary_label"] = 1 - bad[0]["binary_label"]
    with pytest.raises(S.SamplingError, match="binary_label"):
        S.validate_manifest(bad, cfg)
    bad = [dict(r) for r in rows]
    nevus = next(r for r in bad if r["dataset"] == "ph2" and r["class_label"] == "nevus")
    nevus["original_label"] = "clinical=2"
    with pytest.raises(S.SamplingError, match="fora do mapeamento"):
        S.validate_manifest(bad, cfg)


def test_lesion_in_both_classes_is_rejected(cfg, ph2_records, ham_meta):
    mel = next(r for r in ham_meta if r["dx"] == "mel")
    nv = next(r for r in ham_meta if r["dx"] == "nv")
    nv["lesion_id"] = mel["lesion_id"]
    with pytest.raises(S.SamplingError, match="ambas as classes"):
        S.build_manifest(cfg, ph2_records, ham_meta, add_hashes=False)


def test_config_rejects_invalid_setups():
    with pytest.raises(C.ConfigError):
        C.build_config({}, {"datasets": {"ph2": {"melanoma_clinical_codes": [1], "nevus_clinical_codes": [0, 1]}}})
    with pytest.raises(C.ConfigError):
        C.build_config({}, {"inference": {"segmentation_mode": "model"}})
    with pytest.raises(C.ConfigError):
        C.build_config({}, {"sample": {"n_per_class": 0}})
