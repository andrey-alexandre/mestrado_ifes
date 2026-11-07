import pytest

from src.cnn_baseline import bcn20000_split as split_mod


def _train_rows():
    """20 lesions: 6 MEL (sizes 1,1,2,1,3,1 images = 9 images), 14 non-MEL
    (NV/BCC/AK mixed, 1-2 images each = 18 images). Enough variety to
    exercise group-size variation in the stratified split."""
    rows = []
    mel_sizes = [1, 1, 2, 1, 3, 1]
    for i, n_images in enumerate(mel_sizes, start=1):
        lid = f"BCN_MEL_{i:03d}"
        for j in range(n_images):
            rows.append({"bcn_filename": f"{lid}_{j}.jpg", "lesion_id": lid, "diagnosis": "MEL"})
    other_dx = ["NV"] * 8 + ["BCC"] * 4 + ["AK"] * 2
    for i, dx in enumerate(other_dx, start=1):
        lid = f"BCN_OTHER_{i:03d}"
        n_images = 1 if i % 2 == 0 else 2
        for j in range(n_images):
            rows.append({"bcn_filename": f"{lid}_{j}.jpg", "lesion_id": lid, "diagnosis": dx})
    return rows


def _test_rows_unlabeled():
    return [
        {"bcn_filename": "BCN_TEST_001_0.jpg", "lesion_id": "BCN_TEST_001", "diagnosis": ""},
        {"bcn_filename": "BCN_TEST_002_0.jpg", "lesion_id": "BCN_TEST_002", "diagnosis": ""},
    ]


def _test_rows_partially_labeled():
    return _test_rows_unlabeled() + [
        {"bcn_filename": "BCN_TEST_003_0.jpg", "lesion_id": "BCN_TEST_003", "diagnosis": "MEL"},
        {"bcn_filename": "BCN_TEST_004_0.jpg", "lesion_id": "BCN_TEST_004", "diagnosis": "NV"},
    ]


def test_binary_target_for_diagnosis():
    assert split_mod.binary_target_for_diagnosis("MEL") == "melanoma"
    for dx in ("NV", "BCC", "AK", "BKL", "DF", "SCC", "VASC"):
        assert split_mod.binary_target_for_diagnosis(dx) == "nao_melanoma"


def test_select_partition_rows_apenas_train_excludes_all_test():
    selected = split_mod.select_partition_rows(_train_rows(), _test_rows_partially_labeled(), "apenas_train")
    assert all(r["source_partition"] == "train" for r in selected)
    assert len(selected) == len(_train_rows())


def test_select_partition_rows_train_mais_test_rotulado_includes_only_labeled_test_rows():
    selected = split_mod.select_partition_rows(
        _train_rows(), _test_rows_partially_labeled(), "train_mais_test_rotulado"
    )
    test_rows_selected = [r for r in selected if r["source_partition"] == "test"]
    assert len(test_rows_selected) == 2  # only BCN_TEST_003 and BCN_TEST_004 have labels
    assert {r["lesion_id"] for r in test_rows_selected} == {"BCN_TEST_003", "BCN_TEST_004"}


def test_select_partition_rows_unknown_policy_raises():
    with pytest.raises(ValueError):
        split_mod.select_partition_rows(_train_rows(), [], "nao_existe")


def test_build_eligible_lesion_table_excludes_missing_and_conflicting():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    rows = list(rows) + [
        {"bcn_filename": "x.jpg", "lesion_id": "", "diagnosis": "NV", "source_partition": "train"},
        {"bcn_filename": "y.jpg", "lesion_id": "BCN_CONFLICT", "diagnosis": "NV", "source_partition": "train"},
        {"bcn_filename": "z.jpg", "lesion_id": "BCN_CONFLICT", "diagnosis": "MEL", "source_partition": "train"},
    ]
    result = split_mod.build_eligible_lesion_table(rows)
    assert "BCN_CONFLICT" not in result["lesion_table"]
    reasons = {reason for rec in result["excluded_rows"] for reason in rec["reasons"]}
    assert "sem lesion_id" in reasons
    assert "lesion_id com diagnosis conflitante entre suas linhas" in reasons
    assert len(result["lesion_table"]) == 20  # the 20 clean lesions from _train_rows


def test_split_lesions_stratified_sums_to_total_and_is_deterministic():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]

    result_a = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=7)
    result_b = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=7)

    assert result_a == result_b  # determinism
    assert len(result_a["train_lesion_ids"]) + len(result_a["val_lesion_ids"]) == 20
    assert set(result_a["train_lesion_ids"]).isdisjoint(result_a["val_lesion_ids"])


def test_split_lesions_stratified_preserves_class_proportion_per_class():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    # 6 melanoma lesions * 0.2 = 1.2 -> round = 1 val, 5 train
    assert result["per_class_detail"]["melanoma"]["n_val"] == 1
    assert result["per_class_detail"]["melanoma"]["n_train"] == 5
    # 14 non-melanoma lesions * 0.2 = 2.8 -> round = 3 val, 11 train
    assert result["per_class_detail"]["nao_melanoma"]["n_val"] == 3
    assert result["per_class_detail"]["nao_melanoma"]["n_train"] == 11


def test_split_lesions_stratified_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_mod.split_lesions_stratified({}, val_fraction=1.5, seed=1)
    with pytest.raises(ValueError):
        split_mod.split_lesions_stratified({}, val_fraction=0.0, seed=1)


def test_build_split_manifest_rows_keeps_all_images_of_a_lesion_on_one_side():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    split_result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    total_images = sum(len(entry["images"]) for entry in lesion_table.values())
    assert len(manifest_rows) == total_images
    for lid in split_result["train_lesion_ids"]:
        split_values = {r["split"] for r in manifest_rows if r["lesion_id"] == lid}
        assert split_values == {"train"}
    for lid in split_result["val_lesion_ids"]:
        split_values = {r["split"] for r in manifest_rows if r["lesion_id"] == lid}
        assert split_values == {"val"}


def test_verify_no_lesion_overlap_passes_on_clean_split():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    split_result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    result = split_mod.verify_no_lesion_overlap(manifest_rows)
    assert result["ok"] is True
    assert result["overlapping_lesion_ids"] == []


def test_verify_no_lesion_overlap_detects_tampered_overlap():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    split_result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    # Tamper: duplicate one train row into val too.
    tampered = list(manifest_rows)
    leaked = dict(manifest_rows[0])
    leaked["split"] = "val"
    tampered.append(leaked)
    result = split_mod.verify_no_lesion_overlap(tampered)
    assert result["ok"] is False
    assert manifest_rows[0]["lesion_id"] in result["overlapping_lesion_ids"]


def test_summarize_split_reports_both_classes_present_and_fractions():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    split_result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    summary = split_mod.summarize_split(manifest_rows)
    for split in ("train", "val"):
        assert summary[split]["n_melanoma_images"] > 0
        assert summary[split]["n_nao_melanoma_images"] > 0
        assert summary[split]["n_melanoma_lesions"] > 0
        assert summary[split]["n_nao_melanoma_lesions"] > 0
    assert 0.0 < summary["achieved_val_image_fraction"] < 1.0
    assert 0.0 < summary["achieved_val_lesion_fraction"] < 1.0


def test_build_split_report_markdown_smoke():
    rows = split_mod.select_partition_rows(_train_rows(), [], "apenas_train")
    lesion_table = split_mod.build_eligible_lesion_table(rows)["lesion_table"]
    split_result = split_mod.split_lesions_stratified(lesion_table, val_fraction=0.2, seed=1)
    manifest_rows = split_mod.build_split_manifest_rows(
        lesion_table, split_result["train_lesion_ids"], split_result["val_lesion_ids"]
    )
    summary = split_mod.summarize_split(manifest_rows)
    overlap_check = split_mod.verify_no_lesion_overlap(manifest_rows)
    report = split_mod.build_split_report_markdown(
        partition_policy="apenas_train",
        val_fraction=0.2,
        seed=1,
        excluded_rows_count=0,
        split_result=split_result,
        summary=summary,
        overlap_check=overlap_check,
    )
    assert "Divisão Treino/Validação do BCN20000" in report
    assert "preparação de dados, não treino" in report
    assert "HAM10000" in report
