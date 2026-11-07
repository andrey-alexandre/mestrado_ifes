from pathlib import Path

from src.data_audit import sampling


def _fabricated_metadata_rows():
    """Same shape as tests/sampling/test_ham10000_sample.py's fixture:
    50 fictitious lesions across 4 categories (nv=28, mel=12, bcc=6, bkl=4),
    each with 1-2 images."""
    rows = []
    counts = {"nv": 28, "mel": 12, "bcc": 6, "bkl": 4}
    lesion_n = 0
    for dx, n in counts.items():
        for i in range(n):
            lesion_n += 1
            lesion_id = f"HAM_{lesion_n:04d}"
            rows.append({"lesion_id": lesion_id, "image_id": f"ISIC_{lesion_n:04d}_a", "dx": dx})
            if i % 3 == 0:
                rows.append({"lesion_id": lesion_id, "image_id": f"ISIC_{lesion_n:04d}_b", "dx": dx})
    return rows


def _manifest_rows_matching_expected_quotas(target_n=10):
    """Build a manifest (lesion_id/image_id/dx/binary_target rows) whose
    per-category counts exactly match compute_quotas_largest_remainder's
    output for the fabricated population above, picking real lesion_ids/
    image_ids out of that population (so mask-correspondence tests can
    point at files matching these image_ids)."""
    rows = _fabricated_metadata_rows()
    lesion_table = sampling.build_lesion_table(rows)
    counts_by_category = {
        cat: sum(1 for v in lesion_table.values() if v["dx"] == cat)
        for cat in sampling.ELIGIBLE_DX_CATEGORIES
    }
    quotas = sampling.compute_quotas_largest_remainder(counts_by_category, target_n)
    sample = sampling.sample_lesions(lesion_table, quotas["final_quotas"], seed=1)
    return sample["sampled_rows"]


def _write_masks(masks_dir: Path, manifest_rows, skip_image_ids=()):
    masks_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest_rows:
        if row["image_id"] in skip_image_ids:
            continue
        (masks_dir / f"{row['image_id']}_segmentation.png").write_bytes(b"\x00")


# --- criterion 1/2: lesion/image uniqueness ---------------------------------


def test_verify_lesion_and_image_uniqueness_passes_on_clean_manifest():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    result = sampling.verify_lesion_and_image_uniqueness(manifest_rows, expected_n=10)
    assert result["ok"] is True
    assert result["n_distinct_lesion_ids"] == 10
    assert result["lesion_count_matches_expected"] is True
    assert result["one_image_per_lesion"] is True
    assert result["duplicate_lesion_ids"] == []
    assert result["duplicate_image_ids"] == []


def test_verify_lesion_and_image_uniqueness_fails_on_wrong_count():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    result = sampling.verify_lesion_and_image_uniqueness(manifest_rows, expected_n=3000)
    assert result["ok"] is False
    assert result["lesion_count_matches_expected"] is False


def test_verify_lesion_and_image_uniqueness_fails_on_duplicate_lesion_row():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    manifest_rows = list(manifest_rows) + [dict(manifest_rows[0])]  # duplicate one row
    result = sampling.verify_lesion_and_image_uniqueness(manifest_rows, expected_n=10)
    assert result["ok"] is False
    assert result["one_image_per_lesion"] is False
    assert manifest_rows[0]["lesion_id"] in result["duplicate_lesion_ids"]


def test_verify_lesion_and_image_uniqueness_fails_on_duplicate_image_id_different_lesion():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    manifest_rows = list(manifest_rows)
    clashing_row = dict(manifest_rows[1])
    clashing_row["image_id"] = manifest_rows[0]["image_id"]  # reuse another lesion's image_id
    manifest_rows[1] = clashing_row
    result = sampling.verify_lesion_and_image_uniqueness(manifest_rows, expected_n=10)
    assert result["ok"] is False
    assert result["no_duplicate_image_ids"] is False
    assert manifest_rows[0]["image_id"] in result["duplicate_image_ids"]


# --- criterion 3: quota reproducibility -------------------------------------


def test_verify_quota_reproducibility_passes_when_manifest_matches_recomputed_quotas():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    metadata_rows = _fabricated_metadata_rows()
    result = sampling.verify_quota_reproducibility(manifest_rows, metadata_rows, target_n=10)
    assert result["ok"] is True
    assert result["mismatches"] == {}
    assert sum(result["expected_quotas"].values()) == 10


def test_verify_quota_reproducibility_fails_when_manifest_count_diverges():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    metadata_rows = _fabricated_metadata_rows()
    # Tamper: change one row's dx so the manifest's actual per-category
    # count no longer matches what compute_quotas_largest_remainder expects.
    tampered = [dict(r) for r in manifest_rows]
    for r in tampered:
        if r["dx"] == "nv":
            r["dx"] = "bcc"
            break
    result = sampling.verify_quota_reproducibility(tampered, metadata_rows, target_n=10)
    assert result["ok"] is False
    assert "nv" in result["mismatches"] or "bcc" in result["mismatches"]


def test_verify_quota_reproducibility_fails_when_population_has_conflicting_dx():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    metadata_rows = _fabricated_metadata_rows()
    metadata_rows = metadata_rows + [
        {"lesion_id": "HAM_0001", "image_id": "ISIC_conflict", "dx": "mel"}  # HAM_0001 is nv
    ]
    result = sampling.verify_quota_reproducibility(manifest_rows, metadata_rows, target_n=10)
    assert result["ok"] is False
    assert result["population_check_ok"] is False


def test_verify_quota_reproducibility_respects_exclusions():
    metadata_rows = _fabricated_metadata_rows()
    lesion_table = sampling.build_lesion_table(metadata_rows)
    bkl_ids = [lid for lid, v in lesion_table.items() if v["dx"] == "bkl"]
    filtered_table = sampling.apply_exclusions(lesion_table, bkl_ids)["filtered_lesion_table"]
    counts_by_category = {
        cat: sum(1 for v in filtered_table.values() if v["dx"] == cat)
        for cat in sampling.ELIGIBLE_DX_CATEGORIES
    }
    quotas = sampling.compute_quotas_largest_remainder(counts_by_category, target_n=10)
    manifest_rows = sampling.sample_lesions(filtered_table, quotas["final_quotas"], seed=1)[
        "sampled_rows"
    ]

    result = sampling.verify_quota_reproducibility(
        manifest_rows, metadata_rows, target_n=10, excluded_lesion_ids=bkl_ids
    )
    assert result["ok"] is True
    assert result["expected_quotas"].get("bkl", 0) == 0


# --- criterion 4: mask correspondence ---------------------------------------


def test_verify_mask_correspondence_passes_when_all_masks_present(tmp_path):
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    masks_dir = tmp_path / "masks"
    _write_masks(masks_dir, manifest_rows)

    result = sampling.verify_mask_correspondence(manifest_rows, masks_dir)
    assert result["ok"] is True
    assert result["n_with_mask"] == 10
    assert result["n_missing_mask"] == 0
    assert result["missing_image_ids"] == []


def test_verify_mask_correspondence_fails_when_a_mask_is_missing(tmp_path):
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    masks_dir = tmp_path / "masks"
    missing_id = manifest_rows[0]["image_id"]
    _write_masks(masks_dir, manifest_rows, skip_image_ids={missing_id})

    result = sampling.verify_mask_correspondence(manifest_rows, masks_dir)
    assert result["ok"] is False
    assert result["n_missing_mask"] == 1
    assert result["missing_image_ids"] == [missing_id]


def test_verify_mask_correspondence_fails_when_masks_dir_does_not_exist(tmp_path):
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    result = sampling.verify_mask_correspondence(manifest_rows, tmp_path / "does_not_exist")
    assert result["ok"] is False
    assert result["n_missing_mask"] == len(manifest_rows)


# --- read_sample_manifest + report builder ----------------------------------


def test_read_sample_manifest_round_trips_a_written_csv(tmp_path):
    import csv

    manifest_path = tmp_path / "sample_manifest.csv"
    rows = [
        {"lesion_id": "HAM_0001", "image_id": "ISIC_0001", "dx": "nv", "binary_target": "nao_melanoma"},
        {"lesion_id": "HAM_0002", "image_id": "ISIC_0002", "dx": "mel", "binary_target": "melanoma"},
    ]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lesion_id", "image_id", "dx", "binary_target"])
        writer.writeheader()
        writer.writerows(rows)

    read_back = sampling.read_sample_manifest(manifest_path)
    assert read_back == rows


def test_build_verification_report_markdown_reflects_pass_and_fail_states():
    manifest_rows = _manifest_rows_matching_expected_quotas(target_n=10)
    uniqueness = sampling.verify_lesion_and_image_uniqueness(manifest_rows, expected_n=10)
    metadata_rows = _fabricated_metadata_rows()
    quotas = sampling.verify_quota_reproducibility(manifest_rows, metadata_rows, target_n=10)
    masks = {
        "n_checked": 10,
        "n_with_mask": 8,
        "n_missing_mask": 2,
        "missing_image_ids": ["ISIC_0001_a", "ISIC_0002_a"],
        "ok": False,
    }
    overall_ok = uniqueness["ok"] and quotas["ok"] and masks["ok"]

    report = sampling.build_verification_report_markdown(
        manifest_path="dummy.csv",
        uniqueness=uniqueness,
        quotas=quotas,
        masks=masks,
        overall_ok=overall_ok,
    )
    assert overall_ok is False
    assert "AO MENOS UM CRITÉRIO NÃO PASSOU" in report
    assert "ISIC_0001_a" in report
