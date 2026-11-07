from src.cnn_baseline import bcn20000_diagnostics as diag


def _fabricated_rows():
    """12 synthetic rows: 8 clean lesions (2 MEL, 2 NV, 2 BCC, 2 AK), one
    lesion with conflicting diagnosis across its 2 rows, one row missing
    lesion_id, one row with empty diagnosis, one row with an unexpected
    diagnosis value. One filename is deliberately not in the fabricated
    zip listing (simulates a missing image)."""
    rows = [
        {"bcn_filename": "BCN_0000000001.jpg", "lesion_id": "BCN_0000001", "diagnosis": "MEL"},
        {"bcn_filename": "BCN_0000000002.jpg", "lesion_id": "BCN_0000002", "diagnosis": "MEL"},
        {"bcn_filename": "BCN_0000000003.jpg", "lesion_id": "BCN_0000003", "diagnosis": "NV"},
        {"bcn_filename": "BCN_0000000004.jpg", "lesion_id": "BCN_0000003", "diagnosis": "NV"},  # same lesion, 2nd image
        {"bcn_filename": "BCN_0000000005.jpg", "lesion_id": "BCN_0000004", "diagnosis": "NV"},
        {"bcn_filename": "BCN_0000000006.jpg", "lesion_id": "BCN_0000005", "diagnosis": "BCC"},
        {"bcn_filename": "BCN_0000000007.jpg", "lesion_id": "BCN_0000006", "diagnosis": "BCC"},
        {"bcn_filename": "BCN_0000000008.jpg", "lesion_id": "BCN_0000007", "diagnosis": "AK"},
        {"bcn_filename": "BCN_0000000009.jpg", "lesion_id": "BCN_0000008", "diagnosis": "AK"},
        # conflicting diagnosis across rows of the same lesion:
        {"bcn_filename": "BCN_0000000010.jpg", "lesion_id": "BCN_0000009", "diagnosis": "NV"},
        {"bcn_filename": "BCN_0000000011.jpg", "lesion_id": "BCN_0000009", "diagnosis": "MEL"},
        # missing lesion_id:
        {"bcn_filename": "BCN_0000000012.jpg", "lesion_id": "", "diagnosis": "NV"},
        # empty diagnosis:
        {"bcn_filename": "BCN_0000000013.jpg", "lesion_id": "BCN_0000010", "diagnosis": ""},
        # unexpected diagnosis value:
        {"bcn_filename": "BCN_0000000014.jpg", "lesion_id": "BCN_0000011", "diagnosis": "WEIRD"},
        # missing from zip (filename never listed below):
        {"bcn_filename": "BCN_0000000099.jpg", "lesion_id": "BCN_0000012", "diagnosis": "VASC"},
    ]
    return rows


def _zip_filenames_matching(rows):
    """All filenames except the deliberately-missing one."""
    return {r["bcn_filename"] for r in rows if r["bcn_filename"] != "BCN_0000000099.jpg"}


def test_diagnose_partition_counts_usable_labels_and_distinct_lesions():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    assert result["n_csv_rows"] == 15
    # usable labels: all rows with diagnosis in BCN_DIAGNOSIS_CATEGORIES
    # (excludes the empty one and the "WEIRD" one) = 15 - 1(empty) - 1(weird) = 13
    assert result["n_usable_label_rows"] == 13
    assert result["n_empty_label_rows"] == 1
    assert result["n_unexpected_label_rows"] == 1
    assert result["unexpected_label_values"] == ["WEIRD"]


def test_diagnose_partition_flags_missing_lesion_id():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    assert result["n_missing_lesion_id_rows"] == 1


def test_diagnose_partition_flags_conflicting_diagnosis():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    assert "BCN_0000009" in result["conflicting_lesion_id_to_diagnosis"]
    assert set(result["conflicting_lesion_id_to_diagnosis"]["BCN_0000009"]) == {"MEL", "NV"}


def test_diagnose_partition_flags_missing_image_from_zip():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    assert result["n_missing_from_zip"] == 1
    assert "BCN_0000000099.jpg" in result["missing_from_zip_filenames"]


def test_diagnose_partition_distinct_lesion_count_counts_multi_image_lesion_once():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    # BCN_0000001..BCN_0000012 = 12 distinct non-empty lesion_id values
    # (BCN_0000003 has 2 rows but counts once)
    assert result["n_distinct_lesion_ids"] == 12


def test_diagnose_partition_category_counts_by_row_vs_by_lesion_differ_for_multi_image_lesion():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    # NV: by row = BCN_0000003(x2) + BCN_0000004 + BCN_0000009(conflicting,
    # excluded from lesion-level) + missing-lesion_id row = count rows with
    # diagnosis NV among usable labels: rows 3,4,5,10 -> wait row 11 is MEL
    # (same lesion as row 10) and the missing-lesion_id row (12) also NV.
    # By-row NV count = rows: BCN_0000003(x2), BCN_0000004, BCN_0000009(NV
    # row only), missing-lesion_id row = 5.
    assert result["category_counts_by_row"]["NV"] == 5
    # By-lesion NV: BCN_0000003, BCN_0000004 (BCN_0000009 excluded for
    # conflict, missing-lesion_id row has no lesion_id to count) = 2
    assert result["category_counts_by_lesion"]["NV"] == 2


def test_diagnose_partition_ineligible_records_cover_every_reason():
    rows = _fabricated_rows()
    result = diag.diagnose_partition(rows, _zip_filenames_matching(rows), "train")
    all_reasons = {reason for rec in result["ineligible_records"] for reason in rec["reasons"]}
    assert "sem lesion_id" in all_reasons
    assert "diagnosis vazio" in all_reasons
    assert any(r.startswith("diagnosis inesperado") for r in all_reasons)
    assert "lesion_id com diagnosis conflitante entre suas linhas" in all_reasons
    assert "imagem ausente do zip local" in all_reasons


def test_compare_lesion_ids_heuristic_finds_normalized_collision():
    train_ids = {"BCN_0003884", "BCN_0000010"}
    test_ids = {"BCN3884", "BCN9999"}  # BCN3884 normalizes to same int as BCN_0003884
    result = diag.compare_lesion_ids_heuristic(train_ids, test_ids)
    assert result["n_normalized_collisions"] == 1
    assert result["is_confirmed_overlap"] is False
    assert "warning" in result


def test_compare_lesion_ids_heuristic_no_collision_when_disjoint():
    train_ids = {"BCN_0000001", "BCN_0000002"}
    test_ids = {"BCN9999", "BCN8888"}
    result = diag.compare_lesion_ids_heuristic(train_ids, test_ids)
    assert result["n_normalized_collisions"] == 0


def test_build_diagnostic_report_markdown_smoke():
    rows = _fabricated_rows()
    filenames = _zip_filenames_matching(rows)
    train_diag = diag.diagnose_partition(rows, filenames, "train")
    test_diag = diag.diagnose_partition(rows, filenames, "test")
    overlap = diag.compare_lesion_ids_heuristic({"BCN_0000001"}, {"BCN1"})
    report = diag.build_diagnostic_report_markdown(
        train_diag=train_diag, test_diag=test_diag, overlap_heuristic=overlap
    )
    assert "Diagnóstico das Partições Oficiais do BCN20000" in report
    assert "só diagnóstico" in report
    assert "HAM10000" in report
