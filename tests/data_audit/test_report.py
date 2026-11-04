from src.data_audit import report


def _ph2_summary(non_empty_count, melanoma_text_count, clinical_count, value_counts):
    return {
        "dataset": "ph2",
        "status": "completo",
        "lesion_dirs_found": non_empty_count,
        "metadata_rows": non_empty_count,
        "pairing_image_vs_metadata": {"matched": [], "only_in_a": [], "only_in_b": []},
        "pairing_mask_vs_metadata": {"matched": [], "only_in_a": [], "only_in_b": []},
        "pairing_roi_vs_metadata": {"matched": [], "only_in_a": [], "only_in_b": []},
        "pairing_image_vs_mask": {"matched": [], "only_in_a": [], "only_in_b": []},
        "diagnosis_criteria_side_by_side": {
            "histological_diagnosis_non_empty": {"count": non_empty_count, "ids": []},
            "histological_diagnosis_melanoma_text": {"count": melanoma_text_count, "ids": []},
            "histological_diagnosis_values": value_counts,
            "clinical_diagnosis_equals_2": {"count": clinical_count, "ids": []},
        },
    }


def test_render_ph2_paragraph_uses_computed_numbers_not_hardcoded():
    summary = _ph2_summary(
        non_empty_count=3,
        melanoma_text_count=2,
        clinical_count=2,
        value_counts={"Dysplastic Nevus": 1, "Melanoma": 1, "Nodular Melanoma": 1},
    )

    text = report._render_ph2(summary)

    assert "**2**" in text  # melanoma-text count computed, not the real dataset's "28"
    assert "**1**" in text  # non-melanoma remainder (3 - 2)
    assert "28" not in text and "41" not in text and "40" not in text
    assert "Dysplastic Nevus" in text
    assert "melanoma-por-texto" in text


def test_render_ph2_ausente_status_short_circuits():
    text = report._render_ph2({"status": "ausente"})
    assert "ausente" in text.lower()


def _ham10000_summary(hash_status, image_manifest, mask_manifest):
    return {
        "dataset": "ham10000",
        "status": "completo",
        "metadata_rows": 1,
        "image_files_found": 1,
        "mask_files_found": 1,
        "distinct_lesions": 1,
        "pairing_metadata_vs_images": {"matched": [], "only_in_a": [], "only_in_b": []},
        "pairing_metadata_vs_masks": {"matched": [], "only_in_a": [], "only_in_b": []},
        "pairing_images_vs_masks": {"matched": [], "only_in_a": [], "only_in_b": []},
        "lesion_id_to_dx_missing_or_conflicting": {"missing_id": [], "conflicting": {}},
        "dx_counts": {"nv": 1},
        "images_per_lesion_distribution": {"1": 1},
        "image_manifest": image_manifest,
        "mask_manifest": mask_manifest,
        "hash_status": hash_status,
    }


def test_render_ham10000_hashed_note():
    text = report._render_ham10000(_ham10000_summary("hashed", [{"sha256": "a"}], [{"sha256": "b"}]))
    assert "executado nesta rodada" in text


def test_render_ham10000_reused_from_previous_note():
    text = report._render_ham10000(
        _ham10000_summary("reused_from_previous", [{"sha256": "a"}], [{"sha256": "b"}])
    )
    assert "reaproveitados" in text
    assert "preservados" in text


def test_render_ham10000_skipped_no_previous_note():
    text = report._render_ham10000(_ham10000_summary("skipped_no_previous", [], []))
    assert "nenhum manifesto anterior" in text
