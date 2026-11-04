import zipfile

from src.data_audit import adapters


def _make_ham10000_fixture(tmp_path):
    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    (images_dir / "ISIC_0000000.jpg").write_bytes(b"\x00")
    (masks_dir / "ISIC_0000000_segmentation.png").write_bytes(b"\x00")
    metadata_csv = tmp_path / "HAM10000_metadata"
    metadata_csv.write_text(
        "lesion_id,image_id,dx\nHAM_0000001,ISIC_0000000,nv\n", encoding="utf-8"
    )
    return images_dir, masks_dir, metadata_csv


def test_audit_ham10000_hash_files_true_always_hashes(tmp_path):
    images_dir, masks_dir, metadata_csv = _make_ham10000_fixture(tmp_path)

    result = adapters.audit_ham10000(
        images_dir=images_dir, masks_dir=masks_dir, metadata_csv=metadata_csv, hash_files=True
    )

    assert result["hash_status"] == "hashed"
    assert len(result["image_manifest"]) == 1
    assert len(result["mask_manifest"]) == 1


def test_audit_ham10000_skip_without_previous_is_empty(tmp_path):
    images_dir, masks_dir, metadata_csv = _make_ham10000_fixture(tmp_path)

    result = adapters.audit_ham10000(
        images_dir=images_dir,
        masks_dir=masks_dir,
        metadata_csv=metadata_csv,
        hash_files=False,
        previous_manifest=None,
    )

    assert result["hash_status"] == "skipped_no_previous"
    assert result["image_manifest"] == []
    assert result["mask_manifest"] == []


def test_audit_ham10000_skip_reuses_previous_hashes(tmp_path):
    images_dir, masks_dir, metadata_csv = _make_ham10000_fixture(tmp_path)
    previous_manifest = {
        "image_manifest": [{"path": "ISIC_0000000.jpg", "size": 1, "sha256": "aaa"}],
        "mask_manifest": [{"path": "ISIC_0000000_segmentation.png", "size": 1, "sha256": "bbb"}],
    }

    result = adapters.audit_ham10000(
        images_dir=images_dir,
        masks_dir=masks_dir,
        metadata_csv=metadata_csv,
        hash_files=False,
        previous_manifest=previous_manifest,
    )

    assert result["hash_status"] == "reused_from_previous"
    assert result["image_manifest"] == previous_manifest["image_manifest"]
    assert result["mask_manifest"] == previous_manifest["mask_manifest"]


def _make_ph2_fixture(tmp_path, rows):
    """rows: list of (lesion_id, histological_diagnosis, clinical_diagnosis)."""
    root = tmp_path / "PH2"
    root.mkdir()
    header = (
        "||   Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry | "
        "Pigment Network | Dots/Globules | Streaks | Regression Areas | "
        "Blue-Whitish Veil ||           Colors ||\n"
    )
    lines = [header]
    for lesion_id, hist, clinical in rows:
        lesion_dir = root / lesion_id
        (lesion_dir / f"{lesion_id}_Dermoscopic_Image").mkdir(parents=True)
        (lesion_dir / f"{lesion_id}_Dermoscopic_Image" / f"{lesion_id}.bmp").write_bytes(b"\x00")
        (lesion_dir / f"{lesion_id}_lesion").mkdir(parents=True)
        (lesion_dir / f"{lesion_id}_lesion" / f"{lesion_id}_lesion.bmp").write_bytes(b"\x00")
        lines.append(
            f"|| {lesion_id} || {hist} || {clinical} ||         0 |               T |"
            "             A |       A |                A |                 A ||"
            "                4 ||\n"
        )
    metadata_txt = tmp_path / "PH2_dataset.txt"
    metadata_txt.write_text("".join(lines), encoding="utf-8")
    return root, metadata_txt


def test_audit_ph2_melanoma_text_vs_non_empty_counts(tmp_path):
    root, metadata_txt = _make_ph2_fixture(
        tmp_path,
        rows=[
            ("IMD001", "Melanoma", "2"),
            ("IMD002", "Nodular Melanoma", "2"),
            ("IMD003", "Dysplastic Nevus", "0"),
            ("IMD004", "", "0"),
        ],
    )

    result = adapters.audit_ph2(root_dir=root, metadata_txt=metadata_txt)

    criteria = result["diagnosis_criteria_side_by_side"]
    assert criteria["histological_diagnosis_non_empty"]["count"] == 3
    assert sorted(criteria["histological_diagnosis_non_empty"]["ids"]) == [
        "IMD001",
        "IMD002",
        "IMD003",
    ]
    assert criteria["histological_diagnosis_melanoma_text"]["count"] == 2
    assert sorted(criteria["histological_diagnosis_melanoma_text"]["ids"]) == [
        "IMD001",
        "IMD002",
    ]
    assert criteria["clinical_diagnosis_equals_2"]["count"] == 2
    assert criteria["histological_diagnosis_values"] == {
        "Dysplastic Nevus": 1,
        "Melanoma": 1,
        "Nodular Melanoma": 1,
    }


def test_audit_ham10000_skip_ignores_previous_with_empty_manifests(tmp_path):
    images_dir, masks_dir, metadata_csv = _make_ham10000_fixture(tmp_path)
    previous_manifest = {"image_manifest": [], "mask_manifest": []}

    result = adapters.audit_ham10000(
        images_dir=images_dir,
        masks_dir=masks_dir,
        metadata_csv=metadata_csv,
        hash_files=False,
        previous_manifest=previous_manifest,
    )

    assert result["hash_status"] == "skipped_no_previous"
    assert result["image_manifest"] == []
    assert result["mask_manifest"] == []


def _make_images_zip(path, ids):
    with zipfile.ZipFile(path, "w") as zf:
        for image_id in ids:
            zf.writestr(f"ISIC2018_Task1-2_Training_Input/{image_id}.jpg", b"\x00")


def _make_masks_zip(path, ids):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ISIC2018_Task1_Training_GroundTruth/ATTRIBUTION.txt", "x")
        zf.writestr("ISIC2018_Task1_Training_GroundTruth/LICENSE.txt", "x")
        for image_id in ids:
            zf.writestr(
                f"ISIC2018_Task1_Training_GroundTruth/{image_id}_segmentation.png", b"\x00"
            )


def test_audit_isic2018_task1_ausente_when_images_zip_missing(tmp_path):
    masks_zip = tmp_path / "masks.zip"
    _make_masks_zip(masks_zip, ["ISIC_0000000"])

    result = adapters.audit_isic2018_task1(
        images_zip=tmp_path / "does_not_exist.zip", masks_zip=masks_zip
    )
    assert result["status"] == "ausente"


def test_audit_isic2018_task1_ausente_when_masks_zip_missing(tmp_path):
    images_zip = tmp_path / "images.zip"
    _make_images_zip(images_zip, ["ISIC_0000000"])

    result = adapters.audit_isic2018_task1(
        images_zip=images_zip, masks_zip=tmp_path / "does_not_exist.zip"
    )
    assert result["status"] == "ausente"


def test_audit_isic2018_task1_completo_when_ids_match(tmp_path):
    ids = ["ISIC_0000000", "ISIC_0000001"]
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ids)
    _make_masks_zip(masks_zip, ids)

    result = adapters.audit_isic2018_task1(images_zip=images_zip, masks_zip=masks_zip)

    assert result["status"] == "completo"
    assert result["images_zip_image_entries"] == 2
    assert result["masks_zip_mask_entries"] == 2
    pairing = result["pairing_image_vs_mask"]
    assert pairing["matched"] == ids
    assert pairing["only_in_a"] == []
    assert pairing["only_in_b"] == []
    assert len(result["masks_zip_other_entries"]) == 2  # ATTRIBUTION.txt + LICENSE.txt


def test_audit_isic2018_task1_parcial_when_ids_diverge(tmp_path):
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ["ISIC_0000000", "ISIC_0000001"])
    _make_masks_zip(masks_zip, ["ISIC_0000000", "ISIC_0000002"])

    result = adapters.audit_isic2018_task1(images_zip=images_zip, masks_zip=masks_zip)

    assert result["status"] == "parcial"
    pairing = result["pairing_image_vs_mask"]
    assert pairing["matched"] == ["ISIC_0000000"]
    assert pairing["only_in_a"] == ["ISIC_0000001"]
    assert pairing["only_in_b"] == ["ISIC_0000002"]
