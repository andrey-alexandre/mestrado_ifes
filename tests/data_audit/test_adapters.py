import zipfile

from src.data_audit import adapters


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
