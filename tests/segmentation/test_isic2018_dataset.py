import io
import zipfile

import numpy as np
import torch
from PIL import Image

from src.segmentation.isic2018_dataset import (
    ISIC2018Task1Dataset,
    TARGET_SIZE,
    image_id_from_entry_name,
    list_available_image_ids,
    list_available_mask_ids,
    verify_image_mask_correspondence,
)


def test_image_id_from_entry_name_strips_prefix_and_extension():
    assert (
        image_id_from_entry_name("ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg")
        == "ISIC_0000000"
    )


def _make_images_zip(path, image_ids, size=(64, 48)):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ISIC2018_Task1-2_Training_Input/ATTRIBUTION.txt", "x")
        zf.writestr("ISIC2018_Task1-2_Training_Input/LICENSE.txt", "x")
        for image_id in image_ids:
            img = Image.new("RGB", size, color=(100, 150, 200))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            zf.writestr(f"ISIC2018_Task1-2_Training_Input/{image_id}.jpg", buf.getvalue())


def _make_masks_zip(path, image_ids, size=(64, 48)):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("ISIC2018_Task1_Training_GroundTruth/ATTRIBUTION.txt", "x")
        for image_id in image_ids:
            arr = np.zeros(size[::-1], dtype=np.uint8)  # (H, W)
            arr[10:20, 10:20] = 255  # a small "lesion" square
            mask = Image.fromarray(arr, mode="L")
            buf = io.BytesIO()
            mask.save(buf, format="PNG")
            zf.writestr(
                f"ISIC2018_Task1_Training_GroundTruth/{image_id}_segmentation.png", buf.getvalue()
            )


def test_list_available_image_ids_skips_non_jpg_entries(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    images_zip = tmp_path / "images.zip"
    _make_images_zip(images_zip, ids)
    found = list_available_image_ids(images_zip)
    assert found == sorted(ids)


def test_dataset_returns_correctly_shaped_image_and_mask(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002"]
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ids)
    _make_masks_zip(masks_zip, ids)

    dataset = ISIC2018Task1Dataset(ids, images_zip, masks_zip)
    image_tensor, mask_tensor, image_id = dataset[0]

    assert image_id == "ISIC_0000001"
    assert image_tensor.shape == (3, TARGET_SIZE, TARGET_SIZE)
    assert mask_tensor.shape == (1, TARGET_SIZE, TARGET_SIZE)
    assert image_tensor.dtype == torch.float32
    assert mask_tensor.dtype == torch.float32
    # mask must be strictly binary after nearest-neighbor resize + threshold
    unique_vals = set(torch.unique(mask_tensor).tolist())
    assert unique_vals <= {0.0, 1.0}


def test_dataset_pairs_each_image_with_its_own_matching_mask_not_a_neighbor(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ids)
    _make_masks_zip(masks_zip, ids)

    dataset = ISIC2018Task1Dataset(ids, images_zip, masks_zip)
    for idx, expected_id in enumerate(ids):
        _image, _mask, image_id = dataset[idx]
        assert image_id == expected_id


def test_list_available_mask_ids_strips_segmentation_suffix(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002"]
    masks_zip = tmp_path / "masks.zip"
    _make_masks_zip(masks_zip, ids)
    assert list_available_mask_ids(masks_zip) == sorted(ids)


def test_verify_image_mask_correspondence_ok_when_matched(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"]
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ids)
    _make_masks_zip(masks_zip, ids)

    result = verify_image_mask_correspondence(images_zip, masks_zip)
    assert result["ok"] is True
    assert result["n_matched"] == 3
    assert result["n_only_in_images"] == 0
    assert result["n_only_in_masks"] == 0


def test_verify_image_mask_correspondence_detects_mismatch(tmp_path):
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ["ISIC_0000001", "ISIC_0000002"])
    _make_masks_zip(masks_zip, ["ISIC_0000001", "ISIC_0000099"])  # 0000099 only in masks

    result = verify_image_mask_correspondence(images_zip, masks_zip)
    assert result["ok"] is False
    assert result["n_matched"] == 1
    assert result["only_in_images"] == ["ISIC_0000002"]
    assert result["only_in_masks"] == ["ISIC_0000099"]


def test_dataset_len_matches_image_id_count(tmp_path):
    ids = ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003", "ISIC_0000004"]
    images_zip = tmp_path / "images.zip"
    masks_zip = tmp_path / "masks.zip"
    _make_images_zip(images_zip, ids)
    _make_masks_zip(masks_zip, ids)

    dataset = ISIC2018Task1Dataset(ids, images_zip, masks_zip)
    assert len(dataset) == 4
