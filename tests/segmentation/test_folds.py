import json

from src.segmentation import folds as folds_mod


def _image_ids(n: int) -> list[str]:
    return [f"ISIC_{i:07d}" for i in range(n)]


def test_build_fold_groups_falls_back_to_per_image_without_lesion_mapping():
    ids = _image_ids(10)
    result = folds_mod.build_fold_groups(ids, image_id_to_lesion_id=None)
    assert result["lesion_grouping_available"] is False
    assert result["n_images_grouped_by_lesion"] == 0
    assert result["n_groups"] == 10  # one group per image
    for image_id in ids:
        assert result["group_of_image"][image_id] == f"image:{image_id}"


def test_build_fold_groups_uses_lesion_mapping_when_available():
    ids = ["A1", "A2", "B1", "C1"]
    mapping = {"A1": "LESION_A", "A2": "LESION_A", "B1": "LESION_B"}
    # C1 intentionally missing from mapping -> per-image fallback for it only.
    result = folds_mod.build_fold_groups(ids, image_id_to_lesion_id=mapping)
    assert result["lesion_grouping_available"] is True
    assert result["n_images_grouped_by_lesion"] == 3
    assert result["group_of_image"]["A1"] == result["group_of_image"]["A2"] == "lesion:LESION_A"
    assert result["group_of_image"]["B1"] == "lesion:LESION_B"
    assert result["group_of_image"]["C1"] == "image:C1"
    # A1+A2 share a group -> 3 distinct groups total (LESION_A, LESION_B, C1)
    assert result["n_groups"] == 3


def test_assign_folds_covers_every_image_exactly_once():
    ids = _image_ids(23)
    grouping = folds_mod.build_fold_groups(ids)
    assignment = folds_mod.assign_folds(grouping["groups"], n_folds=5, seed=1)
    assert sum(assignment["per_fold_image_count"]) == 23
    assert len(assignment["image_to_fold"]) == 23
    assert set(assignment["image_to_fold"].values()) <= {0, 1, 2, 3, 4}


def test_assign_folds_keeps_a_lesion_group_entirely_in_one_fold():
    ids = ["A1", "A2", "A3", "B1"]
    mapping = {"A1": "LESION_A", "A2": "LESION_A", "A3": "LESION_A"}
    grouping = folds_mod.build_fold_groups(ids, image_id_to_lesion_id=mapping)
    assignment = folds_mod.assign_folds(grouping["groups"], n_folds=5, seed=7)
    fold_a = {assignment["image_to_fold"][iid] for iid in ("A1", "A2", "A3")}
    assert len(fold_a) == 1  # all 3 images of LESION_A in the same fold


def test_build_folds_is_deterministic_for_same_seed():
    ids = _image_ids(50)
    result_a = folds_mod.build_folds(ids, n_folds=5, seed=42)
    result_b = folds_mod.build_folds(ids, n_folds=5, seed=42)
    assert result_a["image_to_fold"] == result_b["image_to_fold"]


def test_build_folds_different_seed_can_change_assignment():
    ids = _image_ids(50)
    result_a = folds_mod.build_folds(ids, n_folds=5, seed=1)
    result_b = folds_mod.build_folds(ids, n_folds=5, seed=2)
    assert result_a["image_to_fold"] != result_b["image_to_fold"]


def test_write_and_load_folds_file_round_trips(tmp_path):
    ids = _image_ids(30)
    result = folds_mod.build_folds(ids, n_folds=5, seed=99)
    path = tmp_path / "folds.json"
    folds_mod.write_folds_file(result, path, dataset="isic2018_task1")

    loaded = folds_mod.load_folds_file(path)
    assert loaded["dataset"] == "isic2018_task1"
    assert loaded["seed"] == 99
    assert loaded["n_folds"] == 5
    assert loaded["lesion_grouping_available"] is False
    assert loaded["image_to_fold"] == result["image_to_fold"]

    # sanity: file is valid, readable JSON independent of load_folds_file too
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["n_images"] == 30
