"""Fixed, reproducible 5-fold split for ISIC2018 Task1 segmentation
development (stage 4). Grouped by lesion when metadata allows it;
otherwise one group per image.

LIMITATION CONFIRMED BY DIRECT INSPECTION (not assumed): ISIC2018 Task1
provides NO lesion-identifier metadata of any kind. Both archives
(``ISIC2018_Task1-2_Training_Input.zip``,
``ISIC2018_Task1_Training_GroundTruth.zip``) contain only
``ATTRIBUTION.txt``/``LICENSE.txt`` alongside the image/mask files --
confirmed by listing each zip's central directory. No CSV or other
metadata file exists next to either zip on disk. Unlike HAM10000/BCN20000
(which have a real ``lesion_id`` column), there is no way to group
ISIC2018 Task1 images by lesion. ``build_fold_groups`` therefore falls
back to one group per image for this dataset -- this is registered here
and in the generated report as an explicit limitation, not silently
glossed over. The grouping function itself DOES support a real
``image_id -> lesion_id`` mapping if one is ever supplied (e.g. from a
future metadata release), so no rework is needed if that changes.

The output fold assignment is written to a fixed file meant to be
reused by any future segmentation architecture (DeepLabV3+, EM-Net,
etc.) evaluated against the same baseline split -- never regenerate a
competing fold assignment for the same dataset once one is frozen.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

N_FOLDS = 5
DEFAULT_SEED = 20261108


def build_fold_groups(
    image_ids: Sequence[str], image_id_to_lesion_id: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Map each ``image_id`` to a group key for fold assignment.

    If ``image_id_to_lesion_id`` is given and covers an image, that
    image's group is its lesion id (so all images of the same lesion
    land in the same fold together). Any image missing from the mapping
    (or when the mapping is ``None`` entirely, the expected case for
    ISIC2018 Task1 today) gets its own group, keyed by its own
    ``image_id`` -- i.e. per-image fallback, never silently merged with
    an unrelated image.
    """
    mapping = image_id_to_lesion_id or {}
    group_of: dict[str, str] = {}
    n_grouped_by_lesion = 0
    for image_id in image_ids:
        lesion_id = mapping.get(image_id)
        if lesion_id:
            group_of[image_id] = f"lesion:{lesion_id}"
            n_grouped_by_lesion += 1
        else:
            group_of[image_id] = f"image:{image_id}"

    groups: dict[str, list[str]] = {}
    for image_id, group_key in group_of.items():
        groups.setdefault(group_key, []).append(image_id)

    return {
        "group_of_image": group_of,
        "groups": groups,
        "n_images": len(image_ids),
        "n_groups": len(groups),
        "n_images_grouped_by_lesion": n_grouped_by_lesion,
        "lesion_grouping_available": bool(mapping),
    }


def assign_folds(
    groups: Mapping[str, list[str]], n_folds: int, seed: int
) -> dict[str, Any]:
    """Shuffle group keys deterministically (fixed seed) and deal them
    round-robin into ``n_folds`` folds, as evenly sized (by group count)
    as the group sizes allow. Returns ``image_id -> fold_index`` (0-based)
    plus per-fold group/image counts for the report.
    """
    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    fold_of_group: dict[str, int] = {}
    for i, key in enumerate(group_keys):
        fold_of_group[key] = i % n_folds

    image_to_fold: dict[str, int] = {}
    per_fold_image_count = [0] * n_folds
    per_fold_group_count = [0] * n_folds
    for key, fold in fold_of_group.items():
        per_fold_group_count[fold] += 1
        for image_id in groups[key]:
            image_to_fold[image_id] = fold
            per_fold_image_count[fold] += 1

    return {
        "n_folds": n_folds,
        "seed": seed,
        "image_to_fold": image_to_fold,
        "per_fold_image_count": per_fold_image_count,
        "per_fold_group_count": per_fold_group_count,
    }


def build_folds(
    image_ids: Sequence[str],
    n_folds: int = N_FOLDS,
    seed: int = DEFAULT_SEED,
    image_id_to_lesion_id: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """End-to-end: group (by lesion if possible, else per-image) then
    assign to ``n_folds`` fixed, reproducible folds. This is the function
    ``scripts/run_isic2018_unet_pilot.py`` and any future training script
    should call (or, more precisely, should load the already-written
    fold-assignment file this produces -- see
    :func:`write_folds_file`/:func:`load_folds_file`)."""
    grouping = build_fold_groups(image_ids, image_id_to_lesion_id)
    assignment = assign_folds(grouping["groups"], n_folds, seed)
    return {
        "n_images": grouping["n_images"],
        "n_groups": grouping["n_groups"],
        "lesion_grouping_available": grouping["lesion_grouping_available"],
        "n_images_grouped_by_lesion": grouping["n_images_grouped_by_lesion"],
        **assignment,
    }


def write_folds_file(result: Mapping[str, Any], output_path: str | Path, dataset: str) -> None:
    """Write the fixed, reusable fold-assignment file. Future scripts
    MUST load this file (via :func:`load_folds_file`) rather than
    recompute their own fold assignment for the same dataset/seed -- the
    whole point of this module is a single frozen split shared by every
    architecture evaluated later."""
    payload = {
        "_doc": (
            "Fixed, reproducible fold assignment -- DO NOT regenerate for a "
            "different architecture evaluated on the same dataset. Reuse this "
            "file's 'image_to_fold' mapping as-is."
        ),
        "dataset": dataset,
        "n_folds": result["n_folds"],
        "seed": result["seed"],
        "lesion_grouping_available": result["lesion_grouping_available"],
        "n_images_grouped_by_lesion": result["n_images_grouped_by_lesion"],
        "n_images": result["n_images"],
        "n_groups": result["n_groups"],
        "per_fold_image_count": result["per_fold_image_count"],
        "per_fold_group_count": result["per_fold_group_count"],
        "image_to_fold": result["image_to_fold"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_folds_file(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
