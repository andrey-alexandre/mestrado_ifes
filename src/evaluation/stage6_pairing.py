"""Stage 6 — pairs saved pipeline results to the frozen HAM10000 sample by
``lesion_id``, not by image path (paths can differ across machines/runs; the
manifest's ``image_id`` -- the image filename stem -- is the stable bridge).

Nothing here reads a real pipeline result file or the real HAM10000 metadata
-- callers supply the manifest rows and the results dict; verification uses
fabricated data (``scripts/verify_stage6_evaluator.py``).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

REQUIRED_MANIFEST_COLUMNS = ("lesion_id", "image_id", "dx", "binary_target")


@dataclass
class ManifestIndex:
    """Indexed view of the frozen sample manifest."""
    lesion_ids: list[str]
    image_id_to_lesion_id: dict[str, str]
    lesion_id_to_dx: dict[str, str]
    lesion_id_to_binary_target: dict[str, str]
    lesion_ids_by_category: dict[str, list[str]]  # category = dx


@dataclass
class PairingReport:
    """Result of pairing ONE system/condition's saved results to the
    manifest. Every field below is populated explicitly -- never omitted,
    even when empty -- so a missing/duplicate/orphan case is always visible
    in the report, not inferred by absence of a key."""
    lesion_id_to_event: dict[str, Any]
    missing_lesion_ids: list[str]
    duplicate_lesion_ids: dict[str, list[str]]  # lesion_id -> list of source paths
    orphan_paths: list[str]  # paths whose derived image_id has no manifest row


def load_manifest_rows(csv_path: str | Path) -> list[dict[str, str]]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    missing_cols = [c for c in REQUIRED_MANIFEST_COLUMNS if rows and c not in rows[0]]
    if missing_cols:
        raise ValueError(f"Manifesto sem colunas esperadas: {missing_cols}")
    return rows


def build_manifest_index(rows: Sequence[Mapping[str, str]]) -> ManifestIndex:
    lesion_ids: list[str] = []
    image_id_to_lesion_id: dict[str, str] = {}
    lesion_id_to_dx: dict[str, str] = {}
    lesion_id_to_binary_target: dict[str, str] = {}
    lesion_ids_by_category: dict[str, list[str]] = {}

    for row in rows:
        lesion_id = row["lesion_id"]
        image_id = row["image_id"]
        dx = row["dx"]
        binary_target = row["binary_target"]

        lesion_ids.append(lesion_id)
        image_id_to_lesion_id[image_id] = lesion_id
        lesion_id_to_dx[lesion_id] = dx
        lesion_id_to_binary_target[lesion_id] = binary_target
        lesion_ids_by_category.setdefault(dx, []).append(lesion_id)

    return ManifestIndex(
        lesion_ids=lesion_ids,
        image_id_to_lesion_id=image_id_to_lesion_id,
        lesion_id_to_dx=lesion_id_to_dx,
        lesion_id_to_binary_target=lesion_id_to_binary_target,
        lesion_ids_by_category=lesion_ids_by_category,
    )


def derive_image_id_from_path(path: str) -> str:
    """'.../ISIC_0024579.jpg' -> 'ISIC_0024579'. Pure string/path operation,
    no filesystem access (the path need not exist on this machine)."""
    return Path(path).stem


def pair_results_by_lesion(
    manifest_index: ManifestIndex, results_by_path: Mapping[str, Any]
) -> PairingReport:
    """Pairs one system/condition's saved results (``path -> event``) to
    the manifest's lesion universe.

    - A path whose derived ``image_id`` is not in the manifest is an
      ``orphan`` -- listed by path, never silently dropped.
    - Two or more paths mapping to the SAME ``lesion_id`` is a
      ``duplicate`` -- all contributing paths are listed; none of them is
      arbitrarily chosen as "the" event here (the caller decides, if it
      must, but the ambiguity is surfaced first).
    - A manifest ``lesion_id`` with no matching path at all is ``missing``.
    """
    lesion_id_to_paths: dict[str, list[str]] = {}
    orphan_paths: list[str] = []

    for path in results_by_path:
        image_id = derive_image_id_from_path(path)
        lesion_id = manifest_index.image_id_to_lesion_id.get(image_id)
        if lesion_id is None:
            orphan_paths.append(path)
            continue
        lesion_id_to_paths.setdefault(lesion_id, []).append(path)

    lesion_id_to_event: dict[str, Any] = {}
    duplicate_lesion_ids: dict[str, list[str]] = {}
    for lesion_id, paths in lesion_id_to_paths.items():
        if len(paths) > 1:
            duplicate_lesion_ids[lesion_id] = sorted(paths)
        # Even when duplicated, still expose ONE event (the first, by sorted
        # path) so downstream indicator computation has a value to use --
        # the duplicate is reported, not silently averaged or hidden.
        lesion_id_to_event[lesion_id] = results_by_path[sorted(paths)[0]]

    missing_lesion_ids = sorted(
        set(manifest_index.lesion_ids) - set(lesion_id_to_event.keys())
    )

    return PairingReport(
        lesion_id_to_event=lesion_id_to_event,
        missing_lesion_ids=missing_lesion_ids,
        duplicate_lesion_ids=duplicate_lesion_ids,
        orphan_paths=sorted(orphan_paths),
    )
