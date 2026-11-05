"""Content-hash (sha256) audits for the zip-only datasets prioritized for
cross-dataset overlap detection: ISIC2018 Task1 and BCN20000.

Kept separate from ``adapters.py`` because this hashing reads every image's
full bytes (slow, ~minutes to tens of minutes), unlike the cheap structural
audits in ``adapters.py`` which only ever touch a zip's central directory or
small CSV entries. Detection only: builds manifests comparable by
``overlap.compare_by_hash`` — does not compare anything itself and does not
decide/implement any exclusion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import bcn_materialize, core


def hash_isic2018_task1_images(images_zip: str | Path) -> dict[str, Any]:
    """Hash every ``.jpg`` entry of the ISIC2018 Task1 images zip directly
    from the zip, without extracting anything to disk.
    """
    all_entries = core.zip_entry_manifest(images_zip)
    jpg_entries = [e for e in all_entries if e["name"].lower().endswith(".jpg")]
    entry_names = [e["name"] for e in jpg_entries]

    hashed = core.sha256_zip_entries(images_zip, entry_names)

    return {
        "dataset": "isic2018_task1_content",
        "source_zip": str(images_zip),
        "expected": len(entry_names),
        "processed": len(hashed["manifest"]),
        "missing": 0,
        "unreadable": hashed["unreadable"],
        "manifest": hashed["manifest"],
    }


def hash_bcn20000_images(
    outer_zip: str | Path,
    train_zip_entry: str,
    test_zip_entry: str,
    extracted_dir: str | Path,
    train_csv_row_count: int,
    test_csv_row_count: int,
) -> dict[str, Any]:
    """Materialize BCN20000's train/test inner zips under ``extracted_dir``
    (idempotent, persistent — also reused by the later visual-duplicate
    task, do not delete), then hash every ``.jpg`` entry of each inner zip
    directly from the now-materialized file, without extracting individual
    images to disk.
    """
    train_materialization = bcn_materialize.materialize_inner_zip(
        outer_zip, train_zip_entry, extracted_dir
    )
    test_materialization = bcn_materialize.materialize_inner_zip(
        outer_zip, test_zip_entry, extracted_dir
    )

    return {
        "dataset": "bcn20000_content",
        "train": _hash_inner_zip_images(
            train_materialization["dest_path"], train_csv_row_count, train_materialization
        ),
        "test": _hash_inner_zip_images(
            test_materialization["dest_path"], test_csv_row_count, test_materialization
        ),
    }


def _hash_inner_zip_images(
    inner_zip_path: str | Path, csv_row_count: int, materialization: dict[str, Any]
) -> dict[str, Any]:
    all_entries = core.zip_entry_manifest(inner_zip_path)
    jpg_entries = [e for e in all_entries if e["name"].lower().endswith(".jpg")]
    entry_names = [e["name"] for e in jpg_entries]

    hashed = core.sha256_zip_entries(inner_zip_path, entry_names)
    expected = len(entry_names)

    return {
        "source_zip": str(inner_zip_path),
        "materialization": materialization,
        "expected": expected,
        "processed": len(hashed["manifest"]),
        "missing": max(0, csv_row_count - expected),
        "unreadable": hashed["unreadable"],
        "csv_row_count": csv_row_count,
        "csv_vs_zip_count_matches": csv_row_count == expected,
        "manifest": hashed["manifest"],
    }
