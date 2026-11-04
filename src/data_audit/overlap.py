"""Cross-dataset overlap/duplicate detection primitives.

Pure functions, detection only: no decision about exclusion is made or
implied here. Operate on manifests/id lists already produced by
:mod:`src.data_audit.core`/`adapters` — no dataset access, no hashing,
no re-extraction.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import core


def compare_by_hash(
    manifest_a: Sequence[Mapping[str, Any]], manifest_b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Find sha256 values present in both manifests (byte-identical files).

    ``manifest_a``/``manifest_b`` are lists of ``{path, size, sha256}`` dicts,
    the format already produced by :func:`core.build_file_manifest`. Returns
    the shared hashes with the paths on each side that carry them.
    """
    paths_by_hash_a: dict[str, list[str]] = {}
    for entry in manifest_a:
        paths_by_hash_a.setdefault(entry["sha256"], []).append(entry["path"])

    paths_by_hash_b: dict[str, list[str]] = {}
    for entry in manifest_b:
        paths_by_hash_b.setdefault(entry["sha256"], []).append(entry["path"])

    shared_hashes = sorted(set(paths_by_hash_a) & set(paths_by_hash_b))
    duplicates = {
        sha256: {"paths_a": sorted(paths_by_hash_a[sha256]), "paths_b": sorted(paths_by_hash_b[sha256])}
        for sha256 in shared_hashes
    }
    return {
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }


def compare_by_id(ids_a: Sequence[str], ids_b: Sequence[str]) -> dict[str, Any]:
    """Find identifiers present in both id collections.

    Thin wrapper over :func:`core.pair_by_id`, reusing its already-tested
    set-pairing logic instead of duplicating it.
    """
    return core.pair_by_id(ids_a, ids_b)
