"""Generic, dataset-agnostic structural-audit primitives.

Pure functions: no dataset-specific knowledge, no network/model calls, no
sampling. Operate on already-extracted files on disk, on zip central
directories (via ``infolist``), or on in-memory rows (list of dict).
"""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def check_path_exists(path: str | Path) -> dict[str, Any]:
    """Report whether ``path`` exists, without raising."""
    p = Path(path)
    exists = p.exists()
    if not exists:
        reason = "not found"
    elif p.is_dir():
        reason = "ok (directory)"
    elif p.is_file():
        reason = "ok (file)"
    else:
        reason = "ok (exists, not a regular file or directory)"
    return {"path": str(p), "exists": exists, "reason": reason}


def pair_by_id(ids_a: Iterable[str], ids_b: Iterable[str]) -> dict[str, list[str]]:
    """Set-pair two id collections, reporting matches and the leftovers on each side."""
    set_a = set(ids_a)
    set_b = set(ids_b)
    return {
        "matched": sorted(set_a & set_b),
        "only_in_a": sorted(set_a - set_b),
        "only_in_b": sorted(set_b - set_a),
    }


def find_missing_or_conflicting(
    rows: Sequence[Mapping[str, Any]], id_field: str, value_field: str
) -> dict[str, Any]:
    """Within ``rows``, find entries missing ``id_field`` and ids whose
    ``value_field`` disagrees across repeated occurrences.

    ``missing_id`` holds the offending rows as-is. ``conflicting`` maps each
    conflicting id to the sorted list of distinct values seen for it.
    """
    values_by_id: dict[str, set[Any]] = defaultdict(set)
    missing_id: list[Mapping[str, Any]] = []
    for row in rows:
        rid = row.get(id_field)
        if rid is None or str(rid).strip() == "":
            missing_id.append(row)
            continue
        values_by_id[rid].add(row.get(value_field))

    conflicting = {
        rid: sorted(v for v in vals if v is not None)
        for rid, vals in values_by_id.items()
        if len(vals) > 1
    }
    return {"missing_id": missing_id, "conflicting": conflicting}


def category_counts(rows: Sequence[Mapping[str, Any]], field: str) -> Counter:
    """Count occurrences of each distinct value of ``field`` across ``rows``."""
    return Counter(row.get(field) for row in rows)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file already on disk, read in chunks (no full read into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(
    file_paths: Iterable[str | Path], base_dir: str | Path
) -> list[dict[str, Any]]:
    """Manifest (relative path, size, sha256) for files already extracted on disk."""
    base = Path(base_dir)
    manifest = []
    for fp in file_paths:
        fp = Path(fp)
        stat = fp.stat()
        manifest.append(
            {
                "path": str(fp.relative_to(base)).replace("\\", "/"),
                "size": stat.st_size,
                "sha256": sha256_file(fp),
            }
        )
    return manifest


def zip_entry_manifest(
    zip_path: str | Path, entry_names: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """List a zip's central directory (name + size), without extracting anything.

    If ``entry_names`` is given, only those entries are returned (still read
    from the central directory, not the entry bodies).
    """
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if entry_names is not None:
            wanted = set(entry_names)
            infos = [i for i in infos if i.filename in wanted]
        return [{"name": i.filename, "size": i.file_size} for i in infos]


def read_zip_entry_text(
    zip_path: str | Path, entry_name: str, encoding: str = "utf-8"
) -> str:
    """Read one small entry's bytes directly from the zip, in memory, as text.

    Does not extract the entry to disk and does not touch any other entry.
    """
    with zipfile.ZipFile(zip_path) as zf:
        return zf.read(entry_name).decode(encoding)


def parse_csv_text(text: str, delimiter: str = ",") -> list[dict[str, Any]]:
    """Parse CSV text (e.g. from :func:`read_zip_entry_text`) into row dicts."""
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))
