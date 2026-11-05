"""Materialize BCN20000's inner per-split zips to a real file on disk.

BCN20000's two image zips (``BCN_20k_train.zip``/``BCN_20k_test.zip``) live
nested inside one outer zip. Python's ``zipfile`` can list/read a nested
zip's entries only by streaming the outer zip sequentially; it cannot open
the inner zip for random access without first giving it a real, seekable
file. This module does that one-time extraction, persistently, so the inner
zips can then be hashed entry-by-entry via ``core.sha256_zip_entries``
without ever extracting the individual images themselves.

Idempotent: re-running does not re-extract a destination file that already
exists with the size recorded in the outer zip's central directory, so an
interrupted run can simply be restarted.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


def materialize_inner_zip(
    outer_zip: str | Path, entry_name: str, dest_dir: str | Path
) -> dict[str, Any]:
    """Extract ``entry_name`` from ``outer_zip`` into ``dest_dir`` if needed.

    If a file already exists at ``dest_dir / Path(entry_name).name`` whose
    size matches the outer zip's central-directory record for
    ``entry_name``, extraction is skipped entirely (``reused_existing``).
    Otherwise the single entry is extracted (no other entry is touched).
    """
    dest_dir = Path(dest_dir)
    dest_path = dest_dir / Path(entry_name).name

    with zipfile.ZipFile(outer_zip) as outer_zf:
        info = outer_zf.getinfo(entry_name)
        expected_size = info.file_size

        if dest_path.is_file() and dest_path.stat().st_size == expected_size:
            return {
                "materialized": False,
                "reused_existing": True,
                "dest_path": str(dest_path),
                "size": dest_path.stat().st_size,
                "size_matches_expected": True,
            }

        dest_dir.mkdir(parents=True, exist_ok=True)
        extracted_path = Path(outer_zf.extract(info, path=dest_dir))
        if extracted_path != dest_path:
            extracted_path.replace(dest_path)

    actual_size = dest_path.stat().st_size
    return {
        "materialized": True,
        "reused_existing": False,
        "dest_path": str(dest_path),
        "size": actual_size,
        "size_matches_expected": actual_size == expected_size,
    }
