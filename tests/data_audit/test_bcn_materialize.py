import zipfile

from src.data_audit import bcn_materialize


def _make_outer_zip(tmp_path, entry_name="inner.zip", content=b"\x00" * 123):
    outer_zip = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.writestr(entry_name, content)
    return outer_zip


def test_materialize_inner_zip_extracts_when_missing(tmp_path):
    content = b"\x01" * 456
    outer_zip = _make_outer_zip(tmp_path, content=content)
    dest_dir = tmp_path / "_extracted"

    result = bcn_materialize.materialize_inner_zip(outer_zip, "inner.zip", dest_dir)

    dest_path = dest_dir / "inner.zip"
    assert result["materialized"] is True
    assert result["reused_existing"] is False
    assert result["dest_path"] == str(dest_path)
    assert result["size"] == len(content)
    assert result["size_matches_expected"] is True
    assert dest_path.read_bytes() == content


def test_materialize_inner_zip_reuses_existing_without_re_extracting(tmp_path):
    content = b"\x02" * 789
    outer_zip = _make_outer_zip(tmp_path, content=content)
    dest_dir = tmp_path / "_extracted"
    dest_dir.mkdir()
    dest_path = dest_dir / "inner.zip"
    # Pre-seed the destination with the exact expected size but different
    # bytes, so a passing "reused" assertion proves extraction was skipped.
    dest_path.write_bytes(b"\x99" * len(content))

    result = bcn_materialize.materialize_inner_zip(outer_zip, "inner.zip", dest_dir)

    assert result["materialized"] is False
    assert result["reused_existing"] is True
    assert dest_path.read_bytes() == b"\x99" * len(content)


def test_materialize_inner_zip_re_extracts_when_size_differs(tmp_path):
    content = b"\x03" * 321
    outer_zip = _make_outer_zip(tmp_path, content=content)
    dest_dir = tmp_path / "_extracted"
    dest_dir.mkdir()
    dest_path = dest_dir / "inner.zip"
    dest_path.write_bytes(b"\x00")  # wrong size -> must be re-extracted

    result = bcn_materialize.materialize_inner_zip(outer_zip, "inner.zip", dest_dir)

    assert result["materialized"] is True
    assert result["reused_existing"] is False
    assert dest_path.read_bytes() == content
