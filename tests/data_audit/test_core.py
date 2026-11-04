import hashlib
import zipfile

import pytest

from src.data_audit import core


def test_check_path_exists_for_missing_path(tmp_path):
    result = core.check_path_exists(tmp_path / "does_not_exist")
    assert result["exists"] is False
    assert "not found" in result["reason"]


def test_check_path_exists_for_existing_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    result = core.check_path_exists(f)
    assert result["exists"] is True
    assert "file" in result["reason"]


def test_pair_by_id_partial_overlap():
    result = core.pair_by_id(["a", "b", "c"], ["b", "c", "d"])
    assert result["matched"] == ["b", "c"]
    assert result["only_in_a"] == ["a"]
    assert result["only_in_b"] == ["d"]


def test_pair_by_id_no_overlap():
    result = core.pair_by_id(["a"], ["z"])
    assert result["matched"] == []
    assert result["only_in_a"] == ["a"]
    assert result["only_in_b"] == ["z"]


def test_find_missing_or_conflicting_flags_missing_id():
    rows = [
        {"id": "x1", "diagnosis": "MEL"},
        {"id": "", "diagnosis": "NV"},
        {"id": None, "diagnosis": "BCC"},
    ]
    result = core.find_missing_or_conflicting(rows, "id", "diagnosis")
    assert len(result["missing_id"]) == 2
    assert result["conflicting"] == {}


def test_find_missing_or_conflicting_flags_conflicting_values():
    rows = [
        {"id": "x1", "diagnosis": "MEL"},
        {"id": "x1", "diagnosis": "NV"},
        {"id": "x2", "diagnosis": "BCC"},
        {"id": "x2", "diagnosis": "BCC"},
    ]
    result = core.find_missing_or_conflicting(rows, "id", "diagnosis")
    assert result["missing_id"] == []
    assert result["conflicting"] == {"x1": ["MEL", "NV"]}


def test_category_counts():
    rows = [{"dx": "mel"}, {"dx": "nv"}, {"dx": "mel"}]
    counts = core.category_counts(rows, "dx")
    assert counts["mel"] == 2
    assert counts["nv"] == 1


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "a.bin"
    content = b"some bytes" * 1000
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert core.sha256_file(f) == expected


def test_build_file_manifest_relative_paths_and_hashes(tmp_path):
    (tmp_path / "sub").mkdir()
    f1 = tmp_path / "sub" / "one.txt"
    f1.write_bytes(b"one")
    f2 = tmp_path / "two.txt"
    f2.write_bytes(b"two")

    manifest = core.build_file_manifest([f1, f2], base_dir=tmp_path)
    by_path = {m["path"]: m for m in manifest}

    assert by_path["sub/one.txt"]["size"] == 3
    assert by_path["sub/one.txt"]["sha256"] == hashlib.sha256(b"one").hexdigest()
    assert by_path["two.txt"]["size"] == 3


def test_zip_entry_manifest_lists_without_extracting(tmp_path):
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("small.csv", "a,b\n1,2\n")
        zf.writestr("big_inner.zip", b"\x00" * 50)

    manifest = core.zip_entry_manifest(zip_path)
    names = {m["name"]: m["size"] for m in manifest}
    assert names["small.csv"] == len("a,b\n1,2\n")
    assert names["big_inner.zip"] == 50
    # nothing should have been extracted to disk alongside the zip
    assert not (tmp_path / "small.csv").exists()
    assert not (tmp_path / "big_inner.zip").exists()


def test_zip_entry_manifest_filters_by_entry_names(tmp_path):
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.csv", "x")
        zf.writestr("b.csv", "yy")

    manifest = core.zip_entry_manifest(zip_path, entry_names=["b.csv"])
    assert len(manifest) == 1
    assert manifest[0]["name"] == "b.csv"
    assert manifest[0]["size"] == 2


def test_read_zip_entry_text_reads_inner_entry_without_extracting(tmp_path):
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("inner/data.csv", "id,val\n1,foo\n")
        zf.writestr("inner/other_big.zip", b"\x00" * 1000)

    text = core.read_zip_entry_text(zip_path, "inner/data.csv")
    assert text == "id,val\n1,foo\n"
    assert not (tmp_path / "inner").exists()


def test_parse_csv_text_returns_row_dicts():
    rows = core.parse_csv_text("id,val\n1,foo\n2,bar\n")
    assert rows == [{"id": "1", "val": "foo"}, {"id": "2", "val": "bar"}]


def test_parse_csv_text_pipe_delimiter():
    rows = core.parse_csv_text("id|val\n1|foo\n", delimiter="|")
    assert rows == [{"id": "1", "val": "foo"}]
