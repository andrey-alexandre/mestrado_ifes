"""Tests for src.data_audit.phash (task 8).

Mirrors tests/data_audit/test_visual_duplicates.py's characterization-test
pattern (measured behavior pinned in code, not just prose) but for pHash.
Real-image validation (task 8 item 2's requirement) lives in
scripts/run_visual_duplicate_pilot.py and its generated report — these
tests use only the synthetic image, same as task 7's dHash tests, to avoid
any dependency on patient data in the automated test suite.
"""
from __future__ import annotations

import zipfile

import pytest

from src.data_audit import phash
from src.data_audit import visual_duplicates as vd


def test_phash_bytes_is_deterministic():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    assert phash.phash_bytes(data) == phash.phash_bytes(data)


def test_phash_bytes_produces_full_width_hash():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    h = phash.phash_bytes(data)
    assert 0 <= h < (1 << phash.PHASH_LOW_FREQ_SIZE**2)


def test_hamming_distance_identical_phash_is_zero():
    h = phash.phash_bytes(vd._image_to_bytes(vd._make_synthetic_base_image()))
    assert vd.hamming_distance(h, h) == 0


# ---------------------------------------------------------------------------
# Characterization: measured pHash distances for the same controlled
# variants used for dHash in task 7, on the synthetic base image. Numbers
# documented here are observed, not invented; real-image validation against
# real reprocessing/hard-pair benchmarks is done separately by
# scripts/run_visual_duplicate_pilot.py (patient data must not appear in the
# automated test suite).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_phash():
    return phash.phash_bytes(vd._image_to_bytes(vd._make_synthetic_base_image()))


def test_identical_copy_has_zero_distance(base_phash):
    variants = vd.generate_controlled_variants()
    h = phash.phash_bytes(variants["identical_copy"])
    assert vd.hamming_distance(base_phash, h) == 0


def test_jpeg_recompression_has_low_distance(base_phash):
    variants = vd.generate_controlled_variants()
    h = phash.phash_bytes(variants["jpeg_recompress_q40"])
    distance = vd.hamming_distance(base_phash, h)
    assert distance <= 10


def test_resize_50_percent_has_low_distance(base_phash):
    variants = vd.generate_controlled_variants()
    h = phash.phash_bytes(variants["resize_50pct"])
    distance = vd.hamming_distance(base_phash, h)
    assert distance <= 10


# ---------------------------------------------------------------------------
# Unreadable input handling
# ---------------------------------------------------------------------------


def test_phash_disk_images_handles_unreadable_file_without_aborting(tmp_path):
    good_path = tmp_path / "good.png"
    good_path.write_bytes(vd._image_to_bytes(vd._make_synthetic_base_image()))
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"this is not a valid image file")

    result = phash.phash_disk_images([good_path, bad_path], base_dir=tmp_path)

    assert result["expected"] == 2
    assert result["processed"] == 1
    assert len(result["unreadable"]) == 1
    assert result["unreadable"][0]["path"] == "bad.jpg"
    assert result["manifest"][0]["path"] == "good.png"


def test_phash_zip_entries_handles_unreadable_entry_without_aborting(tmp_path):
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("good.png", vd._image_to_bytes(vd._make_synthetic_base_image()))
        zf.writestr("bad.jpg", b"not a valid image")

    result = phash.phash_zip_entries(zip_path, ["good.png", "bad.jpg"])

    assert result["expected"] == 2
    assert result["processed"] == 1
    assert len(result["unreadable"]) == 1
    assert result["unreadable"][0]["path"] == "bad.jpg"
    assert result["manifest"][0]["path"] == "good.png"
