"""Tests for src.data_audit.visual_duplicates (task 7).

Includes characterization tests (section further down) that pin down, in
executable code, exactly which kinds of image transformation this dHash
method does and does not detect as a visual-duplicate candidate — so a
future change to the algorithm or threshold cannot silently alter that
behavior without a test failing.
"""
from __future__ import annotations

import zipfile

import pytest
from PIL import Image

from src.data_audit import visual_duplicates as vd


def _make_base_image() -> Image.Image:
    return vd._make_synthetic_base_image()


def _to_bytes(img: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    return vd._image_to_bytes(img, fmt, **kwargs)


# ---------------------------------------------------------------------------
# Basic hash/distance behavior
# ---------------------------------------------------------------------------


def test_dhash_bytes_is_deterministic():
    data = _to_bytes(_make_base_image())
    assert vd.dhash_bytes(data) == vd.dhash_bytes(data)


def test_hamming_distance_identical_is_zero():
    h = vd.dhash_bytes(_to_bytes(_make_base_image()))
    assert vd.hamming_distance(h, h) == 0


def test_hamming_distance_counts_differing_bits():
    assert vd.hamming_distance(0b0000, 0b0000) == 0
    assert vd.hamming_distance(0b0001, 0b0000) == 1
    assert vd.hamming_distance(0b1111, 0b0000) == 4


# ---------------------------------------------------------------------------
# Characterization tests: measured Hamming distances for controlled variants
# of one synthetic base image. These numbers were measured (see module
# docstring / report for the full table), not invented, and they are the
# basis for DEFAULT_CANDIDATE_THRESHOLD: variants that are genuinely the
# same image reprocessed (copy, recompression, resize, small crop) measured
# at distance <= 6; variants with a structural change (rotation, mirror,
# large crop) measured at distance >= 16. The threshold (10) sits strictly
# between those two measured clusters.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_hash():
    return vd.dhash_bytes(_to_bytes(_make_base_image()))


def test_identical_copy_is_detected_as_candidate(base_hash):
    base = _make_base_image()
    h = vd.dhash_bytes(_to_bytes(base, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance == 0
    assert distance <= vd.DEFAULT_CANDIDATE_THRESHOLD


def test_jpeg_recompression_is_detected_as_candidate(base_hash):
    base = _make_base_image()
    h = vd.dhash_bytes(_to_bytes(base, "JPEG", quality=40))
    distance = vd.hamming_distance(base_hash, h)
    assert distance <= 6
    assert distance <= vd.DEFAULT_CANDIDATE_THRESHOLD


def test_resize_50_percent_is_detected_as_candidate(base_hash):
    base = _make_base_image()
    resized = base.resize((128, 128), Image.Resampling.LANCZOS)
    h = vd.dhash_bytes(_to_bytes(resized, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance <= 6
    assert distance <= vd.DEFAULT_CANDIDATE_THRESHOLD


def test_small_border_crop_is_detected_as_candidate(base_hash):
    base = _make_base_image()
    w, h_ = base.size
    margin_w, margin_h = int(w * 0.05), int(h_ * 0.05)
    cropped = base.crop((margin_w, margin_h, w - margin_w, h_ - margin_h))
    h = vd.dhash_bytes(_to_bytes(cropped, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance <= 6
    assert distance <= vd.DEFAULT_CANDIDATE_THRESHOLD


def test_rotation_90_degrees_is_NOT_detected_as_candidate(base_hash):
    """Documented limitation: a 90-degree rotation is the same image but
    scores far above the threshold, so this method misses it."""
    base = _make_base_image()
    rotated = base.rotate(90, expand=True)
    h = vd.dhash_bytes(_to_bytes(rotated, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance >= 16
    assert distance > vd.DEFAULT_CANDIDATE_THRESHOLD


def test_horizontal_mirror_is_NOT_detected_as_candidate(base_hash):
    """Documented limitation: a horizontal mirror is the same image but
    scores far above the threshold, so this method misses it."""
    base = _make_base_image()
    mirrored = base.transpose(Image.FLIP_LEFT_RIGHT)
    h = vd.dhash_bytes(_to_bytes(mirrored, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance >= 16
    assert distance > vd.DEFAULT_CANDIDATE_THRESHOLD


def test_large_border_crop_is_NOT_detected_as_candidate(base_hash):
    """Documented limitation: a significant (30%) crop changes the dHash's
    coarse gradient structure enough to score above the threshold."""
    base = _make_base_image()
    w, h_ = base.size
    cropped = base.crop((int(w * 0.3), int(h_ * 0.3), int(w * 0.7), int(h_ * 0.7)))
    h = vd.dhash_bytes(_to_bytes(cropped, "PNG"))
    distance = vd.hamming_distance(base_hash, h)
    assert distance >= 16
    assert distance > vd.DEFAULT_CANDIDATE_THRESHOLD


def test_measure_controlled_variant_distances_matches_threshold_classification():
    """The function the generated report calls must classify each variant
    the same way the individual characterization tests above do."""
    results = {r["variant"]: r for r in vd.measure_controlled_variant_distances()}

    assert set(results) == {
        "identical_copy",
        "jpeg_recompress_q40",
        "resize_50pct",
        "crop_5pct_border",
        "rotate_90",
        "mirror_horizontal",
        "crop_30pct_border",
    }
    for name in ("identical_copy", "jpeg_recompress_q40", "resize_50pct", "crop_5pct_border"):
        assert results[name]["is_candidate"] is True, name
    for name in ("rotate_90", "mirror_horizontal", "crop_30pct_border"):
        assert results[name]["is_candidate"] is False, name


# ---------------------------------------------------------------------------
# Unreadable input handling
# ---------------------------------------------------------------------------


def test_dhash_disk_images_handles_unreadable_file_without_aborting(tmp_path):
    good_path = tmp_path / "good.png"
    good_path.write_bytes(_to_bytes(_make_base_image()))
    bad_path = tmp_path / "bad.jpg"
    bad_path.write_bytes(b"this is not a valid image file")

    result = vd.dhash_disk_images([good_path, bad_path], base_dir=tmp_path)

    assert result["expected"] == 2
    assert result["processed"] == 1
    assert len(result["unreadable"]) == 1
    assert result["unreadable"][0]["path"] == "bad.jpg"
    assert result["manifest"][0]["path"] == "good.png"


def test_dhash_zip_entries_handles_unreadable_entry_without_aborting(tmp_path):
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("good.png", _to_bytes(_make_base_image()))
        zf.writestr("bad.jpg", b"not a valid image")

    result = vd.dhash_zip_entries(zip_path, ["good.png", "bad.jpg"])

    assert result["expected"] == 2
    assert result["processed"] == 1
    assert len(result["unreadable"]) == 1
    assert result["unreadable"][0]["path"] == "bad.jpg"
    assert result["manifest"][0]["path"] == "good.png"


# ---------------------------------------------------------------------------
# Vectorized candidate search
# ---------------------------------------------------------------------------


def _manifest(entries: dict[str, int]) -> list[dict]:
    return [{"path": path, "dhash": h} for path, h in entries.items()]


def test_find_candidate_pairs_finds_pairs_within_threshold():
    manifest_a = _manifest(
        {
            "a1": 0b0000000000000000000000000000000000000000000000000000000000000000,
            "a2": 0b0000000000000000000000000000000000000000000000000000000000000111,  # distance 3 from a1
            "a3": 0xFFFFFFFFFFFFFFFF,  # far from everything in b
            "a4": 0b1000000000000000000000000000000000000000000000000000000000000000,  # distance 1 from a1
            "a5": 0b0000000000000000000000000000000000000000000000000000000011111111,  # distance 8 from a1
        }
    )
    manifest_b = _manifest(
        {
            "b1": 0b0000000000000000000000000000000000000000000000000000000000000000,  # == a1
            "b2": 0b0000000000000000000000000000000000000000000000000000000000000011,  # distance 2 from a1
        }
    )

    pairs = vd.find_candidate_pairs(manifest_a, manifest_b, threshold=5)
    pairs_by_key = {(p["path_a"], p["path_b"]): p["distance"] for p in pairs}

    assert pairs_by_key[("a1", "b1")] == 0
    assert pairs_by_key[("a1", "b2")] == 2
    assert pairs_by_key[("a2", "b1")] == 3
    assert pairs_by_key[("a2", "b2")] == 1
    assert pairs_by_key[("a4", "b1")] == 1
    assert pairs_by_key[("a4", "b2")] == 3
    assert ("a3", "b1") not in pairs_by_key
    assert ("a3", "b2") not in pairs_by_key
    assert ("a5", "b1") not in pairs_by_key  # distance 8 > threshold 5
    assert ("a5", "b2") not in pairs_by_key


def test_find_candidate_pairs_respects_block_size_smaller_than_input():
    manifest_a = _manifest({f"a{i}": i for i in range(7)})
    manifest_b = _manifest({"b0": 0})

    pairs_block1 = vd.find_candidate_pairs(manifest_a, manifest_b, threshold=1, block_size=1)
    pairs_default = vd.find_candidate_pairs(manifest_a, manifest_b, threshold=1, block_size=1000)

    assert {(p["path_a"], p["distance"]) for p in pairs_block1} == {
        (p["path_a"], p["distance"]) for p in pairs_default
    }
    # hamming(i, 0) == popcount(i): a0->0, a1->1, a2->1, a3->2, a4->1, a5->2, a6->2
    assert {p["path_a"] for p in pairs_block1} == {"a0", "a1", "a2", "a4"}


def test_find_candidate_pairs_empty_inputs_return_empty_list():
    assert vd.find_candidate_pairs([], _manifest({"b": 0}), threshold=5) == []
    assert vd.find_candidate_pairs(_manifest({"a": 0}), [], threshold=5) == []
    assert vd.find_candidate_pairs([], [], threshold=5) == []


# ---------------------------------------------------------------------------
# Null-model baseline (used by the report to contextualize real candidate counts)
# ---------------------------------------------------------------------------


def test_expected_random_candidate_count_threshold_64_is_all_pairs():
    """At threshold == hash width, every pair qualifies (distance can never
    exceed the number of bits), so the expected count must equal n_a * n_b."""
    count = vd.expected_random_candidate_count(10, 20, threshold=64, hash_bits=64)
    assert count == pytest.approx(200.0)


def test_expected_random_candidate_count_threshold_0_is_exact_hash_match_probability():
    count = vd.expected_random_candidate_count(1000, 1000, threshold=0, hash_bits=64)
    expected = 1000 * 1000 / (2**64)
    assert count == pytest.approx(expected)


def test_expected_random_candidate_count_scales_with_pair_count():
    small = vd.expected_random_candidate_count(10, 10, threshold=10)
    large = vd.expected_random_candidate_count(100, 10, threshold=10)
    assert large == pytest.approx(small * 10)
