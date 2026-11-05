"""Tests for src.data_audit.preprocessing_audit (task 8 item 1).

Synthetic images only (no patient data), including a synthetic image with
an explicit EXIF orientation tag to verify PIL.Image.open's documented
non-rotating behavior on real bytes rather than assuming it.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.data_audit import preprocessing_audit as pa


def _make_image(size=(100, 60), color=(255, 255, 255), fill=(200, 0, 0)) -> Image.Image:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, size[0] // 2, size[1] // 2), fill=fill)
    return image


def _to_bytes(image: Image.Image, fmt="JPEG", exif=None) -> bytes:
    buf = io.BytesIO()
    if exif is not None:
        image.save(buf, format=fmt, exif=exif)
    else:
        image.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# sample_paths
# ---------------------------------------------------------------------------


def test_sample_paths_is_deterministic_for_same_seed():
    entries = [{"path": f"img{i}.jpg"} for i in range(100)]
    sample1 = pa.sample_paths(entries, n=10, seed=42)
    sample2 = pa.sample_paths(entries, n=10, seed=42)
    assert sample1 == sample2
    assert len(sample1) == 10


def test_sample_paths_returns_all_when_n_exceeds_population():
    entries = [{"path": f"img{i}.jpg"} for i in range(5)]
    sample = pa.sample_paths(entries, n=100, seed=1)
    assert len(sample) == 5


def test_sample_paths_different_seeds_differ():
    entries = [{"path": f"img{i}.jpg"} for i in range(200)]
    sample_a = pa.sample_paths(entries, n=20, seed=1)
    sample_b = pa.sample_paths(entries, n=20, seed=2)
    assert sample_a != sample_b


# ---------------------------------------------------------------------------
# exif_orientation_audit
# ---------------------------------------------------------------------------


def test_exif_orientation_audit_detects_non_default_orientation():
    base = _make_image()
    exif = base.getexif()
    exif[0x0112] = 6  # non-default orientation
    rotated_data = _to_bytes(base, exif=exif)
    normal_data = _to_bytes(_make_image())

    samples = [("rotated.jpg", rotated_data), ("normal.jpg", normal_data)]
    result = pa.exif_orientation_audit(samples)

    assert result["n_processed"] == 2
    assert result["n_non_default_orientation"] == 1
    assert result["non_default_examples"][0]["path"] == "rotated.jpg"
    assert result["non_default_examples"][0]["orientation"] == 6


def test_verify_open_does_not_apply_exif_orientation():
    """PIL.Image.open must NOT apply EXIF orientation by itself -- this is
    Pillow's documented behavior; here it's measured on real encoded bytes,
    not assumed."""
    result = pa.verify_open_does_not_apply_exif_orientation()

    assert result["open_applies_orientation_automatically"] is False
    assert result["size_from_open"] == result["raw_size"]


def test_exif_orientation_audit_handles_unreadable_sample():
    result = pa.exif_orientation_audit([("bad.jpg", b"not an image")])
    assert result["n_processed"] == 0
    assert len(result["unreadable"]) == 1


# ---------------------------------------------------------------------------
# dimension_audit
# ---------------------------------------------------------------------------


def test_dimension_audit_reports_width_height_aspect_stats():
    samples = [
        (f"img{i}.jpg", _to_bytes(_make_image(size=(100 + i * 10, 50))))
        for i in range(5)
    ]
    result = pa.dimension_audit(samples, dhash_target_size=(9, 8))

    assert result["n_processed"] == 5
    assert result["width"]["min"] == 100
    assert result["width"]["max"] == 140
    assert result["height"]["min"] == 50 == result["height"]["max"]
    assert result["dhash_target_aspect_ratio"] == pytest.approx(9 / 8)
    assert result["mean_abs_deviation_from_target_aspect"] > 0


def test_dimension_audit_handles_unreadable_sample():
    result = pa.dimension_audit([("bad.jpg", b"garbage")])
    assert result["n_processed"] == 0
    assert len(result["unreadable"]) == 1


# ---------------------------------------------------------------------------
# degenerate_image_audit
# ---------------------------------------------------------------------------


def test_degenerate_image_audit_flags_uniform_image():
    uniform = Image.new("L", (50, 50), 128)
    textured = _make_image()

    samples = [("flat.png", _to_bytes(uniform, fmt="PNG")), ("textured.jpg", _to_bytes(textured))]
    result = pa.degenerate_image_audit(samples)

    assert result["n_flagged"] == 1
    assert result["flagged"][0]["path"] == "flat.png"
    assert result["flagged"][0]["near_uniform"] is True


def test_degenerate_image_audit_flags_near_black_image():
    near_black = Image.new("L", (50, 50), 2)
    samples = [("black.png", _to_bytes(near_black, fmt="PNG"))]
    result = pa.degenerate_image_audit(samples)

    assert result["n_flagged"] == 1
    assert result["flagged"][0]["near_extreme_mean"] is True


def test_degenerate_image_audit_does_not_flag_textured_image():
    textured = _make_image()
    samples = [("textured.jpg", _to_bytes(textured))]
    result = pa.degenerate_image_audit(samples)
    assert result["n_flagged"] == 0


# ---------------------------------------------------------------------------
# hub_hash_audit
# ---------------------------------------------------------------------------


def test_hub_hash_audit_counts_edges_per_distinct_hash():
    # 3 candidates, side A paths a1/a2 share the same hash (hub), b1/b2/b3 distinct.
    candidates = [
        {"path_a_ham10000": "a1.jpg", "path_b": "b1.jpg"},
        {"path_a_ham10000": "a2.jpg", "path_b": "b2.jpg"},
        {"path_a_ham10000": "a1.jpg", "path_b": "b3.jpg"},
    ]
    path_to_dhash_a = {"a1.jpg": 111, "a2.jpg": 111}  # same hash value -> a hub
    path_to_dhash_b = {"b1.jpg": 222, "b2.jpg": 333, "b3.jpg": 444}

    result = pa.hub_hash_audit(candidates, path_to_dhash_a, path_to_dhash_b, top_n=5)

    assert result["total_candidates"] == 3
    top_hash_entry = result["top_hashes"][0]
    assert top_hash_entry["dhash"] == 111
    assert top_hash_entry["edge_count"] == 3  # hash 111 appears in all 3 candidate edges (a1 twice, a2 once)
    assert result["n_distinct_hashes_involved"] == 4  # {111, 222, 333, 444}
    # All 3 candidates have hash 111 on one side -> all 3 touch the top hash set.
    assert result["n_candidates_touching_top_hashes"] == 3
    assert result["top_hashes_share_of_candidates"] == pytest.approx(1.0)


def test_hub_hash_audit_share_cannot_exceed_one_even_when_both_sides_are_hubs():
    """Regression test: summing raw edge counts across top hashes can double
    count a candidate whose two endpoints are both hub hashes, which must
    NOT be reflected in top_hashes_share_of_candidates (a share, bounded by
    1.0, not a raw edge-count ratio)."""
    candidates = [{"path_a_ham10000": "a1.jpg", "path_b": "b1.jpg"}]
    path_to_dhash_a = {"a1.jpg": 111}
    path_to_dhash_b = {"b1.jpg": 222}  # both 111 and 222 will be "top" hashes

    result = pa.hub_hash_audit(candidates, path_to_dhash_a, path_to_dhash_b, top_n=5)

    assert result["top_hashes_edge_count_sum"] == 2  # both endpoints counted
    assert result["n_candidates_touching_top_hashes"] == 1  # but only 1 real candidate
    assert result["top_hashes_share_of_candidates"] == pytest.approx(1.0)


def test_hub_hash_audit_empty_candidates_returns_zero_share():
    result = pa.hub_hash_audit([], {}, {}, top_n=5)
    assert result["total_candidates"] == 0
    assert result["top_hashes_share_of_candidates"] == 0.0
