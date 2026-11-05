"""Tests for src.data_audit.visual_duplicate_pilot_v2 (task 9).

Only synthetic images -- no patient data in the automated test suite, same
convention as tests/data_audit/test_visual_duplicates.py,
test_phash.py and test_geometric_verification.py. Real-image validation
(64 vs 256-bit pHash, dHash funnel-entry coverage, unconditional ORB on the
2,000-pair sample) is done separately by
scripts/run_visual_duplicate_pilot_v2.py and its generated report.
"""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.data_audit import phash
from src.data_audit import visual_duplicate_pilot_v2 as v2
from src.data_audit import visual_duplicates as vd

# ---------------------------------------------------------------------------
# phash_bytes at low_freq_size=8 (64 bits) vs 16 (256 bits, task 8's default)
# ---------------------------------------------------------------------------


def test_phash_bytes_64_bit_width():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    h = phash.phash_bytes(data, low_freq_size=v2.PHASH_64_LOW_FREQ_SIZE)
    assert 0 <= h < (1 << 64)


def test_phash_bytes_256_bit_width():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    h = phash.phash_bytes(data, low_freq_size=v2.PHASH_256_LOW_FREQ_SIZE)
    assert 0 <= h < (1 << 256)


def test_phash_bytes_64_and_256_are_both_deterministic_and_identical_for_same_image():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    assert phash.phash_bytes(data, low_freq_size=8) == phash.phash_bytes(data, low_freq_size=8)
    assert phash.phash_bytes(data, low_freq_size=16) == phash.phash_bytes(data, low_freq_size=16)


def test_compute_phash_distances_identical_images_is_zero_both_widths():
    data = vd._image_to_bytes(vd._make_synthetic_base_image())
    result = v2.compute_phash_distances(data, data)
    assert result == {"phash_distance_64": 0, "phash_distance_256": 0}


def test_compute_phash_distances_matches_direct_phash_bytes_calls():
    variants = vd.generate_controlled_variants()
    data_a, data_b = variants["base"], variants["crop_30pct_border"]

    result = v2.compute_phash_distances(data_a, data_b)

    expected_64 = vd.hamming_distance(
        phash.phash_bytes(data_a, low_freq_size=8), phash.phash_bytes(data_b, low_freq_size=8)
    )
    expected_256 = vd.hamming_distance(
        phash.phash_bytes(data_a, low_freq_size=16), phash.phash_bytes(data_b, low_freq_size=16)
    )
    assert result["phash_distance_64"] == expected_64
    assert result["phash_distance_256"] == expected_256


# ---------------------------------------------------------------------------
# normalized_distance
# ---------------------------------------------------------------------------


def test_normalized_distance_scales_by_bit_width():
    assert v2.normalized_distance(32, 64) == 0.5
    assert v2.normalized_distance(128, 256) == 0.5
    assert v2.normalized_distance(0, 64) == 0.0


# ---------------------------------------------------------------------------
# dhash_recovery: whether a transform would ever enter task 7's candidate
# pool at all
# ---------------------------------------------------------------------------


def test_dhash_recovery_identical_copy_is_recovered():
    variants = vd.generate_controlled_variants()
    result = v2.dhash_recovery(variants["base"], variants["identical_copy"])
    assert result["distance"] == 0
    assert result["recovered"] is True


def test_dhash_recovery_large_crop_is_not_recovered():
    variants = vd.generate_controlled_variants()
    result = v2.dhash_recovery(variants["base"], variants["crop_30pct_border"])
    assert result["recovered"] is False


# ---------------------------------------------------------------------------
# derive_threshold_from_gap
# ---------------------------------------------------------------------------


def test_derive_threshold_from_gap_finds_clean_separation():
    variant_distances = {"identical_copy": [0, 1], "jpeg_recompress": [2, 5], "crop_30pct": [60, 70]}
    hard_pair_distances = [58, 90, 112]

    result = v2.derive_threshold_from_gap(variant_distances, hard_pair_distances)

    assert result["has_clean_gap"] is True
    assert set(result["clean_variants"]) == {"identical_copy", "jpeg_recompress"}
    assert result["non_clean_variants"] == ["crop_30pct"]
    assert result["clean_max"] == 5
    assert result["hard_min"] == 58
    assert result["gap_bits"] == 53
    assert result["threshold"] == (5 + 58) // 2


def test_derive_threshold_from_gap_reports_no_clean_separation_explicitly():
    variant_distances = {"identical_copy": [0, 1], "crop_30pct": [60, 70]}
    hard_pair_distances = [1, 50]  # hard_min=1 is below every variant's max

    result = v2.derive_threshold_from_gap(variant_distances, hard_pair_distances)

    assert result["has_clean_gap"] is False
    assert result["clean_variants"] == []
    assert result["clean_max"] is None
    assert result["gap_bits"] is None
    # Falls back to identical_copy's max + 1 rather than inventing a value.
    assert result["threshold"] == 2


# ---------------------------------------------------------------------------
# build_cross_tab
# ---------------------------------------------------------------------------


def _pair(dhash_distance, phash64=False, phash256=False, strength="not_evaluable"):
    return {
        "dhash_distance": dhash_distance,
        "phash_survivor_64": phash64,
        "phash_survivor_256": phash256,
        "geometric_strength": strength,
    }


def test_build_cross_tab_buckets_by_dhash_distance():
    pairs = [
        _pair(0, strength="forte", phash64=True, phash256=False),
        _pair(2, strength="fraco"),
        _pair(5, strength="not_evaluable"),
        _pair(9, strength="forte", phash64=True, phash256=True),
    ]

    cross_tab = v2.build_cross_tab(pairs)

    assert cross_tab["n_total"] == 4
    assert cross_tab["buckets"]["0-2"]["n"] == 2
    assert cross_tab["buckets"]["0-2"]["n_forte"] == 1
    assert cross_tab["buckets"]["3-7"]["n"] == 1
    assert cross_tab["buckets"]["8-10"]["n"] == 1

    assert cross_tab["n_forte_total"] == 2
    assert cross_tab["n_forte_survives_phash64"] == 2
    assert cross_tab["n_forte_survives_phash256"] == 1
    assert cross_tab["n_forte_rejected_by_phash64"] == 0
    assert cross_tab["n_forte_rejected_by_phash256"] == 1

    assert cross_tab["n_hardpair_total"] == 2
    assert cross_tab["n_hardpair_retained_by_phash64"] == 1
    assert cross_tab["n_hardpair_retained_by_phash256"] == 0


def test_build_cross_tab_ignores_out_of_range_distance():
    pairs = [_pair(-1), _pair(999)]
    cross_tab = v2.build_cross_tab(pairs)
    assert cross_tab["n_total"] == 2
    assert sum(b["n"] for b in cross_tab["buckets"].values()) == 0


# ---------------------------------------------------------------------------
# recommend_configuration
# ---------------------------------------------------------------------------


def test_recommend_configuration_falls_back_to_manual_review_when_no_forte_pair_exists():
    cross_tab = v2.build_cross_tab([_pair(0, strength="fraco"), _pair(9, strength="not_evaluable")])

    recommendation = v2.recommend_configuration(cross_tab)

    assert recommendation["decision"] == "manual_review_fallback"
    assert recommendation["residual_risk"] is not None


def test_recommend_configuration_keeps_64_bit_when_only_64_retains_every_forte_pair():
    pairs = [_pair(0, strength="forte", phash64=True, phash256=False)]
    cross_tab = v2.build_cross_tab(pairs)

    recommendation = v2.recommend_configuration(cross_tab)

    assert recommendation["decision"] == "keep_phash_64"
    assert recommendation["residual_risk"] is None


def test_recommend_configuration_keeps_256_bit_when_only_256_retains_every_forte_pair():
    pairs = [_pair(0, strength="forte", phash64=False, phash256=True)]
    cross_tab = v2.build_cross_tab(pairs)

    recommendation = v2.recommend_configuration(cross_tab)

    assert recommendation["decision"] == "keep_phash_256"


def test_recommend_configuration_prefers_config_that_sheds_more_hardpair_noise_when_both_retain_forte():
    pairs = [
        _pair(0, strength="forte", phash64=True, phash256=True),
        _pair(1, strength="not_evaluable", phash64=False, phash256=True),  # hard-pair noise retained only by 256
    ]
    cross_tab = v2.build_cross_tab(pairs)

    recommendation = v2.recommend_configuration(cross_tab)

    assert recommendation["decision"] == "keep_phash_64"


def test_recommend_configuration_drops_phash_when_neither_width_retains_every_forte_pair():
    pairs = [_pair(0, strength="forte", phash64=False, phash256=False)]
    cross_tab = v2.build_cross_tab(pairs)

    recommendation = v2.recommend_configuration(cross_tab)

    assert recommendation["decision"] == "drop_phash_use_dhash_orb_direct"


# ---------------------------------------------------------------------------
# verify_pair_from_bytes: ORB+RANSAC run directly from raw bytes, with no
# pHash gate anywhere in the call path (the "independent of pHash" part of
# task 9)
# ---------------------------------------------------------------------------


def _textured_array(size: int = 300, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size), dtype=np.uint8)
    image = Image.fromarray(arr, "L")
    draw = ImageDraw.Draw(image)
    for i in range(0, size, 20):
        draw.line((0, i, size, i), fill=0, width=1)
        draw.line((i, 0, i, size), fill=255, width=1)
    return np.array(image)


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, "L").save(buf, format="PNG")
    return buf.getvalue()


def test_verify_pair_from_bytes_finds_strong_support_for_cropped_region_without_any_phash_step():
    full = _textured_array()
    cropped = full[40:240, 60:260]

    outcome = v2.verify_pair_from_bytes(_array_to_png_bytes(full), _array_to_png_bytes(cropped))

    assert outcome["result"]["status"] == "evaluated"
    assert outcome["strength"] == "forte"


def test_verify_pair_from_bytes_does_not_call_any_phash_function(monkeypatch):
    """Guards the "independent of pHash" requirement itself: if someone
    later wires a pHash gate into this function, this test fails loudly."""
    import src.data_audit.phash as phash_module

    def _boom(*args, **kwargs):
        raise AssertionError("verify_pair_from_bytes must not call phash.phash_bytes")

    monkeypatch.setattr(phash_module, "phash_bytes", _boom)

    full = _textured_array()
    cropped = full[40:240, 60:260]
    v2.verify_pair_from_bytes(_array_to_png_bytes(full), _array_to_png_bytes(cropped))


def test_verify_pair_from_bytes_unrelated_images_not_forte():
    image_a = _textured_array(seed=1)
    image_b = _textured_array(seed=2)

    outcome = v2.verify_pair_from_bytes(_array_to_png_bytes(image_a), _array_to_png_bytes(image_b))

    assert outcome["strength"] in ("not_evaluable", "fraco")


# ---------------------------------------------------------------------------
# decode_grayscale_array
# ---------------------------------------------------------------------------


def test_decode_grayscale_array_caps_longer_side():
    big = np.zeros((2000, 1000), dtype=np.uint8)
    data = _array_to_png_bytes(big)

    arr = v2.decode_grayscale_array(data, max_dim=800)

    assert max(arr.shape) <= 800


def test_decode_grayscale_array_leaves_small_image_unscaled():
    small = np.zeros((100, 50), dtype=np.uint8)
    data = _array_to_png_bytes(small)

    arr = v2.decode_grayscale_array(data, max_dim=800)

    assert arr.shape == (100, 50)
