"""Tests for src.data_audit.geometric_verification (task 8).

Uses only synthetic, procedurally generated images (a textured checkerboard-
like pattern and a flat/textureless image) -- no patient data in the
automated test suite, matching the convention of
tests/data_audit/test_visual_duplicates.py and test_phash.py.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from src.data_audit import geometric_verification as gv


def _make_textured_image(size: int = 300, seed: int = 0) -> np.ndarray:
    """A grayscale image with enough distinct local structure (random noise
    plus a grid of lines) for ORB to find many keypoints."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size), dtype=np.uint8)
    image = Image.fromarray(arr, "L")
    draw = ImageDraw.Draw(image)
    for i in range(0, size, 20):
        draw.line((0, i, size, i), fill=0, width=1)
        draw.line((i, 0, i, size), fill=255, width=1)
    return np.array(image)


def test_translated_crop_of_same_image_has_strong_geometric_support():
    full = _make_textured_image()
    # A sub-region of the same image: genuinely overlapping content, offset
    # (translated) from the full image -- analogous to the same lesion
    # reframed/cropped across dataset releases.
    cropped = full[40:240, 60:260]

    result = gv.verify_geometric_match(full, cropped)

    assert result["status"] == "evaluated"
    assert result["n_matches"] >= gv.MIN_MATCHES_FOR_HOMOGRAPHY
    assert result["n_inliers"] > 0
    assert result["inlier_ratio"] > 0.5
    assert gv.classify_geometric_support(result) == "forte"


def test_unrelated_random_images_do_not_get_strong_support():
    image_a = _make_textured_image(seed=1)
    image_b = _make_textured_image(seed=2)

    result = gv.verify_geometric_match(image_a, image_b)

    # Either not enough consistent matches to evaluate at all, or evaluated
    # but without strong support -- either way, must not be classified "forte".
    assert gv.classify_geometric_support(result) in ("not_evaluable", "fraco")


def test_textureless_image_is_not_evaluable_not_zero_score():
    flat_a = np.full((100, 100), 128, dtype=np.uint8)
    flat_b = np.full((100, 100), 128, dtype=np.uint8)

    result = gv.verify_geometric_match(flat_a, flat_b)

    assert result["status"] == "not_evaluable"
    assert result["n_keypoints_a"] == 0
    assert result["n_keypoints_b"] == 0
    assert gv.classify_geometric_support(result) == "not_evaluable"


def test_few_keypoints_on_one_side_is_not_evaluable():
    textured = _make_textured_image()
    flat = np.full(textured.shape, 128, dtype=np.uint8)

    result = gv.verify_geometric_match(textured, flat)

    assert result["status"] == "not_evaluable"
    assert result["n_keypoints_b"] == 0


def test_classify_geometric_support_requires_both_min_inliers_and_ratio():
    strong = {"status": "evaluated", "n_inliers": gv.STRONG_MIN_INLIERS, "inlier_ratio": gv.STRONG_MIN_INLIER_RATIO}
    assert gv.classify_geometric_support(strong) == "forte"

    too_few_inliers = {"status": "evaluated", "n_inliers": gv.STRONG_MIN_INLIERS - 1, "inlier_ratio": 1.0}
    assert gv.classify_geometric_support(too_few_inliers) == "fraco"

    too_low_ratio = {"status": "evaluated", "n_inliers": 1000, "inlier_ratio": gv.STRONG_MIN_INLIER_RATIO - 0.01}
    assert gv.classify_geometric_support(too_low_ratio) == "fraco"


def test_draw_match_evidence_returns_an_image_for_evaluated_result():
    full = _make_textured_image()
    cropped = full[40:240, 60:260]
    result = gv.verify_geometric_match(full, cropped)

    evidence = gv.draw_match_evidence(full, cropped, result)

    assert evidence is not None
    assert evidence.shape[0] > 0 and evidence.shape[1] > 0
