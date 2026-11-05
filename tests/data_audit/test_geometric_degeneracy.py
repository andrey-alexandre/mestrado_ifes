"""Tests for src.data_audit.geometric_degeneracy (task 10).

Uses only synthetic point/keypoint fixtures built in-process (no decoded
images, no patient data) -- matching the convention of
tests/data_audit/test_geometric_verification.py.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.data_audit import geometric_degeneracy as gd

IMAGE_SHAPE = (200, 200)  # (height, width)


def _make_result(points_a: list[tuple[float, float]], points_b: list[tuple[float, float]]) -> dict:
    """A minimal ``verify_geometric_match``-shaped "evaluated" result whose
    RANSAC inliers are exactly ``points_a``/``points_b`` (every match is an
    inlier) -- enough structure for extract_inlier_points/compute_dispersion
    without running ORB on a real image.
    """
    assert len(points_a) == len(points_b)
    keypoints_a = [cv2.KeyPoint(x, y, 1) for x, y in points_a]
    keypoints_b = [cv2.KeyPoint(x, y, 1) for x, y in points_b]
    good_matches = [cv2.DMatch(i, i, 0.0) for i in range(len(points_a))]
    inlier_mask = np.ones((len(points_a), 1), dtype=np.uint8)
    return {
        "status": "evaluated",
        "n_matches": len(points_a),
        "n_inliers": len(points_a),
        "inlier_ratio": 1.0,
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "good_matches": good_matches,
        "inlier_mask": inlier_mask,
    }


# A grid of points spanning most of a 200x200 image: high dispersion, the
# genuine-structural-match pattern.
SPREAD_POINTS = [(x, y) for x in (10, 100, 190) for y in (10, 100, 190)]

# All points packed into a tiny corner cluster: the "single point/small
# cluster" degenerate pattern task 9 observed.
CLUSTERED_POINTS = [(10 + dx, 10 + dy) for dx in (0, 1, 2) for dy in (0, 1, 2)]

# Points lying on a single line: the "near-collinear" degenerate pattern.
COLLINEAR_POINTS = [(x, x) for x in range(10, 190, 20)]


def test_convex_hull_normalized_area_spread_points_is_high():
    dispersion = gd.convex_hull_normalized_area(np.float32(SPREAD_POINTS), IMAGE_SHAPE)
    assert dispersion > 0.5


def test_convex_hull_normalized_area_clustered_points_is_near_zero():
    dispersion = gd.convex_hull_normalized_area(np.float32(CLUSTERED_POINTS), IMAGE_SHAPE)
    assert dispersion < 0.01


def test_convex_hull_normalized_area_collinear_points_is_zero():
    dispersion = gd.convex_hull_normalized_area(np.float32(COLLINEAR_POINTS), IMAGE_SHAPE)
    assert dispersion == 0.0


def test_convex_hull_normalized_area_fewer_than_3_points_is_zero():
    assert gd.convex_hull_normalized_area(np.float32([(10, 10), (20, 20)]), IMAGE_SHAPE) == 0.0
    assert gd.convex_hull_normalized_area(np.float32([]).reshape(0, 2), IMAGE_SHAPE) == 0.0
    assert gd.convex_hull_normalized_area(None, IMAGE_SHAPE) == 0.0


def test_extract_inlier_points_returns_none_for_not_evaluable():
    points_a, points_b = gd.extract_inlier_points({"status": "not_evaluable"})
    assert points_a is None
    assert points_b is None


def test_compute_dispersion_matches_direct_hull_computation():
    result = _make_result(SPREAD_POINTS, CLUSTERED_POINTS)
    dispersion = gd.compute_dispersion(result, IMAGE_SHAPE, IMAGE_SHAPE)
    assert dispersion["dispersion_a"] > 0.5
    assert dispersion["dispersion_b"] < 0.01


def test_is_degenerate_either_side_below_threshold_triggers():
    assert gd.is_degenerate(dispersion_a=0.01, dispersion_b=0.9, threshold=0.1) is True
    assert gd.is_degenerate(dispersion_a=0.9, dispersion_b=0.01, threshold=0.1) is True
    assert gd.is_degenerate(dispersion_a=0.9, dispersion_b=0.9, threshold=0.1) is False


def test_classify_with_degeneracy_guard_relabels_degenerate_forte():
    result = _make_result(CLUSTERED_POINTS, SPREAD_POINTS)
    classification = gd.classify_with_degeneracy_guard(
        result, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=0.1, base_strength="forte"
    )
    assert classification["strength"] == "forte"
    assert classification["is_degenerate"] is True
    assert classification["final_strength"] == "forte_degenerada"


def test_classify_with_degeneracy_guard_leaves_genuine_forte_unchanged():
    result = _make_result(SPREAD_POINTS, SPREAD_POINTS)
    classification = gd.classify_with_degeneracy_guard(
        result, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=0.1, base_strength="forte"
    )
    assert classification["is_degenerate"] is False
    assert classification["final_strength"] == "forte"


def test_classify_with_degeneracy_guard_leaves_fraco_unchanged_even_if_degenerate():
    result = _make_result(CLUSTERED_POINTS, CLUSTERED_POINTS)
    classification = gd.classify_with_degeneracy_guard(
        result, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=0.1, base_strength="fraco"
    )
    # The guard only ever relabels "forte" -- a "fraco" pair is reported as
    # degenerate (diagnostic visibility) but its label is not touched.
    assert classification["is_degenerate"] is True
    assert classification["final_strength"] == "fraco"


def test_classify_with_degeneracy_guard_not_evaluable_passthrough():
    classification = gd.classify_with_degeneracy_guard(
        {"status": "not_evaluable"}, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=0.1
    )
    assert classification["strength"] == "not_evaluable"
    assert classification["final_strength"] == "not_evaluable"
    assert classification["is_degenerate"] is False


def test_classify_with_degeneracy_guard_computes_base_strength_when_not_given():
    result = _make_result(CLUSTERED_POINTS, SPREAD_POINTS)
    # n_inliers=9 < STRONG_MIN_INLIERS=10 -> base classifier alone says "fraco".
    assert len(CLUSTERED_POINTS) < 10
    classification = gd.classify_with_degeneracy_guard(result, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=0.1)
    assert classification["strength"] == "fraco"
    assert classification["final_strength"] == "fraco"


def test_pair_dispersion_measure_is_the_min_of_both_sides():
    assert gd.pair_dispersion_measure(0.2, 0.8) == 0.2
    assert gd.pair_dispersion_measure(0.8, 0.2) == 0.2


def test_derive_dispersion_threshold_clean_gap():
    result = gd.derive_dispersion_threshold(
        positive_pair_dispersions=[0.6, 0.7, 0.8], negative_pair_dispersions=[0.01, 0.02, 0.03]
    )
    assert result["has_clean_gap"] is True
    assert result["neg_max"] == 0.03
    assert result["pos_min"] == 0.6
    assert result["threshold"] == (0.03 + 0.6) / 2
    assert result["gap"] == 0.6 - 0.03


def test_derive_dispersion_threshold_no_clean_gap():
    result = gd.derive_dispersion_threshold(
        positive_pair_dispersions=[0.05, 0.6], negative_pair_dispersions=[0.01, 0.2]
    )
    assert result["has_clean_gap"] is False
    assert result["gap"] is None
    assert result["threshold"] == 0.2


def test_full_adjustment_validation_flow_on_synthetic_dataset():
    """End-to-end: build a small synthetic adjustment set (degenerate
    "negative" pairs + genuine-structural "positive" pairs), derive a
    threshold from it alone, then apply that threshold to a disjoint
    synthetic validation set and confirm it separates correctly there too.
    No image decoding, no patient data -- every point set here is
    hand-constructed.
    """
    # Adjustment set.
    adj_negatives = [
        _make_result(CLUSTERED_POINTS, SPREAD_POINTS),  # side A collapses
        _make_result(COLLINEAR_POINTS, SPREAD_POINTS),  # side A collinear
    ]
    adj_positives = [
        _make_result(SPREAD_POINTS, SPREAD_POINTS),
        _make_result(SPREAD_POINTS, [(x + 5, y + 5) for x, y in SPREAD_POINTS]),
    ]

    def pair_measure(result):
        d = gd.compute_dispersion(result, IMAGE_SHAPE, IMAGE_SHAPE)
        return gd.pair_dispersion_measure(d["dispersion_a"], d["dispersion_b"])

    derivation = gd.derive_dispersion_threshold(
        positive_pair_dispersions=[pair_measure(r) for r in adj_positives],
        negative_pair_dispersions=[pair_measure(r) for r in adj_negatives],
    )
    assert derivation["has_clean_gap"] is True
    threshold = derivation["threshold"]

    # Validation set: disjoint synthetic pairs, never used to pick the threshold.
    val_negative_points_a = [(50 + (i % 3), 50 + (i // 3)) for i in range(len(SPREAD_POINTS))]
    val_negative = _make_result(val_negative_points_a, SPREAD_POINTS)
    val_positive = _make_result(
        [(x - 3, y - 3) for x, y in SPREAD_POINTS], [(x + 3, y + 3) for x, y in SPREAD_POINTS]
    )

    neg_classification = gd.classify_with_degeneracy_guard(
        val_negative, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=threshold, base_strength="forte"
    )
    pos_classification = gd.classify_with_degeneracy_guard(
        val_positive, IMAGE_SHAPE, IMAGE_SHAPE, dispersion_threshold=threshold, base_strength="forte"
    )

    assert neg_classification["final_strength"] == "forte_degenerada"
    assert pos_classification["final_strength"] == "forte"
