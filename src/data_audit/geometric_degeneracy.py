"""Degenerate-homography guard for ORB+RANSAC geometric support (task 10).

Why: task 9's manual review of the 3 pairs classified "forte" by
``geometric_verification.classify_geometric_support`` found that none were
genuine duplicates. In all three, the RANSAC-inlier correspondences on one
side of the pair collapsed onto a single point or a tiny cluster instead of
spreading across the image -- a degenerate homography fit (near-collinear or
near-coincident points admit many homographies with low reprojection error,
inflating the inlier count without any real structural correspondence).
``classify_geometric_support``'s existing ``STRONG_MIN_INLIERS``/
``STRONG_MIN_INLIER_RATIO`` thresholds cannot catch this because they only
look at inlier *count*, never inlier *spatial spread*.

This module adds that missing spatial check as an explicit, separately
labeled guard layered on top of (never silently replacing) the existing
classifier -- see ``classify_with_degeneracy_guard``.

Detection/diagnostic only: no exclusion decision for any lesion/image is
made or implied here.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.data_audit import geometric_verification as gv


def extract_inlier_points(result: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """RANSAC-inlier keypoint coordinates on each side of an ``"evaluated"``
    ``verify_geometric_match`` result, as ``(n_inliers, 2)`` float32 arrays.

    Returns ``(None, None)`` for a non-``"evaluated"`` result or one without
    an inlier mask (``findHomography`` failed), so callers can tell "no
    points to measure" apart from "measured and found zero spread".
    """
    if result.get("status") != "evaluated" or result.get("inlier_mask") is None:
        return None, None
    mask = result["inlier_mask"].ravel().astype(bool)
    matches = result["good_matches"]
    inlier_matches = [m for m, keep in zip(matches, mask) if keep]
    if not inlier_matches:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    points_a = np.float32([result["keypoints_a"][m.queryIdx].pt for m in inlier_matches])
    points_b = np.float32([result["keypoints_b"][m.trainIdx].pt for m in inlier_matches])
    return points_a, points_b


def convex_hull_normalized_area(points: np.ndarray | None, image_shape: tuple[int, int]) -> float:
    """Convex-hull area of ``points`` (pixel coordinates), divided by the
    area of the image they were found in.

    Why convex hull area, specifically: it is a single scalar that is near 0
    both when points cluster tightly around one location AND when points lie
    close to a single line -- the two degenerate patterns task 9's manual
    review found across its 3 known false "forte" pairs (points converging
    to "a single point or a very small cluster" in some, near-collinear
    match lines in others). A pure bounding-box or standard-deviation metric
    would catch clustering but not collinearity (a thin, long line of points
    can still have a large bounding box and non-trivial per-axis std while
    spanning essentially zero 2-D area), so convex hull area is used instead
    of either.

    Fewer than 3 points cannot bound any 2-D area at all (a hull needs at
    least 3 non-collinear points) -- always reported as ``0.0``, the
    maximally degenerate value, rather than undefined/NaN.
    """
    height, width = image_shape[:2]
    image_area = height * width
    if points is None or len(points) < 3 or image_area <= 0:
        return 0.0
    hull = cv2.convexHull(points)
    return float(cv2.contourArea(hull)) / image_area


def compute_dispersion(
    result: dict[str, Any], shape_a: tuple[int, int], shape_b: tuple[int, int]
) -> dict[str, float]:
    """Normalized convex-hull dispersion of ``result``'s RANSAC inliers on
    each side, independently -- side A and side B need not share an image
    size, so each side is normalized by its own image's area.
    """
    points_a, points_b = extract_inlier_points(result)
    return {
        "dispersion_a": convex_hull_normalized_area(points_a, shape_a),
        "dispersion_b": convex_hull_normalized_area(points_b, shape_b),
    }


def is_degenerate(dispersion_a: float, dispersion_b: float, threshold: float) -> bool:
    """A configuration is degenerate when the inliers on EITHER side fail to
    spread across at least ``threshold`` fraction of that side's image area.

    Either side collapsing is sufficient to flag the pair: a genuine
    structural correspondence requires both sides to show real, spread-out
    matched structure, not just one -- task 9's 3 known cases all had the
    collapse on only one side (the HAM10000 side in each), with the other
    side's points comparatively more spread, so checking both sides
    independently (rather than, say, their average) is what actually catches
    those cases.
    """
    return dispersion_a < threshold or dispersion_b < threshold


def classify_with_degeneracy_guard(
    result: dict[str, Any],
    shape_a: tuple[int, int],
    shape_b: tuple[int, int],
    dispersion_threshold: float,
    base_strength: str | None = None,
) -> dict[str, Any]:
    """Wrap ``geometric_verification.classify_geometric_support`` with the
    degeneracy guard.

    A pair classified ``"forte"`` whose inlier spread collapses below
    ``dispersion_threshold`` on either side is relabeled ``"forte_degenerada"``
    -- a distinct, explicit label, never a silent downgrade to plain
    ``"fraco"`` with no record of why. ``"fraco"``/``"not_evaluable"`` pairs
    pass through unchanged (the guard only ever acts on ``"forte"``, which is
    the label this task exists to correct), though dispersion is still
    computed and reported for any ``"evaluated"`` result for diagnostic
    visibility, regardless of its base label.

    ``base_strength``, if given, is the already-computed
    ``classify_geometric_support(result)`` (so a caller that computed it for
    its own purposes does not pay for it twice); otherwise it is computed
    here.
    """
    strength = base_strength if base_strength is not None else gv.classify_geometric_support(result)
    dispersion = compute_dispersion(result, shape_a, shape_b)
    degenerate = (
        is_degenerate(dispersion["dispersion_a"], dispersion["dispersion_b"], dispersion_threshold)
        if result.get("status") == "evaluated"
        else False
    )
    final_strength = "forte_degenerada" if strength == "forte" and degenerate else strength
    return {
        "strength": strength,
        "final_strength": final_strength,
        "is_degenerate": degenerate,
        "dispersion_a": dispersion["dispersion_a"],
        "dispersion_b": dispersion["dispersion_b"],
        "dispersion_threshold": dispersion_threshold,
    }


def pair_dispersion_measure(dispersion_a: float, dispersion_b: float) -> float:
    """The scalar used to fit/evaluate a threshold against a whole pair:
    ``min(dispersion_a, dispersion_b)``, matching the "either side collapsing
    is enough" rule in :func:`is_degenerate` -- a pair is only as
    non-degenerate as its worse side.
    """
    return min(dispersion_a, dispersion_b)


def derive_dispersion_threshold(
    positive_pair_dispersions: list[float], negative_pair_dispersions: list[float]
) -> dict[str, Any]:
    """Derive a dispersion threshold from the adjustment set only, using the
    same measured-gap convention as
    ``visual_duplicate_pilot_v2.derive_threshold_from_gap``: if every
    adjustment-set negative's dispersion is strictly below every
    adjustment-set positive's dispersion, the threshold is the midpoint of
    that gap and the gap size is reported; otherwise ``has_clean_gap=False``
    is reported explicitly, with the negatives' max as a non-ideal fallback,
    instead of inventing a threshold the data does not support.

    Each input is expected to already be the per-pair scalar from
    :func:`pair_dispersion_measure` (i.e. ``min(dispersion_a, dispersion_b)``
    for that pair), one value per adjustment-set pair.
    """
    neg_max = max(negative_pair_dispersions)
    pos_min = min(positive_pair_dispersions)
    has_clean_gap = neg_max < pos_min
    if has_clean_gap:
        threshold = (neg_max + pos_min) / 2
        gap = pos_min - neg_max
    else:
        threshold = neg_max
        gap = None
    return {
        "neg_max": neg_max,
        "pos_min": pos_min,
        "has_clean_gap": has_clean_gap,
        "gap": gap,
        "threshold": threshold,
    }
