"""ORB keypoint matching + RANSAC homography estimation — the third-stage,
most expensive filter of the task 8 staged pipeline.

Why: dHash and pHash both summarize an entire image into a fixed-size
global fingerprint, so neither can distinguish "these two images are the
same lesion reframed" from "these two images coincidentally summarize to a
similar fingerprint because of shared acquisition style" (task 7's
dHash-only candidates demonstrated exactly that failure mode). Local
keypoint matching checks for a geometrically consistent correspondence
between specific image regions instead of a global summary — two images of
genuinely the same lesion should have many ORB keypoints whose relative
geometry RANSAC can explain with a single homography; two coincidentally
similar but unrelated lesion photos should not.

Detection only: this module scores a candidate pair's geometric support. No
decision about exclusion is made or implied here — a strong score is
evidence for human review, not an automatic confirmation.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# cv2.findHomography needs at least 4 point correspondences to fit a
# homography (8 degrees of freedom, 2 equations per point). Below that it's
# not a "weak geometric match", it's an input the method cannot evaluate at
# all -- reported as its own "not_evaluable" status, never silently scored
# as 0.
MIN_MATCHES_FOR_HOMOGRAPHY = 4

# Lowe's ratio test threshold (Lowe, 2004): a candidate match is kept only
# if the best match is markedly closer than the second-best, which filters
# out ambiguous matches before they ever reach RANSAC. 0.75 is Lowe's own
# commonly used value.
DEFAULT_LOWE_RATIO = 0.75

DEFAULT_RANSAC_REPROJ_THRESHOLD = 5.0

# Classification thresholds for "strong" vs "weak" geometric support among
# pairs that WERE evaluated (status == "evaluated"). Both conditions must
# hold: a minimum absolute inlier count (so a 100%-inlier-ratio match built
# from only 2 matches, an edge case findHomography already requires >=4
# points for, still doesn't count as strong) and a minimum inlier ratio (so
# a pair with hundreds of matches but only a few consistent with any single
# homography doesn't count as strong either). These are a reasonable
# starting point, not validated against a labeled dataset -- the pilot
# report surfaces the actual distribution so the researcher can revise them.
STRONG_MIN_INLIERS = 10
STRONG_MIN_INLIER_RATIO = 0.5


def verify_geometric_match(
    image_a: np.ndarray,
    image_b: np.ndarray,
    lowe_ratio: float = DEFAULT_LOWE_RATIO,
    ransac_reproj_threshold: float = DEFAULT_RANSAC_REPROJ_THRESHOLD,
) -> dict[str, Any]:
    """Detect ORB keypoints in both (grayscale, uint8) images, match them
    with a brute-force Hamming matcher + Lowe's ratio test (cross-check
    would also reject many-to-one ambiguous matches, but the ratio test is
    used here because it is ORB/BFMatcher's standard pairing in the OpenCV
    tutorials this follows, and composes naturally with ``knnMatch(k=2)``),
    and fit a RANSAC homography to the surviving matches.

    Returns a dict always containing ``status`` (one of ``"not_evaluable"``
    — too few keypoints/matches to attempt a homography fit at all — or
    ``"evaluated"`` — a homography fit was attempted; it may still have
    found 0 inliers) plus the raw counts needed to classify the latter into
    "strong"/"weak" support (see ``classify_geometric_support``).
    """
    orb = cv2.ORB_create()
    keypoints_a, descriptors_a = orb.detectAndCompute(image_a, None)
    keypoints_b, descriptors_b = orb.detectAndCompute(image_b, None)
    n_keypoints_a = len(keypoints_a) if keypoints_a is not None else 0
    n_keypoints_b = len(keypoints_b) if keypoints_b is not None else 0

    if descriptors_a is None or descriptors_b is None or n_keypoints_a < 2 or n_keypoints_b < 2:
        return {
            "status": "not_evaluable",
            "reason": (
                f"too few ORB keypoints to attempt matching (a={n_keypoints_a}, "
                f"b={n_keypoints_b}, need >=2 on each side for knnMatch(k=2))"
            ),
            "n_keypoints_a": n_keypoints_a,
            "n_keypoints_b": n_keypoints_b,
            "n_matches": 0,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    # A query descriptor with fewer than 2 candidate matches on the other
    # side (possible when the other side has very few keypoints) can't be
    # ratio-tested; it's simply dropped rather than treated as a match.
    good_matches = [
        pair[0]
        for pair in raw_matches
        if len(pair) == 2 and pair[0].distance < lowe_ratio * pair[1].distance
    ]

    n_matches = len(good_matches)
    if n_matches < MIN_MATCHES_FOR_HOMOGRAPHY:
        return {
            "status": "not_evaluable",
            "reason": (
                f"only {n_matches} good matches after Lowe's ratio test "
                f"(< {MIN_MATCHES_FOR_HOMOGRAPHY} required by cv2.findHomography)"
            ),
            "n_keypoints_a": n_keypoints_a,
            "n_keypoints_b": n_keypoints_b,
            "n_matches": n_matches,
            "n_inliers": 0,
            "inlier_ratio": 0.0,
        }

    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, inlier_mask = cv2.findHomography(
        points_a, points_b, cv2.RANSAC, ransac_reproj_threshold
    )

    if homography is None or inlier_mask is None:
        n_inliers = 0
    else:
        n_inliers = int(inlier_mask.sum())
    inlier_ratio = n_inliers / n_matches

    return {
        "status": "evaluated",
        "n_keypoints_a": n_keypoints_a,
        "n_keypoints_b": n_keypoints_b,
        "n_matches": n_matches,
        "n_inliers": n_inliers,
        "inlier_ratio": inlier_ratio,
        "keypoints_a": keypoints_a,
        "keypoints_b": keypoints_b,
        "good_matches": good_matches,
        "inlier_mask": inlier_mask,
    }


def classify_geometric_support(result: dict[str, Any]) -> str:
    """Map a ``verify_geometric_match`` result to one of three labels:
    ``"not_evaluable"`` (passed through as-is), ``"forte"`` (evaluated, and
    met both :data:`STRONG_MIN_INLIERS` and :data:`STRONG_MIN_INLIER_RATIO`),
    or ``"fraco"`` (evaluated but did not).
    """
    if result["status"] == "not_evaluable":
        return "not_evaluable"
    if result["n_inliers"] >= STRONG_MIN_INLIERS and result["inlier_ratio"] >= STRONG_MIN_INLIER_RATIO:
        return "forte"
    return "fraco"


def draw_match_evidence(
    image_a: np.ndarray,
    image_b: np.ndarray,
    result: dict[str, Any],
    inliers_only: bool = True,
) -> np.ndarray:
    """Render an OpenCV ``drawMatches`` evidence image for an ``"evaluated"``
    result (connecting lines between matched keypoints on both images).

    ``inliers_only`` draws only the RANSAC-inlier matches (the geometrically
    consistent subset) rather than every Lowe-ratio-surviving match, which
    is the more informative picture for a human reviewing evidence of a
    specific homography.
    """
    matches = result["good_matches"]
    if inliers_only and result.get("inlier_mask") is not None:
        mask = result["inlier_mask"].ravel().astype(bool)
        matches = [m for m, keep in zip(matches, mask) if keep]

    return cv2.drawMatches(
        image_a,
        result["keypoints_a"],
        image_b,
        result["keypoints_b"],
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
