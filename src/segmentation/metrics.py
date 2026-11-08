"""Approved segmentation metrics (stage 4, architecture-comparison pilot),
computed PER IMAGE at the ORIGINAL resolution of the reference mask.

Pipeline order enforced by this module's API: the model's probability map
(sigmoid output, continuous) must be restored to the reference mask's
native resolution via :func:`restore_probability_to_original_resolution`
(continuous/bilinear interpolation) BEFORE binarization --
:func:`compute_image_metrics` takes the already-restored probability map
and binarizes internally with an explicit threshold. The reference mask
itself is never resized with continuous interpolation anywhere in this
pipeline (see ``src.segmentation.isic2018_dataset.load_original_mask``,
which reads it at native resolution with no resize at all).

Metrics:
  - Dice per image (primary metric) -- never a pixel-aggregated/global Dice.
  - IoU/Jaccard -- complementary overlap metric.
  - ISIC failure rate -- proportion of images with IoU < 0.65.
  - Thresholded Jaccard -- 0.0 if IoU < 0.65, else the IoU value itself
    (standard ISIC challenge definition).
  - HD95 symmetric -- max of the two directional 95th-percentile contour
    distances (A->B and B->A), in original-resolution pixels, with a
    pixel_spacing multiplier for a physical-distance variant (see
    :func:`hd95_symmetric` for the exact formula and undefined-case rule).
    ``compute_image_metrics`` additionally reports ``hd95_normalizado_diagonal``
    (HD95 in pixels divided by the original mask's diagonal,
    ``sqrt(h**2 + w**2)``) as a complementary, resolution-independent
    variant -- ``None`` whenever ``hd95`` itself is ``None``, never ``0.0``.

Explicit per-image status values returned by :func:`compute_image_metrics`
(never silently merged or dropped):
  - ``"ok"``             -- valid reference, metrics computed (possibly 0.0
                             for Dice/IoU when the prediction is empty).
  - ``"referencia_invalida"`` -- the reference mask has no lesion pixels at
                             all; overlap metrics are undefined for an
                             empty reference, not "a perfect/trivial
                             prediction" case -- all metric fields are
                             ``None``.
  - ``"mascara_incompativel"`` -- prediction and reference shapes differ;
                             no metric is computed.
  - ``"falha_processamento"`` -- an unexpected exception occurred; the
                             exception string is kept, all metric fields
                             are ``None``.
HD95 is ``None`` (undefined), never ``0.0``, whenever either mask has no
contour pixels (empty prediction, or -- in the reference -- already
excluded by the ``"referencia_invalida"`` status above).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

ISIC_FAILURE_IOU_THRESHOLD = 0.65


def dice_per_image(pred_binary: np.ndarray, ref_binary: np.ndarray) -> float:
    """Dice coefficient for a single image: 2*|A∩B| / (|A|+|B|).

    Both-empty convention (both masks have zero foreground pixels): returns
    1.0 (perfect agreement on "no lesion"). This branch is not expected to
    be reached via :func:`compute_image_metrics` (which routes an empty
    reference to the ``"referencia_invalida"`` status before calling this),
    but is defined here so the function is correct standalone.
    """
    pred = np.asarray(pred_binary, dtype=bool)
    ref = np.asarray(ref_binary, dtype=bool)
    intersection = np.logical_and(pred, ref).sum()
    denom = pred.sum() + ref.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def iou_jaccard(pred_binary: np.ndarray, ref_binary: np.ndarray) -> float:
    """IoU/Jaccard for a single image: |A∩B| / |A∪B|.

    Both-empty convention: returns 1.0 (same rationale as :func:`dice_per_image`).
    """
    pred = np.asarray(pred_binary, dtype=bool)
    ref = np.asarray(ref_binary, dtype=bool)
    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def thresholded_jaccard(iou: float) -> float:
    """0.0 if iou < 0.65 (strict), else the iou value itself (iou == 0.65
    preserves the value -- the cutoff is strictly-less-than, not <=)."""
    if iou < ISIC_FAILURE_IOU_THRESHOLD:
        return 0.0
    return float(iou)


def is_isic_failure(iou: float) -> bool:
    """True iff iou < 0.65 (strict)."""
    return iou < ISIC_FAILURE_IOU_THRESHOLD


def _contour(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary mask: mask AND NOT erosion(mask).

    Convention (changed 2026-11-08, researcher decision after a real pilot
    finding -- see ``docs/reconstruction/13_segmentation_deeplabv3plus_pilot.md``
    section 6): ``border_value=0`` -- the exterior of the image is treated
    as BACKGROUND, not as matching the mask's own fill value. Consequences:
      - A fully-True mask now HAS a contour (the frame's border ring,
        since every border pixel has an outside "background" neighbor) --
        it is no longer treated as "no well-defined boundary". This makes
        HD95 computable for a saturated prediction, which previously came
        out ``None`` under the old ``border_value=1`` convention.
      - A fully-False mask still has NO contour under any border_value
        choice (erosion of an all-False array is all-False regardless) --
        this case is unaffected by the change.
    This was changed because a real pilot result showed a model's
    prediction had saturated the whole frame (confirmed independently via
    the ``dice == 2*iou/(1+iou)`` identity, which only holds when
    ``pred ∪ ref`` covers every pixel) and the researcher wants that
    scenario to produce a defined HD95, not a silent ``None``.
    """
    mask_bool = np.asarray(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask_bool, border_value=0)
    return mask_bool & ~eroded


def hd95_symmetric(pred_binary: np.ndarray, ref_binary: np.ndarray, pixel_spacing: float = 1.0) -> float | None:
    """Symmetric 95th-percentile Hausdorff Distance (HD95), the standard
    definition used in the ISIC challenge and general medical image
    segmentation literature:

      1. Extract the contour (boundary) pixels of each binary mask.
      2. For every contour pixel of A, find the distance to the NEAREST
         contour pixel of B (via a Euclidean distance transform of B's
         background) -- the directed distances A->B.
      3. Repeat symmetrically for B->A.
      4. HD95 = max(percentile_95(A->B distances), percentile_95(B->A distances)).

    Returns ``None`` (undefined) if either mask has zero contour pixels
    (e.g. an empty prediction, or a fully-filled mask -- see
    :func:`_contour`) -- NEVER converted to ``0.0``.

    ``pixel_spacing`` is a flat multiplier applied to the final distance
    (e.g. to report a physical-unit variant); pass 1.0 (default) to keep
    the result in original-resolution pixels. The diagonal-normalized
    variant (HD95 divided by the original mask's diagonal) is computed by
    :func:`compute_image_metrics` (``hd95_normalizado_diagonal``), not
    here -- this function only returns the pixel-space value.
    """
    pred = np.asarray(pred_binary, dtype=bool)
    ref = np.asarray(ref_binary, dtype=bool)

    contour_pred = _contour(pred)
    contour_ref = _contour(ref)
    if not contour_pred.any() or not contour_ref.any():
        return None

    dist_to_ref = ndimage.distance_transform_edt(~contour_ref)
    dist_to_pred = ndimage.distance_transform_edt(~contour_pred)

    pred_to_ref = dist_to_ref[contour_pred]
    ref_to_pred = dist_to_pred[contour_ref]

    hd95 = max(np.percentile(pred_to_ref, 95), np.percentile(ref_to_pred, 95))
    return float(hd95 * pixel_spacing)


def _none_result(status: str, erro: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "dice": None,
        "iou": None,
        "thresholded_jaccard": None,
        "hd95": None,
        "hd95_normalizado_diagonal": None,
        "is_isic_failure": None,
        "pred_positive_pixels": None,
        "ref_positive_pixels": None,
    }
    if erro is not None:
        result["erro"] = erro
    return result


def compute_image_metrics(
    pred_prob_original_res: np.ndarray,
    ref_mask_original_res: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Orchestrates the full per-image metric computation. See module
    docstring for the exact status/edge-case rules. ``pred_prob_original_res``
    must already be restored to ``ref_mask_original_res``'s resolution (via
    :func:`restore_probability_to_original_resolution`) BEFORE calling this
    -- binarization with ``threshold`` happens here, strictly after
    restoration, never before.

    In addition to ``hd95`` (pixels), the returned dict always includes
    ``hd95_normalizado_diagonal`` = ``hd95 / sqrt(h**2 + w**2)`` (``h, w``
    from ``ref_mask_original_res.shape``) -- ``None`` whenever ``hd95`` is
    ``None`` (same "never convert undefined to zero" rule), and ``None``
    for every non-``"ok"`` status, same as the other metric fields.

    Also always includes ``pred_positive_pixels``/``ref_positive_pixels``
    (raw positive-pixel counts of the binarized prediction and the
    reference, ``None`` for non-``"ok"`` statuses) -- a diagnostic added
    after a real pilot run whose Dice/IoU satisfied the
    ``dice == 2*iou/(1+iou)`` identity, which only holds when the
    prediction has saturated the whole frame; these counts let any future
    run confirm frame saturation directly, without re-deriving it from
    that identity by hand.
    """
    try:
        pred_prob = np.asarray(pred_prob_original_res)
        ref_mask = np.asarray(ref_mask_original_res).astype(bool)

        if pred_prob.shape != ref_mask.shape:
            return _none_result("mascara_incompativel")

        if not ref_mask.any():
            return _none_result("referencia_invalida")

        pred_binary = pred_prob > threshold

        dice = dice_per_image(pred_binary, ref_mask)
        iou = iou_jaccard(pred_binary, ref_mask)
        tj = thresholded_jaccard(iou)
        failure = is_isic_failure(iou)
        hd95 = hd95_symmetric(pred_binary, ref_mask)
        if hd95 is None:
            hd95_normalizado_diagonal = None
        else:
            h, w = ref_mask.shape
            diagonal = float(np.sqrt(h ** 2 + w ** 2))
            hd95_normalizado_diagonal = hd95 / diagonal

        return {
            "status": "ok",
            "dice": dice,
            "iou": iou,
            "thresholded_jaccard": tj,
            "hd95": hd95,
            "hd95_normalizado_diagonal": hd95_normalizado_diagonal,
            "is_isic_failure": failure,
            "pred_positive_pixels": int(pred_binary.sum()),
            "ref_positive_pixels": int(ref_mask.sum()),
        }
    except Exception as exc:  # noqa: BLE001 - must report, not propagate
        return _none_result("falha_processamento", erro=f"{type(exc).__name__}: {exc}")


def aggregate_metrics(per_image_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates per-image :func:`compute_image_metrics` outputs.

    Counts cases by status. For ``status == "ok"`` cases only: mean Dice,
    mean IoU, ISIC failure rate, mean Thresholded Jaccard, and for HD95 --
    the median of the FINITE (non-None) values plus a SEPARATE count of how
    many are undefined (None). Undefined HD95 values never enter the
    median and are never treated as zero. The same undefined-count applies
    to ``hd95_normalizado_diagonal`` (an undefined ``hd95`` always implies
    an undefined ``hd95_normalizado_diagonal`` -- see
    :func:`compute_image_metrics`), whose finite-value median is reported
    separately as ``hd95_normalizado_diagonal_median_finite``.
    """
    counts_by_status: dict[str, int] = {}
    for r in per_image_results:
        counts_by_status[r["status"]] = counts_by_status.get(r["status"], 0) + 1

    ok_results = [r for r in per_image_results if r["status"] == "ok"]

    if ok_results:
        mean_dice = float(np.mean([r["dice"] for r in ok_results]))
        mean_iou = float(np.mean([r["iou"] for r in ok_results]))
        isic_failure_rate = float(np.mean([r["is_isic_failure"] for r in ok_results]))
        mean_thresholded_jaccard = float(np.mean([r["thresholded_jaccard"] for r in ok_results]))
        hd95_finite = [r["hd95"] for r in ok_results if r["hd95"] is not None]
        hd95_n_undefined = sum(1 for r in ok_results if r["hd95"] is None)
        hd95_median_finite = float(np.median(hd95_finite)) if hd95_finite else None
        # Same undefined count reused: an undefined hd95 implies an undefined
        # hd95_normalizado_diagonal (see compute_image_metrics), so the two
        # finite-value sets have identical membership.
        hd95_norm_diag_finite = [
            r["hd95_normalizado_diagonal"] for r in ok_results if r["hd95_normalizado_diagonal"] is not None
        ]
        hd95_norm_diag_median_finite = (
            float(np.median(hd95_norm_diag_finite)) if hd95_norm_diag_finite else None
        )
    else:
        mean_dice = mean_iou = isic_failure_rate = mean_thresholded_jaccard = None
        hd95_median_finite = None
        hd95_norm_diag_median_finite = None
        hd95_n_undefined = 0

    return {
        "n_total": len(per_image_results),
        "counts_by_status": counts_by_status,
        "n_ok": len(ok_results),
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "isic_failure_rate": isic_failure_rate,
        "mean_thresholded_jaccard": mean_thresholded_jaccard,
        "hd95_median_finite": hd95_median_finite,
        "hd95_normalizado_diagonal_median_finite": hd95_norm_diag_median_finite,
        "hd95_n_undefined": hd95_n_undefined,
    }


def restore_probability_to_original_resolution(
    prob_map: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    """Restores a continuous probability map (e.g. ``TARGET_SIZE``x``TARGET_SIZE``
    sigmoid output) to ``target_shape`` (the reference mask's native
    resolution) using CONTINUOUS (bilinear) interpolation -- never nearest,
    never applied to an already-binarized mask. Binarization must happen
    strictly AFTER this call (see :func:`compute_image_metrics`).
    """
    import torch
    import torch.nn.functional as F

    tensor = torch.as_tensor(np.asarray(prob_map), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=tuple(target_shape), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()
