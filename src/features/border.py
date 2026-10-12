"""ABCD B border extraction using the frozen Kasmi and Mokrani path."""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline

from src.features.geometry import principal_axis_analysis
from src.features.schemas import EvidenceEnvelope, EvidenceStatus
from src.features.utils import boundary_mask


SOURCE = "Kasmi and Mokrani 2016 ABCD border operationalization"
OCTANT_COUNT = 8
OCTANT_WIDTH_RADIANS = 2.0 * math.pi / OCTANT_COUNT
IRREGULARITY_THRESHOLD_FACTOR = 0.05
DEFAULT_SPLINE_KIND = "smoothing"
DEFAULT_SMOOTHING_FACTOR = 1.0
DEFAULT_REPRESENTATION = "parametric"
DEFAULT_PARAMETRIC_SMOOTHING_FORMULATION = "dimensionless_chord"
SECTOR_ORIENTATION_IMAGE_FIXED = "IMAGE_FIXED"
SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED = "PRINCIPAL_AXIS_ALIGNED"
DEFAULT_SECTOR_ORIENTATION = SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED
MIN_CUBIC_POINTS = 4
CHORD_TOLERANCE = 1e-9


@dataclass(frozen=True)
class SectorSplineResult:
    sector_index: int
    boundary_point_count: int
    rmse: Optional[float]
    threshold: float
    irregular: bool
    insufficient_points: bool
    run_count: int = 1
    unique_angle_count: Optional[int] = None
    arc_length_total: Optional[float] = None
    smooth_arc_length_total: Optional[float] = None
    mean_residual: Optional[float] = None
    max_residual: Optional[float] = None
    spline_smoothing_parameter: Optional[float] = None
    equivalent_dimensional_smoothing_parameter: Optional[float] = None
    local_chord_length: Optional[float] = None
    degenerate_chord: bool = False
    chord_fallback: Optional[str] = None
    theta: Optional[np.ndarray] = None
    normalized_u: Optional[np.ndarray] = None
    normalized_xy: Optional[np.ndarray] = None
    observed_radius: Optional[np.ndarray] = None
    fitted_radius: Optional[np.ndarray] = None
    contour_xy: Optional[np.ndarray] = None
    arc_length: Optional[np.ndarray] = None
    fitted_xy: Optional[np.ndarray] = None


def lesion_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("No lesion pixels available.")
    return float(xs.mean()), float(ys.mean())


def extract_boundary_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(boundary_mask(mask.astype(bool)))
    return np.column_stack([ys.astype(np.float64), xs.astype(np.float64)])


def extract_ordered_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external marching-squares contour as ordered xy points."""
    segments = _marching_squares_segments(mask.astype(bool))
    loops = _stitch_segments_into_loops(segments)
    if not loops:
        points = extract_boundary_points(mask)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        return points[:, ::-1]
    return max(loops, key=_polyline_length)


def normalize_angles(angles: np.ndarray) -> np.ndarray:
    return np.mod(angles, 2.0 * math.pi)


def angles_and_radii(points_yx: np.ndarray, centroid_xy: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    cx, cy = centroid_xy
    dy = points_yx[:, 0] - cy
    dx = points_yx[:, 1] - cx
    angles = normalize_angles(np.arctan2(dy, dx))
    radii = np.hypot(dx, dy)
    return angles, radii


def angles_for_xy(points_xy: np.ndarray, centroid_xy: Tuple[float, float]) -> np.ndarray:
    cx, cy = centroid_xy
    return normalize_angles(np.arctan2(points_xy[:, 1] - cy, points_xy[:, 0] - cx))


def assign_octants(angles: np.ndarray) -> np.ndarray:
    sectors = np.floor(normalize_angles(angles) / OCTANT_WIDTH_RADIANS).astype(int)
    return np.clip(sectors, 0, OCTANT_COUNT - 1)


def assign_octants_relative(angles: np.ndarray, orientation_radians: float = 0.0) -> np.ndarray:
    return assign_octants(angles - orientation_radians)


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    if observed.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def euclidean_rmse(observed_xy: np.ndarray, fitted_xy: np.ndarray) -> float:
    if observed_xy.size == 0:
        return 0.0
    deltas = observed_xy - fitted_xy
    return float(np.sqrt(np.mean(np.sum(deltas * deltas, axis=1))))


def cumulative_arc_length(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.empty((0,), dtype=np.float64)
    deltas = np.diff(points_xy, axis=0)
    distances = np.hypot(deltas[:, 0], deltas[:, 1])
    return np.concatenate([[0.0], np.cumsum(distances)])


def open_polyline_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    deltas = np.diff(points_xy, axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def chord_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    return float(np.hypot(*(points_xy[-1] - points_xy[0])))


def normalized_parametric_coordinates(
    points_xy: np.ndarray,
    chord_tolerance: float = CHORD_TOLERANCE,
) -> Dict:
    s = cumulative_arc_length(points_xy)
    if s.size == 0 or s[-1] <= 0:
        raise ValueError("Arc length must be positive for parametric normalization.")
    d_chord = chord_length(points_xy)
    degenerate = d_chord <= chord_tolerance
    if degenerate:
        d_scale = float(s[-1])
        fallback = "arc_length"
    else:
        d_scale = d_chord
        fallback = None
    if d_scale <= chord_tolerance:
        raise ValueError("No positive local scale available for parametric normalization.")

    origin = points_xy[0]
    return {
        "s": s,
        "u": s / s[-1],
        "normalized_xy": (points_xy - origin) / d_scale,
        "origin_xy": origin,
        "local_scale": d_scale,
        "chord_length": d_chord,
        "degenerate_chord": degenerate,
        "chord_fallback": fallback,
    }


def sector_threshold(boundary_point_count: int) -> float:
    return float(IRREGULARITY_THRESHOLD_FACTOR * boundary_point_count)


def is_irregular_sector(error: Optional[float], boundary_point_count: int) -> bool:
    return bool(error is not None and error > sector_threshold(boundary_point_count))


def is_irregular_by_smoothed_length(error: Optional[float], smooth_arc_length: Optional[float]) -> bool:
    if error is None or smooth_arc_length is None or smooth_arc_length <= 0:
        return False
    return bool((error / smooth_arc_length) > IRREGULARITY_THRESHOLD_FACTOR)


def _marching_squares_segments(mask: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant", constant_values=0)
    table = {
        1: [("left", "bottom")],
        2: [("bottom", "right")],
        3: [("left", "right")],
        4: [("top", "right")],
        5: [("top", "left"), ("bottom", "right")],
        6: [("top", "bottom")],
        7: [("left", "top")],
        8: [("top", "left")],
        9: [("top", "bottom")],
        10: [("top", "right"), ("bottom", "left")],
        11: [("top", "right")],
        12: [("left", "right")],
        13: [("bottom", "right")],
        14: [("left", "bottom")],
    }
    segments = []
    for y in range(padded.shape[0] - 1):
        for x in range(padded.shape[1] - 1):
            case = (
                int(padded[y, x]) * 8
                + int(padded[y, x + 1]) * 4
                + int(padded[y + 1, x + 1]) * 2
                + int(padded[y + 1, x]) * 1
            )
            for edge_a, edge_b in table.get(case, []):
                segments.append((_edge_midpoint(x, y, edge_a), _edge_midpoint(x, y, edge_b)))
    return segments


def _edge_midpoint(x: int, y: int, edge: str) -> Tuple[float, float]:
    ox = float(x - 1)
    oy = float(y - 1)
    if edge == "top":
        return ox + 0.5, oy
    if edge == "right":
        return ox + 1.0, oy + 0.5
    if edge == "bottom":
        return ox + 0.5, oy + 1.0
    if edge == "left":
        return ox, oy + 0.5
    raise ValueError(edge)


def _stitch_segments_into_loops(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[np.ndarray]:
    adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    unused = set()
    for a, b in segments:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
        unused.add(frozenset((a, b)))

    loops = []
    while unused:
        edge = unused.pop()
        a, b = tuple(edge)
        points = [a, b]
        previous, current = a, b
        while current != points[0]:
            candidates = [
                point for point in adjacency[current]
                if point != previous and frozenset((current, point)) in unused
            ]
            if not candidates:
                break
            nxt = candidates[0]
            unused.remove(frozenset((current, nxt)))
            points.append(nxt)
            previous, current = current, nxt
        if len(points) >= MIN_CUBIC_POINTS:
            if points[-1] == points[0]:
                points = points[:-1]
            loops.append(np.asarray(points, dtype=np.float64))
    return loops


def _polyline_length(points_xy: np.ndarray) -> float:
    if points_xy.shape[0] < 2:
        return 0.0
    closed = np.vstack([points_xy, points_xy[0]])
    deltas = np.diff(closed, axis=0)
    return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


def _mean_radius_by_angle(theta: np.ndarray, radius: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_theta, inverse = np.unique(theta, return_inverse=True)
    summed = np.zeros(unique_theta.shape, dtype=np.float64)
    counts = np.zeros(unique_theta.shape, dtype=np.float64)
    np.add.at(summed, inverse, radius)
    np.add.at(counts, inverse, 1.0)
    return unique_theta, summed / counts


def _radial_spline_predictions(
    theta: np.ndarray,
    radius: np.ndarray,
    spline_kind: str,
    smoothing_factor: float,
) -> Tuple[np.ndarray, int, Optional[float]]:
    order = np.argsort(theta)
    sorted_theta = theta[order]
    sorted_radius = radius[order]
    unique_theta, mean_radius = _mean_radius_by_angle(sorted_theta, sorted_radius)
    if unique_theta.size < MIN_CUBIC_POINTS:
        raise ValueError("At least four unique angular samples are required for a cubic spline.")

    if spline_kind == "interpolating":
        spline = CubicSpline(unique_theta, mean_radius, bc_type="natural")
        smoothing = None
    elif spline_kind == "smoothing":
        radial_span = max(float(np.ptp(mean_radius)), 1.0)
        smoothing = float(smoothing_factor * unique_theta.size * radial_span * radial_span)
        spline = UnivariateSpline(unique_theta, mean_radius, k=3, s=smoothing)
    else:
        raise ValueError("spline_kind must be 'smoothing' or 'interpolating'.")

    fitted = np.empty_like(radius, dtype=np.float64)
    fitted[order] = spline(sorted_theta)
    return fitted, int(unique_theta.size), smoothing


def _parametric_spline_predictions(
    points_xy: np.ndarray,
    spline_kind: str,
    smoothing_factor: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict]:
    s = cumulative_arc_length(points_xy)
    unique_s, unique_indices = np.unique(s, return_index=True)
    unique_points = points_xy[unique_indices]
    if unique_s.size < MIN_CUBIC_POINTS or unique_s[-1] <= 0:
        raise ValueError("At least four unique arc-length samples are required for a cubic spline.")

    if spline_kind == "interpolating":
        spline_x = CubicSpline(unique_s, unique_points[:, 0], bc_type="natural")
        spline_y = CubicSpline(unique_s, unique_points[:, 1], bc_type="natural")
        smoothing = None
    elif spline_kind == "smoothing":
        span = max(_polyline_length(unique_points), 1.0)
        smoothing = float(smoothing_factor * unique_s.size * span)
        spline_x = UnivariateSpline(unique_s, unique_points[:, 0], k=3, s=smoothing)
        spline_y = UnivariateSpline(unique_s, unique_points[:, 1], k=3, s=smoothing)
    else:
        raise ValueError("spline_kind must be 'smoothing' or 'interpolating'.")

    fitted = np.column_stack([spline_x(s), spline_y(s)])
    return fitted, s, smoothing, {
        "formulation": "legacy_dimensional",
        "equivalent_dimensional_smoothing_parameter": smoothing,
        "local_scale": None,
        "chord_length": None,
        "degenerate_chord": False,
        "chord_fallback": None,
        "u": None,
        "normalized_xy": None,
    }


def _parametric_spline_predictions_dimensionless(
    points_xy: np.ndarray,
    spline_kind: str,
    smoothing_factor: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict]:
    normalized = normalized_parametric_coordinates(points_xy)
    u = normalized["u"]
    normalized_xy = normalized["normalized_xy"]
    unique_u, unique_indices = np.unique(u, return_index=True)
    unique_points = normalized_xy[unique_indices]
    if unique_u.size < MIN_CUBIC_POINTS or unique_u[-1] <= 0:
        raise ValueError("At least four unique normalized arc samples are required for a cubic spline.")

    if spline_kind == "interpolating":
        spline_x = CubicSpline(unique_u, unique_points[:, 0], bc_type="natural")
        spline_y = CubicSpline(unique_u, unique_points[:, 1], bc_type="natural")
        smoothing = None
    elif spline_kind == "smoothing":
        smoothing = float(smoothing_factor * unique_u.size)
        spline_x = UnivariateSpline(unique_u, unique_points[:, 0], k=3, s=smoothing)
        spline_y = UnivariateSpline(unique_u, unique_points[:, 1], k=3, s=smoothing)
    else:
        raise ValueError("spline_kind must be 'smoothing' or 'interpolating'.")

    fitted_norm = np.column_stack([spline_x(u), spline_y(u)])
    fitted = normalized["origin_xy"] + normalized["local_scale"] * fitted_norm
    equivalent_dimensional = None if smoothing is None else smoothing * normalized["local_scale"] ** 2
    return fitted, normalized["s"], smoothing, {
        "formulation": "dimensionless_chord",
        "equivalent_dimensional_smoothing_parameter": equivalent_dimensional,
        "local_scale": normalized["local_scale"],
        "chord_length": normalized["chord_length"],
        "degenerate_chord": normalized["degenerate_chord"],
        "chord_fallback": normalized["chord_fallback"],
        "u": u,
        "normalized_xy": normalized_xy,
    }


def _contiguous_sector_runs(contour_xy: np.ndarray, sector_ids: np.ndarray, sector_index: int) -> List[np.ndarray]:
    indices = np.nonzero(sector_ids == sector_index)[0]
    if indices.size == 0:
        return []

    breaks = np.where(np.diff(indices) > 1)[0] + 1
    runs = [run for run in np.split(indices, breaks) if run.size > 0]
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == contour_xy.shape[0] - 1:
        runs = [np.concatenate([runs[-1], runs[0]])] + runs[1:-1]
    return [contour_xy[run] for run in runs]


def evaluate_border_octants_radial(
    mask: np.ndarray,
    spline_kind: str = DEFAULT_SPLINE_KIND,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_points: bool = False,
) -> Dict:
    lesion_mask = mask.astype(bool)
    centroid = lesion_centroid(lesion_mask)
    points = extract_boundary_points(lesion_mask)
    if points.size == 0:
        raise ValueError("No lesion boundary pixels available.")

    theta, radius = angles_and_radii(points, centroid)
    sector_ids = assign_octants(theta)
    sector_results: List[SectorSplineResult] = []

    for sector_index in range(OCTANT_COUNT):
        sector_mask = sector_ids == sector_index
        sector_theta = theta[sector_mask]
        sector_radius = radius[sector_mask]
        threshold = sector_threshold(int(sector_theta.size))
        if sector_theta.size < MIN_CUBIC_POINTS:
            sector_results.append(_insufficient_sector(sector_index, int(sector_theta.size), threshold))
            continue
        try:
            fitted, unique_count, smoothing = _radial_spline_predictions(
                sector_theta,
                sector_radius,
                spline_kind=spline_kind,
                smoothing_factor=smoothing_factor,
            )
            residual = np.abs(sector_radius - fitted)
            error = rmse(sector_radius, fitted)
            sector_results.append(
                SectorSplineResult(
                    sector_index=sector_index,
                    boundary_point_count=int(sector_theta.size),
                    unique_angle_count=unique_count,
                    rmse=error,
                    threshold=threshold,
                    irregular=is_irregular_sector(error, int(sector_theta.size)),
                    insufficient_points=False,
                    mean_residual=float(np.mean(residual)),
                    max_residual=float(np.max(residual)),
                    spline_smoothing_parameter=smoothing,
                    observed_radius=sector_radius if include_points else None,
                    fitted_radius=fitted if include_points else None,
                    theta=sector_theta if include_points else None,
                )
            )
        except ValueError:
            sector_results.append(_insufficient_sector(sector_index, int(sector_theta.size), threshold))

    return _border_result(
        sector_results,
        centroid,
        int(points.shape[0]),
        "radial",
        spline_kind,
        smoothing_factor,
        include_points,
    )


def evaluate_border_octants_parametric(
    mask: np.ndarray,
    spline_kind: str = DEFAULT_SPLINE_KIND,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_points: bool = False,
    smoothing_formulation: str = DEFAULT_PARAMETRIC_SMOOTHING_FORMULATION,
    sector_orientation: str = DEFAULT_SECTOR_ORIENTATION,
) -> Dict:
    lesion_mask = mask.astype(bool)
    centroid = lesion_centroid(lesion_mask)
    contour_xy = extract_ordered_contour(lesion_mask)
    if contour_xy.size == 0:
        raise ValueError("No lesion contour available.")
    principal_axis = None
    sector_angle_offset = 0.0
    if sector_orientation == SECTOR_ORIENTATION_IMAGE_FIXED:
        pass
    elif sector_orientation == SECTOR_ORIENTATION_PRINCIPAL_AXIS_ALIGNED:
        principal_axis = principal_axis_analysis(lesion_mask)
        sector_angle_offset = float(principal_axis["angle_radians"])
    else:
        raise ValueError("sector_orientation must be 'IMAGE_FIXED' or 'PRINCIPAL_AXIS_ALIGNED'.")
    return evaluate_ordered_contour_parametric(
        contour_xy,
        centroid,
        spline_kind=spline_kind,
        smoothing_factor=smoothing_factor,
        include_points=include_points,
        smoothing_formulation=smoothing_formulation,
        sector_orientation=sector_orientation,
        sector_angle_offset=sector_angle_offset,
        principal_axis=principal_axis,
    )


def evaluate_ordered_contour_parametric(
    contour_xy: np.ndarray,
    centroid: Tuple[float, float],
    spline_kind: str = DEFAULT_SPLINE_KIND,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_points: bool = False,
    smoothing_formulation: str = DEFAULT_PARAMETRIC_SMOOTHING_FORMULATION,
    sector_orientation: str = SECTOR_ORIENTATION_IMAGE_FIXED,
    sector_angle_offset: float = 0.0,
    principal_axis: Optional[Dict] = None,
) -> Dict:
    theta = angles_for_xy(contour_xy, centroid)
    sector_ids = assign_octants_relative(theta, sector_angle_offset)
    sector_results: List[SectorSplineResult] = []

    for sector_index in range(OCTANT_COUNT):
        runs = _contiguous_sector_runs(contour_xy, sector_ids, sector_index)
        point_count = int(sum(run.shape[0] for run in runs))
        threshold = IRREGULARITY_THRESHOLD_FACTOR
        fitted_parts = []
        arc_parts = []
        valid_runs = []
        residuals = []
        smoothing_parts = []
        equivalent_smoothing_parts = []
        local_scales = []
        chord_lengths = []
        degenerate_chord = False
        chord_fallbacks = []
        normalized_u_parts = []
        normalized_xy_parts = []
        for run in runs:
            if run.shape[0] < MIN_CUBIC_POINTS:
                continue
            try:
                if smoothing_formulation == "legacy_dimensional":
                    fitted, s, smoothing, fit_meta = _parametric_spline_predictions(
                        run,
                        spline_kind,
                        smoothing_factor,
                    )
                elif smoothing_formulation == "dimensionless_chord":
                    fitted, s, smoothing, fit_meta = _parametric_spline_predictions_dimensionless(
                        run,
                        spline_kind,
                        smoothing_factor,
                    )
                else:
                    raise ValueError("smoothing_formulation must be 'dimensionless_chord' or 'legacy_dimensional'.")
            except ValueError:
                continue
            valid_runs.append(run)
            fitted_parts.append(fitted)
            arc_parts.append(s)
            residuals.append(np.sqrt(np.sum((run - fitted) ** 2, axis=1)))
            if smoothing is not None:
                smoothing_parts.append(smoothing)
            if fit_meta["equivalent_dimensional_smoothing_parameter"] is not None:
                equivalent_smoothing_parts.append(fit_meta["equivalent_dimensional_smoothing_parameter"])
            if fit_meta["local_scale"] is not None:
                local_scales.append(fit_meta["local_scale"])
            if fit_meta["chord_length"] is not None:
                chord_lengths.append(fit_meta["chord_length"])
            degenerate_chord = degenerate_chord or bool(fit_meta["degenerate_chord"])
            if fit_meta["chord_fallback"] is not None:
                chord_fallbacks.append(fit_meta["chord_fallback"])
            if include_points and fit_meta["u"] is not None:
                normalized_u_parts.append(fit_meta["u"])
            if include_points and fit_meta["normalized_xy"] is not None:
                normalized_xy_parts.append(fit_meta["normalized_xy"])

        if not valid_runs:
            sector_results.append(_insufficient_sector(sector_index, point_count, threshold, run_count=len(runs)))
            continue

        residual = np.concatenate(residuals)
        error = float(np.sqrt(np.mean(residual * residual)))
        total_arc_length = float(sum(arc[-1] for arc in arc_parts if arc.size > 0))
        total_smooth_arc_length = float(sum(open_polyline_length(fitted) for fitted in fitted_parts))
        sector_results.append(
            SectorSplineResult(
                sector_index=sector_index,
                boundary_point_count=point_count,
                unique_angle_count=None,
                rmse=error,
                threshold=threshold,
                irregular=is_irregular_by_smoothed_length(error, total_smooth_arc_length),
                insufficient_points=False,
                run_count=len(runs),
                arc_length_total=total_arc_length,
                smooth_arc_length_total=total_smooth_arc_length,
                mean_residual=float(np.mean(residual)),
                max_residual=float(np.max(residual)),
                spline_smoothing_parameter=float(sum(smoothing_parts)) if smoothing_parts else None,
                equivalent_dimensional_smoothing_parameter=float(sum(equivalent_smoothing_parts)) if equivalent_smoothing_parts else None,
                local_chord_length=float(np.mean(local_scales)) if local_scales else None,
                degenerate_chord=degenerate_chord,
                chord_fallback=";".join(sorted(set(chord_fallbacks))) if chord_fallbacks else None,
                normalized_u=np.concatenate(normalized_u_parts) if normalized_u_parts else None,
                normalized_xy=np.vstack(normalized_xy_parts) if normalized_xy_parts else None,
                contour_xy=np.vstack(valid_runs) if include_points else None,
                fitted_xy=np.vstack(fitted_parts) if include_points else None,
                arc_length=np.concatenate(arc_parts) if include_points else None,
            )
        )

    return _border_result(
        sector_results,
        centroid,
        int(contour_xy.shape[0]),
        "parametric",
        spline_kind,
        smoothing_factor,
        include_points,
        smoothing_formulation=smoothing_formulation,
        sector_orientation=sector_orientation,
        sector_angle_offset=sector_angle_offset,
        principal_axis=principal_axis,
    )


def evaluate_border_octants(
    mask: np.ndarray,
    spline_kind: str = DEFAULT_SPLINE_KIND,
    smoothing_factor: float = DEFAULT_SMOOTHING_FACTOR,
    include_points: bool = False,
    representation: str = DEFAULT_REPRESENTATION,
    smoothing_formulation: str = DEFAULT_PARAMETRIC_SMOOTHING_FORMULATION,
    sector_orientation: str = DEFAULT_SECTOR_ORIENTATION,
) -> Dict:
    if representation == "radial":
        return evaluate_border_octants_radial(mask, spline_kind, smoothing_factor, include_points)
    if representation == "parametric":
        return evaluate_border_octants_parametric(
            mask,
            spline_kind,
            smoothing_factor,
            include_points,
            smoothing_formulation,
            sector_orientation,
        )
    raise ValueError("representation must be 'radial' or 'parametric'.")


def _insufficient_sector(
    sector_index: int,
    point_count: int,
    threshold: float,
    run_count: int = 1,
) -> SectorSplineResult:
    return SectorSplineResult(
        sector_index=sector_index,
        boundary_point_count=point_count,
        unique_angle_count=None,
        rmse=None,
        threshold=threshold,
        irregular=False,
        insufficient_points=True,
        run_count=run_count,
    )


def _border_result(
    sector_results: List[SectorSplineResult],
    centroid: Tuple[float, float],
    contour_point_count: int,
    representation: str,
    spline_kind: str,
    smoothing_factor: float,
    include_points: bool,
    smoothing_formulation: Optional[str] = None,
    sector_orientation: str = DEFAULT_SECTOR_ORIENTATION,
    sector_angle_offset: float = 0.0,
    principal_axis: Optional[Dict] = None,
) -> Dict:
    irregular_octants = [result.sector_index for result in sector_results if result.irregular]
    out = {
        "border_score": int(len(irregular_octants)),
        "irregular_octants": irregular_octants,
        "centroid_xy": [centroid[0], centroid[1]],
        "boundary_point_count": contour_point_count,
        "representation": representation,
        "spline_kind": spline_kind,
        "smoothing_factor": smoothing_factor if spline_kind == "smoothing" else None,
        "smoothing_formulation": smoothing_formulation,
        "sector_orientation": sector_orientation,
        "sector_angle_offset_radians": float(sector_angle_offset),
        "sector_angle_offset_degrees": float(math.degrees(sector_angle_offset)),
        "octants": [_sector_result_to_dict(result, include_points=include_points) for result in sector_results],
    }
    if principal_axis is not None:
        out["principal_axis"] = {
            "angle_radians": float(principal_axis["angle_radians"]),
            "angle_degrees": float(math.degrees(principal_axis["angle_radians"])),
            "centroid_xy": [float(principal_axis["centroid_xy"][0]), float(principal_axis["centroid_xy"][1])],
            "mu20": float(principal_axis["mu20"]),
            "mu02": float(principal_axis["mu02"]),
            "mu11": float(principal_axis["mu11"]),
            "lambda1": float(principal_axis["lambda1"]),
            "lambda2": float(principal_axis["lambda2"]),
            "axis_anisotropy": float(principal_axis["axis_anisotropy"]),
            "orientation_status": principal_axis["orientation_status"],
        }
    return out


def _float_or_none(value: Optional[float]) -> Optional[float]:
    return None if value is None else float(value)


def _array_or_none(values: Optional[np.ndarray]) -> Optional[List[float]]:
    return None if values is None else [float(value) for value in values.tolist()]


def _points_or_none(points: Optional[np.ndarray]) -> Optional[List[List[float]]]:
    return None if points is None else [[float(x), float(y)] for x, y in points.tolist()]


def _sector_result_to_dict(result: SectorSplineResult, include_points: bool = False) -> Dict:
    e_smooth = (
        None
        if result.rmse is None or result.smooth_arc_length_total is None or result.smooth_arc_length_total <= 0
        else float(result.rmse / result.smooth_arc_length_total)
    )
    out = {
        "sector_index": result.sector_index,
        "angular_extent_degrees": [45 * result.sector_index, 45 * (result.sector_index + 1)],
        "boundary_point_count": result.boundary_point_count,
        "unique_angle_count": result.unique_angle_count,
        "contiguous_contour_run_count": result.run_count,
        "rmse": _float_or_none(result.rmse),
        "mean_residual": _float_or_none(result.mean_residual),
        "max_residual": _float_or_none(result.max_residual),
        "arc_length": _float_or_none(result.arc_length_total),
        "smooth_arc_length": _float_or_none(result.smooth_arc_length_total),
        "E_smooth": _float_or_none(e_smooth),
        "threshold_margin": _float_or_none(None if e_smooth is None else e_smooth - result.threshold),
        "spline_smoothing_parameter": _float_or_none(result.spline_smoothing_parameter),
        "equivalent_dimensional_smoothing_parameter": _float_or_none(result.equivalent_dimensional_smoothing_parameter),
        "local_chord_length": _float_or_none(result.local_chord_length),
        "degenerate_chord": result.degenerate_chord,
        "chord_fallback": result.chord_fallback,
        "threshold": result.threshold,
        "threshold_rule": (
            "Parametric final operationalization: RMSE / L_smooth > 0.05. "
            "The source-specified RMSE > 0.05 * length(y) form remains under-specified for exact reproduction."
        ),
        "irregular": result.irregular,
        "insufficient_points": result.insufficient_points,
    }
    if include_points:
        out.update(
            {
                "theta_radians": _array_or_none(result.theta),
                "normalized_u": _array_or_none(result.normalized_u),
                "normalized_xy": _points_or_none(result.normalized_xy),
                "observed_radius": _array_or_none(result.observed_radius),
                "fitted_radius": _array_or_none(result.fitted_radius),
                "contour_xy": _points_or_none(result.contour_xy),
                "arc_length": _array_or_none(result.arc_length),
                "arc_length_total": _float_or_none(result.arc_length_total),
                "smooth_arc_length": _float_or_none(result.smooth_arc_length_total),
                "fitted_xy": _points_or_none(result.fitted_xy),
            }
        )
    return out


def extract_abcd_border(mask: np.ndarray) -> EvidenceEnvelope:
    try:
        evidence = evaluate_border_octants(mask)
        return EvidenceEnvelope(
            status=EvidenceStatus.DETERMINISTIC_CLINICAL_WITH_LIMITATION,
            value=evidence["border_score"],
            evidence={
                **evidence,
                "sector_rule": "Eight angular sectors of 45 degrees around the lesion centroid.",
                "spline_rule": "Third-order smoothing spline fitted independently to each sector sub-contour.",
                "threshold_rule": "Sector is positive when Euclidean RMSE divided by smoothed local contour length is greater than 0.05.",
                "source_threshold_gap": "Kasmi and Mokrani specify RMSE > 0.05 * length(y), but the executable meaning of y and length(y) is under-specified.",
                "length_y_convention": "AUTHOR_OPERATIONALIZATION: length(y) is operationalized as L_smooth, the fitted local contour arc length.",
                "residual_convention": "AUTHOR_IMPLEMENTATION_CONVENTION: parametric residual is Euclidean distance from contour point to smoothed x(s), y(s).",
                "smoothing_convention": (
                    "AUTHOR_IMPLEMENTATION_CONVENTION: normalized u=s/s_max and normalized "
                    "coordinates by local chord D_i; scipy UnivariateSpline k=3 uses "
                    f"s={DEFAULT_SMOOTHING_FACTOR} * n_samples."
                ),
                "local_scale_convention": "AUTHOR_IMPLEMENTATION_CONVENTION: D_i is the octant sub-contour chord length; degenerate chords fall back to local arc length and are recorded.",
                "sector_orientation_convention": "AUTHOR_IMPLEMENTATION_CONVENTION: sectors are assigned in the lesion principal-axis frame using theta_relative = theta_image - phi.",
                "empty_sector_rule": "Sectors with fewer than four usable arc-length samples are not marked irregular.",
            },
            limitations=[
                "Geometric sub-contour irregularity proxy for ABCD abrupt pigment cutoff; does not directly measure lesion-to-skin pigment transition.",
                "Published computational threshold, not a universal clinical threshold.",
                "No PH2 calibration.",
                "Parametric residual, smoothing factor, and length(y) interpretation are explicit implementation conventions validated only on synthetic canonical masks.",
            ],
            source=SOURCE,
            extractor_id="ABCD-B-001",
        )
    except Exception as exc:
        return EvidenceEnvelope(
            status=EvidenceStatus.UNAVAILABLE,
            value=None,
            evidence={},
            limitations=[
                "ABCD B could not be computed for this mask; unavailable is not clinical absence.",
            ],
            source=SOURCE,
            extractor_id="ABCD-B-001",
            failure_reason=str(exc),
        )
