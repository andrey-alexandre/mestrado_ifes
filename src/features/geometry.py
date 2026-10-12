"""Shared lesion geometry utilities."""
import math
from typing import Dict, Tuple

import numpy as np


def principal_axis_analysis(mask: np.ndarray) -> Dict:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("No lesion pixels available.")
    cx = float(xs.mean())
    cy = float(ys.mean())
    x = xs.astype(np.float64) - cx
    y = ys.astype(np.float64) - cy
    mu20 = float(np.mean(x * x))
    mu02 = float(np.mean(y * y))
    mu11 = float(np.mean(x * y))
    angle = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    covariance = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    lambda1 = float(eigenvalues[0])
    lambda2 = float(eigenvalues[1])
    denominator = lambda1 + lambda2
    anisotropy = 0.0 if denominator <= 0 else float((lambda1 - lambda2) / denominator)
    return {
        "angle_radians": float(angle),
        "centroid_xy": (cx, cy),
        "mu20": mu20,
        "mu02": mu02,
        "mu11": mu11,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "axis_anisotropy": anisotropy,
        "orientation_status": "PRINCIPAL_AXIS_UNDERDETERMINED" if anisotropy <= 1e-12 else "DEFINED",
    }


def principal_axis(mask: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    analysis = principal_axis_analysis(mask)
    return analysis["angle_radians"], analysis["centroid_xy"]


def axial_angle_difference(a: float, b: float) -> float:
    return float(abs(((a - b + math.pi / 2.0) % math.pi) - math.pi / 2.0))
