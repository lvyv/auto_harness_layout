"""Boolean CSG operations on SDF arrays."""

import numpy as np


def sdf_union(*sdfs: np.ndarray) -> np.ndarray:
    """Union of multiple SDFs: region outside ALL obstacles.

    Result = min(d1, d2, ...)
    """
    return np.minimum.reduce(sdfs)


def sdf_intersection(*sdfs: np.ndarray) -> np.ndarray:
    """Intersection of multiple SDFs: region inside ALL shapes.

    Result = max(d1, d2, ...)
    """
    return np.maximum.reduce(sdfs)


def sdf_difference(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """Difference A - B: region inside A but outside B.

    Result = max(d_a, -d_b)
    """
    return np.maximum(sdf_a, -sdf_b)


def sdf_smooth_union(sdf_a: np.ndarray, sdf_b: np.ndarray,
                     k: float = 0.5) -> np.ndarray:
    """Smooth (polynomial) union for blending shapes.

    k controls the blending radius.
    """
    h = np.clip(0.5 + 0.5 * (sdf_b - sdf_a) / k, 0.0, 1.0)
    return sdf_b * (1.0 - h) + sdf_a * h - k * h * (1.0 - h)
