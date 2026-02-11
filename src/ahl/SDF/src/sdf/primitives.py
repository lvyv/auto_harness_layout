"""Analytical SDF functions for basic 2D shapes.

Each function takes an array of points (N, 2) and shape parameters,
returning an (N,) array of signed distances.
Negative = inside, zero = boundary, positive = outside.
"""

import numpy as np


def sdf_circle(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Signed distance to a circle.

    SDF(p) = ||p - c|| - r
    """
    center = np.asarray(center, dtype=float)
    return np.linalg.norm(points - center, axis=-1) - radius


def sdf_rectangle(points: np.ndarray, center: np.ndarray,
                  half_extents: np.ndarray) -> np.ndarray:
    """Signed distance to an axis-aligned rectangle.

    Uses the standard box SDF formula that correctly handles corners:
    outside distance = ||max(d, 0)||
    inside distance  = min(max(dx, dy), 0)
    where d = |p - c| - half_extents
    """
    center = np.asarray(center, dtype=float)
    half_extents = np.asarray(half_extents, dtype=float)
    d = np.abs(points - center) - half_extents
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside = np.minimum(np.max(d, axis=-1), 0.0)
    return outside + inside


def sdf_line_segment(points: np.ndarray, a: np.ndarray,
                     b: np.ndarray, thickness: float = 0.0) -> np.ndarray:
    """Signed distance to a line segment (with optional thickness).

    Useful for modeling wall-like obstacles.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pa = points - a
    ba = b - a
    t = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
    closest = a + t[:, np.newaxis] * ba
    dist = np.linalg.norm(points - closest, axis=-1)
    return dist - thickness
