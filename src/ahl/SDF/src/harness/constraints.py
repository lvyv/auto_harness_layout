"""Wire harness constraint checking: curvature, bend radius, clearance."""

import numpy as np

from src.sdf.grid import SDFGrid


def compute_curvature(path: np.ndarray) -> np.ndarray:
    """Compute discrete (Menger) curvature at each interior point.

    For three consecutive points p[i-1], p[i], p[i+1], the Menger curvature
    is defined as:
        kappa = 2 * |cross(a, b)| / (|a| * |b| * |a + b|)
    where a = p[i] - p[i-1], b = p[i+1] - p[i].

    This equals 1/R where R is the circumscribed circle radius.

    Args:
        path: (N, 2) array of path points.

    Returns:
        (N-2,) array of curvature values at interior points path[1:-1].
    """
    if len(path) < 3:
        return np.array([])

    a = path[1:-1] - path[:-2]   # vectors from p[i-1] to p[i]
    b = path[2:] - path[1:-1]    # vectors from p[i] to p[i+1]

    # 2D cross product magnitude: |ax*by - ay*bx|
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])

    len_a = np.linalg.norm(a, axis=1)
    len_b = np.linalg.norm(b, axis=1)
    len_ab = np.linalg.norm(a + b, axis=1)

    # Avoid division by zero for collinear/coincident points
    denom = len_a * len_b * len_ab
    denom = np.maximum(denom, 1e-15)

    curvature = 2.0 * cross / denom
    return curvature


def compute_bend_radius(path: np.ndarray) -> np.ndarray:
    """Compute bend radius at each interior point.

    Args:
        path: (N, 2) array of path points.

    Returns:
        (N-2,) array of bend radii. INF for straight segments.
    """
    curvature = compute_curvature(path)
    # Avoid division by zero
    radius = np.where(curvature > 1e-12, 1.0 / curvature, np.inf)
    return radius


def check_bend_radius(path: np.ndarray, min_radius: float) -> list[dict]:
    """Check if path violates minimum bend radius constraint.

    Args:
        path: (N, 2) array of path points.
        min_radius: Minimum allowed bend radius.

    Returns:
        List of violations, each a dict with:
            - "index": Index in the original path array.
            - "position": (x, y) world coordinate of the violation.
            - "radius": Actual bend radius at this point.
            - "required": Required minimum bend radius.
            - "severity": How much the radius falls short (ratio).
    """
    radii = compute_bend_radius(path)
    violations = []

    for i, radius in enumerate(radii):
        if radius < min_radius:
            violations.append({
                "index": i + 1,  # offset because curvature is for interior points
                "position": path[i + 1].copy(),
                "radius": float(radius),
                "required": float(min_radius),
                "severity": float(min_radius / max(radius, 1e-12)),
            })

    return violations


def check_clearance(path: np.ndarray, sdf_grid: SDFGrid,
                    min_clearance: float) -> list[dict]:
    """Check if path maintains minimum clearance from obstacles.

    Args:
        path: (N, 2) array of path points.
        sdf_grid: The signed distance field.
        min_clearance: Minimum required distance from obstacles.

    Returns:
        List of violations with index, position, clearance, and required.
    """
    sdf_vals = sdf_grid.sample(path)
    violations = []

    for i, (sdf_val, point) in enumerate(zip(sdf_vals, path)):
        if sdf_val < min_clearance:
            violations.append({
                "index": i,
                "position": point.copy(),
                "clearance": float(sdf_val),
                "required": float(min_clearance),
                "inside_obstacle": sdf_val <= 0,
            })

    return violations


def path_length(path: np.ndarray) -> float:
    """Compute total Euclidean length of a path."""
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_min_clearance(path: np.ndarray, sdf_grid: SDFGrid) -> float:
    """Compute minimum SDF value (clearance) along the path."""
    if len(path) == 0:
        return float('inf')
    sdf_vals = sdf_grid.sample(path)
    return float(np.min(sdf_vals))


def path_statistics(path: np.ndarray, sdf_grid: SDFGrid,
                    min_bend_radius: float = None) -> dict:
    """Compute comprehensive statistics for a path.

    Returns dict with: length, min_clearance, mean_clearance,
    max_curvature, min_bend_radius_actual, bend_violations_count.
    """
    stats = {
        "num_points": len(path),
        "length": path_length(path),
    }

    if len(path) > 0:
        sdf_vals = sdf_grid.sample(path)
        stats["min_clearance"] = float(np.min(sdf_vals))
        stats["mean_clearance"] = float(np.mean(sdf_vals))
    else:
        stats["min_clearance"] = float('inf')
        stats["mean_clearance"] = float('inf')

    if len(path) >= 3:
        curvatures = compute_curvature(path)
        radii = compute_bend_radius(path)
        stats["max_curvature"] = float(np.max(curvatures))
        stats["min_bend_radius_actual"] = float(np.min(radii))
    else:
        stats["max_curvature"] = 0.0
        stats["min_bend_radius_actual"] = float('inf')

    if min_bend_radius is not None:
        violations = check_bend_radius(path, min_bend_radius)
        stats["bend_violations_count"] = len(violations)
        stats["bend_violations"] = violations
    else:
        stats["bend_violations_count"] = 0

    return stats
