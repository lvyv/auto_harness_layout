"""Path smoothing with B-spline fitting and bend radius constraint enforcement."""

import numpy as np
from scipy.interpolate import splprep, splev

from src.sdf.grid import SDFGrid
from .constraints import compute_curvature, check_bend_radius


def smooth_path_bspline(path: np.ndarray, num_output_points: int = 200,
                        smoothing_factor: float = None,
                        degree: int = 3) -> np.ndarray:
    """Smooth a piecewise-linear path using B-spline fitting.

    Args:
        path: (N, 2) path points.
        num_output_points: Number of points in the output smoothed path.
        smoothing_factor: Smoothing factor for splprep. None = auto.
            Larger values = smoother but less faithful to original path.
        degree: B-spline degree (1-5, default 3 = cubic).

    Returns:
        (num_output_points, 2) smoothed path.
    """
    if len(path) < degree + 1:
        return path.copy()

    # Parameterize by cumulative arc length
    diffs = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])

    # Remove duplicate points (zero-length segments cause splprep issues)
    mask = np.concatenate([[True], seg_lengths > 1e-10])
    path_clean = path[mask]
    if len(path_clean) < degree + 1:
        return path.copy()

    # Fit B-spline
    try:
        tck, u = splprep([path_clean[:, 0], path_clean[:, 1]],
                         s=smoothing_factor, k=degree)
    except Exception:
        return path.copy()

    # Resample at uniform parameter values
    u_new = np.linspace(0, 1, num_output_points)
    x_new, y_new = splev(u_new, tck)

    smoothed = np.column_stack([x_new, y_new])

    # Ensure start and end points are preserved exactly
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]

    return smoothed


def _compute_spline_curvature(tck, u: np.ndarray) -> np.ndarray:
    """Compute exact curvature from B-spline parameterization.

    kappa = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
    """
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5
    denominator = np.maximum(denominator, 1e-15)

    return numerator / denominator


def smooth_with_bend_constraint(
    path: np.ndarray,
    min_bend_radius: float,
    sdf_grid: SDFGrid = None,
    min_clearance: float = 0.0,
    num_output_points: int = 200,
    max_iterations: int = 30,
    initial_smoothing: float = None,
) -> dict:
    """Iteratively smooth path while enforcing bend radius and collision constraints.

    Algorithm:
    1. Fit B-spline with increasing smoothing factor
    2. Check bend radius violations
    3. If violations exist, increase smoothing (more relaxation)
    4. Check collisions (SDF > min_clearance at all points)
    5. Return the best path satisfying all constraints

    Args:
        path: (N, 2) initial path.
        min_bend_radius: Minimum allowed bend radius.
        sdf_grid: SDF grid for collision checking (optional).
        min_clearance: Minimum SDF clearance.
        num_output_points: Points in output path.
        max_iterations: Maximum smoothing iterations.
        initial_smoothing: Starting smoothing factor (None = auto).

    Returns:
        Dict with 'path', 'violations', 'iterations', 'converged'.
    """
    if len(path) < 4:
        return {
            'path': path.copy(),
            'violations': [],
            'iterations': 0,
            'converged': True,
        }

    # Estimate initial smoothing factor from path length
    total_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    if initial_smoothing is None:
        s = total_length * 0.001  # Start with low smoothing
    else:
        s = initial_smoothing

    best_path = path.copy()
    best_violation_count = len(check_bend_radius(path, min_bend_radius))
    max_curvature = 1.0 / min_bend_radius

    for iteration in range(max_iterations):
        # Smooth with current factor
        smoothed = smooth_path_bspline(path, num_output_points,
                                       smoothing_factor=s)

        # Check bend radius
        violations = check_bend_radius(smoothed, min_bend_radius)

        # Check clearance if SDF provided
        collision_free = True
        if sdf_grid is not None and len(smoothed) > 0:
            sdf_vals = sdf_grid.sample(smoothed)
            if np.any(sdf_vals < min_clearance):
                collision_free = False

        if len(violations) == 0 and collision_free:
            return {
                'path': smoothed,
                'violations': [],
                'iterations': iteration + 1,
                'converged': True,
            }

        # Track best result
        if len(violations) < best_violation_count and collision_free:
            best_violation_count = len(violations)
            best_path = smoothed

        # Increase smoothing for next iteration
        s *= 1.8

    # Final attempt: if still has violations, try local curvature reduction
    smoothed = _local_curvature_reduction(best_path, min_bend_radius,
                                          sdf_grid, min_clearance)
    violations = check_bend_radius(smoothed, min_bend_radius)

    return {
        'path': smoothed,
        'violations': violations,
        'iterations': max_iterations,
        'converged': len(violations) == 0,
    }


def _local_curvature_reduction(
    path: np.ndarray,
    min_bend_radius: float,
    sdf_grid: SDFGrid = None,
    min_clearance: float = 0.0,
    max_iters: int = 100,
    step_size: float = 0.1,
) -> np.ndarray:
    """Locally push high-curvature points outward to reduce curvature.

    At each violation point, displace it away from the center of curvature
    (i.e., outward from the bend) to increase the bend radius.
    """
    path = path.copy()
    max_curvature = 1.0 / min_bend_radius

    for _ in range(max_iters):
        curvatures = compute_curvature(path)
        if len(curvatures) == 0:
            break

        violation_mask = curvatures > max_curvature
        if not np.any(violation_mask):
            break

        # For each violation, compute displacement direction
        for i in np.where(violation_mask)[0]:
            idx = i + 1  # index in path array
            if idx <= 0 or idx >= len(path) - 1:
                continue

            # Direction away from center of curvature:
            # The center of curvature is on the concave side.
            # We approximate the outward direction using the path normal.
            a = path[idx] - path[idx - 1]
            b = path[idx + 1] - path[idx]

            # Bisector direction (toward the outside of the bend)
            a_norm = a / max(np.linalg.norm(a), 1e-10)
            b_norm = b / max(np.linalg.norm(b), 1e-10)
            outward = a_norm + b_norm
            out_len = np.linalg.norm(outward)
            if out_len < 1e-10:
                continue
            outward = outward / out_len

            # Displace proportional to violation severity
            excess = curvatures[i] - max_curvature
            displacement = step_size * excess / max_curvature * outward
            new_pos = path[idx] + displacement

            # Check clearance at new position
            if sdf_grid is not None:
                sdf_val = sdf_grid.sample(new_pos.reshape(1, 2))[0]
                if sdf_val < min_clearance:
                    continue  # Skip this adjustment if it causes collision

            path[idx] = new_pos

    return path
