"""Path-interaction constraints for multi-harness routing.

Provides attraction (corridor reuse) and repulsion (EMI isolation) constraints
that modify cost/speed fields based on distance to previously routed paths.
"""

from dataclasses import dataclass

import numpy as np

from src.sdf.primitives import sdf_line_segment


@dataclass
class PathConstraint:
    """Describes how a new route should interact with an existing path.

    Attributes:
        reference_path: (N, 2) previously routed path in world coordinates.
        mode: "attract" (reuse corridor) or "repel" (isolate).
        influence_radius: World-coordinate radius of the influence zone.
        strength: Intensity of the effect (0.0 to 5.0, typical 1.0-3.0).
    """
    reference_path: np.ndarray
    mode: str
    influence_radius: float
    strength: float

    def __post_init__(self):
        self.reference_path = np.asarray(self.reference_path, dtype=float)
        if self.mode not in ("attract", "repel"):
            raise ValueError(f"mode must be 'attract' or 'repel', got '{self.mode}'")
        if self.influence_radius <= 0:
            raise ValueError(f"influence_radius must be positive, got {self.influence_radius}")
        if self.strength < 0:
            raise ValueError(f"strength must be non-negative, got {self.strength}")


def compute_path_distance_field(path: np.ndarray, sdf_grid) -> np.ndarray:
    """Compute the unsigned distance from every grid cell to a polyline path.

    Args:
        path: (N, 2) path vertices in world coordinates.
        sdf_grid: SDFGrid providing the grid geometry.

    Returns:
        (ny, nx) array of distances to the nearest path segment.
    """
    X, Y = sdf_grid.meshgrid_world()
    points = np.column_stack([X.ravel(), Y.ravel()])

    dist = np.full(points.shape[0], np.inf)
    for i in range(len(path) - 1):
        seg_dist = sdf_line_segment(points, path[i], path[i + 1], thickness=0.0)
        dist = np.minimum(dist, seg_dist)

    return dist.reshape(sdf_grid.shape)


def _compute_influence(dist_field: np.ndarray, radius: float) -> np.ndarray:
    """Normalized influence: 1 on path, 0 at radius, clipped."""
    return np.clip(1.0 - dist_field / radius, 0.0, 1.0)


def build_cost_modifier(constraints: list, sdf_grid) -> np.ndarray:
    """Build a multiplicative cost modifier for A* from path constraints.

    Attract: lower cost near reference path.
    Repel: raise cost near reference path.

    Returns:
        (ny, nx) modifier array (multiply with cost_field).
    """
    modifier = np.ones(sdf_grid.shape)
    for c in constraints:
        dist = compute_path_distance_field(c.reference_path, sdf_grid)
        influence = _compute_influence(dist, c.influence_radius)
        if c.mode == "attract":
            modifier *= np.maximum(1.0 - c.strength * influence, 0.05)
        else:  # repel
            modifier *= (1.0 + c.strength * influence)
    return modifier


def build_speed_modifier(constraints: list, sdf_grid) -> np.ndarray:
    """Build a multiplicative speed modifier for FMM from path constraints.

    Attract: increase speed near reference path.
    Repel: decrease speed near reference path.

    Returns:
        (ny, nx) modifier array (multiply with speed field).
    """
    modifier = np.ones(sdf_grid.shape)
    for c in constraints:
        dist = compute_path_distance_field(c.reference_path, sdf_grid)
        influence = _compute_influence(dist, c.influence_radius)
        if c.mode == "attract":
            modifier *= (1.0 + c.strength * influence)
        else:  # repel
            modifier *= np.maximum(1.0 - c.strength * influence, 0.05)
    return modifier


def nearest_point_on_path(points: np.ndarray, reference_path: np.ndarray):
    """Find nearest point on a polyline for each query point.

    Args:
        points: (M, 2) query points.
        reference_path: (N, 2) polyline vertices.

    Returns:
        (nearest_points, distances): each (M, 2) and (M,).
    """
    points = np.asarray(points, dtype=float)
    reference_path = np.asarray(reference_path, dtype=float)

    best_dist = np.full(len(points), np.inf)
    best_nearest = np.zeros_like(points)

    for i in range(len(reference_path) - 1):
        a = reference_path[i]
        b = reference_path[i + 1]
        pa = points - a
        ba = b - a
        t = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        closest = a + t[:, np.newaxis] * ba
        dist = np.linalg.norm(points - closest, axis=-1)
        improved = dist < best_dist
        best_dist[improved] = dist[improved]
        best_nearest[improved] = closest[improved]

    return best_nearest, best_dist


def path_constraint_cost_and_gradient(path: np.ndarray, constraints: list):
    """Compute path constraint cost and gradient for GradientOptimizer.

    Interface mirrors gradient.py _obstacle_cost_and_gradient:
    returns (total_cost, grad) where grad has shape (N, 2).

    Attract: C(p) = 0.5 * s * (d/r)^2 — pulls path toward reference.
    Repel:   C(p) = 0.5 * s * ((r-d)/r)^2 — pushes path away from reference.
    """
    total_cost = 0.0
    grad = np.zeros_like(path)

    for c in constraints:
        nearest, dist = nearest_point_on_path(path, c.reference_path)
        r = c.influence_radius
        s = c.strength

        # Direction away from reference path
        direction = path - nearest
        dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        safe_norm = np.maximum(dir_norm, 1e-8)
        unit_outward = direction / safe_norm

        mask = dist < r

        if c.mode == "attract":
            # Cost increases with distance to reference -> gradient descent pulls closer
            d = dist[mask]
            cost_vals = 0.5 * s * (d / r) ** 2
            total_cost += cost_vals.sum()
            # Gradient: d(cost)/d(path) = s * d / r^2 * unit_outward
            grad[mask] += s * (d / (r ** 2))[:, np.newaxis] * unit_outward[mask]
        else:  # repel
            # Cost increases with proximity to reference -> gradient descent pushes away
            d = dist[mask]
            gap = r - d
            cost_vals = 0.5 * s * (gap / r) ** 2
            total_cost += cost_vals.sum()
            # Gradient: negative outward = toward reference,
            # so descent moves *away* from reference
            grad[mask] -= s * (gap / (r ** 2))[:, np.newaxis] * unit_outward[mask]

    return total_cost, grad
