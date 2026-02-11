"""CHOMP-inspired gradient trajectory optimizer.

Takes an initial path (from A* or FMM) and refines it using gradient descent
on a combined smoothness + obstacle avoidance objective:

    F(path) = F_smooth(path) + lambda_obs * F_obstacle(path)

F_smooth penalizes acceleration (second finite differences).
F_obstacle penalizes proximity to obstacles using the SDF.
"""

import time

import numpy as np

from src.sdf.grid import SDFGrid
from .base import PlanResult


class GradientOptimizer:
    """CHOMP-style trajectory optimizer using SDF gradient information."""

    def __init__(self, sdf_grid: SDFGrid, safety_margin: float = 1.5,
                 lambda_obs: float = 15.0, learning_rate: float = 0.02,
                 max_iters: int = 500, convergence_tol: float = 1e-5,
                 path_constraints: list = None, lambda_path: float = 10.0):
        """
        Args:
            sdf_grid: The signed distance field.
            safety_margin: Distance threshold for obstacle cost activation.
            lambda_obs: Weight of obstacle cost relative to smoothness.
            learning_rate: Gradient descent step size.
            max_iters: Maximum optimization iterations.
            convergence_tol: Stop when gradient norm falls below this.
            path_constraints: List of PathConstraint for multi-route interaction.
            lambda_path: Weight of path constraint cost.
        """
        self.sdf_grid = sdf_grid
        self.safety_margin = safety_margin
        self.lambda_obs = lambda_obs
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.convergence_tol = convergence_tol
        self.path_constraints = path_constraints or []
        self.lambda_path = lambda_path

    def _smoothness_gradient(self, path: np.ndarray) -> np.ndarray:
        """Gradient of smoothness cost (discrete Laplacian).

        For interior point i: grad[i] = -(path[i-1] - 2*path[i] + path[i+1])
        This pushes toward a straighter path.
        """
        n = len(path)
        grad = np.zeros_like(path)
        # Interior points only
        grad[1:-1] = -(path[:-2] - 2 * path[1:-1] + path[2:])
        return grad

    def _obstacle_cost_and_gradient(self, path: np.ndarray):
        """Compute obstacle cost and its gradient at each path point.

        Cost at point p:
            c(p) = 0.5 * (margin - sdf(p))^2 / margin^2   if sdf(p) < margin
            c(p) = 0                                         otherwise

        Gradient:
            dc/dp = -(margin - sdf(p)) / margin^2 * grad_sdf(p)  if sdf(p) < margin
            dc/dp = 0                                              otherwise
        """
        sdf_vals = self.sdf_grid.sample(path)        # (N,)
        sdf_grads = self.sdf_grid.gradient(path)     # (N, 2)

        margin = self.safety_margin
        cost = np.zeros(len(path))
        grad = np.zeros_like(path)

        # Points within the safety margin
        mask = sdf_vals < margin
        if np.any(mask):
            diff = margin - sdf_vals[mask]
            cost[mask] = 0.5 * (diff ** 2) / (margin ** 2)
            # Gradient: pushes away from obstacles
            grad[mask] = -(diff / (margin ** 2))[:, np.newaxis] * sdf_grads[mask]

        # Extra strong penalty for points inside obstacles
        inside = sdf_vals <= 0
        if np.any(inside):
            cost[inside] = 10.0 + (margin - sdf_vals[inside]) ** 2
            sdf_grad_norms = np.linalg.norm(sdf_grads[inside], axis=-1, keepdims=True)
            safe_norms = np.maximum(sdf_grad_norms, 1e-8)
            # Push strongly outward along SDF gradient
            grad[inside] = -20.0 * sdf_grads[inside] / safe_norms

        return cost.sum(), grad

    def _path_constraint_cost_and_gradient(self, path: np.ndarray):
        """Compute path-interaction cost and gradient."""
        if not self.path_constraints:
            return 0.0, np.zeros_like(path)
        from src.harness.path_constraints import path_constraint_cost_and_gradient
        return path_constraint_cost_and_gradient(path, self.path_constraints)

    def _project_to_free_space(self, path: np.ndarray) -> np.ndarray:
        """Project points that are inside obstacles back to the surface."""
        sdf_vals = self.sdf_grid.sample(path)
        sdf_grads = self.sdf_grid.gradient(path)

        inside = sdf_vals < 0.01  # Small buffer
        if np.any(inside):
            grad_norms = np.linalg.norm(sdf_grads[inside], axis=-1, keepdims=True)
            safe_norms = np.maximum(grad_norms, 1e-8)
            displacement = (0.02 - sdf_vals[inside])[:, np.newaxis] * \
                           sdf_grads[inside] / safe_norms
            path[inside] += displacement
        return path

    def optimize(self, initial_path: np.ndarray,
                 record_history: bool = False) -> dict:
        """Refine path using gradient descent.

        Args:
            initial_path: (N, 2) initial path from A*/FMM.
            record_history: If True, record path at each iteration.

        Returns:
            Dict with keys: 'path', 'cost_history', 'iterations',
            'time_seconds', and optionally 'path_history'.
        """
        t0 = time.perf_counter()

        path = initial_path.copy()
        cost_history = []
        path_history = [path.copy()] if record_history else None

        for iteration in range(self.max_iters):
            # Compute gradients
            grad_smooth = self._smoothness_gradient(path)
            obs_cost, grad_obs = self._obstacle_cost_and_gradient(path)
            path_cost, grad_path = self._path_constraint_cost_and_gradient(path)

            # Total gradient
            grad_total = (grad_smooth + self.lambda_obs * grad_obs
                          + self.lambda_path * grad_path)

            # Smoothness cost
            diffs = path[:-2] - 2 * path[1:-1] + path[2:]
            smooth_cost = 0.5 * np.sum(diffs ** 2)
            total_cost = (smooth_cost + self.lambda_obs * obs_cost
                          + self.lambda_path * path_cost)
            cost_history.append(total_cost)

            # Gradient norm (interior points only)
            grad_norm = np.linalg.norm(grad_total[1:-1])

            # Update interior points only (fix start and goal)
            path[1:-1] -= self.learning_rate * grad_total[1:-1]

            # Project back to free space
            path = self._project_to_free_space(path)

            if record_history:
                path_history.append(path.copy())

            # Convergence check
            if grad_norm < self.convergence_tol:
                break

        elapsed = time.perf_counter() - t0

        result = {
            'path': path,
            'cost_history': cost_history,
            'iterations': iteration + 1,
            'time_seconds': elapsed,
        }
        if record_history:
            result['path_history'] = path_history

        return result
