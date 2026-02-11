"""Fast Marching Method path planner using scikit-fmm.

The FMM computes a travel-time field from the goal, using a speed field
derived from the SDF. The optimal path is then extracted by tracing the
negative gradient of the travel-time field from start back to goal.
"""

import time

import numpy as np
import skfmm
from scipy.ndimage import map_coordinates

from src.sdf.grid import SDFGrid
from .base import BasePlanner, PlanResult


class FastMarchingPlanner(BasePlanner):
    """Path planner based on the Fast Marching Method.

    Converts the SDF into a speed field (higher speed = farther from obstacles),
    computes travel time from goal, then traces back the optimal path.
    """

    def __init__(self, safety_margin: float = 2.0, speed_exponent: float = 2.0,
                 step_factor: float = 0.4):
        """
        Args:
            safety_margin: SDF distance at which speed reaches maximum.
            speed_exponent: Controls how sharply speed drops near obstacles.
                Higher values push paths further from obstacles.
            step_factor: Gradient descent step size as fraction of grid spacing.
        """
        self.safety_margin = safety_margin
        self.speed_exponent = speed_exponent
        self.step_factor = step_factor

    def _sdf_to_speed(self, sdf_grid: SDFGrid) -> np.ndarray:
        """Convert SDF to a speed field.

        speed = clamp(sdf / safety_margin, 0, 1) ^ exponent
        Zero speed inside obstacles, max speed far from obstacles.
        """
        normalized = np.clip(sdf_grid.values / self.safety_margin, 0.0, 1.0)
        speed = normalized ** self.speed_exponent
        # Ensure minimum positive speed to avoid division issues in FMM
        speed = np.maximum(speed, 1e-10)
        # Zero out obstacle interiors
        speed[sdf_grid.values <= 0] = 1e-10
        return speed

    def _trace_gradient(self, travel_time: np.ndarray, start_idx: np.ndarray,
                        goal_idx: np.ndarray, sdf_grid: SDFGrid,
                        max_steps: int = 50000) -> np.ndarray:
        """Follow steepest descent of travel-time field from start to goal.

        Args:
            travel_time: (ny, nx) travel-time field.
            start_idx: (col, row) grid index of start.
            goal_idx: (col, row) grid index of goal.
            sdf_grid: SDF grid for bounds checking.
            max_steps: Maximum gradient descent steps.

        Returns:
            (N, 2) array of (col, row) grid indices along the path.
        """
        ny, nx = travel_time.shape
        # Compute gradient of travel time: [d/dy, d/dx]
        gt_y, gt_x = np.gradient(travel_time)

        step_size = self.step_factor
        current = start_idx.astype(float).copy()  # (col, row)
        goal_f = goal_idx.astype(float)

        path = [current.copy()]

        for _ in range(max_steps):
            col, row = current[0], current[1]

            # Check if close enough to goal
            if np.linalg.norm(current - goal_f) < 1.5:
                path.append(goal_f.copy())
                break

            # Bounds check
            if not (0.5 < row < ny - 1.5 and 0.5 < col < nx - 1.5):
                break

            # Interpolate gradient at current position
            gx = map_coordinates(gt_x, [[row], [col]], order=1, mode='nearest')[0]
            gy = map_coordinates(gt_y, [[row], [col]], order=1, mode='nearest')[0]

            grad_mag = np.sqrt(gx ** 2 + gy ** 2)
            if grad_mag < 1e-12:
                break

            # Step in negative gradient direction (toward lower travel time)
            dcol = -gx / grad_mag * step_size
            drow = -gy / grad_mag * step_size

            current = current + np.array([dcol, drow])

            # Clamp to grid bounds
            current[0] = np.clip(current[0], 0, nx - 1)
            current[1] = np.clip(current[1], 0, ny - 1)

            path.append(current.copy())

        return np.array(path)

    def plan(self, sdf_grid: SDFGrid, start: np.ndarray,
             goal: np.ndarray, **kwargs) -> PlanResult:
        t0 = time.perf_counter()

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Convert to grid indices
        start_grid = sdf_grid.world_to_grid(start)  # (col, row)
        goal_grid = sdf_grid.world_to_grid(goal)

        start_col, start_row = int(round(start_grid[0])), int(round(start_grid[1]))
        goal_col, goal_row = int(round(goal_grid[0])), int(round(goal_grid[1]))

        ny, nx = sdf_grid.shape

        # Validate
        if not (0 <= start_row < ny and 0 <= start_col < nx):
            return PlanResult(np.empty((0, 2)), float('inf'),
                              time.perf_counter() - t0, False,
                              metadata={"error": "Start outside grid"})
        if not (0 <= goal_row < ny and 0 <= goal_col < nx):
            return PlanResult(np.empty((0, 2)), float('inf'),
                              time.perf_counter() - t0, False,
                              metadata={"error": "Goal outside grid"})
        if sdf_grid.values[start_row, start_col] <= 0:
            return PlanResult(np.empty((0, 2)), float('inf'),
                              time.perf_counter() - t0, False,
                              metadata={"error": "Start inside obstacle"})

        # Build speed field
        speed = self._sdf_to_speed(sdf_grid)
        speed_modifier = kwargs.get("speed_modifier")
        if speed_modifier is not None:
            speed = speed * speed_modifier
            speed = np.maximum(speed, 1e-10)
            speed[sdf_grid.values <= 0] = 1e-10

        # Create phi: masked array with goal as the known zero-arrival point
        phi = np.ones((ny, nx))
        phi[goal_row, goal_col] = -1.0

        # Compute travel time
        travel_time = skfmm.travel_time(phi, speed, dx=sdf_grid.spacing)

        # Handle masked/inf values
        # Handle masked/inf values - 增强版
        if isinstance(travel_time, np.ma.MaskedArray):
            # 它是掩膜数组，但掩膜可能为None
            if travel_time.mask is not None:
                tt_array = np.where(travel_time.mask, np.inf, travel_time.data)
            else:
                # 掩膜为None，表示所有点都有效，直接使用数据
                tt_array = travel_time.data
        else:
            # 它是普通数组，直接转换
            tt_array = np.asarray(travel_time)
        # if hasattr(travel_time, 'data'):
        #     tt_array = np.where(travel_time.mask, np.inf, travel_time.data)
        # else:
        #     tt_array = np.asarray(travel_time)

        # Check if start is reachable
        if np.isinf(tt_array[start_row, start_col]):
            return PlanResult(np.empty((0, 2)), float('inf'),
                              time.perf_counter() - t0, False,
                              metadata={"error": "Start not reachable from goal"})

        # Trace path from start to goal via gradient descent on travel time
        path_grid = self._trace_gradient(
            tt_array,
            np.array([start_col, start_row], dtype=float),
            np.array([goal_col, goal_row], dtype=float),
            sdf_grid,
        )

        # Convert to world coordinates
        path_world = sdf_grid.grid_to_world(path_grid)

        # Compute path cost (sum of segment lengths weighted by SDF cost)
        if len(path_world) > 1:
            diffs = np.diff(path_world, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            total_cost = tt_array[start_row, start_col]
        else:
            total_cost = float('inf')

        elapsed = time.perf_counter() - t0

        return PlanResult(
            path=path_world,
            cost=total_cost,
            time_seconds=elapsed,
            success=len(path_world) > 1,
            metadata={"travel_time_field": tt_array, "speed_field": speed},
        )
