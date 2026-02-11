"""A* path planner on SDF-derived cost grid."""

import heapq
import time

import numpy as np

from src.sdf.grid import SDFGrid
from .base import BasePlanner, PlanResult

# 8-connected neighbor offsets: (drow, dcol, distance_factor)
_NEIGHBORS_8 = [
    (-1, -1, np.sqrt(2)),
    (-1,  0, 1.0),
    (-1,  1, np.sqrt(2)),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    ( 1, -1, np.sqrt(2)),
    ( 1,  0, 1.0),
    ( 1,  1, np.sqrt(2)),
]


class AStarPlanner(BasePlanner):
    """A* planner using SDF-derived cost field.

    The SDF is converted to a traversal cost field where cells near obstacles
    have higher cost, and cells inside obstacles are impassable.
    """

    def __init__(self, safety_margin: float = 2.0, cost_weight: float = 10.0):
        """
        Args:
            safety_margin: Distance (in world units) within which the
                proximity penalty is active.
            cost_weight: Multiplier for the proximity penalty (alpha).
        """
        self.safety_margin = safety_margin
        self.cost_weight = cost_weight

    def plan(self, sdf_grid: SDFGrid, start: np.ndarray,
             goal: np.ndarray, **kwargs) -> PlanResult:
        t0 = time.perf_counter()

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Build cost grid
        cost_grid = sdf_grid.cost_field(self.safety_margin, self.cost_weight)
        cost_modifier = kwargs.get("cost_modifier")
        if cost_modifier is not None:
            cost_grid = cost_grid * cost_modifier
        ny, nx = cost_grid.shape

        # Convert start/goal to grid indices (col, row)
        start_grid = sdf_grid.world_to_grid(start)
        goal_grid = sdf_grid.world_to_grid(goal)

        start_col, start_row = int(round(start_grid[0])), int(round(start_grid[1]))
        goal_col, goal_row = int(round(goal_grid[0])), int(round(goal_grid[1]))

        # Validate bounds
        if not (0 <= start_row < ny and 0 <= start_col < nx):
            return PlanResult(path=np.empty((0, 2)), cost=float('inf'),
                              time_seconds=time.perf_counter() - t0, success=False,
                              metadata={"error": "Start outside grid"})
        if not (0 <= goal_row < ny and 0 <= goal_col < nx):
            return PlanResult(path=np.empty((0, 2)), cost=float('inf'),
                              time_seconds=time.perf_counter() - t0, success=False,
                              metadata={"error": "Goal outside grid"})
        if np.isinf(cost_grid[start_row, start_col]):
            return PlanResult(path=np.empty((0, 2)), cost=float('inf'),
                              time_seconds=time.perf_counter() - t0, success=False,
                              metadata={"error": "Start inside obstacle"})
        if np.isinf(cost_grid[goal_row, goal_col]):
            return PlanResult(path=np.empty((0, 2)), cost=float('inf'),
                              time_seconds=time.perf_counter() - t0, success=False,
                              metadata={"error": "Goal inside obstacle"})

        # A* search
        g_score = np.full((ny, nx), np.inf)
        g_score[start_row, start_col] = 0.0
        came_from = {}

        def heuristic(row, col):
            return np.sqrt((row - goal_row) ** 2 + (col - goal_col) ** 2) * sdf_grid.spacing

        # Priority queue: (f_score, counter, row, col)
        counter = 0
        open_set = [(heuristic(start_row, start_col), counter, start_row, start_col)]
        visited = np.zeros((ny, nx), dtype=bool)
        iterations = 0

        while open_set:
            f, _, cr, cc = heapq.heappop(open_set)
            if visited[cr, cc]:
                continue
            visited[cr, cc] = True
            iterations += 1

            if cr == goal_row and cc == goal_col:
                # Reconstruct path
                path_indices = []
                r, c = goal_row, goal_col
                while (r, c) in came_from:
                    path_indices.append((c, r))  # (col, row) = (x_idx, y_idx)
                    r, c = came_from[(r, c)]
                path_indices.append((start_col, start_row))
                path_indices.reverse()

                # Convert to world coordinates
                path_grid = np.array(path_indices, dtype=float)
                path_world = sdf_grid.grid_to_world(path_grid)

                return PlanResult(
                    path=path_world,
                    cost=g_score[goal_row, goal_col],
                    time_seconds=time.perf_counter() - t0,
                    success=True,
                    iterations=iterations,
                )

            for dr, dc, dist_factor in _NEIGHBORS_8:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < ny and 0 <= nc < nx):
                    continue
                if visited[nr, nc]:
                    continue
                if np.isinf(cost_grid[nr, nc]):
                    continue

                # Movement cost: distance * average cell cost
                move_cost = (dist_factor * sdf_grid.spacing *
                             0.5 * (cost_grid[cr, cc] + cost_grid[nr, nc]))
                tentative_g = g_score[cr, cc] + move_cost

                if tentative_g < g_score[nr, nc]:
                    g_score[nr, nc] = tentative_g
                    came_from[(nr, nc)] = (cr, cc)
                    f_new = tentative_g + heuristic(nr, nc)
                    counter += 1
                    heapq.heappush(open_set, (f_new, counter, nr, nc))

        # No path found
        return PlanResult(
            path=np.empty((0, 2)),
            cost=float('inf'),
            time_seconds=time.perf_counter() - t0,
            success=False,
            iterations=iterations,
            metadata={"error": "No path found"},
        )
