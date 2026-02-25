"""A* path planning algorithm with SDF penalty."""

from typing import List, Tuple, Optional
import numpy as np
import heapq

from .cell_type import CellType
from .grid import Grid
from ..utils.validators import AStarConfig


def astar_search(
    grid: Grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    config: Optional[AStarConfig] = None
) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding with SDF penalty term.

    Implements A* with a cost function that includes SDF penalty:
        cost = movement_cost + sdf_penalty
        sdf_penalty = weight / (sdf_value + epsilon)

    This encourages paths to stay away from obstacles (engineering principle:
    encode preference in search phase, not post-processing).

    Args:
        grid: Grid instance containing the map
        start: Starting position (row, col)
        goal: Goal position (row, col)
        config: A* configuration (default: diagonal=False, sdf_weight=0.5)

    Returns:
        List of (row, col) coordinates from start to goal (inclusive),
        or None if no path exists

    Raises:
        ValueError: If start or goal is invalid or not walkable
    """
    if config is None:
        config = AStarConfig()

    # Validate start and goal
    if not grid.is_valid(*start):
        raise ValueError(f"Start position {start} is out of bounds")
    if not grid.is_valid(*goal):
        raise ValueError(f"Goal position {goal} is out of bounds")

    start_cell = grid.get_cell(*start)
    goal_cell = grid.get_cell(*goal)

    if not CellType.is_walkable(start_cell):
        raise ValueError(f"Start position {start} is not walkable (type={start_cell})")
    if not CellType.is_walkable(goal_cell):
        raise ValueError(f"Goal position {goal} is not walkable (type={goal_cell})")

    # Get SDF for penalty calculation
    sdf = grid.get_sdf()

    # Movement directions (4-connected or 8-connected)
    if config.diagonal_move:
        # 8-connected (includes diagonals)
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
        ]
        move_costs = [1.0, 1.0, 1.0, 1.0, 1.414, 1.414, 1.414, 1.414]
    else:
        # 4-connected (cardinal only)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move_costs = [1.0, 1.0, 1.0, 1.0]

    # A* data structures
    open_set = []  # Priority queue: (f_score, counter, position)
    closed_set = set()
    came_from = {}  # For path reconstruction

    g_score = {start: 0.0}  # Cost from start to node
    f_score = {start: _heuristic(start, goal)}  # Estimated total cost

    counter = 0  # Tie-breaker for heap
    heapq.heappush(open_set, (f_score[start], counter, start))

    iterations = 0

    while open_set and iterations < config.max_iterations:
        iterations += 1

        # Get node with lowest f_score
        _, _, current = heapq.heappop(open_set)

        # Goal reached
        if current == goal:
            return _reconstruct_path(came_from, current)

        closed_set.add(current)

        # Explore neighbors
        for (dr, dc), move_cost in zip(directions, move_costs):
            neighbor = (current[0] + dr, current[1] + dc)

            # Skip if out of bounds
            if not grid.is_valid(*neighbor):
                continue

            # Skip if not walkable
            if not CellType.is_walkable(grid.get_cell(*neighbor)):
                continue

            # Skip if already evaluated
            if neighbor in closed_set:
                continue

            # Calculate SDF penalty
            sdf_value = sdf[neighbor]
            sdf_penalty = config.sdf_weight / (sdf_value + config.epsilon)

            # Tentative g_score
            tentative_g = g_score[current] + move_cost + sdf_penalty

            # If this path to neighbor is better
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal)
                f_score[neighbor] = f

                # Add to open set if not already there
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))

    # No path found
    return None


def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Euclidean distance heuristic.

    Args:
        a: Position (row, col)
        b: Position (row, col)

    Returns:
        Euclidean distance
    """
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def _reconstruct_path(
    came_from: dict[Tuple[int, int], Tuple[int, int]],
    current: Tuple[int, int]
) -> List[Tuple[int, int]]:
    """Reconstruct path from came_from dictionary.

    Args:
        came_from: Dictionary mapping node -> parent node
        current: Goal node

    Returns:
        Path from start to goal (inclusive)
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def batch_astar(
    grid: Grid,
    starts: List[Tuple[int, int]],
    goals: List[Tuple[int, int]],
    config: Optional[AStarConfig] = None
) -> dict[Tuple[int, int], Optional[List[Tuple[int, int]]]]:
    """Run A* for multiple start-goal pairs.

    Args:
        grid: Grid instance
        starts: List of start positions
        goals: List of goal positions
        config: A* configuration

    Returns:
        Dictionary mapping (start, goal) -> path (or None if no path)
    """
    results = {}
    for start in starts:
        for goal in goals:
            key = (start, goal)
            try:
                path = astar_search(grid, start, goal, config)
                results[key] = path
            except ValueError:
                results[key] = None
    return results
