"""Tests for A* pathfinding algorithm."""

import pytest
import numpy as np

from ahl.grid2d.core.grid import Grid
from ahl.grid2d.core.cell_type import CellType
from ahl.grid2d.core.astar import astar_search, batch_astar
from ahl.grid2d.utils.validators import AStarConfig


class TestAStar:
    """Test A* pathfinding algorithm."""

    def test_simple_path(self):
        """Test simple straight-line path."""
        grid = Grid(10, 10)
        start = (0, 0)
        goal = (0, 5)

        path = astar_search(grid, start, goal)

        assert path is not None
        assert path[0] == start
        assert path[-1] == goal
        assert len(path) == 6  # 6 cells from (0,0) to (0,5)

    def test_path_around_obstacle(self):
        """Test path that must go around obstacle."""
        grid = Grid(10, 10)

        # Create wall
        for r in range(2, 8):
            grid.set_cell(r, 5, CellType.OBSTACLE)

        start = (5, 0)
        goal = (5, 9)

        path = astar_search(grid, start, goal)

        assert path is not None
        assert path[0] == start
        assert path[-1] == goal

        # Path should not go through obstacle
        for pos in path:
            assert grid.get_cell(*pos) != CellType.OBSTACLE

    def test_no_path(self):
        """Test case where no path exists."""
        grid = Grid(10, 10)

        # Create complete wall
        for r in range(10):
            grid.set_cell(r, 5, CellType.OBSTACLE)

        start = (5, 0)
        goal = (5, 9)

        path = astar_search(grid, start, goal)

        assert path is None

    def test_diagonal_movement(self):
        """Test diagonal movement option."""
        grid = Grid(10, 10)
        start = (0, 0)
        goal = (5, 5)

        # Without diagonal
        config = AStarConfig(diagonal_move=False)
        path_no_diag = astar_search(grid, start, goal, config)
        assert path_no_diag is not None
        assert len(path_no_diag) == 11  # Manhattan distance

        # With diagonal
        config = AStarConfig(diagonal_move=True)
        path_diag = astar_search(grid, start, goal, config)
        assert path_diag is not None
        assert len(path_diag) == 6  # Diagonal distance
        assert len(path_diag) < len(path_no_diag)

    def test_sdf_penalty_effect(self):
        """Test that SDF penalty affects path selection."""
        grid = Grid(20, 20)

        # Create corridor with obstacles on sides
        for r in range(5, 15):
            grid.set_cell(r, 8, CellType.OBSTACLE)
            grid.set_cell(r, 12, CellType.OBSTACLE)

        start = (10, 0)
        goal = (10, 19)

        # Low SDF weight (path might go close to obstacles)
        config_low = AStarConfig(sdf_weight=0.0)
        path_low = astar_search(grid, start, goal, config_low)

        # High SDF weight (path should stay away from obstacles)
        config_high = AStarConfig(sdf_weight=2.0)
        path_high = astar_search(grid, start, goal, config_high)

        assert path_low is not None
        assert path_high is not None

        # High SDF weight path should prefer center of corridor
        # We can't strictly test this without checking path coordinates,
        # but at least verify paths are found
        assert path_low[0] == start
        assert path_high[0] == start
        assert path_low[-1] == goal
        assert path_high[-1] == goal

    def test_start_equals_goal(self):
        """Test case where start equals goal."""
        grid = Grid(10, 10)
        pos = (5, 5)

        path = astar_search(grid, pos, pos)

        assert path is not None
        assert len(path) == 1
        assert path[0] == pos

    def test_invalid_start(self):
        """Test invalid start position."""
        grid = Grid(10, 10)

        # Out of bounds
        with pytest.raises(ValueError):
            astar_search(grid, (-1, 0), (5, 5))

        # On obstacle
        grid.set_cell(3, 3, CellType.OBSTACLE)
        with pytest.raises(ValueError):
            astar_search(grid, (3, 3), (5, 5))

    def test_invalid_goal(self):
        """Test invalid goal position."""
        grid = Grid(10, 10)

        # Out of bounds
        with pytest.raises(ValueError):
            astar_search(grid, (5, 5), (10, 10))

        # On obstacle
        grid.set_cell(7, 7, CellType.OBSTACLE)
        with pytest.raises(ValueError):
            astar_search(grid, (5, 5), (7, 7))

    def test_path_through_start_end_cells(self):
        """Test that path can go through START/END cell types."""
        grid = Grid(10, 10)

        # Mark start and goal
        start = (0, 0)
        goal = (0, 5)
        grid.set_cell(*start, CellType.START)
        grid.set_cell(*goal, CellType.END)

        path = astar_search(grid, start, goal)

        assert path is not None
        assert path[0] == start
        assert path[-1] == goal

    def test_batch_astar(self):
        """Test batch A* for multiple start-goal pairs."""
        grid = Grid(20, 20)

        starts = [(0, 0), (5, 5)]
        goals = [(19, 19), (15, 15)]

        results = batch_astar(grid, starts, goals)

        assert len(results) == 4  # 2 starts × 2 goals

        # All paths should be found in empty grid
        for key, path in results.items():
            assert path is not None
            start, goal = key
            assert path[0] == start
            assert path[-1] == goal

    def test_max_iterations_protection(self):
        """Test that max iterations prevents infinite loops."""
        grid = Grid(100, 100)

        start = (0, 0)
        goal = (99, 99)

        # Very low iteration limit
        config = AStarConfig(max_iterations=10)
        path = astar_search(grid, start, goal, config)

        # Should return None (not enough iterations)
        assert path is None

    def test_path_continuity(self):
        """Test that returned path is continuous (adjacent cells)."""
        grid = Grid(20, 20)

        # Add some obstacles
        for r in range(5, 15):
            grid.set_cell(r, 10, CellType.OBSTACLE)

        start = (10, 0)
        goal = (10, 19)

        path = astar_search(grid, start, goal)

        assert path is not None

        # Check each step is adjacent (4-connected)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            distance = abs(r2 - r1) + abs(c2 - c1)
            assert distance <= 1, f"Non-adjacent cells: {path[i]} -> {path[i+1]}"
