"""Tests for path planning module."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.sdf import build_sdf_2d
from src.planning import AStarPlanner
from src.harness.constraints import path_min_clearance


def _make_empty_grid():
    """Create an SDF with no obstacles (all positive)."""
    return build_sdf_2d([], ((0, 10), (0, 10)), 0.2)


def _make_simple_grid():
    """Create an SDF with a single obstacle in the center."""
    obstacles = [{"type": "circle", "center": [5.0, 5.0], "radius": 2.0}]
    return build_sdf_2d(obstacles, ((0, 10), (0, 10)), 0.1)


def _make_blocked_grid():
    """Create an SDF where start is fully enclosed."""
    obstacles = [
        {"type": "rectangle", "center": [1.0, 1.0], "half_extents": [1.5, 1.5]},
    ]
    return build_sdf_2d(obstacles, ((0, 10), (0, 10)), 0.1)


class TestAStarPlanner:
    def test_empty_grid_straight_line(self):
        """With no obstacles, path should be roughly a straight line."""
        grid = _make_empty_grid()
        planner = AStarPlanner(safety_margin=1.0, cost_weight=1.0)
        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        result = planner.plan(grid, start, goal)

        assert result.success
        assert len(result.path) > 2
        assert result.cost < float('inf')
        # Path should start and end near start/goal
        np.testing.assert_allclose(result.path[0], start, atol=0.3)
        np.testing.assert_allclose(result.path[-1], goal, atol=0.3)

    def test_obstacle_avoidance(self):
        """Path should avoid obstacles (positive SDF at all points)."""
        grid = _make_simple_grid()
        planner = AStarPlanner(safety_margin=0.5, cost_weight=10.0)
        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        result = planner.plan(grid, start, goal)

        assert result.success
        # All path points should have positive SDF (outside obstacles)
        sdf_vals = grid.sample(result.path)
        assert np.all(sdf_vals > -0.01), \
            f"Path enters obstacle! Min SDF: {sdf_vals.min():.3f}"

    def test_start_inside_obstacle(self):
        """Planning should fail if start is inside an obstacle."""
        grid = _make_blocked_grid()
        planner = AStarPlanner(safety_margin=0.5)
        start = np.array([1.0, 1.0])  # Inside the rectangle
        goal = np.array([8.0, 8.0])
        result = planner.plan(grid, start, goal)
        assert not result.success

    def test_start_outside_grid(self):
        """Planning should fail if start is outside grid bounds."""
        grid = _make_simple_grid()
        planner = AStarPlanner()
        result = planner.plan(grid, np.array([-5.0, -5.0]), np.array([8.0, 8.0]))
        assert not result.success

    def test_safety_margin_effect(self):
        """Higher safety margin should push path further from obstacles."""
        grid = _make_simple_grid()
        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])

        planner_low = AStarPlanner(safety_margin=0.3, cost_weight=10.0)
        planner_high = AStarPlanner(safety_margin=2.0, cost_weight=10.0)

        result_low = planner_low.plan(grid, start, goal)
        result_high = planner_high.plan(grid, start, goal)

        assert result_low.success and result_high.success

        clearance_low = path_min_clearance(result_low.path, grid)
        clearance_high = path_min_clearance(result_high.path, grid)

        # Higher margin should give better (or equal) clearance
        assert clearance_high >= clearance_low - 0.15, \
            f"Expected higher margin path to have more clearance: " \
            f"{clearance_high:.3f} vs {clearance_low:.3f}"

    def test_result_has_timing(self):
        """PlanResult should report positive timing."""
        grid = _make_simple_grid()
        planner = AStarPlanner()
        result = planner.plan(grid, np.array([1.0, 1.0]), np.array([9.0, 9.0]))
        assert result.time_seconds > 0
