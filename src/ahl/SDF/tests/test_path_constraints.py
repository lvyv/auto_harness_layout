"""Tests for path constraint module (attraction & repulsion)."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.sdf.from_obstacles import build_sdf_2d
from src.harness.path_constraints import (
    PathConstraint, compute_path_distance_field,
    build_cost_modifier, build_speed_modifier,
    nearest_point_on_path, path_constraint_cost_and_gradient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_sdf_grid():
    """Small SDF grid with no obstacles (all positive SDF)."""
    bounds = ((0, 10), (0, 10))
    resolution = 0.5
    return build_sdf_2d([], bounds, resolution)


def _straight_path():
    """Horizontal path from (2,5) to (8,5)."""
    return np.array([[2.0, 5.0], [8.0, 5.0]])


def _zigzag_path():
    """Multi-segment path."""
    return np.array([[1.0, 1.0], [3.0, 3.0], [5.0, 1.0], [7.0, 3.0]])


# ---------------------------------------------------------------------------
# PathConstraint validation
# ---------------------------------------------------------------------------

class TestPathConstraintValidation:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            PathConstraint(np.array([[0, 0], [1, 1]]), mode="push",
                           influence_radius=1.0, strength=1.0)

    def test_negative_radius(self):
        with pytest.raises(ValueError, match="influence_radius"):
            PathConstraint(np.array([[0, 0], [1, 1]]), mode="attract",
                           influence_radius=-0.5, strength=1.0)

    def test_zero_radius(self):
        with pytest.raises(ValueError, match="influence_radius"):
            PathConstraint(np.array([[0, 0], [1, 1]]), mode="repel",
                           influence_radius=0.0, strength=1.0)

    def test_negative_strength(self):
        with pytest.raises(ValueError, match="strength"):
            PathConstraint(np.array([[0, 0], [1, 1]]), mode="attract",
                           influence_radius=1.0, strength=-1.0)

    def test_valid_attract(self):
        c = PathConstraint(np.array([[0, 0], [1, 1]]), mode="attract",
                           influence_radius=2.0, strength=1.5)
        assert c.mode == "attract"
        assert c.influence_radius == 2.0

    def test_valid_repel(self):
        c = PathConstraint(np.array([[0, 0], [1, 1]]), mode="repel",
                           influence_radius=3.0, strength=0.0)
        assert c.mode == "repel"
        assert c.strength == 0.0


# ---------------------------------------------------------------------------
# compute_path_distance_field
# ---------------------------------------------------------------------------

class TestPathDistanceField:
    def test_shape(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        dist = compute_path_distance_field(path, grid)
        assert dist.shape == grid.shape

    def test_near_path_distance_small(self):
        grid = _simple_sdf_grid()
        path = _straight_path()  # y=5
        dist = compute_path_distance_field(path, grid)
        # Grid point at (5, 5) should be on the path -> distance ~0
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        assert dist[row, col] < grid.spacing

    def test_far_distance_increases(self):
        grid = _simple_sdf_grid()
        path = _straight_path()  # y=5
        dist = compute_path_distance_field(path, grid)
        # Points at y=5 (on path) should have smaller distance than y=1
        row_on = int(round((5.0 - grid.origin[1]) / grid.spacing))
        row_far = int(round((1.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert dist[row_on, col] < dist[row_far, col]

    def test_non_negative(self):
        grid = _simple_sdf_grid()
        dist = compute_path_distance_field(_zigzag_path(), grid)
        assert np.all(dist >= 0)


# ---------------------------------------------------------------------------
# build_cost_modifier
# ---------------------------------------------------------------------------

class TestCostModifier:
    def test_no_constraints(self):
        grid = _simple_sdf_grid()
        mod = build_cost_modifier([], grid)
        np.testing.assert_array_equal(mod, np.ones(grid.shape))

    def test_attract_lowers_cost(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="attract", influence_radius=2.0, strength=2.0)
        mod = build_cost_modifier([c], grid)
        # On the path (y=5), modifier should be < 1
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert mod[row, col] < 1.0
        # Far from path (y=0), modifier should be ~1
        row_far = int(round((0.0 - grid.origin[1]) / grid.spacing))
        assert mod[row_far, col] == pytest.approx(1.0, abs=0.01)

    def test_repel_raises_cost(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="repel", influence_radius=2.0, strength=2.0)
        mod = build_cost_modifier([c], grid)
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert mod[row, col] > 1.0

    def test_modifier_positive(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="attract", influence_radius=2.0, strength=5.0)
        mod = build_cost_modifier([c], grid)
        assert np.all(mod > 0)


# ---------------------------------------------------------------------------
# build_speed_modifier
# ---------------------------------------------------------------------------

class TestSpeedModifier:
    def test_attract_increases_speed(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="attract", influence_radius=2.0, strength=2.0)
        mod = build_speed_modifier([c], grid)
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert mod[row, col] > 1.0

    def test_repel_decreases_speed(self):
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="repel", influence_radius=2.0, strength=2.0)
        mod = build_speed_modifier([c], grid)
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert mod[row, col] < 1.0

    def test_opposite_to_cost_modifier(self):
        """Speed modifier direction is opposite to cost modifier."""
        grid = _simple_sdf_grid()
        path = _straight_path()
        c = PathConstraint(path, mode="attract", influence_radius=2.0, strength=1.5)
        cost_mod = build_cost_modifier([c], grid)
        speed_mod = build_speed_modifier([c], grid)
        # Where cost goes down, speed goes up
        row = int(round((5.0 - grid.origin[1]) / grid.spacing))
        col = int(round((5.0 - grid.origin[0]) / grid.spacing))
        assert cost_mod[row, col] < 1.0
        assert speed_mod[row, col] > 1.0


# ---------------------------------------------------------------------------
# nearest_point_on_path
# ---------------------------------------------------------------------------

class TestNearestPointOnPath:
    def test_point_on_segment(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        query = np.array([[5.0, 0.0]])
        nearest, dist = nearest_point_on_path(query, ref)
        np.testing.assert_allclose(nearest[0], [5.0, 0.0], atol=1e-10)
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_perpendicular_projection(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        query = np.array([[5.0, 3.0]])
        nearest, dist = nearest_point_on_path(query, ref)
        np.testing.assert_allclose(nearest[0], [5.0, 0.0], atol=1e-10)
        assert dist[0] == pytest.approx(3.0, abs=1e-10)

    def test_endpoint_clamp(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        query = np.array([[-5.0, 0.0]])
        nearest, dist = nearest_point_on_path(query, ref)
        np.testing.assert_allclose(nearest[0], [0.0, 0.0], atol=1e-10)
        assert dist[0] == pytest.approx(5.0, abs=1e-10)

    def test_multi_segment(self):
        ref = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]])
        query = np.array([[5.0, 2.5]])
        nearest, dist = nearest_point_on_path(query, ref)
        np.testing.assert_allclose(nearest[0], [5.0, 2.5], atol=1e-10)
        assert dist[0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# path_constraint_cost_and_gradient
# ---------------------------------------------------------------------------

class TestCostAndGradient:
    def test_attract_nonzero_cost(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        c = PathConstraint(ref, mode="attract", influence_radius=5.0, strength=2.0)
        # Query path above the reference
        query = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]])
        cost, grad = path_constraint_cost_and_gradient(query, [c])
        assert cost > 0
        # Gradient at middle point should push toward y=0 (negative y gradient)
        assert grad[1, 1] > 0  # outward direction is +y, so descent pushes toward ref

    def test_repel_nonzero_cost(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        c = PathConstraint(ref, mode="repel", influence_radius=5.0, strength=2.0)
        query = np.array([[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]])
        cost, grad = path_constraint_cost_and_gradient(query, [c])
        assert cost > 0

    def test_outside_radius_no_cost(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        c = PathConstraint(ref, mode="attract", influence_radius=1.0, strength=2.0)
        # Query path far from reference
        query = np.array([[0.0, 5.0], [5.0, 5.0], [10.0, 5.0]])
        cost, grad = path_constraint_cost_and_gradient(query, [c])
        assert cost == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(grad, 0.0, atol=1e-10)

    def test_gradient_shape(self):
        ref = np.array([[0.0, 0.0], [10.0, 0.0]])
        c = PathConstraint(ref, mode="repel", influence_radius=5.0, strength=1.0)
        query = np.linspace([0, 1], [10, 1], 20)
        cost, grad = path_constraint_cost_and_gradient(query, [c])
        assert grad.shape == query.shape


# ---------------------------------------------------------------------------
# Integration: attract makes path closer, repel makes it farther
# ---------------------------------------------------------------------------

class TestIntegration:
    def _avg_distance_to_ref(self, path, ref_path):
        _, dists = nearest_point_on_path(path, ref_path)
        return dists.mean()

    def test_attract_closer_than_baseline(self):
        """With attraction, the planned path should be closer to reference."""
        obstacles = [
            {"type": "circle", "center": [5.0, 5.0], "radius": 1.0},
        ]
        bounds = ((0, 10), (0, 10))
        resolution = 0.2
        sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

        from src.planning import FastMarchingPlanner

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])

        # Reference path at y=8
        ref_path = np.array([[1.0, 8.0], [9.0, 8.0]])

        # Baseline
        fmm = FastMarchingPlanner(safety_margin=0.5)
        baseline = fmm.plan(sdf_grid, start, goal)
        assert baseline.success

        # With attract
        c = PathConstraint(ref_path, mode="attract",
                           influence_radius=8.0, strength=3.0)
        speed_mod = build_speed_modifier([c], sdf_grid)
        attracted = fmm.plan(sdf_grid, start, goal, speed_modifier=speed_mod)
        assert attracted.success

        d_base = self._avg_distance_to_ref(baseline.path, ref_path)
        d_attr = self._avg_distance_to_ref(attracted.path, ref_path)
        assert d_attr < d_base, (
            f"Attracted path should be closer: {d_attr:.2f} vs {d_base:.2f}")

    def test_repel_farther_than_baseline(self):
        """With repulsion, the planned path should be farther from reference."""
        obstacles = [
            {"type": "circle", "center": [5.0, 5.0], "radius": 1.0},
        ]
        bounds = ((0, 10), (0, 10))
        resolution = 0.2
        sdf_grid = build_sdf_2d(obstacles, bounds, resolution)

        from src.planning import FastMarchingPlanner

        start = np.array([1.0, 2.0])
        goal = np.array([9.0, 2.0])

        # Reference path right through the baseline corridor
        ref_path = np.array([[1.0, 2.0], [9.0, 2.0]])

        # Baseline
        fmm = FastMarchingPlanner(safety_margin=0.5)
        baseline = fmm.plan(sdf_grid, start, goal)
        assert baseline.success

        # With repel
        c = PathConstraint(ref_path, mode="repel",
                           influence_radius=3.0, strength=3.0)
        speed_mod = build_speed_modifier([c], sdf_grid)
        repelled = fmm.plan(sdf_grid, start, goal, speed_modifier=speed_mod)
        assert repelled.success

        d_base = self._avg_distance_to_ref(baseline.path, ref_path)
        d_rep = self._avg_distance_to_ref(repelled.path, ref_path)
        assert d_rep > d_base, (
            f"Repelled path should be farther: {d_rep:.2f} vs {d_base:.2f}")
