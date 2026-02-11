"""Tests for SDF generation module."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.sdf.primitives import sdf_circle, sdf_rectangle, sdf_line_segment
from src.sdf.composite import sdf_union, sdf_intersection, sdf_difference
from src.sdf.grid import SDFGrid
from src.sdf.from_obstacles import build_sdf_2d


class TestCircleSDF:
    def test_on_boundary(self):
        """Points on the circle boundary should have SDF ~= 0."""
        center = np.array([5.0, 5.0])
        radius = 2.0
        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        points = center + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        sdf = sdf_circle(points, center, radius)
        np.testing.assert_allclose(sdf, 0.0, atol=1e-12)

    def test_inside_negative(self):
        """Points inside the circle should have negative SDF."""
        center = np.array([0.0, 0.0])
        radius = 3.0
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        sdf = sdf_circle(points, center, radius)
        assert np.all(sdf < 0)

    def test_outside_positive(self):
        """Points outside the circle should have positive SDF."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        points = np.array([[3.0, 0.0], [0.0, 5.0], [-2.0, -2.0]])
        sdf = sdf_circle(points, center, radius)
        assert np.all(sdf > 0)

    def test_known_distances(self):
        """Verify specific distance values."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        points = np.array([[2.0, 0.0], [0.0, 0.0]])
        sdf = sdf_circle(points, center, radius)
        np.testing.assert_allclose(sdf, [1.0, -1.0], atol=1e-12)

    def test_symmetry(self):
        """SDF should be rotationally symmetric around center."""
        center = np.array([1.0, 2.0])
        radius = 1.5
        d = 3.0
        points = np.array([
            [center[0] + d, center[1]],
            [center[0], center[1] + d],
            [center[0] - d, center[1]],
            [center[0], center[1] - d],
        ])
        sdf = sdf_circle(points, center, radius)
        np.testing.assert_allclose(sdf, sdf[0], atol=1e-12)


class TestRectangleSDF:
    def test_inside_negative(self):
        center = np.array([0.0, 0.0])
        half = np.array([2.0, 1.0])
        points = np.array([[0.0, 0.0], [1.0, 0.5], [-1.0, -0.5]])
        sdf = sdf_rectangle(points, center, half)
        assert np.all(sdf < 0)

    def test_outside_positive(self):
        center = np.array([0.0, 0.0])
        half = np.array([1.0, 1.0])
        points = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]])
        sdf = sdf_rectangle(points, center, half)
        assert np.all(sdf > 0)

    def test_corner_distance(self):
        """Distance from corner should be Euclidean distance to corner."""
        center = np.array([0.0, 0.0])
        half = np.array([1.0, 1.0])
        # Point at (2, 2), corner at (1, 1), distance = sqrt(2)
        point = np.array([[2.0, 2.0]])
        sdf = sdf_rectangle(point, center, half)
        np.testing.assert_allclose(sdf, [np.sqrt(2)], atol=1e-12)

    def test_face_distance(self):
        """Distance from face should be perpendicular distance."""
        center = np.array([0.0, 0.0])
        half = np.array([2.0, 1.0])
        # Point at (0, 3), nearest face at y=1, distance = 2
        point = np.array([[0.0, 3.0]])
        sdf = sdf_rectangle(point, center, half)
        np.testing.assert_allclose(sdf, [2.0], atol=1e-12)

    def test_on_boundary(self):
        center = np.array([0.0, 0.0])
        half = np.array([1.0, 1.0])
        points = np.array([[1.0, 0.5], [-1.0, 0.0], [0.5, 1.0]])
        sdf = sdf_rectangle(points, center, half)
        np.testing.assert_allclose(sdf, [0.0, 0.0, 0.0], atol=1e-12)


class TestComposite:
    def test_union_takes_min(self):
        a = np.array([1.0, 2.0, -1.0])
        b = np.array([0.5, 3.0, -0.5])
        result = sdf_union(a, b)
        np.testing.assert_allclose(result, [0.5, 2.0, -1.0])

    def test_intersection_takes_max(self):
        a = np.array([1.0, 2.0, -1.0])
        b = np.array([0.5, 3.0, -0.5])
        result = sdf_intersection(a, b)
        np.testing.assert_allclose(result, [1.0, 3.0, -0.5])

    def test_difference(self):
        a = np.array([-1.0, -2.0, 1.0])  # inside A
        b = np.array([-0.5, 1.0, -0.5])  # inside/outside B
        result = sdf_difference(a, b)
        # max(a, -b): max(-1, 0.5)=0.5, max(-2, -1)=-1, max(1, 0.5)=1
        np.testing.assert_allclose(result, [0.5, -1.0, 1.0])

    def test_union_multiple(self):
        a = np.array([3.0, 1.0])
        b = np.array([2.0, 4.0])
        c = np.array([1.0, 5.0])
        result = sdf_union(a, b, c)
        np.testing.assert_allclose(result, [1.0, 1.0])


class TestSDFGrid:
    def setup_method(self):
        # Simple 5x5 grid with known SDF values
        values = np.array([
            [2.0, 1.5, 1.0, 1.5, 2.0],
            [1.5, 0.7, 0.0, 0.7, 1.5],
            [1.0, 0.0, -1.0, 0.0, 1.0],
            [1.5, 0.7, 0.0, 0.7, 1.5],
            [2.0, 1.5, 1.0, 1.5, 2.0],
        ], dtype=float)
        self.grid = SDFGrid(values=values, origin=np.array([0.0, 0.0]),
                            spacing=1.0)

    def test_shape(self):
        assert self.grid.shape == (5, 5)
        assert self.grid.nx == 5
        assert self.grid.ny == 5

    def test_bounds(self):
        (xmin, xmax), (ymin, ymax) = self.grid.bounds
        assert xmin == 0.0
        assert xmax == 4.0
        assert ymin == 0.0
        assert ymax == 4.0

    def test_world_grid_roundtrip(self):
        """world_to_grid and grid_to_world should be inverses."""
        world_pt = np.array([2.5, 3.5])
        grid_pt = self.grid.world_to_grid(world_pt)
        recovered = self.grid.grid_to_world(grid_pt)
        np.testing.assert_allclose(recovered, world_pt, atol=1e-12)

    def test_sample_at_grid_points(self):
        """Sampling at exact grid points should return exact values."""
        # Grid point (2, 2) in col, row => world (2, 2)
        val = self.grid.sample(np.array([[2.0, 2.0]]))
        np.testing.assert_allclose(val, [-1.0], atol=1e-6)

    def test_cost_field_inside_obstacle(self):
        cost = self.grid.cost_field(safety_margin=1.5)
        # Center (row=2, col=2) has SDF=-1.0, should be INF
        assert np.isinf(cost[2, 2])

    def test_cost_field_free_space(self):
        cost = self.grid.cost_field(safety_margin=1.0)
        # Corner (row=0, col=0) has SDF=2.0, should be 1.0
        assert cost[0, 0] == 1.0

    def test_cost_field_near_obstacle(self):
        cost = self.grid.cost_field(safety_margin=1.5, alpha=10.0)
        # (row=0, col=2) has SDF=1.0, within margin=1.5
        assert cost[0, 2] > 1.0
        assert not np.isinf(cost[0, 2])

    def test_gradient_shape(self):
        grad = self.grid.gradient()
        assert grad.shape == (2, 5, 5)


class TestBuildSDF2D:
    def test_single_circle(self):
        obstacles = [{"type": "circle", "center": [5.0, 5.0], "radius": 1.0}]
        grid = build_sdf_2d(obstacles, ((0, 10), (0, 10)), 0.1)
        assert grid.shape[0] > 0
        assert grid.shape[1] > 0
        # Center should be inside (negative)
        center_val = grid.sample(np.array([[5.0, 5.0]]))
        assert center_val[0] < 0
        # Far corner should be outside (positive)
        corner_val = grid.sample(np.array([[0.0, 0.0]]))
        assert corner_val[0] > 0

    def test_no_obstacles(self):
        grid = build_sdf_2d([], ((0, 10), (0, 10)), 0.5)
        assert np.all(np.isinf(grid.values))

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown obstacle type"):
            build_sdf_2d([{"type": "triangle"}], ((0, 10), (0, 10)), 0.5)

    def test_multiple_obstacles_union(self):
        obstacles = [
            {"type": "circle", "center": [3.0, 3.0], "radius": 1.0},
            {"type": "circle", "center": [7.0, 7.0], "radius": 1.0},
        ]
        grid = build_sdf_2d(obstacles, ((0, 10), (0, 10)), 0.1)
        # Both centers should be negative
        vals = grid.sample(np.array([[3.0, 3.0], [7.0, 7.0]]))
        assert np.all(vals < 0)
        # Midpoint between them should be positive (they don't overlap)
        mid_val = grid.sample(np.array([[5.0, 5.0]]))
        assert mid_val[0] > 0
