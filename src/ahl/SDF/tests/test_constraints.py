"""Tests for wire harness constraint module."""

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

from src.harness.constraints import (
    compute_curvature, compute_bend_radius, check_bend_radius,
    path_length,
)
from src.harness.cable import Cable, HarnessSpec
from src.harness.smoothing import smooth_path_bspline


class TestCurvature:
    def test_straight_line_zero_curvature(self):
        """A straight line should have zero curvature everywhere."""
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
        curvatures = compute_curvature(path)
        np.testing.assert_allclose(curvatures, 0.0, atol=1e-12)

    def test_known_circle_curvature(self):
        """A semicircle with radius R should have curvature ~1/R."""
        R = 5.0
        angles = np.linspace(0, np.pi, 100)
        path = R * np.column_stack([np.cos(angles), np.sin(angles)])
        curvatures = compute_curvature(path)
        expected = 1.0 / R
        # Allow some error due to discrete approximation
        np.testing.assert_allclose(curvatures, expected, rtol=0.05)

    def test_sharp_corner_high_curvature(self):
        """A sharp 90-degree turn should have high curvature."""
        path = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
        curvatures = compute_curvature(path)
        assert curvatures[0] > 0.5  # Should be high

    def test_less_than_3_points(self):
        """Paths with fewer than 3 points should return empty curvature."""
        path = np.array([[0, 0], [1, 1]], dtype=float)
        curvatures = compute_curvature(path)
        assert len(curvatures) == 0


class TestBendRadius:
    def test_straight_line_infinite_radius(self):
        path = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
        radii = compute_bend_radius(path)
        assert np.all(np.isinf(radii))

    def test_known_circle_radius(self):
        R = 3.0
        angles = np.linspace(0, np.pi, 200)
        path = R * np.column_stack([np.cos(angles), np.sin(angles)])
        radii = compute_bend_radius(path)
        np.testing.assert_allclose(radii, R, rtol=0.05)

    def test_check_bend_radius_no_violations(self):
        """Gentle curve should have no violations with small min radius."""
        R = 5.0
        angles = np.linspace(0, np.pi / 4, 50)
        path = R * np.column_stack([np.cos(angles), np.sin(angles)])
        violations = check_bend_radius(path, min_radius=1.0)
        assert len(violations) == 0

    def test_check_bend_radius_with_violations(self):
        """Sharp zigzag should have bend radius violations."""
        path = np.array([
            [0, 0], [1, 0], [1.1, 1], [2, 0], [3, 0]
        ], dtype=float)
        violations = check_bend_radius(path, min_radius=2.0)
        assert len(violations) > 0
        for v in violations:
            assert v["radius"] < v["required"]
            assert v["severity"] > 1.0


class TestPathLength:
    def test_straight_line(self):
        path = np.array([[0, 0], [3, 4]], dtype=float)
        assert path_length(path) == pytest.approx(5.0)

    def test_multi_segment(self):
        path = np.array([[0, 0], [1, 0], [1, 1]], dtype=float)
        assert path_length(path) == pytest.approx(2.0)

    def test_empty_path(self):
        assert path_length(np.empty((0, 2))) == 0.0

    def test_single_point(self):
        assert path_length(np.array([[1, 2]])) == 0.0


class TestCable:
    def test_automotive_cable(self):
        cable = Cable.automotive("test", diameter_mm=5.0, bend_factor=4.0)
        assert cable.diameter == pytest.approx(0.005)
        assert cable.min_bend_radius == pytest.approx(0.020)

    def test_from_awg(self):
        cable = Cable.from_awg(16)
        assert cable.name == "AWG16"
        assert cable.diameter > 0
        assert cable.min_bend_radius > cable.diameter

    def test_awg_invalid(self):
        with pytest.raises(ValueError):
            Cable.from_awg(99)


class TestHarnessSpec:
    def test_bundle_diameter_single(self):
        cable = Cable("single", 0.005, 0.02)
        spec = HarnessSpec([cable], start=[0, 0], goal=[10, 10])
        assert spec.bundle_diameter == 0.005

    def test_min_bend_radius_takes_max(self):
        c1 = Cable("a", 0.005, 0.02)
        c2 = Cable("b", 0.003, 0.05)  # More restrictive
        spec = HarnessSpec([c1, c2], start=[0, 0], goal=[10, 10])
        assert spec.min_bend_radius == 0.05


class TestSmoothing:
    def test_preserves_endpoints(self):
        path = np.array([
            [0, 0], [1, 0.5], [2, 0], [3, 0.5], [4, 0], [5, 0]
        ], dtype=float)
        smoothed = smooth_path_bspline(path, num_output_points=50)
        np.testing.assert_allclose(smoothed[0], path[0], atol=1e-6)
        np.testing.assert_allclose(smoothed[-1], path[-1], atol=1e-6)

    def test_reduces_curvature(self):
        """Smoothed path should have lower max curvature than zigzag."""
        path = np.array([
            [0, 0], [1, 1], [2, -1], [3, 1], [4, -1], [5, 0]
        ], dtype=float)
        smoothed = smooth_path_bspline(path, num_output_points=100)

        curvatures_raw = compute_curvature(path)
        curvatures_smooth = compute_curvature(smoothed)

        assert curvatures_smooth.max() <= curvatures_raw.max() + 0.1

    def test_short_path_unchanged(self):
        """Paths shorter than degree+1 should be returned unchanged."""
        path = np.array([[0, 0], [1, 1]], dtype=float)
        smoothed = smooth_path_bspline(path)
        np.testing.assert_array_equal(smoothed, path)
