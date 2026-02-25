"""math_utils 单元测试。"""

import numpy as np
import pytest
import math

from ahl.utils.math_utils import (
    normalize, euclidean_distance, manhattan_distance,
    chebyshev_distance, angle_between, cross_product,
    lerp, bounding_box,
)


class TestNormalize:
    def test_unit_vector(self):
        v = np.array([3.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])

    def test_general_vector(self):
        v = np.array([1.0, 1.0, 1.0])
        result = normalize(v)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10

    def test_zero_vector(self):
        v = np.array([0.0, 0.0, 0.0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.0])

    def test_batch(self):
        vs = np.array([[3.0, 0, 0], [0, 4.0, 0]])
        result = normalize(vs)
        np.testing.assert_array_almost_equal(result[0], [1, 0, 0])
        np.testing.assert_array_almost_equal(result[1], [0, 1, 0])


class TestDistances:
    def test_euclidean(self):
        d = euclidean_distance((0, 0, 0), (3, 4, 0))
        assert abs(d - 5.0) < 1e-10

    def test_manhattan(self):
        d = manhattan_distance((0, 0, 0), (3, 4, 5))
        assert abs(d - 12.0) < 1e-10

    def test_chebyshev(self):
        d = chebyshev_distance((0, 0, 0), (3, 4, 5))
        assert abs(d - 5.0) < 1e-10


class TestAngle:
    def test_parallel(self):
        a = angle_between([1, 0, 0], [2, 0, 0])
        assert abs(a) < 1e-10

    def test_perpendicular(self):
        a = angle_between([1, 0, 0], [0, 1, 0])
        assert abs(a - math.pi / 2) < 1e-10

    def test_opposite(self):
        a = angle_between([1, 0, 0], [-1, 0, 0])
        assert abs(a - math.pi) < 1e-10

    def test_zero_vector(self):
        a = angle_between([0, 0, 0], [1, 0, 0])
        assert a == 0.0


class TestCrossProduct:
    def test_basic(self):
        result = cross_product([1, 0, 0], [0, 1, 0])
        np.testing.assert_array_almost_equal(result, [0, 0, 1])


class TestLerp:
    def test_endpoints(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(lerp(a, b, 0.0), a)
        np.testing.assert_array_almost_equal(lerp(a, b, 1.0), b)

    def test_midpoint(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(lerp(a, b, 0.5), [5, 5, 5])


class TestBoundingBox:
    def test_basic(self):
        pts = np.array([[1, 2, 3], [4, 5, 6], [0, 1, 2]])
        bmin, bmax = bounding_box(pts)
        np.testing.assert_array_almost_equal(bmin, [0, 1, 2])
        np.testing.assert_array_almost_equal(bmax, [4, 5, 6])
