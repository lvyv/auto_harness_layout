"""path_ops 单元测试。"""

import numpy as np
import pytest
import math

from ahl.routing.graph.path_ops import (
    path_length, simplify_path, smooth_path,
    count_turns, find_shared_segments, path_edges,
)


class TestPathLength:
    def test_empty(self):
        assert path_length([]) == 0.0

    def test_single_point(self):
        assert path_length([(0, 0, 0)]) == 0.0

    def test_straight_line(self):
        path = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        assert abs(path_length(path) - 3.0) < 1e-10

    def test_diagonal(self):
        path = [(0, 0, 0), (1, 1, 1)]
        assert abs(path_length(path) - math.sqrt(3)) < 1e-10


class TestSimplifyPath:
    def test_straight_line_simplified(self):
        """直线上的中间点应被简化掉。"""
        path = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
        result = simplify_path(path, epsilon=0.1)
        # 直线只保留首尾
        assert len(result) == 2
        assert result[0] == (0, 0, 0)
        assert result[-1] == (4, 0, 0)

    def test_right_angle_preserved(self):
        """直角拐弯点不应被简化掉。"""
        path = [(0, 0, 0), (5, 0, 0), (5, 5, 0)]
        result = simplify_path(path, epsilon=0.1)
        assert len(result) == 3  # 所有点保留

    def test_small_path_unchanged(self):
        path = [(0, 0, 0), (1, 1, 1)]
        result = simplify_path(path, epsilon=0.1)
        assert result == path

    def test_large_epsilon_simplifies_more(self):
        """大容差应简化更多点。"""
        path = [
            (0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 0),
            (4, 0, 0), (5, 1, 0), (6, 0, 0),
        ]
        r_small = simplify_path(path, epsilon=0.1)
        r_large = simplify_path(path, epsilon=2.0)
        assert len(r_large) <= len(r_small)


class TestSmoothPath:
    def test_preserves_endpoints(self):
        path = [(0, 0, 0), (5, 0, 0), (5, 5, 0), (10, 5, 0)]
        result = smooth_path(path, window=3, keep_endpoints=True)
        assert result[0] == (0.0, 0.0, 0.0)
        assert result[-1] == (10.0, 5.0, 0.0)

    def test_short_path_unchanged(self):
        path = [(0, 0, 0), (1, 1, 1)]
        result = smooth_path(path)
        assert len(result) == 2

    def test_returns_floats(self):
        path = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        result = smooth_path(path)
        assert all(isinstance(c, float) for p in result for c in p)


class TestCountTurns:
    def test_straight_no_turns(self):
        path = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        assert count_turns(path) == 0

    def test_one_right_angle(self):
        path = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
        assert count_turns(path) == 1

    def test_zigzag(self):
        path = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0), (2, 2, 0)]
        assert count_turns(path) == 3

    def test_short_path(self):
        assert count_turns([(0, 0, 0)]) == 0
        assert count_turns([(0, 0, 0), (1, 0, 0)]) == 0


class TestFindSharedSegments:
    def test_fully_shared(self):
        path_a = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        path_b = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        segs = find_shared_segments(path_a, path_b)
        assert len(segs) == 1
        assert len(segs[0]) == 3

    def test_partial_shared(self):
        path_a = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)]
        path_b = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]
        segs = find_shared_segments(path_a, path_b)
        assert len(segs) == 1
        assert segs[0] == [(1, 0, 0), (2, 0, 0), (3, 0, 0)]

    def test_no_shared(self):
        path_a = [(0, 0, 0), (1, 0, 0)]
        path_b = [(5, 5, 5), (6, 6, 6)]
        segs = find_shared_segments(path_a, path_b)
        assert len(segs) == 0

    def test_single_point_not_segment(self):
        """单个共享点不算路段（需要 ≥ 2 个连续点）。"""
        path_a = [(0, 0, 0), (1, 0, 0), (5, 5, 5)]
        path_b = [(1, 0, 0), (2, 0, 0)]
        segs = find_shared_segments(path_a, path_b)
        assert len(segs) == 0


class TestPathEdges:
    def test_basic(self):
        path = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        edges = path_edges(path)
        assert len(edges) == 2
        assert frozenset({(0, 0, 0), (1, 0, 0)}) in edges
        assert frozenset({(1, 0, 0), (2, 0, 0)}) in edges

    def test_empty_path(self):
        assert len(path_edges([])) == 0
        assert len(path_edges([(0, 0, 0)])) == 0
