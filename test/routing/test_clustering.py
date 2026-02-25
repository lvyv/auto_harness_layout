"""clustering 单元测试。"""

import numpy as np
import pytest

from ahl.geometry.voxel.grid3d import Grid3D
from ahl.routing.network.clustering import TerminalClusterer, Cluster


class TestTerminalClusterer:
    """终端聚类测试。"""

    def test_basic_clustering(self):
        """基础 K-means 聚类。"""
        terminals = [
            (0, 0, 0), (1, 0, 0), (1, 1, 0),  # 聚类 A
            (10, 10, 0), (11, 10, 0), (10, 11, 0),  # 聚类 B
        ]
        clusterer = TerminalClusterer()
        clusters = clusterer.cluster(terminals, n_clusters=2, seed=42)

        assert len(clusters) == 2

        # 每个聚类应有成员
        for c in clusters:
            assert len(c.members) > 0
            assert c.center is not None
            assert c.center_voxel is not None

        # 所有终端都应被分配
        all_members = []
        for c in clusters:
            all_members.extend(c.members)
        assert len(all_members) == 6

    def test_n_equals_terminals(self):
        """聚类数 = 终端数，每个终端一个聚类。"""
        terminals = [(0, 0, 0), (5, 5, 5), (10, 10, 10)]
        clusterer = TerminalClusterer()
        clusters = clusterer.cluster(terminals, n_clusters=3)

        assert len(clusters) == 3
        for c in clusters:
            assert len(c.members) == 1

    def test_with_grid_snap(self):
        """带 Grid3D 时聚类中心应对齐到自由体素。"""
        grid = Grid3D(20, 20, 1)
        terminals = [
            (2, 2, 0), (3, 3, 0),
            (15, 15, 0), (16, 16, 0),
        ]
        clusterer = TerminalClusterer(grid)
        clusters = clusterer.cluster(terminals, n_clusters=2, seed=42)

        for c in clusters:
            i, j, k = c.center_voxel
            assert grid.is_free(i, j, k)

    def test_invalid_n_clusters(self):
        """非法聚类数。"""
        clusterer = TerminalClusterer()
        with pytest.raises(ValueError):
            clusterer.cluster([(0, 0, 0)], n_clusters=0)

    def test_too_many_clusters(self):
        """聚类数超过终端数。"""
        clusterer = TerminalClusterer()
        with pytest.raises(ValueError):
            clusterer.cluster([(0, 0, 0), (1, 1, 1)], n_clusters=5)

    def test_cluster_labels(self):
        """每个聚类有不同的 label。"""
        terminals = [(i, 0, 0) for i in range(10)]
        clusterer = TerminalClusterer()
        clusters = clusterer.cluster(terminals, n_clusters=3, seed=42)
        labels = [c.label for c in clusters]
        assert len(set(labels)) == 3


class TestKDTree:
    """KDTree 测试。"""

    def test_build_and_query(self):
        points = [(0, 0, 0), (10, 0, 0), (0, 10, 0)]
        tree = TerminalClusterer.build_kdtree(points)

        dist, idx = TerminalClusterer.query_nearest(tree, (1, 0, 0))
        assert idx == 0  # 最近的是 (0,0,0)

    def test_query_k_nearest(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (10, 10, 10)]
        tree = TerminalClusterer.build_kdtree(points)

        dists, indices = TerminalClusterer.query_nearest(tree, (0, 0, 0), k=2)
        assert len(indices) == 2
        assert 0 in indices  # (0,0,0) 自身
        assert 1 in indices  # (1,0,0)
