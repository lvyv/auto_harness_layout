"""graph_builder 单元测试。"""

import numpy as np
import pytest
import networkx as nx

from ahl.geometry.voxel.grid3d import Grid3D, CellType
from ahl.routing.graph.graph_builder import GridGraphBuilder
from ahl.routing.single.cost_function import CostFunction


class TestGridGraphBuilder:
    """图构建测试。"""

    def test_tiny_grid(self):
        """3x3x1 无障碍，6-连通。"""
        grid = Grid3D(3, 3, 1)
        builder = GridGraphBuilder(grid, connectivity=6)
        G = builder.build()

        # 9 个节点
        assert G.number_of_nodes() == 9
        # 6-连通 3x3 单层：水平边 2*3 + 垂直边 3*2 = 12
        assert G.number_of_edges() == 12

    def test_26_connectivity_more_edges(self):
        """26-连通应比 6-连通有更多边。"""
        grid = Grid3D(3, 3, 3)
        G6 = GridGraphBuilder(grid, connectivity=6).build()
        G26 = GridGraphBuilder(grid, connectivity=26).build()
        assert G26.number_of_edges() > G6.number_of_edges()

    def test_obstacles_reduce_nodes(self):
        """障碍物减少图节点数。"""
        grid = Grid3D(5, 5, 1)
        grid.set_cell(2, 2, 0, CellType.OBSTACLE)
        G = GridGraphBuilder(grid, connectivity=6).build()
        assert G.number_of_nodes() == 24  # 25 - 1

    def test_roi_subgraph(self):
        """ROI 限制只构建子区域的图。"""
        grid = Grid3D(10, 10, 10)
        builder = GridGraphBuilder(grid, connectivity=6)
        G = builder.build(roi_min=(0, 0, 0), roi_max=(2, 2, 2))
        # 3x3x3 = 27 nodes
        assert G.number_of_nodes() == 27

    def test_edge_weights_positive(self):
        """所有边权重应为正。"""
        grid = Grid3D(5, 5, 1)
        G = GridGraphBuilder(grid, connectivity=6).build()
        for u, v, d in G.edges(data=True):
            assert d['weight'] > 0

    def test_custom_cost_function(self):
        """自定义代价函数影响边权。"""
        grid = Grid3D(3, 3, 1)

        # 纯距离
        G1 = GridGraphBuilder(
            grid, connectivity=6,
            cost_fn=CostFunction(w_dist=1.0, w_sdf=0.0),
        ).build()

        # 加 SDF 惩罚
        grid.set_cell(0, 0, 0, CellType.OBSTACLE)  # 制造 SDF 梯度
        sdf = grid.get_sdf()
        G2 = GridGraphBuilder(
            grid, connectivity=6,
            cost_fn=CostFunction(w_dist=1.0, w_sdf=1.0, sdf=sdf),
        ).build()

        # 靠近障碍的边在 G2 中应该更贵
        # (1,0,0)→(1,1,0) 在两个图中都存在
        if G1.has_edge((1, 0, 0), (1, 1, 0)) and G2.has_edge((1, 0, 0), (1, 1, 0)):
            w1 = G1[(1, 0, 0)][(1, 1, 0)]['weight']
            w2 = G2[(1, 0, 0)][(1, 1, 0)]['weight']
            assert w2 >= w1

    def test_build_from_nodes(self):
        """从指定节点集构建子图。"""
        grid = Grid3D(10, 10, 1)
        nodes = {(0, 0, 0), (1, 0, 0), (0, 1, 0), (5, 5, 0)}
        builder = GridGraphBuilder(grid, connectivity=6)
        G = builder.build_from_nodes(nodes)
        assert G.number_of_nodes() == 4
        # (0,0,0)-(1,0,0) 和 (0,0,0)-(0,1,0) 是邻居
        assert G.has_edge((0, 0, 0), (1, 0, 0))
        assert G.has_edge((0, 0, 0), (0, 1, 0))
        # (5,5,0) 不与其他节点相邻
        assert G.degree((5, 5, 0)) == 0

    def test_networkx_shortest_path(self):
        """构建的图支持 networkx 最短路查询。"""
        grid = Grid3D(10, 10, 1)
        G = GridGraphBuilder(grid, connectivity=6).build()
        path = nx.shortest_path(G, (0, 0, 0), (9, 9, 0), weight='weight')
        assert path[0] == (0, 0, 0)
        assert path[-1] == (9, 9, 0)
