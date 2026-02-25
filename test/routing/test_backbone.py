"""backbone + branch 路由单元测试。"""

import numpy as np
import pytest

from ahl.geometry.voxel.grid3d import Grid3D, CellType
from ahl.routing.network.backbone import BackboneBuilder, BranchRouter, Backbone
from ahl.routing.graph.path_ops import path_edges


class TestBackboneBuilder:
    """主干网络构建测试。"""

    def _make_grid_with_channel(self):
        """创建一个带通道的测试网格。

        20x20x1 的平面网格，中间有一道墙 (x=10) 留两个缺口。
        """
        grid = Grid3D(20, 20, 1)
        # 中间墙
        for y in range(20):
            if y not in (5, 15):  # 留两个缺口
                grid.set_cell(10, y, 0, CellType.OBSTACLE)
        return grid

    def test_build_basic(self):
        """基础主干构建：3 个终端，2 个聚类。"""
        grid = Grid3D(20, 20, 1)
        terminals = [(0, 0, 0), (19, 0, 0), (10, 19, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build(terminals, n_clusters=2, seed=42)

        assert len(backbone.cluster_centers) == 2
        assert len(backbone.paths) > 0
        assert len(backbone.edges) > 0
        assert len(backbone.nodes) > 0

    def test_build_from_centers(self):
        """直接指定中心点构建主干。"""
        grid = Grid3D(20, 20, 1)
        centers = [(0, 10, 0), (19, 10, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        assert len(backbone.paths) == 1
        # 路径应从一端到另一端
        path = list(backbone.paths.values())[0]
        assert path[0] in centers or path[-1] in centers

    def test_build_with_obstacles(self):
        """带障碍物时主干应绕行。"""
        grid = self._make_grid_with_channel()
        centers = [(2, 10, 0), (18, 10, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        assert len(backbone.paths) == 1
        path = list(backbone.paths.values())[0]

        # 路径不应穿过障碍
        for node in path:
            assert grid.is_free(*node)

    def test_backbone_graph_connected(self):
        """主干图应该是连通的。"""
        grid = Grid3D(20, 20, 1)
        centers = [(0, 0, 0), (10, 10, 0), (19, 19, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        import networkx as nx
        assert nx.is_connected(backbone.graph)

    def test_too_few_terminals(self):
        grid = Grid3D(10, 10, 1)
        builder = BackboneBuilder(grid)
        with pytest.raises(ValueError):
            builder.build([(0, 0, 0)], n_clusters=1)

    def test_default_n_clusters(self):
        """不指定 n_clusters 时自动计算。"""
        grid = Grid3D(30, 30, 1)
        terminals = [(i * 3, i * 3, 0) for i in range(9)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build(terminals, seed=42)

        # 默认 max(2, 9//3) = 3
        assert len(backbone.cluster_centers) == 3


class TestBranchRouter:
    """支线路由测试。"""

    def test_route_terminal_to_backbone(self):
        """终端应能通过支线连接到主干。"""
        grid = Grid3D(20, 20, 1)
        centers = [(5, 10, 0), (15, 10, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        router = BranchRouter(grid, backbone, cost_bias=0.3, w_sdf=0.0)
        terminal = (0, 0, 0)
        path = router.route_terminal(terminal)

        assert path is not None
        assert path[0] == terminal
        # 终点应该是某个聚类中心
        assert path[-1] in centers

    def test_route_all(self):
        """批量支线路由。"""
        grid = Grid3D(20, 20, 1)
        centers = [(5, 10, 0), (15, 10, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        terminals = [(0, 0, 0), (19, 19, 0), (10, 0, 0)]
        router = BranchRouter(grid, backbone, cost_bias=0.3, w_sdf=0.0)
        solution = router.route_all(terminals)

        # 每个终端都应有支线
        assert len(solution.branches) == 3
        # all_paths 包含主干 + 支线
        assert len(solution.all_paths) > 0

    def test_terminal_on_backbone(self):
        """终端已在主干上，支线应为单点。"""
        grid = Grid3D(20, 20, 1)
        centers = [(5, 10, 0), (15, 10, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        # 用主干上的点作为终端
        backbone_node = list(backbone.nodes)[0]
        router = BranchRouter(grid, backbone, cost_bias=0.3, w_sdf=0.0)
        solution = router.route_all([backbone_node])

        assert backbone_node in solution.branches
        assert solution.branches[backbone_node] == [backbone_node]

    def test_cost_bias_attracts_to_backbone(self):
        """有主干偏置时，支线应比无偏置时更靠近主干。"""
        grid = Grid3D(20, 20, 1)
        centers = [(10, 0, 0), (10, 19, 0)]

        builder = BackboneBuilder(grid, w_sdf=0.0)
        backbone = builder.build_from_centers(centers)

        terminal = (0, 10, 0)

        # 无偏置
        router_no_bias = BranchRouter(
            grid, backbone, cost_bias=0.0, w_sdf=0.0
        )
        path_no = router_no_bias.route_terminal(terminal)

        # 有偏置
        router_bias = BranchRouter(
            grid, backbone, cost_bias=0.5, w_sdf=0.0
        )
        path_bias = router_bias.route_terminal(terminal)

        assert path_no is not None
        assert path_bias is not None

        # 有偏置的路径应与主干有更多共享边
        backbone_edge_set = backbone.edges
        shared_no = len(path_edges(path_no) & backbone_edge_set)
        shared_bias = len(path_edges(path_bias) & backbone_edge_set)
        assert shared_bias >= shared_no


class TestCostFunctionEnhancements:
    """代价函数增强功能测试（转弯惩罚 + 主干折扣）。"""

    def test_turn_penalty(self):
        """转弯惩罚计算。"""
        from ahl.routing.single.cost_function import CostFunction

        cf = CostFunction(w_dist=1.0, w_sdf=0.0, w_turn=1.0)

        # 直行：惩罚 = 0
        cost_straight = cf.edge_cost(
            (1, 0, 0), (2, 0, 0), prev=(0, 0, 0)
        )
        # 90° 转弯：惩罚 = 1.0
        cost_turn = cf.edge_cost(
            (1, 0, 0), (1, 1, 0), prev=(0, 0, 0)
        )

        assert cost_turn > cost_straight

    def test_backbone_discount(self):
        """主干折扣使主干边更便宜。"""
        from ahl.routing.single.cost_function import CostFunction

        backbone_edges = {
            frozenset({(0, 0, 0), (1, 0, 0)}),
        }
        cf = CostFunction(
            w_dist=1.0, w_sdf=0.0,
            backbone_edges=backbone_edges, cost_bias=0.5,
        )

        cost_on_backbone = cf.edge_cost((0, 0, 0), (1, 0, 0))
        cost_off_backbone = cf.edge_cost((2, 0, 0), (3, 0, 0))

        # 主干边应打折
        assert cost_on_backbone < cost_off_backbone

    def test_invalid_cost_bias(self):
        from ahl.routing.single.cost_function import CostFunction
        with pytest.raises(ValueError):
            CostFunction(cost_bias=1.5)
