"""3D A* 搜索单元测试。"""

import numpy as np
import pytest

from ahl.geometry.voxel.grid3d import Grid3D, CellType
from ahl.routing.single.astar import AStarSDF
from ahl.routing.single.cost_function import CostFunction


class TestAStarBasic:
    """基础搜索功能测试。"""

    def test_same_start_goal(self):
        """起点 = 终点，应返回单点路径。"""
        grid = Grid3D(5, 5, 5)
        searcher = AStarSDF(grid, w_sdf=0)
        path = searcher.search((2, 2, 2), (2, 2, 2))
        assert path == [(2, 2, 2)]

    def test_adjacent_points(self):
        """相邻两点，路径长度为 2。"""
        grid = Grid3D(5, 5, 5)
        searcher = AStarSDF(grid, w_sdf=0, connectivity=6)
        path = searcher.search((0, 0, 0), (1, 0, 0))
        assert path is not None
        assert len(path) == 2
        assert path[0] == (0, 0, 0)
        assert path[-1] == (1, 0, 0)

    def test_straight_line_6conn(self):
        """无障碍、6-连通下直线路径。"""
        grid = Grid3D(10, 10, 10)
        searcher = AStarSDF(grid, w_sdf=0, connectivity=6)
        path = searcher.search((0, 0, 0), (5, 0, 0))
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1] == (5, 0, 0)
        assert len(path) == 6  # 0,1,2,3,4,5

    def test_diagonal_26conn(self):
        """26-连通下对角线路径应更短。"""
        grid = Grid3D(10, 10, 10)
        searcher = AStarSDF(grid, w_sdf=0, connectivity=26)
        path = searcher.search((0, 0, 0), (5, 5, 5))
        assert path is not None
        # 26-连通下对角线只需 6 步（含首尾）
        assert len(path) == 6


class TestAStarObstacles:
    """带障碍物的搜索测试。"""

    def test_wall_with_gap(self):
        """一面墙阻隔起终点，留一个缺口，路径应绕行。"""
        grid = Grid3D(10, 5, 5)
        # 在 x=4 处设一面墙，y=2 处留缺口
        for j in range(5):
            for k in range(5):
                if j != 2:
                    grid.set_cell(4, j, k, CellType.OBSTACLE)

        searcher = AStarSDF(grid, w_sdf=0, connectivity=6)
        path = searcher.search((0, 0, 0), (9, 0, 0))
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1] == (9, 0, 0)

        # 路径必须经过缺口 (4, 2, *)
        x4_points = [p for p in path if p[0] == 4]
        assert all(p[1] == 2 for p in x4_points)

    def test_no_path(self):
        """完全被障碍包围，应返回 None。"""
        grid = Grid3D(5, 5, 5)
        # 包围终点
        for di, dj, dk in [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ]:
            ni, nj, nk = 3 + di, 3 + dj, 3 + dk
            if grid.is_valid(ni, nj, nk):
                grid.set_cell(ni, nj, nk, CellType.OBSTACLE)

        searcher = AStarSDF(grid, w_sdf=0, connectivity=6)
        path = searcher.search((0, 0, 0), (3, 3, 3))
        assert path is None

    def test_obstacle_start_raises(self):
        """起点是障碍物，应抛异常。"""
        grid = Grid3D(5, 5, 5)
        grid.set_cell(0, 0, 0, CellType.OBSTACLE)
        searcher = AStarSDF(grid, w_sdf=0)
        with pytest.raises(ValueError, match="障碍"):
            searcher.search((0, 0, 0), (4, 4, 4))

    def test_out_of_bounds_raises(self):
        """超出范围应抛异常。"""
        grid = Grid3D(5, 5, 5)
        searcher = AStarSDF(grid, w_sdf=0)
        with pytest.raises(ValueError):
            searcher.search((-1, 0, 0), (4, 4, 4))


class TestAStarSDF:
    """SDF 惩罚效果测试。"""

    def test_sdf_penalty_pushes_away_from_obstacle(self):
        """SDF 惩罚使路径远离障碍物。

        构造一个通道场景：
        - 上方和下方有障碍墙
        - 无 SDF 惩罚时路径可能贴墙
        - 有 SDF 惩罚时路径应趋向通道中心
        """
        grid = Grid3D(20, 10, 1)

        # 上方墙 y=0
        for x in range(20):
            grid.set_cell(x, 0, 0, CellType.OBSTACLE)
        # 下方墙 y=9
        for x in range(20):
            grid.set_cell(x, 9, 0, CellType.OBSTACLE)

        # 无 SDF 惩罚
        searcher_no_sdf = AStarSDF(grid, w_sdf=0.0, connectivity=6)
        path_no = searcher_no_sdf.search((0, 5, 0), (19, 5, 0))
        assert path_no is not None

        # 有 SDF 惩罚
        searcher_sdf = AStarSDF(grid, w_sdf=2.0, connectivity=6)
        path_sdf = searcher_sdf.search((0, 5, 0), (19, 5, 0))
        assert path_sdf is not None

        # 两条路径都应到达目标
        assert path_no[-1] == (19, 5, 0)
        assert path_sdf[-1] == (19, 5, 0)

        # SDF路径的平均 y 应更靠近中心 (y=4.5)
        avg_y_sdf = np.mean([p[1] for p in path_sdf])
        # 应该在通道中心附近
        assert 3.0 <= avg_y_sdf <= 6.0

    def test_sdf_zero_weight_equals_shortest(self):
        """w_sdf=0 时等价于最短路径搜索。"""
        grid = Grid3D(10, 10, 1)
        searcher = AStarSDF(grid, w_sdf=0.0, connectivity=6)
        path = searcher.search((0, 0, 0), (9, 9, 0))
        assert path is not None
        # 6-连通下 Manhattan 距离
        assert len(path) == 19  # 9+9+1


class TestAStarConnectivity:
    """连通性测试。"""

    def test_6_connectivity(self):
        """6-连通路径只走面邻域。"""
        grid = Grid3D(5, 5, 5)
        searcher = AStarSDF(grid, w_sdf=0, connectivity=6)
        path = searcher.search((0, 0, 0), (4, 4, 4))
        assert path is not None

        # 验证每步只走面邻域（距离=1）
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            diff = abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])
            assert diff == 1, f"6-连通下步 {i}: {a} -> {b} 不是面邻域"

    def test_26_connectivity(self):
        """26-连通路径可走对角。"""
        grid = Grid3D(5, 5, 5)
        searcher = AStarSDF(grid, w_sdf=0, connectivity=26)
        path = searcher.search((0, 0, 0), (4, 4, 4))
        assert path is not None

        # 验证每步最大偏移 ≤ 1
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            assert max(abs(a[0]-b[0]), abs(a[1]-b[1]), abs(a[2]-b[2])) <= 1


class TestCostFunction:
    """代价函数独立测试。"""

    def test_distance_only(self):
        """仅距离权重，相邻点代价=1。"""
        cf = CostFunction(w_dist=1.0, w_sdf=0.0)
        cost = cf.edge_cost((0, 0, 0), (1, 0, 0))
        assert abs(cost - 1.0) < 1e-6

    def test_diagonal_cost(self):
        """对角移动距离 = sqrt(3)。"""
        cf = CostFunction(w_dist=1.0, w_sdf=0.0)
        cost = cf.edge_cost((0, 0, 0), (1, 1, 1))
        assert abs(cost - np.sqrt(3)) < 1e-6

    def test_sdf_penalty(self):
        """SDF 惩罚在障碍物附近更高。"""
        sdf = np.ones((5, 5, 5), dtype=np.float32)
        sdf[1, 0, 0] = 0.1  # 靠近障碍
        sdf[3, 0, 0] = 5.0  # 远离障碍

        cf = CostFunction(w_dist=1.0, w_sdf=1.0, epsilon=0.1, sdf=sdf)
        cost_near = cf.edge_cost((0, 0, 0), (1, 0, 0))
        cost_far = cf.edge_cost((2, 0, 0), (3, 0, 0))

        # 靠近障碍的代价应更高
        assert cost_near > cost_far

    def test_heuristic_admissible(self):
        """启发式不超过实际最短距离（可容许）。"""
        cf = CostFunction(w_dist=1.0, w_sdf=0.0)
        h = cf.heuristic((0, 0, 0), (3, 4, 5))
        # 欧氏距离
        actual = np.sqrt(9 + 16 + 25)
        assert abs(h - actual) < 1e-6

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            CostFunction(w_dist=-1.0)
