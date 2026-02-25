"""三维 A* 路径搜索（SDF 引导）。

核心算法：A* 搜索 + SDF 惩罚项。
设计原则：在搜索阶段编码偏好（远离障碍物），而非后处理平滑。
"""

from typing import List, Tuple, Optional
import heapq
import math
import numpy as np

from ahl.geometry.voxel.grid3d import Grid3D, CellType, NEIGHBORS_6, NEIGHBORS_26
from .cost_function import CostFunction


# 预计算邻域移动距离
_MOVE_COSTS_6 = [1.0] * 6  # 面邻域全是1
_MOVE_COSTS_26 = []
for off in NEIGHBORS_26:
    _MOVE_COSTS_26.append(math.sqrt(float(off[0]**2 + off[1]**2 + off[2]**2)))


class AStarSDF:
    """SDF 引导的三维 A* 搜索器。

    代价函数 = w_dist * 移动距离 + w_sdf / (SDF + epsilon)

    Args:
        grid: Grid3D 体素网格
        w_dist: 距离权重（默认 1.0）
        w_sdf: SDF 惩罚权重（默认 0.5，越大越远离障碍）
        epsilon: 防除零小量（默认 0.1）
        connectivity: 邻域连通性，6 或 26（默认 26）
        max_iterations: 最大迭代次数（默认 2_000_000）

    Usage:
        searcher = AStarSDF(grid, w_sdf=0.8)
        path = searcher.search((0, 0, 0), (10, 10, 10))
    """

    def __init__(
        self,
        grid: Grid3D,
        w_dist: float = 1.0,
        w_sdf: float = 0.5,
        epsilon: float = 0.1,
        connectivity: int = 26,
        max_iterations: int = 2_000_000,
    ):
        self.grid = grid
        self.connectivity = connectivity
        self.max_iterations = max_iterations

        # 代价函数
        self.cost_fn = CostFunction(
            w_dist=w_dist,
            w_sdf=w_sdf,
            epsilon=epsilon,
            sdf=grid.get_sdf(),
        )

        # 预计算邻域偏移和移动代价
        if connectivity == 6:
            self._offsets = NEIGHBORS_6
            self._base_costs = _MOVE_COSTS_6
        else:
            self._offsets = NEIGHBORS_26
            self._base_costs = _MOVE_COSTS_26

    def search(
        self,
        start: Tuple[int, int, int],
        goal: Tuple[int, int, int],
    ) -> Optional[List[Tuple[int, int, int]]]:
        """执行 A* 搜索。

        Args:
            start: 起点 (i, j, k)
            goal: 终点 (i, j, k)

        Returns:
            从 start 到 goal 的体素索引路径（包含首尾），
            无路径返回 None

        Raises:
            ValueError: 起点或终点不合法
        """
        grid = self.grid

        # 验证起止点
        if not grid.is_valid(*start):
            raise ValueError(f"起点 {start} 超出网格范围 {grid.shape}")
        if not grid.is_valid(*goal):
            raise ValueError(f"终点 {goal} 超出网格范围 {grid.shape}")
        if not grid.is_free(*start):
            raise ValueError(f"起点 {start} 是障碍物")
        if not grid.is_free(*goal):
            raise ValueError(f"终点 {goal} 是障碍物")

        if start == goal:
            return [start]

        cost_fn = self.cost_fn
        offsets = self._offsets
        nx, ny, nz = grid.shape
        data = grid.data

        # A* 数据结构
        open_heap = []  # (f_score, counter, node)
        came_from = {}
        g_score = {start: 0.0}

        counter = 0
        h0 = cost_fn.heuristic(start, goal)
        heapq.heappush(open_heap, (h0, counter, start))

        closed = set()
        iterations = 0

        while open_heap and iterations < self.max_iterations:
            iterations += 1

            f_cur, _, current = heapq.heappop(open_heap)

            if current == goal:
                return self._reconstruct(came_from, current)

            if current in closed:
                continue
            closed.add(current)

            ci, cj, ck = current

            # 展开邻居
            for idx in range(len(offsets)):
                di, dj, dk = int(offsets[idx][0]), int(offsets[idx][1]), int(offsets[idx][2])
                ni, nj, nk = ci + di, cj + dj, ck + dk

                # 边界检查
                if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                    continue

                # 障碍物检查
                if data[ni, nj, nk] == CellType.OBSTACLE:
                    continue

                neighbor = (ni, nj, nk)

                if neighbor in closed:
                    continue

                # 计算代价
                edge_cost = cost_fn.edge_cost(current, neighbor)
                tentative_g = g_score[current] + edge_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + cost_fn.heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        # 未找到路径
        return None

    @property
    def w_dist(self) -> float:
        return self.cost_fn.w_dist

    @property
    def w_sdf(self) -> float:
        return self.cost_fn.w_sdf

    @staticmethod
    def _reconstruct(
        came_from: dict,
        current: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        """从 came_from 字典重建路径。"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
