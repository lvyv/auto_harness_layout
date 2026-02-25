"""路径搜索代价函数。

核心设计原则（来自 CLAUDE.md）：
- 在搜索阶段编码偏好，而非后处理平滑
- cost = 距离项 + SDF惩罚项 + 转弯惩罚 - 主干折扣
- SDF惩罚使路径远离障碍物表面
- 主干折扣使支线优先复用已有主干（代价偏置，非硬规则）
"""

from typing import Tuple, Optional, Set, FrozenSet
import numpy as np
import math


# 边的标准表示：frozenset({node_a, node_b})
Edge = FrozenSet[Tuple[int, int, int]]


class CostFunction:
    """A* / Dijkstra 通用代价函数。

    总代价 = w_dist * 移动距离
           + w_sdf * SDF惩罚
           + w_turn * 转弯惩罚
           - backbone折扣

    SDF惩罚 = 1 / (sdf_value + epsilon)
    转弯惩罚 = w_turn * (1 - cos(方向变化角))
    主干折扣 = cost_bias * 原代价（仅对主干边生效）

    Attributes:
        w_dist: 距离权重
        w_sdf: SDF 惩罚权重（0 则忽略 SDF）
        w_turn: 转弯惩罚权重（0 则忽略转弯）
        epsilon: 防除零小量
        sdf: (Nx, Ny, Nz) SDF 数组引用（可选）
        backbone_edges: 主干边集合（可选）
        cost_bias: 主干边代价折扣系数 (0~1)，0.3 表示主干边代价打7折
    """

    def __init__(
        self,
        w_dist: float = 1.0,
        w_sdf: float = 0.5,
        w_turn: float = 0.0,
        epsilon: float = 0.1,
        sdf: np.ndarray = None,
        backbone_edges: Optional[Set[Edge]] = None,
        cost_bias: float = 0.0,
    ):
        if w_dist < 0 or w_sdf < 0 or w_turn < 0:
            raise ValueError("权重不能为负数")
        if not (0.0 <= cost_bias <= 1.0):
            raise ValueError(f"cost_bias 必须在 [0, 1] 范围内，收到 {cost_bias}")
        self.w_dist = w_dist
        self.w_sdf = w_sdf
        self.w_turn = w_turn
        self.epsilon = epsilon
        self.sdf = sdf
        self.backbone_edges = backbone_edges or set()
        self.cost_bias = cost_bias

    def set_sdf(self, sdf: np.ndarray) -> None:
        """设置或更新 SDF 数组。"""
        self.sdf = sdf

    def set_backbone(self, edges: Set[Edge], cost_bias: float = 0.3) -> None:
        """设置主干边集合和折扣系数。

        Args:
            edges: 主干边集合，每条边为 frozenset({node_a, node_b})
            cost_bias: 折扣系数，0.3 表示主干边代价 × 0.7
        """
        self.backbone_edges = edges
        self.cost_bias = cost_bias

    def edge_cost(
        self,
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
        prev: Optional[Tuple[int, int, int]] = None,
    ) -> float:
        """计算从 current 到 neighbor 的边代价。

        Args:
            current: 当前体素 (i, j, k)
            neighbor: 邻居体素 (i, j, k)
            prev: 前一个体素（用于计算转弯，可选）

        Returns:
            非负代价值
        """
        # 欧氏移动距离
        di = neighbor[0] - current[0]
        dj = neighbor[1] - current[1]
        dk = neighbor[2] - current[2]
        move_dist = math.sqrt(di * di + dj * dj + dk * dk)

        cost = self.w_dist * move_dist

        # SDF 惩罚（在邻居点计算）
        if self.sdf is not None and self.w_sdf > 0:
            sdf_val = float(self.sdf[neighbor[0], neighbor[1], neighbor[2]])
            cost += self.w_sdf / (sdf_val + self.epsilon)

        # 转弯惩罚
        if prev is not None and self.w_turn > 0:
            cost += self.w_turn * self._turn_penalty(prev, current, neighbor)

        # 主干折扣：如果这条边在主干上，代价打折
        if self.backbone_edges and self.cost_bias > 0:
            edge_key = frozenset({current, neighbor})
            if edge_key in self.backbone_edges:
                cost *= (1.0 - self.cost_bias)

        return cost

    def heuristic(
        self,
        node: Tuple[int, int, int],
        goal: Tuple[int, int, int],
    ) -> float:
        """A* 启发式函数（欧氏距离，可容许）。"""
        di = goal[0] - node[0]
        dj = goal[1] - node[1]
        dk = goal[2] - node[2]
        return self.w_dist * math.sqrt(di * di + dj * dj + dk * dk)

    @staticmethod
    def _turn_penalty(
        prev: Tuple[int, int, int],
        current: Tuple[int, int, int],
        neighbor: Tuple[int, int, int],
    ) -> float:
        """计算转弯惩罚 = 1 - cos(方向变化角)。

        直行 → 0，90° 转弯 → 1，180° 掉头 → 2。
        """
        # 进入方向
        d1x = current[0] - prev[0]
        d1y = current[1] - prev[1]
        d1z = current[2] - prev[2]
        # 离开方向
        d2x = neighbor[0] - current[0]
        d2y = neighbor[1] - current[1]
        d2z = neighbor[2] - current[2]

        dot = d1x * d2x + d1y * d2y + d1z * d2z
        n1 = math.sqrt(d1x * d1x + d1y * d1y + d1z * d1z)
        n2 = math.sqrt(d2x * d2x + d2y * d2y + d2z * d2z)

        if n1 == 0 or n2 == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / (n1 * n2)))
        return 1.0 - cos_angle
