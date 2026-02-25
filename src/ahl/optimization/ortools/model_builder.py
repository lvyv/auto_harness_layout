"""高层约束模型构建器。

从 Grid3D 直接构建完整的 IP 路由模型，封装所有底层细节。
"""

from typing import Tuple, List, Optional

from ahl.geometry.voxel.grid3d import Grid3D
from ahl.optimization.ip.flow_network import FlowNetworkModel, FlowSolution

Node3D = Tuple[int, int, int]


class RoutingModelBuilder:
    """路由 IP 模型构建器。

    高层接口：给定 Grid3D、起终点、约束参数，一步构建并求解。

    Usage:
        builder = RoutingModelBuilder(grid)
        solution = builder.solve_shortest_path(start, goal)
        solution = builder.solve_with_turn_limit(start, goal, max_turns=3)
    """

    def __init__(
        self,
        grid: Grid3D,
        connectivity: int = 6,
        sdf_weight: float = 0.0,
    ):
        self.grid = grid
        self.connectivity = connectivity
        self.sdf_weight = sdf_weight

    def solve_shortest_path(
        self,
        source: Node3D,
        sink: Node3D,
    ) -> FlowSolution:
        """求解最短路（无额外约束）。

        Args:
            source: 起点
            sink: 终点

        Returns:
            FlowSolution
        """
        model = FlowNetworkModel.from_grid3d(
            self.grid, source, sink,
            connectivity=self.connectivity,
            sdf_weight=self.sdf_weight,
        )
        return model.solve()

    def solve_with_turn_limit(
        self,
        source: Node3D,
        sink: Node3D,
        max_turns: int,
    ) -> FlowSolution:
        """求解带转弯限制的最短路。

        Args:
            source: 起点
            sink: 终点
            max_turns: 最大转弯次数

        Returns:
            FlowSolution
        """
        model = FlowNetworkModel.from_grid3d(
            self.grid, source, sink,
            connectivity=self.connectivity,
            sdf_weight=self.sdf_weight,
        )
        model.set_max_turns(max_turns)
        return model.solve()

    def solve_multi_path(
        self,
        pairs: List[Tuple[Node3D, Node3D]],
    ) -> List[FlowSolution]:
        """批量求解多条最短路。

        Args:
            pairs: [(source, sink), ...] 列表

        Returns:
            FlowSolution 列表
        """
        results = []
        for source, sink in pairs:
            sol = self.solve_shortest_path(source, sink)
            results.append(sol)
        return results
