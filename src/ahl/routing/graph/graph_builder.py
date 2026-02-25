"""体素网格 → 图结构转换。

将 Grid3D 转换为 networkx 加权图，供最短路 / Steiner Tree 等图算法使用。
"""

from typing import Tuple, Optional, Set
import numpy as np
import networkx as nx

from ahl.geometry.voxel.grid3d import Grid3D, CellType, NEIGHBORS_6, NEIGHBORS_26
from ahl.routing.single.cost_function import CostFunction

Node3D = Tuple[int, int, int]


class GridGraphBuilder:
    """从 Grid3D 构建 networkx 图。

    节点 = 可通行体素 (i, j, k)
    边 = 邻域连接，权重由 CostFunction 计算

    Usage:
        builder = GridGraphBuilder(grid, connectivity=26)
        G = builder.build()
        # 或指定 ROI 子区域
        G = builder.build(roi_min=(0,0,0), roi_max=(10,10,10))
    """

    def __init__(
        self,
        grid: Grid3D,
        connectivity: int = 26,
        cost_fn: Optional[CostFunction] = None,
    ):
        """
        Args:
            grid: Grid3D 体素网格
            connectivity: 6 或 26
            cost_fn: 代价函数（默认用 w_dist=1, w_sdf=0 的纯距离代价）
        """
        self.grid = grid
        self.connectivity = connectivity
        self.cost_fn = cost_fn or CostFunction(w_dist=1.0, w_sdf=0.0)

        if connectivity == 6:
            self._offsets = NEIGHBORS_6
        else:
            self._offsets = NEIGHBORS_26

    def build(
        self,
        roi_min: Optional[Node3D] = None,
        roi_max: Optional[Node3D] = None,
    ) -> nx.Graph:
        """构建加权无向图。

        Args:
            roi_min: ROI 最小索引 (i,j,k)（含），默认 (0,0,0)
            roi_max: ROI 最大索引 (i,j,k)（含），默认网格最大值

        Returns:
            networkx.Graph，节点为 (i,j,k) 元组，边属性 'weight'
        """
        grid = self.grid
        data = grid.data
        nx_size, ny_size, nz_size = grid.shape

        # ROI 范围
        i0, j0, k0 = roi_min if roi_min else (0, 0, 0)
        i1, j1, k1 = roi_max if roi_max else (nx_size - 1, ny_size - 1, nz_size - 1)

        # 限制在合法范围内
        i0, j0, k0 = max(0, i0), max(0, j0), max(0, k0)
        i1 = min(nx_size - 1, i1)
        j1 = min(ny_size - 1, j1)
        k1 = min(nz_size - 1, k1)

        G = nx.Graph()
        cost_fn = self.cost_fn
        offsets = self._offsets

        # 遍历 ROI 内的自由体素
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    if data[i, j, k] == CellType.OBSTACLE:
                        continue

                    node = (i, j, k)

                    # 添加节点（即使没有边，孤立节点也保留）
                    if not G.has_node(node):
                        G.add_node(node)

                    # 添加到邻居的边（只添加 "前向" 方向避免重复）
                    for off in offsets:
                        ni = i + int(off[0])
                        nj = j + int(off[1])
                        nk = k + int(off[2])

                        if ni < i0 or ni > i1 or nj < j0 or nj > j1 or nk < k0 or nk > k1:
                            continue
                        if data[ni, nj, nk] == CellType.OBSTACLE:
                            continue

                        neighbor = (ni, nj, nk)

                        if not G.has_edge(node, neighbor):
                            w = cost_fn.edge_cost(node, neighbor)
                            G.add_edge(node, neighbor, weight=w)

        return G

    def build_from_nodes(self, nodes: Set[Node3D]) -> nx.Graph:
        """从指定节点集构建子图。

        仅包含 nodes 中的节点及其之间的边。

        Args:
            nodes: 节点集合

        Returns:
            networkx.Graph 子图
        """
        grid = self.grid
        data = grid.data
        nx_size, ny_size, nz_size = grid.shape
        cost_fn = self.cost_fn
        offsets = self._offsets

        G = nx.Graph()
        G.add_nodes_from(nodes)

        for node in nodes:
            i, j, k = node
            for off in offsets:
                ni = i + int(off[0])
                nj = j + int(off[1])
                nk = k + int(off[2])

                if ni < 0 or ni >= nx_size or nj < 0 or nj >= ny_size or nk < 0 or nk >= nz_size:
                    continue

                neighbor = (ni, nj, nk)
                if neighbor in nodes and not G.has_edge(node, neighbor):
                    if data[ni, nj, nk] != CellType.OBSTACLE:
                        w = cost_fn.edge_cost(node, neighbor)
                        G.add_edge(node, neighbor, weight=w)

        return G
