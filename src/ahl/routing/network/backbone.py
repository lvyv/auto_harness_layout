"""主干网络生成与支线路由。

核心策略（来自 CLAUDE.md）：
- 先构建稳定主干 (Backbone)，再分组/逐个生成支线 (Branch)
- 主干稳定可解释，支线容忍局部次优
- 支线"贴主干"用代价偏置而非硬规则

流程：
1. 终端聚类 → 得到 K 个中间节点
2. 连接中间节点 → 主干网络（MST / Steiner 近似）
3. 每个终端通过 A* 连接到最近的主干节点 → 支线

支线搜索时，主干上的边获得代价折扣，使支线自然靠近/复用主干。
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import numpy as np
import networkx as nx

from ahl.geometry.voxel.grid3d import Grid3D
from ahl.routing.single.astar import AStarSDF
from ahl.routing.single.cost_function import CostFunction
from ahl.routing.graph.path_ops import path_edges, path_length
from .clustering import TerminalClusterer, Cluster

Node3D = Tuple[int, int, int]
Path3D = List[Node3D]


@dataclass
class Backbone:
    """主干网络结果。

    Attributes:
        graph: 主干 networkx 图（节点=体素，边=路径段）
        paths: 主干路径字典 {(cluster_i, cluster_j): path}
        cluster_centers: 聚类中心列表
        edges: 主干所有边的集合 (frozenset)
        nodes: 主干所有节点的集合
    """
    graph: nx.Graph = field(default_factory=nx.Graph)
    paths: Dict[Tuple[int, int], Path3D] = field(default_factory=dict)
    cluster_centers: List[Node3D] = field(default_factory=list)
    edges: Set[frozenset] = field(default_factory=set)
    nodes: Set[Node3D] = field(default_factory=set)


@dataclass
class RoutingSolution:
    """完整路由结果（主干 + 支线）。

    Attributes:
        backbone: 主干网络
        branches: 支线路径字典 {terminal: path_to_backbone}
        all_paths: 所有路径列表（主干 + 支线）
    """
    backbone: Backbone = field(default_factory=Backbone)
    branches: Dict[Node3D, Path3D] = field(default_factory=dict)
    all_paths: List[Path3D] = field(default_factory=list)


class BackboneBuilder:
    """主干网络构建器。

    Usage:
        builder = BackboneBuilder(grid, w_sdf=0.5)
        backbone = builder.build(terminals, n_clusters=3)
    """

    def __init__(
        self,
        grid: Grid3D,
        w_dist: float = 1.0,
        w_sdf: float = 0.5,
        connectivity: int = 26,
    ):
        self.grid = grid
        self.w_dist = w_dist
        self.w_sdf = w_sdf
        self.connectivity = connectivity

    def build(
        self,
        terminals: List[Node3D],
        n_clusters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Backbone:
        """构建主干网络。

        步骤：
        1. K-means 聚类终端点
        2. 在聚类中心上构建完全图（A* 最短路作为边权）
        3. 求最小生成树 → 主干骨架
        4. 将 MST 边对应的 A* 路径合并为主干

        Args:
            terminals: 终端点列表
            n_clusters: 聚类数目（默认 = max(2, len(terminals)//3)）
            seed: 随机种子

        Returns:
            Backbone 实例
        """
        n = len(terminals)
        if n < 2:
            raise ValueError(f"至少需要 2 个终端点，收到 {n}")

        # 默认聚类数
        if n_clusters is None:
            n_clusters = max(2, n // 3)
        n_clusters = min(n_clusters, n)

        # 如果终端数 ≤ 聚类数，每个终端就是一个"聚类中心"
        if n <= n_clusters:
            n_clusters = n

        # 1. 聚类
        clusterer = TerminalClusterer(self.grid)
        clusters = clusterer.cluster(terminals, n_clusters, seed=seed)
        centers = [c.center_voxel for c in clusters]

        # 2. 在中心间构建完全图 + A* 路径
        searcher = AStarSDF(
            self.grid,
            w_dist=self.w_dist,
            w_sdf=self.w_sdf,
            connectivity=self.connectivity,
        )

        # 缓存中心间路径
        center_paths: Dict[Tuple[int, int], Path3D] = {}
        full_graph = nx.Graph()

        for i in range(len(centers)):
            full_graph.add_node(i)
            for j in range(i + 1, len(centers)):
                path = searcher.search(centers[i], centers[j])
                if path is not None:
                    length = path_length(path)
                    full_graph.add_edge(i, j, weight=length)
                    center_paths[(i, j)] = path

        # 3. MST
        if full_graph.number_of_edges() == 0:
            # 没有任何可达路径
            backbone = Backbone(cluster_centers=centers)
            return backbone

        mst = nx.minimum_spanning_tree(full_graph)

        # 4. 合并 MST 边的路径
        backbone = Backbone(cluster_centers=centers)

        for u, v in mst.edges():
            key = (min(u, v), max(u, v))
            if key in center_paths:
                path = center_paths[key]
                backbone.paths[key] = path
                backbone.edges |= path_edges(path)
                backbone.nodes |= set(path)

                # 添加到 backbone graph
                for k in range(len(path) - 1):
                    backbone.graph.add_edge(path[k], path[k + 1])

        return backbone

    def build_from_centers(
        self,
        centers: List[Node3D],
    ) -> Backbone:
        """直接从指定中心点构建主干（跳过聚类步骤）。

        Args:
            centers: 主干中间节点列表

        Returns:
            Backbone 实例
        """
        if len(centers) < 2:
            raise ValueError(f"至少需要 2 个中心点，收到 {len(centers)}")

        searcher = AStarSDF(
            self.grid,
            w_dist=self.w_dist,
            w_sdf=self.w_sdf,
            connectivity=self.connectivity,
        )

        center_paths = {}
        full_graph = nx.Graph()

        for i in range(len(centers)):
            full_graph.add_node(i)
            for j in range(i + 1, len(centers)):
                path = searcher.search(centers[i], centers[j])
                if path is not None:
                    length = path_length(path)
                    full_graph.add_edge(i, j, weight=length)
                    center_paths[(i, j)] = path

        if full_graph.number_of_edges() == 0:
            return Backbone(cluster_centers=centers)

        mst = nx.minimum_spanning_tree(full_graph)
        backbone = Backbone(cluster_centers=centers)

        for u, v in mst.edges():
            key = (min(u, v), max(u, v))
            if key in center_paths:
                path = center_paths[key]
                backbone.paths[key] = path
                backbone.edges |= path_edges(path)
                backbone.nodes |= set(path)
                for k in range(len(path) - 1):
                    backbone.graph.add_edge(path[k], path[k + 1])

        return backbone


class BranchRouter:
    """支线路由器（贴主干偏置）。

    通过在 CostFunction 中设置 backbone_edges 和 cost_bias，
    使 A* 搜索自然倾向复用主干路径。

    这是代价偏置，不是硬约束。

    Usage:
        router = BranchRouter(grid, backbone, cost_bias=0.3)
        solution = router.route_all(terminals)
    """

    def __init__(
        self,
        grid: Grid3D,
        backbone: Backbone,
        cost_bias: float = 0.3,
        w_dist: float = 1.0,
        w_sdf: float = 0.5,
        connectivity: int = 26,
    ):
        self.grid = grid
        self.backbone = backbone
        self.cost_bias = cost_bias
        self.w_dist = w_dist
        self.w_sdf = w_sdf
        self.connectivity = connectivity

    def route_terminal(
        self,
        terminal: Node3D,
        target: Optional[Node3D] = None,
    ) -> Optional[Path3D]:
        """将一个终端连接到主干。

        Args:
            terminal: 终端体素坐标
            target: 目标主干节点（默认自动选最近的聚类中心）

        Returns:
            路径，或 None
        """
        if target is None:
            target = self._nearest_backbone_node(terminal)

        if target is None:
            return None

        if terminal == target:
            return [terminal]

        # 创建带主干偏置的代价函数
        cost_fn = CostFunction(
            w_dist=self.w_dist,
            w_sdf=self.w_sdf,
            sdf=self.grid.get_sdf(),
            backbone_edges=self.backbone.edges,
            cost_bias=self.cost_bias,
        )

        # 构建 A* 搜索器（使用自定义代价函数）
        searcher = AStarSDF(
            self.grid,
            w_dist=self.w_dist,
            w_sdf=self.w_sdf,
            connectivity=self.connectivity,
        )
        # 替换代价函数为带主干偏置的版本
        searcher.cost_fn = cost_fn

        return searcher.search(terminal, target)

    def route_all(
        self,
        terminals: List[Node3D],
    ) -> RoutingSolution:
        """将所有终端连接到主干。

        Args:
            terminals: 终端点列表

        Returns:
            RoutingSolution
        """
        solution = RoutingSolution(backbone=self.backbone)

        for terminal in terminals:
            # 跳过已经在主干上的终端
            if terminal in self.backbone.nodes:
                solution.branches[terminal] = [terminal]
                continue

            path = self.route_terminal(terminal)
            if path is not None:
                solution.branches[terminal] = path
                solution.all_paths.append(path)

        # 添加主干路径
        for p in self.backbone.paths.values():
            solution.all_paths.append(p)

        return solution

    def _nearest_backbone_node(self, terminal: Node3D) -> Optional[Node3D]:
        """找距 terminal 最近的聚类中心。"""
        centers = self.backbone.cluster_centers
        if not centers:
            return None

        best = None
        best_dist = float('inf')
        t = np.array(terminal, dtype=np.float64)

        for c in centers:
            d = np.linalg.norm(t - np.array(c, dtype=np.float64))
            if d < best_dist:
                best_dist = d
                best = c

        return best
