"""终端聚类模块。

对接线终端点做 K-means 聚类，确定主干网络的中间节点。
支持 KDTree 加速最近邻查询。
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import KDTree
from scipy.cluster.vq import kmeans2

Node3D = Tuple[int, int, int]


@dataclass
class Cluster:
    """聚类结果。

    Attributes:
        center: 聚类中心坐标 (浮点)
        center_voxel: 聚类中心最近的体素索引
        members: 属于该聚类的终端点列表
        label: 聚类编号
    """
    center: np.ndarray
    center_voxel: Node3D
    members: List[Node3D] = field(default_factory=list)
    label: int = 0


class TerminalClusterer:
    """终端点 K-means 聚类器。

    流程：
    1. 对终端点做 K-means 聚类
    2. 将浮点聚类中心对齐到最近的自由体素
    3. 返回 Cluster 列表

    Usage:
        clusterer = TerminalClusterer(grid)
        clusters = clusterer.cluster(terminals, n_clusters=3)
    """

    def __init__(self, grid=None):
        """
        Args:
            grid: Grid3D 实例（用于将聚类中心对齐到自由体素，可选）
        """
        self.grid = grid

    def cluster(
        self,
        terminals: List[Node3D],
        n_clusters: int,
        seed: Optional[int] = None,
    ) -> List[Cluster]:
        """执行 K-means 聚类。

        Args:
            terminals: 终端点列表 [(i,j,k), ...]
            n_clusters: 聚类数目
            seed: 随机种子（可选，用于可复现性）

        Returns:
            Cluster 列表，按 label 排序

        Raises:
            ValueError: 聚类数目不合法
        """
        n = len(terminals)
        if n_clusters <= 0:
            raise ValueError(f"聚类数必须为正整数，收到 {n_clusters}")
        if n_clusters > n:
            raise ValueError(
                f"聚类数 ({n_clusters}) 不能超过终端点数 ({n})"
            )

        # 特殊情况：每个终端单独一个聚类
        if n_clusters == n:
            clusters = []
            for i, t in enumerate(terminals):
                c = Cluster(
                    center=np.array(t, dtype=np.float64),
                    center_voxel=t,
                    members=[t],
                    label=i,
                )
                clusters.append(c)
            return clusters

        points = np.array(terminals, dtype=np.float64)

        # K-means
        rng = np.random.default_rng(seed)
        # scipy kmeans2 用 minit='points' 从输入中选初始中心
        centers, labels = kmeans2(
            points, n_clusters,
            minit='points',
            seed=rng,
        )

        # 构建聚类结果
        clusters = []
        for k in range(n_clusters):
            member_indices = np.where(labels == k)[0]
            members = [terminals[i] for i in member_indices]

            center = centers[k]
            center_voxel = self._snap_to_voxel(center, members)

            clusters.append(Cluster(
                center=center,
                center_voxel=center_voxel,
                members=members,
                label=k,
            ))

        return clusters

    def _snap_to_voxel(
        self,
        center: np.ndarray,
        members: List[Node3D],
    ) -> Node3D:
        """将浮点聚类中心对齐到最近的自由体素。

        优先在 grid 上找最近自由体素；无 grid 时四舍五入到最近成员。
        """
        if self.grid is not None:
            # 四舍五入到最近整数索引
            ijk = np.round(center).astype(int)
            i, j, k = int(ijk[0]), int(ijk[1]), int(ijk[2])
            if self.grid.is_free(i, j, k):
                return (i, j, k)
            # 如果不是自由体素，找最近的成员
            return self._nearest_member(center, members)
        else:
            return self._nearest_member(center, members)

    @staticmethod
    def _nearest_member(
        center: np.ndarray,
        members: List[Node3D],
    ) -> Node3D:
        """从成员中找距离 center 最近的点。"""
        best = members[0]
        best_dist = float('inf')
        for m in members:
            d = np.linalg.norm(center - np.array(m))
            if d < best_dist:
                best_dist = d
                best = m
        return best

    @staticmethod
    def build_kdtree(points: List[Node3D]) -> KDTree:
        """构建 KDTree 索引。

        Args:
            points: 点列表

        Returns:
            scipy.spatial.KDTree
        """
        return KDTree(np.array(points, dtype=np.float64))

    @staticmethod
    def query_nearest(
        tree: KDTree,
        query: Node3D,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """KDTree 最近邻查询。

        Args:
            tree: KDTree
            query: 查询点
            k: 返回的最近邻数目

        Returns:
            (distances, indices)
        """
        return tree.query(np.array(query, dtype=np.float64), k=k)
