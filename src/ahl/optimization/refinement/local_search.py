"""局部搜索优化。

对已有路径做局部改进，减少总长度或改善平滑度。
基于 2-opt 风格：尝试替换路径子段为更短的连接。
"""

from typing import List, Tuple, Optional, Callable
import math

from ahl.geometry.voxel.grid3d import Grid3D
from ahl.routing.single.astar import AStarSDF

Node3D = Tuple[int, int, int]
Path3D = List[Node3D]


def two_opt_improve(
    path: Path3D,
    grid: Grid3D,
    max_iterations: int = 50,
    w_sdf: float = 0.0,
    connectivity: int = 26,
) -> Path3D:
    """2-opt 局部搜索改进路径。

    对路径中每对 (i, j)，尝试用 A* 重新连接 path[i] → path[j]，
    如果新子段更短则替换。

    Args:
        path: 原始路径
        grid: Grid3D
        max_iterations: 最大改进轮次
        w_sdf: SDF 权重
        connectivity: 邻域连通性

    Returns:
        改进后的路径
    """
    if len(path) <= 3:
        return list(path)

    searcher = AStarSDF(grid, w_sdf=w_sdf, connectivity=connectivity)
    best = list(path)

    for _ in range(max_iterations):
        improved = False
        n = len(best)

        # 跳步采样减少搜索量
        step = max(1, n // 20)

        for i in range(0, n - 2, step):
            for j in range(i + 2, min(i + n // 2, n), step):
                # 当前子段长度
                old_len = _sub_path_length(best, i, j)

                # 尝试 A* 重连
                new_sub = searcher.search(best[i], best[j])
                if new_sub is None:
                    continue

                new_len = _sub_path_length(new_sub, 0, len(new_sub) - 1)

                if new_len < old_len * 0.95:  # 至少改进 5%
                    best = best[:i] + new_sub + best[j + 1:]
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return best


def shortcut_improve(
    path: Path3D,
    grid: Grid3D,
    connectivity: int = 26,
) -> Path3D:
    """快捷路径优化：尝试直接连接间隔较远的路径点。

    从头开始，对每个点尝试跳过中间点直连更远的点。
    比 2-opt 更快但改进幅度较小。

    Args:
        path: 原始路径
        grid: Grid3D
        connectivity: 邻域连通性

    Returns:
        优化后的路径
    """
    if len(path) <= 2:
        return list(path)

    searcher = AStarSDF(grid, w_sdf=0.0, connectivity=connectivity)
    result = [path[0]]
    i = 0

    while i < len(path) - 1:
        # 尝试从 path[i] 直连尽可能远的 path[j]
        best_j = i + 1
        for j in range(min(len(path) - 1, i + 20), i + 1, -1):
            sub = searcher.search(path[i], path[j])
            if sub is not None:
                sub_len = _sub_path_length(sub, 0, len(sub) - 1)
                orig_len = _sub_path_length(path, i, j)
                if sub_len <= orig_len:
                    best_j = j
                    result.extend(sub[1:])
                    break
        else:
            result.append(path[best_j])

        i = best_j

    return result


def _sub_path_length(path: Path3D, start: int, end: int) -> float:
    """计算路径子段长度。"""
    total = 0.0
    for k in range(start, end):
        a, b = path[k], path[k + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total
