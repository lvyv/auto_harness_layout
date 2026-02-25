"""路径操作：简化、平滑、合并、度量。

提供体素路径的后处理工具：
- Douglas-Peucker 简化（减少冗余点）
- 移动平均平滑
- 共享路段检测（用于主干复用分析）
- 路径长度计算
"""

from typing import List, Tuple, Sequence, Optional
import numpy as np
import math

Node3D = Tuple[int, int, int]
Path3D = List[Node3D]


def path_length(path: Path3D) -> float:
    """计算路径的欧氏总长度。

    Args:
        path: 体素索引路径

    Returns:
        总长度（体素单位）
    """
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        dz = b[2] - a[2]
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def path_to_array(path: Path3D) -> np.ndarray:
    """路径转为 (N, 3) numpy 数组。"""
    return np.array(path, dtype=np.float64)


def simplify_path(path: Path3D, epsilon: float = 0.5) -> Path3D:
    """Douglas-Peucker 路径简化。

    去除共线或近共线的冗余节点，保留路径形状。

    Args:
        path: 原始路径
        epsilon: 容差（体素单位），越小保留越多点

    Returns:
        简化后的路径（保留首尾）
    """
    if len(path) <= 2:
        return list(path)

    points = path_to_array(path)
    mask = np.ones(len(points), dtype=bool)
    _dp_recursive(points, mask, 0, len(points) - 1, epsilon)

    return [path[i] for i in range(len(path)) if mask[i]]


def _dp_recursive(
    points: np.ndarray,
    mask: np.ndarray,
    start: int,
    end: int,
    epsilon: float,
) -> None:
    """Douglas-Peucker 递归实现。"""
    if end - start <= 1:
        return

    # 线段方向
    line_vec = points[end] - points[start]
    line_len = np.linalg.norm(line_vec)

    max_dist = 0.0
    max_idx = start

    for i in range(start + 1, end):
        if line_len < 1e-12:
            dist = np.linalg.norm(points[i] - points[start])
        else:
            # 点到线段的距离
            t = np.dot(points[i] - points[start], line_vec) / (line_len * line_len)
            t = max(0.0, min(1.0, t))
            proj = points[start] + t * line_vec
            dist = np.linalg.norm(points[i] - proj)

        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        _dp_recursive(points, mask, start, max_idx, epsilon)
        _dp_recursive(points, mask, max_idx, end, epsilon)
    else:
        # 去掉中间所有点
        for i in range(start + 1, end):
            mask[i] = False


def smooth_path(
    path: Path3D,
    window: int = 3,
    iterations: int = 1,
    keep_endpoints: bool = True,
) -> List[Tuple[float, float, float]]:
    """移动平均平滑。

    对路径点坐标做滑动窗口平均，减少锯齿。
    输出为浮点坐标（不再对齐体素网格）。

    Args:
        path: 原始路径
        window: 窗口大小（奇数，默认3）
        iterations: 迭代次数
        keep_endpoints: 是否保持首尾不动

    Returns:
        平滑后的路径（浮点坐标）
    """
    if len(path) <= 2:
        return [(float(p[0]), float(p[1]), float(p[2])) for p in path]

    points = path_to_array(path)
    half = window // 2

    for _ in range(iterations):
        smoothed = points.copy()
        start_idx = 1 if keep_endpoints else 0
        end_idx = len(points) - 1 if keep_endpoints else len(points)

        for i in range(start_idx, end_idx):
            lo = max(0, i - half)
            hi = min(len(points), i + half + 1)
            smoothed[i] = points[lo:hi].mean(axis=0)

        points = smoothed

    return [(float(p[0]), float(p[1]), float(p[2])) for p in points]


def count_turns(path: Path3D) -> int:
    """统计路径转弯次数。

    转弯定义为连续三个点的方向向量不平行。

    Args:
        path: 路径

    Returns:
        转弯次数
    """
    if len(path) < 3:
        return 0

    turns = 0
    for i in range(1, len(path) - 1):
        prev, curr, nxt = path[i - 1], path[i], path[i + 1]
        d1 = (curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2])
        d2 = (nxt[0] - curr[0], nxt[1] - curr[1], nxt[2] - curr[2])
        if d1 != d2:
            turns += 1

    return turns


def find_shared_segments(
    path_a: Path3D,
    path_b: Path3D,
) -> List[Path3D]:
    """检测两条路径的共享路段。

    共享路段 = 连续的相同节点序列（≥ 2 个点）。

    Args:
        path_a: 第一条路径
        path_b: 第二条路径

    Returns:
        共享路段列表，每段是节点列表
    """
    set_b = set()
    for i, node in enumerate(path_b):
        set_b.setdefault(node, []).append(i) if False else None

    # 用集合快速查找
    nodes_b = {node: i for i, node in enumerate(path_b)}

    segments = []
    current_seg = []
    prev_b_idx = -2  # 上一个在 path_b 中的位置

    for node in path_a:
        if node in nodes_b:
            b_idx = nodes_b[node]
            if b_idx == prev_b_idx + 1:
                # 连续段
                current_seg.append(node)
            else:
                # 新段开始
                if len(current_seg) >= 2:
                    segments.append(current_seg)
                current_seg = [node]
            prev_b_idx = b_idx
        else:
            if len(current_seg) >= 2:
                segments.append(current_seg)
            current_seg = []
            prev_b_idx = -2

    if len(current_seg) >= 2:
        segments.append(current_seg)

    return segments


def path_edges(path: Path3D) -> set:
    """提取路径中所有边的集合。

    返回 frozenset 形式的边集合，用于主干折扣等。

    Args:
        path: 路径

    Returns:
        {frozenset({node_a, node_b}), ...}
    """
    edges = set()
    for i in range(len(path) - 1):
        edges.add(frozenset({path[i], path[i + 1]}))
    return edges
