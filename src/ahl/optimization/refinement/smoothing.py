"""路径平滑（B 样条）。

将体素路径离散点拟合为光滑 B 样条曲线，输出稠密采样点。
用于最终布线可视化和 CAD 导出。
"""

from typing import List, Tuple
import numpy as np
from scipy.interpolate import splprep, splev


def bspline_smooth(
    path: List[Tuple[int, int, int]],
    num_points: int = 100,
    smoothing: float = 0.0,
    degree: int = 3,
) -> np.ndarray:
    """B 样条路径平滑。

    Args:
        path: 体素索引路径 [(i,j,k), ...]
        num_points: 输出采样点数
        smoothing: 平滑因子 (0 = 插值，越大越平滑)
        degree: 样条阶数（1=线性, 2=二次, 3=三次）

    Returns:
        (num_points, 3) 平滑后的路径坐标 (float64)
    """
    if len(path) < 2:
        pts = np.array(path, dtype=np.float64)
        if len(pts) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.tile(pts[0], (num_points, 1))

    points = np.array(path, dtype=np.float64)

    # 去重连续相同点
    mask = np.ones(len(points), dtype=bool)
    for i in range(1, len(points)):
        if np.array_equal(points[i], points[i - 1]):
            mask[i] = False
    points = points[mask]

    if len(points) < 2:
        return np.tile(points[0], (num_points, 1))

    # 确保 degree 不超过点数 - 1
    k = min(degree, len(points) - 1)

    try:
        tck, u = splprep(
            [points[:, 0], points[:, 1], points[:, 2]],
            s=smoothing,
            k=k,
        )
        u_new = np.linspace(0, 1, num_points)
        smooth_pts = splev(u_new, tck)
        return np.column_stack(smooth_pts)
    except Exception:
        # 如果样条拟合失败（如共线点过多），降级为线性插值
        return _linear_resample(points, num_points)


def _linear_resample(points: np.ndarray, num_points: int) -> np.ndarray:
    """线性重采样作为回退。"""
    # 计算累积弧长
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < 1e-12:
        return np.tile(points[0], (num_points, 1))

    # 等间距参数
    target_lengths = np.linspace(0, total_length, num_points)

    result = np.zeros((num_points, 3))
    for i, t in enumerate(target_lengths):
        # 找到对应的线段
        idx = np.searchsorted(cum_length, t, side='right') - 1
        idx = max(0, min(idx, len(points) - 2))

        seg_len = seg_lengths[idx]
        if seg_len < 1e-12:
            result[i] = points[idx]
        else:
            frac = (t - cum_length[idx]) / seg_len
            result[i] = points[idx] + frac * diffs[idx]

    return result
