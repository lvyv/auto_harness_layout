"""数学工具函数（向量/矩阵运算）。

提供三维几何计算中常用的基础运算。
"""

import numpy as np
import math
from typing import Tuple, Union

# 类型别名
Vec3 = Union[np.ndarray, Tuple[float, float, float]]


def normalize(v: np.ndarray) -> np.ndarray:
    """向量归一化。

    Args:
        v: (..., N) 向量或向量数组

    Returns:
        归一化后的向量；零向量返回零向量
    """
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # 避免除零
    norm = np.where(norm == 0, 1.0, norm)
    return v / norm


def euclidean_distance(a: Vec3, b: Vec3) -> float:
    """两点间欧氏距离。

    Args:
        a, b: 三维坐标

    Returns:
        距离标量
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b))


def manhattan_distance(a: Vec3, b: Vec3) -> float:
    """两点间曼哈顿距离。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sum(np.abs(a - b)))


def chebyshev_distance(a: Vec3, b: Vec3) -> float:
    """两点间切比雪夫距离（26-邻域下的最少步数）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """两向量间夹角（弧度）。

    Args:
        v1, v2: 三维向量

    Returns:
        夹角 [0, pi]
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """三维叉积。"""
    return np.cross(np.asarray(v1, dtype=np.float64),
                    np.asarray(v2, dtype=np.float64))


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """线性插值 a + t * (b - a)。

    Args:
        a, b: 起止点
        t: 插值参数 [0, 1]

    Returns:
        插值点
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a + t * (b - a)


def bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算点集的轴对齐包围盒 (AABB)。

    Args:
        points: (N, 3) 点坐标数组

    Returns:
        (min_corner, max_corner)，每个 (3,)
    """
    points = np.asarray(points, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)
