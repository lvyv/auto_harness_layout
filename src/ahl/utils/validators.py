"""输入验证工具。

提供三维场景下常用的参数验证函数。
"""

from typing import Tuple, Sequence
import numpy as np


def validate_point_3d(point, name: str = "point") -> Tuple[int, int, int]:
    """验证并标准化三维整数坐标点。

    Args:
        point: 长度为 3 的序列
        name: 参数名称（用于错误信息）

    Returns:
        (i, j, k) 整数元组

    Raises:
        ValueError: 如果输入不合法
    """
    try:
        if len(point) != 3:
            raise ValueError
        return (int(point[0]), int(point[1]), int(point[2]))
    except (TypeError, ValueError, IndexError):
        raise ValueError(f"{name} 必须是长度为 3 的整数序列，收到 {point}")


def validate_positive(value: float, name: str = "value") -> float:
    """验证正数。"""
    if value <= 0:
        raise ValueError(f"{name} 必须为正数，收到 {value}")
    return float(value)


def validate_non_negative(value: float, name: str = "value") -> float:
    """验证非负数。"""
    if value < 0:
        raise ValueError(f"{name} 不能为负数，收到 {value}")
    return float(value)


def validate_array_shape(
    arr: np.ndarray,
    expected_shape: Tuple,
    name: str = "array",
) -> np.ndarray:
    """验证数组形状。支持 -1 表示任意维度。

    Args:
        arr: numpy 数组
        expected_shape: 期望形状，-1 表示该维度任意
        name: 参数名称

    Raises:
        ValueError: 形状不匹配
    """
    if len(arr.shape) != len(expected_shape):
        raise ValueError(
            f"{name} 维数不匹配：期望 {len(expected_shape)}D，收到 {len(arr.shape)}D"
        )
    for i, (actual, expected) in enumerate(zip(arr.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} 第 {i} 维大小不匹配：期望 {expected}，收到 {actual}"
            )
    return arr
