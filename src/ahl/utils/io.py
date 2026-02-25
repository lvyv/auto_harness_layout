"""文件 IO 工具（npz / json）。

统一的序列化接口，供各模块保存/加载中间结果。
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def save_npz(path: str, compress: bool = True, **arrays) -> None:
    """保存多个 numpy 数组到 npz 文件。

    Args:
        path: 文件路径 (.npz)
        compress: 是否压缩（默认 True）
        **arrays: 关键字参数，名称→数组
    """
    if compress:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """从 npz 文件加载所有数组。

    Args:
        path: 文件路径

    Returns:
        {名称: ndarray} 字典
    """
    npz = np.load(path, allow_pickle=False)
    return dict(npz)


def save_json(path: str, data: Any, indent: int = 2) -> None:
    """保存数据到 JSON 文件。

    支持 numpy 标量和数组的自动转换。

    Args:
        path: 文件路径 (.json)
        data: 可序列化的数据
        indent: 缩进级别
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=_json_default)


def load_json(path: str) -> Any:
    """从 JSON 文件加载数据。

    Args:
        path: 文件路径

    Returns:
        解析后的数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _json_default(obj):
    """JSON 序列化回调，处理 numpy 类型。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"无法序列化类型 {type(obj)}")
