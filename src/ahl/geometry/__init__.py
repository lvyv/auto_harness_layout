"""
几何引擎层 - CAD/Mesh输入与体素化

该模块负责：
- 三维体素网格生成与管理
- SDF（有符号距离场）计算
- Mesh加载与处理
- 空间索引结构（KD-Tree, Octree）
"""

from .voxel.grid3d import Grid3D
from .voxel.sdf import SDFComputer
from .voxel.voxelizer import Voxelizer

__all__ = [
    'Grid3D',
    'SDFComputer',
    'Voxelizer',
]
