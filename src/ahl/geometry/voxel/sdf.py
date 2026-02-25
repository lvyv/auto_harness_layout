"""三维有符号距离场 (SDF) 计算。

利用 scipy EDT 计算体素网格到最近障碍物的欧氏距离场。
SDF 用于 A* 搜索的代价惩罚项，使路径远离障碍物表面。
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import distance_transform_edt


class SDFComputer:
    """三维 SDF 计算器。

    主要功能：
    - 从 Grid3D 计算距离场
    - 计算 SDF 梯度 (∇SDF)
    """

    @staticmethod
    def compute(grid) -> np.ndarray:
        """计算到最近障碍物的欧氏距离场。

        Args:
            grid: Grid3D 实例

        Returns:
            (Nx, Ny, Nz) float32 数组
            - 障碍物处为 0.0
            - 其他位置为到最近障碍物的距离（体素单位）
        """
        from .grid3d import CellType
        obstacle_mask = (grid.data == CellType.OBSTACLE)
        # EDT 计算 ~mask（True=自由空间）中每个点到最近 False（障碍物）的距离
        distances = distance_transform_edt(~obstacle_mask)
        return distances.astype(np.float32)

    @staticmethod
    def compute_from_mask(obstacle_mask: np.ndarray, spacing: float = 1.0) -> np.ndarray:
        """从布尔掩码计算 SDF。

        Args:
            obstacle_mask: (Nx, Ny, Nz) 布尔数组，True = 障碍物
            spacing: 体素间距（用于得到真实世界距离）

        Returns:
            (Nx, Ny, Nz) float32 距离场
        """
        distances = distance_transform_edt(~obstacle_mask, sampling=spacing)
        return distances.astype(np.float32)

    @staticmethod
    def gradient(sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算 SDF 梯度（中心差分）。

        梯度方向指向远离障碍物的方向，|∇SDF| ≈ 1。

        Args:
            sdf: (Nx, Ny, Nz) SDF 数组

        Returns:
            (grad_x, grad_y, grad_z) 三个 (Nx, Ny, Nz) float32 数组
        """
        # numpy.gradient 对每个轴做中心差分，边界用单侧差分
        gx, gy, gz = np.gradient(sdf)
        return gx.astype(np.float32), gy.astype(np.float32), gz.astype(np.float32)
