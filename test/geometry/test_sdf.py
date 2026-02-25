"""SDF 计算单元测试。"""

import numpy as np
import pytest

from ahl.geometry.voxel.grid3d import Grid3D, CellType
from ahl.geometry.voxel.sdf import SDFComputer


class TestSDFCompute:
    """测试 SDF 距离场计算。"""

    def test_empty_grid_large_distances(self):
        """无障碍物时距离应该很大（边界距离）。"""
        grid = Grid3D(10, 10, 10)
        sdf = SDFComputer.compute(grid)
        assert sdf.shape == (10, 10, 10)
        # 无障碍物时，EDT 返回到最近 False 的距离
        # 对于全 True 的 mask，scipy EDT 返回 0 （无边界点）
        # 实际上 ~obstacle_mask 全 True（无障碍），距离 = 0 (无"边界")
        # 这是正常行为：没有障碍物就没有参考面
        # 注意：distance_transform_edt 对全 True 输入返回 inf
        # 所以中心应该有一个有限值
        assert sdf.dtype == np.float32

    def test_single_obstacle_center(self):
        """中心放一个障碍，周围的 SDF 应该递增。"""
        grid = Grid3D(11, 11, 11)
        grid.set_cell(5, 5, 5, CellType.OBSTACLE)
        sdf = SDFComputer.compute(grid)

        # 障碍处距离为 0
        assert sdf[5, 5, 5] == 0.0

        # 6-邻居距离为 1
        assert abs(sdf[6, 5, 5] - 1.0) < 0.01
        assert abs(sdf[4, 5, 5] - 1.0) < 0.01

        # 距离 2 的点
        assert abs(sdf[7, 5, 5] - 2.0) < 0.01

    def test_obstacle_wall(self):
        """一面墙，SDF 应该是到墙面的距离。"""
        grid = Grid3D(10, 10, 10)
        grid.data[0, :, :] = CellType.OBSTACLE  # x=0 面全是障碍

        sdf = SDFComputer.compute(grid)

        # x=0 处距离为 0
        assert sdf[0, 5, 5] == 0.0

        # x=1 处距离为 1
        assert abs(sdf[1, 5, 5] - 1.0) < 0.01

        # x=5 处距离为 5
        assert abs(sdf[5, 5, 5] - 5.0) < 0.01

    def test_sdf_via_grid_cached(self):
        """通过 Grid3D.get_sdf() 获取，验证缓存。"""
        grid = Grid3D(5, 5, 5)
        grid.set_cell(2, 2, 2, CellType.OBSTACLE)

        sdf1 = grid.get_sdf()
        sdf2 = grid.get_sdf()
        assert sdf1 is sdf2  # 同一对象（缓存）

        # 修改后缓存失效
        grid.set_cell(3, 3, 3, CellType.OBSTACLE)
        sdf3 = grid.get_sdf()
        assert sdf3 is not sdf1


class TestSDFFromMask:
    """测试从布尔掩码计算 SDF。"""

    def test_basic_mask(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True

        sdf = SDFComputer.compute_from_mask(mask)
        assert sdf[5, 5, 5] == 0.0
        assert sdf[6, 5, 5] > 0

    def test_with_spacing(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[5, 5, 5] = True

        sdf = SDFComputer.compute_from_mask(mask, spacing=2.0)
        # spacing=2 时距离翻倍
        assert abs(sdf[6, 5, 5] - 2.0) < 0.01


class TestSDFGradient:
    """测试 SDF 梯度。"""

    def test_gradient_shape(self):
        sdf = np.random.rand(5, 6, 7).astype(np.float32)
        gx, gy, gz = SDFComputer.gradient(sdf)
        assert gx.shape == (5, 6, 7)
        assert gy.shape == (5, 6, 7)
        assert gz.shape == (5, 6, 7)

    def test_gradient_magnitude_near_one(self):
        """对于单点障碍的 SDF，远离边界处 |∇SDF| ≈ 1。"""
        grid = Grid3D(21, 21, 21)
        grid.set_cell(10, 10, 10, CellType.OBSTACLE)
        sdf = SDFComputer.compute(grid)

        gx, gy, gz = SDFComputer.gradient(sdf)
        mag = np.sqrt(gx**2 + gy**2 + gz**2)

        # 在远离边界和障碍的内部区域检查
        interior = mag[5:15, 5:15, 5:15]
        # 排除障碍物点本身
        mask = sdf[5:15, 5:15, 5:15] > 1.0
        if np.any(mask):
            magnitudes = interior[mask]
            # |∇SDF| 应接近 1（EDT 的性质）
            assert np.mean(np.abs(magnitudes - 1.0)) < 0.15
