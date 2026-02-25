"""Grid3D 单元测试。"""

import tempfile
import numpy as np
import pytest

from ahl.geometry.voxel.grid3d import Grid3D, CellType, NEIGHBORS_6, NEIGHBORS_26


class TestGrid3DCreation:
    """测试 Grid3D 创建与基本属性。"""

    def test_basic_creation(self):
        grid = Grid3D(10, 20, 30)
        assert grid.shape == (10, 20, 30)
        assert grid.nx == 10
        assert grid.ny == 20
        assert grid.nz == 30
        assert grid.resolution == 1.0
        assert grid.size == 10 * 20 * 30

    def test_custom_resolution_and_origin(self):
        origin = np.array([1.0, 2.0, 3.0])
        grid = Grid3D(5, 5, 5, resolution=0.5, origin=origin)
        assert grid.resolution == 0.5
        np.testing.assert_array_equal(grid.origin, origin)

    def test_all_cells_free_by_default(self):
        grid = Grid3D(3, 3, 3)
        assert np.all(grid.data == CellType.FREE)

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            Grid3D(0, 5, 5)
        with pytest.raises(ValueError):
            Grid3D(5, -1, 5)

    def test_invalid_resolution_raise(self):
        with pytest.raises(ValueError):
            Grid3D(5, 5, 5, resolution=0)


class TestCoordinateTransform:
    """测试坐标变换。"""

    def test_voxel_to_world_default_origin(self):
        grid = Grid3D(10, 10, 10, resolution=2.0)
        # 体素 (0,0,0) 的中心应在世界坐标 (1.0, 1.0, 1.0)
        world = grid.voxel_to_world(np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(world, [1.0, 1.0, 1.0])

    def test_voxel_to_world_with_origin(self):
        grid = Grid3D(10, 10, 10, resolution=1.0, origin=np.array([10.0, 20.0, 30.0]))
        world = grid.voxel_to_world(np.array([0, 0, 0]))
        np.testing.assert_array_almost_equal(world, [10.5, 20.5, 30.5])

    def test_world_to_voxel_roundtrip(self):
        grid = Grid3D(10, 10, 10, resolution=0.5, origin=np.array([5.0, 5.0, 5.0]))
        ijk_orig = np.array([3, 4, 5])
        world = grid.voxel_to_world(ijk_orig)
        ijk_back = grid.world_to_voxel(world)
        np.testing.assert_array_equal(ijk_back, ijk_orig)

    def test_batch_transform(self):
        grid = Grid3D(10, 10, 10, resolution=1.0)
        ijk = np.array([[0, 0, 0], [1, 2, 3], [9, 9, 9]])
        world = grid.voxel_to_world(ijk)
        assert world.shape == (3, 3)
        ijk_back = grid.world_to_voxel(world)
        np.testing.assert_array_equal(ijk_back, ijk)


class TestCellAccess:
    """测试体素读写。"""

    def test_get_set_cell(self):
        grid = Grid3D(5, 5, 5)
        grid.set_cell(1, 2, 3, CellType.OBSTACLE)
        assert grid.get_cell(1, 2, 3) == CellType.OBSTACLE

    def test_out_of_bounds_raises(self):
        grid = Grid3D(5, 5, 5)
        with pytest.raises(IndexError):
            grid.get_cell(5, 0, 0)
        with pytest.raises(IndexError):
            grid.set_cell(-1, 0, 0, CellType.FREE)

    def test_is_free(self):
        grid = Grid3D(5, 5, 5)
        assert grid.is_free(0, 0, 0) is True
        grid.set_cell(0, 0, 0, CellType.OBSTACLE)
        assert grid.is_free(0, 0, 0) is False
        grid.set_cell(0, 0, 0, CellType.SURFACE)
        assert grid.is_free(0, 0, 0) is True

    def test_is_free_out_of_bounds(self):
        grid = Grid3D(5, 5, 5)
        assert grid.is_free(-1, 0, 0) is False
        assert grid.is_free(5, 0, 0) is False


class TestBatchOperations:
    """测试批量操作。"""

    def test_set_obstacle_mask(self):
        grid = Grid3D(5, 5, 5)
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, :, :] = True
        grid.set_obstacle_mask(mask)
        assert grid.obstacle_count() == 25
        assert grid.data[0, 0, 0] == CellType.OBSTACLE
        assert grid.data[1, 0, 0] == CellType.FREE

    def test_mask_shape_mismatch_raises(self):
        grid = Grid3D(5, 5, 5)
        with pytest.raises(ValueError):
            grid.set_obstacle_mask(np.zeros((3, 3, 3), dtype=bool))

    def test_obstacle_count(self):
        grid = Grid3D(10, 10, 10)
        assert grid.obstacle_count() == 0
        grid.set_cell(0, 0, 0, CellType.OBSTACLE)
        grid.set_cell(1, 1, 1, CellType.OBSTACLE)
        assert grid.obstacle_count() == 2


class TestNeighbors:
    """测试邻域查询。"""

    def test_6_neighbors_interior(self):
        grid = Grid3D(5, 5, 5)
        neighbors = grid.get_neighbors(2, 2, 2, connectivity=6)
        assert len(neighbors) == 6

    def test_26_neighbors_interior(self):
        grid = Grid3D(5, 5, 5)
        neighbors = grid.get_neighbors(2, 2, 2, connectivity=26)
        assert len(neighbors) == 26

    def test_6_neighbors_corner(self):
        grid = Grid3D(5, 5, 5)
        neighbors = grid.get_neighbors(0, 0, 0, connectivity=6)
        assert len(neighbors) == 3  # 只有 +x, +y, +z

    def test_free_neighbors_skip_obstacles(self):
        grid = Grid3D(5, 5, 5)
        grid.set_cell(1, 0, 0, CellType.OBSTACLE)
        grid.set_cell(0, 1, 0, CellType.OBSTACLE)
        neighbors = grid.get_free_neighbors(0, 0, 0, connectivity=6)
        assert len(neighbors) == 1  # 只有 (0, 0, 1)


class TestIO:
    """测试保存/加载。"""

    def test_save_load_roundtrip(self, tmp_path):
        grid = Grid3D(8, 6, 4, resolution=0.5, origin=np.array([1.0, 2.0, 3.0]))
        grid.set_cell(0, 0, 0, CellType.OBSTACLE)
        grid.set_cell(7, 5, 3, CellType.SURFACE)

        path = str(tmp_path / "test_grid.npz")
        grid.save(path)

        loaded = Grid3D.load(path)
        assert loaded.shape == grid.shape
        assert loaded.resolution == grid.resolution
        np.testing.assert_array_equal(loaded.origin, grid.origin)
        np.testing.assert_array_equal(loaded.data, grid.data)


class TestDilation:
    """测试膨胀操作。"""

    def test_dilate_expands_obstacle(self):
        grid = Grid3D(11, 11, 11)
        # 中心放一个障碍
        grid.set_cell(5, 5, 5, CellType.OBSTACLE)
        assert grid.obstacle_count() == 1

        grid.dilate_obstacles(radius=1)
        # 膨胀 1 步后，6-连通应该至少有 7 个障碍 (原+6邻居)
        assert grid.obstacle_count() >= 7

    def test_dilate_marks_sdf_dirty(self):
        grid = Grid3D(5, 5, 5)
        grid.set_cell(2, 2, 2, CellType.OBSTACLE)
        _ = grid.get_sdf()  # 计算 SDF，标记为 clean
        assert grid._sdf_dirty is False

        grid.dilate_obstacles(radius=1)
        assert grid._sdf_dirty is True


class TestRepr:
    def test_repr(self):
        grid = Grid3D(5, 5, 5)
        s = repr(grid)
        assert "Grid3D" in s
        assert "(5, 5, 5)" in s
