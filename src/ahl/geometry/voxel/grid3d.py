"""三维体素网格核心类。

Grid3D 是整个几何引擎层的基础数据结构，管理三维体素网格状态，
支持坐标变换、邻域查询和 SDF 缓存。
"""

from enum import IntEnum
from typing import List, Tuple, Optional, Iterator
import numpy as np


class CellType(IntEnum):
    """体素单元类型。

    用 IntEnum 以便直接存储在 numpy int8 数组中。
    """
    FREE = 0        # 自由空间（可通行）
    OBSTACLE = 1    # 障碍物（不可通行）
    SURFACE = 2     # 表面（可布线表面）


# 6-邻域偏移（面连通）
NEIGHBORS_6 = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.int32)

# 26-邻域偏移（全连通）
_offsets_26 = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dy == 0 and dz == 0:
                continue
            _offsets_26.append([dx, dy, dz])
NEIGHBORS_26 = np.array(_offsets_26, dtype=np.int32)


class Grid3D:
    """三维体素网格，支持坐标变换、邻域查询与 SDF 缓存。

    Attributes:
        data: (Nx, Ny, Nz) 体素状态数组 (int8)
        resolution: 体素尺寸 (mm)
        origin: 世界坐标系原点 (3,)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        resolution: float = 1.0,
        origin: Optional[np.ndarray] = None,
    ):
        """初始化三维体素网格。

        Args:
            nx: X 方向格数
            ny: Y 方向格数
            nz: Z 方向格数
            resolution: 体素尺寸 (mm)
            origin: 世界坐标原点，默认 (0,0,0)
        """
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError(f"网格尺寸必须为正整数，收到 ({nx}, {ny}, {nz})")
        if resolution <= 0:
            raise ValueError(f"分辨率必须为正数，收到 {resolution}")

        self.data = np.full((nx, ny, nz), CellType.FREE, dtype=np.int8)
        self.resolution = float(resolution)
        self.origin = np.array(origin, dtype=np.float64) if origin is not None else np.zeros(3)

        # SDF 缓存
        self._sdf: Optional[np.ndarray] = None
        self._sdf_dirty: bool = True

    # ========================= 属性 =========================

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape

    @property
    def nx(self) -> int:
        return self.data.shape[0]

    @property
    def ny(self) -> int:
        return self.data.shape[1]

    @property
    def nz(self) -> int:
        return self.data.shape[2]

    @property
    def size(self) -> int:
        """总体素数量。"""
        return self.data.size

    # ========================= 坐标变换 =========================

    def voxel_to_world(self, ijk: np.ndarray) -> np.ndarray:
        """体素索引 → 世界坐标（体素中心）。

        Args:
            ijk: (..., 3) 体素索引数组

        Returns:
            (..., 3) 世界坐标数组
        """
        ijk = np.asarray(ijk, dtype=np.float64)
        return self.origin + (ijk + 0.5) * self.resolution

    def world_to_voxel(self, coords: np.ndarray) -> np.ndarray:
        """世界坐标 → 体素索引（四舍五入到最近体素）。

        Args:
            coords: (..., 3) 世界坐标数组

        Returns:
            (..., 3) 体素索引数组 (int32)
        """
        coords = np.asarray(coords, dtype=np.float64)
        ijk = (coords - self.origin) / self.resolution - 0.5
        return np.round(ijk).astype(np.int32)

    # ========================= 访问与修改 =========================

    def is_valid(self, i: int, j: int, k: int) -> bool:
        """检查索引是否在网格边界内。"""
        return 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz

    def get_cell(self, i: int, j: int, k: int) -> int:
        """获取体素状态。"""
        if not self.is_valid(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self.shape}")
        return int(self.data[i, j, k])

    def set_cell(self, i: int, j: int, k: int, cell_type: int) -> None:
        """设置体素状态。"""
        if not self.is_valid(i, j, k):
            raise IndexError(f"索引 ({i}, {j}, {k}) 超出网格范围 {self.shape}")
        old = self.data[i, j, k]
        self.data[i, j, k] = cell_type
        if old == CellType.OBSTACLE or cell_type == CellType.OBSTACLE:
            self._sdf_dirty = True

    def is_free(self, i: int, j: int, k: int) -> bool:
        """该体素是否可通行（FREE 或 SURFACE）。"""
        if not self.is_valid(i, j, k):
            return False
        v = int(self.data[i, j, k])
        return v == CellType.FREE or v == CellType.SURFACE

    # ========================= 批量操作 =========================

    def set_obstacle_mask(self, mask: np.ndarray) -> None:
        """用布尔掩码批量设置障碍物。

        Args:
            mask: (Nx, Ny, Nz) 布尔数组，True 位置标记为 OBSTACLE
        """
        if mask.shape != self.shape:
            raise ValueError(f"掩码形状 {mask.shape} 与网格 {self.shape} 不匹配")
        self.data[mask] = CellType.OBSTACLE
        self._sdf_dirty = True

    def set_surface_mask(self, mask: np.ndarray) -> None:
        """用布尔掩码批量设置表面体素。

        Args:
            mask: (Nx, Ny, Nz) 布尔数组，True 位置标记为 SURFACE
        """
        if mask.shape != self.shape:
            raise ValueError(f"掩码形状 {mask.shape} 与网格 {self.shape} 不匹配")
        self.data[mask] = CellType.SURFACE
        self._sdf_dirty = True

    def obstacle_count(self) -> int:
        """障碍物体素数量。"""
        return int(np.count_nonzero(self.data == CellType.OBSTACLE))

    def free_count(self) -> int:
        """自由空间体素数量。"""
        return int(np.count_nonzero(self.data != CellType.OBSTACLE))

    # ========================= 邻域查询 =========================

    def get_neighbors(
        self,
        i: int, j: int, k: int,
        connectivity: int = 6,
    ) -> List[Tuple[int, int, int]]:
        """获取有效邻域体素索引。

        Args:
            i, j, k: 体素索引
            connectivity: 6（面连通）或 26（全连通）

        Returns:
            邻域体素索引列表
        """
        offsets = NEIGHBORS_6 if connectivity == 6 else NEIGHBORS_26
        neighbors = []
        for di, dj, dk in offsets:
            ni, nj, nk = i + di, j + dj, k + dk
            if self.is_valid(ni, nj, nk):
                neighbors.append((ni, nj, nk))
        return neighbors

    def get_free_neighbors(
        self,
        i: int, j: int, k: int,
        connectivity: int = 6,
    ) -> List[Tuple[int, int, int]]:
        """获取可通行的邻域体素索引。"""
        return [
            (ni, nj, nk)
            for ni, nj, nk in self.get_neighbors(i, j, k, connectivity)
            if self.is_free(ni, nj, nk)
        ]

    # ========================= SDF =========================

    def get_sdf(self) -> np.ndarray:
        """获取或计算 SDF（带缓存）。

        Returns:
            (Nx, Ny, Nz) 到最近障碍物的欧氏距离场 (float32)
        """
        if self._sdf is None or self._sdf_dirty:
            from .sdf import SDFComputer
            self._sdf = SDFComputer.compute(self)
            self._sdf_dirty = False
        return self._sdf

    def invalidate_sdf(self) -> None:
        """强制标记 SDF 缓存过期。"""
        self._sdf_dirty = True

    # ========================= IO =========================

    def save(self, path: str) -> None:
        """保存到 npz 文件。

        Args:
            path: 文件路径 (.npz)
        """
        np.savez_compressed(
            path,
            data=self.data,
            resolution=np.array([self.resolution]),
            origin=self.origin,
        )

    @classmethod
    def load(cls, path: str) -> 'Grid3D':
        """从 npz 文件加载。

        Args:
            path: 文件路径 (.npz)

        Returns:
            Grid3D 实例
        """
        npz = np.load(path)
        data = npz['data']
        resolution = float(npz['resolution'][0])
        origin = npz['origin']

        nx, ny, nz = data.shape
        grid = cls(nx, ny, nz, resolution=resolution, origin=origin)
        grid.data = data.astype(np.int8)
        return grid

    # ========================= 膨胀操作 =========================

    def dilate_obstacles(self, radius: int = 1) -> None:
        """对障碍物进行形态学膨胀（安全距离扩展）。

        将障碍物体素向外扩展 radius 个体素，用于生成安全布线区。
        直接在 data 上操作。

        Args:
            radius: 膨胀半径（体素单位）
        """
        from scipy.ndimage import binary_dilation, generate_binary_structure

        obstacle_mask = (self.data == CellType.OBSTACLE)
        struct = generate_binary_structure(3, 1)  # 6-连通结构元素
        dilated = binary_dilation(obstacle_mask, structure=struct, iterations=radius)

        # 只将新扩展的区域标记为障碍（保留原 SURFACE 不被覆盖的逻辑可按需调整）
        new_obstacles = dilated & ~obstacle_mask
        self.data[new_obstacles] = CellType.OBSTACLE
        self._sdf_dirty = True

    # ========================= 表示 =========================

    def __repr__(self) -> str:
        obs = self.obstacle_count()
        return (
            f"Grid3D(shape={self.shape}, resolution={self.resolution}, "
            f"obstacles={obs}/{self.size})"
        )
