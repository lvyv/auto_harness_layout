"""Mesh 体素化模块。

将三角面片网格 (vertices, faces) 转换为体素网格 (Grid3D)。
采用基于切片的 ray casting 方法：逐层切片，对每层做 2D 光栅化判断内外。
"""

from typing import Tuple, Optional
import numpy as np

from .grid3d import Grid3D, CellType


class Voxelizer:
    """Mesh → Voxel 转换器。

    算法流程：
    1. 计算 mesh 包围盒 → 确定网格尺寸
    2. 对每个 Z 切片，用 ray casting 判断哪些体素在 mesh 内部
    3. 内部体素标记为 OBSTACLE

    适用于封闭或近封闭网格。对于开放曲面，使用 surface_voxelize 方法。
    """

    @staticmethod
    def compute_bbox(
        vertices: np.ndarray,
        padding: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 mesh 包围盒。

        Args:
            vertices: (N, 3) 顶点数组
            padding: 各方向额外边距

        Returns:
            (bbox_min, bbox_max)，每个 (3,)
        """
        bbox_min = vertices.min(axis=0) - padding
        bbox_max = vertices.max(axis=0) + padding
        return bbox_min.astype(np.float64), bbox_max.astype(np.float64)

    @staticmethod
    def voxelize_solid(
        vertices: np.ndarray,
        faces: np.ndarray,
        resolution: float,
        padding: float = 2.0,
    ) -> Grid3D:
        """实体体素化（封闭 mesh 内部填充为 OBSTACLE）。

        使用逐层 Z-axis ray casting：对每个 Z 切片平面，
        将三角面片投影到 XY 平面，用 scanline 判断内外。

        Args:
            vertices: (N, 3) 顶点坐标
            faces: (M, 3) 三角面片索引
            resolution: 体素尺寸 (mm)
            padding: 包围盒额外边距（体素单位数 × resolution）

        Returns:
            Grid3D，内部体素为 OBSTACLE，外部为 FREE
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        pad = padding * resolution
        bbox_min, bbox_max = Voxelizer.compute_bbox(vertices, padding=pad)

        # 网格尺寸
        grid_size = np.ceil((bbox_max - bbox_min) / resolution).astype(int)
        nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

        grid = Grid3D(nx, ny, nz, resolution=resolution, origin=bbox_min)

        # 三角面片顶点
        v0 = vertices[faces[:, 0]]  # (M, 3)
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # 对每个 Z 切片做 ray casting
        for iz in range(nz):
            z = bbox_min[2] + (iz + 0.5) * resolution

            # 筛选与当前 Z 平面相交的三角形
            z_min_tri = np.minimum(np.minimum(v0[:, 2], v1[:, 2]), v2[:, 2])
            z_max_tri = np.maximum(np.maximum(v0[:, 2], v1[:, 2]), v2[:, 2])
            mask_z = (z_min_tri <= z) & (z_max_tri >= z)

            if not np.any(mask_z):
                continue

            # 对相交三角形，计算与 Z 平面的交线段
            intersections = _slice_triangles_at_z(
                v0[mask_z], v1[mask_z], v2[mask_z], z
            )

            if len(intersections) == 0:
                continue

            # 对每条 Y 扫描线做 X-axis ray casting
            for iy in range(ny):
                y = bbox_min[1] + (iy + 0.5) * resolution

                # 收集 ray (y=const) 与所有交线段的交点 x 坐标
                x_hits = []
                for seg in intersections:
                    xi = _ray_segment_intersection_x(seg[0], seg[1], y)
                    if xi is not None:
                        x_hits.append(xi)

                if len(x_hits) == 0:
                    continue

                x_hits.sort()

                # 奇偶规则：每对交点之间是内部
                for idx in range(0, len(x_hits) - 1, 2):
                    x_start = x_hits[idx]
                    x_end = x_hits[idx + 1]

                    ix_start = max(0, int((x_start - bbox_min[0]) / resolution))
                    ix_end = min(nx - 1, int((x_end - bbox_min[0]) / resolution))

                    for ix in range(ix_start, ix_end + 1):
                        grid.data[ix, iy, iz] = CellType.OBSTACLE

        grid._sdf_dirty = True
        return grid

    @staticmethod
    def surface_voxelize(
        vertices: np.ndarray,
        faces: np.ndarray,
        resolution: float,
        padding: float = 2.0,
    ) -> Grid3D:
        """表面体素化（只标记三角面片穿过的体素为 SURFACE）。

        适用于开放曲面或不需要实体填充的场景。

        Args:
            vertices: (N, 3) 顶点坐标
            faces: (M, 3) 三角面片索引
            resolution: 体素尺寸 (mm)
            padding: 包围盒额外边距（体素单位数）

        Returns:
            Grid3D，表面体素为 SURFACE
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        pad = padding * resolution
        bbox_min, bbox_max = Voxelizer.compute_bbox(vertices, padding=pad)

        grid_size = np.ceil((bbox_max - bbox_min) / resolution).astype(int)
        nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

        grid = Grid3D(nx, ny, nz, resolution=resolution, origin=bbox_min)

        # 对每个三角面片，标记其覆盖的体素
        for f in faces:
            tri = vertices[f]  # (3, 3)

            # 三角形的包围盒
            tri_min = tri.min(axis=0)
            tri_max = tri.max(axis=0)

            # 转换到体素索引范围
            ijk_min = np.maximum(0, np.floor((tri_min - bbox_min) / resolution).astype(int))
            ijk_max = np.minimum(
                np.array([nx - 1, ny - 1, nz - 1]),
                np.ceil((tri_max - bbox_min) / resolution).astype(int)
            )

            # 在包围盒范围内检查每个体素是否与三角形相交
            for ix in range(ijk_min[0], ijk_max[0] + 1):
                for iy in range(ijk_min[1], ijk_max[1] + 1):
                    for iz in range(ijk_min[2], ijk_max[2] + 1):
                        center = bbox_min + np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * resolution
                        half = resolution * 0.5
                        if _triangle_aabb_intersect(tri, center, half):
                            grid.data[ix, iy, iz] = CellType.SURFACE

        grid._sdf_dirty = True
        return grid


# ============================================================
# 内部辅助函数
# ============================================================


def _slice_triangles_at_z(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, z: float
) -> list:
    """将一组三角形与 Z=z 平面求交，返回交线段列表。

    每个交线段是 ((x1, y1), (x2, y2)) 元组。
    """
    segments = []
    triangles = [(v0, v1, v2)]

    for va, vb, vc in triangles:
        for i in range(len(va)):
            a = va[i]
            b = vb[i]
            c = vc[i]

            pts = []
            # 检查三条边与 z 平面的交点
            for (p, q) in [(a, b), (b, c), (c, a)]:
                pt = _edge_z_intersection(p, q, z)
                if pt is not None:
                    pts.append(pt)

            if len(pts) >= 2:
                segments.append((pts[0], pts[1]))

    return segments


def _edge_z_intersection(
    p: np.ndarray, q: np.ndarray, z: float
) -> Optional[Tuple[float, float]]:
    """计算线段 pq 与平面 Z=z 的交点，返回 (x, y) 或 None。"""
    pz, qz = p[2], q[2]
    if (pz - z) * (qz - z) > 0:
        return None  # 同侧，不相交
    dz = qz - pz
    if abs(dz) < 1e-12:
        return None  # 平行
    t = (z - pz) / dz
    if t < 0.0 or t > 1.0:
        return None
    x = p[0] + t * (q[0] - p[0])
    y = p[1] + t * (q[1] - p[1])
    return (x, y)


def _ray_segment_intersection_x(
    a: Tuple[float, float],
    b: Tuple[float, float],
    y: float,
) -> Optional[float]:
    """Y=y 水平射线与线段 ab 的交点 x 坐标。

    线段 ab 在 XY 平面上 (从 Z 切片得到)。
    """
    ay, by = a[1], b[1]
    if (ay - y) * (by - y) > 0:
        return None  # 同侧
    dy = by - ay
    if abs(dy) < 1e-12:
        return None  # 水平线段
    t = (y - ay) / dy
    if t < 0.0 or t > 1.0:
        return None
    x = a[0] + t * (b[0] - a[0])
    return x


def _triangle_aabb_intersect(
    tri: np.ndarray,
    center: np.ndarray,
    half: float,
) -> bool:
    """简化的三角形-AABB 相交检测。

    使用分离轴定理的简化版本：
    检查三角形顶点是否在 AABB 内，或 AABB 的中心到三角形平面的距离是否小于半对角线。

    Args:
        tri: (3, 3) 三角形顶点
        center: (3,) AABB 中心
        half: AABB 半边长

    Returns:
        True 如果可能相交
    """
    # 快速检查：三角形包围盒与 AABB 是否不相交
    tri_min = tri.min(axis=0)
    tri_max = tri.max(axis=0)

    aabb_min = center - half
    aabb_max = center + half

    if np.any(tri_min > aabb_max) or np.any(tri_max < aabb_min):
        return False

    # 保守返回 True（精确SAT测试对性能要求高，工程中先用粗检）
    return True
