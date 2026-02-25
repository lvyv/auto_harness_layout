"""Blender Mesh → Voxel 转换。

利用 Blender bmesh 的几何查询能力做体素化，
替代纯 numpy 实现以获得更好的精度（尤其是复杂曲面）。
"""

from typing import Optional

try:
    import bpy
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
except ImportError:
    bpy = None
    bmesh = None
    BVHTree = None

import numpy as np

from ahl.geometry.voxel.grid3d import Grid3D, CellType


class BlenderVoxelizer:
    """基于 Blender BVHTree 的体素化器。

    利用 BVHTree.ray_cast 做精确的内外判定。

    Usage (在 Blender 中):
        voxelizer = BlenderVoxelizer()
        grid = voxelizer.voxelize(obj, resolution=1.0)
    """

    @staticmethod
    def voxelize(
        obj,
        resolution: float = 1.0,
        padding: int = 2,
        fill_interior: bool = True,
    ) -> Grid3D:
        """将 Blender mesh 对象体素化。

        Args:
            obj: Blender mesh 对象
            resolution: 体素尺寸 (与 Blender 单位一致)
            padding: 边界额外体素数
            fill_interior: True=填充内部, False=仅标记表面

        Returns:
            Grid3D 实例
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        mesh = obj.data
        mesh.calc_loop_triangles()

        # 构建 BVHTree
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.transform(obj.matrix_world)
        bvh = BVHTree.FromBMesh(bm)

        # 包围盒
        verts = np.array([v.co[:] for v in bm.verts])
        bbox_min = verts.min(axis=0) - padding * resolution
        bbox_max = verts.max(axis=0) + padding * resolution

        grid_size = np.ceil((bbox_max - bbox_min) / resolution).astype(int)
        nx, ny, nz = int(grid_size[0]), int(grid_size[1]), int(grid_size[2])

        grid = Grid3D(nx, ny, nz, resolution=resolution, origin=bbox_min)

        if fill_interior:
            _fill_interior_raycast(grid, bvh, bbox_min, resolution, nx, ny, nz)
        else:
            _mark_surface(grid, bvh, bbox_min, resolution, nx, ny, nz)

        bm.free()
        return grid


def _fill_interior_raycast(grid, bvh, origin, res, nx, ny, nz):
    """用 ray cast 判断内外，填充内部为 OBSTACLE。"""
    ray_dir = Vector((1, 0, 0))  # X 正方向

    for j in range(ny):
        for k in range(nz):
            y = origin[1] + (j + 0.5) * res
            z = origin[2] + (k + 0.5) * res

            # 从左边界发射射线
            ray_origin = Vector((origin[0] - res, y, z))
            hits = []

            pos = ray_origin.copy()
            for _ in range(nx * 2):  # 安全上限
                loc, normal, idx, dist = bvh.ray_cast(pos, ray_dir)
                if loc is None:
                    break
                hits.append(loc.x)
                # 稍微推进以跳过当前交点
                pos = loc + ray_dir * 1e-5

            # 奇偶规则
            hits.sort()
            for idx in range(0, len(hits) - 1, 2):
                x_start = hits[idx]
                x_end = hits[idx + 1]

                ix_start = max(0, int((x_start - origin[0]) / res))
                ix_end = min(nx - 1, int((x_end - origin[0]) / res))

                for i in range(ix_start, ix_end + 1):
                    grid.data[i, j, k] = CellType.OBSTACLE


def _mark_surface(grid, bvh, origin, res, nx, ny, nz):
    """标记与 mesh 表面相交的体素为 SURFACE。"""
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                center = Vector((
                    origin[0] + (i + 0.5) * res,
                    origin[1] + (j + 0.5) * res,
                    origin[2] + (k + 0.5) * res,
                ))
                loc, normal, idx, dist = bvh.find_nearest(center)
                if loc is not None and dist < res * 0.87:  # sqrt(3)/2
                    grid.data[i, j, k] = CellType.SURFACE
