"""布线结果导出（Blender 环境）。

将路径转换为 Blender Curve 对象，用于可视化和 CAD 导出。
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


class CurveExporter:
    """路径 → Blender Curve 导出器。

    Usage (在 Blender 中):
        exporter = CurveExporter()
        obj = exporter.create_curve(path_points, name="harness_01")
    """

    @staticmethod
    def create_curve(
        points: np.ndarray,
        name: str = "HarnessPath",
        bevel_depth: float = 0.5,
        resolution_u: int = 12,
    ):
        """从点序列创建 Blender NURBS Curve 对象。

        Args:
            points: (N, 3) 路径点坐标
            name: 对象名称
            bevel_depth: 管道半径
            resolution_u: 曲线细分级别

        Returns:
            Blender Curve 对象
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        points = np.asarray(points, dtype=np.float64)

        # 创建曲线数据
        curve_data = bpy.data.curves.new(name=name, type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = resolution_u
        curve_data.bevel_depth = bevel_depth

        # 创建样条线
        spline = curve_data.splines.new('NURBS')
        spline.points.add(len(points) - 1)  # 已有 1 个点

        for i, pt in enumerate(points):
            spline.points[i].co = (pt[0], pt[1], pt[2], 1.0)

        spline.use_endpoint_u = True

        # 创建对象并链接到场景
        curve_obj = bpy.data.objects.new(name, curve_data)
        bpy.context.collection.objects.link(curve_obj)

        return curve_obj

    @staticmethod
    def export_paths(
        paths: List[np.ndarray],
        name_prefix: str = "Harness",
        bevel_depth: float = 0.3,
        collection_name: str = "HarnessRouting",
    ) -> list:
        """批量导出多条路径。

        Args:
            paths: 路径列表，每条为 (N, 3) 数组
            name_prefix: 名称前缀
            bevel_depth: 管道半径
            collection_name: Blender Collection 名称

        Returns:
            Blender Curve 对象列表
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        # 创建或获取 Collection
        if collection_name not in bpy.data.collections:
            col = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(col)
        else:
            col = bpy.data.collections[collection_name]

        objects = []
        for i, path in enumerate(paths):
            name = f"{name_prefix}_{i:03d}"
            obj = CurveExporter.create_curve(
                path, name=name, bevel_depth=bevel_depth,
            )
            # 移到指定 Collection
            if obj.name in bpy.context.collection.objects:
                bpy.context.collection.objects.unlink(obj)
            col.objects.link(obj)
            objects.append(obj)

        return objects
