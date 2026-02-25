"""CAD 模型导入（Blender 环境）。

支持 STEP/IGES/STL/OBJ 格式导入。
使用 Blender 原生 API，不依赖 trimesh 等外部库。
"""

from typing import List, Optional
from pathlib import Path

try:
    import bpy
    import bmesh
except ImportError:
    bpy = None
    bmesh = None


class ModelImporter:
    """CAD 模型导入器。

    Usage (在 Blender 中):
        importer = ModelImporter()
        objects = importer.import_file("model.stl")
    """

    @staticmethod
    def import_file(filepath: str, scale: float = 1.0) -> list:
        """导入模型文件。

        Args:
            filepath: 文件路径
            scale: 缩放系数

        Returns:
            导入的 Blender 对象列表

        Raises:
            RuntimeError: 不在 Blender 环境中
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        if bpy is None:
            raise RuntimeError("ModelImporter 需要在 Blender 环境中运行")

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        suffix = path.suffix.lower()
        before = set(bpy.data.objects.keys())

        if suffix == '.stl':
            bpy.ops.import_mesh.stl(filepath=str(path))
        elif suffix == '.obj':
            bpy.ops.wm.obj_import(filepath=str(path))
        elif suffix in ('.fbx',):
            bpy.ops.import_scene.fbx(filepath=str(path))
        elif suffix in ('.step', '.stp'):
            # STEP 需要额外插件 (如 CAD Sketcher 或 FreeCAD bridge)
            raise ValueError(
                f"STEP 格式需要 Blender STEP 导入插件。"
                f"建议先用 FreeCAD 转换为 STL。"
            )
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

        after = set(bpy.data.objects.keys())
        new_names = after - before

        objects = [bpy.data.objects[name] for name in new_names]

        # 应用缩放
        if scale != 1.0:
            for obj in objects:
                obj.scale = (scale, scale, scale)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(scale=True)

        return objects

    @staticmethod
    def get_mesh_data(obj) -> dict:
        """从 Blender 对象提取 mesh 数据。

        Args:
            obj: Blender mesh 对象

        Returns:
            {'vertices': ndarray(N,3), 'faces': ndarray(M,3)}
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        import numpy as np

        mesh = obj.data
        mesh.calc_loop_triangles()

        vertices = np.array([v.co[:] for v in mesh.vertices], dtype=np.float64)
        faces = np.array(
            [tri.vertices[:] for tri in mesh.loop_triangles],
            dtype=np.int32,
        )

        return {'vertices': vertices, 'faces': faces}
