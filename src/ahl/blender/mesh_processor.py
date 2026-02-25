"""Mesh 清理与预处理（Blender 环境）。"""

try:
    import bpy
    import bmesh
except ImportError:
    bpy = None
    bmesh = None


class MeshProcessor:
    """Mesh 预处理器。

    提供清理、修复、简化等操作。
    """

    @staticmethod
    def cleanup(obj, merge_distance: float = 0.001) -> None:
        """清理 mesh：移除重叠顶点、孤立顶点、退化面。

        Args:
            obj: Blender mesh 对象
            merge_distance: 合并距离阈值
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')

        bm = bmesh.from_edit_mesh(obj.data)

        # 合并近距离顶点
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)

        # 删除退化面（面积为0）
        degenerate = [f for f in bm.faces if f.calc_area() < 1e-8]
        bmesh.ops.delete(bm, geom=degenerate, context='FACES')

        # 删除孤立顶点
        loose = [v for v in bm.verts if not v.link_faces]
        bmesh.ops.delete(bm, geom=loose, context='VERTS')

        bmesh.update_edit_mesh(obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def decimate(obj, ratio: float = 0.5) -> None:
        """简化 mesh（减面）。

        Args:
            obj: Blender mesh 对象
            ratio: 面数比例 (0~1)
        """
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
        mod.ratio = ratio
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    @staticmethod
    def recalc_normals(obj) -> None:
        """重新计算法向量（朝外）。"""
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

    @staticmethod
    def get_stats(obj) -> dict:
        """获取 mesh 统计信息。"""
        if bpy is None:
            raise RuntimeError("需要在 Blender 环境中运行")

        mesh = obj.data
        return {
            'vertices': len(mesh.vertices),
            'edges': len(mesh.edges),
            'faces': len(mesh.polygons),
        }
