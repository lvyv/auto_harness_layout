"""Blender 集成模块。

注意：本模块依赖 Blender Python API (bpy)，
只能在 Blender 内嵌 Python 环境中运行。

在标准 Python 环境下导入会抛出 ImportError。
"""

import importlib

_HAS_BPY = importlib.util.find_spec("bpy") is not None

if _HAS_BPY:
    from .importer import ModelImporter
    from .mesh_processor import MeshProcessor
    from .voxelizer import BlenderVoxelizer
    from .exporter import CurveExporter

    __all__ = [
        'ModelImporter', 'MeshProcessor',
        'BlenderVoxelizer', 'CurveExporter',
    ]
else:
    __all__ = []
