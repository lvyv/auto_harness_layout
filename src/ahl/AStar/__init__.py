"""
2D Grid Editor and A* Path Planning System

A complete 2D grid-based path planning system with:
- Interactive PyQt6 editor
- A* algorithm with SDF (Signed Distance Field) penalty
- Multiple start/end points support
- NPZ file I/O
"""

from .core.cell_type import CellType
from .core.grid import Grid, GridConfig
from .core.astar import astar_search, AStarConfig

__all__ = [
    'CellType',
    'Grid',
    'GridConfig',
    'astar_search',
    'AStarConfig',
]
