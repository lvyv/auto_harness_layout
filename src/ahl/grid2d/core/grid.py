"""Core Grid class for 2D grid management."""

from typing import List, Tuple, Optional
import numpy as np

from .cell_type import CellType
from ..utils.validators import GridConfig


class Grid:
    """2D grid for path planning with obstacle management.

    Manages a 2D grid of cells with support for:
    - Multiple start and end points
    - Obstacle editing
    - SDF (Signed Distance Field) caching
    - Path storage

    Attributes:
        cells: (H, W) numpy array of cell types (int8)
        starts: List of (row, col) start point coordinates
        ends: List of (row, col) end point coordinates
        paths: Dictionary mapping (start, end) to path coordinates
        config: Grid configuration
    """

    def __init__(self, width: int, height: int, default_cell: int = 0):
        """Initialize grid.

        Args:
            width: Grid width (columns)
            height: Grid height (rows)
            default_cell: Default cell type (0=FREE)
        """
        # Validate configuration
        self.config = GridConfig(width=width, height=height, default_cell=default_cell)

        # Initialize grid
        self.cells = np.full((height, width), default_cell, dtype=np.int8)

        # Start and end points
        self.starts: List[Tuple[int, int]] = []
        self.ends: List[Tuple[int, int]] = []

        # Path storage: {(start_idx, end_idx): [(row, col), ...]}
        self.paths: dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # SDF cache
        self._sdf: Optional[np.ndarray] = None
        self._sdf_dirty: bool = True

    @property
    def width(self) -> int:
        """Grid width (number of columns)."""
        return self.cells.shape[1]

    @property
    def height(self) -> int:
        """Grid height (number of rows)."""
        return self.cells.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        """Grid shape as (height, width)."""
        return self.cells.shape

    def is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if coordinates are valid
        """
        return 0 <= row < self.height and 0 <= col < self.width

    def get_cell(self, row: int, col: int) -> int:
        """Get cell type at coordinates.

        Args:
            row: Row index
            col: Column index

        Returns:
            Cell type value

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not self.is_valid(row, col):
            raise IndexError(f"Coordinates ({row}, {col}) out of bounds for grid {self.shape}")
        return int(self.cells[row, col])

    def set_cell(self, row: int, col: int, cell_type: int) -> None:
        """Set cell type at coordinates.

        Args:
            row: Row index
            col: Column index
            cell_type: Cell type value (CellType enum value)

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not self.is_valid(row, col):
            raise IndexError(f"Coordinates ({row}, {col}) out of bounds for grid {self.shape}")

        old_type = self.cells[row, col]
        self.cells[row, col] = cell_type

        # Mark SDF dirty if obstacle changed
        if old_type == CellType.OBSTACLE or cell_type == CellType.OBSTACLE:
            self._sdf_dirty = True

    def add_start(self, row: int, col: int) -> int:
        """Add a start point.

        Args:
            row: Row index
            col: Column index

        Returns:
            Index of the added start point

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not self.is_valid(row, col):
            raise IndexError(f"Coordinates ({row}, {col}) out of bounds for grid {self.shape}")

        pos = (row, col)
        if pos not in self.starts:
            self.starts.append(pos)
            self.set_cell(row, col, CellType.START)
        return self.starts.index(pos)

    def remove_start(self, row: int, col: int) -> bool:
        """Remove a start point.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if a start point was removed
        """
        pos = (row, col)
        if pos in self.starts:
            self.starts.remove(pos)
            if self.get_cell(row, col) == CellType.START:
                self.set_cell(row, col, CellType.FREE)
            return True
        return False

    def add_end(self, row: int, col: int) -> int:
        """Add an end point.

        Args:
            row: Row index
            col: Column index

        Returns:
            Index of the added end point

        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not self.is_valid(row, col):
            raise IndexError(f"Coordinates ({row}, {col}) out of bounds for grid {self.shape}")

        pos = (row, col)
        if pos not in self.ends:
            self.ends.append(pos)
            self.set_cell(row, col, CellType.END)
        return self.ends.index(pos)

    def remove_end(self, row: int, col: int) -> bool:
        """Remove an end point.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if an end point was removed
        """
        pos = (row, col)
        if pos in self.ends:
            self.ends.remove(pos)
            if self.get_cell(row, col) == CellType.END:
                self.set_cell(row, col, CellType.FREE)
            return True
        return False

    def clear_paths(self) -> None:
        """Clear all stored paths and path cells."""
        self.paths.clear()

        # Clear path cells
        mask = self.cells == CellType.PATH
        self.cells[mask] = CellType.FREE

    def set_path(self, start_idx: int, end_idx: int, path: List[Tuple[int, int]]) -> None:
        """Store a computed path.

        Args:
            start_idx: Index of start point in self.starts
            end_idx: Index of end point in self.ends
            path: List of (row, col) coordinates forming the path
        """
        key = (start_idx, end_idx)
        self.paths[key] = path

        # Mark path cells (excluding start/end points)
        for row, col in path:
            if self.get_cell(row, col) not in (CellType.START, CellType.END):
                self.set_cell(row, col, CellType.PATH)

    def get_sdf(self) -> np.ndarray:
        """Get or compute SDF (Signed Distance Field).

        SDF is cached and only recomputed when obstacles change.

        Returns:
            (H, W) array of distances to nearest obstacle
        """
        if self._sdf is None or self._sdf_dirty:
            from .sdf import compute_sdf
            self._sdf = compute_sdf(self.cells)
            self._sdf_dirty = False
        return self._sdf

    def fill_rect(self, row1: int, col1: int, row2: int, col2: int, cell_type: int) -> None:
        """Fill a rectangular region with a cell type.

        Args:
            row1: Starting row
            col1: Starting column
            row2: Ending row (inclusive)
            col2: Ending column (inclusive)
            cell_type: Cell type to fill
        """
        # Ensure valid bounds
        row1, row2 = min(row1, row2), max(row1, row2)
        col1, col2 = min(col1, col2), max(col1, col2)

        row1 = max(0, row1)
        col1 = max(0, col1)
        row2 = min(self.height - 1, row2)
        col2 = min(self.width - 1, col2)

        # Fill region
        self.cells[row1:row2+1, col1:col2+1] = cell_type

        # Mark SDF dirty if obstacle changed
        if cell_type == CellType.OBSTACLE:
            self._sdf_dirty = True

    def clear(self, cell_type: int = 0) -> None:
        """Clear entire grid to a cell type.

        Args:
            cell_type: Cell type to fill (default: FREE)
        """
        self.cells.fill(cell_type)
        self.starts.clear()
        self.ends.clear()
        self.clear_paths()
        self._sdf_dirty = True
