"""Tests for Grid class."""

import pytest
import numpy as np

from ahl.grid2d.core.grid import Grid
from ahl.grid2d.core.cell_type import CellType
from ahl.grid2d.utils.validators import GridConfig


class TestGrid:
    """Test Grid class functionality."""

    def test_create_grid(self):
        """Test grid creation with default parameters."""
        grid = Grid(10, 20)
        assert grid.width == 10
        assert grid.height == 20
        assert grid.shape == (20, 10)
        assert grid.cells.dtype == np.int8
        assert np.all(grid.cells == CellType.FREE)

    def test_create_grid_with_obstacles(self):
        """Test grid creation with obstacle default."""
        grid = Grid(5, 5, default_cell=CellType.OBSTACLE)
        assert np.all(grid.cells == CellType.OBSTACLE)

    def test_grid_config_validation(self):
        """Test grid size validation."""
        # Valid sizes
        Grid(1, 1)
        Grid(1000, 1000)

        # Invalid sizes
        with pytest.raises(Exception):
            Grid(0, 10)
        with pytest.raises(Exception):
            Grid(10, 0)
        with pytest.raises(Exception):
            Grid(1001, 10)

    def test_is_valid(self):
        """Test coordinate validation."""
        grid = Grid(10, 20)

        assert grid.is_valid(0, 0)
        assert grid.is_valid(19, 9)
        assert not grid.is_valid(-1, 0)
        assert not grid.is_valid(0, -1)
        assert not grid.is_valid(20, 0)
        assert not grid.is_valid(0, 10)

    def test_get_set_cell(self):
        """Test cell get/set operations."""
        grid = Grid(10, 10)

        # Set and get
        grid.set_cell(5, 5, CellType.OBSTACLE)
        assert grid.get_cell(5, 5) == CellType.OBSTACLE

        # Out of bounds
        with pytest.raises(IndexError):
            grid.get_cell(10, 10)
        with pytest.raises(IndexError):
            grid.set_cell(10, 10, CellType.FREE)

    def test_add_start_point(self):
        """Test adding start points."""
        grid = Grid(10, 10)

        # Add start
        idx = grid.add_start(5, 5)
        assert idx == 0
        assert len(grid.starts) == 1
        assert grid.starts[0] == (5, 5)
        assert grid.get_cell(5, 5) == CellType.START

        # Add another
        idx = grid.add_start(3, 3)
        assert idx == 1
        assert len(grid.starts) == 2

        # Add duplicate (should not add again)
        idx = grid.add_start(5, 5)
        assert idx == 0
        assert len(grid.starts) == 2

    def test_remove_start_point(self):
        """Test removing start points."""
        grid = Grid(10, 10)

        grid.add_start(5, 5)
        assert len(grid.starts) == 1

        result = grid.remove_start(5, 5)
        assert result is True
        assert len(grid.starts) == 0
        assert grid.get_cell(5, 5) == CellType.FREE

        # Remove non-existent
        result = grid.remove_start(3, 3)
        assert result is False

    def test_add_end_point(self):
        """Test adding end points."""
        grid = Grid(10, 10)

        idx = grid.add_end(7, 7)
        assert idx == 0
        assert len(grid.ends) == 1
        assert grid.ends[0] == (7, 7)
        assert grid.get_cell(7, 7) == CellType.END

    def test_remove_end_point(self):
        """Test removing end points."""
        grid = Grid(10, 10)

        grid.add_end(7, 7)
        result = grid.remove_end(7, 7)
        assert result is True
        assert len(grid.ends) == 0
        assert grid.get_cell(7, 7) == CellType.FREE

    def test_clear_paths(self):
        """Test clearing paths."""
        grid = Grid(10, 10)

        # Add some path cells
        path = [(1, 1), (1, 2), (1, 3)]
        grid.set_path(0, 0, path)

        assert len(grid.paths) == 1
        assert grid.get_cell(1, 1) == CellType.PATH

        # Clear
        grid.clear_paths()
        assert len(grid.paths) == 0
        assert grid.get_cell(1, 1) == CellType.FREE

    def test_fill_rect(self):
        """Test rectangular fill."""
        grid = Grid(10, 10)

        grid.fill_rect(2, 2, 4, 4, CellType.OBSTACLE)

        # Check filled area
        for r in range(2, 5):
            for c in range(2, 5):
                assert grid.get_cell(r, c) == CellType.OBSTACLE

        # Check outside area
        assert grid.get_cell(1, 1) == CellType.FREE
        assert grid.get_cell(5, 5) == CellType.FREE

    def test_clear_grid(self):
        """Test clearing entire grid."""
        grid = Grid(10, 10)

        grid.fill_rect(0, 0, 9, 9, CellType.OBSTACLE)
        grid.add_start(5, 5)
        grid.add_end(7, 7)

        grid.clear(CellType.FREE)

        assert np.all(grid.cells == CellType.FREE)
        assert len(grid.starts) == 0
        assert len(grid.ends) == 0

    def test_sdf_caching(self):
        """Test SDF caching mechanism."""
        grid = Grid(10, 10)

        # First call computes SDF
        sdf1 = grid.get_sdf()
        assert sdf1 is not None
        assert grid._sdf is not None
        assert not grid._sdf_dirty

        # Second call returns cached
        sdf2 = grid.get_sdf()
        assert sdf2 is sdf1  # Same object

        # Modify obstacle -> marks dirty
        grid.set_cell(5, 5, CellType.OBSTACLE)
        assert grid._sdf_dirty

        # Next call recomputes
        sdf3 = grid.get_sdf()
        assert not grid._sdf_dirty
        assert sdf3 is not sdf1  # Different object
