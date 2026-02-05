"""Tests for file I/O."""

import pytest
import numpy as np


from ahl.grid2d.core.grid import Grid
from ahl.grid2d.core.cell_type import CellType
from ahl.grid2d.io.npz_handler import save_grid, load_grid


class TestIO:
    """Test NPZ file I/O."""

    def test_save_load_empty_grid(self, tmp_path):
        """Test saving and loading empty grid."""
        grid = Grid(10, 10)

        # Save
        file_path = tmp_path / "test.npz"
        save_grid(grid, file_path)

        assert file_path.exists()

        # Load
        loaded = load_grid(file_path)

        assert loaded.width == grid.width
        assert loaded.height == grid.height
        assert np.array_equal(loaded.cells, grid.cells)

    def test_save_load_grid_with_obstacles(self, tmp_path):
        """Test saving grid with obstacles."""
        grid = Grid(20, 20)

        # Add obstacles
        grid.fill_rect(5, 5, 10, 10, CellType.OBSTACLE)

        # Save and load
        file_path = tmp_path / "test.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert np.array_equal(loaded.cells, grid.cells)

    def test_save_load_start_end_points(self, tmp_path):
        """Test saving grid with start and end points."""
        grid = Grid(15, 15)

        # Add multiple starts and ends
        grid.add_start(1, 1)
        grid.add_start(2, 2)
        grid.add_end(10, 10)
        grid.add_end(11, 11)
        grid.add_end(12, 12)

        # Save and load
        file_path = tmp_path / "test.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert loaded.starts == grid.starts
        assert loaded.ends == grid.ends

    def test_save_load_paths(self, tmp_path):
        """Test saving grid with computed paths."""
        grid = Grid(20, 20)

        # Add starts and ends
        grid.add_start(0, 0)
        grid.add_start(5, 5)
        grid.add_end(19, 19)

        # Add some paths
        path1 = [(0, 0), (0, 1), (0, 2)]
        path2 = [(5, 5), (6, 6), (7, 7)]

        grid.set_path(0, 0, path1)
        grid.set_path(1, 0, path2)

        # Save and load
        file_path = tmp_path / "test.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert len(loaded.paths) == len(grid.paths)
        assert loaded.paths[(0, 0)] == path1
        assert loaded.paths[(1, 0)] == path2

    def test_save_load_config(self, tmp_path):
        """Test that configuration is preserved."""
        grid = Grid(width=25, height=30, default_cell=CellType.FREE)

        file_path = tmp_path / "test.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert loaded.config.width == 25
        assert loaded.config.height == 30
        assert loaded.config.default_cell == CellType.FREE

    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(IOError):
            load_grid("nonexistent.npz")

    def test_save_load_roundtrip(self, tmp_path):
        """Test complete roundtrip preserves all data."""
        # Create complex grid
        grid = Grid(50, 50)

        # Add obstacles
        grid.fill_rect(10, 10, 20, 20, CellType.OBSTACLE)

        # Add starts and ends
        for i in range(5):
            grid.add_start(i, i)
            grid.add_end(40 + i, 40 + i)

        # Add paths
        grid.set_path(0, 0, [(0, 0), (1, 1), (2, 2)])
        grid.set_path(1, 1, [(1, 1), (2, 2), (3, 3)])

        # Save and load
        file_path = tmp_path / "roundtrip.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        # Verify all data
        assert loaded.width == grid.width
        assert loaded.height == grid.height
        assert np.array_equal(loaded.cells, grid.cells)
        assert loaded.starts == grid.starts
        assert loaded.ends == grid.ends
        assert loaded.paths == grid.paths

    def test_compressed_format(self, tmp_path):
        """Test that NPZ uses compression."""
        # Create large grid with repetitive data
        grid = Grid(200, 200)
        grid.fill_rect(0, 0, 199, 199, CellType.OBSTACLE)

        file_path = tmp_path / "compressed.npz"
        save_grid(grid, file_path)

        # File should be much smaller than raw array
        # Raw: 200*200 bytes = 40KB
        # Compressed should be < 5KB
        file_size = file_path.stat().st_size
        assert file_size < 5000  # Less than 5KB

    def test_empty_starts_ends(self, tmp_path):
        """Test saving grid with no start/end points."""
        grid = Grid(10, 10)

        file_path = tmp_path / "empty.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert loaded.starts == []
        assert loaded.ends == []

    def test_large_grid(self, tmp_path):
        """Test save/load of large grid (performance check)."""
        grid = Grid(1000, 1000)

        # Add some data
        grid.fill_rect(400, 400, 600, 600, CellType.OBSTACLE)
        grid.add_start(100, 100)
        grid.add_end(900, 900)

        file_path = tmp_path / "large.npz"
        save_grid(grid, file_path)
        loaded = load_grid(file_path)

        assert loaded.width == 1000
        assert loaded.height == 1000
        assert np.array_equal(loaded.cells, grid.cells)
