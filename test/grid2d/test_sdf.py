"""Tests for SDF computation."""

import pytest
import numpy as np

from ahl.grid2d.core.sdf import compute_sdf
from ahl.grid2d.core.cell_type import CellType


class TestSDF:
    """Test SDF computation."""

    def test_sdf_empty_grid(self):
        """Test SDF on grid with no obstacles."""
        cells = np.zeros((10, 10), dtype=np.int8)
        sdf = compute_sdf(cells)

        # All cells should have large distance (no obstacles)
        assert np.all(sdf > 0)
        # Max distance should be on the order of grid diagonal
        assert np.max(sdf) > 5.0

    def test_sdf_single_obstacle(self):
        """Test SDF with single obstacle."""
        cells = np.zeros((10, 10), dtype=np.int8)
        cells[5, 5] = CellType.OBSTACLE

        sdf = compute_sdf(cells)

        # Obstacle cell has distance 0
        assert sdf[5, 5] == 0.0

        # Adjacent cells have distance ~1
        assert abs(sdf[5, 4] - 1.0) < 0.1
        assert abs(sdf[5, 6] - 1.0) < 0.1
        assert abs(sdf[4, 5] - 1.0) < 0.1
        assert abs(sdf[6, 5] - 1.0) < 0.1

        # Diagonal cells have distance ~sqrt(2)
        assert abs(sdf[4, 4] - np.sqrt(2)) < 0.1

    def test_sdf_wall(self):
        """Test SDF with obstacle wall."""
        cells = np.zeros((10, 10), dtype=np.int8)
        cells[:, 5] = CellType.OBSTACLE  # Vertical wall

        sdf = compute_sdf(cells)

        # Wall cells have distance 0
        assert np.all(sdf[:, 5] == 0.0)

        # Cells one away have distance 1
        assert np.all(abs(sdf[:, 4] - 1.0) < 0.1)
        assert np.all(abs(sdf[:, 6] - 1.0) < 0.1)

        # Cells at edges have max distance
        assert sdf[5, 0] > 4.0
        assert sdf[5, 9] > 3.0

    def test_sdf_shape(self):
        """Test SDF output shape matches input."""
        for shape in [(5, 5), (10, 20), (100, 50)]:
            cells = np.zeros(shape, dtype=np.int8)
            sdf = compute_sdf(cells)
            assert sdf.shape == shape

    def test_sdf_dtype(self):
        """Test SDF output dtype is float32."""
        cells = np.zeros((10, 10), dtype=np.int8)
        sdf = compute_sdf(cells)
        assert sdf.dtype == np.float32

    def test_sdf_all_obstacles(self):
        """Test SDF when entire grid is obstacles."""
        cells = np.ones((10, 10), dtype=np.int8)
        sdf = compute_sdf(cells)

        # All cells should be 0 (all obstacles)
        assert np.all(sdf == 0.0)
