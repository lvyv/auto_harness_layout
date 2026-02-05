"""NPZ file format handler for saving/loading grids."""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np

from ..core.grid import Grid


class NPZHandler:
    """Handler for NPZ file format (compressed numpy arrays).

    File format:
        - cells: (H, W) int8 array
        - starts: (N, 2) int32 array of start positions
        - ends: (M, 2) int32 array of end positions
        - path_<start_idx>_<end_idx>: (L, 2) int32 array for each path
        - config_width: int32
        - config_height: int32
        - config_default_cell: int8
        - metadata_version: string
        - metadata_timestamp: string
    """

    VERSION = "1.0"

    @staticmethod
    def save(grid: Grid, file_path: str | Path) -> None:
        """Save grid to NPZ file.

        Args:
            grid: Grid instance to save
            file_path: Path to save file

        Raises:
            IOError: If file cannot be written
        """
        file_path = Path(file_path)

        # Prepare data dictionary
        data: Dict[str, Any] = {}

        # Core grid data
        data['cells'] = grid.cells

        # Start and end points
        data['starts'] = np.array(grid.starts, dtype=np.int32) if grid.starts else np.empty((0, 2), dtype=np.int32)
        data['ends'] = np.array(grid.ends, dtype=np.int32) if grid.ends else np.empty((0, 2), dtype=np.int32)

        # Paths
        for (start_idx, end_idx), path in grid.paths.items():
            key = f'path_{start_idx}_{end_idx}'
            data[key] = np.array(path, dtype=np.int32)

        # Configuration
        data['config_width'] = np.int32(grid.config.width)
        data['config_height'] = np.int32(grid.config.height)
        data['config_default_cell'] = np.int8(grid.config.default_cell)

        # Metadata
        data['metadata_version'] = NPZHandler.VERSION
        data['metadata_timestamp'] = datetime.now().isoformat()

        # Save compressed
        np.savez_compressed(file_path, **data)

    @staticmethod
    def load(file_path: str | Path) -> Grid:
        """Load grid from NPZ file.

        Args:
            file_path: Path to NPZ file

        Returns:
            Loaded Grid instance

        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise IOError(f"File not found: {file_path}")

        # Load data
        data = np.load(file_path, allow_pickle=False)

        # Check version
        version = str(data.get('metadata_version', ''))
        if not version.startswith('1.'):
            raise ValueError(f"Unsupported file version: {version}")

        # Extract configuration
        width = int(data['config_width'])
        height = int(data['config_height'])
        default_cell = int(data['config_default_cell'])

        # Create grid
        grid = Grid(width=width, height=height, default_cell=default_cell)

        # Load cells
        grid.cells = data['cells'].astype(np.int8)

        # Load starts and ends
        starts = data['starts']
        if starts.size > 0:
            grid.starts = [(int(r), int(c)) for r, c in starts]

        ends = data['ends']
        if ends.size > 0:
            grid.ends = [(int(r), int(c)) for r, c in ends]

        # Load paths
        for key in data.files:
            if key.startswith('path_'):
                # Parse key: 'path_<start_idx>_<end_idx>'
                parts = key.split('_')
                if len(parts) == 3:
                    try:
                        start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        path_array = data[key]
                        path = [(int(r), int(c)) for r, c in path_array]
                        grid.paths[(start_idx, end_idx)] = path
                    except (ValueError, IndexError):
                        pass  # Skip invalid path keys

        return grid


# Convenience functions
def save_grid(grid: Grid, file_path: str | Path) -> None:
    """Save grid to NPZ file.

    Args:
        grid: Grid to save
        file_path: Path to save file
    """
    NPZHandler.save(grid, file_path)


def load_grid(file_path: str | Path) -> Grid:
    """Load grid from NPZ file.

    Args:
        file_path: Path to NPZ file

    Returns:
        Loaded Grid instance
    """
    return NPZHandler.load(file_path)
