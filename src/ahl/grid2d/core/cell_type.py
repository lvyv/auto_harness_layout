"""Cell type enumeration for 2D grid."""

from enum import IntEnum


class CellType(IntEnum):
    """Cell type enumeration with rendering colors.

    Uses IntEnum so values can be stored directly in numpy int8 arrays.
    """

    FREE = 0        # White, free space
    OBSTACLE = 1    # Black, obstacle
    START = 2       # Green, start point
    END = 3         # Red, end point
    PATH = 4        # Blue, path

    @classmethod
    def get_color(cls, cell_type: 'CellType') -> tuple[int, int, int]:
        """Get RGB color for cell type (0-255 range).

        Args:
            cell_type: The cell type

        Returns:
            RGB tuple (r, g, b)
        """
        colors = {
            cls.FREE: (255, 255, 255),      # White
            cls.OBSTACLE: (50, 50, 50),     # Dark gray
            cls.START: (0, 200, 0),         # Green
            cls.END: (200, 0, 0),           # Red
            cls.PATH: (0, 100, 255),        # Blue
        }
        return colors.get(cell_type, (255, 255, 255))

    @classmethod
    def is_walkable(cls, cell_type: 'CellType') -> bool:
        """Check if a cell type is walkable for path planning.

        Args:
            cell_type: The cell type

        Returns:
            True if the cell can be traversed
        """
        return cell_type in (cls.FREE, cls.START, cls.END, cls.PATH)
