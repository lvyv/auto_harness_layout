"""Grid visualization and interaction widget."""

from typing import Optional
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QMouseEvent, QWheelEvent

from ..core.grid import Grid
from ..core.cell_type import CellType


class GridWidget(QWidget):
    """Widget for rendering and interacting with a 2D grid.

    Features:
        - Zoom and pan with mouse
        - Left-click to edit cells
        - Right-click drag to pan
        - Viewport clipping for performance on large grids

    Signals:
        cell_clicked(row, col): Emitted when a cell is clicked
        grid_changed(): Emitted when grid is modified
    """

    cell_clicked = pyqtSignal(int, int)  # row, col
    grid_changed = pyqtSignal()

    def __init__(self, grid: Optional[Grid] = None, parent=None):
        """Initialize grid widget.

        Args:
            grid: Grid instance to display
            parent: Parent widget
        """
        super().__init__(parent)

        self.grid = grid if grid is not None else Grid(20, 20)

        # View parameters
        self.cell_size = 20.0  # Pixels per cell
        self.offset_x = 0.0    # View offset in pixels
        self.offset_y = 0.0

        # Zoom limits
        self.min_cell_size = 2.0
        self.max_cell_size = 100.0

        # Interaction state
        self.edit_mode = CellType.OBSTACLE  # Current edit mode
        self.is_panning = False
        self.last_mouse_pos = None
        self.last_edited_cell = None  # For drag editing

        # Rendering
        self.setMinimumSize(400, 400)
        self.setMouseTracking(False)

    def set_grid(self, grid: Grid) -> None:
        """Set a new grid to display.

        Args:
            grid: Grid instance
        """
        self.grid = grid
        self.update()
        self.grid_changed.emit()

    def set_edit_mode(self, mode: CellType) -> None:
        """Set the current edit mode.

        Args:
            mode: Cell type to draw
        """
        self.edit_mode = mode

    def paintEvent(self, event):
        """Render the grid with viewport clipping."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Background
        painter.fillRect(self.rect(), QColor(200, 200, 200))

        if self.grid is None:
            return

        # Calculate visible cell range
        view_rect = self.rect()
        min_col = max(0, int(-self.offset_x / self.cell_size))
        min_row = max(0, int(-self.offset_y / self.cell_size))
        max_col = min(self.grid.width - 1, int((view_rect.width() - self.offset_x) / self.cell_size) + 1)
        max_row = min(self.grid.height - 1, int((view_rect.height() - self.offset_y) / self.cell_size) + 1)

        # Draw cells (only visible ones)
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                self._draw_cell(painter, row, col)

        # Draw grid lines (if zoomed in enough)
        if self.cell_size >= 4.0:
            self._draw_grid_lines(painter, min_row, max_row, min_col, max_col)

    def _draw_cell(self, painter: QPainter, row: int, col: int) -> None:
        """Draw a single cell.

        Args:
            painter: QPainter instance
            row: Cell row
            col: Cell column
        """
        cell_type = self.grid.get_cell(row, col)
        color = CellType.get_color(cell_type)

        x = col * self.cell_size + self.offset_x
        y = row * self.cell_size + self.offset_y

        painter.fillRect(
            QRectF(x, y, self.cell_size, self.cell_size),
            QColor(*color)
        )

    def _draw_grid_lines(self, painter: QPainter, min_row: int, max_row: int, min_col: int, max_col: int) -> None:
        """Draw grid lines.

        Args:
            painter: QPainter instance
            min_row: Minimum visible row
            max_row: Maximum visible row
            min_col: Minimum visible column
            max_col: Maximum visible column
        """
        pen = QPen(QColor(150, 150, 150), 1)
        painter.setPen(pen)

        # Vertical lines
        for col in range(min_col, max_col + 2):
            x = col * self.cell_size + self.offset_x
            y1 = min_row * self.cell_size + self.offset_y
            y2 = (max_row + 1) * self.cell_size + self.offset_y
            painter.drawLine(QPointF(x, y1), QPointF(x, y2))

        # Horizontal lines
        for row in range(min_row, max_row + 2):
            y = row * self.cell_size + self.offset_y
            x1 = min_col * self.cell_size + self.offset_x
            x2 = (max_col + 1) * self.cell_size + self.offset_x
            painter.drawLine(QPointF(x1, y), QPointF(x2, y))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Edit cell
            row, col = self._screen_to_grid(event.position().x(), event.position().y())
            if self.grid.is_valid(row, col):
                self._edit_cell(row, col)
                self.last_edited_cell = (row, col)

        elif event.button() == Qt.MouseButton.RightButton:
            # Start panning
            self.is_panning = True
            self.last_mouse_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move events."""
        if self.is_panning and self.last_mouse_pos is not None:
            # Pan view
            delta = event.position() - self.last_mouse_pos
            self.offset_x += delta.x()
            self.offset_y += delta.y()
            self.last_mouse_pos = event.position()
            self.update()

        elif event.buttons() & Qt.MouseButton.LeftButton:
            # Drag editing
            row, col = self._screen_to_grid(event.position().x(), event.position().y())
            if self.grid.is_valid(row, col) and (row, col) != self.last_edited_cell:
                self._edit_cell(row, col)
                self.last_edited_cell = (row, col)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release events."""
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.last_edited_cell = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel events for zooming."""
        # Get mouse position in grid coordinates before zoom
        mouse_x = event.position().x()
        mouse_y = event.position().y()
        grid_x_before = (mouse_x - self.offset_x) / self.cell_size
        grid_y_before = (mouse_y - self.offset_y) / self.cell_size

        # Zoom
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        new_cell_size = self.cell_size * zoom_factor

        # Clamp cell size
        new_cell_size = max(self.min_cell_size, min(self.max_cell_size, new_cell_size))

        if new_cell_size != self.cell_size:
            self.cell_size = new_cell_size

            # Adjust offset to keep mouse position fixed in grid coordinates
            self.offset_x = mouse_x - grid_x_before * self.cell_size
            self.offset_y = mouse_y - grid_y_before * self.cell_size

            self.update()

    def _screen_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """Convert screen coordinates to grid coordinates.

        Args:
            x: Screen x coordinate
            y: Screen y coordinate

        Returns:
            (row, col) grid coordinates
        """
        col = int((x - self.offset_x) / self.cell_size)
        row = int((y - self.offset_y) / self.cell_size)
        return row, col

    def _edit_cell(self, row: int, col: int) -> None:
        """Edit a cell based on current edit mode.

        Args:
            row: Cell row
            col: Cell column
        """
        if self.edit_mode == CellType.START:
            self.grid.add_start(row, col)
        elif self.edit_mode == CellType.END:
            self.grid.add_end(row, col)
        elif self.edit_mode == CellType.OBSTACLE:
            current = self.grid.get_cell(row, col)
            # Toggle obstacle: if already obstacle, make free
            if current == CellType.OBSTACLE:
                self.grid.set_cell(row, col, CellType.FREE)
            elif current in (CellType.FREE, CellType.PATH):
                self.grid.set_cell(row, col, CellType.OBSTACLE)
        elif self.edit_mode == CellType.FREE:
            # Eraser mode
            current = self.grid.get_cell(row, col)
            if current == CellType.START:
                self.grid.remove_start(row, col)
            elif current == CellType.END:
                self.grid.remove_end(row, col)
            else:
                self.grid.set_cell(row, col, CellType.FREE)

        self.update()
        self.cell_clicked.emit(row, col)
        self.grid_changed.emit()

    def reset_view(self) -> None:
        """Reset view to show entire grid."""
        if self.grid is None:
            return

        # Calculate cell size to fit grid in view
        view_rect = self.rect()
        h_cell_size = view_rect.width() / self.grid.width
        v_cell_size = view_rect.height() / self.grid.height
        self.cell_size = min(h_cell_size, v_cell_size) * 0.9  # 90% to add margin

        # Clamp
        self.cell_size = max(self.min_cell_size, min(self.max_cell_size, self.cell_size))

        # Center grid
        grid_width = self.grid.width * self.cell_size
        grid_height = self.grid.height * self.cell_size
        self.offset_x = (view_rect.width() - grid_width) / 2
        self.offset_y = (view_rect.height() - grid_height) / 2

        self.update()
