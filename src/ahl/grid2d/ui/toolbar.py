"""Toolbar widget for grid editor."""

from PyQt6.QtWidgets import QToolBar, QWidget, QLabel, QComboBox
from PyQt6.QtGui import QAction
from PyQt6.QtCore import pyqtSignal

from ..core.cell_type import CellType


class EditorToolbar(QToolBar):
    """Toolbar for grid editor actions.

    Signals:
        new_grid(): Create new grid
        save_grid(): Save current grid
        load_grid(): Load grid from file
        edit_mode_changed(CellType): Edit mode changed
        reset_view(): Reset view to fit grid
    """

    new_grid = pyqtSignal()
    save_grid = pyqtSignal()
    load_grid = pyqtSignal()
    edit_mode_changed = pyqtSignal(object)  # CellType
    reset_view = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize toolbar.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setMovable(False)
        self._setup_actions()

    def _setup_actions(self) -> None:
        """Setup toolbar actions."""
        # File actions
        new_action = QAction("New", self)
        new_action.setToolTip("Create a new grid")
        new_action.triggered.connect(self.new_grid.emit)
        self.addAction(new_action)

        save_action = QAction("Save", self)
        save_action.setToolTip("Save grid to file")
        save_action.triggered.connect(self.save_grid.emit)
        self.addAction(save_action)

        load_action = QAction("Load", self)
        load_action.setToolTip("Load grid from file")
        load_action.triggered.connect(self.load_grid.emit)
        self.addAction(load_action)

        self.addSeparator()

        # View actions
        reset_view_action = QAction("Reset View", self)
        reset_view_action.setToolTip("Reset view to fit grid")
        reset_view_action.triggered.connect(self.reset_view.emit)
        self.addAction(reset_view_action)

        self.addSeparator()

        # Edit mode selector
        self.addWidget(QLabel("  Edit Mode: "))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Draw Obstacle", CellType.OBSTACLE)
        self.mode_combo.addItem("Place Start", CellType.START)
        self.mode_combo.addItem("Place End", CellType.END)
        self.mode_combo.addItem("Eraser", CellType.FREE)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.addWidget(self.mode_combo)

    def _on_mode_changed(self, index: int) -> None:
        """Handle edit mode change.

        Args:
            index: Combo box index
        """
        mode = self.mode_combo.itemData(index)
        self.edit_mode_changed.emit(mode)

    def get_current_mode(self) -> CellType:
        """Get current edit mode.

        Returns:
            Current CellType edit mode
        """
        return self.mode_combo.currentData()
