"""Control panel widget for grid editor."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget, QHBoxLayout
)
from PyQt6.QtCore import pyqtSignal

from ..core.grid import Grid
from ..utils.validators import AStarConfig


class ControlPanel(QWidget):
    """Control panel for grid parameters and A* configuration.

    Signals:
        run_astar(): User clicked "Run A*" button
        clear_paths(): User clicked "Clear Paths" button
        config_changed(AStarConfig): A* configuration changed
    """

    run_astar = pyqtSignal()
    clear_paths = pyqtSignal()
    config_changed = pyqtSignal(object)  # AStarConfig

    def __init__(self, grid: Grid, parent=None):
        """Initialize control panel.

        Args:
            grid: Grid instance to control
            parent: Parent widget
        """
        super().__init__(parent)
        self.grid = grid
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Grid Info
        grid_group = QGroupBox("Grid Info")
        grid_layout = QVBoxLayout()
        self.size_label = QLabel(f"Size: {self.grid.width} × {self.grid.height}")
        self.stats_label = QLabel("Starts: 0 | Ends: 0")
        grid_layout.addWidget(self.size_label)
        grid_layout.addWidget(self.stats_label)
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)

        # A* Configuration
        astar_group = QGroupBox("A* Configuration")
        astar_layout = QVBoxLayout()

        # Diagonal movement
        self.diagonal_check = QCheckBox("Allow Diagonal Movement")
        self.diagonal_check.setChecked(False)
        self.diagonal_check.stateChanged.connect(self._on_config_changed)
        astar_layout.addWidget(self.diagonal_check)

        # SDF weight
        sdf_layout = QHBoxLayout()
        sdf_layout.addWidget(QLabel("SDF Weight:"))
        self.sdf_weight_spin = QDoubleSpinBox()
        self.sdf_weight_spin.setRange(0.0, 10.0)
        self.sdf_weight_spin.setSingleStep(0.1)
        self.sdf_weight_spin.setValue(0.5)
        self.sdf_weight_spin.valueChanged.connect(self._on_config_changed)
        sdf_layout.addWidget(self.sdf_weight_spin)
        astar_layout.addLayout(sdf_layout)

        # Max iterations
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("Max Iterations:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(1000, 10_000_000)
        self.max_iter_spin.setSingleStep(10000)
        self.max_iter_spin.setValue(1_000_000)
        self.max_iter_spin.valueChanged.connect(self._on_config_changed)
        iter_layout.addWidget(self.max_iter_spin)
        astar_layout.addLayout(iter_layout)

        astar_group.setLayout(astar_layout)
        layout.addWidget(astar_group)

        # Start/End Points Lists
        points_group = QGroupBox("Points")
        points_layout = QVBoxLayout()

        points_layout.addWidget(QLabel("Start Points:"))
        self.starts_list = QListWidget()
        self.starts_list.setMaximumHeight(100)
        points_layout.addWidget(self.starts_list)

        points_layout.addWidget(QLabel("End Points:"))
        self.ends_list = QListWidget()
        self.ends_list.setMaximumHeight(100)
        points_layout.addWidget(self.ends_list)

        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # Action Buttons
        self.run_button = QPushButton("Run A* Pathfinding")
        self.run_button.clicked.connect(self.run_astar.emit)
        layout.addWidget(self.run_button)

        self.clear_button = QPushButton("Clear All Paths")
        self.clear_button.clicked.connect(self.clear_paths.emit)
        layout.addWidget(self.clear_button)

        layout.addStretch()

    def update_grid_info(self) -> None:
        """Update grid information display."""
        self.size_label.setText(f"Size: {self.grid.width} × {self.grid.height}")
        self.stats_label.setText(f"Starts: {len(self.grid.starts)} | Ends: {len(self.grid.ends)}")

        # Update start points list
        self.starts_list.clear()
        for i, (row, col) in enumerate(self.grid.starts):
            self.starts_list.addItem(f"{i}: ({row}, {col})")

        # Update end points list
        self.ends_list.clear()
        for i, (row, col) in enumerate(self.grid.ends):
            self.ends_list.addItem(f"{i}: ({row}, {col})")

    def get_astar_config(self) -> AStarConfig:
        """Get current A* configuration.

        Returns:
            AStarConfig instance
        """
        return AStarConfig(
            diagonal_move=self.diagonal_check.isChecked(),
            sdf_weight=self.sdf_weight_spin.value(),
            max_iterations=self.max_iter_spin.value()
        )

    def _on_config_changed(self) -> None:
        """Handle configuration change."""
        self.config_changed.emit(self.get_astar_config())
