"""Main window for grid editor application."""

from pathlib import Path
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStatusBar
from PyQt6.QtCore import Qt

from ..core.grid import Grid
from ..core.astar import astar_search, batch_astar
from ..io.npz_handler import save_grid, load_grid
from .grid_widget import GridWidget
from .control_panel import ControlPanel
from .toolbar import EditorToolbar
from .dialogs import (
    NewGridDialog, show_save_dialog, show_load_dialog,
    show_error, show_info, show_warning
)


class GridEditorWindow(QMainWindow):
    """Main window for 2D grid editor and A* path planning.

    Features:
        - Interactive grid editing
        - Multiple start/end points
        - A* pathfinding with SDF penalty
        - Save/load to NPZ format
    """

    def __init__(self):
        """Initialize main window."""
        super().__init__()

        # Create default grid
        self.grid = Grid(50, 50)
        self.current_file = None

        self._setup_ui()
        self._connect_signals()

        # Initial view setup
        self.grid_widget.reset_view()
        self.control_panel.update_grid_info()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        self.setWindowTitle("2D Grid Editor - A* Path Planning")
        self.setGeometry(100, 100, 1200, 800)

        # Toolbar
        self.toolbar = EditorToolbar(self)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self.toolbar)

        # Central widget with grid and control panel
        central = QWidget()
        layout = QHBoxLayout(central)

        # Grid widget (left, takes most space)
        self.grid_widget = GridWidget(self.grid)
        layout.addWidget(self.grid_widget, stretch=3)

        # Control panel (right)
        self.control_panel = ControlPanel(self.grid)
        self.control_panel.setMaximumWidth(300)
        layout.addWidget(self.control_panel, stretch=1)

        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        """Connect signals and slots."""
        # Toolbar signals
        self.toolbar.new_grid.connect(self._on_new_grid)
        self.toolbar.save_grid.connect(self._on_save_grid)
        self.toolbar.load_grid.connect(self._on_load_grid)
        self.toolbar.edit_mode_changed.connect(self.grid_widget.set_edit_mode)
        self.toolbar.reset_view.connect(self.grid_widget.reset_view)

        # Grid widget signals
        self.grid_widget.grid_changed.connect(self._on_grid_changed)

        # Control panel signals
        self.control_panel.run_astar.connect(self._on_run_astar)
        self.control_panel.clear_paths.connect(self._on_clear_paths)

    def _on_new_grid(self) -> None:
        """Handle new grid creation."""
        dialog = NewGridDialog(self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            config = dialog.get_grid_config()
            self.grid = Grid(config.width, config.height, config.default_cell)
            self.grid_widget.set_grid(self.grid)
            self.control_panel.grid = self.grid
            self.control_panel.update_grid_info()
            self.grid_widget.reset_view()
            self.current_file = None
            self.status_bar.showMessage(f"Created new grid: {config.width} × {config.height}")

    def _on_save_grid(self) -> None:
        """Handle grid save."""
        file_path = show_save_dialog(self)
        if file_path:
            try:
                # Ensure .npz extension
                if not file_path.endswith('.npz'):
                    file_path += '.npz'

                save_grid(self.grid, file_path)
                self.current_file = file_path
                self.status_bar.showMessage(f"Saved: {Path(file_path).name}")
                show_info(self, "Success", f"Grid saved to:\n{file_path}")

            except Exception as e:
                show_error(self, "Save Error", f"Failed to save grid:\n{str(e)}")
                self.status_bar.showMessage("Save failed")

    def _on_load_grid(self) -> None:
        """Handle grid load."""
        file_path = show_load_dialog(self)
        if file_path:
            try:
                self.grid = load_grid(file_path)
                self.grid_widget.set_grid(self.grid)
                self.control_panel.grid = self.grid
                self.control_panel.update_grid_info()
                self.grid_widget.reset_view()
                self.current_file = file_path
                self.status_bar.showMessage(f"Loaded: {Path(file_path).name}")

            except Exception as e:
                show_error(self, "Load Error", f"Failed to load grid:\n{str(e)}")
                self.status_bar.showMessage("Load failed")

    def _on_grid_changed(self) -> None:
        """Handle grid change."""
        self.control_panel.update_grid_info()

    def _on_run_astar(self) -> None:
        """Handle A* pathfinding execution."""
        # Validate we have start and end points
        if not self.grid.starts:
            show_warning(self, "No Start Points", "Please add at least one start point (green).")
            return

        if not self.grid.ends:
            show_warning(self, "No End Points", "Please add at least one end point (red).")
            return

        # Clear existing paths
        self.grid.clear_paths()

        # Get configuration
        config = self.control_panel.get_astar_config()

        # Run A* for all start-end pairs
        self.status_bar.showMessage("Running A* pathfinding...")

        try:
            paths_found = 0
            paths_failed = 0

            for start_idx, start in enumerate(self.grid.starts):
                for end_idx, end in enumerate(self.grid.ends):
                    # Run A*
                    path = astar_search(self.grid, start, end, config)

                    if path is not None:
                        self.grid.set_path(start_idx, end_idx, path)
                        paths_found += 1
                    else:
                        paths_failed += 1

            # Update display
            self.grid_widget.update()

            # Show results
            total = len(self.grid.starts) * len(self.grid.ends)
            message = f"Pathfinding complete:\n{paths_found}/{total} paths found"
            if paths_failed > 0:
                message += f"\n{paths_failed} paths failed (no solution)"

            show_info(self, "A* Complete", message)
            self.status_bar.showMessage(f"Found {paths_found} paths")

        except Exception as e:
            show_error(self, "A* Error", f"Pathfinding failed:\n{str(e)}")
            self.status_bar.showMessage("A* failed")

    def _on_clear_paths(self) -> None:
        """Handle path clearing."""
        self.grid.clear_paths()
        self.grid_widget.update()
        self.status_bar.showMessage("Paths cleared")
