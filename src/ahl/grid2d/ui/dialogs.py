"""Dialog widgets for grid editor."""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QSpinBox, QPushButton, QFileDialog, QMessageBox
)

from ..utils.validators import GridConfig


class NewGridDialog(QDialog):
    """Dialog for creating a new grid."""

    def __init__(self, parent=None):
        """Initialize dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Create New Grid")
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Width
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(50)
        width_layout.addWidget(self.width_spin)
        layout.addLayout(width_layout)

        # Height
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 1000)
        self.height_spin.setValue(50)
        height_layout.addWidget(self.height_spin)
        layout.addLayout(height_layout)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("Create")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def get_grid_config(self) -> GridConfig:
        """Get grid configuration from dialog.

        Returns:
            GridConfig instance
        """
        return GridConfig(
            width=self.width_spin.value(),
            height=self.height_spin.value(),
            default_cell=0
        )


def show_save_dialog(parent=None) -> str | None:
    """Show save file dialog.

    Args:
        parent: Parent widget

    Returns:
        Selected file path, or None if cancelled
    """
    file_path, _ = QFileDialog.getSaveFileName(
        parent,
        "Save Grid",
        "",
        "Grid Files (*.npz);;All Files (*)"
    )
    return file_path if file_path else None


def show_load_dialog(parent=None) -> str | None:
    """Show load file dialog.

    Args:
        parent: Parent widget

    Returns:
        Selected file path, or None if cancelled
    """
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Load Grid",
        "",
        "Grid Files (*.npz);;All Files (*)"
    )
    return file_path if file_path else None


def show_error(parent, title: str, message: str) -> None:
    """Show error message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message
    """
    QMessageBox.critical(parent, title, message)


def show_info(parent, title: str, message: str) -> None:
    """Show information message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Information message
    """
    QMessageBox.information(parent, title, message)


def show_warning(parent, title: str, message: str) -> None:
    """Show warning message dialog.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Warning message
    """
    QMessageBox.warning(parent, title, message)
