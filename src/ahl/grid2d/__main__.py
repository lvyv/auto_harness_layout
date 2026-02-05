"""Entry point for running grid editor as a module."""

import sys
from PyQt6.QtWidgets import QApplication
from .ui.main_window import GridEditorWindow


def main():
    """Launch the grid editor application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Grid2D Editor")
    app.setOrganizationName("AHL")

    window = GridEditorWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
