"""Spuštění GUI: ``python -m alphagen_studio``."""

from __future__ import annotations

import sys


def main() -> int:
    from .logging_setup import setup_logging

    setup_logging()
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import APP_NAME
    from .config import APP_ROOT
    from .gui.main_window import MainWindow
    from .gui.theme import apply_theme

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    from PySide6.QtCore import QLocale

    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))  # desetinná tečka v celé aplikaci
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    icon = APP_ROOT / "assets" / "icon.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    apply_theme(app)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_input(sys.argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
