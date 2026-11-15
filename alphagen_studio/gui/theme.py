"""Tmavý vzhled ve stylu postprodukčních aplikací."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Vyšší kontrast: tmavší pozadí, světlejší text, výraznější okraje; akcent tyrkysově modrá
BG = "#0f1114"
PANEL = "#171a1f"
PANEL_2 = "#20242b"
BORDER = "#3a414d"
TEXT = "#f1f3f6"
TEXT_DIM = "#a9b1bd"
ACCENT = "#2bb8e8"
ACCENT_RGB = (43, 184, 232)
ACCENT_TEXT = "#04131a"  # text na akcentu
ACCENT_2 = "#8f7cff"
OK = "#4cc38a"
WARN = "#f2c14e"
ERR = "#ff6b5e"

QSS = f"""
* {{ font-family: 'Segoe UI'; font-size: 9pt; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QWidget#Panel, QScrollArea#Panel > QWidget > QWidget {{ background: {PANEL}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QGroupBox {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; margin-top: 16px;
            padding: 14px 12px 10px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {ACCENT};
                    font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }}
QToolButton#Section {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 4px; padding: 8px 12px;
    color: {ACCENT}; font-weight: 600; text-align: left; }}
QToolButton#Section:checked {{ border-bottom-left-radius: 0; border-bottom-right-radius: 0; }}
QToolButton#Section:hover, QToolButton#SubSection:hover {{ border-color: {ACCENT}; }}
QFrame#SectionBody {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-top: none;
    border-bottom-left-radius: 4px; border-bottom-right-radius: 4px; }}
QToolButton#SubSection {{ background: transparent; border: 1px solid transparent; padding: 4px 2px; color: {TEXT_DIM};
    font-weight: 600; text-align: left; }}
QFrame#SubSectionBody {{ background: transparent; border-left: 2px solid {BORDER}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 3px; top: -1px; }}
QTabBar::tab {{ background: {PANEL_2}; border: 1px solid {BORDER}; padding: 5px 14px; margin-right: 2px;
    border-top-left-radius: 3px; border-top-right-radius: 3px; color: {TEXT_DIM}; }}
QTabBar::tab:selected {{ background: {BG}; color: {ACCENT}; border-bottom-color: {BG}; font-weight: 600; }}
QLabel#PanelTitle {{ color: {TEXT_DIM}; font-weight: 700; letter-spacing: 2px; padding: 8px 6px; }}
QLabel#Dim {{ color: {TEXT_DIM}; }}
QLabel#Warn {{ color: {WARN}; }}
QLabel#Err {{ color: {ERR}; }}
QLabel#Drop {{ border: 2px dashed {BORDER}; border-radius: 6px; color: {TEXT_DIM}; padding: 18px;
               background: {BG}; }}
QLabel#Drop[hover="true"] {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: #6b7380; border-color: #2a2f37; }}
QPushButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; border-color: {ACCENT}; }}
QPushButton#Primary {{ background: {ACCENT}; color: {ACCENT_TEXT}; font-weight: 700; border: none; padding: 7px 18px; }}
QPushButton#Primary:disabled {{ background: #1d4b5c; color: #8aa4ae; }}
QPushButton#Danger {{ background: #6b2a24; border-color: {ERR}; }}
QToolButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px; padding: 3px 6px; }}
QToolButton:checked {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
QToolButton:hover {{ border-color: {ACCENT}; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 3px;
    padding: 4px 8px; selection-background-color: {ACCENT}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER}; selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {BORDER}; background: {BG}; border-radius: 2px; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 12px; margin: -5px 0; background: {TEXT}; border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 3px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 2px; }}
QProgressBar#Vram::chunk {{ background: {ACCENT_2}; }}
QPlainTextEdit {{ background: #0b0c0e; border: 1px solid {BORDER}; font-family: Consolas; font-size: 8.5pt; }}
QTableWidget {{ background: {BG}; gridline-color: {BORDER}; border: 1px solid {BORDER}; }}
QTableWidget::item {{ padding: 4px 10px; }}
QPlainTextEdit {{ padding: 6px 8px; }}
QHeaderView::section {{ background: {PANEL_2}; border: none; border-right: 1px solid {BORDER}; padding: 5px 10px; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {ACCENT}; color: {ACCENT_TEXT}; }}
QMenu {{ background: {PANEL_2}; border: 1px solid {BORDER}; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {ACCENT}; padding: 4px; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(BG))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(PANEL_2))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(ACCENT_TEXT))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
