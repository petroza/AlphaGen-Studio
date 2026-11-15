"""Logování do souboru + do GUI (přes handler s callbackem)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR

LOGGER_NAME = "alphagen"
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(LOGGER_NAME)
    if root.handlers:
        return root
    root.setLevel(level)
    fh = RotatingFileHandler(LOG_DIR / "alphagen.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(sh)
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


class CallbackHandler(logging.Handler):
    """Předává záznamy do GUI (callback musí být thread-safe, např. Qt signál)."""

    def __init__(self, callback: Callable[[str, str], None]) -> None:
        super().__init__()
        self._cb = callback
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._cb(record.levelname, self.format(record))
        except Exception:  # noqa: BLE001 - log nesmí shodit aplikaci
            pass


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
