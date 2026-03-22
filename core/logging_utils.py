"""Shared logging helpers for consistent log formatting."""

from __future__ import annotations

import logging

_ICON_BY_LEVEL = {
    logging.DEBUG: "🐛",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🔥",
}

_INSTALL_STATE = {"installed": False}


def install_level_icons() -> None:
    """Inject a ``level_icon`` attribute into every log record."""
    if _INSTALL_STATE["installed"]:
        return

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.level_icon = _ICON_BY_LEVEL.get(record.levelno, "🔹")
        return record

    logging.setLogRecordFactory(record_factory)
    _INSTALL_STATE["installed"] = True
