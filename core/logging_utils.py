"""Shared logging helpers for consistent log formatting."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

from core.platform_info import IS_WINDOWS

LOGGER = logging.getLogger(__name__)

_ICON_BY_LEVEL = {
    logging.DEBUG: "🐛",
    logging.INFO: "ℹ️",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🔥",
}

_INSTALL_STATE = {"installed": False}


def update_latest_log(log_path: Path, latest_path: Path) -> None:
    """Update a latest-log reference without requiring Windows symlinks.

    Args:
        log_path: Timestamped log file.
        latest_path: Conventional ``*_latest.log`` name in the same directory.

    Raises:
        ValueError: If the reference and log are in different directories.
    """
    if log_path.parent != latest_path.parent:
        raise ValueError("The latest-log reference must share the log directory")

    temporary: Path | None = None
    try:
        if IS_WINDOWS:
            pointer = latest_path.with_suffix(".txt")
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=pointer.parent,
                prefix=f".{pointer.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(f"{log_path.name}\n")
            temporary.replace(pointer)
        else:
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(log_path.name)
    except OSError as error:
        LOGGER.warning(
            "Could not update latest-log reference %s: %s",
            latest_path.name,
            error,
        )
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                LOGGER.warning("Could not remove temporary log reference: %s", error)


def resolve_latest_log(latest_path: Path) -> Path | None:
    """Resolve a platform-specific reference to an existing log file.

    Args:
        latest_path: Conventional ``*_latest.log`` path.

    Returns:
        Referenced log, or None if unavailable. Invalid references are logged.
    """
    if IS_WINDOWS:
        pointer = latest_path.with_suffix(".txt")
        try:
            if pointer.is_file():
                name = pointer.read_text(encoding="utf-8").strip()
                if (
                    not name.endswith(".log")
                    or any(char in name for char in "/\\:\x00\r\n")
                ):
                    raise ValueError("Expected a log filename, not a path")
                target = pointer.parent / name
                if target.is_file():
                    return target
                LOGGER.warning("Latest-log reference points to a missing log")
        except (OSError, ValueError) as error:
            LOGGER.warning("Could not read latest-log reference: %s", error)

    try:
        if latest_path.is_file():
            return latest_path.resolve()
    except OSError as error:
        LOGGER.warning("Could not resolve latest-log reference: %s", error)
    return None


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
