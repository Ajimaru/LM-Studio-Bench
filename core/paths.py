"""User directory paths following XDG Base Directory spec."""

import os
from pathlib import Path
from typing import Union


def _effective_home() -> Path:
    """Return the effective user home directory.

    In Snap environments (e.g. VS Code Snap), ``Path.home()`` often points
    to a snap-scoped home like ``~/snap/code/<rev>``. If ``SNAP_REAL_HOME``
    is available, prefer it so cache/log paths are stable across runs.
    """
    snap_real_home = os.environ.get("SNAP_REAL_HOME")
    if snap_real_home:
        candidate = Path(snap_real_home).expanduser()
        if candidate.is_absolute() and ".." not in candidate.parts:
            return candidate
    return Path.home()


def _resolve_xdg_home(env_var: str, fallback: Path) -> Path:
    """Resolve and validate an XDG base directory.

    The value must be absolute and must not contain explicit parent
    traversal segments. Invalid values fall back to the provided path.
    """
    raw_value = os.environ.get(env_var)
    if not raw_value:
        return fallback

    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        return fallback

    effective_home = _effective_home()
    if str(candidate).startswith(str(effective_home / "snap")):
        return fallback

    return candidate


def format_path_for_logs(path_value: Union[str, Path]) -> str:
    """Format a path for logs without exposing username.

    Replaces the effective home prefix with ``~``.
    """
    try:
        raw = os.fspath(path_value)
    except TypeError:
        raw = str(path_value)

    home_str = str(_effective_home())
    if raw == home_str:
        return "~"
    if raw.startswith(f"{home_str}/"):
        return raw.replace(home_str, "~", 1)
    return raw


def get_user_config_dir() -> Path:
    """Get user config directory (XDG_CONFIG_HOME).

    Returns:
        Path to ~/.config/lm-studio-bench/
    """
    base_dir = _resolve_xdg_home(
        "XDG_CONFIG_HOME",
        _effective_home() / ".config",
    )
    config_dir = base_dir / "lm-studio-bench"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_user_data_dir() -> Path:
    """Get user data directory (XDG_DATA_HOME).

    Returns:
        Path to ~/.local/share/lm-studio-bench/
    """
    base_dir = _resolve_xdg_home(
        "XDG_DATA_HOME",
        _effective_home() / ".local" / "share",
    )
    data_dir = base_dir / "lm-studio-bench"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


USER_CONFIG_DIR = get_user_config_dir()
USER_CONFIG_FILE = USER_CONFIG_DIR / "defaults.json"
USER_PRESETS_DIR = USER_CONFIG_DIR / "presets"
USER_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

USER_DATA_DIR = get_user_data_dir()
USER_RESULTS_DIR = USER_DATA_DIR / "results"
USER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USER_LOGS_DIR = USER_DATA_DIR / "logs"
USER_LOGS_DIR.mkdir(parents=True, exist_ok=True)
