"""User directory paths following XDG Base Directory spec."""
import os
from pathlib import Path


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

    return candidate


def get_user_config_dir() -> Path:
    """Get user config directory (XDG_CONFIG_HOME).

    Returns:
        Path to ~/.config/lm-studio-bench/
    """
    base_dir = _resolve_xdg_home(
        "XDG_CONFIG_HOME",
        Path.home() / ".config",
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
        Path.home() / ".local" / "share",
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
