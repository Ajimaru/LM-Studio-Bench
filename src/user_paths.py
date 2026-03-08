"""User directory paths following XDG Base Directory spec."""
import os
from pathlib import Path


def get_user_config_dir() -> Path:
    """Get user config directory (XDG_CONFIG_HOME).

    Returns:
        Path to ~/.config/lm-studio-bench/
    """
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        config_dir = Path(xdg_config_home) / "lm-studio-bench"
    else:
        config_dir = Path.home() / ".config" / "lm-studio-bench"

    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_user_data_dir() -> Path:
    """Get user data directory (XDG_DATA_HOME).

    Returns:
        Path to ~/.local/share/lm-studio-bench/
    """
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        data_dir = Path(xdg_data_home) / "lm-studio-bench"
    else:
        data_dir = Path.home() / ".local" / "share" / "lm-studio-bench"

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
