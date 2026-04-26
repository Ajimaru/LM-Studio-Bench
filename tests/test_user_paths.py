"""Tests for core/paths.py."""
from pathlib import Path
from typing import cast


class TestGetUserConfigDir:
    """Tests for get_user_config_dir()."""

    def test_returns_xdg_config_home_subdir(self, tmp_path: Path, monkeypatch):
        """XDG_CONFIG_HOME is honoured when set."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / "lm-studio-bench"
        assert result.exists()

    def test_falls_back_to_home_config(self, tmp_path: Path, monkeypatch):
        """Falls back to ~/.config/lm-studio-bench when XDG_CONFIG_HOME unset."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / ".config" / "lm-studio-bench"
        assert result.exists()

    def test_creates_directory(self, tmp_path: Path, monkeypatch):
        """Directory is created when it does not exist yet."""
        target = tmp_path / "xdg_conf"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(target))
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result.exists()
        assert result.is_dir()

    def test_rejects_relative_traversal_xdg_config(
            self, tmp_path: Path, monkeypatch):
        """Relative traversal values fall back to default config path."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "../etc")
        monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / ".config" / "lm-studio-bench"


class TestGetUserDataDir:
    """Tests for get_user_data_dir()."""

    def test_returns_xdg_data_home_subdir(self, tmp_path: Path, monkeypatch):
        """XDG_DATA_HOME is honoured when set."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / "lm-studio-bench"
        assert result.exists()

    def test_falls_back_to_home_local_share(self, tmp_path: Path, monkeypatch):
        """Falls back to ~/.local/share/lm-studio-bench when unset."""
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / ".local" / "share" / "lm-studio-bench"
        assert result.exists()

    def test_creates_directory(self, tmp_path: Path, monkeypatch):
        """Directory is created when it does not exist yet."""
        target = tmp_path / "xdg_data"
        monkeypatch.setenv("XDG_DATA_HOME", str(target))
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result.exists()
        assert result.is_dir()

    def test_rejects_relative_traversal_xdg_data(
        self, tmp_path: Path, monkeypatch
    ):
        """Relative traversal values fall back to default data path."""
        monkeypatch.setenv("XDG_DATA_HOME", "../../var")
        monkeypatch.delenv("SNAP_REAL_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import core.paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / ".local" / "share" / "lm-studio-bench"


class TestModuleLevelConstants:
    """Tests for module-level path constants."""

    def test_user_config_file_is_json(self):
        """USER_CONFIG_FILE has .json suffix."""
        import core.paths as up
        assert up.USER_CONFIG_FILE.suffix == ".json"
        assert up.USER_CONFIG_FILE.name == "defaults.json"

    def test_user_presets_dir_exists(self):
        """USER_PRESETS_DIR is created at import time."""
        import core.paths as up
        assert up.USER_PRESETS_DIR.exists()
        assert up.USER_PRESETS_DIR.is_dir()

    def test_user_results_dir_exists(self):
        """USER_RESULTS_DIR is created at import time."""
        import core.paths as up
        assert up.USER_RESULTS_DIR.exists()
        assert up.USER_RESULTS_DIR.is_dir()

    def test_user_logs_dir_exists(self):
        """USER_LOGS_DIR is created at import time."""
        import core.paths as up
        assert up.USER_LOGS_DIR.exists()
        assert up.USER_LOGS_DIR.is_dir()

    def test_config_dir_is_parent_of_config_file(self):
        """USER_CONFIG_FILE is inside USER_CONFIG_DIR."""
        import core.paths as up
        assert up.USER_CONFIG_FILE.parent == up.USER_CONFIG_DIR

    def test_presets_dir_is_inside_config_dir(self):
        """USER_PRESETS_DIR is inside USER_CONFIG_DIR."""
        import core.paths as up
        assert up.USER_PRESETS_DIR.parent == up.USER_CONFIG_DIR

    def test_results_dir_is_inside_data_dir(self):
        """USER_RESULTS_DIR is inside USER_DATA_DIR."""
        import core.paths as up
        assert up.USER_RESULTS_DIR.parent == up.USER_DATA_DIR

    def test_logs_dir_is_inside_data_dir(self):
        """USER_LOGS_DIR is inside USER_DATA_DIR."""
        import core.paths as up
        assert up.USER_LOGS_DIR.parent == up.USER_DATA_DIR


class TestPathFormattingAndSnapHome:
    """Tests for helper branches in core.paths."""

    def test_effective_home_uses_snap_real_home(self, tmp_path, monkeypatch):
        """SNAP_REAL_HOME is preferred when valid and absolute."""
        monkeypatch.setenv("SNAP_REAL_HOME", str(tmp_path))
        import importlib

        import core.paths as up
        importlib.reload(up)
        assert up._effective_home() == tmp_path

    def test_effective_home_falls_back_on_invalid_snap_path(self, monkeypatch):
        """Invalid SNAP_REAL_HOME falls back to Path.home()."""
        monkeypatch.setenv("SNAP_REAL_HOME", "../bad")
        import importlib

        import core.paths as up
        importlib.reload(up)
        assert up._effective_home() == Path.home()

    def test_format_path_for_logs_exact_home(self, tmp_path, monkeypatch):
        """Exact home path is redacted to '~'."""
        monkeypatch.setenv("SNAP_REAL_HOME", str(tmp_path))
        import importlib

        import core.paths as up
        importlib.reload(up)
        assert up.format_path_for_logs(str(tmp_path)) == "~"

    def test_format_path_for_logs_non_fspath_input(self, tmp_path, monkeypatch):
        """Non-fspath values take the TypeError fallback path."""
        monkeypatch.setenv("SNAP_REAL_HOME", str(tmp_path))
        import importlib

        import core.paths as up
        importlib.reload(up)

        class _NoFspath:
            def __str__(self):
                return str(tmp_path / "logs" / "x.log")

        value = _NoFspath()
        typed_value = cast(str | Path, value)
        assert up.format_path_for_logs(typed_value) == "~/logs/x.log"
