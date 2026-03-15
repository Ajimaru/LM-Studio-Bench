"""Tests for src/user_paths.py."""
from pathlib import Path


class TestGetUserConfigDir:
    """Tests for get_user_config_dir()."""

    def test_returns_xdg_config_home_subdir(self, tmp_path: Path, monkeypatch):
        """XDG_CONFIG_HOME is honoured when set."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / "lm-studio-bench"
        assert result.exists()

    def test_falls_back_to_home_config(self, tmp_path: Path, monkeypatch):
        """Falls back to ~/.config/lm-studio-bench when XDG_CONFIG_HOME unset."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / ".config" / "lm-studio-bench"
        assert result.exists()

    def test_creates_directory(self, tmp_path: Path, monkeypatch):
        """Directory is created when it does not exist yet."""
        target = tmp_path / "xdg_conf"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(target))
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result.exists()
        assert result.is_dir()

    def test_rejects_relative_traversal_xdg_config(
            self, tmp_path: Path, monkeypatch):
        """Relative traversal values fall back to default config path."""
        monkeypatch.setenv("XDG_CONFIG_HOME", "../etc")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_config_dir()
        assert result == tmp_path / ".config" / "lm-studio-bench"


class TestGetUserDataDir:
    """Tests for get_user_data_dir()."""

    def test_returns_xdg_data_home_subdir(self, tmp_path: Path, monkeypatch):
        """XDG_DATA_HOME is honoured when set."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / "lm-studio-bench"
        assert result.exists()

    def test_falls_back_to_home_local_share(self, tmp_path: Path, monkeypatch):
        """Falls back to ~/.local/share/lm-studio-bench when unset."""
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / ".local" / "share" / "lm-studio-bench"
        assert result.exists()

    def test_creates_directory(self, tmp_path: Path, monkeypatch):
        """Directory is created when it does not exist yet."""
        target = tmp_path / "xdg_data"
        monkeypatch.setenv("XDG_DATA_HOME", str(target))
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result.exists()
        assert result.is_dir()

    def test_rejects_relative_traversal_xdg_data(
        self, tmp_path: Path, monkeypatch
    ):
        """Relative traversal values fall back to default data path."""
        monkeypatch.setenv("XDG_DATA_HOME", "../../var")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        import importlib

        import user_paths as up
        importlib.reload(up)
        result = up.get_user_data_dir()
        assert result == tmp_path / ".local" / "share" / "lm-studio-bench"


class TestModuleLevelConstants:
    """Tests for module-level path constants."""

    def test_user_config_file_is_json(self):
        """USER_CONFIG_FILE has .json suffix."""
        import user_paths as up
        assert up.USER_CONFIG_FILE.suffix == ".json"
        assert up.USER_CONFIG_FILE.name == "defaults.json"

    def test_user_presets_dir_exists(self):
        """USER_PRESETS_DIR is created at import time."""
        import user_paths as up
        assert up.USER_PRESETS_DIR.exists()
        assert up.USER_PRESETS_DIR.is_dir()

    def test_user_results_dir_exists(self):
        """USER_RESULTS_DIR is created at import time."""
        import user_paths as up
        assert up.USER_RESULTS_DIR.exists()
        assert up.USER_RESULTS_DIR.is_dir()

    def test_user_logs_dir_exists(self):
        """USER_LOGS_DIR is created at import time."""
        import user_paths as up
        assert up.USER_LOGS_DIR.exists()
        assert up.USER_LOGS_DIR.is_dir()

    def test_config_dir_is_parent_of_config_file(self):
        """USER_CONFIG_FILE is inside USER_CONFIG_DIR."""
        import user_paths as up
        assert up.USER_CONFIG_FILE.parent == up.USER_CONFIG_DIR

    def test_presets_dir_is_inside_config_dir(self):
        """USER_PRESETS_DIR is inside USER_CONFIG_DIR."""
        import user_paths as up
        assert up.USER_PRESETS_DIR.parent == up.USER_CONFIG_DIR

    def test_results_dir_is_inside_data_dir(self):
        """USER_RESULTS_DIR is inside USER_DATA_DIR."""
        import user_paths as up
        assert up.USER_RESULTS_DIR.parent == up.USER_DATA_DIR

    def test_logs_dir_is_inside_data_dir(self):
        """USER_LOGS_DIR is inside USER_DATA_DIR."""
        import user_paths as up
        assert up.USER_LOGS_DIR.parent == up.USER_DATA_DIR
