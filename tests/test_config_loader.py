"""Tests for src/config_loader.py."""
import json
from pathlib import Path
from unittest.mock import patch

from config_loader import (
    BASE_DEFAULT_CONFIG,
    _deep_merge,
    _normalize_ports,
    load_default_config,
)


class TestDeepMerge:
    """Tests for _deep_merge()."""

    def test_simple_key_override(self):
        """Top-level scalar is overridden by non-None override value."""
        result = _deep_merge({"a": 1, "b": 2}, {"a": 10})
        assert result == {"a": 10, "b": 2}

    def test_nested_dict_merged_recursively(self):
        """Nested dicts are merged, not replaced."""
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 99, "c": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}}

    def test_none_values_are_ignored(self):
        """None override values do not overwrite existing base values."""
        result = _deep_merge({"a": 5}, {"a": None})
        assert result["a"] == 5

    def test_new_keys_added(self):
        """Keys absent in base are added from override."""
        result = _deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        """Original base dict is not modified."""
        base = {"a": 1, "nested": {"x": 1}}
        override = {"a": 2, "nested": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": 1, "nested": {"x": 1}}

    def test_empty_override(self):
        """Empty override returns copy of base."""
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {})
        assert result == base
        assert result is not base

    def test_empty_base(self):
        """Empty base returns copy of override (non-None values)."""
        result = _deep_merge({}, {"a": 1, "b": None})
        assert result == {"a": 1}

    def test_override_replaces_non_dict_with_scalar(self):
        """Non-dict value in base is replaced by override scalar."""
        result = _deep_merge({"a": [1, 2]}, {"a": 99})
        assert result["a"] == 99

    def test_deeply_nested_merge(self):
        """Multiple nesting levels are all merged recursively."""
        base = {"l1": {"l2": {"l3": 1}}}
        override = {"l1": {"l2": {"l3": 2, "l3b": 3}}}
        result = _deep_merge(base, override)
        assert result["l1"]["l2"] == {"l3": 2, "l3b": 3}


class TestNormalizePorts:
    """Tests for _normalize_ports()."""

    def test_valid_list_of_ints(self):
        """Valid integer list is returned as-is."""
        result = _normalize_ports([1234, 1235])
        assert result == [1234, 1235]

    def test_string_numbers_are_converted(self):
        """String port numbers are converted to int."""
        result = _normalize_ports(["1234", "1235"])
        assert result == [1234, 1235]

    def test_duplicates_removed(self):
        """Duplicate ports are removed, first occurrence kept."""
        result = _normalize_ports([1234, 1234, 1235])
        assert result == [1234, 1235]

    def test_invalid_strings_skipped(self):
        """Non-numeric strings are skipped silently."""
        result = _normalize_ports([1234, "not_a_port", 1235])
        assert result == [1234, 1235]

    def test_none_values_skipped(self):
        """None entries in list are silently skipped."""
        result = _normalize_ports([1234, None, 1235])
        assert result == [1234, 1235]

    def test_empty_list_returns_defaults(self):
        """Empty list triggers fallback to BASE_DEFAULT_CONFIG ports."""
        result = _normalize_ports([])
        assert result == list(BASE_DEFAULT_CONFIG["lmstudio"]["ports"])

    def test_non_list_returns_defaults(self):
        """Non-list input triggers fallback to BASE_DEFAULT_CONFIG ports."""
        result = _normalize_ports("1234")
        assert result == list(BASE_DEFAULT_CONFIG["lmstudio"]["ports"])

    def test_none_input_returns_defaults(self):
        """None input triggers fallback to BASE_DEFAULT_CONFIG ports."""
        result = _normalize_ports(None)
        assert result == list(BASE_DEFAULT_CONFIG["lmstudio"]["ports"])

    def test_all_invalid_returns_defaults(self):
        """All-invalid entries trigger fallback to defaults."""
        result = _normalize_ports(["bad", "port"])
        assert result == list(BASE_DEFAULT_CONFIG["lmstudio"]["ports"])


class TestLoadDefaultConfig:
    """Tests for load_default_config()."""

    def test_returns_base_keys_when_no_files(self, tmp_path: Path):
        """Returns BASE_DEFAULT_CONFIG keys when no config files exist."""
        non_existent = tmp_path / "no_config.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", non_existent), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        assert "prompt" in result
        assert "lmstudio" in result
        assert "inference" in result
        assert "load" in result

    def test_project_config_overrides_base(self, tmp_path: Path):
        """Project config file values override base defaults."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(
            json.dumps({"prompt": "custom_prompt"}), encoding="utf-8"
        )
        non_existent = tmp_path / "no_user.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        assert result["prompt"] == "custom_prompt"

    def test_user_config_overrides_project(self, tmp_path: Path):
        """User config file overrides project config when both exist."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(
            json.dumps({"prompt": "project_prompt"}), encoding="utf-8"
        )
        user_cfg = tmp_path / "user.json"
        user_cfg.write_text(
            json.dumps({"prompt": "user_prompt"}), encoding="utf-8"
        )
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", user_cfg):
            result = load_default_config()
        assert result["prompt"] == "user_prompt"

    def test_invalid_project_json_falls_back_gracefully(self, tmp_path: Path):
        """Corrupted project JSON is skipped with a warning."""
        bad_cfg = tmp_path / "bad.json"
        bad_cfg.write_text("{ not valid json }", encoding="utf-8")
        non_existent = tmp_path / "no_user.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", bad_cfg), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        assert "prompt" in result

    def test_invalid_user_json_falls_back_gracefully(self, tmp_path: Path):
        """Corrupted user JSON is skipped with a warning."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(json.dumps({"prompt": "p"}), encoding="utf-8")
        bad_user = tmp_path / "bad_user.json"
        bad_user.write_text("not json", encoding="utf-8")
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", bad_user):
            result = load_default_config()
        assert "prompt" in result

    def test_ports_normalized_in_result(self, tmp_path: Path):
        """Ports in returned config are always a list of ints."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(
            json.dumps({"lmstudio": {"ports": ["1234", "1235"]}}),
            encoding="utf-8",
        )
        non_existent = tmp_path / "no_user.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        ports = result["lmstudio"]["ports"]
        assert isinstance(ports, list)
        assert all(isinstance(p, int) for p in ports)

    def test_lmstudio_host_defaults_when_missing(self, tmp_path: Path):
        """lmstudio.host gets default value when absent in config."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(
            json.dumps({"lmstudio": {}}), encoding="utf-8"
        )
        non_existent = tmp_path / "no_user.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        assert result["lmstudio"]["host"] == BASE_DEFAULT_CONFIG["lmstudio"]["host"]

    def test_nested_lmstudio_config_merged(self, tmp_path: Path):
        """Nested lmstudio config is merged, not replaced."""
        project_cfg = tmp_path / "proj.json"
        project_cfg.write_text(
            json.dumps({"lmstudio": {"host": "remotehost"}}),
            encoding="utf-8",
        )
        non_existent = tmp_path / "no_user.json"
        with patch("config_loader.PROJECT_CONFIG_PATH", project_cfg), \
             patch("config_loader.USER_CONFIG_FILE", non_existent):
            result = load_default_config()
        assert result["lmstudio"]["host"] == "remotehost"
        assert "ports" in result["lmstudio"]
