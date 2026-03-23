"""Tests for core/presets.py."""
import json
from pathlib import Path

import pytest

from core.presets import PresetManager


class TestPresetManagerInit:
    """Tests for PresetManager.__init__()."""

    def test_default_presets_dir_used(self):
        """Default presets dir is used when none is passed."""
        pm = PresetManager()
        assert pm._presets_dir.exists()

    def test_custom_presets_dir(self, tmp_presets_dir: Path):
        """Custom presets directory is stored and created."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        assert pm._presets_dir == tmp_presets_dir

    def test_creates_missing_presets_dir(self, tmp_path: Path):
        """Non-existent presets dir is created on init."""
        new_dir = tmp_path / "my_presets"
        PresetManager(presets_dir=new_dir)
        assert new_dir.exists()


class TestListPresets:
    """Tests for PresetManager.list_presets()."""

    def test_always_contains_default(self, tmp_presets_dir: Path):
        """'default_classic' and 'default_compatibility_test' are always listed."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        presets = pm.list_presets()
        assert "default_classic" in presets
        assert "default_compatibility_test" in presets

    def test_predefined_presets_present(self, tmp_presets_dir: Path):
        """All predefined readonly presets are listed."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        presets = pm.list_presets()
        for name in PresetManager.PREDEFINED_PRESETS:
            assert name in presets

    def test_user_presets_included(self, tmp_presets_dir: Path):
        """User-created preset JSON files appear in the list."""
        (tmp_presets_dir / "mypreset.json").write_text(
            json.dumps({"runs": 2}), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        assert "mypreset" in pm.list_presets()

    def test_reserved_names_in_user_dir_ignored(self, tmp_presets_dir: Path):
        """User preset files with reserved names are silently skipped."""
        (tmp_presets_dir / "default.json").write_text(
            json.dumps({}), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        presets = pm.list_presets()
        assert presets.count("default") == 1


class TestListPresetsDetailed:
    """Tests for PresetManager.list_presets_detailed()."""

    def test_returns_tuples_with_readonly_flag(self, tmp_presets_dir: Path):
        """Returns list of (name, is_readonly) tuples."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        detailed = pm.list_presets_detailed()
        assert all(isinstance(item, tuple) and len(item) == 2 for item in detailed)

    def test_default_is_readonly(self, tmp_presets_dir: Path):
        """'default_classic' preset is marked as readonly."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        for name, readonly in pm.list_presets_detailed():
            if name == "default_classic":
                assert readonly is True
                break

    def test_user_preset_not_readonly(self, tmp_presets_dir: Path):
        """User-created preset is not marked as readonly."""
        (tmp_presets_dir / "custom.json").write_text(
            json.dumps({"runs": 1}), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        for name, readonly in pm.list_presets_detailed():
            if name == "custom":
                assert readonly is False
                break


class TestLoadPreset:
    """Tests for PresetManager.load_preset()."""

    def test_load_default_preset(self, tmp_presets_dir: Path):
        """'default' returns a dict with expected keys."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.load_preset("default")
        assert isinstance(preset, dict)
        assert "runs" in preset
        assert "context" in preset

    def test_load_predefined_preset(self, tmp_presets_dir: Path):
        """Predefined presets (e.g. 'quick_test') are merged with defaults."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.load_preset("quick_test")
        assert isinstance(preset, dict)
        assert preset.get("runs") == 1

    def test_load_user_preset_flat(self, tmp_presets_dir: Path):
        """User preset stored as flat JSON dict is loaded and merged."""
        data = {"runs": 4, "context": 4096}
        (tmp_presets_dir / "mytest.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.load_preset("mytest")
        assert preset["runs"] == 4
        assert preset["context"] == 4096

    def test_load_user_preset_with_params_key(self, tmp_presets_dir: Path):
        """User preset stored under 'params' key is unwrapped correctly."""
        payload = {"params": {"runs": 7, "context": 512}}
        (tmp_presets_dir / "wrapped.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.load_preset("wrapped")
        assert preset["runs"] == 7

    def test_load_nonexistent_raises_file_not_found(self, tmp_presets_dir: Path):
        """FileNotFoundError raised for missing user preset."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(FileNotFoundError):
            pm.load_preset("nonexistent_preset_xyz")

    def test_load_invalid_name_raises_value_error(self, tmp_presets_dir: Path):
        """ValueError raised for preset name with invalid characters."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.load_preset("invalid name with spaces")

    def test_load_preset_invalid_format(self, tmp_presets_dir: Path):
        """ValueError raised when preset file contains a non-dict value."""
        (tmp_presets_dir / "badformat.json").write_text(
            json.dumps([1, 2, 3]), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.load_preset("badformat")

    def test_load_preset_invalid_params_value(self, tmp_presets_dir: Path):
        """ValueError raised when 'params' value is not a dict."""
        (tmp_presets_dir / "badparams.json").write_text(
            json.dumps({"params": "string_not_dict"}), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.load_preset("badparams")


class TestGetDefaultPreset:
    """Tests for PresetManager.get_default_preset()."""

    def test_returns_dict_with_expected_keys(self, tmp_presets_dir: Path):
        """Default preset dict contains all expected benchmark keys."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.get_default_preset()
        assert "runs" in preset
        assert "context" in preset
        assert "prompt" in preset
        assert "temperature" in preset
        assert "n_gpu_layers" in preset

    def test_runs_is_int(self, tmp_presets_dir: Path):
        """'runs' field is always an integer."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.get_default_preset()
        assert isinstance(preset["runs"], int)

    def test_context_is_int(self, tmp_presets_dir: Path):
        """'context' field is always an integer."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        preset = pm.get_default_preset()
        assert isinstance(preset["context"], int)


class TestValidatePresetName:
    """Tests for PresetManager.validate_preset_name()."""

    def test_valid_name(self, tmp_presets_dir: Path):
        """Alphanumeric name with underscore is valid."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        valid, reason = pm.validate_preset_name("my_preset_1")
        assert valid is True
        assert reason == ""

    def test_reserved_name_rejected(self, tmp_presets_dir: Path):
        """Reserved preset names are rejected."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        for name in PresetManager.READONLY_PRESETS:
            valid, reason = pm.validate_preset_name(name)
            assert valid is False
            assert reason

    def test_path_separator_rejected(self, tmp_presets_dir: Path):
        """Names with path separators are rejected."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        for bad in ["../evil", "a/b", "a\\b"]:
            valid, _ = pm.validate_preset_name(bad)
            assert valid is False

    def test_empty_name_rejected(self, tmp_presets_dir: Path):
        """Empty name is rejected."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        valid, _ = pm.validate_preset_name("")
        assert valid is False

    def test_too_long_name_rejected(self, tmp_presets_dir: Path):
        """Names longer than 50 characters are rejected."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        valid, _ = pm.validate_preset_name("a" * 51)
        assert valid is False

    def test_special_chars_rejected(self, tmp_presets_dir: Path):
        """Special characters (spaces, $, @) are rejected."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        for bad in ["my preset", "my$preset", "my@preset"]:
            valid, _ = pm.validate_preset_name(bad)
            assert valid is False

    def test_hyphen_allowed(self, tmp_presets_dir: Path):
        """Hyphens are allowed in preset names."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        valid, _ = pm.validate_preset_name("my-preset")
        assert valid is True


class TestSavePreset:
    """Tests for PresetManager.save_preset()."""

    def test_saves_json_file(self, tmp_presets_dir: Path):
        """Preset is persisted as a JSON file."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        pm.save_preset("testpreset", {"runs": 3, "context": 2048})
        saved = (tmp_presets_dir / "testpreset.json")
        assert saved.exists()
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert data["runs"] == 3

    def test_none_values_excluded(self, tmp_presets_dir: Path):
        """None values are excluded from saved preset."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        pm.save_preset("nulltest", {"runs": 3, "prompt": None})
        data = json.loads(
            (tmp_presets_dir / "nulltest.json").read_text(encoding="utf-8")
        )
        assert "prompt" not in data

    def test_raises_for_reserved_name(self, tmp_presets_dir: Path):
        """ValueError raised when trying to save with a reserved name."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.save_preset("default_classic", {"runs": 1})

    def test_raises_for_default_alias(self, tmp_presets_dir: Path):
        """ValueError raised when trying to save using readonly alias."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.save_preset("default", {"runs": 1})


class TestDeletePreset:
    """Tests for PresetManager.delete_preset()."""

    def test_deletes_existing_preset(self, tmp_presets_dir: Path):
        """Existing preset file is removed."""
        (tmp_presets_dir / "todelete.json").write_text(
            json.dumps({}), encoding="utf-8"
        )
        pm = PresetManager(presets_dir=tmp_presets_dir)
        pm.delete_preset("todelete")
        assert not (tmp_presets_dir / "todelete.json").exists()

    def test_raises_for_missing_preset(self, tmp_presets_dir: Path):
        """FileNotFoundError raised when preset does not exist."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(FileNotFoundError):
            pm.delete_preset("no_such_preset")

    def test_raises_for_reserved_name(self, tmp_presets_dir: Path):
        """ValueError raised when trying to delete a reserved preset."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.delete_preset("default_classic")

    def test_raises_for_default_alias(self, tmp_presets_dir: Path):
        """ValueError raised when trying to delete readonly alias."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        with pytest.raises(ValueError):
            pm.delete_preset("default")


class TestComparePresets:
    """Tests for PresetManager.compare_presets()."""

    def test_returns_dict_of_diffs(self, tmp_presets_dir: Path):
        """Returns a dict with preset_a and preset_b keys for each field."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        a = {"runs": 1}
        b = {"runs": 5}
        result = pm.compare_presets(a, b)
        assert "runs" in result
        assert result["runs"]["preset_a"] == 1
        assert result["runs"]["preset_b"] == 5

    def test_handles_empty_presets(self, tmp_presets_dir: Path):
        """Empty dicts are handled by falling back to defaults."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        result = pm.compare_presets({}, {})
        assert isinstance(result, dict)
        assert "runs" in result


class TestNormalizeLegacyKeys:
    """Tests for PresetManager._normalize_legacy_keys()."""

    def test_context_length_to_context(self, tmp_presets_dir: Path):
        """'context_length' key is mapped to 'context'."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        result = pm._normalize_legacy_keys({"context_length": 512})
        assert "context" in result
        assert result["context"] == 512

    def test_num_runs_to_runs(self, tmp_presets_dir: Path):
        """'num_runs' key is mapped to 'runs'."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        result = pm._normalize_legacy_keys({"num_runs": 3})
        assert "runs" in result

    def test_non_legacy_key_passes_through(self, tmp_presets_dir: Path):
        """Unknown keys are passed through unchanged."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        result = pm._normalize_legacy_keys({"my_custom_key": 42})
        assert result["my_custom_key"] == 42

    def test_empty_dict_returns_empty(self, tmp_presets_dir: Path):
        """Empty input returns empty dict."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        result = pm._normalize_legacy_keys({})
        assert result == {}


class TestPresetToCliArgs:
    """Tests for PresetManager.preset_to_cli_args()."""

    def test_runs_flag_added(self, tmp_presets_dir: Path):
        """--runs flag is added when 'runs' is present."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"runs": 5})
        assert "--runs" in args
        assert "5" in args

    def test_prompt_flag_added(self, tmp_presets_dir: Path):
        """--prompt flag is added when 'prompt' is present."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"prompt": "hello"})
        assert "--prompt" in args
        assert "hello" in args

    def test_only_vision_flag_added(self, tmp_presets_dir: Path):
        """--only-vision flag is added as a boolean flag."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"only_vision": True})
        assert "--only-vision" in args

    def test_flash_attention_true(self, tmp_presets_dir: Path):
        """--flash-attention flag is added when flash_attention is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"flash_attention": True})
        assert "--flash-attention" in args

    def test_flash_attention_false(self, tmp_presets_dir: Path):
        """--no-flash-attention flag is added when flash_attention is False."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"flash_attention": False})
        assert "--no-flash-attention" in args

    def test_use_mmap_true(self, tmp_presets_dir: Path):
        """--use-mmap flag is added when use_mmap is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"use_mmap": True})
        assert "--use-mmap" in args

    def test_use_mmap_false(self, tmp_presets_dir: Path):
        """--no-mmap flag is added when use_mmap is False."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"use_mmap": False})
        assert "--no-mmap" in args

    def test_use_mlock_true(self, tmp_presets_dir: Path):
        """--use-mlock flag is added when use_mlock is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"use_mlock": True})
        assert "--use-mlock" in args

    def test_zero_limit_excluded(self, tmp_presets_dir: Path):
        """Zero 'limit' is not emitted as a CLI argument."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"limit": 0})
        assert "--limit" not in args

    def test_positive_limit_included(self, tmp_presets_dir: Path):
        """Positive 'limit' is emitted as --limit."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"limit": 5})
        assert "--limit" in args

    def test_none_values_excluded(self, tmp_presets_dir: Path):
        """None values do not generate CLI flags."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"prompt": None, "runs": None})
        assert args == []

    def test_empty_string_excluded(self, tmp_presets_dir: Path):
        """Empty strings do not generate CLI flags."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"quants": "  "})
        assert "--quants" not in args

    def test_retest_flag(self, tmp_presets_dir: Path):
        """--retest boolean flag is emitted when retest is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"retest": True})
        assert "--retest" in args

    def test_dev_mode_flag(self, tmp_presets_dir: Path):
        """--dev-mode flag is emitted when dev_mode is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"dev_mode": True})
        assert "--dev-mode" in args

    def test_enable_profiling_flag(self, tmp_presets_dir: Path):
        """--enable-profiling flag is emitted when enable_profiling is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"enable_profiling": True})
        assert "--enable-profiling" in args

    def test_disable_gtt_flag(self, tmp_presets_dir: Path):
        """--disable-gtt flag is emitted when disable_gtt is True."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"disable_gtt": True})
        assert "--disable-gtt" in args

    def test_positive_max_size_included(self, tmp_presets_dir: Path):
        """Positive max_size generates --max-size flag."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"max_size": 8.0})
        assert "--max-size" in args

    def test_zero_max_size_excluded(self, tmp_presets_dir: Path):
        """Zero max_size does not generate --max-size flag."""
        pm = PresetManager(presets_dir=tmp_presets_dir)
        args = pm.preset_to_cli_args({"max_size": 0.0})
        assert "--max-size" not in args
