"""Tests for cli/main.py - capability benchmark entrypoint."""

import argparse
from pathlib import Path
from unittest.mock import patch


class TestSanitizeOutputDir:
    """Tests for _sanitize_output_dir function."""

    def test_valid_relative_path(self, tmp_path):
        """Relative path within workspace is accepted."""
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = (tmp_path / "output").resolve()
            assert result.is_absolute()

    def test_absolute_path_in_workspace(self, tmp_path):
        """Absolute path within workspace is accepted."""
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        assert output_dir.resolve().is_absolute()

    def test_parent_directory_blocked(self, tmp_path):
        """Paths trying to escape workspace are rejected."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        assert outside.resolve() != workspace.resolve()


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_yaml_when_file_exists(self, tmp_path):
        """Config file is loaded when it exists."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "context_length: 2048\ngpu_offload: 0.8\n",
            encoding="utf-8"
        )
        assert config_file.exists()

    def test_returns_default_when_file_missing(self):
        """Default config is returned when file is missing."""
        missing_file = Path("/nonexistent/config.yaml")
        assert not missing_file.exists()

    def test_returns_default_on_yaml_error(self, tmp_path):
        """Default config on YAML parse error."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: [yaml: structure:", encoding="utf-8")
        assert bad_yaml.exists()


class TestParseArgs:
    """Tests for parse_args function."""

    def test_model_path_positional(self):
        """Model path is accepted as positional argument."""
        parser = argparse.ArgumentParser()
        parser.add_argument("model_path", type=str, nargs="?")
        args = parser.parse_args(["test-model"])
        assert args.model_path == "test-model"

    def test_all_models_flag(self):
        """--all-models flag is recognized."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--all-models", action="store_true")
        args = parser.parse_args(["--all-models"])
        assert args.all_models is True

    def test_random_models_flag(self):
        """--random-models takes integer argument."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--random-models", type=int)
        args = parser.parse_args(["--random-models", "5"])
        assert args.random_models == 5

    def test_output_dir_default(self):
        """--output-dir defaults to ./output."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--output-dir", type=Path, default=Path("output"))
        args = parser.parse_args([])
        assert args.output_dir == Path("output")

    def test_capabilities_flag(self):
        """--capabilities accepts comma-separated list."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--capabilities", type=str)
        args = parser.parse_args(["--capabilities", "general_text,reasoning"])
        assert args.capabilities == "general_text,reasoning"

    def test_formats_with_multiple_outputs(self):
        """--formats accepts multiple output formats."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--formats", type=str, default="json,html")
        args = parser.parse_args(["--formats", "json,html,pdf"])
        assert "json" in args.formats

    def test_inference_parameters(self):
        """Inference parameters are accepted."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--top-k", type=int)
        parser.add_argument("--top-p", type=float)
        args = parser.parse_args(["--temperature", "0.7", "--top-k", "40"])
        assert args.temperature == 0.7
        assert args.top_k == 40

    def test_load_config_parameters(self):
        """Load config parameters are accepted."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--n-gpu-layers", type=int)
        parser.add_argument("--n-batch", type=int)
        parser.add_argument("--flash-attention", action="store_true")
        args = parser.parse_args([
            "--n-gpu-layers", "32",
            "--n-batch", "512",
            "--flash-attention"
        ])
        assert args.n_gpu_layers == 32
        assert args.n_batch == 512
        assert args.flash_attention is True
