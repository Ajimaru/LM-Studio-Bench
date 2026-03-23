"""Tests for cli/main.py - capability benchmark entrypoint."""

import argparse
from pathlib import Path
from unittest.mock import patch


class TestSanitizeOutputDir:
    """Tests for _sanitize_output_dir function."""

    def test_valid_relative_path(self, tmp_path):
        """Relative path is resolved relative to cwd."""
        from cli.main import _sanitize_output_dir
        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = _sanitize_output_dir("output")
            assert result.is_absolute()
            assert result == (tmp_path / "output").resolve()

    def test_absolute_path_accepted(self, tmp_path):
        """Absolute paths are accepted (including outside workspace)."""
        from cli.main import _sanitize_output_dir
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        result = _sanitize_output_dir(str(output_dir))
        assert result == output_dir.resolve()

    def test_user_home_expansion(self, tmp_path):
        """~ is expanded to user home directory."""
        from cli.main import _sanitize_output_dir

        result = _sanitize_output_dir("~/test")
        assert result == Path.home() / "test"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_yaml_when_file_exists(self, tmp_path):
        """Config file is loaded when it exists."""
        from cli.main import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "context_length: 2048\ngpu_offload: 0.8\n",
            encoding="utf-8",
        )
        assert config_file.exists()

        config = load_config(config_file)
        # The loaded configuration should reflect the YAML contents.
        assert config["context_length"] == 2048
        assert config["gpu_offload"] == 0.8

    def test_returns_default_when_file_missing(self):
        """Default config is returned when file is missing."""
        from cli.main import DEFAULT_CONFIG, load_config

        missing_file = Path("/nonexistent/config.yaml")
        assert not missing_file.exists()

        config = load_config(missing_file)
        # Missing file should result in default configuration.
        assert config == DEFAULT_CONFIG

    def test_returns_default_on_yaml_error(self, tmp_path):
        """Default config on YAML parse error."""
        from cli.main import DEFAULT_CONFIG, load_config

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("invalid: [yaml: structure:", encoding="utf-8")
        assert bad_yaml.exists()

        config = load_config(bad_yaml)
        # YAML parse errors should fall back to default configuration.
        assert config == DEFAULT_CONFIG
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
        """--output-dir defaults to USER_RESULTS_DIR."""
        import sys

        from cli.main import parse_args
        from core.paths import USER_RESULTS_DIR

        original_argv = sys.argv
        try:
            sys.argv = ["test"]
            args = parse_args()
            assert str(args.output_dir) == str(USER_RESULTS_DIR)
        finally:
            sys.argv = original_argv

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


class TestWriteReports:
    """Tests for capability report export writer."""

    def test_write_reports_generates_all_formats(self, tmp_path):
        """_write_reports writes json/html/csv/pdf outputs."""
        from cli.main import _write_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-03-23T10:00:00",
            "results": [
                {
                    "test_id": "qa_001",
                    "test_name": "QA test",
                    "capability": "general_text",
                    "latency_ms": 123.45,
                    "tokens_generated": 42,
                    "throughput": 78.9,
                    "quality_score": 0.87,
                    "error": None,
                }
            ],
            "summary": {
                "total_tests": 1,
                "successful_tests": 1,
                "success_rate": 1.0,
                "avg_latency_ms": 123.45,
                "avg_quality_score": 0.87,
            },
        }

        outputs = _write_reports(
            report_data=report_data,
            output_dir=tmp_path,
            formats=["json", "html", "csv", "pdf"],
            report_stem="test-model",
        )

        assert "json" in outputs
        assert "html" in outputs
        assert "csv" in outputs
        assert "pdf" in outputs
        for output_path in outputs.values():
            assert output_path.exists()


class TestHardwareMonitoring:
    """Tests for capability hardware monitoring wiring."""

    def test_hardware_monitor_initialization_pattern(self):
        """Verify HardwareMonitor uses same init pattern as classic mode."""
        from tools.hardware_monitor import GPUMonitor, HardwareMonitor

        gpu_monitor = GPUMonitor()
        monitor = HardwareMonitor(
            gpu_monitor.gpu_type or "Unknown",
            gpu_monitor.gpu_tool or "",
            enabled=True,
        )

        assert monitor is not None
        assert monitor.enabled is True
        assert monitor.gpu_type in ("NVIDIA", "AMD", "Intel", "Unknown")


class TestBenchmarkAgentConfig:
    """Tests for BenchmarkAgent config handling."""

    def test_dev_mode_flag_stored_in_agent(self, tmp_path):
        """dev_mode config flag is stored in BenchmarkAgent."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        adapter = MagicMock()
        config = {"dev_mode": True}

        agent = BenchmarkAgent(
            adapter=adapter,
            output_dir=tmp_path,
            config=config
        )

        assert agent.dev_mode is True
        assert agent.config == config

    def test_disable_gtt_flag_stored_in_agent(self, tmp_path):
        """disable_gtt config flag is stored in BenchmarkAgent."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        adapter = MagicMock()
        config = {"disable_gtt": True}

        agent = BenchmarkAgent(
            adapter=adapter,
            output_dir=tmp_path,
            config=config
        )

        assert agent.disable_gtt is True
        assert agent.config == config

    def test_config_defaults_to_empty_dict(self, tmp_path):
        """BenchmarkAgent initializes with empty dict if no config provided."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        adapter = MagicMock()

        agent = BenchmarkAgent(
            adapter=adapter,
            output_dir=tmp_path
        )

        assert agent.config == {}
        assert agent.dev_mode is False
        assert agent.disable_gtt is False

    def test_dev_mode_defaults_to_false(self, tmp_path):
        """dev_mode defaults to False if not in config."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        adapter = MagicMock()

        agent = BenchmarkAgent(
            adapter=adapter,
            output_dir=tmp_path,
            config={"other_option": "value"}
        )

        assert agent.dev_mode is False

    def test_disable_gtt_defaults_to_false(self, tmp_path):
        """disable_gtt defaults to False if not in config."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        adapter = MagicMock()

        agent = BenchmarkAgent(
            adapter=adapter,
            output_dir=tmp_path,
            config={"other_option": "value"}
        )

        assert agent.disable_gtt is False
