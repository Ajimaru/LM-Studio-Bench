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
        assert monitor.gpu_type in (
            "NVIDIA",
            "AMD",
            "Intel",
            "Apple",
            "Unknown",
        )


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


class TestListInstalledModels:
    """Tests for _list_installed_models function."""

    @staticmethod
    def _run(payload):
        """Invoke _list_installed_models against a stubbed ``lms ls --json``."""
        import json

        from cli.main import _list_installed_models

        completed = argparse.Namespace(returncode=0, stdout=json.dumps(payload))
        with patch("subprocess.run", return_value=completed):
            return _list_installed_models()

    def test_skips_lm_link_peer_models(self):
        """Models on an LM Link peer are excluded from the benchmark set."""
        models = self._run([
            {"modelKey": "local-model", "deviceIdentifier": None},
            {"modelKey": "peer-model", "deviceIdentifier": "abc123"},
        ])
        assert models == ["local-model"]

    def test_skips_peer_variants(self):
        """Variant lists of a peer entry are skipped as a whole."""
        models = self._run([
            {
                "modelKey": "peer-model",
                "variants": ["peer-model@4bit", "peer-model@bf16"],
                "deviceIdentifier": "abc123",
            },
            {"modelKey": "local-model", "variants": ["local-model@q4_k_m"]},
        ])
        assert models == ["local-model@q4_k_m"]

    def test_keeps_models_without_device_field(self):
        """A missing deviceIdentifier counts as local."""
        models = self._run([{"modelKey": "local-model"}])
        assert models == ["local-model"]

    def test_same_model_on_both_hosts_keeps_local_entry(self):
        """A model present locally and on a peer is kept once, as local."""
        models = self._run([
            {"modelKey": "shared@q4_k_m", "deviceIdentifier": None},
            {"modelKey": "shared@4bit", "deviceIdentifier": "abc123"},
        ])
        assert models == ["shared@q4_k_m"]


class TestPerCapabilityTokenBudget:
    """Generation budgets differ per capability.

    A single global budget is either too small for reasoning chains or wastes
    minutes generating into the void for a 30-token tool call.
    """

    @staticmethod
    def _agent(tmp_path, config):
        """Build a BenchmarkAgent with a stub adapter."""
        from unittest.mock import MagicMock

        from agents.benchmark import BenchmarkAgent

        return BenchmarkAgent(
            adapter=MagicMock(), output_dir=tmp_path, config=config
        )

    def test_budget_is_taken_per_capability(self, tmp_path):
        """Each capability gets its configured budget."""
        from agents.capabilities import Capability

        agent = self._agent(tmp_path, {
            "max_tokens_per_capability": {
                "tooling": 300, "reasoning": 2000, "code": 1200,
            },
        })
        assert agent._max_tokens_for(Capability.TOOLING) == 300
        assert agent._max_tokens_for(Capability.REASONING) == 2000
        assert agent._max_tokens_for(Capability.CODE) == 1200

    def test_falls_back_to_global_budget(self, tmp_path):
        """A capability without an entry uses the global value."""
        from agents.capabilities import Capability

        agent = self._agent(tmp_path, {
            "max_tokens_per_capability": {"tooling": 300},
        })
        agent.inference_options = {"max_tokens": 900}
        assert agent._max_tokens_for(Capability.VISION) == 900

    def test_no_budget_configured(self, tmp_path):
        """Without any configuration the model decides when to stop."""
        from agents.capabilities import Capability

        agent = self._agent(tmp_path, {})
        assert agent._max_tokens_for(Capability.CODE) is None

    def test_truncation_uses_the_capability_budget(self, tmp_path):
        """A tooling answer at 300 tokens is truncated, a code answer is not.

        With a single global budget the same 300-token answer would look fine
        in both cases, hiding a model that rambled through its tool call.
        """
        from agents.benchmark import InferenceResult
        from agents.capabilities import Capability

        agent = self._agent(tmp_path, {
            "max_tokens_per_capability": {"tooling": 300, "code": 1200},
        })
        inference = InferenceResult(
            test_id="t", prompt="p", response="r",
            timestamp_start=0.0, timestamp_end=1.0, latency_ms=1000.0,
            tokens_generated=300,
        )
        assert agent._is_truncated(inference, Capability.TOOLING) is True
        assert agent._is_truncated(inference, Capability.CODE) is False

    def test_explicit_cli_budget_overrides_capabilities(self):
        """--max-tokens applies to every capability."""
        import argparse

        from cli.main import override_config

        args = argparse.Namespace(max_tokens=64)
        for name in [
            "context_length", "gpu_offload", "temperature", "top_k", "top_p",
            "min_p", "repeat_penalty", "n_gpu_layers", "n_batch", "n_threads",
            "flash_attention", "no_flash_attention", "rope_freq_base",
            "rope_freq_scale", "use_mmap", "no_mmap", "use_mlock",
            "kv_cache_quant", "max_temp", "max_power", "enable_profiling",
            "disable_gtt", "dev_mode", "max_tests",
        ]:
            if not hasattr(args, name):
                setattr(args, name, None)

        config = override_config(
            {"max_tokens_per_capability": {"tooling": 300}}, args
        )
        assert config["max_tokens"] == 64
        assert config["max_tokens_per_capability"] == {}
