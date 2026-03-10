"""Tests for src/benchmark.py."""
import io
import sys
import threading
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


def _import_benchmark():
    """Import benchmark module with lmstudio mocked."""
    if "lmstudio" not in sys.modules:
        sys.modules["lmstudio"] = MagicMock()
    if "benchmark" in sys.modules:
        return sys.modules["benchmark"]
    import benchmark
    return benchmark


class TestNoJSONFilter:
    """Tests for benchmark.NoJSONFilter."""

    def test_allows_normal_messages(self):
        """Non-JSON messages are allowed through the filter."""
        bm = _import_benchmark()
        log_filter = bm.NoJSONFilter()
        record = MagicMock()
        record.getMessage.return_value = "Normal log message"
        assert log_filter.filter(record) is True

    def test_blocks_json_messages(self):
        """Messages starting with '{' are filtered out."""
        bm = _import_benchmark()
        log_filter = bm.NoJSONFilter()
        record = MagicMock()
        record.getMessage.return_value = '{"key": "value"}'
        assert log_filter.filter(record) is False

    def test_allows_message_with_embedded_braces(self):
        """Messages not starting with '{' are not filtered."""
        bm = _import_benchmark()
        log_filter = bm.NoJSONFilter()
        record = MagicMock()
        record.getMessage.return_value = "Data: {'key': 'value'}"
        assert log_filter.filter(record) is True


class TestAutoFlushStream:
    """Tests for benchmark.AutoFlushStream."""

    def test_write_calls_flush(self):
        """write() immediately flushes the underlying stream."""
        bm = _import_benchmark()
        mock_stream = MagicMock()
        stream = bm.AutoFlushStream(mock_stream)
        stream.write("hello")
        mock_stream.write.assert_called_once_with("hello")
        mock_stream.flush.assert_called()

    def test_flush_delegates_to_stream(self):
        """flush() delegates to the underlying stream's flush."""
        bm = _import_benchmark()
        mock_stream = MagicMock()
        stream = bm.AutoFlushStream(mock_stream)
        stream.flush()
        mock_stream.flush.assert_called()

    def test_getattr_delegates_to_stream(self):
        """Attribute access falls through to the wrapped stream."""
        bm = _import_benchmark()
        mock_stream = MagicMock()
        mock_stream.some_attr = "value"
        stream = bm.AutoFlushStream(mock_stream)
        assert stream.some_attr == "value"


class TestBenchmarkResultDataclass:
    """Tests for benchmark.BenchmarkResult dataclass."""

    def _make_result(self, **kwargs):
        """Return a minimal BenchmarkResult."""
        bm = _import_benchmark()
        defaults = {
            "model_name": "test-model",
            "quantization": "Q4_K_M",
            "gpu_type": "NVIDIA",
            "gpu_offload": 1.0,
            "vram_mb": "8192",
            "avg_tokens_per_sec": 42.0,
            "avg_ttft": 0.5,
            "avg_gen_time": 1.0,
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "timestamp": "2024-01-01T00:00:00",
            "params_size": "7B",
            "architecture": "llama",
            "max_context_length": 4096,
            "model_size_gb": 4.5,
            "has_vision": False,
            "has_tools": False,
            "tokens_per_sec_per_gb": 9.33,
            "tokens_per_sec_per_billion_params": 6.0,
        }
        defaults.update(kwargs)
        return bm.BenchmarkResult(**defaults)

    def test_creation_with_required_fields(self):
        """BenchmarkResult can be created with required fields."""
        result = self._make_result()
        bm = _import_benchmark()
        assert isinstance(result, bm.BenchmarkResult)

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None."""
        result = self._make_result()
        assert result.temp_celsius_min is None
        assert result.power_watts_avg is None
        assert result.n_gpu_layers is None

    def test_converts_to_dict(self):
        """asdict() works correctly on BenchmarkResult."""
        result = self._make_result()
        d = asdict(result)
        assert "model_name" in d
        assert d["avg_tokens_per_sec"] == 42.0

    def test_has_vision_stored(self):
        """has_vision field is stored correctly."""
        result = self._make_result(has_vision=True)
        assert result.has_vision is True

    def test_speed_delta_pct_optional(self):
        """speed_delta_pct can be set or remain None."""
        result = self._make_result(speed_delta_pct=5.5)
        assert result.speed_delta_pct == 5.5


class TestHardwareMonitor:
    """Tests for benchmark.HardwareMonitor."""

    def test_init_disabled(self):
        """HardwareMonitor starts with enabled=False."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        assert monitor.enabled is False
        assert monitor.monitoring is False

    def test_init_enabled(self):
        """HardwareMonitor with enabled=True stores correct state."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        assert monitor.enabled is True
        assert monitor.gpu_type == "NVIDIA"
        assert monitor.gpu_tool == "nvidia-smi"

    def test_start_when_disabled_does_not_start_thread(self):
        """start() with enabled=False does not start monitoring."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        monitor.start()
        assert monitor.monitoring is False
        assert monitor.thread is None

    def test_start_without_gpu_tool_does_not_start(self):
        """start() with no gpu_tool does not start monitoring."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=True)
        monitor.start()
        assert monitor.monitoring is False

    def test_stop_returns_stats_dict(self):
        """stop() returns a dict with expected statistic keys."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        stats = monitor.stop()
        assert isinstance(stats, dict)
        expected_keys = [
            "temp_celsius_min",
            "temp_celsius_max",
            "temp_celsius_avg",
            "power_watts_min",
            "power_watts_max",
            "power_watts_avg",
        ]
        for key in expected_keys:
            assert key in stats

    def test_stop_returns_none_for_empty_readings(self):
        """stop() returns None for metrics with no readings."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        stats = monitor.stop()
        assert stats["temp_celsius_min"] is None
        assert stats["power_watts_avg"] is None

    def test_stop_computes_stats_from_readings(self):
        """stop() computes min/max/avg from collected readings."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        monitor.temps = [60.0, 70.0, 80.0]
        monitor.powers = [100.0, 120.0]
        stats = monitor.stop()
        assert stats["temp_celsius_min"] == 60.0
        assert stats["temp_celsius_max"] == 80.0
        assert stats["temp_celsius_avg"] == pytest.approx(70.0)

    def test_get_cpu_usage_returns_float_or_none(self):
        """_get_cpu_usage() returns a float or None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        with patch("psutil.cpu_percent", return_value=42.5):
            result = monitor._get_cpu_usage()
        assert result == pytest.approx(42.5)

    def test_get_cpu_usage_handles_exception(self):
        """_get_cpu_usage() returns None on exception."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        with patch("psutil.cpu_percent", side_effect=RuntimeError("fail")):
            result = monitor._get_cpu_usage()
        assert result is None

    def test_get_ram_usage_returns_float(self):
        """_get_ram_usage() returns a smoothed float value."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        mock_mem = MagicMock()
        mock_mem.used = 4 * (1024 ** 3)
        with patch("psutil.virtual_memory", return_value=mock_mem):
            result = monitor._get_ram_usage()
        assert result == pytest.approx(4.0)

    def test_get_temperature_returns_none_without_tool(self):
        """_get_temperature() returns None when gpu_tool is None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_temperature() is None

    def test_get_temperature_nvidia_success(self):
        """_get_temperature() parses NVIDIA temperature correctly."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "72\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_temperature()
        assert result == pytest.approx(72.0)

    def test_get_temperature_nvidia_failure(self):
        """_get_temperature() returns None on non-zero returncode."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_temperature()
        assert result is None

    def test_get_power_draw_returns_none_without_tool(self):
        """_get_power_draw() returns None when gpu_tool is None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_power_draw() is None

    def test_get_power_nvidia_success(self):
        """_get_power_draw() parses NVIDIA power correctly."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "150.5\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_power_draw()
        assert result == pytest.approx(150.5)

    def test_get_vram_usage_returns_none_without_tool(self):
        """_get_vram_usage() returns None when gpu_tool is None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_vram_usage() is None

    def test_get_vram_nvidia_success(self):
        """_get_vram_usage() returns GB for NVIDIA correctly."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "8192\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_vram_usage()
        assert result == pytest.approx(8.0)

    def test_get_gtt_returns_none_for_nvidia(self):
        """_get_gtt_usage() returns None for non-AMD GPUs."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        assert monitor._get_gtt_usage() is None

    def test_get_gtt_returns_none_without_tool(self):
        """_get_gtt_usage() returns None when gpu_tool is None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(gpu_type="AMD", gpu_tool=None, enabled=False)
        assert monitor._get_gtt_usage() is None


class TestBenchmarkCache:
    """Tests for benchmark.BenchmarkCache."""

    def test_init_creates_db(self, tmp_path: Path):
        """BenchmarkCache creates SQLite database on init."""
        bm = _import_benchmark()
        db_path = tmp_path / "test.db"
        cache = bm.BenchmarkCache(db_path=db_path)
        assert db_path.exists()

    def test_compute_params_hash_is_consistent(self, tmp_path: Path):
        """compute_params_hash returns the same hash for same inputs."""
        bm = _import_benchmark()
        h1 = bm.BenchmarkCache.compute_params_hash(
            "prompt", 2048, {"temperature": 0.1}
        )
        h2 = bm.BenchmarkCache.compute_params_hash(
            "prompt", 2048, {"temperature": 0.1}
        )
        assert h1 == h2

    def test_compute_params_hash_differs_for_different_inputs(self, tmp_path: Path):
        """Different inputs produce different hashes."""
        bm = _import_benchmark()
        h1 = bm.BenchmarkCache.compute_params_hash(
            "prompt A", 2048, {"temperature": 0.1}
        )
        h2 = bm.BenchmarkCache.compute_params_hash(
            "prompt B", 2048, {"temperature": 0.1}
        )
        assert h1 != h2

    def test_get_cached_result_returns_none_when_empty(self, tmp_path: Path):
        """Returns None when no cached result exists."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        result = cache.get_cached_result("pub/model", "abc123")
        assert result is None

    def test_list_cached_models_empty(self, tmp_path: Path):
        """Returns empty list when no results cached."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        result = cache.list_cached_models()
        assert result == []

    def test_get_all_results_empty(self, tmp_path: Path):
        """Returns empty list when database is empty."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        result = cache.get_all_results()
        assert result == []

    def test_save_and_retrieve_result(self, tmp_path: Path):
        """Saved result can be retrieved from the cache."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        result = bm.BenchmarkResult(
            model_name="test-model",
            quantization="Q4",
            gpu_type="NVIDIA",
            gpu_offload=1.0,
            vram_mb="8192",
            avg_tokens_per_sec=55.0,
            avg_ttft=0.3,
            avg_gen_time=0.8,
            prompt_tokens=10,
            completion_tokens=50,
            timestamp="2024-01-01T00:00:00",
            params_size="7B",
            architecture="llama",
            max_context_length=4096,
            model_size_gb=4.0,
            has_vision=False,
            has_tools=False,
            tokens_per_sec_per_gb=13.75,
            tokens_per_sec_per_billion_params=7.86,
            inference_params_hash="abcd1234",
        )
        cache.save_result(
            result, "pub/test-model", "abcd1234", "test prompt", 2048
        )
        cached = cache.get_cached_result("pub/test-model", "abcd1234")
        assert cached is not None
        assert cached.model_name == "test-model"
        assert cached.avg_tokens_per_sec == pytest.approx(55.0)

    def test_export_to_json(self, tmp_path: Path):
        """export_to_json creates a JSON file."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        out_file = tmp_path / "export.json"
        cache.export_to_json(out_file)
        import json
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, list)


class TestGPUMonitor:
    """Tests for benchmark.GPUMonitor."""

    def test_init_detects_no_gpu_when_no_tools(self):
        """GPUMonitor falls back to Unknown when no GPU tools found."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        assert monitor.gpu_type == "Unknown"
        assert monitor.gpu_model == "Unknown"

    def test_find_tool_returns_none_when_absent(self):
        """_find_tool returns None when tool is not in PATH."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        result = monitor._find_tool("nonexistent_tool_xyz", ["/nowhere"])
        assert result is None

    def test_find_tool_returns_path_from_which(self):
        """_find_tool returns tool name when found via which."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        with patch("shutil.which", return_value="/usr/bin/mytool"):
            result = monitor._find_tool("mytool", [])
        assert result == "mytool"

    def test_detect_nvidia_gpu(self):
        """GPUMonitor detects NVIDIA GPU when nvidia-smi is available."""
        bm = _import_benchmark()
        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "NVIDIA GeForce RTX 4090\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_run):
            monitor = bm.GPUMonitor()
        assert monitor.gpu_type == "NVIDIA"
        assert monitor.gpu_tool == "nvidia-smi"

    def test_detect_amd_gpu_via_rocm_smi(self):
        """GPUMonitor detects AMD GPU when rocm-smi is available."""
        bm = _import_benchmark()

        def mock_which(name, *args, **kwargs):
            return "/usr/bin/rocm-smi" if name == "rocm-smi" else None

        mock_run = MagicMock()
        mock_run.returncode = 1

        with patch("shutil.which", side_effect=mock_which), \
                patch("subprocess.run", return_value=mock_run), \
                patch("glob.glob", return_value=[]):
            monitor = bm.GPUMonitor()
        assert monitor.gpu_type == "AMD"

    def test_get_vram_usage_returns_na_without_tool(self):
        """get_vram_usage returns 'N/A' when no GPU tool found."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        assert monitor.get_vram_usage() == "N/A"

    def test_get_vram_usage_nvidia_success(self):
        """get_vram_usage returns VRAM string for NVIDIA."""
        bm = _import_benchmark()
        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "RTX 4090\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_run):
            monitor = bm.GPUMonitor()

        mock_vram = MagicMock()
        mock_vram.returncode = 0
        mock_vram.stdout = "8192\n"
        with patch("subprocess.run", return_value=mock_vram):
            result = monitor.get_vram_usage()
        assert "8192" in result

    def test_find_amd_sysfs_path_returns_none_on_failure(self):
        """_find_amd_sysfs_path returns None when lspci fails."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout=""),
        ):
            result = monitor._find_amd_sysfs_path()
        assert result is None

    def test_detect_amd_gpu_model_returns_string(self):
        """_detect_amd_gpu_model always returns a string."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
            result = monitor._detect_amd_gpu_model()
        assert isinstance(result, str)
        assert "AMD" in result

    def test_detect_intel_gpu_model_returns_string(self):
        """_detect_intel_gpu_model always returns a string."""
        bm = _import_benchmark()
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = bm.GPUMonitor()
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
            result = monitor._detect_intel_gpu_model()
        assert isinstance(result, str)


class TestLMStudioServerManager:
    """Tests for benchmark.LMStudioServerManager."""

    def test_is_server_running_returns_false_on_error(self):
        """is_server_running returns False when lms command fails."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr=""),
        ):
            result = bm.LMStudioServerManager.is_server_running()
        assert result is False

    def test_is_server_running_returns_true_when_running(self):
        """is_server_running returns True when output contains 'running'."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="Server is running", stderr=""),
        ):
            result = bm.LMStudioServerManager.is_server_running()
        assert result is True

    def test_is_server_running_exception_returns_false(self):
        """is_server_running returns False on exception."""
        bm = _import_benchmark()
        with patch("subprocess.run", side_effect=FileNotFoundError("lms not found")):
            result = bm.LMStudioServerManager.is_server_running()
        assert result is False

    def test_ensure_server_running_when_already_running(self):
        """ensure_server_running returns True when server is already up."""
        bm = _import_benchmark()
        with patch.object(bm.LMStudioServerManager, "is_server_running", return_value=True):
            result = bm.LMStudioServerManager.ensure_server_running()
        assert result is True


class TestModelDiscovery:
    """Tests for benchmark.ModelDiscovery."""

    def setup_method(self):
        """Clear metadata cache before each test."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}

    def test_get_installed_models_returns_empty_on_error(self):
        """get_installed_models returns empty list when lms fails."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr="error"),
        ):
            result = bm.ModelDiscovery.get_installed_models()
        assert result == []

    def test_get_installed_models_parses_json(self):
        """get_installed_models parses lms JSON output."""
        import json as _json
        bm = _import_benchmark()
        models_data = [
            {
                "type": "llm",
                "modelKey": "pub/model",
                "variants": ["pub/model@q4", "pub/model@q8"],
            }
        ]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _json.dumps(models_data)
        with patch("subprocess.run", return_value=mock_result):
            result = bm.ModelDiscovery.get_installed_models()
        assert "pub/model@q4" in result
        assert "pub/model@q8" in result

    def test_get_installed_models_without_variants(self):
        """get_installed_models returns modelKey when no variants."""
        import json as _json
        bm = _import_benchmark()
        models_data = [{"type": "llm", "modelKey": "pub/model", "variants": []}]
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = _json.dumps(models_data)
        with patch("subprocess.run", return_value=mock_result):
            result = bm.ModelDiscovery.get_installed_models()
        assert "pub/model" in result

    def test_get_model_metadata_returns_defaults(self):
        """get_model_metadata returns default dict for unknown model."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.ModelDiscovery.get_model_metadata("unknown/model")
        assert result["architecture"] == "unknown"
        assert result["has_vision"] is False

    def test_get_model_metadata_strips_quantization(self):
        """get_model_metadata strips @quant suffix before lookup."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {
            "pub/model": {
                "architecture": "llama",
                "params_size": "7B",
                "max_context_length": 4096,
                "model_size_gb": 4.0,
                "has_vision": False,
                "has_tools": False,
            }
        }
        result = bm.ModelDiscovery.get_model_metadata("pub/model@q4")
        assert result["architecture"] == "llama"

    def test_filter_models_no_filter_returns_all(self):
        """filter_models returns all models when no filter args."""
        bm = _import_benchmark()
        models = ["model/a@q4", "model/b@q8"]
        result = bm.ModelDiscovery.filter_models(models, {})
        assert result == models

    def test_filter_models_by_include_pattern(self):
        """filter_models applies include_models regex filter."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}
        models = ["llama/model@q4", "gpt/model@q4"]
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.ModelDiscovery.filter_models(
                models, {"include_models": "llama"}
            )
        assert len(result) == 1
        assert "llama/model@q4" in result

    def test_filter_models_by_exclude_pattern(self):
        """filter_models applies exclude_models regex filter."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}
        models = ["llama/model@q4", "gpt/model@q4"]
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.ModelDiscovery.filter_models(
                models, {"exclude_models": "gpt"}
            )
        assert "gpt/model@q4" not in result

    def test_filter_models_invalid_include_pattern(self):
        """filter_models returns empty list for invalid include regex."""
        bm = _import_benchmark()
        result = bm.ModelDiscovery.filter_models(
            ["model@q4"], {"include_models": "[invalid"}
        )
        assert result == []

    def test_filter_models_invalid_exclude_pattern(self):
        """filter_models returns empty list for invalid exclude regex."""
        bm = _import_benchmark()
        result = bm.ModelDiscovery.filter_models(
            ["model@q4"], {"exclude_models": "[invalid"}
        )
        assert result == []

    def test_filter_models_by_quant(self):
        """filter_models filters by quantization string."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}
        models = ["pub/model@q4_k_m", "pub/model@q8_0"]
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.ModelDiscovery.filter_models(
                models, {"quants": "q4"}
            )
        assert "pub/model@q4_k_m" in result
        assert "pub/model@q8_0" not in result

    def test_get_metadata_cache_empty_on_lms_fail(self):
        """_get_metadata_cache returns empty dict when lms fails."""
        bm = _import_benchmark()
        bm.ModelDiscovery._metadata_cache = {}
        with patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.ModelDiscovery._get_metadata_cache()
        assert result == {}

    def test_get_scraped_metadata_returns_empty_when_no_db(self, tmp_path: Path):
        """get_scraped_metadata returns empty dict when DB missing."""
        bm = _import_benchmark()
        non_existent = tmp_path / "no_metadata.db"
        with patch("benchmark.METADATA_DATABASE_FILE", non_existent):
            result = bm.ModelDiscovery.get_scraped_metadata("pub/model")
        assert result == {}


class TestLMStudioBenchmarkStaticMethods:
    """Tests for LMStudioBenchmark static methods."""

    def test_get_lmstudio_version_from_semver(self):
        """get_lmstudio_version returns semver from lms output."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="LM Studio v1.2.3\n"),
        ):
            result = bm.LMStudioBenchmark.get_lmstudio_version()
        assert "v1.2.3" in result

    def test_get_lmstudio_version_commit_hash(self):
        """get_lmstudio_version returns commit hash string."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout="CLI commit: abc1234def\n",
            ),
        ):
            result = bm.LMStudioBenchmark.get_lmstudio_version()
        assert result is not None

    def test_get_lmstudio_version_returns_none_on_error(self):
        """get_lmstudio_version returns None when lms fails."""
        bm = _import_benchmark()
        with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
            result = bm.LMStudioBenchmark.get_lmstudio_version()
        assert result is None

    def test_get_nvidia_driver_version_success(self):
        """get_nvidia_driver_version returns version string."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="535.104.05\n"),
        ):
            result = bm.LMStudioBenchmark.get_nvidia_driver_version()
        assert "535" in result

    def test_get_nvidia_driver_version_returns_none_on_error(self):
        """get_nvidia_driver_version returns None when nvidia-smi fails."""
        bm = _import_benchmark()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = bm.LMStudioBenchmark.get_nvidia_driver_version()
        assert result is None

    def test_get_intel_driver_version_returns_none_on_error(self):
        """get_intel_driver_version returns None when intel_gpu_top fails."""
        bm = _import_benchmark()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = bm.LMStudioBenchmark.get_intel_driver_version()
        assert result is None

    def test_get_intel_driver_version_found(self):
        """get_intel_driver_version returns version when tool found."""
        bm = _import_benchmark()
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout="version 2.0\n",
                stderr="",
            ),
        ):
            result = bm.LMStudioBenchmark.get_intel_driver_version()
        assert result is None or isinstance(result, str)

    def test_get_os_info_returns_tuple(self):
        """get_os_info returns a (str, str) tuple or (None, None)."""
        bm = _import_benchmark()
        result = bm.LMStudioBenchmark.get_os_info()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_python_version_returns_string(self):
        """get_python_version returns a non-empty string."""
        bm = _import_benchmark()
        result = bm.LMStudioBenchmark.get_python_version()
        assert result is not None
        assert "." in result

    def test_get_cpu_model_returns_string_or_none(self):
        """get_cpu_model returns a string or None."""
        bm = _import_benchmark()
        result = bm.LMStudioBenchmark.get_cpu_model()
        assert result is None or isinstance(result, str)

    def test_get_rocm_driver_version_returns_none_on_error(self):
        """get_rocm_driver_version returns None when rocm-smi is absent."""
        bm = _import_benchmark()
        with patch("glob.glob", return_value=[]), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            result = bm.LMStudioBenchmark.get_rocm_driver_version()
        assert result is None

    def test_get_rocm_driver_version_via_dpkg(self):
        """get_rocm_driver_version tries dpkg as fallback."""
        bm = _import_benchmark()

        def mock_run(cmd, *args, **kwargs):
            if cmd[0] == "dpkg":
                return MagicMock(returncode=0, stdout="ii rocm-smi 5.3 amd64\n")
            return MagicMock(returncode=1, stdout="")

        with patch("glob.glob", return_value=[]), \
                patch("subprocess.run", side_effect=mock_run):
            result = bm.LMStudioBenchmark.get_rocm_driver_version()
        assert result is None or isinstance(result, str)


class TestHardwareMonitorAdvanced:
    """Additional tests for benchmark.HardwareMonitor."""

    def test_init_with_amd_sysfs(self):
        """HardwareMonitor init with AMD sysfs tool calls init paths."""
        bm = _import_benchmark()
        with patch("glob.glob", return_value=[]):
            monitor = bm.HardwareMonitor("AMD", "sysfs", enabled=False)
        assert monitor.gpu_type == "AMD"

    def test_start_enabled_starts_thread(self):
        """start() starts a background thread when enabled."""
        import time
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "nvidia-smi", enabled=True)
        monitor._monitor_loop = MagicMock(return_value=None)
        monitor.start()
        time.sleep(0.05)
        monitor.monitoring = False
        assert monitor.thread is not None

    def test_get_power_draw_returns_none_without_tool(self):
        """_get_power_draw returns None when gpu_tool is empty."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "", enabled=False)
        result = monitor._get_power_draw()
        assert result is None

    def test_get_power_draw_nvidia_success(self):
        """_get_power_draw returns float for NVIDIA GPU."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="150.00\n"),
        ):
            result = monitor._get_power_draw()
        assert result == pytest.approx(150.0, 0.1)

    def test_stop_with_data_returns_averages(self):
        """stop() computes averages when data has been recorded."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        monitor.temps = [70.0, 72.0, 74.0]
        monitor.powers = [120.0, 125.0]
        monitor.vrams = [8.0, 8.5]
        stats = monitor.stop()
        assert stats["temp_celsius_avg"] == pytest.approx(72.0, 0.01)
        assert stats["power_watts_avg"] is not None

    def test_get_cpu_usage_returns_float(self):
        """_get_cpu_usage returns a float value."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        result = monitor._get_cpu_usage()
        assert result is None or isinstance(result, float)

    def test_get_ram_usage_returns_float_or_none(self):
        """_get_ram_usage returns a float or None."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        result = monitor._get_ram_usage()
        assert result is None or isinstance(result, float)


class TestLMStudioBenchmarkInit:
    """Tests for LMStudioBenchmark instantiation."""

    def test_basic_init_with_mocked_dependencies(self, tmp_path: Path):
        """LMStudioBenchmark initializes with mocked subprocess calls."""
        bm = _import_benchmark()
        failed_run = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", return_value=failed_run), \
                patch("shutil.which", return_value=None), \
                patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm.LMStudioBenchmark, "get_lmstudio_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_nvidia_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_rocm_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_intel_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_os_info",
                             return_value=("Linux", "6.0")), \
                patch.object(bm.LMStudioBenchmark, "get_cpu_model",
                             return_value="Test CPU"), \
                patch.object(bm.LMStudioBenchmark, "get_python_version",
                             return_value="3.12.0"):
            benchmark = bm.LMStudioBenchmark(num_runs=1)
        assert benchmark.num_measurement_runs == 1
        assert benchmark.rank_by == "speed"
        assert benchmark.use_cache is True

    def test_init_with_rest_api_enabled(self, tmp_path: Path):
        """LMStudioBenchmark initializes REST client when use_rest_api=True."""
        bm = _import_benchmark()
        failed_run = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", return_value=failed_run), \
                patch("shutil.which", return_value=None), \
                patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm.LMStudioBenchmark, "get_lmstudio_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_nvidia_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_rocm_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_intel_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_os_info",
                             return_value=("Linux", "6.0")), \
                patch.object(bm.LMStudioBenchmark, "get_cpu_model",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_python_version",
                             return_value="3.12.0"):
            benchmark = bm.LMStudioBenchmark(use_rest_api=True)
        assert benchmark.rest_client is not None

    def test_init_with_filter_args(self, tmp_path: Path):
        """LMStudioBenchmark stores filter_args."""
        bm = _import_benchmark()
        failed_run = MagicMock(returncode=1, stdout="", stderr="")
        filter_args = {"quants": "Q4_K_M", "arch": "llama"}
        with patch("subprocess.run", return_value=failed_run), \
                patch("shutil.which", return_value=None), \
                patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm.LMStudioBenchmark, "get_lmstudio_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_nvidia_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_rocm_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_intel_driver_version",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_os_info",
                             return_value=("Linux", "6.0")), \
                patch.object(bm.LMStudioBenchmark, "get_cpu_model",
                             return_value=None), \
                patch.object(bm.LMStudioBenchmark, "get_python_version",
                             return_value="3.12.0"):
            benchmark = bm.LMStudioBenchmark(filter_args=filter_args)
        assert benchmark.filter_args["quants"] == "Q4_K_M"


def _make_benchmark_instance(bm, tmp_path) -> Any:
    """Create a LMStudioBenchmark instance with all deps mocked."""
    failed_run = MagicMock(returncode=1, stdout="", stderr="")
    with patch("subprocess.run", return_value=failed_run), \
            patch("shutil.which", return_value=None), \
            patch.object(bm, "RESULTS_DIR", tmp_path), \
            patch.object(bm.LMStudioBenchmark, "get_lmstudio_version",
                         return_value=None), \
            patch.object(bm.LMStudioBenchmark, "get_nvidia_driver_version",
                         return_value=None), \
            patch.object(bm.LMStudioBenchmark, "get_rocm_driver_version",
                         return_value=None), \
            patch.object(bm.LMStudioBenchmark, "get_intel_driver_version",
                         return_value=None), \
            patch.object(bm.LMStudioBenchmark, "get_os_info",
                         return_value=("Linux", "6.0")), \
            patch.object(bm.LMStudioBenchmark, "get_cpu_model",
                         return_value=None), \
            patch.object(bm.LMStudioBenchmark, "get_python_version",
                         return_value="3.12.0"):
        instance = bm.LMStudioBenchmark(num_runs=1)
    return instance


def _make_result(bm, model_name="test/model", quantization="Q4_K_M",
                 speed=50.0, ttft=0.3, vision=False, tools=False,
                 arch="llama") -> Any:
    """Create a BenchmarkResult for testing."""
    return bm.BenchmarkResult(
        model_name=model_name,
        quantization=quantization,
        gpu_type="NVIDIA",
        gpu_offload=1.0,
        vram_mb="8192",
        avg_tokens_per_sec=speed,
        avg_ttft=ttft,
        avg_gen_time=0.8,
        prompt_tokens=10,
        completion_tokens=50,
        timestamp="2024-01-01T00:00:00",
        params_size="7B",
        architecture=arch,
        max_context_length=4096,
        model_size_gb=4.0,
        has_vision=vision,
        has_tools=tools,
        tokens_per_sec_per_gb=speed / 4.0,
        tokens_per_sec_per_billion_params=speed / 7.0,
    )


class TestLMStudioBenchmarkFiltersAndSorting:
    """Tests for LMStudioBenchmark data-processing methods."""

    def test_sort_results_by_speed(self, tmp_path: Path):
        """sort_results('speed') returns descending order by tok/s."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [
            _make_result(bm, speed=30.0),
            _make_result(bm, speed=60.0),
            _make_result(bm, speed=45.0),
        ]
        sorted_results = bench.sort_results("speed")
        speeds = [r.avg_tokens_per_sec for r in sorted_results]
        assert speeds == sorted(speeds, reverse=True)

    def test_sort_results_by_efficiency(self, tmp_path: Path):
        """sort_results('efficiency') ranks by tokens/s/GB."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        r1 = _make_result(bm, speed=30.0)
        r2 = _make_result(bm, speed=60.0)
        bench.results = [r1, r2]
        sorted_results = bench.sort_results("efficiency")
        assert sorted_results[0].tokens_per_sec_per_gb >= sorted_results[-1].tokens_per_sec_per_gb

    def test_sort_results_by_ttft(self, tmp_path: Path):
        """sort_results('ttft') ranks ascending by TTFT."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [
            _make_result(bm, ttft=0.5),
            _make_result(bm, ttft=0.1),
            _make_result(bm, ttft=0.3),
        ]
        sorted_results = bench.sort_results("ttft")
        ttfts = [r.avg_ttft for r in sorted_results]
        assert ttfts == sorted(ttfts)

    def test_sort_results_by_vram(self, tmp_path: Path):
        """sort_results('vram') ranks ascending by VRAM usage."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        r1 = _make_result(bm)
        r1.vram_mb = "4096"
        r2 = _make_result(bm)
        r2.vram_mb = "8192"
        bench.results = [r2, r1]
        sorted_results = bench.sort_results("vram")
        assert sorted_results[0].vram_mb == "4096"

    def test_sort_results_unknown_key_falls_back_to_speed(self, tmp_path: Path):
        """sort_results with unknown key defaults to speed sort."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [
            _make_result(bm, speed=40.0),
            _make_result(bm, speed=80.0),
        ]
        sorted_results = bench.sort_results("unknown_key")
        assert sorted_results[0].avg_tokens_per_sec >= sorted_results[-1].avg_tokens_per_sec

    def test_analyze_best_quantizations_empty_results(self, tmp_path: Path):
        """_analyze_best_quantizations returns empty dict with no results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = []
        result = bench._analyze_best_quantizations()
        assert result == {}

    def test_analyze_best_quantizations_with_data(self, tmp_path: Path):
        """_analyze_best_quantizations tracks best speed per model."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [
            _make_result(bm, model_name="pub/model", quantization="Q4_K_M", speed=50.0),
            _make_result(bm, model_name="pub/model", quantization="Q8_0", speed=60.0),
        ]
        result = bench._analyze_best_quantizations()
        assert "pub/model" in result
        assert result["pub/model"]["best_speed"].avg_tokens_per_sec == 60.0

    def test_calculate_percentile_stats_empty(self, tmp_path: Path):
        """calculate_percentile_stats returns empty dict with < 3 results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [_make_result(bm)]
        result = bench.calculate_percentile_stats()
        assert result == {}

    def test_calculate_percentile_stats_with_data(self, tmp_path: Path):
        """calculate_percentile_stats returns speed percentiles."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [_make_result(bm, speed=s) for s in [30, 50, 70, 90, 110]]
        result = bench.calculate_percentile_stats()
        assert isinstance(result, dict)
        assert "speed" in result

    def test_generate_quantization_comparison_empty(self, tmp_path: Path):
        """generate_quantization_comparison returns dict with no results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = []
        result = bench.generate_quantization_comparison()
        assert isinstance(result, dict)

    def test_generate_quantization_comparison_with_data(self, tmp_path: Path):
        """generate_quantization_comparison groups by model."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [
            _make_result(bm, model_name="pub/m", quantization="Q4_K_M", speed=50),
            _make_result(bm, model_name="pub/m", quantization="Q8_0", speed=60),
        ]
        result = bench.generate_quantization_comparison()
        assert "pub/m" in result


class TestLMStudioBenchmarkFiltering:
    """Tests for LMStudioBenchmark._matches_filters()."""

    def test_no_filters_always_matches(self, tmp_path: Path):
        """_matches_filters returns True when no filters active."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        result = _make_result(bm)
        assert bench._matches_filters(result) is True

    def test_only_vision_filter_excludes_non_vision(self, tmp_path: Path):
        """_matches_filters returns False for non-vision when only_vision is set."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"only_vision": True}
        result = _make_result(bm, vision=False)
        assert bench._matches_filters(result) is False

    def test_only_vision_filter_includes_vision(self, tmp_path: Path):
        """_matches_filters returns True for vision model when only_vision is set."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"only_vision": True}
        result = _make_result(bm, vision=True)
        assert bench._matches_filters(result) is True

    def test_only_tools_filter_excludes_non_tool(self, tmp_path: Path):
        """_matches_filters returns False for non-tool when only_tools is set."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"only_tools": True}
        result = _make_result(bm, tools=False)
        assert bench._matches_filters(result) is False

    def test_quants_filter_matches(self, tmp_path: Path):
        """_matches_filters passes result with matching quantization."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"quants": "q4_k_m"}
        result = _make_result(bm, quantization="Q4_K_M")
        assert bench._matches_filters(result) is True

    def test_quants_filter_excludes_mismatch(self, tmp_path: Path):
        """_matches_filters excludes result with mismatched quantization."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"quants": "Q8_0"}
        result = _make_result(bm, quantization="Q4_K_M")
        assert bench._matches_filters(result) is False

    def test_arch_filter_excludes_mismatch(self, tmp_path: Path):
        """_matches_filters excludes result with wrong architecture."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"arch": "mistral"}
        result = _make_result(bm, arch="llama")
        assert bench._matches_filters(result) is False

    def test_include_models_filter_excludes_non_matching(self, tmp_path: Path):
        """_matches_filters excludes model not matching include_models regex."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"include_models": "^other/"}
        result = _make_result(bm, model_name="pub/model")
        assert bench._matches_filters(result) is False

    def test_exclude_models_filter_excludes_matching(self, tmp_path: Path):
        """_matches_filters excludes model matching exclude_models regex."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"exclude_models": "pub/"}
        result = _make_result(bm, model_name="pub/model")
        assert bench._matches_filters(result) is False

    def test_min_context_excludes_too_small(self, tmp_path: Path):
        """_matches_filters excludes model below min_context."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"min_context": 8192}
        result = _make_result(bm)
        result.max_context_length = 4096
        assert bench._matches_filters(result) is False

    def test_max_size_excludes_too_large(self, tmp_path: Path):
        """_matches_filters excludes model larger than max_size."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.filter_args = {"max_size": 2.0}
        result = _make_result(bm)
        result.model_size_gb = 4.0
        assert bench._matches_filters(result) is False


class TestLMStudioBenchmarkDelta:
    """Tests for LMStudioBenchmark._calculate_delta()."""

    def test_returns_none_without_previous_results(self, tmp_path: Path):
        """_calculate_delta returns None when no previous results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        current = _make_result(bm)
        result = bench._calculate_delta(current)
        assert result is None

    def test_returns_delta_when_matching_previous(self, tmp_path: Path):
        """_calculate_delta returns delta dict for matching model+quant."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        prev = _make_result(bm, speed=40.0)
        current = _make_result(bm, speed=50.0)
        bench.previous_results = [prev]
        delta = bench._calculate_delta(current)
        assert delta is not None
        assert delta["speed_delta"] == pytest.approx(10.0, 0.01)
        assert delta["speed_delta_pct"] == pytest.approx(25.0, 0.01)

    def test_returns_none_when_no_match(self, tmp_path: Path):
        """_calculate_delta returns None when no matching previous result."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        prev = _make_result(bm, model_name="other/model")
        current = _make_result(bm, model_name="pub/model")
        bench.previous_results = [prev]
        result = bench._calculate_delta(current)
        assert result is None


class TestLMStudioBenchmarkBestPractices:
    """Tests for LMStudioBenchmark._generate_best_practices()."""

    def test_returns_list(self, tmp_path: Path):
        """_generate_best_practices returns a list."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [_make_result(bm, speed=50.0)]
        result = bench._generate_best_practices()
        assert isinstance(result, list)

    def test_returns_empty_without_results(self, tmp_path: Path):
        """_generate_best_practices returns empty list with no results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = []
        result = bench._generate_best_practices()
        assert isinstance(result, list)
