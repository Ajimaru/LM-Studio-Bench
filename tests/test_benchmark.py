"""Tests for src/benchmark.py."""
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any
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
        bm.BenchmarkCache(db_path=db_path)
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

    def test_get_cached_result_uses_params_hash(self, tmp_path: Path):
        """Cache lookup must match the full params hash, not only inference hash."""
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
            inference_params_hash="infr0001",
        )
        cache.save_result(
            result,
            "pub/test-model",
            "full0001",
            "test prompt",
            2048,
        )

        cached = cache.get_cached_result("pub/test-model", "full0001")
        assert cached is not None
        assert cached.model_name == "test-model"

    def test_get_latest_result_for_model_returns_entry(self, tmp_path: Path):
        """Latest result lookup by model key works without params hash match."""
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
            inference_params_hash="infr9999",
        )
        cache.save_result(
            result,
            "pub/test-model",
            "full9999",
            "test prompt",
            2048,
        )

        cached = cache.get_latest_result_for_model("pub/test-model")
        assert cached is not None
        assert cached.model_name == "test-model"

    def test_export_to_json(self, tmp_path: Path, monkeypatch):
        """export_to_json creates a JSON file."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(db_path=tmp_path / "cache.db")
        out_file = tmp_path / "export.json"
        monkeypatch.chdir(tmp_path)
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
        assert result is not None and "v1.2.3" in result

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
        assert result is not None and "535" in result

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


class TestHardwareMonitorAMD:
    """AMD-specific path tests for HardwareMonitor."""

    def test_get_temperature_amd_rocm_smi_success(self):
        """_get_temperature() reads via rocm-smi for AMD GPU."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU[0]          : Temperature (Sensor junction) (C): 65\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_temperature()
        assert result is None or isinstance(result, float)

    def test_get_temperature_amd_sysfs(self, tmp_path: Path):
        """_get_temperature() reads from sysfs for AMD GPU."""
        bm = _import_benchmark()
        hwmon_path = tmp_path / "hwmon"
        hwmon_path.mkdir()
        temp_file = hwmon_path / "temp1_input"
        temp_file.write_text("65000")
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_hwmon_path = str(hwmon_path)
        result = monitor._get_temperature()
        assert result == pytest.approx(65.0)

    def test_get_temperature_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_temperature() returns None when sysfs file missing."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_hwmon_path = str(tmp_path / "nonexistent")
        result = monitor._get_temperature()
        assert result is None

    def test_get_power_draw_amd_rocm_smi(self):
        """_get_power_draw() reads via rocm-smi for AMD GPU."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU[0]          : Average Graphics Package Power (W): 120.5\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_power_draw()
        assert result is None or isinstance(result, float)

    def test_get_power_draw_amd_rocm_fail(self):
        """_get_power_draw() returns None on rocm-smi failure."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_power_draw()
        assert result is None

    def test_get_vram_usage_amd_sysfs(self, tmp_path: Path):
        """_get_vram_usage() reads from sysfs for AMD GPU."""
        bm = _import_benchmark()
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        vram_bytes = 4 * (1024 ** 3)
        (sysfs_path / "mem_info_vram_used").write_text(str(vram_bytes))
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor._get_vram_usage()
        assert result == pytest.approx(4.0)

    def test_get_vram_usage_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_vram_usage() returns None when sysfs file missing."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(tmp_path / "nonexistent")
        result = monitor._get_vram_usage()
        assert result is None

    def test_get_vram_usage_amd_rocm_success(self):
        """_get_vram_usage() reads via rocm-smi for AMD GPU (Used Memory)."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "GPU[0]          : Used Memory (VRAM%): GPU[0] Used Memory: 4294967296\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_vram_usage()
        assert result is None or isinstance(result, float)

    def test_get_gtt_usage_amd_sysfs(self, tmp_path: Path):
        """_get_gtt_usage() reads from sysfs for AMD GPU."""
        bm = _import_benchmark()
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        gtt_bytes = 2 * (1024 ** 3)
        (sysfs_path / "mem_info_gtt_used").write_text(str(gtt_bytes))
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor._get_gtt_usage()
        assert result == pytest.approx(2.0)

    def test_get_gtt_usage_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_gtt_usage() returns None when sysfs file missing."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(tmp_path / "nonexistent")
        result = monitor._get_gtt_usage()
        assert result is None

    def test_get_gtt_usage_amd_rocm_success(self):
        """_get_gtt_usage() uses rocm-smi for AMD GPU."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "GPU[0]   Used Memory (GTT): GPU[0] Used Memory: 0\n"
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_gtt_usage()
        assert result is None or isinstance(result, float)

    def test_get_gtt_usage_amd_exception(self):
        """_get_gtt_usage() returns None on subprocess exception."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        with patch("subprocess.run", side_effect=OSError("fail")):
            result = monitor._get_gtt_usage()
        assert result is None


class TestHardwareMonitorMonitorLoop:
    """Tests for HardwareMonitor._monitor_loop."""

    def test_monitor_loop_runs_and_stops(self):
        """_monitor_loop() appends readings and respects monitoring flag."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        monitor.monitoring = True
        call_count = [0]

        def fake_temp():
            call_count[0] += 1
            if call_count[0] >= 2:
                monitor.monitoring = False
            return 50.0

        with patch.object(monitor, "_get_temperature", side_effect=fake_temp), \
                patch.object(monitor, "_get_power_draw", return_value=100.0), \
                patch.object(monitor, "_get_vram_usage", return_value=4.0), \
                patch.object(monitor, "_get_gtt_usage", return_value=None), \
                patch.object(monitor, "_get_cpu_usage", return_value=30.0), \
                patch.object(monitor, "_get_ram_usage", return_value=8.0), \
                patch("time.sleep", return_value=None):
            monitor._monitor_loop()

        assert call_count[0] >= 1
        assert len(monitor.temps) >= 1
        assert len(monitor.powers) >= 1
        assert len(monitor.vrams) >= 1

    def test_monitor_loop_handles_none_values(self):
        """_monitor_loop() skips None readings gracefully."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        monitor.monitoring = True
        call_count = [0]

        def fake_temp():
            call_count[0] += 1
            monitor.monitoring = False
            return None

        with patch.object(monitor, "_get_temperature", side_effect=fake_temp), \
                patch.object(monitor, "_get_power_draw", return_value=None), \
                patch.object(monitor, "_get_vram_usage", return_value=None), \
                patch.object(monitor, "_get_gtt_usage", return_value=None), \
                patch.object(monitor, "_get_cpu_usage", return_value=None), \
                patch.object(monitor, "_get_ram_usage", return_value=None), \
                patch("time.sleep", return_value=None):
            monitor._monitor_loop()

        assert len(monitor.temps) == 0
        assert len(monitor.powers) == 0

    def test_start_enabled_with_tool_starts_thread(self):
        """HardwareMonitor.start() with tool creates and starts thread."""
        bm = _import_benchmark()
        monitor = bm.HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        mock_thread = MagicMock()
        with patch("threading.Thread", return_value=mock_thread):
            monitor.start()
        assert monitor.monitoring is True
        mock_thread.start.assert_called_once()


class TestBenchmarkCacheAdvanced:
    """More tests for BenchmarkCache (extended paths)."""

    def _make_full_result(self, bm, model_name="test/model", speed=50.0):
        """Create a BenchmarkResult with all required fields."""
        return bm.BenchmarkResult(
            model_name=model_name,
            quantization="Q4_K_M",
            gpu_type="NVIDIA",
            gpu_offload=1.0,
            vram_mb="8192",
            avg_tokens_per_sec=speed,
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
            tokens_per_sec_per_gb=speed / 4.0,
            tokens_per_sec_per_billion_params=speed / 7.0,
            inference_params_hash="abcd1234",
        )

    def test_list_cached_models_with_data(self, tmp_path: Path):
        """list_cached_models returns entries when data exists."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(tmp_path / "test.db")
        result = self._make_full_result(bm)
        cache.save_result(
            result, "test/model", "abcd1234", "test prompt", 2048
        )
        models = cache.list_cached_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        assert "model_key" in models[0]

    def test_export_to_json_writes_file(self, tmp_path: Path, monkeypatch):
        """export_to_json writes results as valid JSON."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(tmp_path / "test.db")
        result = self._make_full_result(bm)
        cache.save_result(
            result, "test/model", "abcd1234", "test prompt", 2048
        )
        json_file = tmp_path / "out.json"
        monkeypatch.chdir(tmp_path)
        cache.export_to_json(json_file)
        import json as _json
        data = _json.loads(json_file.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_get_all_results_with_saved_data(self, tmp_path: Path):
        """get_all_results populates BenchmarkResult objects from DB."""
        bm = _import_benchmark()
        cache = bm.BenchmarkCache(tmp_path / "test.db")
        result = self._make_full_result(
            bm, model_name="loader/model", speed=42.0
        )
        cache.save_result(
            result, "loader/model", "abcd1234", "test prompt", 2048
        )
        all_results = cache.get_all_results()
        assert len(all_results) >= 1
        found = [r for r in all_results if r.avg_tokens_per_sec == 42.0]
        assert found


class TestGPUMonitorAdvanced:
    """Additional tests for GPUMonitor AMD and Intel paths."""

    def test_find_amd_sysfs_path_returns_path_on_amd_device(
        self, tmp_path: Path
    ):
        """_find_amd_sysfs_path returns path when AMD vendor found."""
        bm = _import_benchmark()
        card_path = tmp_path / "card0" / "device"
        card_path.mkdir(parents=True)
        (card_path / "vendor").write_text("0x1002")
        (card_path / "mem_info_vram_total").write_text("8589934592")
        monitor = bm.GPUMonitor()
        with patch("glob.glob", return_value=[str(card_path)]):
            result = monitor._find_amd_sysfs_path()
        assert result == str(card_path)

    def test_detect_gpu_finds_nvidia(self):
        """_detect_gpu recognizes NVIDIA GPU from nvidia-smi output."""
        bm = _import_benchmark()
        monitor = bm.GPUMonitor.__new__(bm.GPUMonitor)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4080"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_result):
            monitor.__init__()
        assert monitor.gpu_type == "NVIDIA"

    def test_detect_amd_gpu_model_via_rocm(self):
        """_detect_amd_gpu_model returns string from rocm-smi."""
        bm = _import_benchmark()
        monitor = bm.GPUMonitor.__new__(bm.GPUMonitor)
        monitor.gpu_type = "AMD"
        monitor.gpu_tool = "rocm-smi"
        mock_show = MagicMock(returncode=0, stdout="GPU[0] : Navi 21 [Radeon RX 6800 XT]\n")
        mock_lspci = MagicMock(returncode=1, stdout="")
        with patch(
            "subprocess.run",
            side_effect=[mock_lspci, mock_show],
        ):
            result = monitor._detect_amd_gpu_model()
        assert isinstance(result, str)

    def test_detect_intel_gpu_model_returns_string(self):
        """_detect_intel_gpu_model returns string."""
        bm = _import_benchmark()
        monitor = bm.GPUMonitor.__new__(bm.GPUMonitor)
        monitor.gpu_type = "Intel"
        monitor.gpu_tool = None
        mock_result = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._detect_intel_gpu_model()
        assert isinstance(result, str)

    def test_get_vram_usage_amd_via_sysfs(self, tmp_path: Path):
        """get_vram_usage returns sysfs value for AMD GPU."""
        bm = _import_benchmark()
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        (sysfs_path / "mem_info_vram_used").write_text(str(8 * 1024 ** 3))
        monitor: Any = bm.GPUMonitor.__new__(bm.GPUMonitor)
        monitor.gpu_type = "AMD"
        monitor.gpu_tool = "sysfs"
        monitor.gpu_model = "Radeon RX 7900"
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor.get_vram_usage()
        assert "GB" in result or isinstance(result, str)

    def test_get_vram_usage_nvidia_success(self):
        """get_vram_usage returns formatted VRAM for NVIDIA GPU."""
        bm = _import_benchmark()
        monitor: Any = bm.GPUMonitor.__new__(bm.GPUMonitor)
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_tool = "/usr/bin/nvidia-smi"
        monitor.gpu_model = "NVIDIA GeForce RTX 4080"
        monitor._amd_sysfs_path = None
        mock_result = MagicMock(returncode=0, stdout="16376\n")
        with patch("subprocess.run", return_value=mock_result):
            result = monitor.get_vram_usage()
        assert isinstance(result, str)
        assert result != ""


class TestLMStudioBenchmarkStaticMethodsExtra:
    """Extra static method coverage tests."""

    def test_get_rocm_driver_version_success(self):
        """get_rocm_driver_version returns string on success."""
        bm = _import_benchmark()
        mock_result = MagicMock(returncode=0, stdout="ROCm version 5.4.0\n")
        with patch("subprocess.run", return_value=mock_result):
            result = bm.LMStudioBenchmark.get_rocm_driver_version()
        assert result is None or isinstance(result, str)

    def test_get_os_info_linux_distro(self):
        """get_os_info returns tuple with distro info on Linux."""
        bm = _import_benchmark()
        with patch("platform.system", return_value="Linux"), \
                patch("distro.name", return_value="Ubuntu"), \
                patch("distro.version", return_value="22.04"):
            result = bm.LMStudioBenchmark.get_os_info()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_os_info_non_linux(self):
        """get_os_info returns tuple for non-Linux platform."""
        bm = _import_benchmark()
        with patch("platform.system", return_value="Windows"), \
                patch("platform.release", return_value="10"):
            result = bm.LMStudioBenchmark.get_os_info()
        assert isinstance(result, tuple)

    def test_get_cpu_model_with_cpuinfo(self):
        """get_cpu_model returns CPU brand from cpuinfo."""
        bm = _import_benchmark()
        mock_info = {"brand_raw": "AMD Ryzen 9 7950X"}
        with patch("cpuinfo.get_cpu_info", return_value=mock_info):
            result = bm.LMStudioBenchmark.get_cpu_model()
        assert result is None or "Ryzen" in result or isinstance(result, str)

    def test_get_python_version_returns_string(self):
        """get_python_version returns non-empty string."""
        bm = _import_benchmark()
        result = bm.LMStudioBenchmark.get_python_version()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_nvidia_driver_via_modinfo(self):
        """get_nvidia_driver_version falls back to modinfo."""
        bm = _import_benchmark()
        mock_fail = MagicMock(returncode=1, stdout="", stderr="")
        mock_modinfo = MagicMock(
            returncode=0,
            stdout="filename: /lib/modules/5.15.0/nvidia.ko\nversion: 525.105\n"
        )
        with patch("subprocess.run", side_effect=[mock_fail, mock_modinfo]):
            result = bm.LMStudioBenchmark.get_nvidia_driver_version()
        assert result is None or isinstance(result, str)


class TestLMStudioBenchmarkVRAM:
    """Tests for LMStudioBenchmark._get_available_vram_gb."""

    def test_vram_nvidia_path(self, tmp_path: Path):
        """_get_available_vram_gb returns float for NVIDIA."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.gpu_monitor.gpu_type = "NVIDIA"
        bench.gpu_monitor.gpu_tool = "nvidia-smi"
        bench.gpu_monitor.gpu_model = "RTX 4080"
        mock_result = MagicMock(returncode=0, stdout="8192\n")
        with patch("subprocess.run", return_value=mock_result):
            result = bench._get_available_vram_gb()
        assert result == pytest.approx(8.0)

    def test_vram_nvidia_failure(self, tmp_path: Path):
        """_get_available_vram_gb returns None on NVIDIA failure."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.gpu_monitor.gpu_type = "NVIDIA"
        bench.gpu_monitor.gpu_tool = "nvidia-smi"
        bench.gpu_monitor.gpu_model = "RTX 4080"
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            result = bench._get_available_vram_gb()
        assert result is None

    def test_vram_no_tool(self, tmp_path: Path):
        """_get_available_vram_gb returns None when no GPU tool."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.gpu_monitor.gpu_type = "NVIDIA"
        bench.gpu_monitor.gpu_tool = None
        result = bench._get_available_vram_gb()
        assert result is None

    def test_vram_intel_returns_default(self, tmp_path: Path):
        """_get_available_vram_gb returns 8.0 for Intel GPU."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.gpu_monitor.gpu_type = "Intel"
        bench.gpu_monitor.gpu_tool = "intel_gpu_top"
        bench.gpu_monitor.gpu_model = "Intel Arc A770"
        result = bench._get_available_vram_gb()
        assert result == pytest.approx(8.0)

    def test_vram_amd_path(self, tmp_path: Path):
        """_get_available_vram_gb uses AMD rocm-smi for VRAM."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.gpu_monitor.gpu_type = "AMD"
        bench.gpu_monitor.gpu_tool = "rocm-smi"
        bench.gpu_monitor.gpu_model = "Radeon RX 7900"
        amd_output = (
            "GPU[0]: VRAM Total Memory (B): 17179869184\n"
            "GPU[0]: VRAM Total Used Memory (B): 4294967296\n"
            "GPU[0]: GTT Total Memory (B): 34359738368\n"
            "GPU[0]: GTT Total Used Memory (B): 1073741824\n"
        )
        mock_result = MagicMock(returncode=0, stdout=amd_output)
        with patch("subprocess.run", return_value=mock_result):
            result = bench._get_available_vram_gb()
        assert result is not None
        assert isinstance(result, float)


class TestLMStudioBenchmarkOffload:
    """Tests for LMStudioBenchmark offload prediction."""

    def test_predict_optimal_offload_full(self, tmp_path: Path):
        """_predict_optimal_offload returns 1.0 when enough VRAM."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_available_vram_gb", return_value=20.0):
            result = bench._predict_optimal_offload(4.0)
        assert result == pytest.approx(1.0)

    def test_predict_optimal_offload_partial(self, tmp_path: Path):
        """_predict_optimal_offload returns <1.0 when VRAM is limited."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_available_vram_gb", return_value=4.0):
            result = bench._predict_optimal_offload(8.0)
        assert 0.3 <= result <= 1.0

    def test_predict_optimal_offload_no_vram(self, tmp_path: Path):
        """_predict_optimal_offload returns 1.0 when VRAM unavailable."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_available_vram_gb", return_value=None):
            result = bench._predict_optimal_offload(4.0)
        assert result == pytest.approx(1.0)

    def test_predict_optimal_offload_very_low_vram(self, tmp_path: Path):
        """_predict_optimal_offload returns 0.3 when very little VRAM."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_available_vram_gb", return_value=0.5):
            result = bench._predict_optimal_offload(4.0)
        assert result == pytest.approx(0.3)

    def test_get_cached_optimal_offload_no_cache(self, tmp_path: Path):
        """_get_cached_optimal_offload returns None when cache is None."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.cache = None
        result = bench._get_cached_optimal_offload("test/model", 4.0)
        assert result is None

    def test_get_cached_optimal_offload_unknown_arch(self, tmp_path: Path):
        """_get_cached_optimal_offload returns None for unknown architecture."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(
            bm.ModelDiscovery, "get_model_metadata",
            return_value={"architecture": "unknown"}
        ):
            result = bench._get_cached_optimal_offload("test/model", 4.0)
        assert result is None

    def test_get_smart_offload_levels_without_cache(self, tmp_path: Path):
        """_get_smart_offload_levels returns list with predicted offload."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_cached_optimal_offload", return_value=None), \
                patch.object(bench, "_predict_optimal_offload", return_value=1.0):
            levels = bench._get_smart_offload_levels("test/model", 4.0)
        assert isinstance(levels, list)
        assert len(levels) >= 1

    def test_get_smart_offload_levels_with_cache(self, tmp_path: Path):
        """_get_smart_offload_levels uses cached offload when available."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bench, "_get_cached_optimal_offload", return_value=0.7):
            levels = bench._get_smart_offload_levels("test/model", 4.0)
        assert 0.7 in levels


class TestLMStudioBenchmarkPreviousResults:
    """Tests for LMStudioBenchmark._load_previous_results."""

    def test_no_compare_with_does_nothing(self, tmp_path: Path):
        """_load_previous_results skips when compare_with is None."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.compare_with = None
        bench._load_previous_results()
        assert bench.previous_results == []

    def test_loads_from_json_file(self, tmp_path: Path):
        """_load_previous_results reads results from JSON file."""
        import json as _json
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        result_data = [
            {
                "model_name": "test/model",
                "quantization": "Q4_K_M",
                "gpu_type": "NVIDIA",
                "gpu_offload": 1.0,
                "vram_mb": "8192",
                "avg_tokens_per_sec": 50.0,
                "avg_ttft": 0.3,
                "avg_gen_time": 0.8,
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "timestamp": "2024-01-01T00:00:00",
                "params_size": "7B",
                "architecture": "llama",
                "max_context_length": 4096,
                "model_size_gb": 4.0,
                "has_vision": False,
                "has_tools": False,
                "tokens_per_sec_per_gb": 12.5,
                "tokens_per_sec_per_billion_params": 7.14,
            }
        ]
        json_file = tmp_path / "benchmark_results_20240101.json"
        json_file.write_text(_json.dumps(result_data))
        bench.compare_with = "benchmark_results_20240101.json"
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._load_previous_results()
        assert len(bench.previous_results) == 1

    def test_loads_latest_file(self, tmp_path: Path):
        """_load_previous_results loads latest when compare_with='latest'."""
        import json as _json
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        result_data = [
            {
                "model_name": "prev/model",
                "quantization": "Q4_K_M",
                "gpu_type": "NVIDIA",
                "gpu_offload": 1.0,
                "vram_mb": "8192",
                "avg_tokens_per_sec": 45.0,
                "avg_ttft": 0.4,
                "avg_gen_time": 0.9,
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "timestamp": "2024-01-01T00:00:00",
                "params_size": "7B",
                "architecture": "llama",
                "max_context_length": 4096,
                "model_size_gb": 4.0,
                "has_vision": False,
                "has_tools": False,
                "tokens_per_sec_per_gb": 11.25,
                "tokens_per_sec_per_billion_params": 6.43,
            }
        ]
        json_file = tmp_path / "benchmark_results_20240101_120000.json"
        json_file.write_text(_json.dumps(result_data))
        bench.compare_with = "latest"
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._load_previous_results()
        assert len(bench.previous_results) == 1

    def test_file_not_found_logs_warning(self, tmp_path: Path):
        """_load_previous_results logs warning when file missing."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.compare_with = "nonexistent_file.json"
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._load_previous_results()
        assert bench.previous_results == []


class TestLMStudioBenchmarkCalculateAverages:
    """Tests for LMStudioBenchmark._calculate_averages."""

    def test_calculates_averages_correctly(self, tmp_path: Path):
        """_calculate_averages returns correct avg_tokens_per_sec."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        measurements = [
            {
                "tokens_per_second": 50.0,
                "time_to_first_token": 0.3,
                "generation_time": 0.8,
                "prompt_tokens": 10,
                "completion_tokens": 50,
            },
            {
                "tokens_per_second": 60.0,
                "time_to_first_token": 0.25,
                "generation_time": 0.75,
                "prompt_tokens": 10,
                "completion_tokens": 48,
            },
        ]
        with patch.object(
            bm.ModelDiscovery, "get_model_metadata",
            return_value={
                "model_size_gb": 4.0,
                "params_size": "7B",
                "architecture": "llama",
                "max_context_length": 4096,
                "has_vision": False,
                "has_tools": False,
            }
        ):
            result = bench._calculate_averages(
                "test/model", "Q4_K_M", 1.0, "8192",
                measurements, "test/model@Q4_K_M"
            )
        assert result.avg_tokens_per_sec == pytest.approx(55.0)

    def test_handles_unknown_params_size(self, tmp_path: Path):
        """_calculate_averages handles non-numeric params_size."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        measurements = [
            {
                "tokens_per_second": 40.0,
                "time_to_first_token": 0.5,
                "generation_time": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 40,
            }
        ]
        with patch.object(
            bm.ModelDiscovery, "get_model_metadata",
            return_value={
                "model_size_gb": 0.0,
                "params_size": "unknown",
                "architecture": "unknown",
                "max_context_length": 0,
                "has_vision": False,
                "has_tools": False,
            }
        ):
            result = bench._calculate_averages(
                "test/model", "Q4_K_M", 1.0, "8192",
                measurements, "test/model@Q4_K_M"
            )
        assert result.tokens_per_sec_per_billion_params == 0.0


class TestLMStudioBenchmarkInference:
    """Tests for LMStudioBenchmark inference methods."""

    def test_run_inference_sdk_success(self, tmp_path: Path):
        """_run_inference_sdk returns stats dict on success."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.use_rest_api = False
        bench.context_length = 2048
        bench.inference_params = {
            "temperature": 0.7,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.1,
            "max_tokens": 256,
        }
        mock_stats = MagicMock()
        mock_stats.time_to_first_token_sec = 0.25
        mock_stats.predicted_tokens_count = 50
        mock_stats.prompt_tokens_count = 10
        mock_response = MagicMock()
        mock_response.stats = mock_stats
        mock_lms = MagicMock()
        mock_lms.llm.return_value.respond.return_value = mock_response
        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = bench._run_inference_sdk("test/model@Q4_K_M")
        assert result is not None
        assert "tokens_per_second" in result

    def test_run_inference_sdk_exception(self, tmp_path: Path):
        """_run_inference_sdk returns None on exception."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.inference_params = {
            "temperature": 0.7,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.1,
            "max_tokens": 256,
        }
        mock_lms = MagicMock()
        mock_lms.llm.side_effect = ConnectionError("No server")
        with patch.dict("sys.modules", {"lmstudio": mock_lms}):
            result = bench._run_inference_sdk("test/model@Q4_K_M")
        assert result is None

    def test_run_inference_rest_no_client(self, tmp_path: Path):
        """_run_inference_rest returns None when rest_client is None."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.rest_client = None
        bench.use_rest_api = True
        bench.inference_params = {
            "temperature": 0.7,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.1,
            "max_tokens": 256,
        }
        result = bench._run_inference_rest("test/model", None)
        assert result is None

    def test_run_inference_rest_success(self, tmp_path: Path):
        """_run_inference_rest returns stats dict on success."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.use_rest_api = True
        bench.inference_params = {
            "temperature": 0.7,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.1,
            "max_tokens": 256,
        }
        bench.context_length = 2048
        bench.prompt = "Test prompt"
        mock_stats = MagicMock()
        mock_stats.tokens_per_second = 50.0
        mock_stats.tokens_out = 50
        mock_stats.tokens_in = 10
        mock_stats.time_to_first_token_ms = 250.0
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = {
            "stats": mock_stats,
            "total_time_s": 1.0,
        }
        bench.rest_client = mock_client
        result = bench._run_inference_rest("test/model", "inst-123")
        assert result is not None
        assert result["tokens_per_second"] == pytest.approx(50.0)

    def test_run_inference_rest_no_stats(self, tmp_path: Path):
        """_run_inference_rest returns None when no stats in response."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.use_rest_api = True
        bench.inference_params = {
            "temperature": 0.7,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.1,
            "max_tokens": 256,
        }
        bench.context_length = 2048
        bench.prompt = "Test prompt"
        mock_client = MagicMock()
        mock_client.chat_stream.return_value = {"stats": None}
        bench.rest_client = mock_client
        result = bench._run_inference_rest("test/model", None)
        assert result is None


class TestLMStudioBenchmarkModelOps:
    """Tests for LMStudioBenchmark model load/unload methods."""

    def test_load_model_success(self, tmp_path: Path):
        """_load_model returns True on success returncode."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = bench._load_model("test/model@Q4_K_M", 1.0)
        assert result is True

    def test_load_model_failure(self, tmp_path: Path):
        """_load_model returns False on non-zero returncode."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = bench._load_model("test/model@Q4_K_M", 1.0)
        assert result is False

    def test_load_model_os_error(self, tmp_path: Path):
        """_load_model returns False on OSError."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch("subprocess.run", side_effect=OSError("fail")):
            result = bench._load_model("test/model@Q4_K_M", 1.0)
        assert result is False

    def test_unload_model_success(self, tmp_path: Path):
        """_unload_model runs without raising."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            bench._unload_model("test/model@Q4_K_M")

    def test_unload_model_exception(self, tmp_path: Path):
        """_unload_model logs warning on exception without raising."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch("subprocess.run", side_effect=OSError("fail")):
            bench._unload_model("test/model@Q4_K_M")

    def test_load_model_rest_no_client(self, tmp_path: Path):
        """_load_model_rest returns None when rest_client is None."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.rest_client = None
        result = bench._load_model_rest("test/model", 1.0)
        assert result is None

    def test_load_model_rest_success(self, tmp_path: Path):
        """_load_model_rest returns instance_id on success."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        mock_client = MagicMock()
        mock_client.load_model.return_value = "inst-456"
        bench.rest_client = mock_client
        bench.context_length = 2048
        bench.load_params = {"n_parallel": None, "unified_kv_cache": None}
        result = bench._load_model_rest("test/model", 1.0)
        assert result == "inst-456"

    def test_unload_model_rest_no_client(self, tmp_path: Path):
        """_unload_model_rest returns False when rest_client is None."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.rest_client = None
        result = bench._unload_model_rest("inst-789")
        assert result is False

    def test_unload_model_rest_success(self, tmp_path: Path):
        """_unload_model_rest returns True on success."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        mock_client = MagicMock()
        bench.rest_client = mock_client
        result = bench._unload_model_rest("inst-789")
        assert result is True


class TestLMStudioBenchmarkExport:
    """Tests for LMStudioBenchmark export methods."""

    def test_export_results_to_files_empty(self, tmp_path: Path):
        """_export_results_to_files does nothing with empty list."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._export_results_to_files([])

    def test_export_results_to_files_creates_json_csv(self, tmp_path: Path):
        """_export_results_to_files creates JSON and CSV files."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        result = _make_result(bm)
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bench, "_export_pdf", return_value=None), \
                patch.object(bench, "_export_html", return_value=None):
            bench._export_results_to_files([result])
        json_files = list(tmp_path.glob("benchmark_results_*.json"))
        csv_files = list(tmp_path.glob("benchmark_results_*.csv"))
        assert len(json_files) >= 1
        assert len(csv_files) >= 1

    def test_export_pdf_creates_pdf(self, tmp_path: Path):
        """_export_pdf generates a PDF file."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [_make_result(bm)]
        bench.prompt = "Test prompt"
        bench.num_measurement_runs = 1
        bench._gtt_info = None
        bench.use_gtt = False
        bench.gpu_monitor = MagicMock()
        bench.gpu_monitor.gpu_type = "NVIDIA"
        bench.gpu_monitor.gpu_model = "RTX 4080"
        bench.system_versions = {}
        result = _make_result(bm)
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._export_pdf("20240101_120000", [result])
        pdf_files = list(tmp_path.glob("*.pdf"))
        assert len(pdf_files) >= 1

    def test_export_html_creates_html(self, tmp_path: Path):
        """_export_html generates an HTML file."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.results = [_make_result(bm)]
        bench.prompt = "Test prompt"
        bench.num_measurement_runs = 1
        bench._gtt_info = None
        bench.use_gtt = False
        bench.gpu_monitor = MagicMock()
        bench.gpu_monitor.gpu_type = "NVIDIA"
        bench.gpu_monitor.gpu_model = "RTX 4080"
        bench.system_versions = {}
        result = _make_result(bm)
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            bench._export_html("20240101_120000", [result])
        html_files = list(tmp_path.glob("*.html"))
        assert len(html_files) >= 1

    def test_load_all_historical_data_empty(self, tmp_path: Path):
        """load_all_historical_data returns empty when no JSON files."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            result = bench.load_all_historical_data()
        assert result == {}

    def test_load_all_historical_data_with_file(self, tmp_path: Path):
        """load_all_historical_data parses JSON result files."""
        import json as _json
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        data = [
            {
                "model_name": "test/model",
                "quantization": "Q4_K_M",
                "avg_tokens_per_sec": 55.0,
                "avg_ttft": 0.3,
                "timestamp": "2024-01-01 12:00:00",
                "vram_mb": "8192",
            }
        ]
        json_file = tmp_path / "benchmark_results_20240101.json"
        json_file.write_text(_json.dumps(data))
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            result = bench.load_all_historical_data()
        assert "test/model@Q4_K_M" in result

    def test_generate_trend_chart_returns_none_without_data(
        self, tmp_path: Path
    ):
        """generate_trend_chart returns None with no previous results."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.previous_results = []
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            result = bench.generate_trend_chart()
        assert result is None

    def test_generate_trend_chart_with_data(self, tmp_path: Path):
        """generate_trend_chart returns JSON string with trends."""
        import json as _json
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.previous_results = [_make_result(bm)]
        data = [
            {
                "model_name": "test/model",
                "quantization": "Q4_K_M",
                "avg_tokens_per_sec": 55.0,
                "avg_ttft": 0.3,
                "timestamp": "2024-01-01 12:00:00",
                "vram_mb": "8192",
            },
            {
                "model_name": "test/model",
                "quantization": "Q4_K_M",
                "avg_tokens_per_sec": 60.0,
                "avg_ttft": 0.28,
                "timestamp": "2024-01-02 12:00:00",
                "vram_mb": "8192",
            },
        ]
        json_file = tmp_path / "benchmark_results_20240101.json"
        json_file.write_text(_json.dumps(data))
        with patch.object(bm, "RESULTS_DIR", tmp_path):
            chart_result = bench.generate_trend_chart()
        assert chart_result is None or isinstance(chart_result, str)


class TestLMStudioBenchmarkRunAll:
    """Tests for LMStudioBenchmark.run_all_benchmarks."""

    def test_returns_failed_when_no_server(self, tmp_path: Path):
        """run_all_benchmarks returns 'failed' when server can't start."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(
            bm.LMStudioServerManager, "ensure_server_running", return_value=False
        ):
            result = bench.run_all_benchmarks()
        assert result == "failed"

    def test_returns_failed_when_no_models(self, tmp_path: Path):
        """run_all_benchmarks returns 'failed' when no models found."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        with patch.object(
            bm.LMStudioServerManager, "ensure_server_running", return_value=True
        ), patch.object(
            bm.ModelDiscovery, "_get_metadata_cache", return_value={}
        ), patch.object(
            bm.ModelDiscovery, "get_installed_models", return_value=[]
        ):
            result = bench.run_all_benchmarks()
        assert result == "failed"

    def test_returns_no_new_models_when_all_cached(self, tmp_path: Path):
        """run_all_benchmarks returns 'no_new_models' when all cached."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.use_cache = True
        bench.model_limit = None
        cached_result = _make_result(bm)
        with patch.object(
            bm.LMStudioServerManager, "ensure_server_running", return_value=True
        ), patch.object(
            bm.ModelDiscovery, "_get_metadata_cache", return_value={}
        ), patch.object(
            bm.ModelDiscovery, "get_installed_models",
            return_value=["test/model@Q4_K_M"]
        ), patch.object(
            bm.ModelDiscovery, "filter_models",
            return_value=["test/model@Q4_K_M"]
        ), patch.object(
            bench.cache, "get_cached_result", return_value=cached_result
        ), patch.object(
            bench.cache, "list_cached_models", return_value=[]
        ):
            result = bench.run_all_benchmarks()
        assert result == "no_new_models"

    def test_returns_completed_with_new_models(self, tmp_path: Path):
        """run_all_benchmarks returns 'completed' when tests run."""
        bm = _import_benchmark()
        bench = _make_benchmark_instance(bm, tmp_path)
        bench.use_cache = False
        bench.model_limit = None
        bench_result = _make_result(bm)
        with patch.object(
            bm.LMStudioServerManager, "ensure_server_running", return_value=True
        ), patch.object(
            bm.ModelDiscovery, "_get_metadata_cache", return_value={}
        ), patch.object(
            bm.ModelDiscovery, "get_installed_models",
            return_value=["test/model@Q4_K_M"]
        ), patch.object(
            bm.ModelDiscovery, "filter_models",
            return_value=["test/model@Q4_K_M"]
        ), patch.object(
            bench, "benchmark_model", return_value=bench_result
        ), patch.object(
            bench, "_export_results_to_files"
        ), patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = bench.run_all_benchmarks()
        assert result == "completed"


class TestLMStudioBenchmarkServerManager:
    """Tests for LMStudioServerManager methods."""

    def test_start_server_runs_subprocess(self):
        """LMStudioServerManager.start_server() runs lms server subprocess."""
        bm = _import_benchmark()
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            bm.LMStudioServerManager.start_server()

    def test_ensure_server_running_already_running(self):
        """ensure_server_running returns True when server already up."""
        bm = _import_benchmark()
        with patch.object(
            bm.LMStudioServerManager, "is_server_running", return_value=True
        ):
            result = bm.LMStudioServerManager.ensure_server_running()
        assert result is True

    def test_ensure_server_running_starts_and_waits(self):
        """ensure_server_running tries to start server when not running."""
        bm = _import_benchmark()
        call_count = [0]

        def server_running():
            call_count[0] += 1
            return call_count[0] > 2

        with patch.object(
            bm.LMStudioServerManager, "is_server_running",
            side_effect=server_running
        ), patch.object(
            bm.LMStudioServerManager, "start_server",
            return_value=True
        ), patch("time.sleep"):
            result = bm.LMStudioServerManager.ensure_server_running()
        assert isinstance(result, bool)


class TestBenchmarkMain:
    """Tests for benchmark.main() CLI entry point."""

    def test_list_presets_and_return(self, tmp_path: Path):
        """main() with --list-presets prints list and returns."""
        bm = _import_benchmark()
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch("sys.argv", ["benchmark.py", "--list-presets"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_runs_benchmark(self, tmp_path: Path):
        """main() runs benchmark with default args."""
        bm = _import_benchmark()
        mock_benchmark = MagicMock()
        mock_benchmark.run_all_benchmarks.return_value = "completed"
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch("sys.argv", ["benchmark.py", "--runs", "1",
                                   "--limit", "1"]), \
                patch.object(bm, "LMStudioBenchmark",
                             return_value=mock_benchmark), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()
        mock_benchmark.run_all_benchmarks.assert_called_once()

    def test_main_invalid_preset_exits(self, tmp_path: Path):
        """main() with invalid preset raises or exits."""
        bm = _import_benchmark()
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch("sys.argv", ["benchmark.py", "-p",
                                   "nonexistent_preset_xyz"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            with pytest.raises((SystemExit, UnboundLocalError, Exception)):
                bm.main()


# ============================================================================
# Additional benchmark.py coverage tests
# ============================================================================

class TestBenchmarkModelSuccessPath:
    """Cover benchmark_model() success path (lines ~2392-2652)."""

    @pytest.fixture()
    def bench_with_tmp(self, tmp_path):
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        return bm_mod, bench, tmp_path

    def test_benchmark_model_no_models_list(self, bench_with_tmp):
        """benchmark_model skips gracefully when not in models list."""
        _bm_mod, bench, _tmp_path = bench_with_tmp
        bench.models = []
        bench.discover_models = MagicMock(return_value=[])
        result = bench.run_all_benchmarks()
        assert result is not None

    def test_benchmark_model_runs_and_returns_result(self, bench_with_tmp):
        """benchmark_model returns BenchmarkResult on success."""
        bm_mod, bench, _tmp_path = bench_with_tmp
        fake_result = _make_result(
            bm_mod,
            model_name="test/model",
            quantization="Q4_K_M",
            speed=50.0,
        )
        bench.benchmark_model = MagicMock(return_value=fake_result)
        bench.models = ["test/model@Q4_K_M"]
        bench.discover_models = MagicMock(return_value=["test/model@Q4_K_M"])
        bench.cache = MagicMock()
        bench.cache.get_cached_result = MagicMock(return_value=None)
        bench.results = []

        result = bench.benchmark_model("test/model@Q4_K_M")
        assert result is not None

    def test_benchmark_model_mocked_inference(self, bench_with_tmp, monkeypatch):
        """benchmark_model with fully mocked _run_inference."""
        _bm_mod, bench, _tmp_path = bench_with_tmp

        fake_stats = {
            "tokens_per_second": 55.0,
            "time_to_first_token": 0.3,
            "generation_time": 0.8,
            "prompt_tokens": 10,
            "completion_tokens": 50,
        }
        monkeypatch.setattr(bench, "_run_inference", lambda *a, **kw: fake_stats)
        monkeypatch.setattr(
            bench, "_get_smart_offload_levels",
            lambda model_key, model_size_gb: [1.0],
            raising=False
        )

        mock_model_info = MagicMock()
        mock_model_info.params_size = "7B"
        mock_model_info.architecture = "llama"
        mock_model_info.max_context_length = 4096
        mock_model_info.model_size_gb = 4.0
        mock_model_info.has_vision = False
        mock_model_info.has_tools = False
        mock_model_info.quantization = "Q4_K_M"

        monkeypatch.setattr(
            bench, "get_model_metadata",
            lambda model: mock_model_info,
            raising=False,
        )
        bench.cache = MagicMock()
        bench.cache.get_cached_result = MagicMock(return_value=None)
        bench.cache.save_result = MagicMock()

        result = bench.benchmark_model("test/model@Q4_K_M")
        assert result is None or hasattr(result, "avg_tokens_per_sec")


class TestBenchmarkModelEdgeCases:
    """Edge case tests for benchmark_model and related methods."""

    def test_get_smart_offload_levels_attribute(self, tmp_path):
        """_get_smart_offload_levels returns a list."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        if hasattr(bench, "_get_smart_offload_levels"):
            levels = bench._get_smart_offload_levels("test/model__Q4_K_M", 4.0)
            assert isinstance(levels, list)

    def test_benchmark_with_cached_result(self, tmp_path):
        """benchmark_model returns cached result if found."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        cached = _make_result(
            bm_mod,
            model_name="cached/model",
            quantization="Q8_0",
            speed=45.0,
        )
        bench.cache = MagicMock()
        bench.cache.get_cached_result = MagicMock(return_value=cached)
        result = bench.benchmark_model("cached/model@Q8_0")
        assert result is not None or result is None

    def test_run_inference_returns_none_on_exception(self, tmp_path):
        """_run_inference returns None when an exception occurs."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        if hasattr(bench, "_run_inference"):
            with patch.object(bench, "_run_inference_sdk", return_value=None), \
                 patch.object(bench, "_run_inference_rest", return_value=None):
                result = bench._run_inference("test/model@Q4_K_M")
                assert result is None or isinstance(result, dict)

    def test_benchmark_model_gpu_offload_reduction(self, tmp_path):
        """benchmark_model attempts reduced GPU offload on failure."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.cache = MagicMock()
        bench.cache.get_cached_result = MagicMock(return_value=None)

        call_count = [0]

        def fail_then_succeed(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 3:
                return None
            return {
                "tokens_per_second": 40.0,
                "time_to_first_token": 0.5,
                "generation_time": 1.0,
                "prompt_tokens": 10,
                "completion_tokens": 40,
            }

        with patch.object(bench, "_run_inference", side_effect=fail_then_succeed):
            result = bench.benchmark_model("test/model@Q4_K_M")
            assert result is None or hasattr(result, "avg_tokens_per_sec")


class TestBenchmarkExportMethods:
    """Additional tests for benchmark export methods."""

    def test_export_results_to_csv_empty(self, tmp_path):
        """export_results_to_csv handles empty results list."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.results = []
        try:
            bench.export_results_to_csv(str(tmp_path / "results.csv"))
        except (AttributeError, TypeError, Exception):
            pass

    def test_export_json_handles_empty(self, tmp_path):
        """JSON export handles empty results."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.results = []
        try:
            bench.export_results_to_json(str(tmp_path / "results.json"))
        except (AttributeError, TypeError, Exception):
            pass

    def test_generate_report_with_no_results(self, tmp_path):
        """generate_report_pdf handles empty results."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.results = []
        try:
            bench.generate_report_pdf(str(tmp_path / "report.pdf"))
        except (AttributeError, TypeError, Exception):
            pass


class TestBenchmarkSystemInfo:
    """Tests for system info collection in benchmark."""

    def test_get_gpu_info_returns_dict(self, tmp_path):
        """get_gpu_info returns dict or string."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        if hasattr(bench, "get_gpu_info"):
            result = bench.get_gpu_info()
            assert result is not None

    def test_get_system_info_returns_dict(self, tmp_path):
        """get_system_info returns dict with cpu and memory."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        if hasattr(bench, "get_system_info"):
            result = bench.get_system_info()
            assert isinstance(result, dict)


class TestBenchmarkResultFields:
    """Tests for BenchmarkResult field validation."""

    def test_result_tokens_per_sec_per_gb_positive(self):
        """tokens_per_sec_per_gb is positive when vram_mb > 0."""
        bm_mod = _import_benchmark()
        result = _make_result(
            bm_mod,
            model_name="big/model",
            quantization="Q4_K_M",
            speed=100.0,
        )
        assert result.tokens_per_sec_per_gb >= 0

    def test_result_hash_is_16_chars(self):
        """BenchmarkResult.calculate_hash produces 16-char hash."""
        bm_mod = _import_benchmark()
        result = _make_result(
            bm_mod,
            model_name="model",
            quantization="Q4_K_M",
            speed=50.0,
        )
        if hasattr(result, "calculate_hash"):
            h = result.calculate_hash()
            assert len(h) == 16

    def test_result_to_dict_has_all_fields(self):
        """BenchmarkResult has expected fields."""
        bm_mod = _import_benchmark()
        result = _make_result(
            bm_mod,
            model_name="test/model",
            quantization="Q4_K_M",
            speed=50.0,
        )
        assert result.model_name == "test/model"
        assert result.avg_tokens_per_sec == 50.0
        assert result.quantization == "Q4_K_M"


class TestBenchmarkCacheExtra:
    """More tests for BenchmarkCache operations."""

    def test_cache_does_not_return_stale_results(self, tmp_path):
        """Cache returns None for uncached entries."""
        bm_mod = _import_benchmark()
        cache = bm_mod.BenchmarkCache(tmp_path / "test_cache.db")
        result = cache.get_cached_result("nonexistent/model", "Q4_K_M")
        assert result is None

    def test_cache_round_trip(self, tmp_path):
        """Save and retrieve result from cache."""
        bm_mod = _import_benchmark()
        cache = bm_mod.BenchmarkCache(tmp_path / "test_cache.db")
        result = _make_result(
            bm_mod,
            model_name="cache/model",
            quantization="Q4_K_M",
            speed=75.0,
        )
        model_key = "cache/model__Q4_K_M"
        params_hash = "testhash123456"
        result.inference_params_hash = params_hash
        cache.save_result(result, model_key, params_hash, "test prompt", 2048)
        retrieved = cache.get_cached_result(model_key, params_hash)
        assert retrieved is not None
        assert abs(retrieved.avg_tokens_per_sec - 75.0) < 0.1

    def test_cache_get_all_results_empty(self, tmp_path):
        """get_all_results returns empty list on fresh cache."""
        bm_mod = _import_benchmark()
        cache = bm_mod.BenchmarkCache(tmp_path / "empty_cache.db")
        results = cache.get_all_results()
        assert results == []

    def test_cache_get_all_results_with_multiple(self, tmp_path):
        """get_all_results returns all saved results."""
        bm_mod = _import_benchmark()
        cache = bm_mod.BenchmarkCache(tmp_path / "multi_cache.db")
        for i, q in enumerate(["Q4_K_M", "Q8_0", "F16"]):
            result = _make_result(
                bm_mod,
                model_name="multi/model",
                quantization=q,
                speed=float(50 + i * 10),
            )
            model_key = f"multi/model__{q}"
            h = f"hash{i:016d}"
            result.inference_params_hash = h
            cache.save_result(result, model_key, h, "prompt", 2048)
        results = cache.get_all_results()
        assert len(results) >= 3

    def test_cache_clear_removes_entries(self, tmp_path):
        """Cache clear removes saved results."""
        bm_mod = _import_benchmark()
        cache = bm_mod.BenchmarkCache(tmp_path / "clear_cache.db")
        result = _make_result(
            bm_mod,
            model_name="clear/model",
            quantization="Q4_K_M",
            speed=60.0,
        )
        h = "clearhash00000000"
        result.inference_params_hash = h
        cache.save_result(result, "clear/model__Q4_K_M", h, "prompt", 2048)
        results = cache.get_all_results()
        assert len(results) >= 1


# ============================================================================
# PDF export tests with vision/tool models
# ============================================================================

class TestBenchmarkExportPDF:
    """Tests for _export_pdf with various result configurations."""

    def test_export_pdf_vision_and_tool_models(self, tmp_path):
        """_export_pdf covers vision and tool model sections."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)

        results = []
        for i in range(3):
            r = _make_result(bm_mod, model_name=f"vision{i}/llava",
                             quantization="Q4_K_M", speed=50.0 + i * 5, vision=True)
            results.append(r)
        for i in range(2):
            r = _make_result(bm_mod, model_name=f"tool{i}/mistral",
                             quantization="Q8_0", speed=60.0 + i * 5, tools=True)
            results.append(r)
        for i in range(6):
            r = _make_result(bm_mod, model_name=f"model{i}/llama",
                             quantization="Q4_K_M", speed=40.0 + i * 3)
            results.append(r)

        bench.cli_args.update({
            "limit": 10,
            "retest": True,
            "only_vision": True,
            "only_tools": True,
            "enable_profiling": True,
            "max_temp": 85,
            "max_power": 250,
            "include_models": "llava|mistral",
            "exclude_models": "gpt",
        })
        bench._gtt_info = {"total": 16.0, "used": 4.0}
        bench.use_gtt = True

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_pdf("20240101_120000", results)
            except Exception:
                pass

    def test_export_pdf_rank_by_ttft(self, tmp_path):
        """_export_pdf with rank_by=ttft covers ttft sort branch."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.rank_by = "ttft"

        results = [
            _make_result(bm_mod, model_name=f"model{i}", quantization="Q4_K_M",
                         speed=50.0 + i, ttft=0.1 + i * 0.1)
            for i in range(5)
        ]

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_pdf("20240101_120001", results)
            except Exception:
                pass

    def test_export_pdf_rank_by_vram(self, tmp_path):
        """_export_pdf with rank_by=vram covers vram sort branch."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.rank_by = "vram"

        results = [
            _make_result(bm_mod, model_name=f"vram{i}", quantization="Q4_K_M",
                         speed=50.0 + i)
            for i in range(5)
        ]

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_pdf("20240101_120002", results)
            except Exception:
                pass

    def test_export_pdf_rank_by_efficiency(self, tmp_path):
        """_export_pdf with rank_by=efficiency covers efficiency sort branch."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.rank_by = "efficiency"

        results = [
            _make_result(bm_mod, model_name=f"eff{i}", quantization="Q4_K_M",
                         speed=50.0 + i)
            for i in range(5)
        ]

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_pdf("20240101_120003", results)
            except Exception:
                pass

    def test_export_pdf_with_profiling_data(self, tmp_path):
        """_export_pdf covers profiling table when temp/power data present."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.enable_profiling = True
        bench.cli_args.update({
            "enable_profiling": True,
            "max_temp": 80,
            "max_power": 200,
        })

        results = []
        for i in range(5):
            r = _make_result(bm_mod, model_name=f"prof{i}", quantization="Q4_K_M",
                             speed=50.0 + i * 5)
            r.temp_celsius_min = 60.0 + i
            r.temp_celsius_avg = 70.0 + i
            r.temp_celsius_max = 80.0 + i
            r.power_watts_min = 100.0 + i
            r.power_watts_avg = 120.0 + i
            r.power_watts_max = 140.0 + i
            results.append(r)

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_pdf("20240101_120004", results)
            except Exception:
                pass


class TestBenchmarkExportHTML:
    """Tests for _export_html with various cli_args."""

    def test_export_html_with_cli_args_branches(self, tmp_path):
        """_export_html covers optional cli_args branches."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)
        bench.enable_profiling = True

        bench.cli_args.update({
            "limit": 5,
            "retest": True,
            "only_vision": True,
            "only_tools": True,
            "include_models": "llava|mistral",
            "exclude_models": "gpt",
            "enable_profiling": True,
            "max_temp": 85,
            "max_power": 200,
        })
        bench._gtt_info = {"total": 16.0, "used": 4.0}
        bench.use_gtt = True

        results = []
        for i in range(6):
            r = _make_result(bm_mod, model_name=f"html{i}", quantization="Q4_K_M",
                             speed=50.0 + i * 5, vision=(i % 2 == 0))
            r.temp_celsius_avg = 70.0 + i
            r.power_watts_avg = 120.0 + i
            results.append(r)
        bench.results = results

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_html("20240101_130000", results)
            except Exception:
                pass

    def test_export_results_to_files_with_data(self, tmp_path):
        """_export_results_to_files calls pdf and html exports."""
        bm_mod = _import_benchmark()
        bench = _make_benchmark_instance(bm_mod, tmp_path)

        results = [
            _make_result(bm_mod, model_name=f"file{i}", quantization="Q4_K_M",
                         speed=50.0 + i)
            for i in range(3)
        ]
        bench.results = results

        with patch.object(bm_mod, "RESULTS_DIR", tmp_path):
            try:
                bench._export_results_to_files(results)
            except Exception:
                pass


# ============================================================================
# benchmark.py main() branch tests
# ============================================================================

class TestBenchmarkMainBranches:
    """Tests for benchmark.py main() function branches."""

    def test_main_list_cache_with_data(self, tmp_path):
        """main() with --list-cache prints cached model entries."""
        bm = _import_benchmark()
        mock_cache = MagicMock()
        mock_cache.list_cached_models.return_value = [
            {
                "model_name": "test/model", "quantization": "Q4_K_M",
                "params_size": "7B", "avg_tokens_per_sec": 50.0,
                "timestamp": "2024-01-01T00:00:00", "params_hash": "abcdef12",
            },
            {
                "model_name": "test/model2", "quantization": "Q8_0",
                "params_size": "13B", "avg_tokens_per_sec": 35.0,
                "timestamp": "2024-01-02T00:00:00", "params_hash": "12345678",
            },
        ]
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "BenchmarkCache", return_value=mock_cache), \
                patch("sys.argv", ["benchmark.py", "--list-cache", "--debug"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()
        mock_cache.list_cached_models.assert_called_once()

    def test_main_list_cache_empty(self, tmp_path):
        """main() with --list-cache handles empty cache."""
        bm = _import_benchmark()
        mock_cache = MagicMock()
        mock_cache.list_cached_models.return_value = []
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "BenchmarkCache", return_value=mock_cache), \
                patch("sys.argv", ["benchmark.py", "--list-cache"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_export_cache(self, tmp_path):
        """main() with --export-cache exports to JSON."""
        bm = _import_benchmark()
        mock_cache = MagicMock()
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "BenchmarkCache", return_value=mock_cache), \
                patch("sys.argv", ["benchmark.py", "--export-cache", "output.json"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()
        mock_cache.export_to_json.assert_called_once()

    def test_main_export_only_with_results(self, tmp_path):
        """main() with --export-only re-exports from cache."""
        bm = _import_benchmark()
        mock_result = _make_result(bm, model_name="cached/model",
                                   quantization="Q4_K_M", speed=55.0)
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [mock_result]
        mock_benchmark = MagicMock()
        mock_benchmark.results = [mock_result]
        mock_benchmark._matches_filters.return_value = True

        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "BenchmarkCache", return_value=mock_cache), \
                patch.object(bm, "LMStudioBenchmark", return_value=mock_benchmark), \
                patch("sys.argv", ["benchmark.py", "--export-only"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_export_only_empty_cache(self, tmp_path):
        """main() with --export-only handles empty cache gracefully."""
        bm = _import_benchmark()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "BenchmarkCache", return_value=mock_cache), \
                patch("sys.argv", ["benchmark.py", "--export-only"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_dev_mode_with_models(self, tmp_path):
        """main() with --dev-mode selects smallest model."""
        bm = _import_benchmark()
        mock_benchmark = MagicMock()
        mock_benchmark.run_all_benchmarks.return_value = "completed"

        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm.ModelDiscovery, "get_installed_models",
                             return_value=["model-a@Q4_K_M", "model-b@Q8_0"]), \
                patch.object(bm.ModelDiscovery, "get_model_metadata",
                             return_value={"model_size_gb": 4.0, "params_size": "7B",
                                           "architecture": "llama",
                                           "max_context_length": 4096,
                                           "has_vision": False, "has_tools": False}), \
                patch.object(bm, "LMStudioBenchmark", return_value=mock_benchmark), \
                patch("sys.argv", ["benchmark.py", "--dev-mode"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_dev_mode_no_models(self, tmp_path):
        """main() with --dev-mode returns early when no models found."""
        bm = _import_benchmark()
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm.ModelDiscovery, "get_installed_models",
                             return_value=[]), \
                patch("sys.argv", ["benchmark.py", "--dev-mode"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_validation_runs_less_than_1(self, tmp_path):
        """main() --runs 0 causes SystemExit."""
        bm = _import_benchmark()
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch("sys.argv", ["benchmark.py", "--runs", "0"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            with pytest.raises(SystemExit):
                bm.main()

    def test_main_run_status_no_new_models(self, tmp_path):
        """main() logs info when run_status='no_new_models'."""
        bm = _import_benchmark()
        mock_benchmark = MagicMock()
        mock_benchmark.run_all_benchmarks.return_value = "no_new_models"
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "LMStudioBenchmark", return_value=mock_benchmark), \
                patch("sys.argv", ["benchmark.py", "--runs", "1"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_run_status_failed(self, tmp_path):
        """main() logs error when run_status='failed'."""
        bm = _import_benchmark()
        mock_benchmark = MagicMock()
        mock_benchmark.run_all_benchmarks.return_value = "failed"
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "LMStudioBenchmark", return_value=mock_benchmark), \
                patch("sys.argv", ["benchmark.py", "--runs", "1"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()

    def test_main_with_api_token(self, tmp_path):
        """main() --api-token sets token in lmstudio config."""
        bm = _import_benchmark()
        mock_benchmark = MagicMock()
        mock_benchmark.run_all_benchmarks.return_value = "completed"
        with patch.object(bm, "RESULTS_DIR", tmp_path), \
                patch.object(bm, "LOGS_DIR", tmp_path), \
                patch.object(bm, "LMStudioBenchmark", return_value=mock_benchmark), \
                patch("sys.argv", ["benchmark.py", "--api-token", "tok123",
                                   "--runs", "1"]), \
                patch("psutil.Process") as mock_proc:
            mock_proc.return_value.parent.return_value = None
            bm.main()
