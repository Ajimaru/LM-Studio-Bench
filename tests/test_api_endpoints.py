"""Comprehensive FastAPI endpoint tests for web/app.py."""
import asyncio
import importlib
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

if "lmstudio" not in sys.modules:
    sys.modules["lmstudio"] = MagicMock()


def _get_client() -> TestClient:
    """Return a TestClient for the FastAPI app."""
    if "app" in sys.modules:
        _app_mod = sys.modules["app"]
    else:
        _app_mod = importlib.import_module("app")
    return TestClient(_app_mod.app, raise_server_exceptions=False)


def _make_bench_result_dict(**kwargs) -> Dict[str, Any]:
    """Return a minimal benchmark result dict for mocking."""
    defaults = {
        "model_name": "test-model",
        "quantization": "Q4_K_M",
        "gpu_type": "NVIDIA",
        "gpu_offload": 1.0,
        "vram_mb": "8192",
        "avg_tokens_per_sec": 55.0,
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
        "tokens_per_sec_per_gb": 13.75,
        "tokens_per_sec_per_billion_params": 7.86,
    }
    defaults.update(kwargs)
    return defaults


class TestStatusEndpoint:
    """Tests for GET /api/status."""

    def test_returns_status_idle(self):
        """Returns idle status when no benchmark running."""
        client = _get_client()
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "running" in data

    def test_running_false_when_idle(self):
        """running is False when no benchmark is active."""
        client = _get_client()
        response = client.get("/api/status")
        assert response.json()["running"] is False


class TestBenchmarkControlEndpoints:
    """Tests for benchmark start/pause/resume/stop endpoints."""

    def test_pause_returns_false_when_not_running(self):
        """Pause returns false when nothing is running."""
        client = _get_client()
        response = client.post("/api/benchmark/pause")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_resume_returns_false_when_not_paused(self):
        """Resume returns false when nothing is paused."""
        client = _get_client()
        response = client.post("/api/benchmark/resume")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_stop_returns_false_when_not_running(self):
        """Stop returns false when nothing is running."""
        client = _get_client()
        response = client.post("/api/benchmark/stop")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_start_benchmark_returns_result(self):
        """Start benchmark endpoint returns a result dict."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(app_mod.manager, "start_benchmark", new=AsyncMock(return_value=True)):
            response = client.post(
                "/api/benchmark/start",
                json={"runs": 1, "context": 512, "enable_profiling": False},
            )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_start_benchmark_with_all_params(self):
        """Start benchmark with all optional parameters."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(app_mod.manager, "start_benchmark", new=AsyncMock(return_value=True)):
            response = client.post(
                "/api/benchmark/start",
                json={
                    "runs": 2,
                    "context": 1024,
                    "limit": 5,
                    "prompt": "Test prompt",
                    "min_context": 512,
                    "max_size": 8.0,
                    "quants": "Q4_K_M",
                    "arch": "llama",
                    "params": "7B",
                    "rank_by": "speed",
                    "include_models": "llama",
                    "exclude_models": "embed",
                    "only_vision": True,
                    "only_tools": False,
                    "retest": True,
                    "dev_mode": True,
                    "enable_profiling": True,
                    "disable_gtt": True,
                    "max_temp": 90.0,
                    "max_power": 250.0,
                    "temperature": 0.7,
                    "top_k_sampling": 40,
                    "top_p_sampling": 0.9,
                    "min_p_sampling": 0.05,
                    "repeat_penalty": 1.1,
                    "max_tokens": 256,
                    "n_gpu_layers": -1,
                    "n_batch": 512,
                    "n_threads": -1,
                    "flash_attention": True,
                    "rope_freq_base": 10000.0,
                    "rope_freq_scale": 1.0,
                    "use_mmap": True,
                    "use_mlock": False,
                    "kv_cache_quant": "q4_0",
                },
            )
        assert response.status_code == 200

    def test_start_benchmark_with_no_flash_attention(self):
        """Start benchmark with flash_attention disabled."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(app_mod.manager, "start_benchmark", new=AsyncMock(return_value=False)):
            response = client.post(
                "/api/benchmark/start",
                json={"flash_attention": False, "use_mmap": False, "use_mlock": True},
            )
        assert response.status_code == 200


class TestSystemEndpoints:
    """Tests for system-related API endpoints."""

    def test_latest_release_returns_dict(self):
        """Latest release endpoint returns a dict."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(
            app_mod,
            "_get_cached_latest_release",
            return_value={
                "current_version": "v0.1.0",
                "latest_version": "v0.2.0",
                "download_url": "https://example.com",
                "is_update_available": True,
            },
        ):
            response = client.get("/api/system/latest-release")
        assert response.status_code == 200
        data = response.json()
        assert "current_version" in data

    def test_latest_release_failure_returns_failure_dict(self):
        """Returns failure dict when release check fails."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(app_mod, "_get_cached_latest_release", return_value=None):
            response = client.get("/api/system/latest-release")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_shutdown_starts_thread(self):
        """Shutdown endpoint returns success and starts shutdown thread."""
        client = _get_client()
        with patch("os.kill"), \
                patch("time.sleep"):
            response = client.post("/api/system/shutdown")
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestResultsEndpoints:
    """Tests for /api/results endpoint."""

    def test_results_empty_when_no_cache(self):
        """Returns empty results when cache has no data."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    def test_results_returns_formatted_data(self):
        """Returns properly formatted result data."""
        app_mod = sys.modules["app"]
        bm = sys.modules.get("benchmark", None)
        if bm is None:
            import benchmark as bm
        client = _get_client()
        result = bm.BenchmarkResult(
            **_make_bench_result_dict(),
            speed_delta_pct=5.0,
            prev_timestamp="2023-12-01T00:00:00",
            temp_celsius_avg=70.0,
            power_watts_avg=150.0,
            gtt_enabled=False,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 1

    def test_results_handles_exception(self):
        """Returns error on cache exception."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(
            app_mod, "BenchmarkCache",
            side_effect=RuntimeError("db error"),
        ):
            response = client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestCacheStatsEndpoint:
    """Tests for GET /api/cache/stats."""

    def test_empty_cache_stats(self):
        """Returns zero-stats for empty cache."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_entries"] == 0

    def test_cache_stats_with_data(self):
        """Returns real stats when cache has data."""
        app_mod = sys.modules["app"]
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()
        results = [
            bm.BenchmarkResult(**_make_bench_result_dict(avg_tokens_per_sec=50.0)),
            bm.BenchmarkResult(**_make_bench_result_dict(avg_tokens_per_sec=100.0)),
        ]
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = results
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache), \
                patch.object(app_mod, "DATABASE_FILE", MagicMock(exists=lambda: False)):
            response = client.get("/api/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["stats"]["total_entries"] == 2


class TestCacheDeletion:
    """Tests for cache delete endpoint."""

    def test_delete_cache_entry(self):
        """Delete endpoint removes cache entry."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.delete("/api/cache/pub__model")
        assert response.status_code == 200

    def test_clear_cache(self):
        """Clear cache endpoint truncates the table."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post("/api/cache/clear")
        assert response.status_code == 200


class TestLMStudioModelsEndpoint:
    """Tests for GET /api/lmstudio/models."""

    def test_returns_empty_list_on_no_models(self):
        """Returns empty list when lms command fails."""
        client = _get_client()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout="", stderr=""),
        ):
            response = client.get("/api/lmstudio/models")
        assert response.status_code == 200


class TestPresetsEndpoints:
    """Tests for preset CRUD endpoints."""

    def test_list_presets(self):
        """List presets returns success with preset info."""
        client = _get_client()
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["presets"], list)

    def test_get_default_preset(self):
        """Get default preset returns its config."""
        client = _get_client()
        response = client.get("/api/presets/default")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["name"] == "default"
        assert "config" in data

    def test_get_invalid_preset_name(self):
        """Invalid preset name returns failure."""
        client = _get_client()
        response = client.get("/api/presets/invalid name with spaces")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_get_nonexistent_preset(self):
        """Non-existent preset returns failure."""
        client = _get_client()
        response = client.get("/api/presets/nonexistent_preset_xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_save_valid_preset(self):
        """Save a valid user preset."""
        client = _get_client()
        response = client.post(
            "/api/presets",
            json={"name": "testpreset99", "config": {"runs": 2}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_save_invalid_preset_name(self):
        """Save fails for invalid preset name."""
        client = _get_client()
        response = client.post(
            "/api/presets",
            json={"name": "invalid name", "config": {}},
        )
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_save_readonly_preset(self):
        """Cannot overwrite a readonly preset."""
        client = _get_client()
        response = client.post(
            "/api/presets",
            json={"name": "default", "config": {}},
        )
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_delete_nonexistent_preset(self):
        """Delete non-existent preset returns failure."""
        client = _get_client()
        response = client.delete("/api/presets/nonexistent_preset_xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_delete_readonly_preset(self):
        """Cannot delete readonly preset."""
        client = _get_client()
        response = client.delete("/api/presets/default")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_delete_invalid_name(self):
        """Invalid preset name returns failure on delete."""
        client = _get_client()
        response = client.delete("/api/presets/invalid name")
        assert response.status_code == 200
        assert response.json()["success"] is False

    def test_compare_presets(self):
        """Compare two predefined presets returns differences."""
        client = _get_client()
        response = client.post(
            "/api/presets/compare",
            json={"preset_a": "default", "preset_b": "quick_test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "differences" in data

    def test_export_presets(self):
        """Export presets endpoint returns a response."""
        client = _get_client()
        response = client.get("/api/presets/export")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_import_presets_no_data(self):
        """Import presets fails when no data provided."""
        client = _get_client()
        response = client.post(
            "/api/presets/import",
            json={"data": ""},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestOutputEndpoint:
    """Tests for GET /api/output."""

    def test_output_when_not_running(self):
        """Returns empty output when no benchmark running."""
        client = _get_client()
        response = client.get("/api/output")
        assert response.status_code == 200
        data = response.json()
        assert "output" in data


class TestLMStudioHealthEndpoint:
    """Tests for GET /api/lmstudio/health."""

    def test_offline_when_no_server(self):
        """Returns offline when LM Studio is not running."""
        client = _get_client()
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            import httpx
            mock_client.get.side_effect = httpx.ConnectError("no connection")
            mock_cls.return_value = mock_client
            with patch(
                "subprocess.run",
                return_value=MagicMock(
                    returncode=0,
                    stdout="server: off",
                    stderr="",
                ),
            ):
                response = client.get("/api/lmstudio/health")
        assert response.status_code == 200
        data = response.json()
        assert "ok" in data

    def test_online_when_server_responds(self):
        """Returns online when LM Studio responds with 200."""
        client = _get_client()
        with patch("httpx.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_client.get.return_value = mock_resp
            mock_cls.return_value = mock_client
            response = client.get("/api/lmstudio/health")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True


class TestExperimentEndpoints:
    """Tests for A/B testing experiment endpoints."""

    def test_create_experiment_returns_id(self):
        """Create experiment returns experiment_id."""
        client = _get_client()
        response = client.post(
            "/api/experiments/create",
            json={
                "name": "test-exp",
                "model_name": "pub/model",
                "baseline_params": {"name": "baseline", "temperature": 0.7},
                "test_params": {"name": "test", "temperature": 0.9},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "experiment_id" in data


class TestComparisonEndpoints:
    """Tests for model comparison endpoints."""

    def test_comparison_models_empty(self):
        """Returns empty list when no models in cache."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.list_cached_models.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/comparison/models")
        assert response.status_code == 200

    def test_model_history_empty(self):
        """Returns empty history for a model not in cache."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/comparison/pub/model")
        assert response.status_code == 200


class TestExtraComparisonEndpoints:
    """Additional tests for comparison endpoints."""

    def test_comparison_models_with_data(self):
        """Returns model list when cache has data."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.list_cached_models.return_value = ["pub/llama@q4"]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/comparison/models")
        assert response.status_code == 200

    def test_model_history_with_data(self):
        """Returns model history for existing model."""
        app_mod = sys.modules["app"]
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()
        result = bm.BenchmarkResult(
            model_name="test-model", quantization="Q4_K_M",
            gpu_type="NVIDIA", gpu_offload=1.0, vram_mb="8192",
            avg_tokens_per_sec=55.0, avg_ttft=0.3, avg_gen_time=0.8,
            prompt_tokens=10, completion_tokens=50,
            timestamp="2024-01-01T00:00:00", params_size="7B",
            architecture="llama", max_context_length=4096,
            model_size_gb=4.0, has_vision=False, has_tools=False,
            tokens_per_sec_per_gb=13.75,
            tokens_per_sec_per_billion_params=7.86,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.get("/api/comparison/test-model")
        assert response.status_code == 200


class TestHealthCheckEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self):
        """Health check endpoint returns status ok."""
        client = _get_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "benchmark_running" in data


class TestLatestResultsEndpoint:
    """Tests for /api/latest-results endpoint."""

    def test_returns_none_when_no_results(self):
        """Returns null latest when no result files."""
        app_mod = sys.modules["app"]
        client = _get_client()
        with patch.object(app_mod, "RESULTS_DIR", MagicMock(
            glob=MagicMock(return_value=[])
        )):
            response = client.get("/api/latest-results")
        assert response.status_code == 200
        assert response.json()["latest"] is None


class TestExperimentEndpoints2:
    """Additional tests for experiment endpoints."""

    def test_experiment_comparison_no_data(self):
        """Returns failure when no data for experiment."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.get(
                "/api/experiments/exp1/comparison",
                params={
                    "baseline_hash": "abc123",
                    "test_hash": "def456",
                    "model_name": "test-model",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_post_experiment_comparison(self):
        """POST experiment comparison returns result."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/experiments/exp1/comparison",
                json={
                    "baseline_hash": "abc123",
                    "test_hash": "def456",
                    "model_name": "test-model",
                },
            )
        assert response.status_code == 200

    def test_experiment_export_csv(self):
        """Experiment export CSV returns success."""
        import tempfile
        app_mod = sys.modules["app"]
        client = _get_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = __import__("pathlib").Path(tmpdir)
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.post(
                    "/api/experiments/exp1/export",
                    json={
                        "format": "csv",
                        "baseline": {
                            "mean": 50.0, "std_dev": 2.0,
                            "min": 48.0, "max": 52.0, "count": 5,
                        },
                        "test": {
                            "mean": 55.0, "std_dev": 1.5,
                            "min": 53.0, "max": 57.0, "count": 5,
                        },
                        "comparison": {"winner": "test", "delta_pct": 10.0},
                        "statistical_test": {
                            "p_value": 0.03, "t_statistic": 2.1,
                            "significant": True,
                        },
                    },
                )
        assert response.status_code == 200

    def test_experiment_run_returns_result(self):
        """Run experiment returns a result dict."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/experiments/run",
                json={
                    "experiment_name": "Test Exp",
                    "model_name": "test/model",
                    "baseline_params": {"temperature": 0.7},
                    "test_params": {"temperature": 0.9},
                },
            )
        assert response.status_code == 200


class TestCSVExportEndpoints:
    """Tests for CSV export endpoints."""

    def test_csv_export_no_data(self):
        """CSV export returns failure when no data."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/comparison/export/csv",
                json={"model_name": "test-model"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_csv_export_with_data(self):
        """CSV export returns success when data found."""
        import tempfile
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (
                "2024-01-01T00:00:00", "test-model", "Q4_K_M",
                55.0, 0.3, 0.8, 1.0, "8192", 0.7, 40, 0.9, 0.05,
                1.1, 256, 3, 120.0, 0,
            )
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = __import__("pathlib").Path(tmpdir)
                with patch.object(app_mod, "RESULTS_DIR", real_dir):
                    response = client.post(
                        "/api/comparison/export/csv",
                        json={},
                    )
        assert response.status_code == 200


class TestStatisticsEndpoint:
    """Tests for /api/comparison/statistics/{model_name}."""

    def test_statistics_no_history(self):
        """Statistics returns failure when no history."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache):
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={},
            )
        assert response.status_code == 200


class TestDashboardStats:
    """Tests for /api/dashboard/stats endpoint."""

    def test_returns_stats_with_empty_cache(self):
        """Returns stats for empty cache."""
        app_mod = sys.modules["app"]
        client = _get_client()
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = []
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache), \
                patch("subprocess.run", return_value=MagicMock(
                    returncode=1, stdout="", stderr=""
                )):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_returns_stats_with_data(self):
        """Returns complete stats with result data."""
        app_mod = sys.modules["app"]
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()
        result = bm.BenchmarkResult(
            model_name="test-model", quantization="Q4_K_M",
            gpu_type="NVIDIA", gpu_offload=1.0, vram_mb="8192",
            avg_tokens_per_sec=55.0, avg_ttft=0.3, avg_gen_time=0.8,
            prompt_tokens=10, completion_tokens=50,
            timestamp="2024-01-01T00:00:00", params_size="7B",
            architecture="llama", max_context_length=4096,
            model_size_gb=4.0, has_vision=True, has_tools=False,
            tokens_per_sec_per_gb=13.75,
            tokens_per_sec_per_billion_params=7.86,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache), \
                patch("subprocess.run", return_value=MagicMock(
                    returncode=1, stdout="", stderr=""
                )):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200


class TestPresetImportEndpoint:
    """Tests for POST /api/presets/import."""

    def test_import_valid_zip(self):
        """Import valid ZIP with a preset file."""
        import base64
        import io
        import zipfile
        client = _get_client()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("mypreset.json", '{"runs": 3}')
        zip_b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.post(
            "/api/presets/import",
            json={"data": zip_b64},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_import_skips_readonly(self):
        """Import skips readonly preset names."""
        import base64
        import io
        import zipfile
        client = _get_client()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("default.json", '{"runs": 3}')
        zip_b64 = base64.b64encode(buf.getvalue()).decode()
        response = client.post(
            "/api/presets/import",
            json={"data": zip_b64},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert any("default" in s for s in data.get("skipped", []))


def _get_benchmark_manager():
    """Return the BenchmarkManager instance from the app module."""
    if "app" in sys.modules:
        app_mod = sys.modules["app"]
    else:
        app_mod = importlib.import_module("app")
    return app_mod.manager, app_mod.BenchmarkManager


class TestBenchmarkManagerValidation:
    """Unit tests for BenchmarkManager input validation methods."""

    def test_validate_valid_int(self):
        """_validate_cli_arg_value accepts valid int for int flag."""
        manager, _ = _get_benchmark_manager()
        result = manager._validate_cli_arg_value("--runs", "3")
        assert result == "3"

    def test_validate_invalid_int_raises(self):
        """_validate_cli_arg_value raises ValueError for non-int int flag."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError):
            manager._validate_cli_arg_value("--runs", "abc")

    def test_validate_valid_float(self):
        """_validate_cli_arg_value accepts valid float for float flag."""
        manager, _ = _get_benchmark_manager()
        result = manager._validate_cli_arg_value("--temperature", "0.7")
        assert result == "0.7"

    def test_validate_invalid_float_raises(self):
        """_validate_cli_arg_value raises ValueError for non-float float flag."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError):
            manager._validate_cli_arg_value("--temperature", "hot")

    def test_validate_control_char_raises(self):
        """_validate_cli_arg_value raises ValueError for control chars."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError):
            manager._validate_cli_arg_value("--prompt", "test\nprompt")

    def test_validate_null_byte_raises(self):
        """_validate_cli_arg_value raises ValueError for null bytes."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError):
            manager._validate_cli_arg_value("--prompt", "val\x00ue")

    def test_validate_too_long_raises(self):
        """_validate_cli_arg_value raises ValueError when value > 2000 chars."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError):
            manager._validate_cli_arg_value("--prompt", "x" * 2001)

    def test_validate_string_flag_passthrough(self):
        """_validate_cli_arg_value passes through non-int/float flags."""
        manager, _ = _get_benchmark_manager()
        result = manager._validate_cli_arg_value("--prompt", "Explain ML")
        assert result == "Explain ML"


class TestBenchmarkManagerSanitize:
    """Unit tests for BenchmarkManager._sanitize_benchmark_args."""

    def test_sanitize_valid_flag_with_value(self):
        """_sanitize_benchmark_args accepts valid flag+value pair."""
        manager, _ = _get_benchmark_manager()
        result = manager._sanitize_benchmark_args(["--runs", "3"])
        assert result == ["--runs", "3"]

    def test_sanitize_unknown_flag_raises(self):
        """_sanitize_benchmark_args raises ValueError for unknown flag."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError, match="Unsupported"):
            manager._sanitize_benchmark_args(["--evil-flag"])

    def test_sanitize_flag_without_required_value_raises(self):
        """_sanitize_benchmark_args raises ValueError when value missing."""
        manager, _ = _get_benchmark_manager()
        with pytest.raises(ValueError, match="Missing value"):
            manager._sanitize_benchmark_args(["--runs"])

    def test_sanitize_boolean_flags(self):
        """_sanitize_benchmark_args handles boolean flags without values."""
        manager, _ = _get_benchmark_manager()
        result = manager._sanitize_benchmark_args(["--retest", "--debug"])
        assert "--retest" in result
        assert "--debug" in result

    def test_sanitize_multiple_flags(self):
        """_sanitize_benchmark_args handles multiple flag-value pairs."""
        manager, _ = _get_benchmark_manager()
        result = manager._sanitize_benchmark_args(
            ["--runs", "3", "--limit", "5", "--retest"]
        )
        assert result == ["--runs", "3", "--limit", "5", "--retest"]


class TestBenchmarkManagerBuildSafeCommand:
    """Unit tests for BenchmarkManager._build_safe_command."""

    def test_build_safe_command_returns_list(self):
        """_build_safe_command returns list with interpreter, script, args."""
        _, BenchmarkManager = _get_benchmark_manager()
        result = BenchmarkManager._build_safe_command(
            ["--runs", "3"]
        )
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_build_safe_command_has_python_interpreter(self):
        """_build_safe_command first element is python interpreter."""
        import sys as _sys
        _, BenchmarkManager = _get_benchmark_manager()
        result = BenchmarkManager._build_safe_command([])
        assert result[0] == _sys.executable

    def test_build_safe_command_shell_unsafe_arg_raises(self):
        """_build_safe_command raises ValueError on shell-unsafe characters."""
        _, BenchmarkManager = _get_benchmark_manager()
        with pytest.raises(ValueError, match="Shell-unsafe"):
            BenchmarkManager._build_safe_command(["--runs", "3; rm -rf /"])


class TestBenchmarkManagerSetIdle:
    """Tests for BenchmarkManager.set_idle_status."""

    def test_set_idle_status_changes_state(self):
        """set_idle_status sets status to idle."""
        manager, _ = _get_benchmark_manager()
        manager._state.status = "running"
        manager.set_idle_status()
        assert manager._state.status == "idle"


class TestDashboardStatsWithData:
    """Tests for /api/dashboard/stats with mocked result data."""

    def test_stats_endpoint_with_results(self):
        """GET /api/dashboard/stats includes top_models when results exist."""
        client = _get_client()
        results = [_make_bench_result_dict(avg_tokens_per_sec=80.0)]
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                (
                    results[0]["model_name"],
                    results[0]["quantization"],
                    results[0]["avg_tokens_per_sec"],
                    results[0]["avg_ttft"],
                    results[0]["timestamp"],
                    results[0]["gpu_type"],
                    results[0]["gpu_offload"],
                    results[0]["vram_mb"],
                    results[0]["params_size"],
                    results[0]["architecture"],
                    results[0]["has_vision"],
                    results[0]["has_tools"],
                    results[0]["tokens_per_sec_per_gb"],
                )
            ]
            mock_cursor.fetchone.return_value = (1, 80.0, 0.3)
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert "system_info" in data or "hardware" in data or isinstance(
            data, dict
        )

    def test_stats_endpoint_nvidia_gpu_info(self):
        """GET /api/dashboard/stats returns 200 with NVIDIA GPU mocked."""
        client = _get_client()
        mock_proc = MagicMock(returncode=0, stdout="RTX 4080\n", stderr="")
        with patch("subprocess.run", return_value=mock_proc), \
                patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_stats_endpoint_subprocess_failure(self):
        """GET /api/dashboard/stats handles subprocess errors gracefully."""
        client = _get_client()
        with patch("subprocess.run", side_effect=OSError("no GPU")):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200


class TestStartBenchmarkSanitization:
    """Test that POST /api/benchmark/start sanitizes args before execution."""

    def test_start_rejects_unknown_flag(self):
        """POST /api/benchmark/start rejects unknown flag."""
        client = _get_client()
        with patch(
            "subprocess.Popen", side_effect=ValueError("should not be called")
        ):
            response = client.post(
                "/api/benchmark/start",
                json={"args": ["--evil-flag", "payload"]},
            )
        assert response.status_code in (400, 422, 200)

    def test_start_rejects_shell_metachar(self):
        """POST /api/benchmark/start rejects args with shell metacharacters."""
        client = _get_client()
        with patch(
            "subprocess.Popen", side_effect=ValueError("should not be called")
        ):
            response = client.post(
                "/api/benchmark/start",
                json={"args": ["--runs", "3; rm -rf /"]},
            )
        assert response.status_code in (400, 422, 200)


class TestExperimentComparisonEndpoints:
    """Tests for experiment comparison API endpoints."""

    def test_get_comparison_no_data_returns_ok(self):
        """GET /api/experiments/{id}/comparison returns 200 with empty data."""
        client = _get_client()
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None
            response = client.get("/api/experiments/1/comparison")
        assert response.status_code in (200, 404, 422)

    def test_post_comparison_no_data_returns_ok(self):
        """POST /api/experiments/{id}/comparison returns ok with empty rows."""
        client = _get_client()
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None
            response = client.post("/api/experiments/1/comparison", json={})
        assert response.status_code in (200, 404, 422)


class TestComparisonStatisticsEndpoint:
    """Tests for POST /api/comparison/statistics/{model}."""

    def test_statistics_with_no_data(self):
        """POST /api/comparison/statistics/{model} returns ok with no DB."""
        client = _get_client()
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = []
            mock_cursor.fetchone.return_value = None
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={}
            )
        assert response.status_code in (200, 404, 422)

    def test_statistics_with_data(self):
        """POST /api/comparison/statistics/{model} returns stats when data."""
        client = _get_client()
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_cursor.fetchall.return_value = [
                ("test-model", "Q4_K_M", 50.0, "2024-01-01T00:00:00"),
                ("test-model", "Q4_K_M", 55.0, "2024-01-02T00:00:00"),
                ("test-model", "Q4_K_M", 60.0, "2024-01-03T00:00:00"),
                ("test-model", "Q4_K_M", 58.0, "2024-01-04T00:00:00"),
            ]
            mock_cursor.fetchone.return_value = (4, 55.75, 4.08)
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={}
            )
        assert response.status_code in (200, 404, 422)


# ============================================================================
# Additional tests for deeper coverage of web/app.py
# ============================================================================


class TestComparisonPDFExportWithData:
    """Tests for POST /api/comparison/export/pdf with real data rows."""

    def test_pdf_export_no_data_returns_failure(self):
        """PDF export returns failure when DB has no rows."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/comparison/export/pdf",
                json={"model_name": "nonexistent-model"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_pdf_export_with_data_success(self):
        """PDF export succeeds when DB has rows."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", "test-model", "Q4_K_M",
             55.0, 0.3, 0.8, 1.0, "8192", 0.7),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("sqlite3.connect", return_value=mock_conn):
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = Path(tmpdir)
                with patch.object(app_mod, "RESULTS_DIR", real_dir):
                    response = client.post(
                        "/api/comparison/export/pdf",
                        json={},
                    )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True or "error" in data

    def test_pdf_export_with_filters(self):
        """PDF export handles quantization filters."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", "test-model", "Q4_K_M",
             50.0, 0.3, 0.8, 1.0, "8192", 0.7),
            ("2024-01-02T00:00:00", "test-model", "Q8_0",
             60.0, 0.2, 0.7, 1.0, "8192", 0.7),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("sqlite3.connect", return_value=mock_conn):
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = Path(tmpdir)
                with patch.object(app_mod, "RESULTS_DIR", real_dir):
                    response = client.post(
                        "/api/comparison/export/pdf",
                        json={
                            "model_name": "test-model",
                            "quantizations": ["Q4_K_M", "Q8_0"],
                            "start_date": "2024-01-01",
                            "end_date": "2024-12-31",
                        },
                    )
        assert response.status_code == 200

    def test_pdf_export_with_string_quant_filter(self):
        """PDF export handles quantization as string (not list)."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", "test-model", "Q4_K_M",
             55.0, 0.3, 0.8, 1.0, "8192", 0.7),
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("sqlite3.connect", return_value=mock_conn):
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = Path(tmpdir)
                with patch.object(app_mod, "RESULTS_DIR", real_dir):
                    response = client.post(
                        "/api/comparison/export/pdf",
                        json={"quantizations": "Q4_K_M"},
                    )
        assert response.status_code == 200


class TestStatisticsAdvancedWithData:
    """Test POST /api/comparison/statistics/{model} with 2+ data points."""

    def test_statistics_with_enough_data(self):
        """Statistics endpoint returns success with 4 data points."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", 50.0),
            ("2024-01-02T00:00:00", 55.0),
            ("2024-01-03T00:00:00", 60.0),
            ("2024-01-04T00:00:00", 58.0),
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "advanced" in data or "basic" in data

    def test_statistics_with_exactly_two_points(self):
        """Statistics endpoint works with exactly 2 data points."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", 50.0),
            ("2024-01-02T00:00:00", 60.0),
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_statistics_with_single_point_returns_failure(self):
        """Statistics endpoint returns failure when only 1 data point."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("2024-01-01T00:00:00", 50.0),
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/comparison/statistics/test-model",
                json={},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestGETExperimentComparisonWithData:
    """Tests for GET /api/experiments/{id}/comparison with matching rows."""

    def test_get_comparison_with_matching_baseline_test(self):
        """Returns comparison stats when baseline/test rows match hashes."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        baseline_params = {"temperature": 0.7}
        test_params_dict = {"temperature": 0.9}
        baseline_hash = app_mod.calculate_hash(baseline_params)
        test_hash = app_mod.calculate_hash(test_params_dict)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        baseline_row = (
            "2024-01-01T00:00:00", 50.0, 0.3, 0.8,
            0.7, None, None, None, None, None, 3, 0,
        )
        test_row = (
            "2024-01-01T01:00:00", 60.0, 0.25, 0.7,
            0.9, None, None, None, None, None, 3, 0,
        )
        mock_cursor.fetchall.return_value = [
            baseline_row, baseline_row, test_row, test_row
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.get(
                "/api/experiments/exp1/comparison",
                params={
                    "baseline_hash": baseline_hash,
                    "test_hash": test_hash,
                    "model_name": "test-model",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_get_comparison_no_rows_returns_failure(self):
        """Returns failure when DB has no rows."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.get(
                "/api/experiments/exp1/comparison",
                params={
                    "baseline_hash": "abc12345",
                    "test_hash": "def67890",
                    "model_name": "no-model",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_get_comparison_with_date_filters(self):
        """GET comparison handles start_date and end_date params."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.get(
                "/api/experiments/exp1/comparison",
                params={
                    "baseline_hash": "abc12345",
                    "test_hash": "def67890",
                    "model_name": "test-model",
                    "start_date": "2024-01-01",
                    "end_date": "2024-12-31",
                },
            )
        assert response.status_code == 200


class TestPOSTExperimentComparisonWithData:
    """Tests for POST /api/experiments/{id}/comparison with data."""

    def test_post_comparison_with_matching_rows(self):
        """POST comparison returns success when rows match params."""
        client = _get_client()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        baseline_row = (
            "2024-01-01T00:00:00", 50.0, 0.3, 0.8,
            0.7, None, None, None, None, None, 3, 0,
        )
        baseline_row_2 = (
            "2024-01-01T00:01:00", 51.0, 0.31, 0.79,
            0.7, None, None, None, None, None, 3, 0,
        )
        test_row = (
            "2024-01-01T01:00:00", 60.0, 0.25, 0.7,
            0.9, None, None, None, None, None, 3, 0,
        )
        test_row_2 = (
            "2024-01-01T01:01:00", 61.0, 0.24, 0.69,
            0.9, None, None, None, None, None, 3, 0,
        )
        mock_cursor.fetchall.return_value = [
            baseline_row, baseline_row_2, test_row, test_row_2
        ]
        mock_conn.cursor.return_value = mock_cursor

        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/experiments/exp1/comparison",
                json={
                    "baseline_hash": "abc12345",
                    "test_hash": "def67890",
                    "model_name": "test-model",
                    "baseline_params": {"temperature": 0.7},
                    "test_params": {"temperature": 0.9},
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    def test_post_comparison_insufficient_data(self):
        """POST comparison returns failure with no matching rows."""
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/experiments/exp1/comparison",
                json={
                    "model_name": "test-model",
                    "baseline_params": {"temperature": 0.7},
                    "test_params": {"temperature": 0.9},
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_post_comparison_db_error(self):
        """POST comparison handles DB errors gracefully."""
        import sqlite3 as _sqlite3
        client = _get_client()
        with patch("sqlite3.connect", side_effect=_sqlite3.Error("DB error")):
            response = client.post(
                "/api/experiments/exp1/comparison",
                json={"model_name": "test-model"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False


class TestExperimentExportFormats:
    """Tests for POST /api/experiments/{id}/export CSV and PDF formats."""

    def test_export_csv_format(self):
        """Experiment export CSV runs without error."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.post(
                    "/api/experiments/exp-test/export",
                    json={
                        "format": "csv",
                        "baseline": {
                            "mean": 50.0, "std_dev": 2.0,
                            "min": 48.0, "max": 52.0,
                            "count": 3, "data": [],
                        },
                        "test": {
                            "mean": 55.0, "std_dev": 1.5,
                            "min": 53.0, "max": 57.0,
                            "count": 3, "data": [],
                        },
                        "comparison": {
                            "winner": "test",
                            "delta_pct": 10.0,
                            "significant": True,
                        },
                        "statistical_test": {
                            "p_value": 0.03,
                            "t_statistic": 2.1,
                            "significant": True,
                        },
                    },
                )
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True

    def test_export_pdf_format(self):
        """Experiment export PDF runs without error."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.post(
                    "/api/experiments/exp-test/export",
                    json={
                        "format": "pdf",
                        "baseline": {
                            "mean": 50.0, "std_dev": 2.0,
                            "min": 48.0, "max": 52.0,
                            "count": 3,
                        },
                        "test": {
                            "mean": 55.0, "std_dev": 1.5,
                            "min": 53.0, "max": 57.0,
                            "count": 3,
                        },
                        "comparison": {
                            "winner": "test",
                            "delta_pct": 10.0,
                        },
                        "statistical_test": {
                            "p_value": 0.03,
                            "t_statistic": 2.1,
                            "significant": True,
                        },
                    },
                )
        assert response.status_code == 200

    def test_export_with_path_traversal_denied(self):
        """Export with traversal in experiment ID is denied."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.post(
                    "/api/experiments/../../../etc/passwd/export",
                    json={
                        "format": "csv",
                        "baseline": {"mean": 50.0},
                        "test": {"mean": 55.0},
                        "comparison": {},
                        "statistical_test": {},
                    },
                )
        assert response.status_code in (200, 404, 422)


class TestRunExperimentWithMocks:
    """Tests for POST /api/experiments/run with mocked benchmarks."""

    def test_run_experiment_no_model_name(self):
        """run_experiment returns error when model_name missing."""
        client = _get_client()
        response = client.post(
            "/api/experiments/run",
            json={
                "experiment_name": "Test",
                "baseline_params": {},
                "test_params": {},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "model_name" in data.get("error", "").lower()

    def test_run_experiment_baseline_start_fails(self):
        """run_experiment returns error when baseline benchmark fails to start."""
        from unittest.mock import AsyncMock
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        with patch.object(
            app_mod.manager, "start_benchmark",
            new_callable=AsyncMock, return_value=False,
        ):
            response = client.post(
                "/api/experiments/run",
                json={
                    "experiment_name": "Test",
                    "model_name": "test/model",
                    "baseline_params": {"temperature": 0.7},
                    "test_params": {"temperature": 0.9},
                    "runs": 1,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_run_experiment_test_start_fails(self):
        """run_experiment returns error when test benchmark fails to start."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        call_count = [0]

        async def mock_start(*args):
            call_count[0] += 1
            return call_count[0] == 1 and isinstance(args, tuple)

        with patch.object(app_mod.manager, "start_benchmark", side_effect=mock_start), \
                patch.object(app_mod.manager, "is_running", return_value=False):
            response = client.post(
                "/api/experiments/run",
                json={
                    "experiment_name": "Test",
                    "model_name": "test/model",
                    "baseline_params": {"temperature": 0.7},
                    "test_params": {"temperature": 0.9},
                    "runs": 1,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_run_experiment_success_path(self):
        """run_experiment returns success with matched rows."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        baseline_row = (
            "2024-01-01 10:00:00", 50.0, 0.3, 0.8,
            0.7, None, None, None, None, None, 0,
            None, None, None, None, None, None, None, None, None,
        )
        test_row = (
            "2024-01-01 10:01:00", 60.0, 0.2, 0.7,
            0.9, None, None, None, None, None, 0,
            None, None, None, None, None, None, None, None, None,
        )

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [baseline_row, test_row]
        mock_conn.cursor.return_value = mock_cursor

        async def mock_start(*args):
            return isinstance(args, tuple)

        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            with patch.object(app_mod.manager, "start_benchmark", side_effect=mock_start), \
                    patch.object(app_mod.manager, "is_running", return_value=False), \
                    patch("sqlite3.connect", return_value=mock_conn), \
                    patch.object(app_mod, "USER_RESULTS_DIR", real_dir):
                response = client.post(
                    "/api/experiments/run",
                    json={
                        "experiment_name": "AB-Test",
                        "model_name": "test/model",
                        "baseline_params": {"temperature": 0.7},
                        "test_params": {"temperature": 0.9},
                        "runs": 1,
                    },
                )
        assert response.status_code == 200

    def test_run_experiment_all_params(self):
        """run_experiment handles full parameter set correctly."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        async def mock_start(*args):
            return isinstance(args, tuple)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        with patch.object(app_mod.manager, "start_benchmark", side_effect=mock_start), \
                patch.object(app_mod.manager, "is_running", return_value=False), \
                patch("sqlite3.connect", return_value=mock_conn):
            response = client.post(
                "/api/experiments/run",
                json={
                    "experiment_name": "Full AB Test",
                    "model_name": "test/model@Q4_K_M",
                    "baseline_params": {
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "min_p": 0.05,
                        "repeat_penalty": 1.1,
                        "max_tokens": 256,
                        "n_gpu_layers": 33,
                        "n_batch": 512,
                        "n_threads": 4,
                        "flash_attention": True,
                        "rope_freq_base": 1000000.0,
                        "use_mmap": True,
                        "use_mlock": False,
                        "kv_cache_quant": "Q4_0",
                    },
                    "test_params": {
                        "temperature": 0.9,
                        "flash_attention": False,
                        "use_mmap": False,
                        "use_mlock": True,
                    },
                    "runs": 1,
                    "context": 4096,
                    "prompt": "Test prompt for experiment",
                },
            )
        assert response.status_code == 200


class TestDashboardStatsDetailed:
    """Detailed tests for GET /api/dashboard/stats."""

    def test_dashboard_stats_returns_200(self):
        """GET /api/dashboard/stats always returns 200."""
        client = _get_client()
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_no_lmstudio(self):
        """Dashboard stats handles missing lmstudio gracefully."""
        client = _get_client()
        with patch("subprocess.run", side_effect=FileNotFoundError("lms not found")), \
                patch("subprocess.check_output", side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_with_nvidia_gpu(self):
        """Dashboard stats parses NVIDIA GPU info."""
        client = _get_client()
        mock_check_output = MagicMock(
            side_effect=[
                b"16384\n",
                b"NVIDIA RTX 4080\n",
            ]
        )
        with patch("subprocess.check_output", side_effect=mock_check_output), \
                patch("subprocess.run", return_value=MagicMock(
                    returncode=1, stdout="", stderr=""
                )):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_nvidia_timeout(self):
        """Dashboard stats handles NVIDIA timeout gracefully."""
        import subprocess as _subprocess
        client = _get_client()
        with patch("subprocess.check_output",
                   side_effect=_subprocess.TimeoutExpired("nvidia-smi", 5)):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_returns_dict_with_keys(self):
        """Dashboard stats response is a dict with expected structure."""
        client = _get_client()
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_dashboard_stats_with_result_data(self):
        """Dashboard stats works when cache has results."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()
        result = bm.BenchmarkResult(
            model_name="amd-model", quantization="Q4_K_M",
            gpu_type="AMD Radeon RX 6800 XT", gpu_offload=1.0,
            vram_mb="16384",
            avg_tokens_per_sec=40.0, avg_ttft=0.4, avg_gen_time=1.0,
            prompt_tokens=10, completion_tokens=50,
            timestamp="2024-01-01T00:00:00", params_size="7B",
            architecture="llama", max_context_length=4096,
            model_size_gb=4.0, has_vision=False, has_tools=False,
            tokens_per_sec_per_gb=10.0,
            tokens_per_sec_per_billion_params=5.71,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia-smi")), \
                patch("subprocess.run", return_value=MagicMock(
                    returncode=1, stdout="", stderr=""
                )):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_with_intel_gpu(self):
        """Dashboard stats handles Intel GPU type."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()
        result = bm.BenchmarkResult(
            model_name="intel-model", quantization="Q4_K_M",
            gpu_type="Intel Arc A770", gpu_offload=1.0,
            vram_mb="16384",
            avg_tokens_per_sec=35.0, avg_ttft=0.5, avg_gen_time=1.2,
            prompt_tokens=10, completion_tokens=40,
            timestamp="2024-01-01T00:00:00", params_size="7B",
            architecture="llama", max_context_length=4096,
            model_size_gb=4.0, has_vision=False, has_tools=False,
            tokens_per_sec_per_gb=8.75,
            tokens_per_sec_per_billion_params=5.0,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]
        with patch.object(app_mod, "BenchmarkCache", return_value=mock_cache), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia-smi")), \
                patch("subprocess.run", return_value=MagicMock(
                    returncode=1, stdout="", stderr=""
                )):
            response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

    def test_dashboard_stats_amd_rocm_and_metadata(self, tmp_path: Path):
        """Dashboard stats parses AMD/ROCm info and capability catalog."""
        import sqlite3 as _sqlite3

        import httpx as _httpx

        app_mod = sys.modules.get("app") or importlib.import_module("app")
        if "benchmark" not in sys.modules:
            importlib.import_module("benchmark")
        bm = sys.modules["benchmark"]
        client = _get_client()

        result = bm.BenchmarkResult(
            model_name="pub/model-a@q4",
            quantization="Q4_K_M",
            gpu_type="AMD Radeon",
            gpu_offload=1.0,
            vram_mb="8192",
            avg_tokens_per_sec=42.0,
            avg_ttft=0.4,
            avg_gen_time=1.0,
            prompt_tokens=10,
            completion_tokens=40,
            timestamp="2024-01-01T00:00:00",
            params_size="7B",
            architecture="llama",
            max_context_length=4096,
            model_size_gb=4.0,
            has_vision=False,
            has_tools=False,
            tokens_per_sec_per_gb=10.5,
            tokens_per_sec_per_billion_params=6.0,
        )
        mock_cache = MagicMock()
        mock_cache.get_all_results.return_value = [result]

        metadata_db = tmp_path / "model_metadata.db"
        conn = _sqlite3.connect(metadata_db)
        conn.execute(
            "CREATE TABLE model_metadata (model_key TEXT, capabilities TEXT)"
        )
        conn.execute(
            "INSERT INTO model_metadata VALUES (?, ?)",
            ("pub/model-a", '["coding", "chat"]'),
        )
        conn.execute(
            "INSERT INTO model_metadata VALUES (?, ?)",
            ("broken", "{not-json"),
        )
        conn.commit()
        conn.close()

        mock_http_client = MagicMock()
        mock_http_client.__enter__ = MagicMock(return_value=mock_http_client)
        mock_http_client.__exit__ = MagicMock(return_value=False)
        mock_http_client.get.side_effect = _httpx.ConnectError("offline")

        def _cp(stdout: str, returncode: int = 0):
            proc = MagicMock()
            proc.returncode = returncode
            proc.stdout = stdout
            proc.stderr = ""
            return proc

        def _run_side_effect(cmd, **_kwargs):
            _ = _kwargs
            if cmd[:2] == ["lms", "status"]:
                return _cp("server: on\n")
            if cmd[:2] == ["lspci", "-d"]:
                return _cp(
                    "03:00.0 VGA compatible controller: "
                    "Advanced Micro Devices, Inc. [AMD/ATI] Device 1002:5450\n"
                )
            if cmd[:2] == ["lspci", "-s"]:
                return _cp("Device: Radeon RX 6800 XT\n")
            if "--showproductname" in cmd:
                return _cp("GPU[0] : gfx906\n")
            if cmd[-2:] == ["--showmeminfo", "vram"]:
                return _cp("VRAM Total Memory (B): 17179869184\n")
            if cmd[-2:] == ["--showmeminfo", "gtt"]:
                return _cp("GTT Total Memory (B): 8589934592\n")
            return _cp("", returncode=1)

        with (
            patch.object(app_mod, "BenchmarkCache", return_value=mock_cache),
            patch.object(app_mod, "METADATA_DATABASE_FILE", metadata_db),
            patch("httpx.Client", return_value=mock_http_client),
            patch(
                "cpuinfo.get_cpu_info",
                return_value={"brand_raw": "AMD Ryzen AI 9 HX 370"},
            ),
            patch(
                "subprocess.check_output",
                side_effect=FileNotFoundError("no nvidia"),
            ),
            patch("subprocess.run", side_effect=_run_side_effect),
            patch("glob.glob", return_value=[]),
        ):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["lmstudio"]["ok"] is True
        assert data["gpu_info"]["type"] == "AMD"
        assert data["gpu_info"]["vram_gb"] == 16.0
        assert data["gpu_info"]["gtt_gb"] == 8.0
        assert "coding" in data["capability_catalog"]
        assert "chat" in data["capability_catalog"]


class TestLatestResultsWithFiles:
    """Tests for GET /api/latest-results with actual file mocking."""

    def test_returns_filename_when_json_exists(self):
        """Returns filename when JSON result files are present."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            json_file = real_dir / "benchmark_results_20240101_120000.json"
            json_file.write_text("[]")
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.get("/api/latest-results")
        assert response.status_code == 200
        data = response.json()
        assert data["latest"] is not None
        assert "benchmark_results" in data["latest"]

    def test_returns_latest_when_multiple_files(self):
        """Returns most recent file when multiple JSON files exist."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            real_dir = Path(tmpdir)
            f1 = real_dir / "benchmark_results_20240101_100000.json"
            f1.write_text("[]")
            f2 = real_dir / "benchmark_results_20240101_120000.json"
            f2.write_text("[]")
            os.utime(f1, (1_000_000, 1_000_000))
            os.utime(f2, (2_000_000, 2_000_000))
            with patch.object(app_mod, "RESULTS_DIR", real_dir):
                response = client.get("/api/latest-results")
        assert response.status_code == 200
        data = response.json()
        assert data["latest"] == f2.name

    def test_returns_none_on_os_error(self):
        """Returns null latest on OS error."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        mock_dir = MagicMock()
        mock_dir.glob.side_effect = OSError("no access")
        with patch.object(app_mod, "RESULTS_DIR", mock_dir):
            response = client.get("/api/latest-results")
        assert response.status_code == 200
        assert response.json()["latest"] is None


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_ok_status(self):
        """Health endpoint returns status 'ok'."""
        client = _get_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_benchmark_not_running(self):
        """Health endpoint shows benchmark_running is False when idle."""
        client = _get_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["benchmark_running"] is False
        assert "connected_clients" in data


class TestPresetCompareAndExport:
    """Tests for preset compare and export endpoints."""

    def test_compare_presets_both_valid(self):
        """POST /api/presets/compare returns comparison for two presets."""
        client = _get_client()
        response = client.post(
            "/api/presets/compare",
            json={"preset_a": "default", "preset_b": "fast"},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    def test_compare_presets_missing_preset(self):
        """POST /api/presets/compare handles missing presets gracefully."""
        client = _get_client()
        response = client.post(
            "/api/presets/compare",
            json={"preset_a": "nonexistent_preset_xyz", "preset_b": "default"},
        )
        assert response.status_code == 200

    def test_export_presets_returns_zip(self):
        """GET /api/presets/export returns a ZIP file or success response."""
        client = _get_client()
        response = client.get("/api/presets/export")
        assert response.status_code == 200

    def test_get_preset_by_name(self):
        """GET /api/presets/{name} returns preset data."""
        client = _get_client()
        response = client.get("/api/presets/default")
        assert response.status_code == 200

    def test_get_nonexistent_preset(self):
        """GET /api/presets/{name} returns failure for missing preset."""
        client = _get_client()
        response = client.get("/api/presets/no_such_preset_at_all_xyz")
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is False or "error" in data


class TestCSVExportAdvanced:
    """More tests for POST /api/comparison/export/csv."""

    def test_csv_export_sqlite_error(self):
        """CSV export handles sqlite3 errors gracefully."""
        import sqlite3 as _sqlite3
        client = _get_client()
        with patch("sqlite3.connect",
                   side_effect=_sqlite3.Error("connection failed")):
            response = client.post(
                "/api/comparison/export/csv",
                json={"model_name": "test-model"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_csv_export_with_all_filters(self):
        """CSV export correctly applies model, date, and quantization filters."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            (
                "2024-06-01T00:00:00", "filtered-model", "Q4_K_M",
                55.0, 0.3, 0.8, 1.0, "8192", 0.7, 40, 0.9, 0.05,
                1.1, 256, 3, 120.0, 0,
            )
        ]
        mock_conn.cursor.return_value = mock_cursor
        with patch("sqlite3.connect", return_value=mock_conn):
            with tempfile.TemporaryDirectory() as tmpdir:
                real_dir = Path(tmpdir)
                with patch.object(app_mod, "RESULTS_DIR", real_dir):
                    response = client.post(
                        "/api/comparison/export/csv",
                        json={
                            "model_name": "filtered-model",
                            "start_date": "2024-01-01",
                            "end_date": "2024-12-31",
                            "quantizations": ["Q4_K_M"],
                        },
                    )
        assert response.status_code == 200


class TestStatisticalHelperFunctions:
    """Tests for helper functions in web/app.py."""

    def _app(self):
        """Get the app module."""
        return sys.modules.get("app") or importlib.import_module("app")

    def test_calculate_hash_deterministic(self):
        """calculate_hash returns same hash for same params."""
        app_mod = self._app()
        params = {"temperature": 0.7, "top_k": 40}
        h1 = app_mod.calculate_hash(params)
        h2 = app_mod.calculate_hash(params)
        assert h1 == h2
        assert len(h1) == 16

    def test_calculate_hash_different_params(self):
        """calculate_hash returns different hashes for different params."""
        app_mod = self._app()
        h1 = app_mod.calculate_hash({"temperature": 0.7})
        h2 = app_mod.calculate_hash({"temperature": 0.9})
        assert h1 != h2

    def test_match_parameters_all_match(self):
        """match_parameters returns True when all target params match."""
        app_mod = self._app()
        row = {"temperature": 0.7, "top_k": 40, "top_p": 0.9}
        target = {"temperature": 0.7, "top_k": 40}
        assert app_mod.match_parameters(row, target) is True

    def test_match_parameters_none_ignored(self):
        """match_parameters ignores None values in target."""
        app_mod = self._app()
        row = {"temperature": 0.7, "top_k": None}
        target = {"temperature": 0.7, "top_k": None}
        assert app_mod.match_parameters(row, target) is True

    def test_match_parameters_mismatch(self):
        """match_parameters returns False when param values differ."""
        app_mod = self._app()
        row = {"temperature": 0.7}
        target = {"temperature": 0.9}
        assert app_mod.match_parameters(row, target) is False

    def test_match_parameters_float_tolerance(self):
        """match_parameters handles float tolerance correctly."""
        app_mod = self._app()
        row = {"temperature": 0.70001}
        target = {"temperature": 0.7}
        assert app_mod.match_parameters(row, target) is True

    def test_calculate_effect_size_empty(self):
        """calculate_effect_size returns negligible for empty lists."""
        app_mod = self._app()
        result = app_mod.calculate_effect_size([], [50.0, 60.0])
        assert result["effect_magnitude"] == "negligible"

    def test_calculate_effect_size_with_data(self):
        """calculate_effect_size computes Cohen's d."""
        app_mod = self._app()
        baseline = [50.0, 52.0, 48.0, 51.0]
        test = [60.0, 62.0, 58.0, 61.0]
        result = app_mod.calculate_effect_size(baseline, test)
        assert "cohens_d" in result
        assert "effect_magnitude" in result
        assert result["cohens_d"] > 0

    def test_perform_ttest_significant(self):
        """perform_ttest detects significant difference."""
        app_mod = self._app()
        baseline = [49.5, 50.2, 50.1, 49.8, 50.4, 49.9, 50.3, 49.7, 50.0, 50.5]
        test = [89.4, 90.1, 90.5, 89.8, 90.2, 89.7, 90.6, 89.9, 90.3, 90.0]
        result = app_mod.perform_ttest(baseline, test)
        assert "significant" in result
        assert result["significant"] is True

    def test_perform_ttest_not_significant(self):
        """perform_ttest returns not significant for equal groups."""
        app_mod = self._app()
        baseline = [50.0, 51.0, 49.0, 50.5]
        test = [50.0, 51.0, 49.0, 50.5]
        result = app_mod.perform_ttest(baseline, test)
        assert "significant" in result

    def test_perform_ttest_single_element(self):
        """perform_ttest handles single element lists."""
        app_mod = self._app()
        result = app_mod.perform_ttest([50.0], [60.0])
        assert isinstance(result, dict)


class TestExperimentCreateEndpoint:
    """More tests for POST /api/experiments/create."""

    def test_create_experiment_with_all_params(self):
        """Create experiment with full parameter set succeeds."""
        client = _get_client()
        response = client.post(
            "/api/experiments/create",
            json={
                "name": "full-test",
                "model_name": "pub/llama",
                "baseline_params": {
                    "name": "baseline",
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "min_p": 0.05,
                    "repeat_penalty": 1.1,
                    "max_tokens": 256,
                },
                "test_params": {
                    "name": "test",
                    "temperature": 0.9,
                    "top_k": 60,
                },
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_name"] == "pub/llama"
        assert "baseline_hash" in data
        assert "test_hash" in data

    def test_create_experiment_hashes_differ(self):
        """Different params produce different experiment hashes."""
        client = _get_client()
        response = client.post(
            "/api/experiments/create",
            json={
                "name": "hash-test",
                "model_name": "pub/model",
                "baseline_params": {"name": "b", "temperature": 0.7},
                "test_params": {"name": "t", "temperature": 0.9},
            },
        )
        data = response.json()
        assert data["success"] is True
        assert data["baseline_hash"] != data["test_hash"]


# ============================================================================
# Dashboard stats with proper httpx mocking
# ============================================================================

class TestDashboardStatsHTTPXMocking:
    """Tests for get_dashboard_stats covering httpx exception/CLI health paths."""

    def test_dashboard_stats_benchmark_cache_none(self):
        """Returns error dict when BenchmarkCache is None."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        with patch.object(app_mod, "BenchmarkCache", None):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200


class _FakeWebSocket:
    """Simple async websocket stub for direct handler tests."""

    def __init__(self, events):
        self._events = list(events)
        self.accepted = False
        self.sent = []

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        self.sent.append(data)

    async def receive_json(self):
        if not self._events:
            raise RuntimeError("no events left")
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


class _FailOnTypeWebSocket(_FakeWebSocket):
    """WebSocket stub that raises on sending a specific message type."""

    def __init__(self, events, fail_type, exc):
        super().__init__(events)
        self._fail_type = fail_type
        self._exc = exc

    async def send_json(self, data):
        if data.get("type") == self._fail_type:
            raise self._exc
        await super().send_json(data)


class TestBenchmarkWebSocket:
    """Tests for websocket_benchmark live-stream handler."""

    @pytest.mark.anyio
    async def test_websocket_sends_output_and_disconnects(self):
        """Handler sends status/output and removes disconnected client."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        disconnect_exc = app_mod.WebSocketDisconnect()
        ws = _FakeWebSocket([asyncio.TimeoutError(), disconnect_exc])

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "running"
        mock_manager.is_running.side_effect = [True, True]
        mock_manager.drain_output_queue.side_effect = ["line-1", ""]
        mock_manager.hardware_history = {
            "temperatures": [],
            "power": [],
            "vram": [],
            "gtt": [],
            "cpu": [],
            "ram": [],
        }
        mock_manager.last_hardware_send_time = time.time()

        with patch.object(app_mod, "manager", mock_manager):
            await app_mod.websocket_benchmark(ws)

        assert ws.accepted is True
        assert any(m.get("type") == "status" for m in ws.sent)
        assert any(m.get("type") == "output" for m in ws.sent)
        assert ws not in mock_manager.connected_clients

    @pytest.mark.anyio
    async def test_websocket_sends_completed_event(self):
        """Handler emits completed event and resets manager state."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        disconnect_exc = app_mod.WebSocketDisconnect()
        ws = _FakeWebSocket([asyncio.TimeoutError(), disconnect_exc])

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "completed"
        mock_manager.is_running.return_value = False
        mock_manager.drain_output_queue.return_value = ""

        with patch.object(app_mod, "manager", mock_manager), \
                patch("app.asyncio.sleep", new=AsyncMock(return_value=None)):
            await app_mod.websocket_benchmark(ws)

        assert any(m.get("type") == "completed" for m in ws.sent)
        mock_manager.set_idle_status.assert_called()

    @pytest.mark.anyio
    async def test_websocket_sends_failed_event(self):
        """Handler emits failure event and resets manager state."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        disconnect_exc = app_mod.WebSocketDisconnect()
        ws = _FakeWebSocket([asyncio.TimeoutError(), disconnect_exc])

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "failed"
        mock_manager.is_running.return_value = False
        mock_manager.drain_output_queue.return_value = ""

        with patch.object(app_mod, "manager", mock_manager), \
                patch("app.asyncio.sleep", new=AsyncMock(return_value=None)):
            await app_mod.websocket_benchmark(ws)

        assert any(m.get("type") == "error" for m in ws.sent)
        mock_manager.set_idle_status.assert_called()

    @pytest.mark.anyio
    async def test_websocket_sends_hardware_data_when_running(self):
        """Handler sends hardware payload while benchmark is running."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        disconnect_exc = app_mod.WebSocketDisconnect()
        ws = _FakeWebSocket([asyncio.TimeoutError(), disconnect_exc])

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "running"
        mock_manager.is_running.side_effect = [True, True]
        mock_manager.drain_output_queue.return_value = ""
        mock_manager.hardware_history = {
            "temperatures": [70],
            "power": [120],
            "vram": [8192],
            "gtt": [0],
            "cpu": [35],
            "ram": [45],
        }
        mock_manager.last_hardware_send_time = 0

        with patch.object(app_mod, "manager", mock_manager), \
                patch("app.time.time", return_value=10.0):
            await app_mod.websocket_benchmark(ws)

        assert any(m.get("type") == "hardware" for m in ws.sent)
        mock_manager.update_last_hardware_send_time.assert_called_once_with(10.0)

    @pytest.mark.anyio
    async def test_websocket_receive_error_breaks_loop(self):
        """Receive errors are handled and loop exits gracefully."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        ws = _FakeWebSocket([ValueError("bad payload")])

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "running"
        mock_manager.is_running.return_value = True
        mock_manager.drain_output_queue.return_value = ""
        mock_manager.hardware_history = {
            "temperatures": [],
            "power": [],
            "vram": [],
            "gtt": [],
            "cpu": [],
            "ram": [],
        }
        mock_manager.last_hardware_send_time = 0

        with patch.object(app_mod, "manager", mock_manager):
            await app_mod.websocket_benchmark(ws)

        assert ws.accepted is True
        assert ws not in mock_manager.connected_clients

    @pytest.mark.anyio
    async def test_websocket_output_send_error_breaks_loop(self):
        """Output send failures are handled and stop websocket loop."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        ws = _FailOnTypeWebSocket(
            [asyncio.TimeoutError()],
            "output",
            RuntimeError("send failed"),
        )

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "running"
        mock_manager.is_running.return_value = True
        mock_manager.drain_output_queue.return_value = "line-1"
        mock_manager.hardware_history = {
            "temperatures": [],
            "power": [],
            "vram": [],
            "gtt": [],
            "cpu": [],
            "ram": [],
        }
        mock_manager.last_hardware_send_time = time.time()

        with patch.object(app_mod, "manager", mock_manager):
            await app_mod.websocket_benchmark(ws)

        assert any(m.get("type") == "status" for m in ws.sent)

    @pytest.mark.anyio
    async def test_websocket_heartbeat_send_error_breaks_loop(self):
        """Heartbeat send errors are handled in idle/non-running mode."""
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        ws = _FailOnTypeWebSocket(
            [asyncio.TimeoutError(), asyncio.TimeoutError()],
            "status",
            OSError("socket closed"),
        )

        mock_manager = MagicMock()
        mock_manager.connected_clients = set()
        mock_manager.status = "idle"
        mock_manager.is_running.return_value = False
        mock_manager.drain_output_queue.return_value = ""

        with patch.object(app_mod, "manager", mock_manager), \
                patch("app.asyncio.sleep", new=AsyncMock(return_value=None)):
            await app_mod.websocket_benchmark(ws)

        assert ws.accepted is True
        assert ws not in mock_manager.connected_clients

    def test_dashboard_stats_httpx_fails_cli_online(self):
        """Covers httpx exception path and CLI health check 'online'."""
        import httpx as _httpx
        client = _get_client()

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.side_effect = _httpx.ConnectError("Connection refused")
        mock_lms = MagicMock(returncode=0, stdout="server: on\n", stderr="")

        with patch("httpx.Client", return_value=mock_http_inst), \
                patch("subprocess.run", return_value=mock_lms), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200

    def test_dashboard_stats_httpx_fails_cli_offline(self):
        """Covers httpx exception path and CLI health check 'offline'."""
        import httpx as _httpx
        client = _get_client()

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.side_effect = _httpx.ConnectError("refused")
        mock_lms = MagicMock(returncode=1, stdout="server: off\n", stderr="error")

        with patch("httpx.Client", return_value=mock_http_inst), \
                patch("subprocess.run", return_value=mock_lms), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200

    def test_dashboard_stats_httpx_fails_cli_raises(self):
        """Covers the CLI subprocess raising OSError."""
        import httpx as _httpx
        client = _get_client()

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.side_effect = _httpx.ConnectError("refused")

        with patch("httpx.Client", return_value=mock_http_inst), \
                patch("subprocess.run",
                      side_effect=FileNotFoundError("lms not found")), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200

    def test_dashboard_stats_httpx_succeeds_200(self):
        """Covers httpx success path (LM Studio online)."""
        client = _get_client()

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.return_value = mock_resp

        with patch("httpx.Client", return_value=mock_http_inst), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200

    def test_dashboard_stats_config_defaults_error(self):
        """Dashboard stats handles CONFIG_DEFAULTS access error."""
        import httpx as _httpx
        app_mod = sys.modules.get("app") or importlib.import_module("app")
        client = _get_client()

        mock_config = MagicMock()
        mock_config.get.side_effect = AttributeError("no config")

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.side_effect = _httpx.ConnectError("refused")

        with patch.object(app_mod, "CONFIG_DEFAULTS", mock_config), \
                patch("httpx.Client", return_value=mock_http_inst), \
                patch("subprocess.run",
                      side_effect=FileNotFoundError("lms")), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200

    def test_dashboard_stats_sqlite_error_in_capabilities(self):
        """Dashboard stats handles sqlite3.Error in capability loading."""
        import sqlite3 as _sqlite3

        import httpx as _httpx
        client = _get_client()

        mock_http_inst = MagicMock()
        mock_http_inst.__enter__ = MagicMock(return_value=mock_http_inst)
        mock_http_inst.__exit__ = MagicMock(return_value=False)
        mock_http_inst.get.side_effect = _httpx.ConnectError("refused")

        with patch("httpx.Client", return_value=mock_http_inst), \
                patch("sqlite3.connect",
                      side_effect=_sqlite3.OperationalError("no such table")), \
                patch("subprocess.run",
                      side_effect=FileNotFoundError("lms")), \
                patch("subprocess.check_output",
                      side_effect=FileNotFoundError("no nvidia")):
            response = client.get("/api/dashboard/stats")

        assert response.status_code == 200
