"""Comprehensive FastAPI endpoint tests for web/app.py."""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))

if "lmstudio" not in sys.modules:
    sys.modules["lmstudio"] = MagicMock()


def _get_client():
    """Return a TestClient for the FastAPI app."""
    if "app" in sys.modules:
        app_mod = sys.modules["app"]
    else:
        import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=False)


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
        with patch("os.kill") as mock_kill, \
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
        with patch.object(app_mod, "BenchmarkCache", side_effect=Exception("db error")):
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
            import benchmark
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
        app_mod = sys.modules["app"]
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
        app_mod = sys.modules["app"]
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
            import benchmark
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
        client = _get_client()
        app_mod = sys.modules["app"]
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
        client = _get_client()
        app_mod = sys.modules["app"]
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
            import benchmark
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
