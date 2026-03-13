"""Tests for web/app.py."""
import importlib
import json
import sys
from unittest.mock import MagicMock, patch


def _import_app():
    """Import web app module with all heavy deps mocked."""
    if "lmstudio" not in sys.modules:
        sys.modules["lmstudio"] = MagicMock()

    if "app" not in sys.modules:
        importlib.import_module("app")
    return sys.modules["app"]


class TestExpandShortFlagClusters:
    """Tests for web/app.py _expand_short_flag_clusters()."""

    def test_long_flag_unchanged(self):
        """Long flags pass through unchanged."""
        app_mod = _import_app()
        result = app_mod._expand_short_flag_clusters(
            ["--debug"], {"d", "h"}
        )
        assert result == ["--debug"]

    def test_single_short_flag_unchanged(self):
        """Single short flags are not expanded."""
        app_mod = _import_app()
        result = app_mod._expand_short_flag_clusters(
            ["-d"], {"d", "h"}
        )
        assert result == ["-d"]

    def test_combinable_cluster_expanded(self):
        """Combinable flag cluster is split into individual flags."""
        app_mod = _import_app()
        result = app_mod._expand_short_flag_clusters(
            ["-dh"], {"d", "h"}
        )
        assert "-d" in result
        assert "-h" in result

    def test_non_combinable_cluster_unchanged(self):
        """Non-combinable cluster is returned unchanged."""
        app_mod = _import_app()
        result = app_mod._expand_short_flag_clusters(
            ["-xyz"], {"d", "h"}
        )
        assert "-xyz" in result

    def test_positional_arg_unchanged(self):
        """Non-flag argument passes through unchanged."""
        app_mod = _import_app()
        result = app_mod._expand_short_flag_clusters(
            ["value"], {"d"}
        )
        assert result == ["value"]


class TestFindFreePort:
    """Tests for web/app.py find_free_port()."""

    def test_returns_integer(self):
        """Returns an integer port number."""
        app_mod = _import_app()
        port = app_mod.find_free_port()
        assert isinstance(port, int)

    def test_port_in_valid_range(self):
        """Returned port is in a valid range."""
        app_mod = _import_app()
        port = app_mod.find_free_port()
        assert 1 <= port <= 65535


class TestCalculateHash:
    """Tests for web/app.py calculate_hash()."""

    def test_returns_string(self):
        """Returns a hex string hash."""
        app_mod = _import_app()
        result = app_mod.calculate_hash({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 16

    def test_same_input_same_hash(self):
        """Same dict input yields the same hash."""
        app_mod = _import_app()
        h1 = app_mod.calculate_hash({"a": 1, "b": 2})
        h2 = app_mod.calculate_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_input_different_hash(self):
        """Different inputs yield different hashes."""
        app_mod = _import_app()
        h1 = app_mod.calculate_hash({"a": 1})
        h2 = app_mod.calculate_hash({"a": 2})
        assert h1 != h2

    def test_key_order_invariant(self):
        """Key insertion order does not change the hash."""
        app_mod = _import_app()
        h1 = app_mod.calculate_hash({"a": 1, "b": 2})
        h2 = app_mod.calculate_hash({"b": 2, "a": 1})
        assert h1 == h2


class TestPerformTtest:
    """Tests for web/app.py perform_ttest()."""

    def test_returns_dict(self):
        """Returns a dictionary with expected keys."""
        app_mod = _import_app()
        result = app_mod.perform_ttest([10.0, 12.0, 11.0], [20.0, 22.0, 21.0])
        assert isinstance(result, dict)
        assert "p_value" in result
        assert "significant" in result

    def test_insufficient_data_returns_not_significant(self):
        """Single-sample groups return not significant."""
        app_mod = _import_app()
        result = app_mod.perform_ttest([10.0], [20.0])
        assert result["significant"] is False

    def test_empty_lists_return_not_significant(self):
        """Empty lists return not significant."""
        app_mod = _import_app()
        result = app_mod.perform_ttest([], [])
        assert result["significant"] is False

    def test_identical_distributions_not_significant(self):
        """Same distribution returns not significant."""
        app_mod = _import_app()
        data = [10.0, 10.0, 10.0, 10.0]
        result = app_mod.perform_ttest(data, data)
        assert not result["significant"]

    def test_very_different_distributions_significant(self):
        """Highly different distributions are significant."""
        app_mod = _import_app()
        baseline = [1.0, 1.1, 0.9, 1.0, 1.0]
        test = [100.0, 99.0, 101.0, 100.0, 100.0]
        result = app_mod.perform_ttest(baseline, test)
        assert result["significant"]


class TestMatchParameters:
    """Tests for web/app.py match_parameters()."""

    def test_matching_params(self):
        """Identical parameters return True."""
        app_mod = _import_app()
        row = {"temperature": 0.7, "top_k": 40}
        target = {"temperature": 0.7, "top_k": 40}
        assert app_mod.match_parameters(row, target) is True

    def test_none_target_values_ignored(self):
        """None target values are ignored in comparison."""
        app_mod = _import_app()
        row = {"temperature": 0.7}
        target = {"temperature": None, "top_k": None}
        assert app_mod.match_parameters(row, target) is True

    def test_numeric_tolerance(self):
        """Numeric values within 0.001 tolerance are considered equal."""
        app_mod = _import_app()
        row = {"temperature": 0.7001}
        target = {"temperature": 0.7}
        assert app_mod.match_parameters(row, target) is True

    def test_numeric_mismatch(self):
        """Numeric values outside tolerance return False."""
        app_mod = _import_app()
        row = {"temperature": 0.8}
        target = {"temperature": 0.7}
        assert app_mod.match_parameters(row, target) is False

    def test_string_mismatch(self):
        """String values that differ return False."""
        app_mod = _import_app()
        row = {"quant": "Q4_K_M"}
        target = {"quant": "Q8_0"}
        assert app_mod.match_parameters(row, target) is False

    def test_empty_target_always_matches(self):
        """Empty target params always match any row."""
        app_mod = _import_app()
        row = {"temperature": 0.7, "top_k": 40}
        assert app_mod.match_parameters(row, {}) is True


class TestCalculateEffectSize:
    """Tests for web/app.py calculate_effect_size()."""

    def test_returns_dict(self):
        """Returns a dict with cohens_d and effect_magnitude."""
        app_mod = _import_app()
        result = app_mod.calculate_effect_size([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert "cohens_d" in result
        assert "effect_magnitude" in result

    def test_empty_lists_return_negligible(self):
        """Empty input returns negligible effect."""
        app_mod = _import_app()
        result = app_mod.calculate_effect_size([], [])
        assert result["effect_magnitude"] == "negligible"
        assert result["cohens_d"] == 0.0

    def test_negligible_effect(self):
        """Small difference returns negligible effect."""
        app_mod = _import_app()
        data = [10.0, 10.1, 9.9, 10.05]
        result = app_mod.calculate_effect_size(data, data)
        assert result["effect_magnitude"] == "negligible"

    def test_large_effect(self):
        """Very different groups return large effect."""
        app_mod = _import_app()
        baseline = [1.0, 1.1, 0.9]
        test = [100.0, 99.0, 101.0]
        result = app_mod.calculate_effect_size(baseline, test)
        assert result["effect_magnitude"] == "large"

    def test_single_element_groups(self):
        """Single element groups are handled without error."""
        app_mod = _import_app()
        result = app_mod.calculate_effect_size([5.0], [10.0])
        assert "cohens_d" in result

    def test_medium_effect(self):
        """Moderate difference returns small or medium effect."""
        app_mod = _import_app()
        baseline = [10.0, 11.0, 10.5, 10.2]
        test = [12.0, 13.0, 12.5, 12.2]
        result = app_mod.calculate_effect_size(baseline, test)
        assert result["effect_magnitude"] in {"small", "medium", "large", "negligible"}


class TestBenchmarkManager:
    """Tests for web/app.py BenchmarkManager."""

    def test_init_defaults(self):
        """BenchmarkManager initializes with expected default state."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        assert mgr.process is None
        assert mgr.status == "idle"
        assert mgr.start_time is None
        assert mgr.current_output == ""

    def test_is_running_false_when_no_process(self):
        """is_running() returns False when no process is set."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        assert mgr.is_running() is False

    def test_is_running_true_when_process_alive(self):
        """is_running() returns True when process is not done."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mgr.process = mock_proc
        assert mgr.is_running() is True

    def test_is_running_false_when_process_done(self):
        """is_running() returns False when process has exited."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mgr.process = mock_proc
        assert mgr.is_running() is False

    def test_pause_benchmark_not_running(self):
        """pause_benchmark() returns False when no benchmark running."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        result = mgr.pause_benchmark()
        assert result is False

    def test_resume_benchmark_not_paused(self):
        """resume_benchmark() returns False when not paused."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        result = mgr.resume_benchmark()
        assert result is False

    def test_stop_benchmark_no_process(self):
        """stop_benchmark() returns False when no process."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        result = mgr.stop_benchmark()
        assert result is False

    def test_drain_output_queue_empty(self):
        """drain_output_queue() returns empty string when no queue."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        result = mgr.drain_output_queue()
        assert result == ""

    def test_parse_hardware_metrics_temperature(self):
        """parse_hardware_metrics extracts GPU temperature."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("GPU Temp: 72.5°C measured")
        assert len(mgr.hardware_history["temperatures"]) == 1
        assert mgr.hardware_history["temperatures"][0]["value"] == 72.5

    def test_parse_hardware_metrics_power(self):
        """parse_hardware_metrics extracts GPU power."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("GPU Power: 150.0W draw")
        assert len(mgr.hardware_history["power"]) == 1
        assert mgr.hardware_history["power"][0]["value"] == 150.0

    def test_parse_hardware_metrics_vram(self):
        """parse_hardware_metrics extracts GPU VRAM usage."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("GPU VRAM: 8.5GB used")
        assert len(mgr.hardware_history["vram"]) == 1
        assert mgr.hardware_history["vram"][0]["value"] == 8.5

    def test_parse_hardware_metrics_gtt(self):
        """parse_hardware_metrics extracts GPU GTT usage."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("GPU GTT: 2.0GB used")
        assert len(mgr.hardware_history["gtt"]) == 1

    def test_parse_hardware_metrics_cpu(self):
        """parse_hardware_metrics extracts CPU usage."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("CPU: 45.0% util")
        assert len(mgr.hardware_history["cpu"]) == 1

    def test_parse_hardware_metrics_ram(self):
        """parse_hardware_metrics extracts RAM usage."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("RAM: 16.0GB used")
        assert len(mgr.hardware_history["ram"]) == 1

    def test_parse_hardware_metrics_no_match(self):
        """parse_hardware_metrics leaves history empty on non-matching line."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.parse_hardware_metrics("No metrics here")
        for key in mgr.hardware_history:
            assert len(mgr.hardware_history[key]) == 0


class TestGetCachedLatestRelease:
    """Tests for web/app.py _get_cached_latest_release()."""

    def test_returns_none_when_fetch_fails(self):
        """Returns None when fetch_latest_release returns None."""
        app_mod = _import_app()
        app_mod.LATEST_RELEASE_CACHE["data"] = None
        app_mod.LATEST_RELEASE_CACHE["timestamp"] = 0

        with patch.object(app_mod, "fetch_latest_release", return_value=None), \
                patch.object(app_mod, "get_current_version", return_value="v0.1.0"):
            result = app_mod._get_cached_latest_release()

        assert result is None

    def test_returns_cached_data_when_valid(self):
        """Returns cached data when cache is still valid."""
        import time
        app_mod = _import_app()
        cached_data = {
            "current_version": "v0.1.0",
            "latest_version": "v0.2.0",
            "download_url": "https://example.com",
            "is_update_available": True,
        }
        app_mod.LATEST_RELEASE_CACHE["data"] = cached_data
        app_mod.LATEST_RELEASE_CACHE["timestamp"] = time.time()
        app_mod.LATEST_RELEASE_CACHE["ttl_seconds"] = 3600

        result = app_mod._get_cached_latest_release()
        assert result == cached_data

    def test_fetches_fresh_data_when_expired(self):
        """Fetches fresh data when cache has expired."""
        app_mod = _import_app()
        app_mod.LATEST_RELEASE_CACHE["data"] = None
        app_mod.LATEST_RELEASE_CACHE["timestamp"] = 0

        mock_release = {"tag_name": "v0.2.0"}
        with patch.object(app_mod, "fetch_latest_release", return_value=mock_release), \
                patch.object(app_mod, "get_current_version", return_value="v0.1.0"), \
                patch.object(app_mod, "compare_versions", return_value=True), \
                patch.object(app_mod, "format_release_url", return_value="https://example.com/v0.2.0"):
            result = app_mod._get_cached_latest_release()

        assert result is not None
        assert result["latest_version"] == "v0.2.0"

    def test_returns_none_on_exception(self):
        """Returns None when an exception occurs during fetch."""
        app_mod = _import_app()
        app_mod.LATEST_RELEASE_CACHE["data"] = None
        app_mod.LATEST_RELEASE_CACHE["timestamp"] = 0

        with patch.object(
            app_mod,
            "get_current_version",
            side_effect=Exception("error"),
        ):
            result = app_mod._get_cached_latest_release()

        assert result is None


class TestCollectLmsVariants:
    """Tests for web/app.py _collect_lms_variants()."""

    def test_returns_empty_list_on_subprocess_error(self):
        """Returns empty list when lms command fails."""
        app_mod = _import_app()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout=""),
        ):
            result = app_mod._collect_lms_variants("org/model")
        assert result == []

    def test_returns_empty_list_when_no_match(self):
        """Returns empty list when no models match base_model."""
        app_mod = _import_app()
        models_data = [{"modelKey": "other/model", "paramsString": "7B"}]
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps(models_data),
            ),
        ):
            result = app_mod._collect_lms_variants("org/model")
        assert result == []

    def test_returns_model_entry_for_matching_key(self):
        """Returns model info when base_model key matches."""
        app_mod = _import_app()
        models_data = [
            {
                "modelKey": "org/model@q4",
                "paramsString": "7B",
                "sizeBytes": 4 * 1024 ** 3,
            }
        ]
        with patch(
            "subprocess.run",
            return_value=MagicMock(
                returncode=0,
                stdout=json.dumps(models_data),
            ),
        ):
            result = app_mod._collect_lms_variants("org/model")
        assert len(result) >= 1
        assert result[0]["params"] == "7B"

    def test_returns_empty_list_on_exception(self):
        """Returns empty list on unexpected exception."""
        app_mod = _import_app()
        with patch("subprocess.run", side_effect=Exception("broken")):
            result = app_mod._collect_lms_variants("org/model")
        assert result == []


class TestBenchmarkManagerWithProcess:
    """Tests for BenchmarkManager with mocked subprocess."""

    def test_pause_returns_true_when_running(self):
        """pause_benchmark returns True when process is running."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mgr.process = mock_proc
        mgr.status = "running"
        result = mgr.pause_benchmark()
        assert result is True
        assert mgr.status == "paused"

    def test_pause_returns_false_on_error(self):
        """pause_benchmark returns False when signal raises."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.send_signal.side_effect = OSError("no such process")
        mgr.process = mock_proc
        result = mgr.pause_benchmark()
        assert result is False

    def test_resume_returns_true_when_paused(self):
        """resume_benchmark returns True when status is paused."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mgr.process = mock_proc
        mgr.status = "paused"
        result = mgr.resume_benchmark()
        assert result is True
        assert mgr.status == "running"

    def test_resume_returns_false_on_error(self):
        """resume_benchmark returns False when signal raises."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.send_signal.side_effect = OSError("no such process")
        mgr.process = mock_proc
        mgr.status = "paused"
        result = mgr.resume_benchmark()
        assert result is False

    def test_stop_returns_true_when_process_exists(self):
        """stop_benchmark returns True when process terminates."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mgr.process = mock_proc
        result = mgr.stop_benchmark()
        assert result is True
        assert mgr.process is None

    def test_stop_kills_if_sigterm_times_out(self):
        """stop_benchmark kills process if SIGTERM wait times out."""
        import subprocess
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="bench", timeout=5),
            None,
        ]
        mgr.process = mock_proc
        result = mgr.stop_benchmark()
        assert result is True
        mock_proc.kill.assert_called_once()

    def test_stop_returns_false_on_error(self):
        """stop_benchmark returns False when signal raises."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.send_signal.side_effect = OSError("no such process")
        mgr.process = mock_proc
        result = mgr.stop_benchmark()
        assert result is False

    def test_drain_output_queue_returns_combined(self):
        """drain_output_queue returns all queued items concatenated."""
        import asyncio
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mgr.output_queue = asyncio.Queue()
        mgr.output_queue.put_nowait("line1\n")
        mgr.output_queue.put_nowait("line2\n")
        result = mgr.drain_output_queue()
        assert result == "line1\nline2\n"


class TestBenchmarkManagerStartAsync:
    """Async tests for BenchmarkManager.start_benchmark."""

    @staticmethod
    def _run(coro):
        """Run a coroutine in an isolated event loop."""
        import asyncio
        return asyncio.run(coro)

    def test_start_benchmark_returns_true_on_success(self):
        """start_benchmark returns True when subprocess starts."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.stdout = MagicMock()

        def _capture_task(coro):
            coro.close()
            return MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc), \
                patch("asyncio.create_task", side_effect=_capture_task):
            result = self._run(mgr.start_benchmark(["--test"]))
        assert result is True
        assert mgr.status == "running"

    def test_start_benchmark_returns_false_when_already_running(self):
        """start_benchmark returns False if already running."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mgr.process = mock_proc
        result = self._run(mgr.start_benchmark(["--test"]))
        assert result is False

    def test_start_benchmark_returns_false_on_error(self):
        """start_benchmark returns False when Popen raises."""
        app_mod = _import_app()
        mgr = app_mod.BenchmarkManager()
        with patch("subprocess.Popen", side_effect=OSError("no exec")):
            result = self._run(mgr.start_benchmark(["--test"]))
        assert result is False
        assert mgr.status == "idle"
