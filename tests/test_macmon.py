"""Tests for tools/macmon.py (sudoless Apple Silicon power/temperature).

The JSON fixtures below are trimmed copies of real ``macmon pipe`` output
(macmon 0.8.0), so the tests do not need the binary installed.
"""

import json
import subprocess
import threading
from unittest.mock import MagicMock, patch

import tools.macmon as mm
from tools.macmon import MacmonSampler

SAMPLE = {
    "all_power": 0.5507163405418396,
    "ane_power": 0.0,
    "cpu_power": 0.2923669219017029,
    "gpu_active_ratio": 0.32043036818504333,
    "gpu_freq_mhz": 338,
    "gpu_power": 0.2583494186401367,
    "memory": {
        "ram_total": 68719476736,
        "ram_usage": 16828694528,
        "swap_total": 2147483648,
        "swap_usage": 1135280128,
    },
    "sys_power": 7.6835761070251465,
    "temp": {
        "cpu_temp_avg": 50.0324592590332,
        "gpu_temp_avg": 47.14311599731445,
    },
    "timestamp": "2026-07-31T02:28:40.726672+00:00",
}

NDJSON = json.dumps(SAMPLE) + "\n"


def _sampler_with(sample):
    """Build a sampler with a preloaded latest sample."""
    sampler = MacmonSampler.__new__(MacmonSampler)
    sampler._lock = threading.Lock()
    sampler._latest = sample
    return sampler


class TestFindMacmon:
    """Tests for find_macmon() and is_macmon_available()."""

    def test_returns_none_off_macos(self, monkeypatch):
        """Non-macOS platforms never look for the binary."""
        monkeypatch.setattr(mm, "IS_MACOS", False)
        with patch("shutil.which") as mock_which:
            assert mm.find_macmon() is None
        mock_which.assert_not_called()

    def test_finds_binary_on_path(self, monkeypatch):
        """A macmon on PATH is returned directly."""
        monkeypatch.setattr(mm, "IS_MACOS", True)
        with patch("shutil.which", return_value="/opt/homebrew/bin/macmon"):
            assert mm.find_macmon() == "/opt/homebrew/bin/macmon"

    def test_falls_back_to_homebrew_prefixes(self, monkeypatch):
        """A GUI-launched process may lack Homebrew on PATH."""
        monkeypatch.setattr(mm, "IS_MACOS", True)

        def fake_which(_name, path=None):
            if path == "/opt/homebrew/bin":
                return "/opt/homebrew/bin/macmon"
            return None

        with patch("shutil.which", side_effect=fake_which):
            assert mm.find_macmon() == "/opt/homebrew/bin/macmon"

    def test_availability_reflects_lookup(self, monkeypatch):
        """is_macmon_available mirrors find_macmon."""
        monkeypatch.setattr(mm, "find_macmon", lambda: None)
        assert mm.is_macmon_available() is False
        monkeypatch.setattr(mm, "find_macmon", lambda: "/usr/local/bin/macmon")
        assert mm.is_macmon_available() is True


class TestMetricAccessors:
    """Tests for the typed accessors over a parsed sample."""

    def test_reads_temperatures(self):
        """GPU and CPU temperatures come from the nested temp object."""
        sampler = _sampler_with(SAMPLE)
        assert round(sampler.get_gpu_temperature(), 2) == 47.14
        assert round(sampler.get_cpu_temperature(), 2) == 50.03

    def test_reads_power_fields(self):
        """Power fields are returned in watts."""
        sampler = _sampler_with(SAMPLE)
        assert round(sampler.get_gpu_power(), 3) == 0.258
        assert round(sampler.get_cpu_power(), 3) == 0.292
        assert sampler.get_ane_power() == 0.0
        assert round(sampler.get_total_power(), 3) == 0.551

    def test_total_power_falls_back_to_sys_power(self):
        """When all_power is absent, sys_power is used."""
        sample = {k: v for k, v in SAMPLE.items() if k != "all_power"}
        sampler = _sampler_with(sample)
        assert round(sampler.get_total_power(), 2) == 7.68

    def test_gpu_utilization_is_percentage(self):
        """gpu_active_ratio is scaled from 0-1 to 0-100."""
        sampler = _sampler_with(SAMPLE)
        assert round(sampler.get_gpu_utilization(), 2) == 32.04

    def test_returns_none_without_sample(self):
        """Accessors return None before the first sample arrives."""
        sampler = _sampler_with(None)
        assert sampler.get_gpu_temperature() is None
        assert sampler.get_gpu_power() is None
        assert sampler.get_gpu_utilization() is None

    def test_returns_none_for_missing_keys(self):
        """A sample missing the requested field yields None."""
        sampler = _sampler_with({"temp": {}})
        assert sampler.get_gpu_temperature() is None
        assert sampler.get_gpu_power() is None

    def test_rejects_non_numeric_values(self):
        """Non-numeric values are not coerced."""
        sampler = _sampler_with(
            {"gpu_power": "hot", "temp": {"gpu_temp_avg": None}}
        )
        assert sampler.get_gpu_power() is None
        assert sampler.get_gpu_temperature() is None

    def test_rejects_bool_values(self):
        """Booleans are not treated as numbers."""
        sampler = _sampler_with({"gpu_power": True})
        assert sampler.get_gpu_power() is None


class TestSamplerLifecycle:
    """Tests for start/stop and the NDJSON reader loop."""

    def test_start_returns_false_without_binary(self, monkeypatch):
        """An unavailable macmon does not spawn a process."""
        monkeypatch.setattr(mm, "find_macmon", lambda: None)
        sampler = MacmonSampler()
        with patch("subprocess.Popen") as mock_popen:
            assert sampler.start() is False
        mock_popen.assert_not_called()

    def test_start_spawns_pipe_with_interval(self, monkeypatch):
        """macmon is launched in pipe mode with the configured interval."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler(interval_ms=500)

        proc = MagicMock()
        proc.stdout.__iter__.return_value = iter([])
        proc.poll.return_value = None

        with patch("subprocess.Popen", return_value=proc) as mock_popen:
            assert sampler.start() is True

        command = mock_popen.call_args[0][0]
        assert command[0] == "/opt/homebrew/bin/macmon"
        assert command[1] == "pipe"
        assert "--interval" in command
        assert "500" in command
        sampler._running = False

    def test_start_handles_launch_failure(self, monkeypatch):
        """An OSError while spawning is reported as a failed start."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()
        with patch("subprocess.Popen", side_effect=OSError("boom")):
            assert sampler.start() is False

    def test_interval_has_a_floor(self, monkeypatch):
        """Absurdly small intervals are clamped."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        assert MacmonSampler(interval_ms=1).interval_ms == 100

    def test_read_loop_parses_ndjson(self, monkeypatch):
        """Each NDJSON line replaces the latest sample."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()

        proc = MagicMock()
        proc.stdout.__iter__.return_value = iter([NDJSON])
        sampler._process = proc
        sampler._running = True
        sampler._read_loop()

        assert round(sampler.get_gpu_temperature(), 2) == 47.14

    def test_read_loop_skips_malformed_lines(self, monkeypatch):
        """Unparsable lines are ignored, valid ones still land."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()

        proc = MagicMock()
        proc.stdout.__iter__.return_value = iter(["not json\n", "\n", NDJSON])
        sampler._process = proc
        sampler._running = True
        sampler._read_loop()

        assert round(sampler.get_gpu_power(), 3) == 0.258

    def test_read_loop_ignores_non_object_json(self, monkeypatch):
        """A JSON array is not a sample and is discarded."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()

        proc = MagicMock()
        proc.stdout.__iter__.return_value = iter(["[1, 2, 3]\n"])
        sampler._process = proc
        sampler._running = True
        sampler._read_loop()

        assert sampler.latest() is None

    def test_stop_terminates_process(self, monkeypatch):
        """stop() terminates a live process and clears state."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()

        proc = MagicMock()
        proc.poll.return_value = None
        sampler._process = proc
        sampler._running = True

        sampler.stop()

        proc.terminate.assert_called_once()
        assert sampler._process is None

    def test_stop_kills_when_terminate_times_out(self, monkeypatch):
        """A process that ignores terminate is killed."""
        monkeypatch.setattr(
            mm, "find_macmon", lambda: "/opt/homebrew/bin/macmon"
        )
        sampler = MacmonSampler()

        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(
            cmd="macmon", timeout=2
        )
        sampler._process = proc
        sampler._running = True

        sampler.stop()

        proc.kill.assert_called_once()

    def test_stop_is_safe_without_process(self):
        """stop() on a never-started sampler does nothing."""
        sampler = MacmonSampler.__new__(MacmonSampler)
        sampler._process = None
        sampler._thread = None
        sampler._running = False
        sampler.stop()
        assert sampler._process is None
