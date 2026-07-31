"""Tests for core/platform_info.py (cross-platform OS and GPU detection)."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import core.platform_info as pi


SYSTEM_PROFILER_APPLE = json.dumps(
    {
        "SPDisplaysDataType": [
            {
                "_name": "Apple M4 Max",
                "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
                "spdisplays_vendor": "sppci_vendor_Apple",
                "sppci_bus": "spdisplays_builtin",
                "sppci_cores": "40",
                "sppci_device_type": "spdisplays_gpu",
                "sppci_model": "Apple M4 Max",
            }
        ]
    }
)

IOREG_OUTPUT = (
    '  | |   "PerformanceStatistics" = {"In use system memory (driver)"=0,'
    '"Alloc system memory"=52967440384,"Tiler Utilization %"=10,'
    '"recoveryCount"=0,"Device Utilization %"=16,'
    '"In use system memory"=2363801600}\n'
)


def _ok(stdout: str) -> MagicMock:
    """Build a successful CompletedProcess-like mock."""
    return MagicMock(returncode=0, stdout=stdout)


class TestRunTextCommand:
    """Tests for _run_text_command()."""

    def test_returns_stdout_on_success(self):
        """Successful command returns its stdout."""
        with patch("subprocess.run", return_value=_ok("value\n")):
            assert pi._run_text_command(["true"]) == "value\n"

    def test_returns_none_on_missing_binary(self):
        """A missing binary yields None instead of raising."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert pi._run_text_command(["nope"]) is None

    def test_returns_none_on_timeout(self):
        """A timeout yields None instead of raising."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5),
        ):
            assert pi._run_text_command(["slow"]) is None

    def test_returns_none_on_nonzero_exit(self):
        """Non-zero exit codes yield None."""
        with patch(
            "subprocess.run", return_value=MagicMock(returncode=1, stdout="")
        ):
            assert pi._run_text_command(["false"]) is None

    def test_returns_none_on_blank_stdout(self):
        """Whitespace-only stdout is treated as no result."""
        with patch("subprocess.run", return_value=_ok("   \n")):
            assert pi._run_text_command(["blank"]) is None


class TestMacosNameVersion:
    """Tests for get_macos_name_version()."""

    def test_maps_known_major_release(self):
        """A known major release gains its marketing name."""
        with patch(
            "platform.mac_ver", return_value=("26.6", ("", "", ""), "arm64")
        ):
            assert pi.get_macos_name_version() == ("macOS Tahoe", "26.6")

    def test_maps_legacy_ten_dot_release(self):
        """Legacy 10.x versions resolve on the two-part key."""
        with patch(
            "platform.mac_ver", return_value=("10.15.7", ("", "", ""), "x86_64")
        ):
            assert pi.get_macos_name_version() == ("macOS Catalina", "10.15.7")

    def test_unknown_release_falls_back_to_plain_name(self):
        """An unmapped release still reports a usable name and version."""
        with patch(
            "platform.mac_ver", return_value=("99.0", ("", "", ""), "arm64")
        ):
            assert pi.get_macos_name_version() == ("macOS", "99.0")

    def test_empty_mac_ver_falls_back_to_release(self):
        """An empty mac_ver falls back to platform.release()."""
        with patch("platform.mac_ver", return_value=("", ("", "", ""), "")), \
                patch("platform.release", return_value="25.6.0"):
            name, version = pi.get_macos_name_version()
        assert name == "macOS"
        assert version == "25.6.0"


class TestGetOsNameVersion:
    """Tests for get_os_name_version()."""

    def test_darwin_returns_macos_name(self):
        """Darwin is reported as macOS, not as the Darwin kernel version."""
        mac_ver = ("26.6", ("", "", ""), "arm64")
        with patch("platform.system", return_value="Darwin"), \
                patch("platform.mac_ver", return_value=mac_ver):
            assert pi.get_os_name_version() == ("macOS Tahoe", "26.6")

    def test_other_platform_uses_platform_values(self):
        """Unknown systems fall back to platform name and release."""
        with patch("platform.system", return_value="FreeBSD"), \
                patch("platform.release", return_value="14.0"):
            assert pi.get_os_name_version() == ("FreeBSD", "14.0")


class TestOpenUrlCommand:
    """Tests for get_open_url_command()."""

    def test_macos_uses_open(self, monkeypatch):
        """macOS uses the `open` command."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        monkeypatch.setattr(pi, "IS_LINUX", False)
        assert pi.get_open_url_command() == "open"

    def test_linux_uses_xdg_open(self, monkeypatch):
        """Linux uses `xdg-open`."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        monkeypatch.setattr(pi, "IS_LINUX", True)
        assert pi.get_open_url_command() == "xdg-open"


class TestDetectAppleGpu:
    """Tests for detect_apple_gpu()."""

    def test_returns_none_off_macos(self, monkeypatch):
        """Non-macOS platforms never probe system_profiler."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        with patch("subprocess.run") as mock_run:
            assert pi.detect_apple_gpu() is None
        mock_run.assert_not_called()

    def test_parses_apple_silicon_gpu(self, monkeypatch):
        """Apple Silicon GPU fields are extracted from system_profiler JSON."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok(SYSTEM_PROFILER_APPLE)):
            info = pi.detect_apple_gpu()

        assert info == {
            "model": "Apple M4 Max",
            "cores": 40,
            "metal_family": "metal4",
            "vendor": "Apple",
        }

    def test_handles_missing_core_count(self, monkeypatch):
        """A GPU entry without a core count still resolves."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        payload = json.dumps(
            {
                "SPDisplaysDataType": [
                    {
                        "sppci_model": "AMD Radeon Pro 5500M",
                        "spdisplays_vendor": "sppci_vendor_amd",
                    }
                ]
            }
        )
        with patch("subprocess.run", return_value=_ok(payload)):
            info = pi.detect_apple_gpu()

        assert info["model"] == "AMD Radeon Pro 5500M"
        assert info["cores"] is None
        assert info["vendor"] == "amd"

    def test_returns_none_on_invalid_json(self, monkeypatch):
        """Malformed system_profiler output is handled gracefully."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok("not json")):
            assert pi.detect_apple_gpu() is None

    def test_returns_none_when_command_missing(self, monkeypatch):
        """A missing system_profiler yields None."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert pi.detect_apple_gpu() is None


class TestReadAppleGpuStats:
    """Tests for read_apple_gpu_stats()."""

    def test_parses_ioreg_performance_statistics(self, monkeypatch):
        """Utilization and in-use memory are read from ioreg."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok(IOREG_OUTPUT)):
            stats = pi.read_apple_gpu_stats()

        assert stats["utilization_percent"] == 16.0
        assert round(stats["vram_used_gb"], 2) == 2.2

    def test_returns_empty_stats_off_macos(self, monkeypatch):
        """Non-macOS platforms return empty stats without probing."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        with patch("subprocess.run") as mock_run:
            stats = pi.read_apple_gpu_stats()

        assert stats == {"utilization_percent": None, "vram_used_gb": None}
        mock_run.assert_not_called()

    def test_returns_empty_stats_without_statistics_block(self, monkeypatch):
        """Output lacking PerformanceStatistics yields empty stats."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok("+-o IOAccelerator\n")):
            stats = pi.read_apple_gpu_stats()

        assert stats == {"utilization_percent": None, "vram_used_gb": None}


class TestUnifiedMemoryAndChip:
    """Tests for get_apple_unified_memory_gb() and get_apple_chip_name()."""

    def test_unified_memory_converts_bytes_to_gb(self, monkeypatch):
        """hw.memsize bytes are converted to GB."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok("68719476736\n")):
            assert pi.get_apple_unified_memory_gb() == 64.0

    def test_unified_memory_handles_non_numeric(self, monkeypatch):
        """Unparsable sysctl output yields None."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok("banana\n")):
            assert pi.get_apple_unified_memory_gb() is None

    def test_unified_memory_none_off_macos(self, monkeypatch):
        """Non-macOS platforms return None."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        assert pi.get_apple_unified_memory_gb() is None

    def test_chip_name_from_sysctl(self, monkeypatch):
        """The CPU brand string is returned verbatim."""
        monkeypatch.setattr(pi, "IS_MACOS", True)
        with patch("subprocess.run", return_value=_ok("Apple M4 Max\n")):
            assert pi.get_apple_chip_name() == "Apple M4 Max"

    def test_chip_name_none_off_macos(self, monkeypatch):
        """Non-macOS platforms return None."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        assert pi.get_apple_chip_name() is None


class TestMetalDriverVersion:
    """Tests for get_metal_driver_version()."""

    def test_includes_metal_family_and_build(self, monkeypatch):
        """Metal family and macOS build are combined."""
        monkeypatch.setattr(pi, "IS_MACOS", True)

        def fake_run(cmd, **_kwargs):
            if cmd[0] == "system_profiler":
                return _ok(SYSTEM_PROFILER_APPLE)
            return _ok("25G72\n")

        with patch("subprocess.run", side_effect=fake_run):
            result = pi.get_metal_driver_version()

        assert result == "metal4 (macOS build 25G72)"

    def test_falls_back_without_gpu_info(self, monkeypatch):
        """Only the build is reported when no GPU is detected."""
        monkeypatch.setattr(pi, "IS_MACOS", True)

        def fake_run(cmd, **_kwargs):
            if cmd[0] == "system_profiler":
                return MagicMock(returncode=1, stdout="")
            return _ok("25G72\n")

        with patch("subprocess.run", side_effect=fake_run):
            assert pi.get_metal_driver_version() == "macOS build 25G72"

    def test_none_off_macos(self, monkeypatch):
        """Non-macOS platforms return None."""
        monkeypatch.setattr(pi, "IS_MACOS", False)
        assert pi.get_metal_driver_version() is None


class TestPlatformFlags:
    """Tests for the module-level platform flags."""

    def test_at_most_one_flag_matches_current_platform(self):
        """The flags are mutually exclusive for the running platform."""
        assert sum([pi.IS_MACOS, pi.IS_LINUX, pi.IS_WINDOWS]) <= 1
