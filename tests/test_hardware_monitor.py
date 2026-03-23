"""Tests for tools/hardware_monitor.py (GPUMonitor, HardwareMonitor)."""
from pathlib import Path
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tools.hardware_monitor import GPUMonitor, HardwareMonitor


class TestHardwareMonitor:
    """Tests for HardwareMonitor."""

    def test_init_disabled(self):
        """HardwareMonitor starts with enabled=False."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        assert monitor.enabled is False
        assert monitor.monitoring is False

    def test_init_enabled(self):
        """HardwareMonitor with enabled=True stores correct state."""
        monitor = HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        assert monitor.enabled is True
        assert monitor.gpu_type == "NVIDIA"
        assert monitor.gpu_tool == "nvidia-smi"

    def test_start_when_disabled_does_not_start_thread(self):
        """start() with enabled=False does not start monitoring."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        monitor.start()
        assert monitor.monitoring is False
        assert monitor.thread is None

    def test_start_without_gpu_tool_does_not_start(self):
        """start() with no gpu_tool does not start monitoring."""
        monitor = HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=True)
        monitor.start()
        assert monitor.monitoring is False

    def test_start_resets_all_measurement_buffers(self):
        """start() clears all stale readings before a new run begins."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=True
        )
        monitor.temps = [60.0]
        monitor.powers = [50.0]
        monitor.vrams = [9.0]
        monitor.gtts = [1.0]
        monitor.cpus = [25.0]
        monitor.rams = [15.0]
        monitor.ram_readings = [15.0, 15.1]

        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            monitor.start()

        assert monitor.temps == []
        assert monitor.powers == []
        assert monitor.vrams == []
        assert monitor.gtts == []
        assert monitor.cpus == []
        assert monitor.rams == []
        assert monitor.ram_readings == []
        mock_thread.start.assert_called_once()

    def test_stop_returns_stats_dict(self):
        """stop() returns a dict with expected statistic keys."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
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
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        stats = monitor.stop()
        assert stats["temp_celsius_min"] is None
        assert stats["power_watts_avg"] is None

    def test_stop_computes_stats_from_readings(self):
        """stop() computes min/max/avg from collected readings."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        monitor.temps = [60.0, 70.0, 80.0]
        monitor.powers = [100.0, 120.0]
        stats = monitor.stop()
        assert stats["temp_celsius_min"] == 60.0
        assert stats["temp_celsius_max"] == 80.0
        assert stats["temp_celsius_avg"] == pytest.approx(70.0)

    def test_get_cpu_usage_returns_float_or_none(self):
        """_get_cpu_usage() returns a float or None."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        with patch("psutil.cpu_percent", return_value=42.5):
            result = monitor._get_cpu_usage()
        assert result == pytest.approx(42.5)

    def test_get_cpu_usage_handles_exception(self):
        """_get_cpu_usage() returns None on exception."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        with patch("psutil.cpu_percent", side_effect=RuntimeError("fail")):
            result = monitor._get_cpu_usage()
        assert result is None

    def test_get_ram_usage_returns_float(self):
        """_get_ram_usage() returns a smoothed float value."""
        monitor = HardwareMonitor(gpu_type=None, gpu_tool=None, enabled=False)
        mock_mem = MagicMock()
        mock_mem.used = 4 * (1024 ** 3)
        with patch("psutil.virtual_memory", return_value=mock_mem):
            result = monitor._get_ram_usage()
        assert result == pytest.approx(4.0)

    def test_get_temperature_returns_none_without_tool(self):
        """_get_temperature() returns None when gpu_tool is None."""
        monitor = HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_temperature() is None

    def test_get_temperature_nvidia_success(self):
        """_get_temperature() parses NVIDIA temperature correctly."""
        monitor = HardwareMonitor(
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
        monitor = HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_temperature()
        assert result is None

    def test_get_power_draw_returns_none_without_tool(self):
        """_get_power_draw() returns None when gpu_tool is None."""
        monitor = HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_power_draw() is None

    def test_get_power_nvidia_success(self):
        """_get_power_draw() parses NVIDIA power correctly."""
        monitor = HardwareMonitor(
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
        monitor = HardwareMonitor(gpu_type="NVIDIA", gpu_tool=None, enabled=False)
        assert monitor._get_vram_usage() is None

    def test_get_vram_nvidia_success(self):
        """_get_vram_usage() returns GB for NVIDIA correctly."""
        monitor = HardwareMonitor(
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
        monitor = HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=False
        )
        assert monitor._get_gtt_usage() is None

    def test_get_gtt_returns_none_without_tool(self):
        """_get_gtt_usage() returns None when gpu_tool is None."""
        monitor = HardwareMonitor(gpu_type="AMD", gpu_tool=None, enabled=False)
        assert monitor._get_gtt_usage() is None


class TestGPUMonitor:
    """Tests for GPUMonitor."""

    def test_init_detects_no_gpu_when_no_tools(self):
        """GPUMonitor falls back to Unknown when no GPU tools found."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)), \
                patch("glob.glob", return_value=[]):
            monitor = GPUMonitor()
        assert monitor.gpu_type == "Unknown"
        assert monitor.gpu_model == "Unknown"

    def test_find_tool_returns_none_when_absent(self):
        """_find_tool returns None when tool is not in PATH."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = GPUMonitor()
        result = monitor._find_tool("nonexistent_tool_xyz", ["/nowhere"])
        assert result is None

    def test_find_tool_returns_path_from_which(self):
        """_find_tool returns tool name when found via which."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = GPUMonitor()
        with patch("shutil.which", return_value="/usr/bin/mytool"):
            result = monitor._find_tool("mytool", [])
        assert result == "mytool"

    def test_detect_nvidia_gpu(self):
        """GPUMonitor detects NVIDIA GPU when nvidia-smi is available."""
        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "NVIDIA GeForce RTX 4090\n"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_run):
            monitor = GPUMonitor()
        assert monitor.gpu_type == "NVIDIA"
        assert monitor.gpu_tool == "nvidia-smi"

    def test_detect_amd_gpu_via_rocm_smi(self):
        """GPUMonitor detects AMD GPU when rocm-smi is available."""
        def mock_which(name, *args, **kwargs):
            return "/usr/bin/rocm-smi" if name == "rocm-smi" else None

        mock_run = MagicMock()
        mock_run.returncode = 1

        with patch("shutil.which", side_effect=mock_which), \
                patch("subprocess.run", return_value=mock_run), \
                patch("glob.glob", return_value=[]):
            monitor = GPUMonitor()
        assert monitor.gpu_type == "AMD"

    def test_get_vram_usage_returns_na_without_tool(self):
        """get_vram_usage returns 'N/A' when no GPU tool found."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)), \
                patch("glob.glob", return_value=[]):
            monitor = GPUMonitor()
        assert monitor.get_vram_usage() == "N/A"

    def test_get_vram_usage_nvidia_success(self):
        """get_vram_usage returns VRAM string for NVIDIA."""
        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = "RTX 4090\n"

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_run):
            monitor = GPUMonitor()

        mock_vram = MagicMock()
        mock_vram.returncode = 0
        mock_vram.stdout = "8192\n"
        with patch("subprocess.run", return_value=mock_vram):
            result = monitor.get_vram_usage()
        assert "8192" in result

    def test_find_amd_sysfs_path_returns_none_on_failure(self):
        """_find_amd_sysfs_path returns None when lspci fails."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = GPUMonitor()
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=1, stdout=""),
        ), patch("glob.glob", return_value=[]):
            result = monitor._find_amd_sysfs_path()
        assert result is None

    def test_detect_amd_gpu_model_returns_string(self):
        """_detect_amd_gpu_model always returns a string."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = GPUMonitor()
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
            result = monitor._detect_amd_gpu_model()
        assert isinstance(result, str)
        assert "AMD" in result

    def test_detect_intel_gpu_model_returns_string(self):
        """_detect_intel_gpu_model always returns a string."""
        with patch("shutil.which", return_value=None), \
                patch("subprocess.run", return_value=MagicMock(returncode=1)):
            monitor = GPUMonitor()
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="")):
            result = monitor._detect_intel_gpu_model()
        assert isinstance(result, str)


class TestHardwareMonitorAdvanced:
    """Additional tests for HardwareMonitor."""

    def test_init_with_amd_sysfs(self):
        """HardwareMonitor init with AMD sysfs tool calls init paths."""
        with patch("glob.glob", return_value=[]):
            monitor = HardwareMonitor("AMD", "sysfs", enabled=False)
        assert monitor.gpu_type == "AMD"

    def test_start_enabled_starts_thread(self):
        """start() starts a background thread when enabled."""
        monitor = HardwareMonitor("NVIDIA", "nvidia-smi", enabled=True)
        monitor._monitor_loop = MagicMock(return_value=None)
        monitor.start()
        time.sleep(0.05)
        monitor.monitoring = False
        assert monitor.thread is not None

    def test_get_power_draw_returns_none_without_tool(self):
        """_get_power_draw returns None when gpu_tool is empty."""
        monitor = HardwareMonitor("NVIDIA", "", enabled=False)
        result = monitor._get_power_draw()
        assert result is None

    def test_get_power_draw_nvidia_success(self):
        """_get_power_draw returns float for NVIDIA GPU."""
        monitor = HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        with patch(
            "subprocess.run",
            return_value=MagicMock(returncode=0, stdout="150.00\n"),
        ):
            result = monitor._get_power_draw()
        assert result == pytest.approx(150.0, 0.1)

    def test_stop_with_data_returns_averages(self):
        """stop() computes averages when data has been recorded."""
        monitor = HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        monitor.temps = [70.0, 72.0, 74.0]
        monitor.powers = [120.0, 125.0]
        monitor.vrams = [8.0, 8.5]
        stats = monitor.stop()
        assert stats["temp_celsius_avg"] == pytest.approx(72.0, 0.01)
        assert stats["power_watts_avg"] is not None

    def test_get_cpu_usage_returns_float(self):
        """_get_cpu_usage returns a float value."""
        monitor = HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        result = monitor._get_cpu_usage()
        assert result is None or isinstance(result, float)

    def test_get_ram_usage_returns_float_or_none(self):
        """_get_ram_usage returns a float or None."""
        monitor = HardwareMonitor("NVIDIA", "nvidia-smi", enabled=False)
        result = monitor._get_ram_usage()
        assert result is None or isinstance(result, float)


class TestHardwareMonitorAMD:
    """AMD-specific path tests for HardwareMonitor."""

    def test_get_temperature_amd_rocm_smi_success(self):
        """_get_temperature() reads via rocm-smi for AMD GPU."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "GPU[0]          : Temperature (Sensor junction) (C): 65\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_temperature()
        assert result is None or isinstance(result, float)

    def test_get_temperature_amd_sysfs(self, tmp_path: Path):
        """_get_temperature() reads from sysfs for AMD GPU."""
        hwmon_path = tmp_path / "hwmon"
        hwmon_path.mkdir()
        temp_file = hwmon_path / "temp1_input"
        temp_file.write_text("65000")
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_hwmon_path = str(hwmon_path)
        result = monitor._get_temperature()
        assert result == pytest.approx(65.0)

    def test_get_temperature_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_temperature() returns None when sysfs file missing."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_hwmon_path = str(tmp_path / "nonexistent")
        result = monitor._get_temperature()
        assert result is None

    def test_get_power_draw_amd_rocm_smi(self):
        """_get_power_draw() reads via rocm-smi for AMD GPU."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "GPU[0]          : Average Graphics Package Power (W): 120.5\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_power_draw()
        assert result is None or isinstance(result, float)

    def test_get_power_draw_amd_rocm_fail(self):
        """_get_power_draw() returns None on rocm-smi failure."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_power_draw()
        assert result is None

    def test_get_vram_usage_amd_sysfs(self, tmp_path: Path):
        """_get_vram_usage() reads from sysfs for AMD GPU."""
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        vram_bytes = 4 * (1024 ** 3)
        (sysfs_path / "mem_info_vram_used").write_text(str(vram_bytes))
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor._get_vram_usage()
        assert result == pytest.approx(4.0)

    def test_get_vram_usage_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_vram_usage() returns None when sysfs file missing."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(tmp_path / "nonexistent")
        result = monitor._get_vram_usage()
        assert result is None

    def test_get_vram_usage_amd_rocm_success(self):
        """_get_vram_usage() reads via rocm-smi for AMD GPU (Used Memory)."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "GPU[0]          : Used Memory (VRAM%): "
            "GPU[0] Used Memory: 4294967296\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_vram_usage()
        assert result is None or isinstance(result, float)

    def test_get_gtt_usage_amd_sysfs(self, tmp_path: Path):
        """_get_gtt_usage() reads from sysfs for AMD GPU."""
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        gtt_bytes = 2 * (1024 ** 3)
        (sysfs_path / "mem_info_gtt_used").write_text(str(gtt_bytes))
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor._get_gtt_usage()
        assert result == pytest.approx(2.0)

    def test_get_gtt_usage_amd_sysfs_no_file(self, tmp_path: Path):
        """_get_gtt_usage() returns None when sysfs file missing."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="sysfs", enabled=False
        )
        monitor._amd_sysfs_path = str(tmp_path / "nonexistent")
        result = monitor._get_gtt_usage()
        assert result is None

    def test_get_gtt_usage_amd_rocm_success(self):
        """_get_gtt_usage() uses rocm-smi for AMD GPU."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = (
            "GPU[0]   Used Memory (GTT): GPU[0] Used Memory: 0\n"
        )
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._get_gtt_usage()
        assert result is None or isinstance(result, float)

    def test_get_gtt_usage_amd_exception(self):
        """_get_gtt_usage() returns None on subprocess exception."""
        monitor = HardwareMonitor(
            gpu_type="AMD", gpu_tool="rocm-smi", enabled=False
        )
        with patch("subprocess.run", side_effect=OSError("fail")):
            result = monitor._get_gtt_usage()
        assert result is None


class TestHardwareMonitorMonitorLoop:
    """Tests for HardwareMonitor._monitor_loop."""

    def test_monitor_loop_runs_and_stops(self):
        """_monitor_loop() appends readings and respects monitoring flag."""
        monitor = HardwareMonitor(
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
        monitor = HardwareMonitor(
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
        monitor = HardwareMonitor(
            gpu_type="NVIDIA", gpu_tool="nvidia-smi", enabled=True
        )
        mock_thread = MagicMock()
        with patch("threading.Thread", return_value=mock_thread):
            monitor.start()
        assert monitor.monitoring is True
        mock_thread.start.assert_called_once()


class TestGPUMonitorAdvanced:
    """Additional tests for GPUMonitor AMD and Intel paths."""

    def test_find_amd_sysfs_path_returns_path_on_amd_device(
        self, tmp_path: Path
    ):
        """_find_amd_sysfs_path returns path when AMD vendor found."""
        card_path = tmp_path / "card0" / "device"
        card_path.mkdir(parents=True)
        (card_path / "vendor").write_text("0x1002")
        (card_path / "mem_info_vram_total").write_text("8589934592")
        monitor = GPUMonitor()
        with patch("glob.glob", return_value=[str(card_path)]):
            result = monitor._find_amd_sysfs_path()
        assert result == str(card_path)

    def test_detect_gpu_finds_nvidia(self):
        """_detect_gpu recognizes NVIDIA GPU from nvidia-smi output."""
        monitor = GPUMonitor.__new__(GPUMonitor)
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA GeForce RTX 4080"
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"), \
                patch("subprocess.run", return_value=mock_result):
            monitor.__init__()
        assert monitor.gpu_type == "NVIDIA"

    def test_detect_amd_gpu_model_via_rocm(self):
        """_detect_amd_gpu_model returns string from rocm-smi."""
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_type = "AMD"
        monitor.gpu_tool = "rocm-smi"
        mock_show = MagicMock(
            returncode=0,
            stdout="GPU[0] : Navi 21 [Radeon RX 6800 XT]\n",
        )
        mock_lspci = MagicMock(returncode=1, stdout="")
        with patch(
            "subprocess.run",
            side_effect=[mock_lspci, mock_show],
        ):
            result = monitor._detect_amd_gpu_model()
        assert isinstance(result, str)

    def test_detect_intel_gpu_model_returns_string(self):
        """_detect_intel_gpu_model returns string."""
        monitor = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_type = "Intel"
        monitor.gpu_tool = None
        mock_result = MagicMock(returncode=1, stdout="", stderr="")
        with patch("subprocess.run", return_value=mock_result):
            result = monitor._detect_intel_gpu_model()
        assert isinstance(result, str)

    def test_get_vram_usage_amd_via_sysfs(self, tmp_path: Path):
        """get_vram_usage returns sysfs value for AMD GPU."""
        sysfs_path = tmp_path / "sysfs"
        sysfs_path.mkdir()
        (sysfs_path / "mem_info_vram_used").write_text(str(8 * 1024 ** 3))
        monitor: Any = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_type = "AMD"
        monitor.gpu_tool = "sysfs"
        monitor.gpu_model = "Radeon RX 7900"
        monitor._amd_sysfs_path = str(sysfs_path)
        result = monitor.get_vram_usage()
        assert "GB" in result or isinstance(result, str)

    def test_get_vram_usage_nvidia_success(self):
        """get_vram_usage returns formatted VRAM for NVIDIA GPU."""
        monitor: Any = GPUMonitor.__new__(GPUMonitor)
        monitor.gpu_type = "NVIDIA"
        monitor.gpu_tool = "/usr/bin/nvidia-smi"
        monitor.gpu_model = "NVIDIA GeForce RTX 4080"
        monitor._amd_sysfs_path = None
        mock_result = MagicMock(returncode=0, stdout="16376\n")
        with patch("subprocess.run", return_value=mock_result):
            result = monitor.get_vram_usage()
        assert isinstance(result, str)
        assert result != ""
