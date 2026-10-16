"""Shared GPU detection and hardware monitoring helpers."""

import glob
import logging
import math
import os
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess  # nosec B404 - vendor CLIs are the only source for this data
import threading
import time
from typing import Any, Dict, List, Optional

import psutil

from core.platform_info import (
    IS_MACOS,
    IS_WINDOWS,
    detect_apple_gpu,
    get_apple_unified_memory_gb,
    read_apple_gpu_stats,
)
from core.runtime import child_creation_flags
from tools.gpu_devices import (
    GpuAggregate,
    GpuSample,
    device_names,
    parse_gpu_number,
    sample_devices,
    serialize_aggregates,
)
from tools.macmon import MacmonSampler, is_macmon_available
from tools.windows_gpu import (
    WindowsGpuDevice,
    WindowsGpuError,
    WindowsGpuSampler,
    WindowsGpuSnapshot,
)
from tools.windows_gpu import query_snapshot as query_windows_snapshot

try:
    import cpuinfo
except (ImportError, ModuleNotFoundError):
    cpuinfo = None


logger = logging.getLogger(__name__)


PER_GPU_MIN_DEVICES = 2
WINDOWS_NATIVE_TOOL = "windows-native"

# lspci lives in sbin on distributions that keep it off a user's PATH.
LSPCI_SEARCH_PATHS = ("/usr/sbin", "/sbin", "/usr/bin", "/bin")


def _resolve_tool(
    tool_name: str,
    search_paths: tuple[str, ...] = (),
) -> Optional[str]:
    """Return the absolute path of a CLI, or None when it is not installed.

    Args:
        tool_name: Executable to look for.
        search_paths: Directories to try when PATH does not contain it.
    """
    found = shutil.which(tool_name)
    if found:
        return found

    for path in search_paths:
        found = shutil.which(tool_name, path=path)
        if found:
            return found

    return None


def _run_tool(command: List[str], timeout: int) -> Optional[str]:
    """Run a hardware query CLI and return its stdout.

    Every vendor call in this module goes through here, so the subprocess
    keyword arguments are defined once and a security audit has a single
    place to look. Returns None when the tool is missing, times out or
    exits non-zero - all of which mean "no reading available" to callers.
    """
    try:
        result = subprocess.run(  # nosec B603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
            creationflags=child_creation_flags(),
        )
    except (subprocess.SubprocessError, OSError) as error:
        logger.debug("Hardware query failed (%s): %s", command[0], error)
        return None

    if result.returncode != 0:
        logger.debug("Hardware query %s exited with %s", command[0], result.returncode)
        return None
    return result.stdout


def nvidia_search_paths() -> list[str]:
    """Return platform-appropriate NVIDIA executable search directories."""
    if not IS_WINDOWS:
        return ["/usr/bin", "/usr/local/bin", "/usr/local/cuda/bin"]
    paths = []
    windows = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if windows:
        paths.append(str(Path(windows) / "System32"))
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        paths.append(str(Path(program_files) / "NVIDIA Corporation" / "NVSMI"))
    return paths


def nvidia_smi_path() -> Optional[str]:
    """Resolve NVIDIA's CLI through PATH and native installation locations."""
    return _resolve_tool("nvidia-smi", tuple(nvidia_search_paths()))


def nvidia_driver_version() -> Optional[str]:
    """Return the first NVIDIA adapter's driver version, if available."""
    tool = nvidia_smi_path()
    if tool is None:
        return None
    output = _run_tool(
        [tool, "--query-gpu=driver_version", "--format=csv,noheader"], timeout=5
    )
    return output.strip().splitlines()[0] if output and output.strip() else None


def evaluate_hardware_limits(
    stats: dict[str, Any],
    max_temp: float | None = None,
    max_power: float | None = None,
) -> dict[str, bool | None]:
    """Evaluate warning thresholds without calling missing readings a pass.

    Args:
        stats: Recorded hardware statistics.
        max_temp: Temperature warning limit; None/zero disables it.
        max_power: Power warning limit; None/zero disables it.

    Returns:
        Exceeded flags; None means a configured threshold cannot be checked.

    Raises:
        ValueError: If a configured limit is not finite.
    """
    results: dict[str, bool | None] = {}
    for label, metric, key, limit in (
        ("temperature", "temp_celsius_max", "max_temp_exceeded", max_temp),
        ("power", "power_watts_max", "max_power_exceeded", max_power),
    ):
        if not limit:
            results[key] = False
            continue
        if not math.isfinite(limit):
            raise ValueError(f"The {label} limit must be finite")
        value = stats.get(metric)
        if value is None:
            results[key] = None
            logger.warning(
                "%s threshold cannot be checked: the GPU sensor is unavailable",
                label.capitalize(),
            )
            continue
        results[key] = value > limit
        if results[key]:
            logger.warning("Max. %s exceeded: %s > %s", label, value, limit)
    return results


class HardwareMonitor:
    """Real-time monitoring of GPU temperature and power draw."""

    def __init__(
        self,
        gpu_type: Optional[str],
        gpu_tool: Optional[str],
        enabled: bool = False,
        *,
        device_id: str | None = None,
    ) -> None:
        """Initialize the selected backend without starting a sampling thread."""
        self.gpu_type = gpu_type
        self.gpu_tool = gpu_tool
        self.enabled = enabled
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.temps: List[float] = []
        self.powers: List[float] = []
        self.vrams: List[float] = []
        self.gtts: List[float] = []
        self.cpus: List[float] = []
        self.rams: List[float] = []
        self.ram_readings: List[float] = []
        self.lock = threading.Lock()
        self._amd_sysfs_path: Optional[str] = None
        self._amd_hwmon_path: Optional[str] = None
        self._native_device_id = device_id
        self._native_snapshot: WindowsGpuSnapshot | None = None
        self._native_indices: dict[str, int] = {}
        self._native_sampler = (
            WindowsGpuSampler(nvidia_smi_path())
            if gpu_tool == WINDOWS_NATIVE_TOOL else None
        )

        # Per-device tracking is only set up when the vendor tool actually
        # reports more than one GPU. On a single-GPU machine the aggregate
        # readers already tell the whole story, and an empty per-device list
        # keeps reports and dashboards free of placeholder charts.
        self.gpu_devices: Dict[int, GpuAggregate] = {}
        self.device_count: int = 0
        self._device_names: Dict[int, str] = {}
        if self.enabled:
            self._init_gpu_devices()

        # Apple Silicon exposes temperature and power only through
        # powermetrics (root) or macmon (sudoless). macmon is optional: when
        # absent, those metrics simply stay unset.
        self._macmon: Optional[MacmonSampler] = None
        if self.gpu_type == "Apple" and is_macmon_available():
            self._macmon = MacmonSampler()

        if self.gpu_type == "AMD" and self.gpu_tool == "sysfs":
            self._init_amd_sysfs_paths()

    def _init_gpu_devices(self) -> None:
        """Detect whether per-device sampling applies to this machine."""
        if self._native_sampler is not None:
            return
        try:
            devices = sample_devices(self.gpu_type, self.gpu_tool)
        except (OSError, ValueError) as error:  # pragma: no cover - defensive
            logger.debug("Per-GPU detection failed: %s", error)
            return

        self.device_count = len(devices)
        if self.device_count < PER_GPU_MIN_DEVICES:
            return

        self._device_names = device_names(self.gpu_type, self.gpu_tool)
        logger.info(
            "🎛️ %s GPUs detected, recording per-device metrics",
            self.device_count,
        )

    def _sample_gpu_devices(self) -> None:
        """Fold one per-device reading into the aggregates."""
        if self._native_sampler is not None:
            snapshot = self._native_snapshot
            if snapshot is None:
                return
            with self.lock:
                for device in snapshot.devices:
                    index = self._native_indices.setdefault(
                        device.id, len(self._native_indices)
                    )
                    aggregate = self.gpu_devices.setdefault(
                        index, GpuAggregate(index=index, name=device.name)
                    )
                    aggregate.metadata = {
                        "device_id": device.id,
                        "vendor": device.vendor,
                        "driver_version": device.driver_version,
                        "integrated": device.integrated,
                        "sensor_status": device.status.model_dump(),
                        "sensor_messages": dict(device.messages),
                        "sensor_sources": dict(device.sources),
                    }
                    aggregate.add(GpuSample(
                        index=index, name=device.name,
                        temp_celsius=device.metrics.temp_celsius,
                        power_watts=device.metrics.power_watts,
                        vram_gb=device.metrics.vram_gb,
                        gtt_gb=device.metrics.gtt_gb,
                    ))
                self.device_count = len(self._native_indices)
            return
        if self.device_count < PER_GPU_MIN_DEVICES:
            return

        samples = sample_devices(self.gpu_type, self.gpu_tool)
        if not samples:
            return

        with self.lock:
            for sample in samples:
                if sample.name is None:
                    sample.name = self._device_names.get(sample.index)
                aggregate = self.gpu_devices.setdefault(
                    sample.index,
                    GpuAggregate(index=sample.index, name=sample.name),
                )
                aggregate.add(sample)

    def _log_per_gpu_reading(self) -> None:
        """Log the latest reading of every device on a multi-GPU machine."""
        if self.device_count < PER_GPU_MIN_DEVICES:
            return

        def display(value: float | None, digits: int) -> str:
            return f"{value:.{digits}f}" if value is not None else "N/A"

        with self.lock:
            parts = []
            for index in sorted(self.gpu_devices):
                aggregate = self.gpu_devices[index]
                latest = {
                    metric: readings[-1]
                    for metric, readings in aggregate.values.items()
                    if readings
                }
                status = aggregate.metadata.get("sensor_status", {})
                for metric, availability in status.items():
                    if availability != "ok":
                        latest.pop(metric, None)
                if not latest:
                    continue
                label = aggregate.name or f"GPU{index}"
                parts.append(
                    f"[{index}] {label}: "
                    f"{display(latest.get('vram_gb'), 2)}GB VRAM, "
                    f"{display(latest.get('gtt_gb'), 2)}GB GTT, "
                    f"{display(latest.get('temp_celsius'), 0)}°C, "
                    f"{display(latest.get('power_watts'), 0)}W"
                )

        if parts:
            logger.info("🎛️ %s", " | ".join(parts))

    def per_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Per-device min/max/avg, empty on single-GPU systems."""
        with self.lock:
            aggregates = [
                self.gpu_devices[index]
                for index in sorted(self.gpu_devices)
            ]
        return [
            aggregate.as_dict()
            for aggregate in aggregates
            if aggregate.samples > 0
        ]

    def per_gpu_metrics_json(self) -> Optional[str]:
        """Per-device metrics serialized for a TEXT column."""
        with self.lock:
            aggregates = [
                self.gpu_devices[index]
                for index in sorted(self.gpu_devices)
            ]
        return serialize_aggregates(aggregates)

    def _reset_measurements(self) -> None:
        """Clear all collected measurements before a new profiling run."""
        with self.lock:
            self.temps.clear()
            self.powers.clear()
            self.vrams.clear()
            self.gtts.clear()
            self.cpus.clear()
            self.rams.clear()
            self.ram_readings.clear()
            self.gpu_devices.clear()
            self._native_indices.clear()
            self._native_snapshot = None
        self._amd_sysfs_path = None
        self._amd_hwmon_path = None

        if self.gpu_type == "AMD" and self.gpu_tool == "sysfs":
            self._init_amd_sysfs_paths()

    def _init_amd_sysfs_paths(self) -> None:
        """Initialize AMD sysfs paths at startup."""
        try:
            for cardpath in glob.glob("/sys/class/drm/card*/device"):
                vendor_file = Path(cardpath) / "vendor"
                if vendor_file.exists():
                    vendor = vendor_file.read_text().strip()
                    if vendor == "0x1002":
                        vram_file = Path(cardpath) / "mem_info_vram_total"
                        if vram_file.exists():
                            self._amd_sysfs_path = cardpath
                            hwmon_paths = glob.glob(f"{cardpath}/hwmon/hwmon*")
                            if hwmon_paths:
                                self._amd_hwmon_path = hwmon_paths[0]
                            break
        except OSError:
            pass

    def start(self) -> None:
        """Start background monitoring."""
        if not self.enabled:
            logger.info(
                "⚠️ Hardware monitoring disabled (--enable-profiling not set)"
            )
            return

        if self.monitoring:
            logger.warning("Hardware monitoring is already running")
            return
        if not self.gpu_tool:
            logger.warning("No GPU tools found; recording CPU and RAM only")

        logger.info(
            "🔥 Starting Hardware-Monitoring (GPU: %s, Tool: %s)",
            self.gpu_type,
            self.gpu_tool,
        )
        self.monitoring = True
        self._reset_measurements()

        if self._native_sampler is not None:
            logger.info("Initializing native Windows GPU monitoring")
            self._native_sampler.start()
            if not self._native_sampler.wait_ready():
                logger.warning(
                    "Windows GPU data is unavailable: %s",
                    self._native_sampler.error or "initialization timed out",
                )

        if self._macmon is not None and not self._macmon.start():
            logger.debug("macmon could not be started; temp/power stay unset")

        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return collected statistics."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5 if self._native_sampler is not None else 2)

        if self._native_sampler is not None:
            self._native_sampler.stop()

        if self._macmon is not None:
            self._macmon.stop()

        with self.lock:
            temps = self.temps.copy()
            powers = self.powers.copy()
            vrams = self.vrams.copy()
            gtts = self.gtts.copy()
            cpus = self.cpus.copy()
            rams = self.rams.copy()

        return {
            "temp_celsius_min": min(temps) if temps else None,
            "temp_celsius_max": max(temps) if temps else None,
            "temp_celsius_avg": mean(temps) if temps else None,
            "power_watts_min": min(powers) if powers else None,
            "power_watts_max": max(powers) if powers else None,
            "power_watts_avg": mean(powers) if powers else None,
            "vram_gb_min": min(vrams) if vrams else None,
            "vram_gb_max": max(vrams) if vrams else None,
            "vram_gb_avg": mean(vrams) if vrams else None,
            "gtt_gb_min": min(gtts) if gtts else None,
            "gtt_gb_max": max(gtts) if gtts else None,
            "gtt_gb_avg": mean(gtts) if gtts else None,
            "cpu_percent_min": min(cpus) if cpus else None,
            "cpu_percent_max": max(cpus) if cpus else None,
            "cpu_percent_avg": mean(cpus) if cpus else None,
            "ram_gb_min": min(rams) if rams else None,
            "ram_gb_max": max(rams) if rams else None,
            "ram_gb_avg": mean(rams) if rams else None,
            "gpu_count": self.device_count,
            "per_gpu": self.per_gpu_metrics(),
        }

    def _monitor_loop(self) -> None:
        """Background thread for continuous measurements."""
        logger.info("🔍 Hardware-Monitor thread started")
        while self.monitoring:
            try:
                if self._native_sampler is not None:
                    self._native_snapshot = self._native_sampler.sample()
                temp = self._get_temperature()
                power = self._get_power_draw()
                vram = self._get_vram_usage()
                gtt = self._get_gtt_usage()
                cpu = self._get_cpu_usage()
                ram = self._get_ram_usage()
                self._sample_gpu_devices()

                with self.lock:
                    if temp is not None:
                        self.temps.append(temp)
                        logger.info("🌡️ GPU Temp: %s°C", temp)
                    else:
                        logger.debug(
                            "⚠️ No temperature read (gpu_type=%s, tool=%s)",
                            self.gpu_type,
                            self.gpu_tool,
                        )

                    if power is not None:
                        self.powers.append(power)
                        logger.info("⚡ GPU Power: %sW", power)
                    else:
                        logger.debug(
                            "⚠️ No power read (gpu_type=%s, tool=%s)",
                            self.gpu_type,
                            self.gpu_tool,
                        )

                    if vram is not None:
                        self.vrams.append(vram)
                        logger.info("💾 GPU VRAM: %sGB", vram)
                    else:
                        logger.debug(
                            "⚠️ No VRAM read (gpu_type=%s, tool=%s)",
                            self.gpu_type,
                            self.gpu_tool,
                        )

                    if gtt is not None:
                        self.gtts.append(gtt)
                        logger.info("🧠 GPU GTT: %sGB", gtt)
                    else:
                        logger.debug(
                            "⚠️ No GTT read (gpu_type=%s, tool=%s)",
                            self.gpu_type,
                            self.gpu_tool,
                        )

                    if cpu is not None:
                        self.cpus.append(cpu)
                        logger.info("🖥️ CPU: %s%%", cpu)

                    if ram is not None:
                        self.rams.append(ram)
                        logger.info("💾 RAM: %sGB", ram)

                self._log_per_gpu_reading()

                time.sleep(1)
            except (
                WindowsGpuError, subprocess.SubprocessError, OSError, ValueError
            ) as error:
                logger.warning("Monitoring error: %s", error)
                time.sleep(2)
        logger.info("🛑 Hardware-Monitor thread stopped")

    def _get_temperature(self) -> Optional[float]:
        """Read current GPU temperature."""
        if self._native_sampler is not None:
            device = self._native_device()
            return device.metrics.temp_celsius if device is not None else None
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "Apple":
                if self._macmon is None:
                    return None
                return self._macmon.get_gpu_temperature()

            if self.gpu_type == "NVIDIA":
                output = _run_tool(
                    [
                        self.gpu_tool,
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=3,
                )
                if output is not None:
                    temp_str = output.strip().split("\n")[0]
                    return parse_gpu_number(temp_str)

            if self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs" and self._amd_hwmon_path:
                    temp_file = Path(self._amd_hwmon_path) / "temp1_input"
                    if temp_file.exists():
                        temp_millic = int(temp_file.read_text().strip())
                        return float(temp_millic) / 1000.0

                output = _run_tool([self.gpu_tool, "--showtemp"], timeout=3)
                if output is not None:
                    for line in output.split("\n"):
                        if "GPU[" in line and ("(C):" in line or "c" in line.lower()):
                            try:
                                match = re.search(r"[\d.]+\s*$", line.strip())
                                if match:
                                    return float(match.group())
                                temp_str = line.split(":")[-1].strip()
                                temp_str = temp_str.replace("c", "").replace(
                                    "C", ""
                                )
                                return float(temp_str)
                            except (ValueError, IndexError):
                                pass
        except (subprocess.SubprocessError, OSError):
            pass

        return None

    def _get_power_draw(self) -> Optional[float]:
        """Read GPU power draw in watts."""
        if self._native_sampler is not None:
            device = self._native_device()
            return device.metrics.power_watts if device is not None else None
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "Apple":
                if self._macmon is None:
                    return None
                return self._macmon.get_gpu_power()

            if self.gpu_type == "NVIDIA":
                output = _run_tool(
                    [
                        self.gpu_tool,
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=3,
                )
                if output is not None:
                    power_str = output.strip().split("\n")[0]
                    return parse_gpu_number(power_str)

            if self.gpu_type == "AMD":
                output = _run_tool([self.gpu_tool, "--showpower"], timeout=3)
                if output is not None:
                    for line in output.split("\n"):
                        if "GPU[" in line and ("(W):" in line or "W" in line):
                            try:
                                match = re.search(r"[\d.]+\s*$", line.strip())
                                if match:
                                    return float(match.group())
                                power_str = line.split(":")[-1].strip()
                                power_str = power_str.replace("W", "")
                                return float(power_str)
                            except (ValueError, IndexError):
                                pass
        except (subprocess.SubprocessError, OSError):
            pass

        return None

    def _get_vram_usage(self) -> Optional[float]:
        """Read GPU VRAM usage in GB."""
        if self._native_sampler is not None:
            device = self._native_device()
            return device.metrics.vram_gb if device is not None else None
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "NVIDIA":
                output = _run_tool(
                    [
                        self.gpu_tool,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=3,
                )
                if output is not None:
                    vram_mb = parse_gpu_number(output.strip().split("\n")[0])
                    return vram_mb / 1024.0 if vram_mb is not None else None

            if self.gpu_type == "Apple":
                return read_apple_gpu_stats()["vram_used_gb"]

            if self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs" and self._amd_sysfs_path:
                    vram_file = Path(self._amd_sysfs_path) / "mem_info_vram_used"
                    if vram_file.exists():
                        vram_bytes = int(vram_file.read_text().strip())
                        return float(vram_bytes) / (1024**3)

                output = _run_tool(
                    [self.gpu_tool, "--showmeminfo", "vram"], timeout=3
                )
                if output is not None:
                    for line in output.split("\n"):
                        if "GPU[" in line and "Used Memory" in line:
                            match = re.search(r"(\d+)\s*$", line.strip())
                            if match:
                                vram_bytes = float(match.group(1))
                                return vram_bytes / (1024**3)
        except (subprocess.SubprocessError, OSError):
            pass

        return None

    def _get_gtt_usage(self) -> Optional[float]:
        """Read GTT usage in GB for AMD GPUs."""
        if self._native_sampler is not None:
            device = self._native_device()
            return device.metrics.gtt_gb if device is not None else None
        try:
            if self.gpu_type != "AMD" or not self.gpu_tool:
                return None

            if self.gpu_tool == "sysfs" and self._amd_sysfs_path:
                gtt_file = Path(self._amd_sysfs_path) / "mem_info_gtt_used"
                if gtt_file.exists():
                    gtt_bytes = int(gtt_file.read_text().strip())
                    return float(gtt_bytes) / (1024**3)

            output = _run_tool([self.gpu_tool, "--showmeminfo", "gtt"], timeout=3)
            if output is not None:
                for line in output.split("\n"):
                    if "GPU[" in line and "Used Memory" in line:
                        match = re.search(r"(\d+)\s*$", line.strip())
                        if match:
                            gtt_bytes = float(match.group(1))
                            return gtt_bytes / (1024**3)
        except (subprocess.SubprocessError, OSError):
            pass

        return None

    @property
    def windows_sampler(self) -> WindowsGpuSampler | None:
        """Expose the owned sampler for consistent per-run VRAM readings."""
        return self._native_sampler

    def _native_device(self) -> WindowsGpuDevice | None:
        if self._native_snapshot is None:
            return None
        if self._native_device_id is not None:
            return next(
                (device for device in self._native_snapshot.devices
                 if device.id == self._native_device_id),
                None,
            )
        return self._native_snapshot.primary_device(self.gpu_type)

    def _get_cpu_usage(self) -> Optional[float]:
        """Read system CPU utilization in percent."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except (OSError, RuntimeError):
            return None

    def _get_ram_usage(self) -> Optional[float]:
        """Read system RAM usage in GB with smoothing."""
        try:
            mem = psutil.virtual_memory()
            current_ram = mem.used / (1024**3)

            self.ram_readings.append(current_ram)
            if len(self.ram_readings) > 7:
                self.ram_readings.pop(0)

            return sum(self.ram_readings) / len(self.ram_readings)
        except OSError:
            return None


class GPUMonitor:
    """Detect GPU type and expose VRAM usage helpers."""

    def __init__(self):
        self.gpu_type: Optional[str] = None
        self.gpu_model: Optional[str] = None
        self.gpu_tool: Optional[str] = None
        self.native_snapshot: WindowsGpuSnapshot | None = None
        self.native_device_id: str | None = None
        self._windows_sampler: WindowsGpuSampler | None = None
        self.shared_memory_info: dict[str, float] | None = None
        self._detect_gpu()

    def get_gpu_info(self) -> Dict[str, Optional[str]]:
        """Return detected GPU metadata."""
        return {
            "gpu_type": self.gpu_type,
            "gpu_model": self.gpu_model,
            "gpu_tool": self.gpu_tool,
        }

    def get_total_vram_gb(self) -> Optional[float]:
        """Return primary GPU capacity without inventing a RAM-based fallback."""
        if self.native_snapshot is not None:
            device = self.native_snapshot.primary_device()
            return device.vram_total_gb if device is not None else None
        if self.gpu_type == "Apple":
            return get_apple_unified_memory_gb()
        if self.gpu_type != "NVIDIA" or not self.gpu_tool:
            return None
        output = _run_tool(
            [
                self.gpu_tool, "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
        )
        if not output or not output.strip():
            return None
        memory_mb = parse_gpu_number(output.strip().splitlines()[0])
        return memory_mb / 1024 if memory_mb is not None else None

    def attach_windows_sampler(self, sampler: WindowsGpuSampler | None) -> None:
        """Reuse the profiling sampler rather than launching competing probes."""
        self._windows_sampler = sampler

    def get_windows_device(self) -> WindowsGpuDevice | None:
        """Read the selected adapter, never substitute a differently named GPU."""
        if self.gpu_tool != WINDOWS_NATIVE_TOOL:
            return None
        if self._windows_sampler is not None and self._windows_sampler.running:
            snapshot = self._windows_sampler.latest_sample()
        else:
            try:
                snapshot = query_windows_snapshot(nvidia_smi_path())
            except WindowsGpuError as error:
                logger.warning("Windows GPU readings unavailable: %s", error)
                return None
        if snapshot is None:
            return None
        return next(
            (device for device in snapshot.devices
             if device.id == self.native_device_id),
            None,
        )

    def get_available_windows_memory(self, use_shared: bool) -> float | None:
        """Return measured memory headroom, bounded by available system RAM.

        Args:
            use_shared: Allow AMD/Intel shared memory for offload prediction.

        Returns:
            Available GiB, or None if the necessary measurements are absent.
        """
        device = self.get_windows_device()
        self.shared_memory_info = None
        if device is None:
            return None
        if device.vram_total_gb is None or device.metrics.vram_gb is None:
            return None
        available = max(0.0, device.vram_total_gb - device.metrics.vram_gb)
        if (
            device.vendor in ("AMD", "Intel")
            and device.shared_total_gb is not None
            and device.metrics.gtt_gb is not None
        ):
            self.shared_memory_info = {
                "total": device.shared_total_gb,
                "used": device.metrics.gtt_gb,
                "free": max(0.0, device.shared_total_gb - device.metrics.gtt_gb),
            }
        if (
            use_shared and device.vendor in ("AMD", "Intel")
            and device.shared_total_gb is not None
            and device.metrics.gtt_gb is not None
        ):
            shared = max(0.0, device.shared_total_gb - device.metrics.gtt_gb)
            available += min(shared, psutil.virtual_memory().available / 1024**3)
        return available

    def _find_tool(self, tool_name: str, search_paths: List[str]) -> Optional[str]:
        """Search for a tool in PATH and well-known fallback paths.

        Returns the absolute path, so every later call names the executable
        it resolved instead of leaving the lookup to PATH.
        """
        return _resolve_tool(tool_name, tuple(search_paths))

    def _find_amd_sysfs_path(self) -> Optional[str]:
        """Find AMD GPU sysfs path for direct monitoring."""
        lspci = _resolve_tool("lspci", LSPCI_SEARCH_PATHS)
        output = _run_tool([lspci, "-d", "1002:", "-n"], timeout=5) if lspci else None
        has_amd_lspci = bool(output and output.strip())

        for cardpath in glob.glob("/sys/class/drm/card*/device"):
            vendor_file = Path(cardpath) / "vendor"
            if not vendor_file.exists():
                continue
            vendor = vendor_file.read_text().strip()
            if vendor != "0x1002":
                continue
            vram_file = Path(cardpath) / "mem_info_vram_total"
            if vram_file.exists():
                return str(Path(cardpath))

        if has_amd_lspci:
            logger.debug(
                "AMD device seen via lspci, but no readable sysfs"
                " mem_info_vram_total path found"
            )
        return None

    def _detect_apple_gpu(self) -> bool:
        """Detect an Apple Silicon GPU on macOS.

        Returns:
            True when an Apple GPU was detected and state was set.
        """
        gpu_info = detect_apple_gpu()
        if not gpu_info or gpu_info.get("vendor") != "Apple":
            return False

        self.gpu_type = "Apple"
        # VRAM always comes from ioreg; macmon adds sudoless temperature and
        # power on top when it is installed.
        has_macmon = is_macmon_available()
        self.gpu_tool = "macmon+ioreg" if has_macmon else "ioreg"

        model = gpu_info["model"]
        cores = gpu_info.get("cores")
        self.gpu_model = f"{model} ({cores}-core GPU)" if cores else model

        unified_memory_gb = get_apple_unified_memory_gb()
        logger.info(
            "🍎 Apple GPU detected: %s, unified memory: %s GB, Tool: %s",
            self.gpu_model,
            unified_memory_gb if unified_memory_gb is not None else "unknown",
            self.gpu_tool,
        )
        if not has_macmon:
            logger.info(
                "💡 Install macmon (brew install macmon) for sudoless GPU "
                "temperature and power metrics."
            )
        return True

    def _detect_gpu(self) -> None:
        """Detect GPU type and corresponding monitoring tool."""
        if IS_MACOS and self._detect_apple_gpu():
            return

        if IS_WINDOWS:
            try:
                snapshot = query_windows_snapshot(nvidia_smi_path())
                device = snapshot.primary_device()
                if device is not None:
                    self.native_snapshot = snapshot
                    self.native_device_id = device.id
                    self.gpu_type = device.vendor
                    self.gpu_model = device.name
                    self.gpu_tool = WINDOWS_NATIVE_TOOL
                    logger.info(
                        "Windows GPU detected: %s (%s)", device.name, device.id
                    )
                    for error in snapshot.errors:
                        logger.warning("Windows GPU telemetry: %s", error)
                    return
            except WindowsGpuError as error:
                logger.warning("Native Windows GPU monitoring unavailable: %s", error)

        nvidia_paths = nvidia_search_paths()
        nvidia_tool = self._find_tool("nvidia-smi", nvidia_paths)
        if nvidia_tool:
            self.gpu_type = "NVIDIA"
            self.gpu_tool = nvidia_tool
            output = _run_tool(
                [nvidia_tool, "--query-gpu=name", "--format=csv,noheader"],
                timeout=5,
            )
            if output is not None:
                self.gpu_model = output.strip().split("\n")[0]
            else:
                self.gpu_model = "NVIDIA GPU"
            logger.info(
                "🟢 NVIDIA GPU detected: %s, Tool: %s",
                self.gpu_model,
                nvidia_tool,
            )
            return

        if IS_WINDOWS:
            self.gpu_type = "Unknown"
            self.gpu_model = "Unknown"
            logger.warning(
                "No supported Windows GPU monitoring backend is available"
            )
            return

        amd_paths = ["/usr/bin", "/usr/local/bin", "/opt/rocm/bin"]
        rocm_versions = glob.glob("/opt/rocm-*/bin")
        amd_paths.extend(rocm_versions)

        amd_tool = self._find_tool("rocm-smi", amd_paths)
        if amd_tool:
            self.gpu_type = "AMD"
            self.gpu_tool = amd_tool
            self.gpu_model = self._detect_amd_gpu_model()
            logger.info("🔴 AMD GPU detected: %s, Tool: %s", self.gpu_model, amd_tool)
            return

        amd_sysfs = self._find_amd_sysfs_path()
        if amd_sysfs:
            self.gpu_type = "AMD"
            self.gpu_tool = "sysfs"
            self.gpu_model = self._detect_amd_gpu_model()
            logger.info(
                "🔴 AMD GPU detected (sysfs): %s, Path: %s",
                self.gpu_model,
                amd_sysfs,
            )
            return

        intel_paths = ["/usr/bin", "/usr/local/bin", "/usr/lib/xpu"]
        intel_tool = self._find_tool("intel_gpu_top", intel_paths)
        if intel_tool:
            self.gpu_type = "Intel"
            self.gpu_tool = intel_tool
            self.gpu_model = self._detect_intel_gpu_model()
            logger.info(
                "🔵 Intel GPU detected: %s, Tool: %s",
                self.gpu_model,
                intel_tool,
            )
            return

        logger.warning(
            "⚠️ No GPU monitoring tools found. VRAM measurement not available."
        )
        self.gpu_type = "Unknown"
        self.gpu_model = "Unknown"

    def _detect_amd_gpu_model(self) -> str:
        """Name the AMD GPU the metrics are read from.

        Order matters. The aggregate readers report ``GPU[0]`` of rocm-smi,
        so that device decides the name. Two lower-ranked sources exist for
        machines where rocm-smi cannot answer, and both are wrong on a
        system with a discrete card next to an integrated one:

        * ``lspci`` lists whatever it finds first, which need not be the
          measured device - an OcuLink GPU may not show up at all.
        * The CPU brand string ("Ryzen AI 9 HX 370 w/ Radeon 890M") always
          describes the *integrated* GPU. Asking it first labelled every
          result on this machine "AMD Radeon 890M" while the numbers came
          from an RX 7600M XT.
        """
        # Same parser the per-device sampling uses, keyed by device index.
        names = device_names(self.gpu_type or "AMD", self.gpu_tool)
        if names:
            return names[min(names)]

        amd_device_mapping = {
            "150e": "Radeon Graphics",
            "7340": "Radeon RX 5700 XT",
            "731f": "Radeon RX 5700",
            "7360": "Radeon RX 6700 XT",
            "73bf": "Radeon RX 6600 XT",
            "73df": "Radeon RX 6600",
            "15c8": "Radeon RX 7600 XT",
            "5450": "Radeon RX 6800 XT",
            "5498": "Radeon RX 6900 XT",
            "gfx906": "Radeon RX 5700 XT",
            "gfx1103": "Radeon 890M",
        }

        device_id = None
        lspci = _resolve_tool("lspci", LSPCI_SEARCH_PATHS)
        output = _run_tool([lspci, "-d", "1002:"], timeout=5) if lspci else None
        if output:
            for line in output.strip().split("\n"):
                if "1002:" in line:
                    parts = line.split("1002:")
                    if len(parts) > 1:
                        device_id = parts[1].split()[0].lower()
                        if device_id in amd_device_mapping:
                            return f"AMD {amd_device_mapping[device_id]}"
                        break

        # Last resort: the CPU brand names the integrated GPU. Only correct
        # when there is no discrete card, which is why it runs last.
        try:
            if cpuinfo is not None:
                brand = cpuinfo.get_cpu_info().get("brand_raw", "")
                if "Radeon" in brand:
                    model = brand.split("Radeon")[1].strip().split()[0]
                    if model:
                        return f"AMD Radeon {model}"
        except OSError:
            pass

        if device_id:
            return f"AMD GPU (1002:{device_id})"
        return "AMD GPU"

    def _detect_intel_gpu_model(self) -> str:
        """Detect Intel GPU model name."""
        lspci = _resolve_tool("lspci", LSPCI_SEARCH_PATHS)
        output = _run_tool([lspci, "-d", "8086::0300"], timeout=5) if lspci else None
        if output:
            line = output.strip().split("\n")[0]
            if "Intel" in line:
                parts = line.split(": ")
                if len(parts) > 1:
                    return parts[1].split("[")[0].strip()
        return "Intel GPU"

    def get_vram_usage(self) -> str:
        """Measure current VRAM usage."""
        if not self.gpu_tool:
            return "N/A"

        if self.gpu_tool == WINDOWS_NATIVE_TOOL:
            device = self.get_windows_device()
            value = device.metrics.vram_gb if device is not None else None
            return str(round(value * 1024)) if value is not None else "N/A"
        try:
            if self.gpu_type == "Apple":
                vram_used_gb = read_apple_gpu_stats()["vram_used_gb"]
                if vram_used_gb is not None:
                    return f"{int(vram_used_gb * 1024)}"
                return "N/A"

            if self.gpu_type == "NVIDIA":
                output = _run_tool(
                    [
                        self.gpu_tool,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                )
                if output is not None:
                    return output.strip().split("\n")[0]

            if self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs":
                    sysfs_path = self._find_amd_sysfs_path()
                    if sysfs_path:
                        vram_file = Path(sysfs_path) / "mem_info_vram_used"
                        if vram_file.exists():
                            vram_bytes = int(vram_file.read_text().strip())
                            mb_used = vram_bytes / (1024 * 1024)
                            return f"{int(mb_used)}"

                output = _run_tool(
                    [self.gpu_tool, "--showmeminfo", "vram"], timeout=5
                )
                if output is not None:
                    for line in output.split("\n"):
                        if "VRAM Total Used Memory" in line:
                            parts = line.split(":")
                            if len(parts) >= 3:
                                bytes_used = parts[-1].strip()
                                try:
                                    mb_used = int(bytes_used) / (1024 * 1024)
                                    return f"{int(mb_used)}"
                                except ValueError:
                                    pass

            if self.gpu_type == "Intel":
                return "N/A"

        except (subprocess.SubprocessError, OSError, ValueError) as error:
            logger.warning("⚠️ VRAM measurement failed: %s", error)

        return "N/A"


__all__ = ["GPUMonitor", "HardwareMonitor"]
