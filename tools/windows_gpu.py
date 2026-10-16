"""Validated Windows GPU probe IPC and NVIDIA/WDDM adapter association."""

from __future__ import annotations

from collections import deque
import csv
import logging
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Literal, Self, TextIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.runtime import PROJECT_ROOT, child_creation_flags
from tools.gpu_devices import parse_gpu_number

LOGGER = logging.getLogger(__name__)
PROBE_TIMEOUT = 20
MAX_SAMPLE_AGE = 3.0
MAX_FRAME_BYTES = 512 * 1024
SENSOR_NAMES = ("temp_celsius", "power_watts", "vram_gb", "gtt_gb")
MetricStatus = Literal["ok", "unsupported", "unavailable", "error"]
PCI_ADDRESS = re.compile(
    r"^[0-9a-f]+:(?P<bus>[0-9a-f]{2}):(?P<device>[0-9a-f]{2})"
    r"\.(?P<function>[0-7])$", re.IGNORECASE
)


class WindowsGpuError(RuntimeError):
    """A missing, failed or malformed native monitoring provider."""


class SensorValues(BaseModel):
    """A sensor batch in degrees Celsius, watts and GiB."""

    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)
    temp_celsius: float | None
    power_watts: float | None = Field(ge=0)
    vram_gb: float | None = Field(ge=0)
    gtt_gb: float | None = Field(ge=0)


class SensorStatus(BaseModel):
    """Explicit availability of each sensor in a batch."""

    model_config = ConfigDict(strict=True, frozen=True)
    temp_celsius: MetricStatus
    power_watts: MetricStatus
    vram_gb: MetricStatus
    gtt_gb: MetricStatus


class WindowsGpuDevice(BaseModel):
    """One proven Windows adapter identity and its current readings."""

    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)
    id: str = Field(min_length=1)
    index: int = Field(ge=0)
    vendor: Literal["AMD", "Intel", "NVIDIA", "Unknown"]
    name: str = Field(min_length=1)
    vram_total_gb: float | None = Field(ge=0)
    shared_total_gb: float | None = Field(ge=0)
    driver_version: str | None
    integrated: bool | None = None
    pci_bus: int | None = Field(default=None, ge=0, le=255)
    pci_device: int | None = Field(default=None, ge=0, le=31)
    pci_function: int | None = Field(default=None, ge=0, le=7)
    nvidia_index: int | None = Field(default=None, ge=0)
    metrics: SensorValues
    status: SensorStatus
    messages: dict[str, str]
    sources: dict[str, str]

    @model_validator(mode="after")
    def validate_sensor_status(self) -> Self:
        """Reject contradictory readings instead of displaying false zeros."""
        for sensor in SENSOR_NAMES:
            value = getattr(self.metrics, sensor)
            status = getattr(self.status, sensor)
            if (status == "ok") != (value is not None):
                raise ValueError(f"Inconsistent {sensor} value/status")
        return self


class WindowsGpuSnapshot(BaseModel):
    """Versioned native IPC envelope with a real acquisition timestamp."""

    model_config = ConfigDict(strict=True, frozen=True, allow_inf_nan=False)
    protocol_version: int
    sequence: int = Field(ge=0)
    timestamp_unix_ms: int = Field(gt=0)
    devices: list[WindowsGpuDevice]
    errors: list[str]

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        """Reject incompatible protocols and duplicate device identities."""
        if self.protocol_version != 1:
            raise ValueError("Unsupported Windows GPU protocol version")
        if len({device.id for device in self.devices}) != len(self.devices):
            raise ValueError("Duplicate GPU adapter identities")
        if len({device.index for device in self.devices}) != len(self.devices):
            raise ValueError("Duplicate GPU adapter indices")
        return self

    def primary_device(self, vendor: str | None = None) -> WindowsGpuDevice | None:
        """Select the existing NVIDIA/AMD/Intel priority without name matching."""
        priority = {"NVIDIA": 0, "AMD": 1, "Intel": 2, "Unknown": 3}
        return min(
            [device for device in self.devices
             if vendor is None or device.vendor == vendor],
            key=lambda device: (
                priority[device.vendor],
                device.nvidia_index is None,
                device.nvidia_index or 0,
                device.integrated is True,
                device.index,
            ),
            default=None,
        )


def probe_path() -> Path:
    """Locate a packaged or locally built native probe.

    Returns:
        Absolute executable path.

    Raises:
        WindowsGpuError: If the native component has not been built/installed.
    """
    bundle = Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    candidates = [
        bundle / "native" / "gpu_probe.exe",
        PROJECT_ROOT / "build" / "windows_gpu" / "bin" / "gpu_probe.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise WindowsGpuError(
        "Native Windows GPU probe is missing. Build native/gpu_probe "
        "or install a Windows package containing the native component."
    )


def parse_snapshot(text: str) -> WindowsGpuSnapshot:
    """Validate one native JSON frame.

    Args:
        text: Complete UTF-8 JSON frame.

    Returns:
        Typed, validated snapshot.

    Raises:
        WindowsGpuError: If the envelope or its readings are malformed.
    """
    if len(text) > MAX_FRAME_BYTES:
        raise WindowsGpuError("Native GPU frame exceeds the protocol size limit")
    try:
        return WindowsGpuSnapshot.model_validate_json(text)
    except ValueError as error:
        raise WindowsGpuError(f"Invalid native GPU response: {error}") from error


def _nvidia_rows(tool: str) -> dict[tuple[int, int, int], list[list[str]]]:
    """Read NVIDIA data keyed by PCI address, never display-name or row order."""
    fields = (
        "pci.bus_id,index,name,temperature.gpu,power.draw,"
        "memory.used,memory.total,driver_version"
    )
    try:
        result = subprocess.run(
            [tool, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=3, check=False, creationflags=child_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise WindowsGpuError(f"NVIDIA telemetry query failed: {error}") from error
    if result.returncode != 0:
        raise WindowsGpuError(
            f"NVIDIA telemetry exited with {result.returncode}: "
            f"{result.stderr.strip()[:1000]}"
        )
    rows: dict[tuple[int, int, int], list[list[str]]] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 8:
            raise WindowsGpuError("Malformed NVIDIA telemetry row")
        row = [field.strip() for field in row]
        address = PCI_ADDRESS.fullmatch(row[0])
        if address is None or not row[1].isdigit():
            raise WindowsGpuError("Invalid NVIDIA PCI identity or device index")
        key = (
            int(address.group("bus"), 16),
            int(address.group("device"), 16),
            int(address.group("function"), 16),
        )
        rows.setdefault(key, []).append(row)
    return rows


def merge_nvidia(
    snapshot: WindowsGpuSnapshot, tool: str | None
) -> WindowsGpuSnapshot:
    """Merge NVIDIA readings only when PCI identity has one unambiguous match.

    Args:
        snapshot: Validated DXGI/WDDM/ADLX/IGCL frame.
        tool: Resolved nvidia-smi executable, if installed.

    Returns:
        Frame preserving non-NVIDIA sensors and explicit NVIDIA query errors.
    """
    if not any(device.vendor == "NVIDIA" for device in snapshot.devices):
        return snapshot
    problem = "nvidia-smi is unavailable"
    rows: dict[tuple[int, int, int], list[list[str]]] = {}
    if tool:
        try:
            rows = _nvidia_rows(tool)
            problem = ""
        except WindowsGpuError as error:
            problem = str(error)

    devices = []
    errors = list(snapshot.errors)
    for device in snapshot.devices:
        if device.vendor != "NVIDIA":
            devices.append(device)
            continue
        data = device.model_dump()
        matches = []
        if (
            device.pci_bus is not None
            and device.pci_device is not None
            and device.pci_function is not None
        ):
            key = (device.pci_bus, device.pci_device, device.pci_function)
            matches = rows.get(key, [])
        failure = problem or (
            "NVIDIA PCI identity cannot be uniquely associated with this LUID"
            if len(matches) != 1 else ""
        )
        if failure:
            for sensor in ("temp_celsius", "power_watts"):
                data["metrics"][sensor] = None
                data["status"][sensor] = "error" if tool else "unavailable"
                data["messages"][sensor] = failure
                data["sources"][sensor] = "nvidia-smi"
            errors.append(f"{device.id}: {failure}")
        else:
            row = matches[0]
            data["nvidia_index"] = int(row[1])
            data["name"] = row[2] or device.name
            data["driver_version"] = row[7] or device.driver_version
            total_mb = parse_gpu_number(row[6])
            if total_mb is not None and total_mb >= 0:
                data["vram_total_gb"] = total_mb / 1024
            for sensor, raw, divisor in (
                ("temp_celsius", row[3], 1),
                ("power_watts", row[4], 1),
                ("vram_gb", row[5], 1024),
            ):
                value = parse_gpu_number(raw)
                invalid_negative = (
                    value is not None and value < 0 and sensor != "temp_celsius"
                )
                if invalid_negative:
                    value = None
                data["metrics"][sensor] = value / divisor if value is not None else None
                data["status"][sensor] = (
                    "ok" if value is not None else
                    "unsupported" if "not supported" in raw.lower() else
                    "unavailable" if raw.upper() in ("N/A", "[N/A]") else "error"
                )
                data["messages"][sensor] = (
                    "" if value is not None else f"NVIDIA reading unavailable: {raw}"
                )
                data["sources"][sensor] = "nvidia-smi"
        devices.append(WindowsGpuDevice.model_validate(data))
    return snapshot.model_copy(update={"devices": devices, "errors": errors})


def query_snapshot(nvidia_tool: str | None = None) -> WindowsGpuSnapshot:
    """Read one bounded native snapshot without starting continuous profiling.

    Args:
        nvidia_tool: Optional resolved NVIDIA CLI.

    Returns:
        Validated snapshot with available NVIDIA fields merged by PCI identity.

    Raises:
        WindowsGpuError: If the native component fails or violates the protocol.
    """
    try:
        result = subprocess.run(
            [str(probe_path()), "--once"],
            capture_output=True, text=True, encoding="utf-8",
            timeout=PROBE_TIMEOUT, check=False, creationflags=child_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise WindowsGpuError(f"Windows GPU probe failed: {error}") from error
    if result.returncode != 0:
        raise WindowsGpuError(
            f"Windows GPU probe exited with {result.returncode}: "
            f"{result.stderr.strip()[:2000]}"
        )
    return merge_nvidia(parse_snapshot(result.stdout), nvidia_tool)


class WindowsGpuSampler:
    """Own a continuously drained native probe only while profiling is active."""

    def __init__(self, nvidia_tool: str | None = None) -> None:
        """Initialize an inactive sampler with an optional NVIDIA CLI."""
        self.nvidia_tool = nvidia_tool
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._raw: WindowsGpuSnapshot | None = None
        self._sample: WindowsGpuSnapshot | None = None
        self._error: str | None = None
        self._seen_errors: set[str] = set()

    @property
    def running(self) -> bool:
        """Whether the sampler owns an active reader thread."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str | None:
        """The latest provider failure, not a fabricated sensor value."""
        with self._lock:
            if self._error is not None:
                return self._error
            if (
                self._raw is not None
                and time.time() - self._raw.timestamp_unix_ms / 1000 > MAX_SAMPLE_AGE
            ):
                return "Native GPU sample is stale"
            return None

    def start(self) -> None:
        """Start one native reader; repeated starts are idempotent."""
        if self.running:
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._read, name="windows-gpu-probe", daemon=True
        )
        self._thread.start()

    def wait_ready(self, timeout: float = PROBE_TIMEOUT) -> bool:
        """Wait for startup or an explicit failure, returning whether data exists."""
        self._ready.wait(timeout)
        return self.raw_snapshot() is not None

    def _record_error(self, message: str) -> None:
        with self._lock:
            changed = self._error != message
            self._error = message
            self._raw = None
            self._sample = None
        if changed:
            LOGGER.warning("%s", message)
        self._ready.set()

    @staticmethod
    def _drain_errors(stream: TextIO) -> None:
        seen: deque[str] = deque(maxlen=64)
        try:
            for line in stream:
                message = line.strip()
                if message and message not in seen:
                    LOGGER.warning("Native GPU probe: %s", message)
                    seen.append(message)
        except (OSError, ValueError) as error:
            if not stream.closed:
                LOGGER.warning("Could not read native GPU diagnostics: %s", error)
        finally:
            stream.close()

    def _read(self) -> None:
        while not self._stop.is_set():
            stderr_thread = None
            try:
                with subprocess.Popen(
                    [str(probe_path()), "--watch", "--interval-ms", "1000"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, encoding="utf-8",
                    creationflags=child_creation_flags(),
                ) as process:
                    with self._lock:
                        self._process = process
                    if process.stdout is None or process.stderr is None:
                        raise WindowsGpuError("Native GPU pipes are unavailable")
                    stderr_thread = threading.Thread(
                        target=self._drain_errors, args=(process.stderr,), daemon=True
                    )
                    stderr_thread.start()
                    try:
                        while not self._stop.is_set():
                            line = process.stdout.readline(MAX_FRAME_BYTES + 1)
                            if not line:
                                break
                            snapshot = parse_snapshot(line)
                            age = time.time() - snapshot.timestamp_unix_ms / 1000
                            if age > MAX_SAMPLE_AGE or age < -1:
                                self._record_error("Discarded stale native GPU frame")
                                continue
                            with self._lock:
                                self._raw = snapshot
                                self._error = None
                            self._ready.set()
                        if not self._stop.is_set():
                            code = process.wait(timeout=PROBE_TIMEOUT)
                            raise WindowsGpuError(
                                f"Native GPU probe stopped with exit code {code}"
                            )
                    finally:
                        if process.poll() is None:
                            process.kill()
                        process.wait(timeout=5)
            except (
                WindowsGpuError, OSError, UnicodeError, subprocess.SubprocessError
            ) as error:
                if not self._stop.is_set():
                    self._record_error(str(error))
            finally:
                if stderr_thread is not None:
                    stderr_thread.join(timeout=2)
                with self._lock:
                    self._process = None
            if self._stop.wait(5):
                break

    def raw_snapshot(self) -> WindowsGpuSnapshot | None:
        """Return only a fresh native frame, including after a long pause."""
        with self._lock:
            snapshot = self._raw
        if snapshot is not None:
            age = time.time() - snapshot.timestamp_unix_ms / 1000
            if -1 <= age <= MAX_SAMPLE_AGE:
                return snapshot
        return None

    def sample(self) -> WindowsGpuSnapshot | None:
        """Sample all adapters without blocking the native pipe on NVIDIA I/O."""
        raw = self.raw_snapshot()
        if raw is None:
            return None
        sample = merge_nvidia(raw, self.nvidia_tool)
        current = self.raw_snapshot()
        if current is None:
            return None
        # Keep other vendors fresh if a NVIDIA query took most of a tick.
        nvidia = {device.id: device for device in sample.devices
                  if device.vendor == "NVIDIA"}
        sample = current.model_copy(update={
            "devices": [nvidia.get(device.id, device) for device in current.devices],
            "errors": sample.errors,
        })
        errors = set(sample.errors)
        for message in errors - self._seen_errors:
            LOGGER.warning("Windows GPU telemetry: %s", message)
        self._seen_errors = errors
        with self._lock:
            self._sample = sample
        return sample

    def latest_sample(self) -> WindowsGpuSnapshot | None:
        """Return the latest merged reading only while it remains fresh."""
        with self._lock:
            sample = self._sample
        if sample is not None:
            age = time.time() - sample.timestamp_unix_ms / 1000
            if -1 <= age <= MAX_SAMPLE_AGE:
                return sample
        return None

    def stop(self) -> None:
        """Stop and reap only this sampler's native process."""
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write("stop\n")
                    process.stdin.flush()
                process.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                LOGGER.warning("Terminating an unresponsive GPU probe: %s", error)
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=7)
            if self._thread.is_alive():
                raise WindowsGpuError("Native GPU reader did not stop")
        with self._lock:
            self._raw = None
            self._sample = None
