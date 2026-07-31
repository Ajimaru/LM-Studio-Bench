"""Sudoless Apple Silicon power and temperature sampling via ``macmon``.

macOS exposes GPU temperature and power draw only through ``powermetrics``,
which requires root. `macmon <https://github.com/vladkens/macmon>`_ reads the
same counters through a private macOS API without elevated privileges, which
makes it usable from an unattended benchmark run.

``macmon pipe`` streams newline-delimited JSON, one sample per interval::

    {"gpu_power": 0.258, "temp": {"gpu_temp_avg": 47.1, ...}, ...}

The tool is entirely optional. When it is missing, or when its output cannot
be parsed, callers fall back to leaving temperature and power unset rather
than failing the benchmark.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from typing import Any, Dict, Optional

from core.platform_info import IS_MACOS

logger = logging.getLogger(__name__)

MACMON_BINARY = "macmon"

# macmon is a Homebrew/cargo install, so it is not always on the PATH of a
# GUI-launched process. These are the standard Homebrew prefixes.
_MACMON_SEARCH_PATHS = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
]


def find_macmon() -> Optional[str]:
    """Locate the macmon binary.

    Returns:
        Path or bare command name when found, otherwise None.
    """
    if not IS_MACOS:
        return None

    found = shutil.which(MACMON_BINARY)
    if found:
        return found

    for path in _MACMON_SEARCH_PATHS:
        found = shutil.which(MACMON_BINARY, path=path)
        if found:
            return found

    return None


def is_macmon_available() -> bool:
    """Report whether sudoless power/temperature sampling is possible."""
    return find_macmon() is not None


class MacmonSampler:
    """Background reader for a streaming ``macmon pipe`` process.

    A single long-lived subprocess is spawned and its NDJSON output is read
    on a daemon thread, keeping only the most recent sample. This avoids
    paying process start-up cost on every poll, which a per-metric invocation
    would incur once per second.
    """

    def __init__(self, interval_ms: int = 1000):
        """Initialize the sampler.

        Args:
            interval_ms: macmon sampling interval in milliseconds.
        """
        self.interval_ms = max(int(interval_ms), 100)
        self.binary = find_macmon()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, Any]] = None
        self._running = False

    @property
    def available(self) -> bool:
        """Whether the macmon binary was found."""
        return self.binary is not None

    def start(self) -> bool:
        """Start the background macmon process.

        Returns:
            True when sampling started, False when macmon is unavailable or
            could not be launched.
        """
        if not self.available or self._running:
            return self._running

        command = [
            str(self.binary),
            "pipe",
            "--interval",
            str(self.interval_ms),
        ]

        try:
            self._process = subprocess.Popen(  # nosec B603
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (OSError, subprocess.SubprocessError) as error:
            logger.debug("Could not start macmon: %s", error)
            self._process = None
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name="macmon-sampler",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "🍏 macmon sampling started (sudoless power/temperature, %sms)",
            self.interval_ms,
        )
        return True

    def _read_loop(self) -> None:
        """Consume NDJSON lines and keep the most recent sample."""
        process = self._process
        if process is None or process.stdout is None:
            return

        try:
            for line in process.stdout:
                if not self._running:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping unparsable macmon line")
                    continue

                if isinstance(sample, dict):
                    with self._lock:
                        self._latest = sample
        except (OSError, ValueError) as error:
            logger.debug("macmon reader stopped: %s", error)
        finally:
            try:
                process.stdout.close()
            except OSError:
                pass

    def stop(self) -> None:
        """Terminate the macmon process and stop the reader thread."""
        self._running = False

        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=2)
            except (subprocess.SubprocessError, OSError, TimeoutError):
                try:
                    process.kill()
                except (subprocess.SubprocessError, OSError):
                    pass

        if self._thread is not None:
            self._thread.join(timeout=2)

        self._process = None
        self._thread = None

    def latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent sample, or None when nothing was read yet."""
        with self._lock:
            return self._latest

    def _read_float(self, *path: str) -> Optional[float]:
        """Read a nested numeric field from the latest sample.

        Args:
            *path: Nested key path, e.g. ``("temp", "gpu_temp_avg")``.

        Returns:
            The value as a float, or None when absent or non-numeric.
        """
        sample: Any = self.latest()
        if sample is None:
            return None

        for key in path:
            if not isinstance(sample, dict) or key not in sample:
                return None
            sample = sample[key]

        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            return None
        return float(sample)

    def get_gpu_temperature(self) -> Optional[float]:
        """Return average GPU temperature in degrees Celsius."""
        return self._read_float("temp", "gpu_temp_avg")

    def get_cpu_temperature(self) -> Optional[float]:
        """Return average CPU temperature in degrees Celsius."""
        return self._read_float("temp", "cpu_temp_avg")

    def get_gpu_power(self) -> Optional[float]:
        """Return GPU power draw in watts.

        This mirrors the meaning of ``nvidia-smi --query-gpu=power.draw`` so
        results stay comparable across platforms.
        """
        return self._read_float("gpu_power")

    def get_cpu_power(self) -> Optional[float]:
        """Return CPU package power draw in watts."""
        return self._read_float("cpu_power")

    def get_ane_power(self) -> Optional[float]:
        """Return Apple Neural Engine power draw in watts."""
        return self._read_float("ane_power")

    def get_total_power(self) -> Optional[float]:
        """Return total SoC power in watts.

        Falls back to whole-system power when the SoC total is unavailable.
        """
        total = self._read_float("all_power")
        if total is None:
            total = self._read_float("sys_power")
        return total

    def get_gpu_utilization(self) -> Optional[float]:
        """Return GPU active residency as a percentage (0-100)."""
        ratio = self._read_float("gpu_active_ratio")
        if ratio is None:
            return None
        return ratio * 100.0

    def __enter__(self) -> "MacmonSampler":
        """Start sampling on context entry."""
        self.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        """Stop sampling on context exit."""
        self.stop()


__all__ = [
    "MacmonSampler",
    "find_macmon",
    "is_macmon_available",
]
