"""Per-device GPU sampling for systems with more than one GPU.

The aggregate readers in :mod:`tools.hardware_monitor` answer "how hot is the
GPU" by taking the first device they find. That is correct on a single-GPU
box and wrong on a laptop with a discrete card next to an integrated one:
whatever the second device contributes stays invisible, so a model split
across both looks like it uses half the memory it really does.

This module samples every device separately. It is deliberately narrow: it
only handles the two vendor tools that report a device index (``rocm-smi``
and ``nvidia-smi``), and returns an empty list when it cannot tell devices
apart. Callers keep their existing aggregate values in that case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import shutil
import subprocess  # nosec B404 - vendor CLIs are the only source for this data
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

QUERY_TIMEOUT_SECONDS = 5

# "GPU[1]  : Temperature (Sensor edge) (C): 47.0"
ROCM_LINE = re.compile(
    r"GPU\[(?P<index>\d+)\]\s*:\s*(?P<label>[^:]+):\s*(?P<value>[-\d.]+)\s*$"
)


@dataclass
class GpuSample:
    """One reading of one GPU device.

    Attributes:
        index: Device index as reported by the vendor tool.
        name: Human readable device name, when the tool provides one.
        temp_celsius: Edge temperature.
        power_watts: Current power draw.
        vram_gb: Dedicated video memory in use.
        gtt_gb: Memory borrowed from system RAM (AMD only).
    """

    index: int
    name: Optional[str] = None
    temp_celsius: Optional[float] = None
    power_watts: Optional[float] = None
    vram_gb: Optional[float] = None
    gtt_gb: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return the sample as a plain dictionary."""
        return {
            "index": self.index,
            "name": self.name,
            "temp_celsius": self.temp_celsius,
            "power_watts": self.power_watts,
            "vram_gb": self.vram_gb,
            "gtt_gb": self.gtt_gb,
        }


@dataclass
class GpuAggregate:
    """Min/max/avg of one device across a whole benchmark run."""

    index: int
    name: Optional[str] = None
    samples: int = 0
    values: Dict[str, List[float]] = field(default_factory=dict)

    def add(self, sample: GpuSample) -> None:
        """Fold one sample into this aggregate."""
        self.samples += 1
        if sample.name and not self.name:
            self.name = sample.name
        for metric in ("temp_celsius", "power_watts", "vram_gb", "gtt_gb"):
            value = getattr(sample, metric)
            if value is not None:
                self.values.setdefault(metric, []).append(float(value))

    def as_dict(self) -> Dict[str, Any]:
        """Return min/max/avg per metric, omitting metrics never seen."""
        result: Dict[str, Any] = {
            "index": self.index,
            "name": self.name,
            "samples": self.samples,
        }
        for metric, readings in self.values.items():
            if not readings:
                continue
            result[f"{metric}_min"] = min(readings)
            result[f"{metric}_max"] = max(readings)
            result[f"{metric}_avg"] = sum(readings) / len(readings)
        return result


def _run(command: List[str]) -> Optional[str]:
    """Run a vendor CLI and return stdout, or None when it fails.

    The audited call site of this module: ``command`` is always built here
    from a vendor tool path plus literal flags, is passed as an argv list
    with no shell, and never carries user input.
    """
    try:
        # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
        completed = subprocess.run(  # nosec B603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as error:
        logger.debug("GPU query failed (%s): %s", command[0], error)
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout


def _parse_rocm_block(output: str, metric_key: str) -> Dict[int, float]:
    """Extract ``{device index: value}`` from one rocm-smi report."""
    values: Dict[int, float] = {}
    for line in output.splitlines():
        match = ROCM_LINE.match(line.strip())
        if not match:
            continue
        label = match.group("label").lower()
        if metric_key not in label:
            continue
        try:
            values[int(match.group("index"))] = float(match.group("value"))
        except ValueError:
            continue
    return values


def _sample_rocm(tool: str) -> List[GpuSample]:
    """Sample every AMD device via rocm-smi."""
    samples: Dict[int, GpuSample] = {}

    def ensure(index: int) -> GpuSample:
        return samples.setdefault(index, GpuSample(index=index))

    temp_out = _run([tool, "--showtemp"])
    if temp_out:
        for index, value in _parse_rocm_block(temp_out, "temperature").items():
            ensure(index).temp_celsius = value

    power_out = _run([tool, "--showpower"])
    if power_out:
        for index, value in _parse_rocm_block(power_out, "power").items():
            ensure(index).power_watts = value

    vram_out = _run([tool, "--showmeminfo", "vram"])
    if vram_out:
        for index, value in _parse_rocm_block(vram_out, "used memory").items():
            ensure(index).vram_gb = value / 1024**3

    gtt_out = _run([tool, "--showmeminfo", "gtt"])
    if gtt_out:
        for index, value in _parse_rocm_block(gtt_out, "used memory").items():
            ensure(index).gtt_gb = value / 1024**3

    return [samples[index] for index in sorted(samples)]


def _sample_nvidia(tool: str) -> List[GpuSample]:
    """Sample every NVIDIA device via nvidia-smi."""
    output = _run([
        tool,
        "--query-gpu=index,name,temperature.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []

    def to_float(text: str) -> Optional[float]:
        try:
            return float(text)
        except ValueError:
            return None

    samples: List[GpuSample] = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        memory_mb = to_float(parts[4])
        samples.append(
            GpuSample(
                index=index,
                name=parts[1] or None,
                temp_celsius=to_float(parts[2]),
                power_watts=to_float(parts[3]),
                vram_gb=memory_mb / 1024 if memory_mb is not None else None,
            )
        )
    return samples


def _rocm_device_names(tool: str) -> Dict[int, str]:
    """Map device index to card series as reported by rocm-smi."""
    output = _run([tool, "--showproductname"])
    if not output:
        return {}

    names: Dict[int, str] = {}
    for line in output.splitlines():
        match = re.match(
            r"GPU\[(\d+)\]\s*:\s*Card Series:\s*(?P<name>.+?)\s*$",
            line.strip(),
        )
        if match:
            names[int(match.group(1))] = match.group("name")
    return names


def sample_devices(
    gpu_type: Optional[str],
    gpu_tool: Optional[str],
) -> List[GpuSample]:
    """Sample all GPUs of the detected vendor.

    Args:
        gpu_type: Vendor label from GPUMonitor ("AMD", "NVIDIA", ...).
        gpu_tool: Tool path or name from GPUMonitor.

    Returns:
        One sample per device, empty when per-device data is unavailable.
    """
    if not gpu_tool:
        return []

    tool_name = gpu_tool.rsplit("/", 1)[-1]

    if gpu_type == "AMD" and tool_name.startswith("rocm-smi"):
        return _sample_rocm(gpu_tool)

    if gpu_type == "NVIDIA" and tool_name.startswith("nvidia-smi"):
        return _sample_nvidia(gpu_tool)

    # sysfs, intel_gpu_top and macmon do not expose a stable device index
    # here; the aggregate readers stay authoritative for those.
    return []


def device_names(
    gpu_type: Optional[str],
    gpu_tool: Optional[str],
) -> Dict[int, str]:
    """Return human readable names per device index, when available."""
    if not gpu_tool:
        return {}
    tool_name = gpu_tool.rsplit("/", 1)[-1]
    if gpu_type == "AMD" and tool_name.startswith("rocm-smi"):
        return _rocm_device_names(gpu_tool)
    return {}


def detect_device_count(
    gpu_type: Optional[str],
    gpu_tool: Optional[str],
) -> int:
    """Number of GPUs the vendor tool reports, 0 when unknown."""
    if gpu_type == "NVIDIA" and gpu_tool and shutil.which("nvidia-smi"):
        return len(_sample_nvidia(gpu_tool))
    return len(sample_devices(gpu_type, gpu_tool))


def serialize_aggregates(aggregates: List[GpuAggregate]) -> Optional[str]:
    """Serialize per-device aggregates for storage in a TEXT column.

    Returns:
        Compact JSON, or None when there is nothing worth storing.
    """
    payload = [
        aggregate.as_dict()
        for aggregate in aggregates
        if aggregate.samples > 0
    ]
    if not payload:
        return None
    return json.dumps(payload, separators=(",", ":"))
