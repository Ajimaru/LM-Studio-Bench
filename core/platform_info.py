"""Cross-platform OS and GPU detection helpers.

Centralizes the platform-specific probes used by the benchmark CLI, the web
dashboard and the hardware monitor so that Linux, macOS and other systems are
handled in one place instead of being spread across call sites.

On macOS the equivalents of the Linux tooling are:

===================  ==========================================
Linux                macOS
===================  ==========================================
``lspci``            ``system_profiler SPDisplaysDataType``
``nvidia-smi``       ``ioreg -c IOAccelerator`` (Apple Silicon)
``/sys/class/drm``   ``ioreg`` / ``sysctl``
``xdg-open``         ``open``
``lm-sensors``       ``powermetrics`` (requires root)
===================  ==========================================
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"
IS_WINDOWS = platform.system() == "Windows"

_SUBPROCESS_TIMEOUT = 5

# Marketing names keyed by the major macOS release. ``platform.mac_ver``
# reports the product version, so only the leading component is looked up.
_MACOS_RELEASE_NAMES = {
    "26": "Tahoe",
    "15": "Sequoia",
    "14": "Sonoma",
    "13": "Ventura",
    "12": "Monterey",
    "11": "Big Sur",
    "10.15": "Catalina",
}

_IOREG_STATS_RE = re.compile(r'"PerformanceStatistics"\s*=\s*\{(?P<body>[^}]*)\}')


def _run_text_command(command: List[str]) -> Optional[str]:
    """Run a command and return stdout, or None when it is unavailable.

    Args:
        command: Argument vector to execute.

    Returns:
        Captured stdout on success, otherwise None.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
            check=False,
        )
    except (subprocess.SubprocessError, OSError) as error:
        logger.debug("Command %s unavailable: %s", command[0], error)
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def get_macos_name_version() -> Tuple[str, str]:
    """Return the macOS product name and version.

    Returns:
        Tuple such as ``("macOS Tahoe", "26.6")``. The marketing name is
        appended only when the release is known.
    """
    product_version = platform.mac_ver()[0] or platform.release()

    major = product_version.split(".", maxsplit=1)[0]
    release_name = _MACOS_RELEASE_NAMES.get(major)
    if release_name is None and product_version.startswith("10."):
        release_name = _MACOS_RELEASE_NAMES.get(
            ".".join(product_version.split(".")[:2])
        )

    name = f"macOS {release_name}" if release_name else "macOS"
    return name, product_version


def get_os_name_version() -> Tuple[Optional[str], Optional[str]]:
    """Return a human readable OS name and version for any platform.

    Linux distributions are resolved through ``distro`` when installed;
    macOS uses ``platform.mac_ver``. Everything else falls back to the
    generic ``platform`` values.
    """
    system = platform.system()

    if system == "Darwin":
        return get_macos_name_version()

    if system == "Linux":
        try:
            import distro  # pylint: disable=import-outside-toplevel

            distro_name = distro.name()
            if distro_name:
                return distro_name, distro.version()
        except (ImportError, OSError):
            logger.debug("distro module unavailable, using platform fallback")

    return system, platform.release()


def get_open_url_command() -> Optional[str]:
    """Return the platform command that opens a URL in the default browser."""
    if IS_MACOS:
        return "open"
    if IS_LINUX:
        return "xdg-open"
    if IS_WINDOWS:
        return "start"
    return None


def detect_apple_gpu() -> Optional[Dict[str, Any]]:
    """Detect the integrated Apple Silicon / Apple-vendored GPU.

    Uses ``system_profiler SPDisplaysDataType`` which is present on every
    macOS install and needs no elevated privileges.

    Returns:
        Mapping with ``model``, ``cores``, ``metal_family`` and ``vendor``,
        or None when no GPU is reported.
    """
    if not IS_MACOS:
        return None

    output = _run_text_command(
        ["system_profiler", "-json", "SPDisplaysDataType"]
    )
    if not output:
        return None

    try:
        displays = json.loads(output).get("SPDisplaysDataType", [])
    except (json.JSONDecodeError, AttributeError) as error:
        logger.debug("Could not parse system_profiler output: %s", error)
        return None

    for entry in displays:
        if not isinstance(entry, dict):
            continue

        model = entry.get("sppci_model") or entry.get("_name")
        if not model:
            continue

        cores_raw = entry.get("sppci_cores")
        try:
            cores = int(cores_raw) if cores_raw is not None else None
        except (TypeError, ValueError):
            cores = None

        metal_family = entry.get("spdisplays_mtlgpufamilysupport")
        if isinstance(metal_family, str):
            metal_family = metal_family.replace("spdisplays_", "")

        vendor_raw = entry.get("spdisplays_vendor", "")
        vendor = str(vendor_raw).replace("sppci_vendor_", "") or "Unknown"

        return {
            "model": model,
            "cores": cores,
            "metal_family": metal_family,
            "vendor": vendor,
        }

    return None


def read_apple_gpu_stats() -> Dict[str, Optional[float]]:
    """Read live Apple GPU statistics via ``ioreg``.

    Apple Silicon uses unified memory, so "VRAM in use" is reported as the
    driver's in-use system memory. Temperature and power draw are only
    exposed by ``powermetrics``, which requires root, and are therefore not
    included here.

    Returns:
        Mapping with ``utilization_percent`` and ``vram_used_gb``. Values are
        None when they cannot be read.
    """
    stats: Dict[str, Optional[float]] = {
        "utilization_percent": None,
        "vram_used_gb": None,
    }

    if not IS_MACOS:
        return stats

    output = _run_text_command(
        ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"]
    )
    if not output:
        return stats

    match = _IOREG_STATS_RE.search(output)
    if not match:
        return stats

    body = match.group("body")

    utilization = re.search(r'"Device Utilization %"\s*=\s*(\d+)', body)
    if utilization:
        stats["utilization_percent"] = float(utilization.group(1))

    in_use = re.search(r'"In use system memory"\s*=\s*(\d+)', body)
    if in_use:
        stats["vram_used_gb"] = float(in_use.group(1)) / (1024**3)

    return stats


def get_apple_unified_memory_gb() -> Optional[float]:
    """Return total unified memory in GB, which doubles as usable VRAM."""
    if not IS_MACOS:
        return None

    output = _run_text_command(["sysctl", "-n", "hw.memsize"])
    if not output:
        return None

    try:
        return round(int(output.strip()) / (1024**3), 2)
    except ValueError:
        return None


def get_metal_driver_version() -> Optional[str]:
    """Return the Metal support level as the macOS GPU "driver version".

    macOS exposes no user-facing GPU driver version; the Metal family
    supported by the GPU plus the OS build is the closest equivalent.
    """
    if not IS_MACOS:
        return None

    gpu_info = detect_apple_gpu()
    metal_family = gpu_info.get("metal_family") if gpu_info else None

    build = _run_text_command(["sw_vers", "-buildVersion"])
    build_version = build.strip() if build else platform.release()

    if metal_family:
        return f"{metal_family} (macOS build {build_version})"
    return f"macOS build {build_version}"


def get_apple_chip_name() -> Optional[str]:
    """Return the Apple Silicon chip name (e.g. ``Apple M4 Max``)."""
    if not IS_MACOS:
        return None

    output = _run_text_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    if output and output.strip():
        return output.strip()
    return None


__all__ = [
    "IS_LINUX",
    "IS_MACOS",
    "IS_WINDOWS",
    "detect_apple_gpu",
    "get_apple_chip_name",
    "get_apple_unified_memory_gb",
    "get_macos_name_version",
    "get_metal_driver_version",
    "get_open_url_command",
    "get_os_name_version",
    "read_apple_gpu_stats",
]
