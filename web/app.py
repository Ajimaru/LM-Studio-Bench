#!/usr/bin/env python3
"""
FastAPI Web Dashboard for LM Studio Benchmark

Controls benchmark.py via subprocess and provides live monitoring via
WebSocket.
"""

import argparse
import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime
import glob
import hashlib
from io import BytesIO, StringIO
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import sqlite3
import statistics
import subprocess
from subprocess import TimeoutExpired
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote
import uuid
import webbrowser
import zipfile

import cpuinfo
import distro
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from jinja2 import Environment, FileSystemLoader
import psutil
from pydantic import BaseModel

from core.config import DEFAULT_CONFIG
from core.paths import USER_LOGS_DIR, USER_RESULTS_DIR, format_path_for_logs
from core.presets import PresetManager

try:
    from core.version import (
        compare_versions,
        fetch_latest_release,
        format_release_url,
        get_current_version,
    )
except ImportError as e:
    logging.getLogger(__name__).error("❌ Could not import core.version: %s", e)
    get_current_version = None
    fetch_latest_release = None
    compare_versions = None
    format_release_url = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_PAGE_SIZE = A4
    REPORTLAB_AVAILABLE = True
except ImportError:
    colors = None
    REPORTLAB_PAGE_SIZE = None
    landscape = None
    getSampleStyleSheet = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None
    REPORTLAB_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

PROJECT_ROOT = Path(__file__).parent.parent

SCIPY_AVAILABLE = scipy_stats is not None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


GENERIC_API_ERROR = "Internal server error"


def _safe_api_error(
    extras: Optional[Dict[str, Any]] = None,
    message: str = GENERIC_API_ERROR,
) -> Dict[str, Any]:
    """Build a sanitized API error response without exposing internals."""
    response: Dict[str, Any] = {"success": False, "error": message}
    if extras:
        response.update(extras)
    return response


# ============================================================================
# Helper functions
# ============================================================================


def _collect_lms_variants(base_model: str) -> list[dict]:
    """Return a list of variant entries for the given base model.

    The LM Studio CLI output (JSON) contains metadata about each installed
    model, including a ``variants`` array.  ``base_model`` is something like
    ``qwen/qwen2.5-vl-7b`` and we want to return a list of dicts with keys
    ``name``, ``params`` and ``size`` so the dashboard can populate selects.
    If we fail or nothing matches we return an empty list.
    """
    try:
        result = subprocess.run(
            ["lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return []

        data = json.loads(result.stdout)
        out_models: list[dict] = []
        for m in data:
            model_key = m.get("modelKey")
            if not model_key:
                continue
            if not model_key.startswith(base_model):
                continue

            params_str = m.get("paramsString", "?")
            size_bytes = m.get("sizeBytes", 0) or 0
            size_gb = round(size_bytes / 1024**3, 2)
            size_label = f"{size_gb} GB" if size_gb else "--"

            variants = m.get("variants") or []
            if variants:
                for v in variants:
                    out_models.append(
                        {
                            "name": v,
                            "params": params_str,
                            "size": size_label,
                        }
                    )
                return out_models

            out_models.append(
                {
                    "name": model_key,
                    "params": params_str,
                    "size": size_label,
                }
            )
            return out_models
        return out_models
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ):
        return []


def _expand_short_flag_clusters(
    cli_args: list[str],
    combinable: set[str],
) -> list[str]:
    """Expand combined short flags for CLI parsing.

    Example: ``-dh`` becomes ``-d -h`` when both flags are combinable.

    Args:
        cli_args: Raw command line arguments.
        combinable: Allowed short flags for cluster expansion.

    Returns:
        Normalized argument list.
    """
    normalized: list[str] = []
    for arg in cli_args:
        if arg.startswith("--") or not arg.startswith("-"):
            normalized.append(arg)
            continue
        if len(arg) <= 2:
            normalized.append(arg)
            continue

        cluster = arg[1:]
        if all(flag in combinable for flag in cluster):
            normalized.extend(f"-{flag}" for flag in cluster)
        else:
            normalized.append(arg)
    return normalized


# ============================================================================
# System tray helper (GTK3-based, moved to separate module)
# ============================================================================


def setup_webapp_logger():
    """Creates a separate WebApp startup log file"""
    logs_dir = USER_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"webapp_{timestamp}.log"
    latest_link = logs_dir / "webapp_latest.log"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(log_file.name)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    return log_file


# ============================================================================
# Helper functions
# ============================================================================


def find_free_port() -> int:
    """Finds a free port on the system"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        free_port = s.getsockname()[1]
    return free_port


BENCHMARK_SCRIPT = PROJECT_ROOT / "cli" / "benchmark.py"
TEMPLATES_DIR = Path(__file__).parent / "templates"
RESULTS_DIR = USER_RESULTS_DIR
DATABASE_FILE = RESULTS_DIR / "benchmark_cache.db"
METADATA_DATABASE_FILE = RESULTS_DIR / "model_metadata.db"
SCRAPER_SCRIPT = PROJECT_ROOT / "tools" / "scrape_metadata.py"


try:
    from cli.benchmark import BenchmarkCache
except ImportError as e:
    logger.error("❌ Could not import cli.benchmark: %s", e)
    BenchmarkCache = None

CONFIG_DEFAULTS = DEFAULT_CONFIG
LMSTUDIO_HOST = CONFIG_DEFAULTS.get("lmstudio", {}).get("host", "localhost")
LMSTUDIO_PORTS = CONFIG_DEFAULTS.get("lmstudio", {}).get("ports", [1234, 1235])
template_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR), autoescape=True)
DEBUG_MODE = False

LATEST_RELEASE_CACHE: Dict[str, Any] = {
    "data": None,
    "timestamp": 0,
    "ttl_seconds": 3600,
}


def _get_cached_latest_release() -> Optional[dict]:
    """Fetch latest release with caching.

    Uses 1-hour cache to prevent GitHub API rate-limiting.
    Returns None if cache is stale or fetch fails.

    Returns:
        Dict with 'current_version', 'latest_version', 'download_url',
        'is_update_available' or None on failure.
    """
    cache = LATEST_RELEASE_CACHE
    now = time.time()

    if cache["data"] is not None and (now - cache["timestamp"]) < cache["ttl_seconds"]:
        logger.debug("Cache hit for latest release")
        return cache["data"]

    logger.debug("Cache miss or expired, fetching from GitHub")

    current_version_fn = get_current_version
    fetch_latest_release_fn = fetch_latest_release
    compare_versions_fn = compare_versions
    format_release_url_fn = format_release_url

    if (
        current_version_fn is None
        or fetch_latest_release_fn is None
        or compare_versions_fn is None
        or format_release_url_fn is None
    ):
        logger.warning(
            "Version checker functions not available, skipping update check"
        )
        return None

    try:
        current = current_version_fn()
        latest_data = fetch_latest_release_fn()

        if latest_data is None:
            logger.warning("Failed to fetch latest release from GitHub")
            return None

        latest_version = str(latest_data.get("tag_name", "unknown"))
        is_update = compare_versions_fn(current, latest_version)
        download_url = format_release_url_fn(latest_version)

        result = {
            "current_version": current,
            "latest_version": latest_version,
            "download_url": download_url,
            "is_update_available": is_update,
        }

        cache["data"] = result
        cache["timestamp"] = now

        logger.info(
            "Latest release: %s (update available: %s)",
            latest_version,
            is_update,
        )
        return result
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.error("Error in latest release check: %s", exc)
        return None


@dataclass
class _BenchmarkManagerState:
    """Mutable runtime state for the benchmark process lifecycle."""

    process: Optional[subprocess.Popen] = None
    status: str = "idle"
    start_time: Optional[datetime] = None
    current_output: str = ""
    connected_clients: set[WebSocket] = field(default_factory=set)
    benchmark_log_file: Optional[Path] = None
    output_queue: Optional[asyncio.Queue[str]] = None
    output_task: Optional[asyncio.Task] = None
    hardware_history: Dict[str, List[Dict[str, Union[str, float]]]] = field(
        default_factory=lambda: {
            "temperatures": [],
            "power": [],
            "vram": [],
            "gtt": [],
            "cpu": [],
            "ram": [],
        }
    )
    last_hardware_send_time: float = 0.0


class BenchmarkManager:
    """Manage benchmark execution, streaming output, and runtime telemetry."""

    def __init__(self):
        self._state = _BenchmarkManagerState()

    def __getattr__(self, name: str) -> Any:
        """Delegate state fields to the internal runtime state object."""
        state = object.__getattribute__(self, "_state")
        if hasattr(state, name):
            return getattr(state, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Route state field updates to the internal runtime state object."""
        if name == "_state":
            object.__setattr__(self, name, value)
            return
        if "_state" in self.__dict__ and hasattr(self._state, name):
            setattr(self._state, name, value)
            return
        object.__setattr__(self, name, value)

    def is_running(self) -> bool:
        """Return whether the benchmark process is currently active.

        Returns:
            True if a process exists and has not exited, otherwise False.
        """
        return self.process is not None and self.process.poll() is None

    async def _consume_output(self):
        """Continuously reads stdout and puts new chunks into output_queue."""
        if self._state.output_queue is None:
            self._state.output_queue = asyncio.Queue()

        logger.info("🔄 Output consumer task started")

        try:
            loop = asyncio.get_event_loop()

            while True:
                if not self.process or not self.process.stdout:
                    break

                try:
                    line = await loop.run_in_executor(
                        None, self.process.stdout.readline
                    )

                    if not line:
                        break

                    if self._state.benchmark_log_file:
                        try:
                            with open(
                                self._state.benchmark_log_file, "a", encoding="utf-8"
                            ) as f:
                                f.write(line)
                        except (OSError, UnicodeError, ValueError) as log_error:
                            logger.error("❌ Log write error: %s", log_error)

                    self.parse_hardware_metrics(line)

                    await self._state.output_queue.put(line)
                    self._state.current_output += line

                except (OSError, RuntimeError, ValueError) as read_error:
                    logger.error("❌ Read error: %s", read_error)
                    break

            if self.process is not None:
                return_code = self.process.poll()
                if return_code == 0:
                    self._state.status = "completed"
                else:
                    self._state.status = "failed"
                    failure_msg = (
                        f"❌ Benchmark process exited with code {return_code}\n"
                    )
                    await self._state.output_queue.put(failure_msg)
                    self._state.current_output += failure_msg

            logger.info("🔄 Output consumer task ended (EOF reached)")

            if (
                self._state.benchmark_log_file
                and self._state.benchmark_log_file.exists()
            ):
                logger.info(
                    "✅ Benchmark-Log: %s",
                    format_path_for_logs(self._state.benchmark_log_file),
                )

        except asyncio.CancelledError:
            logger.info("ℹ️ Output consumer task cancelled")
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as consume_error:
            logger.error("❌ Error in output consumer: %s", consume_error)

    def drain_output_queue(self) -> str:
        """Fetches all currently available output chunks without blocking."""
        if not self.output_queue:
            return ""
        chunks: List[str] = []
        try:
            while True:
                chunks.append(self.output_queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return "".join(chunks)

    def _validate_cli_arg_value(self, flag: str, value: str) -> str:
        """Validate and sanitize benchmark CLI argument values."""
        if any(char in value for char in ("\x00", "\n", "\r")):
            raise ValueError(f"Invalid control characters in {flag}")
        if len(value) > 2000:
            raise ValueError(f"Value too long for {flag}")

        int_flags = {
            "--runs",
            "--context",
            "--limit",
            "--min-context",
            "--random-models",
            "--top-k",
            "--max-tokens",
            "--n-gpu-layers",
            "--n-batch",
            "--n-threads",
        }
        float_flags = {
            "--max-size",
            "--max-temp",
            "--max-power",
            "--temperature",
            "--top-p",
            "--min-p",
            "--repeat-penalty",
            "--rope-freq-base",
            "--rope-freq-scale",
        }

        if flag in int_flags:
            int(value)
        elif flag in float_flags:
            float(value)

        return value

    def _sanitize_benchmark_args(self, cli_args: list[str]) -> list[str]:
        """Whitelist and sanitize CLI args before subprocess execution."""
        flags_with_values = {
            "--runs",
            "--context",
            "--limit",
            "--prompt",
            "--min-context",
            "--max-size",
            "--quants",
            "--arch",
            "--params",
            "--rank-by",
            "--include-models",
            "--exclude-models",
            "--max-temp",
            "--max-power",
            "--temperature",
            "--top-k",
            "--top-p",
            "--min-p",
            "--repeat-penalty",
            "--max-tokens",
            "--n-gpu-layers",
            "--n-batch",
            "--n-threads",
            "--rope-freq-base",
            "--rope-freq-scale",
            "--kv-cache-quant",
        }
        flags_without_values = {
            "--only-vision",
            "--only-tools",
            "--retest",
            "--test",
            "--dev-mode",
            "--enable-profiling",
            "--disable-gtt",
            "--flash-attention",
            "--no-flash-attention",
            "--use-mmap",
            "--no-mmap",
            "--use-mlock",
            "--debug",
        }
        allowed_flags = flags_with_values | flags_without_values

        sanitized: list[str] = []
        index = 0
        while index < len(cli_args):
            current = str(cli_args[index])

            if current not in allowed_flags:
                raise ValueError(f"Unsupported benchmark argument: {current}")

            sanitized.append(current)

            if current in flags_with_values:
                if index + 1 >= len(cli_args):
                    raise ValueError(f"Missing value for benchmark argument: {current}")
                value = self._validate_cli_arg_value(
                    current, str(cli_args[index + 1])
                )
                sanitized.append(value)
                index += 2
                continue

            index += 1

        return sanitized

    def _sanitize_agent_args(self, cli_args: list[str]) -> list[str]:
        """Whitelist and sanitize capability-agent CLI args."""
        if not cli_args:
            raise ValueError("Missing model id/path for capability benchmark")

        flags_with_values = {
            "--model-name",
            "--capabilities",
            "--max-tests",
            "--random-models",
            "--context-length",
            "--gpu-offload",
            "--temperature",
        }
        flags_without_values = {"--all-models", "--verbose"}
        allowed_flags = flags_with_values | flags_without_values

        sanitized: list[str] = []
        index = 0

        first_arg = str(cli_args[0])
        if not first_arg.startswith("--"):
            model = first_arg.strip()
            if not model:
                raise ValueError(
                    "Invalid empty model id/path for capability benchmark"
                )

            model_pattern = re.compile(r"^[A-Za-z0-9_./:@%+\-=,]+$")
            if not model_pattern.fullmatch(model):
                raise ValueError(f"Unsupported model id/path: {model}")

            sanitized.append(model)
            index = 1

        if index >= len(cli_args):
            return sanitized

        while index < len(cli_args):
            current = str(cli_args[index])

            if current not in allowed_flags:
                raise ValueError(f"Unsupported agent argument: {current}")

            sanitized.append(current)

            if current in flags_with_values:
                if index + 1 >= len(cli_args):
                    raise ValueError(f"Missing value for agent argument: {current}")
                value = self._validate_cli_arg_value(
                    current,
                    str(cli_args[index + 1]),
                )
                sanitized.append(value)
                index += 2
                continue

            index += 1

        has_model = bool(sanitized and not sanitized[0].startswith("--"))
        has_random = "--random-models" in sanitized
        has_all = "--all-models" in sanitized
        if not has_model and not has_random and not has_all:
            raise ValueError(
                "Capability benchmark requires model id/path, "
                "--random-models or --all-models"
            )

        return sanitized

    @staticmethod
    def _build_safe_command(
        sanitized_args: list[str],
        mode: str = "classic",
    ) -> list[str]:
        """Build a fully-validated command list for subprocess execution.

        Verifies that the Python interpreter and benchmark script are absolute
        paths, and that no shell metacharacters remain in any argument.

        Args:
            sanitized_args: Pre-sanitized benchmark CLI arguments.

        Returns:
            A safe command list safe to pass to subprocess.Popen.

        Raises:
            ValueError: If any component contains shell-unsafe characters.
        """
        shell_unsafe_pattern = re.compile(r"[;&|`$<>\\!]")

        interpreter = str(sys.executable)
        if mode == "capability":
            base_cmd = [interpreter, "-u", "-m", "cli.main"]
        else:
            script = str(BENCHMARK_SCRIPT.resolve())
            base_cmd = [interpreter, script]

        for component in base_cmd + sanitized_args:
            if shell_unsafe_pattern.search(component):
                raise ValueError(
                    f"Shell-unsafe characters detected in argument: "
                    f"{component!r}"
                )

        return base_cmd + [str(a) for a in sanitized_args]

    async def start_benchmark(
        self,
        cli_args: list[str],
        mode: str = "classic",
    ) -> bool:
        """Starts a new benchmark process."""
        if self.is_running():
            logger.warning("Benchmark is already running")
            return False
        if self._state.output_queue is None:
            self._state.output_queue = asyncio.Queue()
        try:
            if mode == "capability":
                sanitized_args = self._sanitize_agent_args(cli_args)
            else:
                sanitized_args = self._sanitize_benchmark_args(cli_args)
            self._state.output_queue = asyncio.Queue()
            if self._state.output_task and not self._state.output_task.done():
                self._state.output_task.cancel()

            safe_cmd = self._build_safe_command(sanitized_args, mode=mode)

            self._state.process = subprocess.Popen(
                safe_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                shell=False,
            )
            self._state.status = "running"
            self._state.start_time = datetime.now()
            self._state.current_output = ""
            logs_dir = USER_LOGS_DIR
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = self._state.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{timestamp_str}.log"
            self._state.benchmark_log_file = logs_dir / filename
            self._state.output_task = asyncio.create_task(self._consume_output())

            logger.info("✅ Benchmark started with PID %s", self._state.process.pid)
            logger.info(
                "📝 Benchmark log: %s",
                format_path_for_logs(self._state.benchmark_log_file),
            )
            return True
        except (OSError, RuntimeError, ValueError) as start_error:
            logger.error("❌ Error starting benchmark: %s", start_error)
            self._state.status = "idle"
            return False

    def pause_benchmark(self) -> bool:
        """Pauses running benchmark."""
        if not self.is_running() or not self.process:
            logger.warning("No running benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGSTOP)
            self._state.status = "paused"
            logger.info("⏸️ Benchmark paused")
            return True
        except (OSError, ProcessLookupError, ValueError) as pause_error:
            logger.error("❌ Error pausing: %s", pause_error)
            return False

    def resume_benchmark(self) -> bool:
        """Resumes paused benchmark."""
        if self.status != "paused" or not self.process:
            logger.warning("No paused benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGCONT)
            self._state.status = "running"
            logger.info("▶️ Benchmark resumed")
            return True
        except (OSError, ProcessLookupError, ValueError) as resume_error:
            logger.error("❌ Error resuming: %s", resume_error)
            return False

    def stop_benchmark(self) -> bool:
        """Stops running benchmark."""
        if not self.process:
            logger.warning("No running benchmark")
            return False

        try:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
                logger.info("⏹️ Benchmark stopped (SIGTERM)")
            except TimeoutExpired:
                self.process.kill()
                self.process.wait()
                logger.warning("⏹️ Benchmark forcefully stopped (SIGKILL)")

            self._state.status = "stopped"
            self._state.process = None
            if self._state.output_task and not self._state.output_task.done():
                self._state.output_task.cancel()
            return True
        except (OSError, ProcessLookupError, ValueError) as stop_error:
            logger.error("❌ Error stopping: %s", stop_error)
            return False

    def parse_hardware_metrics(self, output_line: str):
        """Parse hardware metrics from benchmark output"""

        temp_pat = r"GPU\s+Temp\s*:\s*(\d+(?:\.\d+)?)°?C"
        temp_match = re.search(temp_pat, output_line, re.IGNORECASE)
        if temp_match:
            temp_value = float(temp_match.group(1))
            self.hardware_history["temperatures"].append(
                {"timestamp": datetime.now().isoformat(), "value": temp_value}
            )

        power_pat = r"GPU\s+Power\s*:\s*(\d+(?:\.\d+)?)W"
        power_match = re.search(power_pat, output_line, re.IGNORECASE)
        if power_match:
            power_value = float(power_match.group(1))
            self.hardware_history["power"].append(
                {"timestamp": datetime.now().isoformat(), "value": power_value}
            )

        vram_pat = r"GPU\s+VRAM\s*:\s*(\d+(?:\.\d+)?)GB"
        vram_match = re.search(vram_pat, output_line, re.IGNORECASE)
        if vram_match:
            vram_value = float(vram_match.group(1))
            self.hardware_history["vram"].append(
                {"timestamp": datetime.now().isoformat(), "value": vram_value}
            )

        gtt_pat = r"GPU\s+GTT\s*:\s*(\d+(?:\.\d+)?)GB"
        gtt_match = re.search(gtt_pat, output_line, re.IGNORECASE)
        if gtt_match:
            gtt_value = float(gtt_match.group(1))
            self.hardware_history["gtt"].append(
                {"timestamp": datetime.now().isoformat(), "value": gtt_value}
            )

        cpu_pat = r"CPU\s*:\s*(\d+(?:\.\d+)?)%"
        cpu_match = re.search(cpu_pat, output_line, re.IGNORECASE)
        if cpu_match:
            cpu_value = float(cpu_match.group(1))
            self.hardware_history["cpu"].append(
                {"timestamp": datetime.now().isoformat(), "value": cpu_value}
            )

        ram_pattern = r"(?<![V])RAM\s*:\s*(\d+(?:\.\d+)?)GB"
        ram_match = re.search(ram_pattern, output_line, re.IGNORECASE)
        if ram_match:
            ram_value = float(ram_match.group(1))
            self.hardware_history["ram"].append(
                {"timestamp": datetime.now().isoformat(), "value": ram_value}
            )

    async def read_output(self) -> str:
        """Reads ALL available lines from process without blocking"""
        if not self.process or not self.process.stdout:
            return ""

        try:
            loop = asyncio.get_event_loop()
            lines = []

            while True:
                try:
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, self.process.stdout.readline),
                        timeout=0.1,
                    )

                    if output:
                        lines.append(output)
                    else:
                        break
                except asyncio.TimeoutError:
                    break

            if lines:
                combined_output = "".join(lines)
                self._state.current_output += combined_output
                return combined_output

            return ""
        except (OSError, RuntimeError, TypeError, ValueError) as read_error:
            logger.error("❌ Error reading output: %s", read_error)
            return ""

    def update_last_hardware_send_time(self, current_time: float) -> None:
        """Update timestamp of last hardware WebSocket send."""
        self._state.last_hardware_send_time = current_time

    def set_idle_status(self) -> None:
        """Set benchmark status to idle."""
        self._state.status = "idle"


manager = BenchmarkManager()


async def run_metadata_scraper():
    """Runs the metadata scraper script best-effort."""
    if not SCRAPER_SCRIPT.exists():
        logger.warning("⚠️ Scraper script not found: %s", SCRAPER_SCRIPT)
        return
    try:
        await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(SCRAPER_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        logger.info("📝 Metadata scraper executed (missing models only)")
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Scraper timeout reached (>=300s), aborting")
    except subprocess.CalledProcessError as scrape_err:
        logger.warning("⚠️ Scraper error: %s", scrape_err.stderr or scrape_err)
    except (OSError, RuntimeError, ValueError) as scrape_exc:
        logger.warning("⚠️ Scraper execution failed: %s", scrape_exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Handle FastAPI startup/shutdown for benchmark manager."""
    try:
        asyncio.create_task(run_metadata_scraper())
        yield
    finally:
        if manager.is_running():
            logger.info("🛑 Stopping benchmark on shutdown...")
            manager.stop_benchmark()
            await asyncio.sleep(1)


app = FastAPI(
    title="LM Studio Benchmark Dashboard",
    description=("Web dashboard for controlling and monitoring LM Studio benchmarks"),
    lifespan=lifespan,
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# ============================================================================
# Pydantic Models
# ============================================================================


class BenchmarkParams(BaseModel):
    """Parameters for benchmark start"""

    runs: Optional[int] = None
    context: Optional[int] = None
    limit: Optional[int] = None
    prompt: Optional[str] = None
    benchmark_mode: str = "classic"

    min_context: Optional[int] = None
    max_size: Optional[float] = None
    quants: Optional[str] = None
    arch: Optional[str] = None
    params: Optional[str] = None
    rank_by: Optional[str] = None

    only_vision: bool = False
    only_tools: bool = False
    include_models: Optional[str] = None
    exclude_models: Optional[str] = None

    retest: bool = False
    dev_mode: bool = False
    enable_profiling: bool = True
    disable_gtt: bool = False

    max_temp: Optional[float] = None
    max_power: Optional[float] = None

    temperature: Optional[float] = None
    top_k_sampling: Optional[int] = None
    top_p_sampling: Optional[float] = None
    min_p_sampling: Optional[float] = None
    repeat_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

    n_gpu_layers: Optional[int] = -1
    n_batch: Optional[int] = 512
    n_threads: Optional[int] = -1
    flash_attention: Optional[bool] = True
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    use_mmap: Optional[bool] = True
    use_mlock: Optional[bool] = False
    kv_cache_quant: Optional[str] = None

    agent_model: Optional[str] = None
    agent_capabilities: Optional[str] = None
    agent_max_tests: Optional[int] = None


class InferenceParamSet(BaseModel):
    """Set of inference parameters for A/B test"""

    name: str
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    max_tokens: Optional[int] = None

    n_gpu_layers: Optional[int] = None
    n_batch: Optional[int] = None
    n_threads: Optional[int] = None
    flash_attention: Optional[bool] = None
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    kv_cache_quant: Optional[str] = None


class CreateExperimentRequest(BaseModel):
    """Request zum Erstellen eines A/B Experiments"""

    name: str
    model_name: str
    baseline_params: InferenceParamSet
    test_params: InferenceParamSet
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ExperimentResult(BaseModel):
    """A/B Test Result with Statistics"""

    experiment_id: str
    model_name: str
    baseline_data: Dict[str, Any]
    test_data: Dict[str, Any]
    statistical_test: Dict[str, Any]
    winner: str


class PresetSaveRequest(BaseModel):
    """Request to save a new user preset"""

    name: str
    config: Dict[str, Any]


class PresetCompareRequest(BaseModel):
    """Request to compare two presets"""

    preset_a: str
    preset_b: str


# ============================================================================
# Statistical Analysis Functions
# ============================================================================


def calculate_hash(params: Dict[str, Any]) -> str:
    """Erstelle SHA256-Hash aus Parameter-Dictionary"""
    params_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(params_str.encode()).hexdigest()[:16]


def _to_float_scalar(value: Any) -> float:
    """Convert scalar-like SciPy results to float safely."""
    if isinstance(value, tuple):
        if not value:
            raise ValueError("Empty tuple cannot be converted to float")
        value = value[0]
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty list cannot be converted to float")
        value = value[0]
    return float(value)


def perform_ttest(
    baseline_speeds: List[float],
    test_speeds: List[float],
) -> Dict[str, Any]:
    """Perform independent samples t-test"""
    if len(baseline_speeds) < 2 or len(test_speeds) < 2:
        return {
            "test_name": "t-test",
            "p_value": None,
            "t_statistic": None,
            "significant": False,
            "reason": "Insufficient data (min. 2 Proben pro Gruppe)",
        }

    try:
        baseline_var = (
            statistics.variance(baseline_speeds) if len(baseline_speeds) > 1 else 0
        )
        test_var = statistics.variance(test_speeds) if len(test_speeds) > 1 else 0

        if baseline_var == 0 and test_var == 0:
            baseline_mean = statistics.mean(baseline_speeds)
            test_mean = statistics.mean(test_speeds)
            if math.isclose(baseline_mean, test_mean, rel_tol=1e-12):
                return {
                    "test_name": "Welch's t-test",
                    "t_statistic": 0.0,
                    "p_value": 1.0,
                    "significant": False,
                    "alpha": 0.05,
                    "reason": "Identical distributions",
                }

        if SCIPY_AVAILABLE and scipy_stats is not None:
            t_stat_raw, p_value_raw = scipy_stats.ttest_ind(
                baseline_speeds,
                test_speeds,
                equal_var=False,
            )
            t_stat = _to_float_scalar(t_stat_raw)
            p_value = _to_float_scalar(p_value_raw)
            return {
                "test_name": "Welch's t-test",
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "alpha": 0.05,
            }

        baseline_mean = statistics.mean(baseline_speeds)
        test_mean = statistics.mean(test_speeds)

        n1, n2 = len(baseline_speeds), len(test_speeds)
        se = math.sqrt((baseline_var / n1) + (test_var / n2))

        if se == 0:
            return {
                "test_name": "t-test",
                "p_value": None,
                "significant": False,
                "reason": "No variance",
            }

        t_stat = (baseline_mean - test_mean) / se

        p_value = 0.01 if abs(t_stat) > 2.5 else 0.1

        return {
            "test_name": "t-test (approximiert)",
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "alpha": 0.05,
        }

    except (
        statistics.StatisticsError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
    ) as ttest_error:
        logger.error("Error in t-test: %s", ttest_error)
        return {
            "test_name": "t-test",
            "error": str(ttest_error),
            "significant": False,
        }


def match_parameters(row_params: Dict[str, Any], target_params: Dict[str, Any]) -> bool:
    """
    Checks if the parameters of a DB entry match the target parameters;
    None-values in target_params will be ignored.
    """
    for key, target_value in target_params.items():
        if target_value is None:
            continue

        row_value = row_params.get(key)

        if isinstance(target_value, (int, float)) and isinstance(
            row_value, (int, float)
        ):
            if abs(float(row_value) - float(target_value)) > 0.001:
                return False
        else:
            if row_value != target_value:
                return False

    return True


def calculate_effect_size(
    baseline_speeds: List[float], test_speeds: List[float]
) -> Dict[str, Union[float, str]]:
    """Calculate Cohen's d effect size"""
    if not baseline_speeds or not test_speeds:
        return {"cohens_d": 0.0, "effect_magnitude": "negligible"}

    baseline_mean = statistics.mean(baseline_speeds)
    test_mean = statistics.mean(test_speeds)
    if len(baseline_speeds) > 1:
        baseline_var = statistics.variance(baseline_speeds)
    else:
        baseline_var = 0
    if len(test_speeds) > 1:
        test_var = statistics.variance(test_speeds)
    else:
        test_var = 0

    n1, n2 = len(baseline_speeds), len(test_speeds)
    denom = n1 + n2 - 2

    if denom <= 0:
        return {
            "cohens_d": 0.0,
            "effect_magnitude": "negligible",
            "reason": "Insufficient data for effect size",
        }

    pooled_var = ((n1 - 1) * baseline_var + (n2 - 1) * test_var) / denom
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1

    cohens_d = (test_mean - baseline_mean) / pooled_sd if pooled_sd > 0 else 0

    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {"cohens_d": round(cohens_d, 4), "effect_magnitude": magnitude}


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root() -> HTMLResponse:
    """Hauptseite - Dashboard"""
    template = template_env.get_template("dashboard.html.jinja")
    html = template.render(config=CONFIG_DEFAULTS)
    return HTMLResponse(content=html)


@app.get("/api/status")
async def get_status() -> dict:
    """Aktueller Status des Benchmarks"""
    return {
        "status": manager.status,
        "running": manager.is_running(),
        "start_time": (manager.start_time.isoformat() if manager.start_time else None),
        "uptime_seconds": (
            (datetime.now() - manager.start_time).total_seconds()
            if manager.start_time and manager.is_running()
            else None
        ),
        "connected_clients": len(manager.connected_clients),
    }


@app.get("/api/lmstudio/health")
async def get_lmstudio_health() -> dict:
    """LM Studio Healthcheck - Live Status ohne Cache"""
    lmstudio_ports = LMSTUDIO_PORTS

    for lm_port in lmstudio_ports:
        try:
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(f"http://{LMSTUDIO_HOST}:{lm_port}/v1/models")
                if resp.status_code == 200:
                    status_msg = f"online ({LMSTUDIO_HOST}:{lm_port})"
                    return {"ok": True, "status": status_msg, "version": None}
        except (httpx.HTTPError, OSError, ValueError):
            continue

    try:
        result = subprocess.run(
            ["lms", "status"], capture_output=True, text=True, timeout=2, check=False
        )
        text = (result.stdout + result.stderr).lower()

        offline_keywords = [
            "server:  off",
            "server: off",
            "off",
            "not running",
            "stopped",
            "offline",
        ]
        online_keywords = [
            "server:  on",
            "server: on",
            "running",
            "listening",
            "ready",
        ]

        is_offline = (
            any(kw in text for kw in offline_keywords) or result.returncode != 0
        )
        is_online = any(kw in text for kw in online_keywords) and not is_offline

        if is_online:
            return {"ok": True, "status": "online (cli)", "version": None}
    except (FileNotFoundError, TimeoutExpired):
        pass

    return {"ok": False, "status": "offline"}


@app.post("/api/benchmark/start")
async def start_benchmark(params: BenchmarkParams) -> dict:
    """Start a new benchmark."""
    mode = params.benchmark_mode
    if mode not in {"classic", "capability"}:
        return {
            "success": False,
            "status": manager.status,
            "message": "❌ Invalid benchmark mode",
        }

    benchmark_args = []

    if mode == "capability":
        if params.retest:
            if params.limit is not None:
                benchmark_args.extend([
                    "--random-models",
                    str(params.limit),
                ])
            else:
                benchmark_args.append("--all-models")
        elif not params.agent_model:
            return {
                "success": False,
                "status": manager.status,
                "message": "❌ Agent mode requires a model id/path",
            }

        if params.agent_model and not params.retest:
            benchmark_args.append(params.agent_model)
        if params.agent_capabilities:
            benchmark_args.extend([
                "--capabilities",
                params.agent_capabilities,
            ])
        if params.agent_max_tests is not None:
            benchmark_args.extend([
                "--max-tests",
                str(params.agent_max_tests),
            ])
        if params.context:
            benchmark_args.extend([
                "--context-length",
                str(params.context),
            ])
        if params.temperature is not None:
            benchmark_args.extend([
                "--temperature",
                str(params.temperature),
            ])
        if DEBUG_MODE:
            benchmark_args.append("--verbose")

        logger.info("🔧 Capability benchmark args: %s", benchmark_args)
        success = await manager.start_benchmark(benchmark_args, mode=mode)
        message = (
            "✅ Capability benchmark started"
            if success
            else "❌ Error starting capability benchmark"
        )
        if success and params.retest:
            if params.limit is not None:
                message = (
                    "✅ Capability benchmark started "
                    f"(random {params.limit} models)"
                )
            else:
                message = "✅ Capability benchmark started (all models)"
        return {
            "success": success,
            "status": manager.status,
            "message": message,
        }

    if params.runs:
        benchmark_args.extend(["--runs", str(params.runs)])
    if params.context:
        benchmark_args.extend(["--context", str(params.context)])
    if params.limit:
        benchmark_args.extend(["--limit", str(params.limit)])
    if params.prompt:
        benchmark_args.extend(["--prompt", params.prompt])
    if params.min_context:
        benchmark_args.extend(["--min-context", str(params.min_context)])
    if params.max_size:
        benchmark_args.extend(["--max-size", str(params.max_size)])
    if params.quants:
        benchmark_args.extend(["--quants", params.quants])
    if params.arch:
        benchmark_args.extend(["--arch", params.arch])
    if params.params:
        benchmark_args.extend(["--params", params.params])
    if params.rank_by:
        benchmark_args.extend(["--rank-by", params.rank_by])
    if params.include_models:
        benchmark_args.extend(["--include-models", params.include_models])
    if params.exclude_models:
        benchmark_args.extend(["--exclude-models", params.exclude_models])
    if params.only_vision:
        benchmark_args.append("--only-vision")
    if params.only_tools:
        benchmark_args.append("--only-tools")
    if params.retest:
        benchmark_args.append("--retest")
    if params.dev_mode:
        benchmark_args.append("--dev-mode")
    if params.enable_profiling:
        benchmark_args.append("--enable-profiling")
    if params.disable_gtt:
        benchmark_args.append("--disable-gtt")
    if params.max_temp:
        benchmark_args.extend(["--max-temp", str(params.max_temp)])
    if params.max_power:
        benchmark_args.extend(["--max-power", str(params.max_power)])
    if params.temperature is not None:
        benchmark_args.extend(["--temperature", str(params.temperature)])
    if params.top_k_sampling is not None:
        benchmark_args.extend(["--top-k", str(params.top_k_sampling)])
    if params.top_p_sampling is not None:
        benchmark_args.extend(["--top-p", str(params.top_p_sampling)])
    if params.min_p_sampling is not None:
        benchmark_args.extend(["--min-p", str(params.min_p_sampling)])
    if params.repeat_penalty is not None:
        benchmark_args.extend(["--repeat-penalty", str(params.repeat_penalty)])
    if params.max_tokens is not None:
        benchmark_args.extend(["--max-tokens", str(params.max_tokens)])
    if params.n_gpu_layers is not None:
        benchmark_args.extend(["--n-gpu-layers", str(params.n_gpu_layers)])
    if params.n_batch is not None:
        benchmark_args.extend(["--n-batch", str(params.n_batch)])
    if params.n_threads is not None:
        benchmark_args.extend(["--n-threads", str(params.n_threads)])
    if params.flash_attention is not None:
        if params.flash_attention:
            benchmark_args.append("--flash-attention")
        else:
            benchmark_args.append("--no-flash-attention")
    if params.rope_freq_base is not None:
        benchmark_args.extend(["--rope-freq-base", str(params.rope_freq_base)])
    if params.rope_freq_scale is not None:
        benchmark_args.extend(["--rope-freq-scale", str(params.rope_freq_scale)])
    if params.use_mmap is not None:
        if params.use_mmap:
            benchmark_args.append("--use-mmap")
        else:
            benchmark_args.append("--no-mmap")
    if params.use_mlock is not None and params.use_mlock:
        benchmark_args.append("--use-mlock")
    if params.kv_cache_quant:
        benchmark_args.extend(["--kv-cache-quant", params.kv_cache_quant])

    if DEBUG_MODE:
        benchmark_args.append("--debug")

    logger.info("🔧 Benchmark-Args: %s", benchmark_args)
    logger.info(
        "📊 enable_profiling=%s, disable_gtt=%s",
        params.enable_profiling,
        params.disable_gtt,
    )

    success = await manager.start_benchmark(benchmark_args, mode=mode)
    message = "✅ Benchmark started" if success else "❌ Error starting"
    return {"success": success, "status": manager.status, "message": message}


@app.post("/api/benchmark/pause")
async def pause_benchmark() -> dict:
    """Pauses running benchmark"""
    success = manager.pause_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏸️ Paused" if success else "❌ Error",
    }


@app.post("/api/benchmark/resume")
async def resume_benchmark() -> dict:
    """Resumes paused benchmark"""
    success = manager.resume_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "▶️ Resumed" if success else "❌ Error",
    }


@app.post("/api/benchmark/stop")
async def stop_benchmark() -> dict:
    """Stoppt laufenden Benchmark"""
    success = manager.stop_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏹️ Stopped" if success else "❌ Error",
    }


@app.post("/api/system/shutdown")
async def shutdown_system() -> dict:
    """Shutdown the web application gracefully.

    This endpoint allows clients (like the tray app) to request a graceful
    shutdown of the entire dashboard. The benchmark is stopped first, then
    the application exits via SIGTERM signal.
    """
    logger.info("🛑 Shutdown requested via API")

    try:
        success = manager.stop_benchmark()
        logger.info("✅ Benchmark stopped: %s", success)
    except (OSError, ProcessLookupError, RuntimeError, ValueError) as exc:
        logger.warning("Error stopping benchmark during shutdown: %s", exc)

    def _send_shutdown_signal() -> None:
        """Send SIGTERM to self after brief delay to allow response."""
        time.sleep(0.5)
        logger.info("🛑 Sending SIGTERM to shutdown process...")
        os.kill(os.getpid(), signal.SIGTERM)

    shutdown_thread = threading.Thread(target=_send_shutdown_signal, daemon=True)
    shutdown_thread.start()

    return {"success": True, "message": "🛑 Shutting down..."}


@app.get("/api/system/latest-release")
async def get_latest_release() -> dict:
    """Check for updates and return latest release info.

    Fetches latest release from GitHub Releases with 1-hour caching
    to prevent API rate-limiting. Returns version comparison info.

    Returns:
        Dict with 'current_version', 'latest_version', 'download_url',
        'is_update_available' and 'success' flag.
    """
    result = _get_cached_latest_release()

    if result is None:
        return {
            "success": False,
            "current_version": "unknown",
            "latest_version": "unknown",
            "download_url": "",
            "is_update_available": False,
            "message": "Failed to check for updates",
        }

    return {
        "success": True,
        "current_version": result["current_version"],
        "latest_version": result["latest_version"],
        "download_url": result["download_url"],
        "is_update_available": result["is_update_available"],
    }


@app.get("/api/results")
async def get_results() -> dict:
    """Returns all cached benchmark results"""
    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache not available",
            "results": [],
        }

    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()
        results_data = []
        for result in results:
            model_key = f"{result.model_name}@{result.quantization}"

            result_dict = {
                "model_key": model_key,
                "model_name": result.model_name,
                "quantization": result.quantization,
                "gpu_type": result.gpu_type,
                "gpu_offload": result.gpu_offload,
                "vram_mb": result.vram_mb,
                "avg_tokens_per_sec": result.avg_tokens_per_sec,
                "avg_ttft": result.avg_ttft,
                "avg_gen_time": result.avg_gen_time,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "timestamp": result.timestamp,
                "params_size": result.params_size,
                "architecture": result.architecture,
                "max_context_length": result.max_context_length,
                "model_size_gb": result.model_size_gb,
                "has_vision": result.has_vision,
                "has_tools": result.has_tools,
                "tokens_per_sec_per_gb": result.tokens_per_sec_per_gb,
                "tokens_per_sec_per_billion_params": (
                    result.tokens_per_sec_per_billion_params
                ),
                "speed_delta_pct": result.speed_delta_pct,
                "prev_timestamp": result.prev_timestamp,
            }

            if hasattr(result, "temp_celsius_avg") and result.temp_celsius_avg:
                result_dict["temp_celsius_avg"] = result.temp_celsius_avg
            if hasattr(result, "power_watts_avg") and result.power_watts_avg:
                result_dict["power_watts_avg"] = result.power_watts_avg
            if hasattr(result, "gtt_enabled"):
                result_dict["gtt_enabled"] = result.gtt_enabled

            results_data.append(result_dict)

        return {"success": True, "count": len(results_data), "results": results_data}
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error loading results: %s", e)
        return _safe_api_error({"results": []})


@app.get("/api/cache/stats")
async def get_cache_stats() -> dict:
    """Returns cache statistics"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()

        if not results:
            return {
                "success": True,
                "stats": {
                    "total_entries": 0,
                    "avg_tokens_per_sec": 0,
                    "fastest_model": "No data",
                    "fastest_speed": 0,
                    "slowest_model": "No data",
                    "slowest_speed": 0,
                    "db_size_mb": 0,
                },
            }

        speeds = [r.avg_tokens_per_sec for r in results]
        fastest = max(results, key=lambda r: r.avg_tokens_per_sec)
        slowest = min(results, key=lambda r: r.avg_tokens_per_sec)

        if DATABASE_FILE.exists():
            db_size_mb = DATABASE_FILE.stat().st_size / (1024 * 1024)
        else:
            db_size_mb = 0

        fastest_model_key = f"{fastest.model_name}@{fastest.quantization}"
        slowest_model_key = f"{slowest.model_name}@{slowest.quantization}"
        return {
            "success": True,
            "stats": {
                "total_entries": len(results),
                "avg_tokens_per_sec": sum(speeds) / len(speeds),
                "fastest_model": fastest_model_key,
                "fastest_speed": fastest.avg_tokens_per_sec,
                "slowest_model": slowest_model_key,
                "slowest_speed": slowest.avg_tokens_per_sec,
                "db_size_mb": round(db_size_mb, 2),
            },
        }
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error fetching cache statistics: %s", e)
        return _safe_api_error()


@app.delete("/api/cache/{model_key}")
async def delete_cache_entry(model_key: str) -> dict:
    """Deletes a single cache entry"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM benchmark_results WHERE model_key = ?",
            (model_key,),
        )
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            return {"success": False, "error": f"Model {model_key} not found in cache"}

        cursor.execute(
            "DELETE FROM benchmark_results WHERE model_key = ?",
            (model_key,),
        )
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()

        logger.info("🗑️ Cache entry deleted: %s", model_key)
        return {
            "success": True,
            "message": f"✅ {deleted_count} entry(ies) deleted",
            "model_key": model_key,
        }
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error deleting cache entry: %s", e)
        return _safe_api_error()


@app.post("/api/cache/clear")
async def clear_cache() -> dict:
    """Clears entire cache with backup"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        backup_dir = RESULTS_DIR / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_dir / f"benchmark_cache_{timestamp}_backup.db"

        try:
            shutil.copy2(DATABASE_FILE, backup_file)
            logger.info("💾 Backup created: %s", backup_file)
        except (shutil.Error, OSError) as backup_error:
            logger.warning(
                "⚠️ Backup error during copy (will still be cleared): %s", backup_error
            )
            backup_file = None

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM benchmark_results")
        count_before = cursor.fetchone()[0]
        cursor.execute("DELETE FROM benchmark_results")
        conn.commit()
        conn.close()

        logger.warning("⚠️ Cache cleared: %d entries", count_before)
        logger.warning("💾 Backup available at: %s", backup_file)

        return {
            "success": True,
            "message": f"✅ Cache cleared: {count_before} entries deleted",
            "deleted_count": count_before,
            "backup_file": str(backup_file) if backup_file else None,
        }
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error clearing cache: %s", e)
        return _safe_api_error()


@app.get("/api/lmstudio/models")
async def get_lmstudio_models() -> dict:
    """Returns local LM Studio models."""
    try:
        result = subprocess.run(
            ["lms", "ls"], capture_output=True, text=True, timeout=10, check=False
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": f"LM Studio CLI error: {result.stderr}",
                "models": [],
            }

        models: List[dict] = []
        for line in result.stdout.strip().split("\n"):
            text = line.strip()
            if not text or "LLM" in text or "PARAMS" in text or "You have" in text:
                continue

            variant_pattern = r"^([^\s]+(?:/[^\s]+)?)\s+\((\d+)\s+variants?\)"
            match = re.match(variant_pattern, text)
            if not match:
                continue

            base_model = match.group(1)
            models.extend(_collect_lms_variants(base_model))

        return {
            "success": True,
            "models": models,
            "count": len(models),
        }

    except TimeoutExpired:
        return {"success": False, "error": "LM Studio CLI Timeout", "models": []}
    except (OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error("❌ Error fetching LM Studio models: %s", e)
        return _safe_api_error({"models": []})


@app.get("/api/models/installed")
async def get_installed_models() -> dict:
    """Returns installed LM Studio model IDs via ``lms ls --json``."""
    try:
        result = subprocess.run(
            ["lms", "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if result.returncode != 0:
            return {"success": False, "models": []}

        parsed = json.loads(result.stdout)
        model_ids: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            variants = item.get("variants") or []
            if variants:
                for variant in variants:
                    if variant and variant not in seen:
                        model_ids.append(variant)
                        seen.add(variant)
                continue
            model_key = item.get("modelKey")
            if model_key and model_key not in seen:
                model_ids.append(model_key)
                seen.add(model_key)

        return {"success": True, "models": model_ids}
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to parse lms ls --json output: %s", e)
        return {"success": False, "models": []}
    except (OSError, TimeoutExpired) as e:
        logger.error("❌ Failed to list installed models: %s", e)
        return {"success": False, "models": []}


@app.get("/api/comparison/models")
async def get_comparison_models() -> dict:
    """Returns all models with historical data"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available", "models": []}

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT model_name, COUNT(*) as entry_count
            FROM benchmark_results
            GROUP BY model_name
            ORDER BY entry_count DESC, model_name ASC
        """)

        models = []
        for model_name, count in cursor.fetchall():
            cursor.execute(
                """
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (model_name,),
            )
            latest = cursor.fetchone()

            cursor.execute(
                """
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp ASC
                LIMIT 1
            """,
                (model_name,),
            )
            oldest = cursor.fetchone()

            if latest and oldest:
                delta = (
                    ((latest[1] - oldest[1]) / oldest[1] * 100) if oldest[1] > 0 else 0
                )
                models.append(
                    {
                        "model_name": model_name,
                        "entry_count": count,
                        "latest_speed": round(latest[1], 2),
                        "latest_timestamp": latest[0],
                        "oldest_timestamp": oldest[0],
                        "speed_delta_pct": round(delta, 2),
                    }
                )

        conn.close()
        return {"success": True, "models": models}
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error fetching comparison models: %s", e)
        return _safe_api_error({"models": []})


@app.get("/api/comparison/{model_name:path}")
async def get_model_history(model_name: str) -> dict:
    """Returns history for a specific model"""
    model_name = unquote(model_name)

    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache not available",
            "history": [],
        }

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                timestamp, quantization, avg_tokens_per_sec, avg_ttft,
                avg_gen_time, gpu_offload, vram_mb, temperature,
                top_k_sampling, top_p_sampling, min_p_sampling,
                repeat_penalty, max_tokens, num_runs,
                benchmark_duration_seconds, error_count
            FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        """,
            (model_name,),
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "timestamp": row[0],
                    "quantization": row[1],
                    "speed_tokens_sec": round(row[2], 2),
                    "ttft": round(row[3], 3),
                    "gen_time": round(row[4], 3),
                    "gpu_offload": row[5],
                    "vram_mb": row[6],
                    "temperature": row[7],
                    "top_k_sampling": row[8],
                    "top_p_sampling": row[9],
                    "min_p_sampling": row[10],
                    "repeat_penalty": row[11],
                    "max_tokens": row[12],
                    "num_runs": row[13],
                    "benchmark_duration_seconds": row[14],
                    "error_count": row[15],
                }
            )

        if history:
            speeds = [h["speed_tokens_sec"] for h in history]
            stats = {
                "min_speed": round(min(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "total_runs": len(history),
                "first_run": history[0]["timestamp"],
                "last_run": history[-1]["timestamp"],
                "trend": (
                    "up"
                    if speeds[-1] > speeds[0]
                    else "down" if speeds[-1] < speeds[0] else "stable"
                ),
            }
        else:
            stats = {}

        conn.close()
        return {
            "success": True,
            "model_name": model_name,
            "history": history,
            "stats": stats,
        }
    except (
        AttributeError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Error fetching model history: %s", e)
        return _safe_api_error({"history": []})


@app.post("/api/comparison/export/csv")
async def export_comparison_csv(
    request: Request,
    model_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Exports comparison data as CSV with optional filters"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        payload = {}
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            payload = {}

        model_filter = payload.get("model_name", model_name)
        start_filter = payload.get("start_date", start_date)
        end_filter = payload.get("end_date", end_date)
        quant_filters = payload.get("quantizations", []) or []
        if isinstance(quant_filters, str):
            quant_filters = [quant_filters]

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = (
            "SELECT timestamp, model_name, quantization, "
            "avg_tokens_per_sec, avg_ttft, avg_gen_time, "
            "gpu_offload, vram_mb, temperature, top_k_sampling, "
            "top_p_sampling, min_p_sampling, repeat_penalty, "
            "max_tokens, num_runs, benchmark_duration_seconds, "
            "error_count FROM benchmark_results WHERE 1=1"
        )
        params: list = []

        if model_filter:
            query += " AND model_name = ?"
            params.append(model_filter)
        if start_filter:
            query += " AND timestamp >= ?"
            params.append(start_filter)
        if end_filter:
            query += " AND timestamp <= ?"
            params.append(end_filter)
        if quant_filters:
            placeholders = ",".join(["?"] * len(quant_filters))
            query += f" AND quantization IN ({placeholders})"
            params.extend(quant_filters)

        query += " ORDER BY timestamp ASC"
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"success": False, "error": "No data found"}

        def safe_round(value, digits):
            try:
                return round(value, digits)
            except (TypeError, ValueError, OverflowError):
                return value if value is not None else ""

        output = StringIO()
        writer = csv.writer(output)

        headers = [
            "Timestamp",
            "Model",
            "Quantization",
            "Speed (tok/s)",
            "TTFT (ms)",
            "Gen-Time (ms)",
            "GPU-Offload",
            "VRAM (MB)",
            "Temperature",
            "Top-K",
            "Top-P",
            "Min-P",
            "Repeat-Penalty",
            "Max-Tokens",
            "Num-Runs",
            "Duration (s)",
            "Error-Count",
        ]
        writer.writerow(headers)

        for row in rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    safe_round(row[3], 2),
                    safe_round(row[4], 3),
                    safe_round(row[5], 3),
                    safe_round(row[6], 2),
                    row[7],
                    row[8],
                    row[9],
                    row[10],
                    row[11],
                    row[12],
                    row[13],
                    row[14],
                    safe_round(row[15], 2),
                    row[16],
                ]
            )

        csv_content = output.getvalue()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_export_{timestamp}.csv"
        export_file = RESULTS_DIR / filename
        with open(export_file, "w", encoding="utf-8") as f:
            f.write(csv_content)

        logger.info("📊 CSV Export: %s", export_file)

        return {
            "success": True,
            "message": f"CSV exported: {len(rows)} entries",
            "file": str(export_file),
            "url": f"/results/{export_file.name}",
            "rows": len(rows),
            "filters": {
                "model": model_filter,
                "start": start_filter,
                "end": end_filter,
                "quantizations": quant_filters,
            },
            "csv_preview": csv_content.split("\n")[:5],
        }
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ CSV Export Error: %s", e)
        return _safe_api_error()


@app.post("/api/comparison/export/pdf")
async def export_comparison_pdf(request: Request) -> dict:
    """Exports comparison data as simple PDF summary"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        payload = {}
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            payload = {}

        model_filter = payload.get("model_name")
        start_filter = payload.get("start_date")
        end_filter = payload.get("end_date")
        quant_filters = payload.get("quantizations", []) or []
        if isinstance(quant_filters, str):
            quant_filters = [quant_filters]

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = """
            SELECT timestamp, model_name, quantization,
                   avg_tokens_per_sec, avg_ttft,
                   avg_gen_time, gpu_offload, vram_mb, temperature
            FROM benchmark_results
            WHERE 1=1
        """
        params: list = []

        if model_filter:
            query += " AND model_name = ?"
            params.append(model_filter)
        if start_filter:
            query += " AND timestamp >= ?"
            params.append(start_filter)
        if end_filter:
            query += " AND timestamp <= ?"
            params.append(end_filter)
        if quant_filters:
            placeholders = ",".join(["?"] * len(quant_filters))
            query += f" AND quantization IN ({placeholders})"
            params.extend(quant_filters)

        query += " ORDER BY timestamp ASC"
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"success": False, "error": "No data found"}

        speeds = [row[3] for row in rows if row[3] is not None]
        stats = {
            "min_speed": round(min(speeds), 2) if speeds else None,
            "max_speed": round(max(speeds), 2) if speeds else None,
            "avg_speed": round(statistics.mean(speeds), 2) if speeds else None,
            "entries": len(rows),
        }

        def generate_pdf_bytes(lines):
            pdf_bytes = bytearray()
            pdf_bytes.extend(b"%PDF-1.4\n")
            offsets = []

            def add_obj(content: str):
                offsets.append(len(pdf_bytes))
                pdf_bytes.extend(content.encode("latin-1"))

            add_obj("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
            add_obj("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

            stream_lines = []
            y = 770
            for line in lines:
                safe_line = line.replace("(", "\\(").replace(")", "\\)")
                stream_lines.append(f"BT /F1 11 Tf 50 {y} Td ({safe_line}) Tj ET")
                y -= 14
                if y < 60:
                    break
            stream_content = "\n".join(stream_lines).encode("latin-1")

            add_obj(
                "3 0 obj\n<< /Type /Page /Parent 2 0 R "
                "/MediaBox [0 0 612 792] /Contents 4 0 R "
                "/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            )
            add_obj(f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n")
            pdf_bytes.extend(stream_content)
            pdf_bytes.extend(b"\nendstream\nendobj\n")
            add_obj(
                "5 0 obj\n<< /Type /Font /Subtype /Type1 "
                "/BaseFont /Helvetica >>\nendobj\n"
            )

            xref_offset = len(pdf_bytes)
            pdf_bytes.extend(f"xref\n0 {len(offsets) + 1}\n".encode("latin-1"))
            pdf_bytes.extend(b"0000000000 65535 f \n")
            for off in offsets:
                pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
            pdf_bytes.extend(b"trailer\n")
            pdf_bytes.extend(
                f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode("latin-1")
            )
            pdf_bytes.extend(b"startxref\n")
            pdf_bytes.extend(f"{xref_offset}\n".encode("latin-1"))
            pdf_bytes.extend(b"%%EOF")
            return bytes(pdf_bytes)

        quant_str = ", ".join(quant_filters) if quant_filters else "All"
        min_spd = stats["min_speed"]
        max_spd = stats["max_speed"]
        avg_spd = stats["avg_speed"]
        header_lines = [
            "Historical Comparison Export",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {model_filter or 'All models'}",
            f"Time period: {start_filter or '---'} to {end_filter or '---'}",
            f"Quantization: {quant_str}",
            "",
            "Statistics:",
            f"  Min Speed: {min_spd if min_spd is not None else '-'} tok/s",
            f"  Max Speed: {max_spd if max_spd is not None else '-'} tok/s",
            f"  Avg Speed: {avg_spd if avg_spd is not None else '-'} tok/s",
            f"  Runs: {stats['entries']}",
            "",
            "Top 50 Runs:",
        ]

        for row in rows[:50]:
            spd = round(row[3], 2) if row[3] is not None else "-"
            ttft = round(row[4], 3) if row[4] is not None else "-"
            gen = round(row[5], 3) if row[5] is not None else "-"
            header_lines.append(
                f"{row[0]} | {row[1]} | {row[2]} | {spd} tok/s | "
                f"TTFT {ttft} | Gen {gen}"
            )

        pdf_bytes = generate_pdf_bytes(header_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = RESULTS_DIR / f"comparison_export_{timestamp}.pdf"
        with open(export_file, "wb") as f:
            f.write(pdf_bytes)

        logger.info("🧾 PDF Export: %s", export_file)

        return {
            "success": True,
            "message": f"PDF exported: {len(rows)} entries",
            "file": str(export_file),
            "url": f"/results/{export_file.name}",
            "rows": len(rows),
            "stats": stats,
            "filters": {
                "model": model_filter,
                "start": start_filter,
                "end": end_filter,
                "quantizations": quant_filters,
            },
        }
    except (sqlite3.Error, OSError, ValueError, TypeError) as e:
        logger.error("❌ PDF Export Error: %s", e)
        return _safe_api_error()


@app.post("/api/comparison/statistics/{model_name:path}")
async def get_advanced_statistics(model_name: str) -> dict:
    """Calculates advanced statistics (Volatility, Regression, Alerts)"""
    model_name = unquote(model_name)

    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        """,
            (model_name,),
        )

        data = cursor.fetchall()
        conn.close()

        if not data or len(data) < 2:
            return {
                "success": False,
                "error": "Insufficient data for statistical analysis",
            }

        speeds = [row[1] for row in data]
        timestamps = [row[0] for row in data]

        mean = statistics.mean(speeds)
        variance = statistics.variance(speeds) if len(speeds) > 1 else 0
        std_dev = math.sqrt(variance)

        volatility = (std_dev / mean * 100) if mean > 0 else 0

        n = len(speeds)
        x_values = list(range(n))
        x_mean = statistics.mean(x_values)
        y_mean = mean

        numerator = sum((x_values[i] - x_mean) * (speeds[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean

        forecast = []
        for i in range(n, n + 3):
            predicted_speed = slope * i + intercept
            forecast.append(round(max(0, predicted_speed), 2))

        z_scores = []
        if std_dev > 0:
            for speed in speeds:
                z = (speed - mean) / std_dev
                z_scores.append(round(z, 2))

        anomalies = []
        for i, z in enumerate(z_scores):
            if abs(z) > 2:
                anomalies.append(
                    {
                        "index": i,
                        "timestamp": timestamps[i],
                        "speed": speeds[i],
                        "z_score": z,
                        "alert": "🔴 ANOMALY" if abs(z) > 2.5 else "🟠 WARNING",
                    }
                )

        recent_avg = statistics.mean(speeds[-3:]) if len(speeds) >= 3 else speeds[-1]
        overall_avg = mean
        performance_delta = (
            (recent_avg - overall_avg) / overall_avg * 100 if overall_avg > 0 else 0
        )

        alert = ""
        if performance_delta < -10:
            alert = "🔴 PERFORMANCE REGRESSION"
        elif performance_delta > 10:
            alert = "🟢 PERFORMANCE IMPROVEMENT"
        else:
            alert = "⚪ STABLE"

        logger.info(
            "📈 Advanced stats for %s: σ=%.2f, slope=%.4f", model_name, std_dev, slope
        )

        return {
            "success": True,
            "model_name": model_name,
            "basic": {
                "mean": round(mean, 2),
                "median": round(statistics.median(speeds), 2),
                "min": round(min(speeds), 2),
                "max": round(max(speeds), 2),
            },
            "advanced": {
                "std_dev": round(std_dev, 2),
                "variance": round(variance, 2),
                "volatility_pct": round(volatility, 2),
                "coefficient_of_variation": (
                    round(std_dev / mean * 100, 2) if mean > 0 else 0
                ),
            },
            "trend": {
                "slope": round(slope, 4),
                "intercept": round(intercept, 2),
                "direction": (
                    "📈 UPWARD"
                    if slope > 0.01
                    else "📉 DOWNWARD" if slope < -0.01 else "➡️ FLAT"
                ),
            },
            "forecast": {
                "next_3_runs": forecast,
                "confidence": "Medium" if len(speeds) >= 10 else "Low",
            },
            "anomalies": {"count": len(anomalies), "items": anomalies},
            "alert": {
                "status": alert,
                "recent_avg": round(recent_avg, 2),
                "overall_avg": round(overall_avg, 2),
                "delta_pct": round(performance_delta, 2),
            },
        }
    except (sqlite3.Error, ValueError, ZeroDivisionError, TypeError) as e:
        logger.error("❌ Advanced Statistics Error: %s", e)
        return _safe_api_error()


@app.get("/api/output")
async def get_output() -> dict:
    """Gibt aktuellen Output"""
    return {"output": manager.current_output, "status": manager.status}


# ============================================================================
# PRESET MANAGEMENT ENDPOINTS
# ============================================================================

preset_mgr = PresetManager()


@app.get("/api/presets")
async def list_presets() -> dict:
    """List all available presets (readonly + user-created)"""
    try:
        all_presets = preset_mgr.list_presets_detailed()

        result = {
            "success": True,
            "presets": [
                {
                    "name": name,
                    "readonly": is_readonly,
                    "type": "readonly" if is_readonly else "user",
                }
                for name, is_readonly in all_presets
            ],
        }

        logger.info("📜 Listed %s presets", len(all_presets))
        return result
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.error("❌ Error listing presets: %s", e)
        return _safe_api_error()


@app.get("/api/presets/{name}")
async def get_preset(name: str) -> dict:
    """Get a single preset by name"""
    try:
        is_valid_name, _ = preset_mgr.validate_preset_name(name)
        if not is_valid_name and name not in preset_mgr.READONLY_PRESETS:
            return {"success": False, "error": f"Invalid preset name: {name}"}

        preset_config = preset_mgr.load_preset(name)

        all_presets = preset_mgr.list_presets_detailed()
        is_readonly = any(pname == name and ro for pname, ro in all_presets)

        logger.info("📦 Loaded preset: %s", name)
        return {
            "success": True,
            "name": name,
            "readonly": is_readonly,
            "config": preset_config,
        }
    except FileNotFoundError:
        return {"success": False, "error": f"Preset not found: {name}"}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.error("❌ Error loading preset %s: %s", name, e)
        return _safe_api_error()


@app.post("/api/presets")
async def save_preset(request: PresetSaveRequest) -> dict:
    """Save a new user preset"""
    try:
        is_valid_name, _ = preset_mgr.validate_preset_name(request.name)
        if not is_valid_name:
            return {"success": False, "error": f"Invalid preset name: {request.name}"}

        if request.name in preset_mgr.READONLY_PRESETS:
            return {
                "success": False,
                "error": f"Cannot overwrite readonly preset: {request.name}",
            }

        preset_mgr.save_preset(request.name, request.config)

        logger.info("💾 Saved user preset: %s", request.name)
        return {
            "success": True,
            "message": f"Preset '{request.name}' saved successfully",
        }
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        logger.error("❌ Error saving preset %s: %s", request.name, e)
        return _safe_api_error()


@app.delete("/api/presets/{name}")
async def delete_preset(name: str) -> dict:
    """Delete a user preset"""
    try:
        is_valid_name, _ = preset_mgr.validate_preset_name(name)
        if not is_valid_name:
            return {"success": False, "error": f"Invalid preset name: {name}"}

        if name in preset_mgr.READONLY_PRESETS:
            return {"success": False, "error": f"Cannot delete readonly preset: {name}"}

        preset_mgr.delete_preset(name)

        logger.info("🗑️ Deleted user preset: %s", name)
        return {"success": True, "message": f"Preset '{name}' deleted successfully"}
    except FileNotFoundError:
        return {"success": False, "error": f"Preset not found: {name}"}
    except (OSError, PermissionError) as e:
        logger.error("❌ Error deleting preset %s: %s", name, e)
        return _safe_api_error()


@app.post("/api/presets/compare")
async def compare_presets(request: PresetCompareRequest) -> dict:
    """Compare two presets and return differences"""
    try:
        preset_a = preset_mgr.load_preset(request.preset_a)
        preset_b = preset_mgr.load_preset(request.preset_b)

        differences = preset_mgr.compare_presets(preset_a, preset_b)

        logger.info("🔍 Compared presets: %s vs %s", request.preset_a, request.preset_b)

        return {
            "success": True,
            "preset_a": request.preset_a,
            "preset_b": request.preset_b,
            "differences": differences,
        }
    except FileNotFoundError as e:
        return {"success": False, "error": f"Preset not found: {str(e)}"}
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.error(
            "❌ Error comparing presets %s vs %s: %s",
            request.preset_a,
            request.preset_b,
            e,
        )
        return _safe_api_error()


@app.get("/api/presets/export")
async def export_presets() -> dict:
    """Export all user presets as ZIP archive"""
    try:
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            user_presets = [
                name for name, is_ro in preset_mgr.list_presets_detailed() if not is_ro
            ]

            for preset_name in user_presets:
                preset_config = preset_mgr.load_preset(preset_name)
                json_str = json.dumps(preset_config, indent=2)

                zip_file.writestr(f"{preset_name}.json", json_str)

        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()
        zip_base64 = base64.b64encode(zip_bytes).decode("utf-8")

        logger.info("📤 Exported %s user presets", len(user_presets))

        return {
            "success": True,
            "filename": "lmstudio_presets.zip",
            "data": zip_base64,
            "count": len(user_presets),
        }
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        zipfile.BadZipFile,
    ) as e:
        logger.error("❌ Error exporting presets: %s", e)
        return _safe_api_error()


@app.post("/api/presets/import")
async def import_presets(request: Request) -> dict:
    """Import user presets from ZIP archive"""
    try:
        body = await request.json()
        zip_base64 = body.get("data", "")

        if not zip_base64:
            return {"success": False, "error": "No ZIP data provided"}

        zip_bytes = base64.b64decode(zip_base64)
        zip_buffer = BytesIO(zip_bytes)

        imported_count = 0
        skipped = []

        with zipfile.ZipFile(zip_buffer, "r") as zip_file:
            for file_info in zip_file.namelist():
                if not file_info.endswith(".json"):
                    continue

                preset_name = Path(file_info).stem

                is_valid_name, _ = preset_mgr.validate_preset_name(preset_name)
                if not is_valid_name:
                    skipped.append(f"{preset_name} (invalid name)")
                    continue

                if preset_name in preset_mgr.READONLY_PRESETS:
                    skipped.append(f"{preset_name} (readonly)")
                    continue

                json_content = zip_file.read(file_info).decode("utf-8")
                preset_config = json.loads(json_content)

                preset_mgr.save_preset(preset_name, preset_config)
                imported_count += 1

        logger.info("📥 Imported %s presets, skipped %s", imported_count, len(skipped))

        return {"success": True, "imported": imported_count, "skipped": skipped}
    except (
        json.JSONDecodeError,
        ValueError,
        zipfile.BadZipFile,
        OSError,
        binascii.Error,
    ) as e:
        logger.error("❌ Error importing presets: %s", e)
        return _safe_api_error()


# ============================================================================
# A/B TESTING ENDPOINTS (PHASE 15)
# ============================================================================


@app.post("/api/experiments/create")
async def create_experiment(request: CreateExperimentRequest) -> dict:
    """Creates a new A/B Testing Experiment"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        experiment_id = str(uuid.uuid4())[:8]

        baseline_dict = request.baseline_params.model_dump(exclude_none=True)
        test_dict = request.test_params.model_dump(exclude_none=True)

        baseline_hash = calculate_hash(baseline_dict)
        test_hash = calculate_hash(test_dict)

        logger.info("🧪 Experiment created: %s", experiment_id)
        logger.info("   Model: %s", request.model_name)
        logger.info("   Baseline: %s (hash: %s)", baseline_dict, baseline_hash)
        logger.info("   Test: %s (hash: %s)", test_dict, test_hash)

        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": request.model_name,
            "baseline_hash": baseline_hash,
            "test_hash": test_hash,
            "baseline_params": baseline_dict,
            "test_params": test_dict,
            "created_at": datetime.now().isoformat(),
        }
    except (ValueError, TypeError, AttributeError) as e:
        logger.error("❌ Experiment Creation Error: %s", e)
        return _safe_api_error()


@app.get("/api/experiments/{experiment_id}/comparison")
async def get_experiment_comparison(
    experiment_id: str,
    baseline_hash: str,
    test_hash: str,
    model_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Compares two parameter combinations for a model"""
    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache not available",
            "comparison": {},
        }

    try:
        model_name = unquote(model_name)

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = """
            SELECT
                timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                temperature, top_k_sampling, top_p_sampling, min_p_sampling,
                repeat_penalty, max_tokens, num_runs, error_count
            FROM benchmark_results
            WHERE model_name = ?
        """
        params: List[Any] = [model_name]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp ASC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "success": False,
                "error": "No data for this model",
                "comparison": {},
            }

        baseline_data = []
        test_data = []

        for row in rows:
            (
                ts,
                speed,
                ttft,
                gen_time,
                temp,
                topk,
                topp,
                minp,
                penalty,
                maxts,
                _runs,
                _errors,
            ) = row

            params_dict = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts,
            }

            current_hash = calculate_hash(params_dict)

            if current_hash.startswith(baseline_hash[:8]) or baseline_hash.startswith(
                current_hash[:8]
            ):
                baseline_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                    }
                )
            elif current_hash.startswith(test_hash[:8]) or test_hash.startswith(
                current_hash[:8]
            ):
                test_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                    }
                )

        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]

        if not baseline_speeds or not test_speeds:
            error_msg = (
                f"Insufficient data: Baseline={len(baseline_speeds)} entries, "
                f"Test={len(test_speeds)} entries"
            )
            return {"success": False, "error": error_msg, "comparison": {}}

        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": (
                round(statistics.stdev(baseline_speeds), 2)
                if len(baseline_speeds) > 1
                else 0
            ),
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data,
        }

        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": (
                round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0
            ),
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data,
        }

        test_result = perform_ttest(baseline_speeds, test_speeds)
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)
        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        if baseline_mean and baseline_mean != 0:
            delta_pct = (test_mean - baseline_mean) / baseline_mean * 100
        else:
            delta_pct = None
            logger.warning("⚠️ Baseline-Mean is 0 – Delta% will not be calculated")
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            winner = (
                "test"
                if test_mean > baseline_mean
                else ("baseline" if baseline_mean > test_mean else "tie")
            )

        logger.info("🧪 Experiment %s: %s", experiment_id, winner.upper())
        logger.info(
            "   Baseline: %s ± %s tok/s",
            baseline_stats["mean"],
            baseline_stats["std_dev"],
        )
        logger.info("   Test: %s ± %s tok/s", test_stats["mean"], test_stats["std_dev"])
        delta_str = (
            f"{delta_pct:.1f}%" if isinstance(delta_pct, (int, float)) else "n/a"
        )
        logger.info("   Delta: %s | p-value: %s", delta_str, test_result.get("p_value"))

        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (
                    round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None
                ),
                "winner": winner,
                "significant": test_result.get("significant", False),
                "confidence": (
                    "High"
                    if test_result.get("p_value", 1) < 0.01
                    else "Medium" if test_result.get("p_value", 1) < 0.05 else "Low"
                ),
            },
        }
    except (sqlite3.Error, ValueError, ZeroDivisionError, TypeError) as e:
        logger.error("❌ Experiment Comparison Error: %s", e)
        return _safe_api_error({"comparison": {}})


@app.post("/api/experiments/{experiment_id}/comparison")
async def post_experiment_comparison(experiment_id: str, request: Request) -> dict:
    """Compares two parameter sets for a model (payload contains params)"""
    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache not available",
            "comparison": {},
        }

    try:
        payload = await request.json()
        model_name = payload.get("model_name")
        baseline_params = payload.get("baseline_params", {})
        test_params = payload.get("test_params", {})
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")

        if not model_name:
            return {"success": False, "error": "model_name missing", "comparison": {}}

        if "@" in model_name:
            model_name = model_name.split("@")[0]

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = """
            SELECT
                timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                temperature, top_k_sampling, top_p_sampling, min_p_sampling,
                repeat_penalty, max_tokens, num_runs, error_count
            FROM benchmark_results
            WHERE model_name = ?
        """
        params: list = [model_name]
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp ASC"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {
                "success": False,
                "error": "No data for this model",
                "comparison": {},
            }

        baseline_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        for row in rows:
            (
                ts,
                speed,
                ttft,
                gen_time,
                temp,
                topk,
                topp,
                minp,
                penalty,
                maxts,
                _runs,
                _errors,
            ) = row
            params_dict = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts,
            }

            if match_parameters(params_dict, baseline_params):
                baseline_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                    }
                )
            if match_parameters(params_dict, test_params):
                test_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                    }
                )

        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]

        if not baseline_speeds or not test_speeds:
            error_msg = (
                f"Insufficient data: Baseline={len(baseline_speeds)} entries, "
                f"Test={len(test_speeds)} entries"
            )
            return {
                "success": False,
                "error": error_msg,
                "comparison": {},
            }

        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": (
                round(statistics.stdev(baseline_speeds), 2)
                if len(baseline_speeds) > 1
                else 0
            ),
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data,
        }

        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": (
                round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0
            ),
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data,
        }

        test_result = perform_ttest(baseline_speeds, test_speeds)
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)

        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        delta_pct = None
        if baseline_mean and baseline_mean != 0:
            delta_pct = (test_mean - baseline_mean) / baseline_mean * 100
        else:
            logger.warning("⚠️ Baseline-Mean is 0 – Delta% will not be calculated")
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            winner = (
                "test"
                if test_mean > baseline_mean
                else ("baseline" if baseline_mean > test_mean else "tie")
            )

        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (
                    round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None
                ),
                "winner": winner,
                "significant": test_result.get("significant", False),
                "confidence": (
                    "High"
                    if test_result.get("p_value", 1) < 0.01
                    else "Medium" if test_result.get("p_value", 1) < 0.05 else "Low"
                ),
            },
        }
    except (
        json.JSONDecodeError,
        sqlite3.Error,
        ValueError,
        ZeroDivisionError,
        TypeError,
    ) as e:
        logger.error("❌ Experiment Comparison Error: %s", e)
        return _safe_api_error({"comparison": {}})


@app.post("/api/experiments/{experiment_id}/export")
async def export_experiment(experiment_id: str, request: Request) -> dict:
    """Exports experiment results as CSV/PDF"""
    try:
        payload = await request.json()
        export_format = payload.get("format", "csv")
        baseline_data = payload.get("baseline", {})
        test_data = payload.get("test", {})
        comparison = payload.get("comparison", {})
        test_result = payload.get("statistical_test", {})

        if export_format == "csv":
            output = StringIO()
            writer = csv.writer(output)

            writer.writerow(
                [
                    "Experiment ID",
                    "Type",
                    "Mean (tok/s)",
                    "StdDev",
                    "Min",
                    "Max",
                    "Count",
                ]
            )

            writer.writerow(
                [
                    experiment_id,
                    "Baseline",
                    baseline_data.get("mean", "-"),
                    baseline_data.get("std_dev", "-"),
                    baseline_data.get("min", "-"),
                    baseline_data.get("max", "-"),
                    baseline_data.get("count", 0),
                ]
            )

            writer.writerow(
                [
                    experiment_id,
                    "Test",
                    test_data.get("mean", "-"),
                    test_data.get("std_dev", "-"),
                    test_data.get("min", "-"),
                    test_data.get("max", "-"),
                    test_data.get("count", 0),
                ]
            )

            writer.writerow([])
            writer.writerow(["Statistical Test Results"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["p-value", test_result.get("p_value", "-")])
            writer.writerow(["t-statistic", test_result.get("t_statistic", "-")])
            writer.writerow(["Significant", test_result.get("significant", False)])
            writer.writerow(["Winner", comparison.get("winner", "-")])
            writer.writerow(["Delta %", comparison.get("delta_pct", "-")])

            csv_content = output.getvalue()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_exp_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", experiment_id)
            export_file = (
                RESULTS_DIR / f"experiment_{safe_exp_id}_{timestamp}.csv"
            ).resolve()
            if not str(export_file).startswith(str(RESULTS_DIR.resolve())):
                raise ValueError("Path traversal detected in experiment export")

            with open(export_file, "w", encoding="utf-8") as f:
                f.write(csv_content)

            logger.info("📊 CSV Experiment Export: %s", export_file)

            return {
                "success": True,
                "format": "csv",
                "file": str(export_file),
                "url": f"/results/{export_file.name}",
            }

        lines = [
            f"Experiment {experiment_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Baseline Results:",
            f"  Mean: {baseline_data.get('mean')} "
            f"± {baseline_data.get('std_dev')} tok/s",
            f"  Range: {baseline_data.get('min')} - {baseline_data.get('max')}",
            f"  Runs: {baseline_data.get('count')}",
            "",
            "Test Results:",
            f"  Mean: {test_data.get('mean')} ± {test_data.get('std_dev')} tok/s",
            f"  Range: {test_data.get('min')} - {test_data.get('max')}",
            f"  Runs: {test_data.get('count')}",
            "",
            "Statistical Analysis:",
            f"  p-value: {test_result.get('p_value', '-')}",
            f"  Significant: {test_result.get('significant', False)}",
            f"  Winner: {comparison.get('winner', '-')}",
            f"  Performance Delta: {comparison.get('delta_pct', '-')}%",
        ]

        def generate_simple_pdf(text_lines):
            pdf_bytes = bytearray()
            pdf_bytes.extend(b"%PDF-1.4\n")
            offsets = []

            def add_obj(content: str):
                offsets.append(len(pdf_bytes))
                pdf_bytes.extend(content.encode("latin-1"))

            add_obj("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
            add_obj("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

            stream_lines = []
            y = 770
            for line in text_lines:
                safe_line = line.replace("(", "\\(").replace(")", "\\)")
                stream_lines.append(f"BT /F1 10 Tf 50 {y} Td ({safe_line}) Tj ET")
                y -= 12
            stream_content = "\n".join(stream_lines).encode("latin-1")

            add_obj(
                "3 0 obj\n"
                "<< /Type /Page /Parent 2 0 R "
                "/MediaBox [0 0 612 792] /Contents 4 0 R "
                "/Resources << /Font << /F1 5 0 R >> >> >>\n"
                "endobj\n"
            )
            add_obj(f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n")
            pdf_bytes.extend(stream_content)
            pdf_bytes.extend(b"\nendstream\nendobj\n")
            add_obj(
                "5 0 obj\n"
                "<< /Type /Font /Subtype /Type1 "
                "/BaseFont /Helvetica >>\n"
                "endobj\n"
            )

            xref_offset = len(pdf_bytes)
            pdf_bytes.extend(f"xref\n0 {len(offsets) + 1}\n".encode("latin-1"))
            pdf_bytes.extend(b"0000000000 65535 f \n")
            for off in offsets:
                pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
            pdf_bytes.extend(b"trailer\n")
            pdf_bytes.extend(
                f"<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode("latin-1")
            )
            pdf_bytes.extend(b"startxref\n")
            pdf_bytes.extend(f"{xref_offset}\n".encode("latin-1"))
            pdf_bytes.extend(b"%%EOF")
            return bytes(pdf_bytes)

        pdf_bytes = generate_simple_pdf(lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_exp_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", experiment_id)
        export_file = (
            RESULTS_DIR / f"experiment_{safe_exp_id}_{timestamp}.pdf"
        ).resolve()
        if not str(export_file).startswith(str(RESULTS_DIR.resolve())):
            raise ValueError("Path traversal detected in experiment export")

        with open(export_file, "wb") as f:
            f.write(pdf_bytes)

        logger.info("📋 PDF Experiment Export: %s", export_file)

        return {
            "success": True,
            "format": "pdf",
            "file": str(export_file),
            "url": f"/results/{export_file.name}",
        }

    except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
        logger.error("❌ Experiment Export Error: %s", e)
        return _safe_api_error()


@app.post("/api/experiments/run")
async def run_experiment(request: Request) -> dict:
    """Actively runs A/B experiment: two benchmarks with specified parameters"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        payload = await request.json()
        experiment_name = payload.get("experiment_name", "A/B Test")
        model_name = payload.get("model_name")
        baseline_params = payload.get("baseline_params", {})
        test_params = payload.get("test_params", {})
        runs = payload.get("runs", 3)
        context = payload.get("context", 2048)
        prompt = payload.get("prompt", "Explain machine learning in 3 sentences")

        if not model_name:
            return {"success": False, "error": "model_name missing"}

        if "@" in model_name:
            model_name = model_name.split("@")[0]

        logger.info("🎯 Normalized model_name: %s", model_name)

        baseline_params.pop("name", None)
        test_params.pop("name", None)

        def build_args(param_set: Dict[str, Any]) -> List[str]:
            benchmark_args: List[str] = []
            benchmark_args.extend(["--runs", str(runs)])
            benchmark_args.extend(["--context", str(context)])
            benchmark_args.extend(["--limit", "1"])
            benchmark_args.extend(["--prompt", prompt])
            benchmark_args.extend(["--include-models", model_name])
            benchmark_args.append("--retest")
            if param_set.get("temperature") is not None:
                benchmark_args.extend(
                    ["--temperature", str(param_set["temperature"])]
                )
            if param_set.get("top_k") is not None:
                benchmark_args.extend(["--top-k", str(param_set["top_k"])])
            if param_set.get("top_p") is not None:
                benchmark_args.extend(["--top-p", str(param_set["top_p"])])
            if param_set.get("min_p") is not None:
                benchmark_args.extend(["--min-p", str(param_set["min_p"])])
            if param_set.get("repeat_penalty") is not None:
                benchmark_args.extend(
                    ["--repeat-penalty", str(param_set["repeat_penalty"])]
                )
            if param_set.get("max_tokens") is not None:
                benchmark_args.extend(["--max-tokens", str(param_set["max_tokens"])])
            if param_set.get("n_gpu_layers") is not None:
                benchmark_args.extend(
                    ["--n-gpu-layers", str(param_set["n_gpu_layers"])]
                )
            if param_set.get("n_batch") is not None:
                benchmark_args.extend(["--n-batch", str(param_set["n_batch"])])
            if param_set.get("n_threads") is not None:
                benchmark_args.extend(["--n-threads", str(param_set["n_threads"])])
            if param_set.get("flash_attention") is not None:
                if param_set["flash_attention"]:
                    benchmark_args.append("--flash-attention")
                else:
                    benchmark_args.append("--no-flash-attention")
            if param_set.get("rope_freq_base") is not None:
                benchmark_args.extend(
                    ["--rope-freq-base", str(param_set["rope_freq_base"])]
                )
            if param_set.get("rope_freq_scale") is not None:
                benchmark_args.extend(
                    ["--rope-freq-scale", str(param_set["rope_freq_scale"])]
                )
            if param_set.get("use_mmap") is not None:
                if param_set["use_mmap"]:
                    benchmark_args.append("--use-mmap")
                else:
                    benchmark_args.append("--no-mmap")
            if param_set.get("use_mlock") is not None and param_set["use_mlock"]:
                benchmark_args.append("--use-mlock")
            if param_set.get("kv_cache_quant"):
                benchmark_args.extend(
                    ["--kv-cache-quant", param_set["kv_cache_quant"]]
                )
            benchmark_args.append("--enable-profiling")
            return benchmark_args

        async def run_once(cli_args: List[str]) -> bool:
            return await manager.start_benchmark(cli_args)

        baseline_args = build_args(baseline_params)
        logger.info("🎯 Baseline Args: %s", baseline_args)
        baseline_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        baseline_ok = await run_once(baseline_args)
        if not baseline_ok:
            return {"success": False, "error": "Could not start Baseline Benchmark"}

        while manager.is_running():
            await asyncio.sleep(1.0)
        await asyncio.sleep(2.0)

        test_args = build_args(test_params)
        logger.info("🎯 Test Args: %s", test_args)
        test_ok = await run_once(test_args)
        if not test_ok:
            return {"success": False, "error": "Could not start Test Benchmark"}

        while manager.is_running():
            await asyncio.sleep(1.0)
        await asyncio.sleep(2.0)

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                   temperature, top_k_sampling, top_p_sampling, min_p_sampling,
                   repeat_penalty, max_tokens, run_index,
                   n_gpu_layers, n_batch, n_threads, flash_attention,
                   rope_freq_base, rope_freq_scale, use_mmap, use_mlock, kv_cache_quant
            FROM benchmark_results
            WHERE model_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC, run_index DESC
            LIMIT 20
        """,
            (model_name, baseline_start_str),
        )
        all_rows = cursor.fetchall()

        conn.close()

        baseline_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        logger.info("🔍 Found total entries: %s", len(all_rows))

        for row in all_rows:
            (
                ts,
                speed,
                ttft,
                gen_time,
                temp,
                topk,
                topp,
                minp,
                penalty,
                maxts,
                run_idx,
                gpu_layers,
                batch,
                threads,
                flash,
                rope_base,
                rope_scale,
                mmap,
                mlock,
                kv_quant,
            ) = row

            row_params = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts,
                "n_gpu_layers": gpu_layers,
                "n_batch": batch,
                "n_threads": threads,
                "flash_attention": bool(flash) if flash is not None else None,
                "rope_freq_base": rope_base,
                "rope_freq_scale": rope_scale,
                "use_mmap": bool(mmap) if mmap is not None else None,
                "use_mlock": bool(mlock) if mlock is not None else None,
                "kv_cache_quant": kv_quant,
            }

            if match_parameters(row_params, baseline_params):
                baseline_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                        "run_index": run_idx,
                    }
                )
                logger.info(
                    "✅ Baseline Match: run_index=%s, speed=%s, params=%s",
                    run_idx,
                    speed,
                    row_params,
                )
            elif match_parameters(row_params, test_params):
                test_data.append(
                    {
                        "timestamp": ts,
                        "speed": speed,
                        "ttft": ttft,
                        "gen_time": gen_time,
                        "run_index": run_idx,
                    }
                )
                logger.info(
                    "✅ Test Match: run_index=%s, speed=%s, params=%s",
                    run_idx,
                    speed,
                    row_params,
                )
            else:
                logger.info("❌ No Match: run_index=%s, params=%s", run_idx, row_params)

        logger.info(
            "🔍 After filtering: Baseline=%s, Test=%s",
            len(baseline_data),
            len(test_data),
        )

        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]

        if not baseline_speeds or not test_speeds:
            return {
                "success": False,
                "error": (
                    "Insufficient data after execution: "
                    f"Baseline={len(baseline_speeds)}, "
                    f"Test={len(test_speeds)}"
                ),
                "baseline": {"count": len(baseline_speeds)},
                "test": {"count": len(test_speeds)},
            }

        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": (
                round(statistics.stdev(baseline_speeds), 2)
                if len(baseline_speeds) > 1
                else 0
            ),
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data,
        }

        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": (
                round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0
            ),
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data,
        }

        test_result = perform_ttest(baseline_speeds, test_speeds)
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)
        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        delta_pct = None
        if baseline_mean and baseline_mean != 0:
            delta_pct = (test_mean - baseline_mean) / baseline_mean * 100
        else:
            logger.warning(
                "⚠️ Baseline-Mean is 0 – Delta% will not be calculated (active run)"
            )
        winner = "tie"
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            winner = (
                "test"
                if test_mean > baseline_mean
                else ("baseline" if baseline_mean > test_mean else "tie")
            )

        results_data = {
            "success": True,
            "mode": "active",
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (
                    round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None
                ),
                "winner": winner,
                "significant": test_result.get("significant", False),
            },
            "experiment_info": {
                "name": experiment_name,
                "baseline_params": baseline_params,
                "test_params": test_params,
                "runs": runs,
                "context": context,
                "prompt": prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = USER_RESULTS_DIR
            results_dir.mkdir(exist_ok=True)

            json_file = results_dir / f"ab_test_results_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            logger.info("📄 A/B Test JSON saved: %s", json_file)

            csv_file = results_dir / f"ab_test_results_{timestamp}.csv"
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write(f"Experiment Name: {experiment_name}\n")
                f.write("Metric,Baseline,Test,Delta\n")
                f.write(f"Model,{model_name},{model_name},-\n")
                f.write(
                    "Mean Speed (tok/s),"
                    f"{baseline_stats['mean']},"
                    f"{test_stats['mean']},"
                    f"{results_data['comparison']['delta_pct']}%\n"
                )
                f.write(
                    f"Min Speed (tok/s),{baseline_stats['min']},{test_stats['min']},-\n"
                )
                f.write(
                    f"Max Speed (tok/s),{baseline_stats['max']},{test_stats['max']},-\n"
                )
                f.write(
                    f"Std Dev,{baseline_stats['std_dev']},{test_stats['std_dev']},-\n"
                )
                f.write(f"Count,{baseline_stats['count']},{test_stats['count']},-\n")
                f.write(
                    "Temperature,"
                    f"{baseline_params.get('temperature', 'N/A')},"
                    f"{test_params.get('temperature', 'N/A')},-\n"
                )
                f.write(
                    f"Top-K,{baseline_params.get('top_k', 'N/A')},"
                    f"{test_params.get('top_k', 'N/A')},-\n"
                )
                f.write(
                    f"Top-P,{baseline_params.get('top_p', 'N/A')},"
                    f"{test_params.get('top_p', 'N/A')},-\n"
                )
                f.write(f"Winner,-,-,{winner}\n")
                f.write(f"Significant,-,-,{test_result.get('significant', False)}\n")
            logger.info("📊 A/B Test CSV saved: %s", csv_file)

            html_file = results_dir / f"ab_test_results_{timestamp}.html"
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{experiment_name} - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{
            max-width: 1200px; margin: 0 auto; background: white;
            padding: 30px; border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .winner {{ font-weight: bold; color: #27ae60; font-size: 1.2em; }}
        .metric {{ font-weight: 600; }}
        .params {{
            background: #ecf0f1; padding: 15px;
            border-radius: 5px; margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 {experiment_name}</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Timestamp:</strong>
        {results_data['experiment_info']['timestamp']}</p>

        <h2>📊 Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Baseline</th>
                <th>Test</th>
                <th>Delta</th>
            </tr>
            <tr>
                <td class="metric">Mean Speed</td>
                <td>{baseline_stats['mean']} tok/s</td>
                <td>{test_stats['mean']} tok/s</td>
                <td>{results_data['comparison']['delta_pct']}%</td>
            </tr>
            <tr>
                <td class="metric">Min Speed</td>
                <td>{baseline_stats['min']} tok/s</td>
                <td>{test_stats['min']} tok/s</td>
                <td>-</td>
            </tr>
            <tr>
                <td class="metric">Max Speed</td>
                <td>{baseline_stats['max']} tok/s</td>
                <td>{test_stats['max']} tok/s</td>
                <td>-</td>
            </tr>
            <tr>
                <td class="metric">Std Dev</td>
                <td>±{baseline_stats['std_dev']}</td>
                <td>±{test_stats['std_dev']}</td>
                <td>-</td>
            </tr>
            <tr>
                <td class="metric">Sample Count</td>
                <td>{baseline_stats['count']}</td>
                <td>{test_stats['count']}</td>
                <td>-</td>
            </tr>
        </table>

        <h2>⚙️ Parameters</h2>
        <div class="params">
            <h3>Baseline</h3>
            <p>
                Temperature: {baseline_params.get('temperature', 'N/A')},
                Top-K: {baseline_params.get('top_k', 'N/A')},
                Top-P: {baseline_params.get('top_p', 'N/A')}
            </p>
        </div>
        <div class="params">
            <h3>Test</h3>
            <p>
                Temperature: {test_params.get('temperature', 'N/A')},
                Top-K: {test_params.get('top_k', 'N/A')},
                Top-P: {test_params.get('top_p', 'N/A')}
            </p>
        </div>

        <h2>📈 Statistical Analysis</h2>
        <p><strong>Winner:</strong> <span class="winner">{winner.upper()}</span></p>
        <p><strong>Statistically Significant:</strong>
        {test_result.get('significant', False)}</p>
        <p><strong>p-value:</strong> {test_result.get('p_value', 'N/A')}</p>
        <p><strong>Effect Size (Cohen's d):</strong>
        {effect_size.get('cohens_d', 'N/A')}
        ({effect_size.get('effect_magnitude', 'N/A')})</p>
    </div>
</body>
</html>"""
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info("🌐 A/B Test HTML saved: %s", html_file)

            if REPORTLAB_AVAILABLE:
                simple_doc_template = SimpleDocTemplate
                landscape_fn = landscape
                reportlab_page_size = REPORTLAB_PAGE_SIZE
                stylesheet_factory = getSampleStyleSheet
                paragraph_cls = Paragraph
                spacer_cls = Spacer
                table_cls = Table
                table_style_cls = TableStyle
                reportlab_colors = colors

                if (
                    simple_doc_template is None
                    or landscape_fn is None
                    or reportlab_page_size is None
                    or stylesheet_factory is None
                    or paragraph_cls is None
                    or spacer_cls is None
                    or table_cls is None
                    or table_style_cls is None
                    or reportlab_colors is None
                ):
                    logger.warning(
                        "⚠️ reportlab symbols unavailable - PDF export skipped"
                    )
                else:
                    try:
                        pdf_file = results_dir / f"ab_test_results_{timestamp}.pdf"
                        doc = simple_doc_template(
                            str(pdf_file),
                            pagesize=landscape_fn(reportlab_page_size),
                        )
                        elements = []
                        styles = stylesheet_factory()

                        elements.append(
                            paragraph_cls(
                                f"<b>{experiment_name}</b>",
                                styles["Title"],
                            )
                        )
                        elements.append(
                            paragraph_cls(
                                f"Model: {model_name}",
                                styles["Heading2"],
                            )
                        )
                        elements.append(spacer_cls(1, 12))
                        timestamp_str = results_data["experiment_info"]["timestamp"]
                        elements.append(
                            paragraph_cls(
                                f"Timestamp: {timestamp_str}",
                                styles["Normal"],
                            )
                        )
                        elements.append(spacer_cls(1, 20))

                        data = [
                            ["Metric", "Baseline", "Test", "Delta"],
                            [
                                "Mean Speed (tok/s)",
                                f"{baseline_stats['mean']}",
                                f"{test_stats['mean']}",
                                f"{results_data['comparison']['delta_pct']}%",
                            ],
                            [
                                "Min Speed",
                                f"{baseline_stats['min']}",
                                f"{test_stats['min']}",
                                "-",
                            ],
                            [
                                "Max Speed",
                                f"{baseline_stats['max']}",
                                f"{test_stats['max']}",
                                "-",
                            ],
                            [
                                "Std Dev",
                                f"±{baseline_stats['std_dev']}",
                                f"±{test_stats['std_dev']}",
                                "-",
                            ],
                            [
                                "Count",
                                str(baseline_stats["count"]),
                                str(test_stats["count"]),
                                "-",
                            ],
                        ]

                        table = table_cls(data)
                        table.setStyle(
                            table_style_cls(
                                [
                                    (
                                        "BACKGROUND",
                                        (0, 0),
                                        (-1, 0),
                                        reportlab_colors.grey,
                                    ),
                                    (
                                        "TEXTCOLOR",
                                        (0, 0),
                                        (-1, 0),
                                        reportlab_colors.whitesmoke,
                                    ),
                                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("FONTSIZE", (0, 0), (-1, 0), 12),
                                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                                    (
                                        "BACKGROUND",
                                        (0, 1),
                                        (-1, -1),
                                        reportlab_colors.beige,
                                    ),
                                    (
                                        "GRID",
                                        (0, 0),
                                        (-1, -1),
                                        1,
                                        reportlab_colors.black,
                                    ),
                                ]
                            )
                        )
                        elements.append(table)
                        elements.append(spacer_cls(1, 20))
                        elements.append(
                            paragraph_cls(
                                f"<b>Winner:</b> {winner.upper()}",
                                styles["Heading2"],
                            )
                        )
                        elements.append(
                            paragraph_cls(
                                f"<b>Significant:</b> "
                                f"{test_result.get('significant', False)}",
                                styles["Normal"],
                            )
                        )

                        doc.build(elements)
                        logger.info("📑 A/B Test PDF saved: %s", pdf_file)
                    except (OSError, ValueError, TypeError) as pdf_error:
                        logger.error("❌ PDF Export Error: %s", pdf_error)
            else:
                logger.warning("⚠️ reportlab not installed - PDF export skipped")

            results_data["exports"] = {
                "json": str(json_file),
                "csv": str(csv_file),
                "html": str(html_file),
            }
        except (OSError, ValueError, TypeError) as export_error:
            logger.error("❌ Export Error: %s", export_error)

        return results_data

    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as e:
        logger.error("❌ Experiment Run Error: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())
        return _safe_api_error()


@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> dict:
    """Dashboard statistics for Home view"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()

        capabilities_by_model: Dict[str, List[str]] = {}
        distinct_capabilities: set[str] = set()
        try:
            if METADATA_DATABASE_FILE.exists():
                mconn = sqlite3.connect(METADATA_DATABASE_FILE)
                mcur = mconn.cursor()
                mcur.execute(
                    "SELECT model_key, capabilities FROM model_metadata "
                    "WHERE capabilities IS NOT NULL "
                    "AND TRIM(capabilities) <> ''"
                )
                for mk, caps_json in mcur.fetchall():
                    try:
                        caps = json.loads(caps_json) if caps_json else []
                        if isinstance(caps, list):
                            capabilities_by_model[mk] = caps
                            for c in caps:
                                distinct_capabilities.add(c)
                    except (json.JSONDecodeError, ValueError):
                        continue
                mconn.close()
        except sqlite3.Error:
            pass

        lmstudio_health = {"ok": False, "status": "offline"}
        lmstudio_ports = LMSTUDIO_PORTS

        for lm_port in lmstudio_ports:
            try:
                with httpx.Client(timeout=1.5) as client:
                    resp = client.get(f"http://{LMSTUDIO_HOST}:{lm_port}/v1/models")
                    if resp.status_code == 200:
                        lmstudio_health = {
                            "ok": True,
                            "status": f"online ({LMSTUDIO_HOST}:{lm_port})",
                            "version": None,
                        }
                        break
            except (httpx.HTTPError, OSError):
                continue

        if not lmstudio_health["ok"]:
            try:
                result = subprocess.run(
                    ["lms", "status"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                text = (result.stdout + result.stderr).lower()
                offline_keywords = [
                    "server:  off",
                    "server: off",
                    "off",
                    "not running",
                    "stopped",
                    "offline",
                    "no server",
                    "not loaded",
                    "error",
                    "failed",
                ]
                is_offline = (
                    any(kw in text for kw in offline_keywords) or result.returncode != 0
                )
                online_keywords = [
                    "server:  on",
                    "server: on",
                    "running",
                    "listening",
                    "ready",
                ]
                is_online = any(kw in text for kw in online_keywords) and not is_offline
                if is_online:
                    lmstudio_health = {
                        "ok": True,
                        "status": "online (cli)",
                        "version": None,
                    }
            except (subprocess.SubprocessError, OSError):
                pass

        try:
            use_rest = CONFIG_DEFAULTS.get("lmstudio", {}).get("use_rest_api", False)
            lmstudio_health["mode"] = "REST API" if use_rest else "Python SDK"
        except (KeyError, AttributeError):
            lmstudio_health["mode"] = "???"

        system_info = {
            "os": platform.system(),
            "os_version": platform.release(),
            "python_version": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

        if system_info["os"] == "Linux":
            try:

                distro_name = distro.name()
                distro_version = distro.version()
                if distro_name:
                    system_info["os"] = f"{distro_name} {distro_version}".strip()
            except (ImportError, OSError):
                try:
                    with open("/etc/os-release", "r", encoding="utf-8") as f:
                        os_release = {}
                        for line in f:
                            if "=" in line:
                                key, val = line.strip().split("=", 1)
                                os_release[key] = val.strip('"')
                        if "PRETTY_NAME" in os_release:
                            system_info["os"] = os_release["PRETTY_NAME"]
                        elif "NAME" in os_release:
                            version = os_release.get("VERSION", "")
                            system_info["os"] = (
                                f"{os_release['NAME']} {version}".strip()
                            )
                except OSError:
                    pass

        cpu_gpu_series = None
        try:
            cpu_data = cpuinfo.get_cpu_info()
            if "brand_raw" in cpu_data and cpu_data["brand_raw"]:
                raw_cpu = (
                    cpu_data["brand_raw"].replace("®", "").replace("™", "").strip()
                )
                system_info["cpu"] = raw_cpu

                if "Radeon" in raw_cpu:
                    radeon_match = re.search(r"Radeon\s+(\d+[A-Za-z]*)", raw_cpu)
                    if radeon_match:
                        cpu_gpu_series = f"AMD Radeon {radeon_match.group(1)}"
        except (ImportError, OSError, KeyError):
            pass

        gpu_info = None
        try:
            gpu_type = "Unknown"
            gpu_model = "Unknown"
            vram_total_gb = None
            gtt_total_gb = None

            if results:
                gpu_model = results[0].gpu_type
                if (
                    "NVIDIA" in gpu_model
                    or "GeForce" in gpu_model
                    or "RTX" in gpu_model
                    or "GTX" in gpu_model
                ):
                    gpu_type = "NVIDIA"
                elif "AMD" in gpu_model or "Radeon" in gpu_model:
                    gpu_type = "AMD"
                elif "Intel" in gpu_model or "Arc" in gpu_model or "Iris" in gpu_model:
                    gpu_type = "Intel"
                else:
                    gpu_type = gpu_model

            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    timeout=5,
                )
                vram_total_mb = int(
                    output.decode().strip().split("\n", maxsplit=1)[0]
                )
                vram_total_gb = round(vram_total_mb / 1024, 2)
                gpu_type = "NVIDIA"

                try:
                    model_output = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        timeout=5,
                    )
                    gpu_model = model_output.decode().strip().split(
                        "\n", maxsplit=1
                    )[0]
                except (subprocess.SubprocessError, OSError):
                    gpu_model = "NVIDIA GPU"

            except (TimeoutExpired, FileNotFoundError, ValueError):
                pass

            if not vram_total_gb:
                try:
                    rocm_tool = None
                    for path in [
                        "/usr/bin/rocm-smi",
                        "/usr/local/bin/rocm-smi",
                    ] + glob.glob("/opt/rocm-*/bin/rocm-smi"):
                        if Path(path).exists():
                            rocm_tool = path
                            break

                    if not rocm_tool:
                        rocm_tool = "rocm-smi"

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
                        "5780": "Radeon Pro W6800X",
                        "gfx906": "Radeon RX 5700 XT (Navi)",
                        "gfx908": "MI100",
                        "gfx90a": "MI250",
                    }

                    gpu_series = None
                    device_id = None

                    try:
                        lspci_output = subprocess.run(
                            ["lspci", "-d", "1002::0300,1002::0380"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )
                        if lspci_output.returncode == 0 and lspci_output.stdout:
                            for line in lspci_output.stdout.strip().split("\n"):
                                if "1002:" in line:
                                    parts = line.split("1002:")
                                    if len(parts) > 1:
                                        dev_part = parts[1].split()[0]
                                        device_id = dev_part.lower()
                                        lspci_slot = line.split()[0]
                                        try:
                                            detail_output = subprocess.run(
                                                ["lspci", "-s", lspci_slot, "-v"],
                                                capture_output=True,
                                                text=True,
                                                timeout=5,
                                                check=False,
                                            )
                                            if detail_output.returncode == 0:
                                                detail_text = detail_output.stdout
                                                if "Radeon" in detail_text:
                                                    for (
                                                        detail_line
                                                    ) in detail_text.split("\n"):
                                                        if "Radeon" in detail_line:
                                                            gpu_series = (
                                                                detail_line.strip()
                                                            )
                                                            break
                                        except (subprocess.SubprocessError, OSError):
                                            pass
                                        break
                    except (FileNotFoundError, TimeoutExpired):
                        pass

                    if not device_id:
                        try:
                            for gpu_path in Path("/sys/devices").glob(
                                "**/pci*/*/0000:c7:00.0/device"
                            ):
                                with open(gpu_path, "r", encoding="utf-8") as f:
                                    dev_id_hex = f.read().strip()
                                    device_id = dev_id_hex.replace("0x", "")
                                    break
                        except OSError:
                            pass

                    gfx_code = None
                    try:
                        result = subprocess.run(
                            [rocm_tool, "--showproductname"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )
                        if result.returncode == 0:
                            for line in result.stdout.split("\n"):
                                if "GPU[0]" in line:
                                    parts = line.split(":")
                                    if len(parts) > 1:
                                        gfx_code = parts[1].strip()
                                    break
                    except (subprocess.SubprocessError, OSError):
                        pass

                    if cpu_gpu_series:
                        gpu_model = cpu_gpu_series
                    elif gpu_series and "Radeon" in gpu_series:
                        gpu_model = gpu_series
                    elif device_id and device_id in amd_device_mapping:
                        gpu_model = f"AMD {amd_device_mapping[device_id]}"
                    elif gfx_code and gfx_code in amd_device_mapping:
                        gpu_model = f"AMD {amd_device_mapping[gfx_code]}"
                    elif device_id:
                        gpu_model = f"AMD GPU (Device: 1002:{device_id})"
                    elif gfx_code:
                        gpu_model = f"AMD {gfx_code}"
                    else:
                        gpu_model = "AMD GPU"

                    result = subprocess.run(
                        [rocm_tool, "--showmeminfo", "vram"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=False,
                    )

                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "VRAM Total Memory" in line:
                                match = re.search(r":\s*(\d+)\s*$", line.strip())
                                if match:
                                    vram_bytes = int(match.group(1))
                                    vram_total_gb = round(vram_bytes / (1024**3), 2)
                                    gpu_type = "AMD"
                                    break

                    if gpu_type == "AMD":
                        result = subprocess.run(
                            [rocm_tool, "--showmeminfo", "gtt"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                            check=False,
                        )

                        if result.returncode == 0:
                            for line in result.stdout.split("\n"):
                                if "GTT Total Memory" in line:
                                    match = re.search(r":\s*(\d+)\s*$", line.strip())
                                    if match:
                                        gtt_bytes = int(match.group(1))
                                        gtt_total_gb = round(gtt_bytes / (1024**3), 2)
                                        break

                except (TimeoutExpired, FileNotFoundError):
                    pass

            gpu_info = {
                "type": gpu_type,
                "model": gpu_model,
                "vram_gb": vram_total_gb,
                "gtt_gb": gtt_total_gb if gtt_total_gb else system_info["ram_gb"],
                "total_gb": (
                    (vram_total_gb + gtt_total_gb)
                    if (vram_total_gb and gtt_total_gb)
                    else (vram_total_gb or system_info["ram_gb"])
                ),
            }

        except (subprocess.SubprocessError, OSError, ValueError):
            gpu_info = {
                "type": "Unknown",
                "vram_gb": None,
                "gtt_gb": system_info["ram_gb"],
                "total_gb": system_info["ram_gb"],
            }

        cache_stats = {
            "total_models": len(results),
            "total_runs": len(results),
            "db_size_mb": (
                round(DATABASE_FILE.stat().st_size / (1024 * 1024), 2)
                if DATABASE_FILE.exists()
                else 0
            ),
        }

        def calc_percentile(values: List[float], percentile: float) -> float:
            """Calculate percentile with linear interpolation."""
            if not values:
                return 0.0
            sorted_values = sorted(values)
            if len(sorted_values) == 1:
                return float(sorted_values[0])
            position = (len(sorted_values) - 1) * percentile
            lower_idx = int(math.floor(position))
            upper_idx = int(math.ceil(position))
            if lower_idx == upper_idx:
                return float(sorted_values[lower_idx])
            lower_val = sorted_values[lower_idx]
            upper_val = sorted_values[upper_idx]
            weight = position - lower_idx
            return float(lower_val + (upper_val - lower_val) * weight)

        perf_stats = {}
        speed_summary = {
            "min": 0,
            "p50": 0,
            "avg": 0,
            "p95": 0,
            "max": 0,
        }
        if results:
            speeds = [r.avg_tokens_per_sec for r in results]
            perf_stats = {
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "min_speed": round(min(speeds), 2),
            }
            speed_summary = {
                "min": round(min(speeds), 2),
                "p50": round(calc_percentile(speeds, 0.5), 2),
                "avg": round(sum(speeds) / len(speeds), 2),
                "p95": round(calc_percentile(speeds, 0.95), 2),
                "max": round(max(speeds), 2),
            }

        quantization_distribution: Dict[str, int] = {}
        architecture_distribution: Dict[str, int] = {}
        for result in results:
            quant_key = result.quantization or "unknown"
            arch_key = result.architecture or "unknown"
            quantization_distribution[quant_key] = (
                quantization_distribution.get(quant_key, 0) + 1
            )
            architecture_distribution[arch_key] = (
                architecture_distribution.get(arch_key, 0) + 1
            )

        top_models = []
        top_models_extended = []
        fastest_model = None
        if results:
            sorted_results = sorted(
                results, key=lambda r: r.avg_tokens_per_sec, reverse=True
            )
            for i, r in enumerate(sorted_results[:5]):
                base_key = r.model_name.split("@")[0]
                top_models.append(
                    {
                        "model_name": r.model_name,
                        "quantization": r.quantization,
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "vram_mb": r.vram_mb,
                        "params_size": r.params_size,
                        "capabilities": capabilities_by_model.get(r.model_name, []),
                        "source_url": f"https://lmstudio.ai/models/{base_key}",
                    }
                )
                if i == 0:
                    fastest_model = {
                        "name": r.model_name,
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "capabilities": capabilities_by_model.get(r.model_name, []),
                    }

            for r in sorted_results[:10]:
                base_key = r.model_name.split("@")[0]
                top_models_extended.append(
                    {
                        "model_name": r.model_name,
                        "quantization": r.quantization,
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "vram_mb": r.vram_mb,
                        "params_size": r.params_size,
                        "architecture": r.architecture,
                        "capabilities": capabilities_by_model.get(
                            r.model_name, []
                        ),
                        "source_url": f"https://lmstudio.ai/models/{base_key}",
                    }
                )

        efficiency_top = []
        if results:
            efficiency_results = [
                r
                for r in results
                if getattr(r, "tokens_per_sec_per_gb", None) is not None
            ]
            efficiency_sorted = sorted(
                efficiency_results,
                key=lambda r: float(r.tokens_per_sec_per_gb),
                reverse=True,
            )
            for r in efficiency_sorted[:5]:
                efficiency_top.append(
                    {
                        "model_name": r.model_name,
                        "quantization": r.quantization,
                        "tokens_per_sec_per_gb": round(
                            float(r.tokens_per_sec_per_gb), 2
                        ),
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "vram_mb": r.vram_mb,
                    }
                )

        recent_runs = []
        last_run_timestamp = None
        if results:
            sorted_by_time = sorted(results, key=lambda r: r.timestamp, reverse=True)
            for i, r in enumerate(sorted_by_time[:10]):
                base_key = r.model_name.split("@")[0]
                recent_runs.append(
                    {
                        "model_name": r.model_name,
                        "quantization": r.quantization,
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "timestamp": r.timestamp,
                        "gpu_offload": r.gpu_offload,
                        "capabilities": capabilities_by_model.get(r.model_name, []),
                        "source_url": f"https://lmstudio.ai/models/{base_key}",
                    }
                )
                if i == 0:
                    last_run_timestamp = r.timestamp

        return {
            "success": True,
            "system_info": system_info,
            "gpu_info": gpu_info,
            "cache_stats": cache_stats,
            "perf_stats": perf_stats,
            "speed_summary": speed_summary,
            "top_models": top_models,
            "top_models_extended": top_models_extended,
            "recent_runs": recent_runs,
            "fastest_model": fastest_model,
            "quantization_distribution": quantization_distribution,
            "architecture_distribution": architecture_distribution,
            "efficiency_top": efficiency_top,
            "capability_catalog": (
                sorted(list(distinct_capabilities)) if distinct_capabilities else []
            ),
            "last_run": last_run_timestamp,
            "lmstudio": lmstudio_health,
        }
    except (
        AttributeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        sqlite3.Error,
        subprocess.SubprocessError,
        psutil.Error,
        statistics.StatisticsError,
    ) as e:
        logger.error("❌ Error loading dashboard stats: %s", e)
        return _safe_api_error()


# ============================================================================
# WebSocket - Live Streaming
# ============================================================================


@app.websocket("/ws/benchmark")
async def websocket_benchmark(websocket: WebSocket):
    """WebSocket for live-streaming of benchmark output"""
    await websocket.accept()
    manager.connected_clients.add(websocket)

    heartbeat_count = 0

    try:
        logger.info(
            "✅ WebSocket Client connected (Total: %d)",
            len(manager.connected_clients),
        )

        await websocket.send_json(
            {
                "type": "status",
                "status": manager.status,
                "running": manager.is_running(),
            }
        )

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=0.5,
                )
                logger.debug("📨 Client message: %s", data)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                logger.info("⚠️ WebSocket Client has disconnected")
                break
            except (OSError, RuntimeError, ValueError) as e:
                logger.error("❌ WebSocket Receive Error: %s", e)
                break

            output = manager.drain_output_queue()
            if output:
                try:
                    await websocket.send_json(
                        {"type": "output", "line": output, "status": manager.status}
                    )
                    heartbeat_count = 0
                except (OSError, RuntimeError, ValueError) as e:
                    logger.error("❌ WebSocket Send Error: %s", e)
                    break

            if manager.is_running():

                current_time = time.time()
                if current_time - manager.last_hardware_send_time >= 2.0:
                    hw = manager.hardware_history
                    has_hw_data = any([
                        hw["temperatures"],
                        hw["power"],
                        hw["vram"],
                        hw["gtt"],
                        hw["cpu"],
                        hw["ram"],
                    ])
                    if has_hw_data:
                        try:
                            max_history = 60
                            hardware_data = {
                                "temperatures": hw["temperatures"][-max_history:],
                                "power": hw["power"][-max_history:],
                                "vram": hw["vram"][-max_history:],
                                "gtt": hw["gtt"][-max_history:],
                                "cpu": hw["cpu"][-max_history:],
                                "ram": hw["ram"][-max_history:],
                            }

                            await websocket.send_json(
                                {"type": "hardware", "data": hardware_data}
                            )
                            manager.update_last_hardware_send_time(current_time)
                        except (OSError, RuntimeError, ValueError) as e:
                            logger.error(
                                "❌ WebSocket Hardware Send Error: %s",
                                e,
                            )
            else:
                heartbeat_count += 1
                if heartbeat_count % 2 == 0:
                    try:
                        await websocket.send_json(
                            {
                                "type": "status",
                                "status": manager.status,
                                "running": manager.is_running(),
                            }
                        )
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.error("❌ WebSocket Heartbeat Error: %s", e)
                        break

                if manager.status == "completed":
                    try:
                        await websocket.send_json(
                            {
                                "type": "completed",
                                "message": "✅ Benchmark completed",
                            }
                        )
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.warning(
                            "⚠️ Could not send Completion message: %s",
                            e,
                        )
                    finally:
                        manager.set_idle_status()

                if manager.status == "failed":
                    try:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "message": "❌ Benchmark failed",
                            }
                        )
                    except (OSError, RuntimeError, ValueError) as e:
                        logger.warning(
                            "⚠️ Could not send Failure message: %s",
                            e,
                        )
                    finally:
                        manager.set_idle_status()

                await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("ℹ️ WebSocket normal disconnect")
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("❌ WebSocket Error: %s", e)
    finally:
        manager.connected_clients.discard(websocket)
        logger.info(
            "❌ WebSocket Client disconnected (Total: %d)",
            len(manager.connected_clients),
        )


# ============================================================================
# Latest Results Export
# ============================================================================


@app.get("/api/latest-results")
async def get_latest_results() -> dict:
    """Finds the latest benchmark results"""
    try:
        results_dir = RESULTS_DIR

        json_files = list(results_dir.glob("benchmark_results_*.json"))

        if not json_files:
            return {"latest": None}

        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)

        return {"latest": latest_file.name}
    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Error finding latest results: %s", e)
        return {"latest": None}


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check() -> dict:
    """Health Check Endpoint"""
    return {
        "status": "ok",
        "benchmark_running": manager.is_running(),
        "connected_clients": len(manager.connected_clients),
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(
        description="FastAPI Web Dashboard for LM Studio Benchmark"
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=None,
        help=(
            "Port for web dashboard " "(default: automatically search for a free port)"
        ),
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable DEBUG logging for detailed output",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not auto-open browser on startup",
    )
    normalized_args = _expand_short_flag_clusters(
        sys.argv[1:],
        combinable={"d", "h"},
    )
    args = parser.parse_args(args=normalized_args)

    DEBUG_MODE = args.debug

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("🐛 DEBUG mode enabled")

    webapp_log_file = setup_webapp_logger()

    logger.info("🌐 Starting FastAPI web dashboard...")
    logger.info("📝 WebApp log: %s", format_path_for_logs(webapp_log_file))
    logger.info("📁 Project root: %s", format_path_for_logs(PROJECT_ROOT))
    logger.info("📄 Benchmark script: %s", format_path_for_logs(BENCHMARK_SCRIPT))

    if not BENCHMARK_SCRIPT.exists():
        logger.error(
            "❌ Benchmark script not found: %s",
            format_path_for_logs(BENCHMARK_SCRIPT),
        )
        sys.exit(1)

    if args.port:
        port = args.port
        logger.info("🔧 Using specified port: %s", port)
    else:
        port = find_free_port()
        logger.info("🎲 Using automatically found free port: %s", port)

    DASHBOARD_URL = f"http://localhost:{port}"
    logger.info("🚀 Dashboard available at %s", DASHBOARD_URL)
    logger.info("📊 API Docs: %s/docs", DASHBOARD_URL)

    def open_browser():
        """Open the dashboard URL in the default web browser."""
        time.sleep(1.5)
        try:
            logger.info("🌐 Opening browser: %s", DASHBOARD_URL)
            webbrowser.open(DASHBOARD_URL)
        except (OSError, webbrowser.Error) as e:
            logger.warning("⚠️ Could not open browser: %s", e)

    if args.no_browser:
        logger.info("🌐 Browser auto-open disabled (--no-browser)")
    else:
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()

    UVICORN_LOG_LEVEL = "debug" if args.debug else "info"
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=UVICORN_LOG_LEVEL)
