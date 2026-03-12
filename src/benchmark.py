#!/usr/bin/env python3
"""
LM Studio Model Benchmark Tool

Automatically tests all locally installed LM Studio models and their
quantizations. Measures token/s speed with a standardized prompt.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import glob
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sqlite3
from statistics import mean, median, quantiles
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import psutil
from tqdm import tqdm

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    REPORTLAB_AVAILABLE = False

from config_loader import BASE_DEFAULT_CONFIG, DEFAULT_CONFIG
from preset_manager import PresetManager
from rest_client import LMStudioRESTClient
from user_paths import USER_LOGS_DIR, USER_RESULTS_DIR, format_path_for_logs

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    go = None
    PLOTLY_AVAILABLE = False


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOGS_DIR = USER_LOGS_DIR
LOGS_DIR.mkdir(exist_ok=True)
log_filename = None


class NoJSONFilter(logging.Filter):
    def filter(self, record):
        if record.getMessage().startswith("{"):
            return False
        return True


class AutoFlushStream:
    """Wrapper that automatically flushes on every write()"""

    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = AutoFlushStream(sys.stdout)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.root.handlers = [stream_handler]
logger = logging.getLogger(__name__)

for handler in logging.root.handlers:
    handler.addFilter(NoJSONFilter())

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmstudio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)

BASE_STANDARD_PROMPT = "Explain machine learning in 3 sentences"
BASE_CONTEXT_LENGTH = 2048
BASE_NUM_MEASUREMENT_RUNS = 3
GPU_OFFLOAD_LEVELS = [1.0, 0.7, 0.5, 0.3]
NUM_WARMUP_RUNS = 1
STANDARD_PROMPT = DEFAULT_CONFIG.get("prompt", BASE_STANDARD_PROMPT)
CONTEXT_LENGTH = int(DEFAULT_CONFIG.get("context_length", BASE_CONTEXT_LENGTH))
NUM_MEASUREMENT_RUNS = int(DEFAULT_CONFIG.get("num_runs", BASE_NUM_MEASUREMENT_RUNS))
VRAM_SAFETY_HEADROOM_GB = 1.0
CONTEXT_VRAM_FACTOR = 0.000002
RESULTS_DIR = USER_RESULTS_DIR
DATABASE_FILE = RESULTS_DIR / "benchmark_cache.db"
METADATA_DATABASE_FILE = RESULTS_DIR / "model_metadata.db"
BASE_OPTIMIZED_INFERENCE_PARAMS = {
    "temperature": 0.1,
    "top_k_sampling": 40,
    "top_p_sampling": 0.9,
    "min_p_sampling": 0.05,
    "repeat_penalty": 1.2,
    "max_tokens": 256,
}

BASE_LOAD_PARAMS = dict(BASE_DEFAULT_CONFIG.get("load", {}))

OPTIMIZED_INFERENCE_PARAMS = {
    **BASE_OPTIMIZED_INFERENCE_PARAMS,
    **(DEFAULT_CONFIG.get("inference") or {}),
}

DEFAULT_LOAD_PARAMS = {**BASE_LOAD_PARAMS, **(DEFAULT_CONFIG.get("load") or {})}


def get_app_version() -> str:
    """Read app version from VERSION file"""
    try:
        version_file = PROJECT_ROOT / "VERSION"
        if version_file.exists():
            return version_file.read_text().strip()
    except Exception:
        pass
    return "unknown"


APP_VERSION = get_app_version()


@dataclass
class BenchmarkResult:
    """Results of a single benchmark run"""

    model_name: str
    quantization: str
    gpu_type: str
    gpu_offload: float
    vram_mb: str
    avg_tokens_per_sec: float
    avg_ttft: float
    avg_gen_time: float
    prompt_tokens: int
    completion_tokens: int
    timestamp: str
    params_size: str
    architecture: str
    max_context_length: int
    model_size_gb: float
    has_vision: bool
    has_tools: bool
    tokens_per_sec_per_gb: float
    tokens_per_sec_per_billion_params: float
    temp_celsius_min: Optional[float] = None
    temp_celsius_max: Optional[float] = None
    temp_celsius_avg: Optional[float] = None
    power_watts_min: Optional[float] = None
    power_watts_max: Optional[float] = None
    power_watts_avg: Optional[float] = None
    gtt_enabled: Optional[bool] = None
    gtt_total_gb: Optional[float] = None
    gtt_used_gb: Optional[float] = None
    temperature: Optional[float] = None
    top_k_sampling: Optional[int] = None
    top_p_sampling: Optional[float] = None
    min_p_sampling: Optional[float] = None
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
    num_runs: Optional[int] = None
    runs_averaged_from: Optional[int] = None
    warmup_runs: Optional[int] = None
    run_index: Optional[int] = None
    lmstudio_version: Optional[str] = None
    app_version: Optional[str] = None
    nvidia_driver_version: Optional[str] = None
    rocm_driver_version: Optional[str] = None
    intel_driver_version: Optional[str] = None
    context_length: Optional[int] = None
    model_key: Optional[str] = None
    prompt_hash: Optional[str] = None
    params_hash: Optional[str] = None
    os_name: Optional[str] = None
    os_version: Optional[str] = None
    cpu_model: Optional[str] = None
    python_version: Optional[str] = None
    benchmark_duration_seconds: Optional[float] = None
    error_count: Optional[int] = None
    inference_params_hash: Optional[str] = None
    speed_delta_pct: Optional[float] = None
    prev_timestamp: Optional[str] = None


class HardwareMonitor:
    """Real-time monitoring of GPU temperature and power draw"""

    def __init__(
        self, gpu_type: Optional[str], gpu_tool: Optional[str], enabled: bool = False
    ):
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

        if self.gpu_type == "AMD" and self.gpu_tool == "sysfs":
            self._init_amd_sysfs_paths()
        self._amd_sysfs_path: Optional[str] = None
        self._amd_hwmon_path: Optional[str] = None

        if self.gpu_type == "AMD" and self.gpu_tool == "sysfs":
            self._init_amd_sysfs_paths()

    def _init_amd_sysfs_paths(self):
        """Initializes AMD sysfs paths at startup"""
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
        except Exception:
            pass

    def start(self):
        """Start background monitoring"""
        if not self.enabled:
            logger.info("⚠️ Hardware monitoring disabled (--enable-profiling not set)")
            return

        if not self.gpu_tool:
            logger.warning("⚠️ No GPU tools found - hardware monitoring not available")
            return

        logger.info(
            f"🔥 Starting Hardware-Monitoring (GPU: {self.gpu_type}, Tool: {self.gpu_tool})"
        )
        self.monitoring = True
        self.temps.clear()
        self.powers.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Optional[float]]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)

        with self.lock:
            temps = self.temps.copy()
            powers = self.powers.copy()
            vrams = self.vrams.copy()
            gtts = self.gtts.copy()
            cpus = self.cpus.copy()
            rams = self.rams.copy()

        stats = {
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
        }
        return stats

    def _monitor_loop(self):
        """Background thread for continuous measurements"""
        logger.info("🔍 Hardware-Monitor thread started")
        while self.monitoring:
            try:
                temp = self._get_temperature()
                power = self._get_power_draw()
                vram = self._get_vram_usage()
                gtt = self._get_gtt_usage()
                cpu = self._get_cpu_usage()
                ram = self._get_ram_usage()

                with self.lock:
                    if temp is not None:
                        self.temps.append(temp)
                        logger.info("🌡️ GPU Temp: %s°C", temp)
                    else:
                        logger.debug("⚠️ No temperature read (gpu_type=%s, tool=%s)",
                                self.gpu_type,
                                self.gpu_tool)

                    if power is not None:
                        self.powers.append(power)
                        logger.info("⚡ GPU Power: %sW", power)
                    else:
                        logger.debug("⚠️ No power read (gpu_type=%s, tool=%s)",
                                self.gpu_type,
                                self.gpu_tool)

                    if vram is not None:
                        self.vrams.append(vram)
                        logger.info("💾 GPU VRAM: %sGB", vram)
                    else:
                        logger.debug("⚠️ No VRAM read (gpu_type=%s, tool=%s)",
                                self.gpu_type,
                                self.gpu_tool)

                    if gtt is not None:
                        self.gtts.append(gtt)
                        logger.info("🧠 GPU GTT: %sGB", gtt)
                    else:
                        logger.debug("⚠️ No GTT read (gpu_type=%s, tool=%s)",
                                self.gpu_type,
                                self.gpu_tool)

                    if cpu is not None:
                        self.cpus.append(cpu)
                        logger.info("🖥️ CPU: %s%%", cpu)

                    if ram is not None:
                        self.rams.append(ram)
                        logger.info("💾 RAM: %sGB", ram)

                time.sleep(1)
            except Exception as e:
                logger.debug("Monitoring error: %s", e)
                time.sleep(2)
        logger.info("🛑 Hardware-Monitor thread stopped")

    def _get_temperature(self) -> Optional[float]:
        """Liest aktuelle GPU-Temperatur"""
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [
                        self.gpu_tool,
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip().split("\n")[0]
                    return float(temp_str)

            elif self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs" and self._amd_hwmon_path:
                    temp_file = Path(self._amd_hwmon_path) / "temp1_input"
                    if temp_file.exists():
                        temp_millic = int(temp_file.read_text().strip())
                        return float(temp_millic) / 1000.0
                else:
                    result = subprocess.run(
                        [self.gpu_tool, "--showtemp"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "GPU[" in line and (
                                "(C):" in line or "c" in line.lower()
                            ):
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
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None

    def _get_power_draw(self) -> Optional[float]:
        """Liest GPU Power-Draw in Watts"""
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [
                        self.gpu_tool,
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    power_str = result.stdout.strip().split("\n")[0]
                    return float(power_str)

            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, "--showpower"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    import re

                    for line in result.stdout.split("\n"):
                        if "GPU[" in line and ("(W):" in line or "W" in line):
                            try:
                                match = re.search(r"[\d.]+\s*$", line.strip())
                                if match:
                                    return float(match.group())
                                power_str = line.split(":")[-1].strip().replace("W", "")
                                return float(power_str)
                            except (ValueError, IndexError):
                                pass
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None

    def _get_vram_usage(self) -> Optional[float]:
        """Reads GPU VRAM usage in GB"""
        try:
            if not self.gpu_tool:
                return None

            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [
                        self.gpu_tool,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    vram_mb = float(result.stdout.strip().split("\n")[0])
                    return vram_mb / 1024.0

            elif self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs" and self._amd_sysfs_path:
                    vram_file = Path(self._amd_sysfs_path) / "mem_info_vram_used"
                    if vram_file.exists():
                        vram_bytes = int(vram_file.read_text().strip())
                        return float(vram_bytes) / (1024**3)
                else:
                    result = subprocess.run(
                        [self.gpu_tool, "--showmeminfo", "vram"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "GPU[" in line and "Used Memory" in line:
                                match = re.search(r"(\d+)\s*$", line.strip())
                                if match:
                                    vram_bytes = float(match.group(1))
                                    return vram_bytes / (1024**3)
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None

    def _get_gtt_usage(self) -> Optional[float]:
        """Reads GTT usage in GB (System RAM for AMD GPUs)"""
        try:
            if self.gpu_type != "AMD" or not self.gpu_tool:
                return None

            if self.gpu_tool == "sysfs" and self._amd_sysfs_path:
                gtt_file = Path(self._amd_sysfs_path) / "mem_info_gtt_used"
                if gtt_file.exists():
                    gtt_bytes = int(gtt_file.read_text().strip())
                    return float(gtt_bytes) / (1024**3)
            else:
                result = subprocess.run(
                    [self.gpu_tool, "--showmeminfo", "gtt"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "GPU[" in line and "Used Memory" in line:
                            match = re.search(r"(\d+)\s*$", line.strip())
                            if match:
                                gtt_bytes = float(match.group(1))
                                return gtt_bytes / (1024**3)
        except (subprocess.TimeoutExpired, Exception):
            pass

        return None

    def _get_cpu_usage(self) -> Optional[float]:
        """Reads system CPU utilization in %"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return None

    def _get_ram_usage(self) -> Optional[float]:
        """Reads system RAM usage in GB with smoothing (moving average over 7 measurements)"""
        try:
            mem = psutil.virtual_memory()
            current_ram = mem.used / (1024**3)

            self.ram_readings.append(current_ram)
            if len(self.ram_readings) > 7:
                self.ram_readings.pop(0)

            return sum(self.ram_readings) / len(self.ram_readings)
        except Exception:
            return None


class BenchmarkCache:
    """SQLite cache for benchmark results"""

    def __init__(self, db_path: Path = DATABASE_FILE):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Creates database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                quantization TEXT NOT NULL,
                inference_params_hash TEXT NOT NULL,
                gpu_type TEXT NOT NULL,
                gpu_offload REAL NOT NULL,
                vram_mb TEXT NOT NULL,
                avg_tokens_per_sec REAL NOT NULL,
                avg_ttft REAL NOT NULL,
                avg_gen_time REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                params_size TEXT NOT NULL,
                architecture TEXT NOT NULL,
                max_context_length INTEGER NOT NULL,
                model_size_gb REAL NOT NULL,
                has_vision INTEGER NOT NULL,
                has_tools INTEGER NOT NULL,
                tokens_per_sec_per_gb REAL,
                tokens_per_sec_per_billion_params REAL,
                speed_delta_pct REAL,
                prev_timestamp TEXT,
                prompt TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                temperature REAL,
                top_k_sampling INTEGER,
                top_p_sampling REAL,
                min_p_sampling REAL,
                repeat_penalty REAL,
                max_tokens INTEGER,
                num_runs INTEGER,
                runs_averaged_from INTEGER,
                warmup_runs INTEGER,
                run_index INTEGER,
                lmstudio_version TEXT,
                app_version TEXT,
                nvidia_driver_version TEXT,
                rocm_driver_version TEXT,
                intel_driver_version TEXT,
                prompt_hash TEXT,
                params_hash TEXT,
                os_name TEXT,
                os_version TEXT,
                cpu_model TEXT,
                python_version TEXT,
                benchmark_duration_seconds REAL,
                error_count INTEGER,
                n_gpu_layers INTEGER,
                n_batch INTEGER,
                n_threads INTEGER,
                flash_attention INTEGER,
                rope_freq_base REAL,
                rope_freq_scale REAL,
                use_mmap INTEGER,
                use_mlock INTEGER,
                kv_cache_quant TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_key_hash
            ON benchmark_results(model_key, inference_params_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_key_params_hash
            ON benchmark_results(model_key, params_hash)
        """)

        try:
            cursor.execute("SELECT run_index FROM benchmark_results LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("📦 Migration: Adding run_index column...")
            cursor.execute("ALTER TABLE benchmark_results ADD COLUMN run_index INTEGER")
            conn.commit()
            logger.info("✅ Migration successful")

        new_columns = [
            ("n_gpu_layers", "INTEGER"),
            ("n_batch", "INTEGER"),
            ("n_threads", "INTEGER"),
            ("flash_attention", "INTEGER"),
            ("rope_freq_base", "REAL"),
            ("rope_freq_scale", "REAL"),
            ("use_mmap", "INTEGER"),
            ("use_mlock", "INTEGER"),
            ("kv_cache_quant", "TEXT"),
        ]
        # Whitelist validation: ensure column names are alphanumeric
        for col_name, col_type in new_columns:
            if not col_name.replace("_", "").isalnum():
                raise ValueError(
                    f"Invalid column name: {col_name}"
                )
            if not col_type.replace(" ", "").isalpha():
                raise ValueError(
                    f"Invalid column type: {col_type}"
                )
            try:
                # Safe: col_name is validated against alphanumeric
                query = (
                    f"SELECT {col_name} FROM benchmark_results LIMIT 1"
                )
                cursor.execute(query)  # nosec B608
            except sqlite3.OperationalError:
                logger.info("📦 Migration: Adding %s column...", col_name)
                # Safe: both col_name and col_type are validated
                alter_query = (
                    f"ALTER TABLE benchmark_results "
                    f"ADD COLUMN {col_name} {col_type}"
                )
                cursor.execute(alter_query)  # nosec B608
                conn.commit()

        conn.commit()
        conn.close()

    @staticmethod
    def compute_params_hash(
        prompt: str,
        context_length: int,
        inference_params: dict,
        load_params: Optional[dict] = None,
    ) -> str:
        """Calculates hash from all relevant parameters"""
        params_dict = {
            "prompt": prompt,
            "context_length": context_length,
            **inference_params,
        }
        if load_params:
            params_dict["load_config"] = load_params
        hash_input = json.dumps(params_dict, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    def get_cached_result(
        self, model_key: str, params_hash: str
    ) -> Optional[BenchmarkResult]:
        """Retrieves cached result from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(benchmark_results)")
        columns = {row[1]: row[0] for row in cursor.fetchall()}

        if "params_hash" in columns:
            cursor.execute(
                """
                SELECT * FROM benchmark_results
                WHERE model_key = ? AND params_hash = ?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (model_key, params_hash),
            )
            row = cursor.fetchone()
            if row is None:
                cursor.execute(
                    """
                    SELECT * FROM benchmark_results
                    WHERE model_key = ? AND inference_params_hash = ?
                    ORDER BY timestamp DESC LIMIT 1
                """,
                    (model_key, params_hash),
                )
                row = cursor.fetchone()
        else:
            cursor.execute(
                """
                SELECT * FROM benchmark_results
                WHERE model_key = ? AND inference_params_hash = ?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (model_key, params_hash),
            )
            row = cursor.fetchone()
        conn.close()

        if row:
            result_dict = {
                "model_name": row[2],
                "quantization": row[3],
                "gpu_type": row[5],
                "gpu_offload": row[6],
                "vram_mb": row[7],
                "avg_tokens_per_sec": row[8],
                "avg_ttft": row[9],
                "avg_gen_time": row[10],
                "prompt_tokens": row[11],
                "completion_tokens": row[12],
                "timestamp": row[13],
                "params_size": row[14],
                "architecture": row[15],
                "max_context_length": row[16],
                "model_size_gb": row[17],
                "has_vision": bool(row[18]),
                "has_tools": bool(row[19]),
                "tokens_per_sec_per_gb": row[20],
                "tokens_per_sec_per_billion_params": row[21],
                "speed_delta_pct": row[22],
                "prev_timestamp": row[23],
            }

            if "n_gpu_layers" in columns:
                result_dict["n_gpu_layers"] = row[columns["n_gpu_layers"]]
            if "n_batch" in columns:
                result_dict["n_batch"] = row[columns["n_batch"]]
            if "n_threads" in columns:
                result_dict["n_threads"] = row[columns["n_threads"]]
            if "flash_attention" in columns:
                val = row[columns["flash_attention"]]
                result_dict["flash_attention"] = bool(val) if val is not None else None
            if "rope_freq_base" in columns:
                result_dict["rope_freq_base"] = row[columns["rope_freq_base"]]
            if "rope_freq_scale" in columns:
                result_dict["rope_freq_scale"] = row[columns["rope_freq_scale"]]
            if "use_mmap" in columns:
                val = row[columns["use_mmap"]]
                result_dict["use_mmap"] = bool(val) if val is not None else None
            if "use_mlock" in columns:
                val = row[columns["use_mlock"]]
                result_dict["use_mlock"] = bool(val) if val is not None else None
            if "kv_cache_quant" in columns:
                result_dict["kv_cache_quant"] = row[columns["kv_cache_quant"]]

            return BenchmarkResult(**result_dict)
        return None

    def get_latest_result_for_model(self, model_key: str) -> Optional[BenchmarkResult]:
        """Returns latest cached result for model regardless of params hash."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(benchmark_results)")
        columns = {row[1]: row[0] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT * FROM benchmark_results
            WHERE model_key = ?
            ORDER BY timestamp DESC LIMIT 1
        """,
            (model_key,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            result_dict = {
                "model_name": row[2],
                "quantization": row[3],
                "gpu_type": row[5],
                "gpu_offload": row[6],
                "vram_mb": row[7],
                "avg_tokens_per_sec": row[8],
                "avg_ttft": row[9],
                "avg_gen_time": row[10],
                "prompt_tokens": row[11],
                "completion_tokens": row[12],
                "timestamp": row[13],
                "params_size": row[14],
                "architecture": row[15],
                "max_context_length": row[16],
                "model_size_gb": row[17],
                "has_vision": bool(row[18]),
                "has_tools": bool(row[19]),
                "tokens_per_sec_per_gb": row[20],
                "tokens_per_sec_per_billion_params": row[21],
                "speed_delta_pct": row[22],
                "prev_timestamp": row[23],
            }

            if "n_gpu_layers" in columns:
                result_dict["n_gpu_layers"] = row[columns["n_gpu_layers"]]
            if "n_batch" in columns:
                result_dict["n_batch"] = row[columns["n_batch"]]
            if "n_threads" in columns:
                result_dict["n_threads"] = row[columns["n_threads"]]
            if "flash_attention" in columns:
                val = row[columns["flash_attention"]]
                result_dict["flash_attention"] = bool(val) if val is not None else None
            if "rope_freq_base" in columns:
                result_dict["rope_freq_base"] = row[columns["rope_freq_base"]]
            if "rope_freq_scale" in columns:
                result_dict["rope_freq_scale"] = row[columns["rope_freq_scale"]]
            if "use_mmap" in columns:
                val = row[columns["use_mmap"]]
                result_dict["use_mmap"] = bool(val) if val is not None else None
            if "use_mlock" in columns:
                val = row[columns["use_mlock"]]
                result_dict["use_mlock"] = bool(val) if val is not None else None
            if "kv_cache_quant" in columns:
                result_dict["kv_cache_quant"] = row[columns["kv_cache_quant"]]

            return BenchmarkResult(**result_dict)
        return None

    def save_result(
        self,
        result: BenchmarkResult,
        model_key: str,
        params_hash: str,
        prompt: str,
        context_length: int,
    ):
        """Saves benchmark result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            values = (
                model_key,
                result.model_name,
                result.quantization,
                result.inference_params_hash,
                result.gpu_type,
                result.gpu_offload,
                result.vram_mb,
                result.avg_tokens_per_sec,
                result.avg_ttft,
                result.avg_gen_time,
                result.prompt_tokens,
                result.completion_tokens,
                result.timestamp,
                result.params_size,
                result.architecture,
                result.max_context_length,
                result.model_size_gb,
                int(result.has_vision),
                int(result.has_tools),
                result.tokens_per_sec_per_gb,
                result.tokens_per_sec_per_billion_params,
                result.speed_delta_pct,
                result.prev_timestamp,
                prompt,
                context_length,
                result.temperature,
                result.top_k_sampling,
                result.top_p_sampling,
                result.min_p_sampling,
                result.repeat_penalty,
                result.max_tokens,
                result.num_runs,
                result.runs_averaged_from,
                result.warmup_runs,
                result.run_index,
                result.lmstudio_version,
                result.app_version,
                result.nvidia_driver_version,
                result.rocm_driver_version,
                result.intel_driver_version,
                result.prompt_hash,
                params_hash,
                result.os_name,
                result.os_version,
                result.cpu_model,
                result.python_version,
                result.benchmark_duration_seconds,
                result.error_count,
                result.n_gpu_layers,
                result.n_batch,
                result.n_threads,
                (
                    int(result.flash_attention)
                    if result.flash_attention is not None
                    else None
                ),
                result.rope_freq_base,
                result.rope_freq_scale,
                int(result.use_mmap) if result.use_mmap is not None else None,
                int(result.use_mlock) if result.use_mlock is not None else None,
                result.kv_cache_quant,
            )

            logger.debug("📊 INSERT: %s values for 56 columns", len(values))

            placeholders = ", ".join("?" for _ in values)

            cursor.execute(
                f"""
                INSERT INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, tokens_per_sec_per_gb,
                    tokens_per_sec_per_billion_params, speed_delta_pct, prev_timestamp,
                    prompt, context_length, temperature, top_k_sampling, top_p_sampling,
                    min_p_sampling, repeat_penalty, max_tokens, num_runs, runs_averaged_from,
                    warmup_runs, run_index, lmstudio_version, app_version, nvidia_driver_version, rocm_driver_version,
                    intel_driver_version, prompt_hash, params_hash, os_name, os_version,
                    cpu_model, python_version, benchmark_duration_seconds, error_count,
                    n_gpu_layers, n_batch, n_threads, flash_attention, rope_freq_base,
                    rope_freq_scale, use_mmap, use_mlock, kv_cache_quant
                ) VALUES ({placeholders})
            """,
                values,
            )

            conn.commit()
        except Exception as e:
            logger.error("❌ Error saving to cache: %s", e)
        finally:
            conn.close()

    def get_all_results(self) -> List[BenchmarkResult]:
        """Loads all benchmark results from the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(benchmark_results)")
            columns = {row[1] for row in cursor.fetchall()}

            base_cols = (
                "model_name, quantization, gpu_type, gpu_offload, vram_mb, "
                "avg_tokens_per_sec, avg_ttft, avg_gen_time, prompt_tokens, "
                "completion_tokens, "
                "timestamp, params_size, architecture, max_context_length, "
                "model_size_gb, "
                "has_vision, has_tools, tokens_per_sec_per_gb, "
                "tokens_per_sec_per_billion_params"
            )

            optional_cols = []
            if "temp_celsius_min" in columns:
                optional_cols.extend(
                    ["temp_celsius_min", "temp_celsius_max", "temp_celsius_avg"]
                )
            if "power_watts_min" in columns:
                optional_cols.extend(
                    ["power_watts_min", "power_watts_max", "power_watts_avg"]
                )
            if "gtt_enabled" in columns:
                optional_cols.extend(
                    ["gtt_enabled", "gtt_total_gb", "gtt_used_gb"]
                )
            if "speed_delta_pct" in columns:
                optional_cols.extend(["speed_delta_pct", "prev_timestamp"])
            if "temperature" in columns:
                optional_cols.extend(
                    [
                        "temperature",
                        "top_k_sampling",
                        "top_p_sampling",
                        "min_p_sampling",
                        "repeat_penalty",
                        "max_tokens",
                    ]
                )
            if "num_runs" in columns:
                optional_cols.extend(
                    ["num_runs", "runs_averaged_from", "warmup_runs"]
                )
            if "lmstudio_version" in columns:
                optional_cols.extend(
                    [
                        "lmstudio_version",
                        "nvidia_driver_version",
                        "rocm_driver_version",
                        "intel_driver_version",
                    ]
                )
            if "prompt_hash" in columns:
                optional_cols.extend(
                    [
                        "prompt_hash",
                        "params_hash",
                        "os_name",
                        "os_version",
                        "cpu_model",
                        "python_version",
                        "benchmark_duration_seconds",
                        "error_count",
                    ]
                )
            if "inference_params_hash" in columns:
                optional_cols.append("inference_params_hash")

            # Validate all column names are alphanumeric (SQL injection)
            all_cols_str = base_cols + (
                ", " + ", ".join(optional_cols) if optional_cols else ""
            )
            for col in all_cols_str.replace(",", "").split():
                if not col.replace("_", "").isalnum():
                    raise ValueError(
                        f"Invalid column name detected: {col}"
                    )

            # Safe: all column names have been validated
            select_query = f"""
                SELECT {all_cols_str}
                FROM benchmark_results
                ORDER BY timestamp DESC
            """
            cursor.execute(select_query)  # nosec B608

            results = []
            for row in cursor.fetchall():
                idx = 0
                result_dict = {
                    "model_name": row[idx],
                    "quantization": row[idx + 1],
                    "gpu_type": row[idx + 2],
                    "gpu_offload": row[idx + 3],
                    "vram_mb": row[idx + 4],
                    "avg_tokens_per_sec": row[idx + 5],
                    "avg_ttft": row[idx + 6],
                    "avg_gen_time": row[idx + 7],
                    "prompt_tokens": row[idx + 8],
                    "completion_tokens": row[idx + 9],
                    "timestamp": row[idx + 10],
                    "params_size": row[idx + 11],
                    "architecture": row[idx + 12],
                    "max_context_length": row[idx + 13],
                    "model_size_gb": row[idx + 14],
                    "has_vision": bool(row[idx + 15]),
                    "has_tools": bool(row[idx + 16]),
                    "tokens_per_sec_per_gb": row[idx + 17],
                    "tokens_per_sec_per_billion_params": row[idx + 18],
                }
                idx = 19

                if "temp_celsius_min" in columns:
                    result_dict["temp_celsius_min"] = row[idx]
                    result_dict["temp_celsius_max"] = row[idx + 1]
                    result_dict["temp_celsius_avg"] = row[idx + 2]
                    idx += 3

                if "power_watts_min" in columns:
                    result_dict["power_watts_min"] = row[idx]
                    result_dict["power_watts_max"] = row[idx + 1]
                    result_dict["power_watts_avg"] = row[idx + 2]
                    idx += 3

                if "gtt_enabled" in columns:
                    result_dict["gtt_enabled"] = (
                        bool(row[idx]) if row[idx] is not None else None
                    )
                    result_dict["gtt_total_gb"] = row[idx + 1]
                    result_dict["gtt_used_gb"] = row[idx + 2]
                    idx += 3

                if "speed_delta_pct" in columns:
                    result_dict["speed_delta_pct"] = row[idx]
                    result_dict["prev_timestamp"] = row[idx + 1]
                    idx += 2

                if "temperature" in columns:
                    result_dict["temperature"] = row[idx]
                    result_dict["top_k_sampling"] = row[idx + 1]
                    result_dict["top_p_sampling"] = row[idx + 2]
                    result_dict["min_p_sampling"] = row[idx + 3]
                    result_dict["repeat_penalty"] = row[idx + 4]
                    result_dict["max_tokens"] = row[idx + 5]
                    idx += 6

                if "num_runs" in columns:
                    result_dict["num_runs"] = row[idx]
                    result_dict["runs_averaged_from"] = row[idx + 1]
                    result_dict["warmup_runs"] = row[idx + 2]
                    idx += 3

                if "lmstudio_version" in columns:
                    result_dict["lmstudio_version"] = row[idx]
                    result_dict["nvidia_driver_version"] = row[idx + 1]
                    result_dict["rocm_driver_version"] = row[idx + 2]
                    result_dict["intel_driver_version"] = row[idx + 3]
                    idx += 4

                if "prompt_hash" in columns:
                    result_dict["prompt_hash"] = row[idx]
                    result_dict["params_hash"] = row[idx + 1]
                    result_dict["os_name"] = row[idx + 2]
                    result_dict["os_version"] = row[idx + 3]
                    result_dict["cpu_model"] = row[idx + 4]
                    result_dict["python_version"] = row[idx + 5]
                    result_dict["benchmark_duration_seconds"] = row[idx + 6]
                    result_dict["error_count"] = row[idx + 7]
                    idx += 8

                if "inference_params_hash" in columns:
                    result_dict["inference_params_hash"] = row[idx]

                result = BenchmarkResult(**result_dict)
                results.append(result)

            return results
        except Exception as e:
            logger.error("❌ Error loading all results: %s", e)
            return []
        finally:
            conn.close()

    def list_cached_models(self) -> List[Dict]:
        """Returns all cached models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT model_key, model_name, quantization, avg_tokens_per_sec,
                   timestamp, inference_params_hash, params_size
            FROM benchmark_results
            ORDER BY timestamp DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "model_key": row[0],
                    "model_name": row[1],
                    "quantization": row[2],
                    "avg_tokens_per_sec": row[3],
                    "timestamp": row[4],
                    "params_hash": row[5],
                    "params_size": row[6],
                }
            )

        conn.close()
        return results

    def export_to_json(self, output_file: Path):
        """Exports cache as JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM benchmark_results ORDER BY timestamp DESC")

        columns = [desc[0] for desc in cursor.description]
        results = []

        for row in cursor.fetchall():
            result_dict = dict(zip(columns, row))
            result_dict["has_vision"] = bool(result_dict["has_vision"])
            result_dict["has_tools"] = bool(result_dict["has_tools"])
            results.append(result_dict)

        conn.close()

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info("💾 Cache exported: %s", output_file)


class GPUMonitor:
    """Detects GPU type and measures VRAM usage"""

    def __init__(self):
        self.gpu_type: Optional[str] = None
        self.gpu_model: Optional[str] = None
        self.gpu_tool: Optional[str] = None
        self._detect_gpu()

    def _find_tool(self, tool_name: str, search_paths: List[str]) -> Optional[str]:
        """Searches for tool in PATH and specific paths"""
        if shutil.which(tool_name):
            return tool_name

        for path in search_paths:
            found = shutil.which(tool_name, path=path)
            if found:
                return found

        return None

    def _find_amd_sysfs_path(self) -> Optional[str]:
        """Finds AMD GPU sysfs path for direct monitoring"""
        try:
            result = subprocess.run(
                ["lspci", "-d", "1002:", "-n"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0 or not result.stdout:
                return None

            import glob

            for cardpath in glob.glob("/sys/class/drm/card*/device"):
                vendor_file = Path(cardpath) / "vendor"
                if vendor_file.exists():
                    vendor = vendor_file.read_text().strip()
                    if vendor == "0x1002":
                        vram_file = Path(cardpath) / "mem_info_vram_total"
                        if vram_file.exists():
                            return str(Path(cardpath))
            return None
        except Exception:
            return None

    def _detect_gpu(self):
        """Detects GPU type and finds corresponding monitoring tool"""
        nvidia_paths = ["/usr/bin", "/usr/local/bin", "/usr/local/cuda/bin"]
        nvidia_tool = self._find_tool("nvidia-smi", nvidia_paths)
        if nvidia_tool:
            self.gpu_type = "NVIDIA"
            self.gpu_tool = nvidia_tool
            try:
                result = subprocess.run(
                    [nvidia_tool, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    self.gpu_model = result.stdout.strip().split("\n")[0]
                else:
                    self.gpu_model = "NVIDIA GPU"
            except Exception:
                self.gpu_model = "NVIDIA GPU"
            logger.info(
                f"🟢 NVIDIA GPU detected: {self.gpu_model}, Tool: {nvidia_tool}"
            )
            return

        amd_paths = ["/usr/bin", "/usr/local/bin", "/opt/rocm/bin"]
        import glob

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
                f"🔴 AMD GPU detected (sysfs): {self.gpu_model}, " f"Path: {amd_sysfs}"
            )
            return

        intel_paths = ["/usr/bin", "/usr/local/bin", "/usr/lib/xpu"]
        intel_tool = self._find_tool("intel_gpu_top", intel_paths)
        if intel_tool:
            self.gpu_type = "Intel"
            self.gpu_tool = intel_tool
            self.gpu_model = self._detect_intel_gpu_model()
            logger.info("🔵 Intel GPU detected: %s, Tool: %s", self.gpu_model, intel_tool)
            return

        logger.warning(
            "⚠️ No GPU monitoring tools found. VRAM measurement not available."
        )
        self.gpu_type = "Unknown"
        self.gpu_model = "Unknown"

    def _detect_amd_gpu_model(self) -> str:
        """Detects AMD GPU model name with fallback chain"""
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

        try:
            import cpuinfo

            cpu = cpuinfo.get_cpu_info()
            brand = cpu.get("brand_raw", "")
            if "Radeon" in brand:
                radeon_part = brand.split("Radeon")[1].strip()
                model = radeon_part.split()[0]
                if model:
                    return f"AMD Radeon {model}"
        except Exception:
            pass

        device_id = None
        try:
            result = subprocess.run(
                ["lspci", "-d", "1002:"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if "1002:" in line:
                        parts = line.split("1002:")
                        if len(parts) > 1:
                            device_id = parts[1].split()[0].lower()
                            if device_id in amd_device_mapping:
                                return f"AMD {amd_device_mapping[device_id]}"
                            break
        except Exception:
            pass

        if self.gpu_tool:
            try:
                result = subprocess.run(
                    [self.gpu_tool, "--showproductname"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if "GPU[0]" in line:
                            parts = line.split(":")
                            if len(parts) > 1:
                                gfx_code = parts[1].strip()
                                if gfx_code in amd_device_mapping:
                                    return f"AMD {amd_device_mapping[gfx_code]}"
                                return f"AMD {gfx_code}"
            except Exception:
                pass

        if device_id:
            return f"AMD GPU (1002:{device_id})"
        return "AMD GPU"

    def _detect_intel_gpu_model(self) -> str:
        """Detects Intel GPU model name"""
        try:
            result = subprocess.run(
                ["lspci", "-d", "8086::0300"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout:
                line = result.stdout.strip().split("\n")[0]
                if "Intel" in line:
                    parts = line.split(": ")
                    if len(parts) > 1:
                        return parts[1].split("[")[0].strip()
        except Exception:
            pass
        return "Intel GPU"

    def get_vram_usage(self) -> str:
        """Measures current VRAM usage"""
        if not self.gpu_tool:
            return "N/A"

        try:
            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [
                        self.gpu_tool,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return result.stdout.strip().split("\n")[0]

            elif self.gpu_type == "AMD":
                if self.gpu_tool == "sysfs":
                    sysfs_path = self._find_amd_sysfs_path()
                    if sysfs_path:
                        vram_file = Path(sysfs_path) / "mem_info_vram_used"
                        if vram_file.exists():
                            vram_bytes = int(vram_file.read_text().strip())
                            mb_used = vram_bytes / (1024 * 1024)
                            return f"{int(mb_used)}"
                else:
                    result = subprocess.run(
                        [self.gpu_tool, "--showmeminfo", "vram"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0:
                        for line in result.stdout.split("\n"):
                            if "VRAM Total Used Memory" in line:
                                parts = line.split(":")
                                if len(parts) >= 3:
                                    bytes_used = parts[-1].strip()
                                    try:
                                        mb_used = int(bytes_used) / (1024 * 1024)
                                        return f"{int(mb_used)}"
                                    except ValueError:
                                        pass

            elif self.gpu_type == "Intel":
                return "N/A"

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning("⚠️ VRAM measurement failed: %s", e)

        return "N/A"


class LMStudioServerManager:
    """Manages LM Studio Server lifecycle"""

    @staticmethod
    def is_server_running() -> bool:
        """Checks if LM Studio Server is running"""
        try:
            result = subprocess.run(
                ["lms", "server", "status"], capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            return result.returncode == 0 and (
                "running" in output.lower() or "port" in output.lower()
            )
        except Exception as e:
            logger.error("❌ Error checking server status: %s", e)
            return False

    @staticmethod
    def start_server():
        """Startet LM Studio Server"""
        try:
            logger.info("🚀 Starting LM Studio Server...")
            subprocess.Popen(
                ["lms", "server", "start"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            max_retries = 30
            for i in range(max_retries):
                time.sleep(2)
                if LMStudioServerManager.is_server_running():
                    logger.info("✅ LM Studio Server started successfully")
                    return True

            logger.error("❌ Server start timeout after 60 seconds")
            return False

        except Exception as e:
            logger.error("❌ Error starting server: %s", e)
            return False

    @staticmethod
    def ensure_server_running():
        """Ensures that server is running"""
        if not LMStudioServerManager.is_server_running():
            logger.info("⚠️ Server not running, starting server...")
            return LMStudioServerManager.start_server()
        logger.info("✅ Server already running")
        return True


class ModelDiscovery:
    """Finds all locally installed models"""

    _metadata_cache: Dict[str, Dict] = {}

    @staticmethod
    def _get_metadata_cache() -> Dict[str, Dict]:
        """Cache for model metadata (loaded once at startup)"""
        if not ModelDiscovery._metadata_cache:
            try:
                result = subprocess.run(
                    ["lms", "ls", "--json"], capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for model_data in data:
                        if model_data.get("type") == "llm":
                            key = model_data.get("modelKey")
                            ModelDiscovery._metadata_cache[key] = {
                                "architecture": model_data.get(
                                    "architecture", "unknown"
                                ),
                                "params_size": model_data.get(
                                    "paramsString", "unknown"
                                ),
                                "max_context_length": model_data.get(
                                    "maxContextLength", 0
                                ),
                                "model_size_gb": round(
                                    model_data.get("sizeBytes", 0) / 1024**3, 2
                                ),
                                "has_vision": model_data.get("vision", False),
                                "has_tools": model_data.get("trainedForToolUse", False),
                            }
            except Exception as e:
                logger.warning("⚠️ Error loading metadata cache: %s", e)
        return ModelDiscovery._metadata_cache

    @staticmethod
    def get_model_metadata(model_key: str) -> Dict:
        """Get metadata for a specific model"""
        cache = ModelDiscovery._get_metadata_cache()
        base_model = model_key.split("@")[0] if "@" in model_key else model_key
        return cache.get(
            base_model,
            {
                "architecture": "unknown",
                "params_size": "unknown",
                "max_context_length": 0,
                "model_size_gb": 0.0,
                "has_vision": False,
                "has_tools": False,
            },
        )

    @staticmethod
    def get_scraped_metadata(model_key: str) -> Dict:
        """Reads optional, scraped metadata from model_metadata.db."""
        if not METADATA_DATABASE_FILE.exists():
            return {}
        base_key = model_key.split("@")[0] if "@" in model_key else model_key
        try:
            conn = sqlite3.connect(METADATA_DATABASE_FILE)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            row = cur.execute(
                "SELECT * FROM model_metadata WHERE model_key = ?",
                (base_key,),
            ).fetchone()
            conn.close()
            return dict(row) if row else {}
        except Exception as e:
            logger.warning("⚠️ Could not read scraped metadata: %s", e)
            return {}

    @staticmethod
    def get_installed_models() -> List[str]:
        """Lists all locally installed models and quantizations"""
        try:
            result = subprocess.run(
                ["lms", "ls", "--json"], capture_output=True, text=True, timeout=30
            )

            if result.returncode != 0:
                logger.error("❌ Error with 'lms ls': %s", result.stderr)
                return []

            models = []
            import json

            data = json.loads(result.stdout)

            for model_data in data:
                if model_data.get("type") == "llm":
                    variants = model_data.get("variants", [])
                    if variants:
                        models.extend(variants)
                    else:
                        models.append(model_data.get("modelKey"))

            logger.info("🔍 %s models found", len(models))
            if models:
                logger.info("📋 First 5 models:")
                for model in models[:5]:
                    logger.info("  • %s", model)
            return models

        except Exception as e:
            logger.error("❌ Error fetching models: %s", e)
            return []

    @staticmethod
    def filter_models(models: List[str], filter_args: Dict) -> List[str]:
        """Filters models based on CLI arguments"""
        if not filter_args:
            return models

        metadata_cache = ModelDiscovery._get_metadata_cache()
        filtered = []

        include_pattern = None
        exclude_pattern = None

        if filter_args.get("include_models"):
            try:
                include_pattern = re.compile(
                    filter_args["include_models"], re.IGNORECASE
                )
            except re.error as e:
                logger.error("❌ Invalid include-models pattern: %s", e)
                return []

        if filter_args.get("exclude_models"):
            try:
                exclude_pattern = re.compile(
                    filter_args["exclude_models"], re.IGNORECASE
                )
            except re.error as e:
                logger.error("❌ Invalid exclude-models pattern: %s", e)
                return []

        for model_key in models:
            metadata = ModelDiscovery.get_model_metadata(model_key)

            if include_pattern and not include_pattern.search(model_key):
                continue

            if exclude_pattern and exclude_pattern.search(model_key):
                continue

            if filter_args.get("only_vision") and not metadata["has_vision"]:
                continue

            if filter_args.get("only_tools") and not metadata["has_tools"]:
                continue

            if filter_args.get("quants"):
                quants_list = [
                    q.strip().lower() for q in filter_args["quants"].split(",")
                ]
                quant = model_key.split("@")[-1].lower() if "@" in model_key else ""
                if not any(q in quant for q in quants_list):
                    continue

            if filter_args.get("arch"):
                arch_list = [a.strip().lower() for a in filter_args["arch"].split(",")]
                if metadata["architecture"].lower() not in arch_list:
                    continue

            if filter_args.get("params"):
                params_list = [
                    p.strip().upper() for p in filter_args["params"].split(",")
                ]
                if metadata["params_size"].upper() not in params_list:
                    continue

            if filter_args.get("min_context"):
                if metadata["max_context_length"] < filter_args["min_context"]:
                    continue

            if filter_args.get("max_size"):
                if metadata["model_size_gb"] > filter_args["max_size"]:
                    continue

            filtered.append(model_key)

        logger.info(
            f"✔️ After filtering: {len(filtered)}/{len(models)} models remaining"
        )
        return filtered


class LMStudioBenchmark:
    """Main benchmark class"""

    @staticmethod
    def get_lmstudio_version() -> Optional[str]:
        """Retrieves LM Studio version"""
        try:
            result = subprocess.run(
                ["lms", "version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                stdout = result.stdout.strip()
                m = re.search(r"v\d+\.\d+\.\d+", stdout)
                if m:
                    return m.group(0)

                m2 = re.search(r"CLI commit:\s*([0-9a-fA-F]{6,40})", stdout)
                if m2:
                    commit = m2.group(1)
                    try:
                        import lmstudio

                        pkg_ver = getattr(lmstudio, "__version__", None)
                        if pkg_ver:
                            return f"{pkg_ver} (commit:{commit})"
                    except Exception:
                        pass
                    return f"commit:{commit}"

                first_line = stdout.split("\n")[0] if stdout else None
                return first_line if first_line else None
        except Exception as e:
            logger.debug("Error fetching LM Studio version: %s", e)
        return None

    @staticmethod
    def get_nvidia_driver_version() -> Optional[str]:
        """Retrieves NVIDIA Driver version"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return version if version else None
        except Exception:
            logger.debug("NVIDIA driver not available")
        return None

    @staticmethod
    def get_rocm_driver_version() -> Optional[str]:
        """Retrieves AMD ROCm/Driver version"""
        try:
            rocm_paths = ["/usr/bin/rocm-smi", "/usr/local/bin/rocm-smi"]
            rocm_paths.extend(glob.glob("/opt/rocm-*/bin/rocm-smi"))

            for rocm_smi in rocm_paths:
                try:
                    result = subprocess.run(
                        [rocm_smi, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        output = result.stdout.strip()
                        lines = output.split("\n")
                        if lines and "version" in lines[0].lower():
                            parts = lines[0].split(":")
                            if len(parts) > 1:
                                version = parts[1].strip()
                                if "/rocm-" in rocm_smi and "/bin/" in rocm_smi:
                                    tool_version = rocm_smi.split("/rocm-")[1].split(
                                        "/"
                                    )[0]
                                    return f"{version} (tool: {tool_version})"
                                return version
                        return output if output else None
                except Exception:
                    pass

            try:
                result = subprocess.run(
                    ["dpkg", "-l", "rocm-smi"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("ii") and "rocm-smi" in line:
                            parts = line.split()
                            if len(parts) >= 3:
                                return f"{parts[2]} (package)"
            except Exception:
                pass

            try:
                result = subprocess.run(
                    ["modinfo", "amdgpu"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split("\n"):
                        if line.startswith("srcversion:"):
                            srcver = line.split(":")[1].strip()[:12]
                            return f"amdgpu:{srcver}"
            except Exception:
                pass

        except Exception:
            logger.debug("ROCm driver not available")
        return None

    @staticmethod
    def get_intel_driver_version() -> Optional[str]:
        """Retrieves Intel GPU Driver version"""
        try:
            result = subprocess.run(
                ["intel_gpu_top", "--help"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split("\n"):
                if "version" in line.lower():
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "version" in part.lower() and i + 1 < len(parts):
                            return parts[i + 1]
                    return line.strip()
        except Exception:
            logger.debug("Intel GPU driver not available")
        return None

    @staticmethod
    def get_os_info() -> Tuple[Optional[str], Optional[str]]:
        """Retrieves operating system and kernel version"""
        try:
            import platform

            os_system = platform.system()

            if os_system == "Linux":
                try:
                    import distro

                    os_name = distro.name()
                    os_version = distro.version()
                    return os_name, os_version
                except Exception:
                    os_name = os_system
                    os_version = platform.release()
                    return os_name, os_version
            else:
                os_name = os_system
                os_version = platform.release()
                return os_name, os_version
        except Exception:
            logger.debug("OS info not available")
        return None, None

    @staticmethod
    def get_cpu_model() -> Optional[str]:
        """Retrieves CPU model"""
        try:
            import cpuinfo

            cpu = cpuinfo.get_cpu_info()
            brand = cpu.get("brand_raw", "")
            return brand if brand else None
        except Exception:
            logger.debug("CPU info not available")
        return None

    @staticmethod
    def get_python_version() -> Optional[str]:
        """Retrieves Python version"""
        try:
            import sys

            return (
                f"{sys.version_info.major}."
                f"{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            )
        except Exception:
            logger.debug("Python version not available")
        return None

    def __init__(
        self,
        num_runs: int = NUM_MEASUREMENT_RUNS,
        context_length: int = CONTEXT_LENGTH,
        prompt: str = STANDARD_PROMPT,
        model_limit: Optional[int] = None,
        filter_args: Optional[Dict] = None,
        compare_with: Optional[str] = None,
        rank_by: str = "speed",
        use_cache: bool = True,
        enable_profiling: bool = False,
        max_temp: Optional[float] = None,
        max_power: Optional[float] = None,
        use_gtt: bool = True,
        inference_overrides: Optional[Dict] = None,
        load_params: Optional[Dict] = None,
        use_rest_api: Optional[bool] = None,
    ):
        self.gpu_monitor = GPUMonitor()
        self.results: List[BenchmarkResult] = []
        self.num_measurement_runs = num_runs
        self.context_length = context_length
        self.prompt = prompt
        self.model_limit = model_limit
        self.filter_args = filter_args or {}
        self.compare_with = compare_with
        self.rank_by = rank_by
        self.use_cache = use_cache
        self.enable_profiling = enable_profiling
        self.max_temp = max_temp
        self.max_power = max_power
        self.use_gtt = use_gtt
        self.previous_results: List[BenchmarkResult] = []
        self._gtt_info = {}
        lmstudio_config = DEFAULT_CONFIG.get("lmstudio", {})
        self.use_rest_api = (
            use_rest_api
            if use_rest_api is not None
            else lmstudio_config.get("use_rest_api", False)
        )
        self.rest_client: Optional[LMStudioRESTClient] = None

        if self.use_rest_api:
            host = lmstudio_config.get("host", "localhost")
            ports = lmstudio_config.get("ports", [1234])
            port = ports[0] if ports else 1234
            base_url = f"http://{host}:{port}"
            api_token = lmstudio_config.get("api_token")
            self.rest_client = LMStudioRESTClient(
                base_url=base_url, api_token=api_token
            )
            logger.info("🌐 REST API mode enabled: %s", base_url)
        else:
            logger.info("🔧 SDK/CLI mode enabled")

        self.load_params = dict(DEFAULT_LOAD_PARAMS)
        if load_params:
            for key, value in load_params.items():
                if value is not None:
                    self.load_params[key] = value

        self.cache = BenchmarkCache()
        self.params_hash = BenchmarkCache.compute_params_hash(
            prompt, context_length, OPTIMIZED_INFERENCE_PARAMS, self.load_params
        )

        self.hardware_monitor = HardwareMonitor(
            self.gpu_monitor.gpu_type or "Unknown",
            self.gpu_monitor.gpu_tool or "",
            enabled=enable_profiling,
        )

        self.cli_args = {
            "runs": num_runs,
            "context": context_length,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "limit": model_limit,
            "only_vision": self.filter_args.get("only_vision", False),
            "only_tools": self.filter_args.get("only_tools", False),
            "quants": self.filter_args.get("quants"),
            "arch": self.filter_args.get("arch"),
            "params": self.filter_args.get("params"),
            "include_models": self.filter_args.get("include_models"),
            "exclude_models": self.filter_args.get("exclude_models"),
            "compare_with": compare_with,
            "rank_by": rank_by,
            "retest": not use_cache,
            "enable_profiling": enable_profiling,
            "max_temp": max_temp,
            "max_power": max_power,
            "use_gtt": use_gtt,
        }

        self.system_versions = {
            "lmstudio_version": self.get_lmstudio_version(),
            "app_version": APP_VERSION,
            "nvidia_driver_version": self.get_nvidia_driver_version(),
            "rocm_driver_version": self.get_rocm_driver_version(),
            "intel_driver_version": self.get_intel_driver_version(),
        }

        logger.info("📋 Collected version information:")
        for key, value in self.system_versions.items():
            logger.info("   • %s: %s", key, value if value else 'N/A')

        os_name, os_version = self.get_os_info()
        self.system_info = {
            "os_name": os_name,
            "os_version": os_version,
            "cpu_model": self.get_cpu_model(),
            "python_version": self.get_python_version(),
        }
        logger.info("💻 System information:")
        logger.info(
            f"   • OS: {self.system_info['os_name']} {self.system_info['os_version']}"
        )
        logger.info("   • CPU: %s", self.system_info['cpu_model'] or 'N/A')
        logger.info("   • Python: %s", self.system_info['python_version'])

        self.inference_params = OPTIMIZED_INFERENCE_PARAMS.copy()
        self.inference_overrides = inference_overrides or {}
        for k, v in (self.inference_overrides or {}).items():
            if v is not None and k in self.inference_params:
                self.inference_params[k] = v
        self.prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        self.params_hash = BenchmarkCache.compute_params_hash(
            prompt, context_length, self.inference_params, self.load_params
        )

        RESULTS_DIR.mkdir(exist_ok=True)

        if self.compare_with:
            self._load_previous_results()

    def __del__(self):
        """Cleanup REST client."""
        if self.rest_client:
            try:
                self.rest_client.close()
            except Exception:
                pass

    def _get_available_vram_gb(self) -> Optional[float]:
        """Determines available VRAM in GB (incl. GTT if enabled)"""
        try:
            gpu_type = self.gpu_monitor.gpu_type
            gpu_model = self.gpu_monitor.gpu_model or self.gpu_monitor.gpu_type
            gpu_tool = self.gpu_monitor.gpu_tool

            if not gpu_tool:
                return None

            if gpu_type == "NVIDIA":
                result = subprocess.run(
                    [
                        gpu_tool,
                        "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    vram_mb = float(result.stdout.strip().split("\n")[0])
                    return vram_mb / 1024

            elif gpu_type == "AMD":
                result = subprocess.run(
                    [gpu_tool, "--showmeminfo", "all"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    vram_total = vram_used = gtt_total = gtt_used = 0.0

                    for line in result.stdout.split("\n"):
                        if "VRAM Total Memory (B):" in line:
                            vram_total = float(line.split(":")[-1].strip()) / (1024**3)
                        elif "VRAM Total Used Memory (B):" in line:
                            vram_used = float(line.split(":")[-1].strip()) / (1024**3)
                        elif "GTT Total Memory (B):" in line:
                            gtt_total = float(line.split(":")[-1].strip()) / (1024**3)
                        elif "GTT Total Used Memory (B):" in line:
                            gtt_used = float(line.split(":")[-1].strip()) / (1024**3)

                    vram_free = vram_total - vram_used
                    gtt_free = gtt_total - gtt_used

                    self._gtt_info = {
                        "total": gtt_total,
                        "used": gtt_used,
                        "free": gtt_free,
                    }

                    if self.use_gtt and gtt_total > 0:
                        total_available = vram_free + gtt_free
                        logger.info("💾 Memory: %sGB VRAM + %sGB GTT = %sGB total",
                                vram_free,
                                gtt_free,
                                total_available)
                        return total_available
                    else:
                        logger.info("💾 Memory: %sGB VRAM (GTT disabled)", vram_free)
                        return vram_free

            elif gpu_type == "Intel":
                return 8.0

            return None

        except Exception as e:
            logger.debug("VRAM query failed: %s", e)
            return None

    def _predict_optimal_offload(self, model_size_gb: float) -> float:
        """Calculates optimal GPU offload based on VRAM and model size"""
        try:
            available_vram = self._get_available_vram_gb()

            if available_vram is None:
                logger.debug("VRAM not available, using default levels")
                return 1.0

            estimated_vram = model_size_gb * 1.2

            estimated_vram += self.context_length * CONTEXT_VRAM_FACTOR

            safe_vram = available_vram - VRAM_SAFETY_HEADROOM_GB

            if safe_vram <= 0:
                logger.warning("⚠️ Too little VRAM available: %sGB", available_vram)
                return 0.3

            if estimated_vram <= safe_vram:
                optimal_offload = 1.0
            else:
                optimal_offload = safe_vram / estimated_vram
                optimal_offload = max(0.3, min(1.0, optimal_offload))

            logger.info(
                f"📊 VRAM prediction: {available_vram:.1f}GB available, "
                f"{estimated_vram:.1f}GB estimated -> Offload "
                f"{optimal_offload:.2f}"
            )

            return round(optimal_offload, 1)

        except Exception as e:
            logger.debug("Offload prediction failed: %s", e)
            return 1.0

    def _get_cached_optimal_offload(
        self, model_key: str, model_size_gb: float
    ) -> Optional[float]:
        """Gets optimal offload from cache for similar models"""
        if not self.cache:
            return None

        try:
            metadata = ModelDiscovery.get_model_metadata(model_key)
            architecture = metadata.get("architecture", "unknown")

            if architecture == "unknown":
                return None

            conn = sqlite3.connect(self.cache.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT gpu_offload, model_size_gb
                FROM benchmark_results
                WHERE architecture = ?
                AND model_size_gb BETWEEN ? AND ?
                AND gpu_offload > 0
                ORDER BY timestamp DESC
                LIMIT 5
            """,
                (architecture, model_size_gb * 0.8, model_size_gb * 1.2),
            )

            results = cursor.fetchall()
            conn.close()

            if results:
                offloads = [r[0] for r in results]
                avg_offload = sum(offloads) / len(offloads)
                logger.info(
                    f"📚 Cache-Hit: Using average offload {avg_offload:.2f} "
                    f"for {architecture} (~{model_size_gb:.1f}GB)"
                )
                return round(avg_offload, 1)

            return None

        except Exception as e:
            logger.debug("Cache lookup failed: %s", e)
            return None

    def _get_smart_offload_levels(
        self, model_key: str, model_size_gb: float
    ) -> List[float]:
        """Generates intelligent GPU offload levels based on prediction and cache"""
        cached_offload = self._get_cached_optimal_offload(model_key, model_size_gb)
        if cached_offload:
            levels = [cached_offload]
            if cached_offload > 0.5:
                levels.append(round(cached_offload - 0.2, 1))
            if cached_offload < 0.9:
                levels.append(round(cached_offload + 0.1, 1))
            return sorted(set(levels), reverse=True)

        predicted_offload = self._predict_optimal_offload(model_size_gb)

        levels = [predicted_offload]

        if predicted_offload >= 0.8:
            levels.extend([1.0, 0.7, 0.5])
        elif predicted_offload >= 0.5:
            levels.extend([0.7, 0.5, 0.3])
        else:
            levels.extend([0.5, 0.3, 0.7])

        return sorted(set(levels), reverse=True)

    def _load_previous_results(self):
        """Loads previous benchmark results for comparison"""
        try:
            if not self.compare_with:
                return

            if self.compare_with.endswith(".json"):
                json_file = RESULTS_DIR / self.compare_with
            else:
                if self.compare_with.lower() == "latest":
                    json_files = sorted(RESULTS_DIR.glob("benchmark_results_*.json"))
                    if not json_files:
                        logger.warning("⚠️ No previous benchmark files found")
                        return
                    json_file = json_files[-1]
                else:
                    json_file = (
                        RESULTS_DIR / f"benchmark_results_{self.compare_with}.json"
                    )

            if not json_file.exists():
                logger.warning("⚠️ File not found: %s", json_file)
                return

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.previous_results = [BenchmarkResult(**item) for item in data]

            logger.info(
                f"✓ {len(self.previous_results)} previous results loaded from {json_file.name}"
            )
        except Exception as e:
            logger.error("❌ Error loading previous results: %s", e)

    def _matches_filters(self, result: BenchmarkResult) -> bool:
        """Checks if a BenchmarkResult passes the active filters"""
        if self.filter_args.get("only_vision") and not result.has_vision:
            return False

        if self.filter_args.get("only_tools") and not result.has_tools:
            return False

        if self.filter_args.get("quants"):
            quants = [q.strip().lower() for q in self.filter_args["quants"].split(",")]
            if not any(q in result.quantization.lower() for q in quants):
                return False

        if self.filter_args.get("arch"):
            archs = [a.strip().lower() for a in self.filter_args["arch"].split(",")]
            if not any(a in result.architecture.lower() for a in archs):
                return False

        if self.filter_args.get("params"):
            params = [p.strip().upper() for p in self.filter_args["params"].split(",")]
            if result.params_size.upper() not in params:
                return False

        if self.filter_args.get("min_context"):
            if result.max_context_length < self.filter_args["min_context"]:
                return False

        if self.filter_args.get("max_size"):
            if result.model_size_gb > self.filter_args["max_size"]:
                return False

        if self.filter_args.get("include_models"):
            import re

            try:
                pattern = re.compile(self.filter_args["include_models"], re.IGNORECASE)
                model_full = f"{result.model_name}@{result.quantization}"
                if not pattern.search(model_full):
                    return False
            except re.error:
                pass

        if self.filter_args.get("exclude_models"):
            import re

            try:
                pattern = re.compile(self.filter_args["exclude_models"], re.IGNORECASE)
                model_full = f"{result.model_name}@{result.quantization}"
                if pattern.search(model_full):
                    return False
            except re.error:
                pass

        return True

    def _calculate_delta(self, current: BenchmarkResult) -> Optional[Dict]:
        """Calculates delta to previous benchmark for same model+quantization"""
        if not self.previous_results:
            return None

        for prev in self.previous_results:
            if (
                prev.model_name == current.model_name
                and prev.quantization == current.quantization
            ):
                speed_delta = current.avg_tokens_per_sec - prev.avg_tokens_per_sec
                speed_delta_pct = (
                    (speed_delta / prev.avg_tokens_per_sec * 100)
                    if prev.avg_tokens_per_sec > 0
                    else 0
                )

                return {
                    "prev_speed": prev.avg_tokens_per_sec,
                    "current_speed": current.avg_tokens_per_sec,
                    "speed_delta": speed_delta,
                    "speed_delta_pct": speed_delta_pct,
                    "prev_timestamp": prev.timestamp,
                }

        return None

    def benchmark_model(self, model_key: str) -> Optional[BenchmarkResult]:
        """Performs benchmark for a specific model"""
        logger.info("🎯 Starting benchmark for %s", model_key)

        benchmark_start_time = time.time()
        error_count = 0

        if self.use_rest_api and self.rest_client:
            try:
                models = self.rest_client.list_models()
                for model in models:
                    for instance in model.loaded_instances:
                        self._unload_model_rest(instance.instance_id)
                logger.info("🧹 All models unloaded via REST")
                time.sleep(1)
            except Exception as e:
                logger.warning("⚠️ REST error unloading: %s", e)
        else:
            try:
                subprocess.run(
                    ["lms", "unload", "--all"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.info("🧹 All models unloaded")
                time.sleep(1)
            except Exception as e:
                logger.warning("⚠️ Error unloading all models: %s", e)

        if "@" in model_key:
            model_name, quantization = model_key.split("@", 1)
        else:
            model_name = model_key
            quantization = "unknown"

        metadata = ModelDiscovery.get_model_metadata(model_key)
        model_size_gb = metadata.get("model_size_gb", 0)

        smart_offload_levels = self._get_smart_offload_levels(model_key, model_size_gb)
        logger.info("🎯 Intelligente Offload-Levels: %s", smart_offload_levels)

        used_offload = smart_offload_levels[0] if smart_offload_levels else 1.0
        instance_id: Optional[str] = None

        if self.use_rest_api and self.rest_client:
            try:
                instance_id = self._load_model_rest(model_key, used_offload)
                if not instance_id:
                    logger.info(
                        f"⬇️ Model not loaded; starting download for {model_key}"
                    )
                    try:
                        self.rest_client.download_model(model_key)
                        time.sleep(3)
                        instance_id = self._load_model_rest(model_key, used_offload)
                    except Exception as ex:
                        logger.warning(
                            f"⚠️ Download/Load via REST failed: {ex}"
                        )
                if instance_id:
                    logger.info("🧩 Using REST instance id: %s", instance_id)
                else:
                    logger.warning("⚠️ Could not load model via REST: %s", model_key)
            except Exception as e:
                logger.warning("⚠️ REST error while loading model: %s", e)

        try:
            logger.info("🔥 Warmup for %s...", model_key)
            for _ in range(NUM_WARMUP_RUNS):
                warmup_result = self._run_inference(model_key, instance_id)
                if not warmup_result:
                    logger.error("❌ Warmup for %s failed", model_key)
                    return None

            self.hardware_monitor.start()

            logger.info("📊 Performing %s measurements...", self.num_measurement_runs)
            measurements = []
            vram_after = "N/A"
            for run in range(self.num_measurement_runs):
                vram_before = self.gpu_monitor.get_vram_usage()
                stats = self._run_inference(model_key, instance_id)
                vram_after = self.gpu_monitor.get_vram_usage()

                if stats:
                    measurements.append(stats)
                    logger.info(
                        "⚡ Run %d/%d: %.2f tokens/s",
                        run + 1,
                        self.num_measurement_runs,
                        stats["tokens_per_second"],
                    )
                else:
                    logger.warning(
                        "⚠️ Run %d/%d failed",
                        run + 1,
                        self.num_measurement_runs,
                    )

            profiling_stats = self.hardware_monitor.stop()

            if measurements:
                result = self._calculate_averages(
                    model_name,
                    quantization,
                    used_offload,
                    vram_after,
                    measurements,
                    model_key,
                )

                if self.enable_profiling:
                    result.temp_celsius_min = profiling_stats.get("temp_celsius_min")
                    result.temp_celsius_max = profiling_stats.get("temp_celsius_max")
                    result.temp_celsius_avg = profiling_stats.get("temp_celsius_avg")
                    result.power_watts_min = profiling_stats.get("power_watts_min")
                    result.power_watts_max = profiling_stats.get("power_watts_max")
                    result.power_watts_avg = profiling_stats.get("power_watts_avg")

                    if (
                        self.max_temp
                        and result.temp_celsius_max
                        and result.temp_celsius_max > self.max_temp
                    ):
                        logger.warning("⚠️ Max. temperature exceeded: %s°C > %s°C",
                                result.temp_celsius_max,
                                self.max_temp)

                    if (
                        self.max_power
                        and result.power_watts_max
                        and result.power_watts_max > self.max_power
                    ):
                        logger.warning("⚠️ Max. power exceeded: %sW > %sW",
                                result.power_watts_max,
                                self.max_power)

                benchmark_end_time = time.time()
                result.benchmark_duration_seconds = round(
                    benchmark_end_time - benchmark_start_time, 2
                )
                result.error_count = error_count
                result.model_key = model_key
                result.prompt_hash = self.prompt_hash
                result.params_hash = self.params_hash
                result.context_length = self.context_length
                result.os_name = self.system_info.get("os_name")
                result.os_version = self.system_info.get("os_version")
                result.cpu_model = self.system_info.get("cpu_model")
                result.python_version = self.system_info.get("python_version")
                result.inference_params_hash = hashlib.sha256(
                    json.dumps(self.inference_params, sort_keys=True).encode()
                ).hexdigest()[:8]

                if self.cache:
                    for run_idx, measurement in enumerate(measurements):
                        tps_per_gb = (
                            measurement["tokens_per_second"] / result.model_size_gb
                            if result.model_size_gb > 0
                            else 0.0
                        )
                        params_billions = (
                            float(result.params_size.replace("B", ""))
                            if result.params_size.endswith("B")
                            and result.params_size[:-1].replace(".", "", 1).isdigit()
                            else 0.0
                        )
                        tps_per_billion = (
                            measurement["tokens_per_second"] / params_billions
                            if params_billions > 0
                            else 0.0
                        )

                        run_result = BenchmarkResult(
                            model_name=model_name,
                            quantization=quantization,
                            gpu_type=result.gpu_type,
                            gpu_offload=used_offload,
                            vram_mb=vram_after,
                            avg_tokens_per_sec=measurement["tokens_per_second"],
                            avg_ttft=measurement["time_to_first_token"],
                            avg_gen_time=measurement["generation_time"],
                            prompt_tokens=measurement["prompt_tokens"],
                            completion_tokens=measurement["completion_tokens"],
                            timestamp=result.timestamp,
                            params_size=result.params_size,
                            architecture=result.architecture,
                            max_context_length=result.max_context_length,
                            model_size_gb=result.model_size_gb,
                            has_vision=result.has_vision,
                            has_tools=result.has_tools,
                            tokens_per_sec_per_gb=tps_per_gb,
                            tokens_per_sec_per_billion_params=tps_per_billion,
                            lmstudio_version=result.lmstudio_version,
                            nvidia_driver_version=result.nvidia_driver_version,
                            rocm_driver_version=result.rocm_driver_version,
                            intel_driver_version=result.intel_driver_version,
                            temperature=result.temperature,
                            top_k_sampling=result.top_k_sampling,
                            top_p_sampling=result.top_p_sampling,
                            min_p_sampling=result.min_p_sampling,
                            repeat_penalty=result.repeat_penalty,
                            max_tokens=result.max_tokens,
                            num_runs=1,
                            runs_averaged_from=len(measurements),
                            warmup_runs=NUM_WARMUP_RUNS,
                            run_index=run_idx,
                            model_key=model_key,
                            prompt_hash=self.prompt_hash,
                            params_hash=self.params_hash,
                            context_length=self.context_length,
                            os_name=result.os_name,
                            os_version=result.os_version,
                            cpu_model=result.cpu_model,
                            python_version=result.python_version,
                            benchmark_duration_seconds=result.benchmark_duration_seconds
                            / len(measurements),
                            error_count=error_count,
                            inference_params_hash=result.inference_params_hash,
                            n_gpu_layers=self.load_params.get("n_gpu_layers"),
                            n_batch=self.load_params.get("n_batch"),
                            n_threads=self.load_params.get("n_threads"),
                            flash_attention=self.load_params.get("flash_attention"),
                            rope_freq_base=self.load_params.get("rope_freq_base"),
                            rope_freq_scale=self.load_params.get("rope_freq_scale"),
                            use_mmap=self.load_params.get("use_mmap"),
                            use_mlock=self.load_params.get("use_mlock"),
                            kv_cache_quant=self.load_params.get("kv_cache_quant"),
                        )

                        self.cache.save_result(
                            run_result,
                            model_key,
                            self.params_hash,
                            self.prompt,
                            self.context_length,
                        )

                logger.info("✓ %s: %s tokens/s (Duration: %ss)", model_key,
                        result.avg_tokens_per_sec,
                        result.benchmark_duration_seconds)
                try:
                    if self.use_rest_api and instance_id:
                        self._unload_model_rest(instance_id)
                except Exception:
                    pass

                return result
            else:
                logger.error("❌ No successful measurements for %s", model_key)
                return None

        except Exception as e:
            logger.error("❌ Error benchmarking %s: %s", model_key, e)
            error_count += 1
            return None

    def _load_model(self, model_key: str, gpu_offload: float) -> bool:
        """Loads a model into memory"""
        try:
            cmd = [
                "lms",
                "load",
                model_key,
                "--gpu",
                str(gpu_offload),
                "--context-length",
                str(CONTEXT_LENGTH),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            return result.returncode == 0

        except Exception as e:
            logger.error("❌ Error loading %s: %s", model_key, e)
            return False

    def _unload_model(self, model_key: str):
        """Unloads a model from memory"""
        try:
            subprocess.run(
                ["lms", "unload", model_key], capture_output=True, timeout=30
            )
        except Exception as e:
            logger.warning("⚠️ Error unloading %s: %s", model_key, e)

    def _run_inference(
        self, model_key: str, instance_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Performs inference and returns stats (dispatcher)."""
        if self.use_rest_api:
            return self._run_inference_rest(model_key, instance_id)
        else:
            return self._run_inference_sdk(model_key)

    def _run_inference_sdk(self, model_key: str) -> Optional[Dict]:
        """Performs inference via SDK and returns stats."""
        try:
            import lmstudio as lms

            load_config_params: Dict[str, Any] = {
                "context_length": self.context_length,
            }

            flash_attn = self.load_params.get("flash_attention")
            if flash_attn is not None and isinstance(flash_attn, bool):
                load_config_params["flash_attention"] = flash_attn

            use_mmap = self.load_params.get("use_mmap")
            if use_mmap is not None and isinstance(use_mmap, bool):
                load_config_params["try_mmap"] = use_mmap

            kv_quant = self.load_params.get("kv_cache_quant")
            if (
                kv_quant
                and isinstance(kv_quant, str)
                and kv_quant
                in ["f32", "f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
            ):
                load_config_params["llama_k_cache_quantization_type"] = kv_quant
                load_config_params["llama_v_cache_quantization_type"] = kv_quant

            logger.info(
                f"⚙️ Load Config: context={self.context_length}, "
                f"n_gpu_layers={self.load_params.get('n_gpu_layers')}, "
                f"n_batch={self.load_params.get('n_batch')}, "
                f"n_threads={self.load_params.get('n_threads')}, "
                f"flash_attention={self.load_params.get('flash_attention')}, "
                f"rope_freq_base={self.load_params.get('rope_freq_base')}, "
                f"rope_freq_scale={self.load_params.get('rope_freq_scale')}, "
                f"use_mmap={self.load_params.get('use_mmap')}, "
                f"use_mlock={self.load_params.get('use_mlock')}, "
                f"kv_cache_quant={kv_quant}"
            )

            model = lms.llm(
                model_key, config=lms.LlmLoadModelConfig(**load_config_params)
            )

            prediction_config = lms.LlmPredictionConfig(
                temperature=self.inference_params["temperature"],
                top_k_sampling=self.inference_params["top_k_sampling"],
                top_p_sampling=self.inference_params["top_p_sampling"],
                min_p_sampling=self.inference_params["min_p_sampling"],
                repeat_penalty=self.inference_params["repeat_penalty"],
                max_tokens=self.inference_params["max_tokens"],
            )

            logger.info(
                f"⚙️ Inference Config: temp={prediction_config.temperature}, "
                f"top_k={prediction_config.top_k_sampling}, "
                f"top_p={prediction_config.top_p_sampling}, "
                f"min_p={prediction_config.min_p_sampling}, "
                f"repeat_penalty={prediction_config.repeat_penalty}, "
                f"max_tokens={prediction_config.max_tokens}"
            )

            start_time = time.time()
            result = model.respond(self.prompt, config=prediction_config)
            end_time = time.time()

            generation_time = end_time - start_time

            stats = result.stats

            tokens_per_sec = 0.0
            if generation_time > 0 and stats.predicted_tokens_count:
                tokens_per_sec = float(stats.predicted_tokens_count) / generation_time

            return {
                "tokens_per_second": tokens_per_sec,
                "time_to_first_token": stats.time_to_first_token_sec,
                "generation_time": generation_time,
                "prompt_tokens": stats.prompt_tokens_count or 0,
                "completion_tokens": stats.predicted_tokens_count or 0,
            }

        except Exception as e:
            logger.error("❌ Error during inference with %s: %s", model_key, e)
            return None

    def _run_inference_rest(
        self, model_key: str, instance_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Performs inference via REST API and returns stats."""
        if not self.rest_client:
            logger.error("❌ REST Client not initialized")
            return None

        try:
            messages = [{"role": "user", "content": self.prompt}]

            result = self.rest_client.chat_stream(
                messages=messages,
                model=instance_id or model_key,
                context_length=self.context_length,
                temperature=self.inference_params["temperature"],
                max_tokens=self.inference_params["max_tokens"],
            )

            stats = result.get("stats")
            if not stats:
                logger.warning("⚠️ No stats in REST response")
                return None

            tokens_per_sec = stats.tokens_per_second
            if not tokens_per_sec and stats.tokens_out > 0:
                total_time_s = result.get("total_time_s", 0)
                if total_time_s > 0:
                    tokens_per_sec = stats.tokens_out / total_time_s

            return {
                "tokens_per_second": tokens_per_sec,
                "time_to_first_token": stats.time_to_first_token_ms / 1000.0,
                "generation_time": result.get("total_time_s", 0),
                "prompt_tokens": stats.tokens_in,
                "completion_tokens": stats.tokens_out,
            }

        except Exception as e:
            logger.error("❌ REST error during inference with %s: %s", model_key, e)
            return None

    def _load_model_rest(
        self, model_key: str, gpu_offload: Optional[float] = None
    ) -> Optional[str]:
        """Loads a model via REST API."""
        if not self.rest_client:
            return None

        try:
            n_parallel = self.load_params.get("n_parallel")
            unified_kv = self.load_params.get("unified_kv_cache")

            instance_id = self.rest_client.load_model(
                model_key=model_key,
                context_length=self.context_length,
                n_parallel=n_parallel,
                unified_kv_cache=unified_kv,
                gpu_offload=gpu_offload,
            )

            logger.info("✓ Model loaded via REST: %s", instance_id)
            return instance_id

        except Exception as e:
            logger.error("❌ REST error loading %s: %s", model_key, e)
            return None

    def _unload_model_rest(self, instance_id: str) -> bool:
        """Unloads a model via REST API."""
        if not self.rest_client:
            return False

        try:
            self.rest_client.unload_model(instance_id)
            logger.info("✓ Model unloaded via REST: %s", instance_id)
            return True
        except Exception as e:
            logger.warning("⚠️ REST error unloading: %s", e)
            return False

    def _calculate_averages(
        self,
        model_name: str,
        quantization: str,
        gpu_offload: float,
        vram_mb: str,
        measurements: List[Dict],
        model_key: str,
    ) -> BenchmarkResult:
        """Calculates average values from measurements"""
        avg_tokens_per_sec = sum(m["tokens_per_second"] for m in measurements) / len(
            measurements
        )
        avg_ttft = sum(m["time_to_first_token"] for m in measurements) / len(
            measurements
        )
        avg_gen_time = sum(m["generation_time"] for m in measurements) / len(
            measurements
        )
        prompt_tokens = measurements[0]["prompt_tokens"]
        completion_tokens = int(
            sum(m["completion_tokens"] for m in measurements) / len(measurements)
        )

        metadata = ModelDiscovery.get_model_metadata(model_key)

        model_size_gb = metadata.get("model_size_gb", 0.0)
        params_size_str = metadata.get("params_size", "unknown")

        tokens_per_sec_per_gb = (
            round(avg_tokens_per_sec / model_size_gb, 2) if model_size_gb > 0 else 0.0
        )

        try:
            params_billion = float(params_size_str.upper().replace("B", "").strip())
            tokens_per_sec_per_billion_params = round(
                avg_tokens_per_sec / params_billion, 2
            )
        except (ValueError, AttributeError):
            tokens_per_sec_per_billion_params = 0.0

        result = BenchmarkResult(
            model_name=model_name,
            quantization=quantization,
            gpu_type=self.gpu_monitor.gpu_model
            or self.gpu_monitor.gpu_type
            or "Unknown",
            gpu_offload=gpu_offload,
            vram_mb=vram_mb,
            avg_tokens_per_sec=round(avg_tokens_per_sec, 2),
            avg_ttft=round(avg_ttft, 3),
            avg_gen_time=round(avg_gen_time, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            params_size=metadata.get("params_size", "unknown"),
            architecture=metadata.get("architecture", "unknown"),
            max_context_length=metadata.get("max_context_length", 0),
            model_size_gb=model_size_gb,
            has_vision=metadata.get("has_vision", False),
            has_tools=metadata.get("has_tools", False),
            tokens_per_sec_per_gb=tokens_per_sec_per_gb,
            tokens_per_sec_per_billion_params=tokens_per_sec_per_billion_params,
            gtt_enabled=self.use_gtt if self._gtt_info else None,
            gtt_total_gb=(
                round(self._gtt_info.get("total", 0), 2) if self._gtt_info else None
            ),
            gtt_used_gb=(
                round(self._gtt_info.get("used", 0), 2) if self._gtt_info else None
            ),
            temperature=self.inference_params.get("temperature"),
            top_k_sampling=self.inference_params.get("top_k_sampling"),
            top_p_sampling=self.inference_params.get("top_p_sampling"),
            min_p_sampling=self.inference_params.get("min_p_sampling"),
            repeat_penalty=self.inference_params.get("repeat_penalty"),
            max_tokens=self.inference_params.get("max_tokens"),
            num_runs=self.num_measurement_runs,
            runs_averaged_from=len(measurements),
            warmup_runs=NUM_WARMUP_RUNS,
            lmstudio_version=self.system_versions.get("lmstudio_version"),
            app_version=APP_VERSION,
            nvidia_driver_version=self.system_versions.get("nvidia_driver_version"),
            rocm_driver_version=self.system_versions.get("rocm_driver_version"),
            intel_driver_version=self.system_versions.get("intel_driver_version"),
        )

        delta = self._calculate_delta(result)
        if delta:
            result.speed_delta_pct = delta["speed_delta_pct"]
            result.prev_timestamp = delta["prev_timestamp"]

        return result

    def run_all_benchmarks(self) -> str:
        """Performs benchmarks for all available models.

        Returns:
            Status string: "completed", "no_new_models", or "failed".
        """
        if not LMStudioServerManager.ensure_server_running():
            logger.error("❌ Server could not be started, aborting")
            return "failed"

        ModelDiscovery._get_metadata_cache()

        models = ModelDiscovery.get_installed_models()
        if not models:
            logger.error("❌ No models found")
            return "failed"

        models = ModelDiscovery.filter_models(models, self.filter_args)
        if not models:
            logger.error("❌ No models remaining after filtering")
            return "failed"

        logger.info("")
        logger.info("📊 Models detected: %d total", len(models))
        logger.info("🔍 App Version: %s", APP_VERSION)
        logger.info("")

        newly_tested_models = []
        if self.cache and self.use_cache:
            try:
                cache_count = len(self.cache.list_cached_models())
                logger.info("💽 Cache DB: %s", format_path_for_logs(DATABASE_FILE))
                logger.info("💽 Cached entries available: %d", cache_count)
            except Exception as e:
                logger.debug("Cache stats read failed: %s", e)
        if self.cache and self.use_cache:
            cached_models = []
            new_models = []
            exact_hits = 0
            fallback_hits = 0

            for model_key in models:
                cached = self.cache.get_cached_result(model_key, self.params_hash)
                is_fallback = False
                if not cached:
                    cached = self.cache.get_latest_result_for_model(model_key)
                    is_fallback = cached is not None
                if cached:
                    cached_models.append((model_key, cached))
                    if is_fallback:
                        fallback_hits += 1
                    else:
                        exact_hits += 1
                else:
                    new_models.append(model_key)

            if self.model_limit and self.model_limit < len(new_models):
                logger.info("⚙️ Model limit set: Testing max. %s new models (+ %s cached)",
                        self.model_limit,
                        len(cached_models))
                new_models = new_models[: self.model_limit]
            elif self.model_limit:
                logger.info("⚙️ Model limit: %s new models + %s cached = %s total",
                        len(new_models),
                        len(cached_models),
                        len(new_models)
                        + len(cached_models))

            if cached_models:
                logger.info("")
                logger.info("📦 === Cached Models ===")
                logger.info(
                    f"💾 {len(cached_models)} models already tested "
                    "(will be loaded from cache):"
                )
                for model_key, cached in cached_models[:10]:
                    date_part = (
                        cached.timestamp.split("T")[0]
                        if "T" in cached.timestamp
                        else cached.timestamp[:10]
                    )
                    logger.info(
                        f"  • {model_key}: {cached.avg_tokens_per_sec:.2f} "
                        f"tok/s (last tested: {date_part})"
                    )
                if len(cached_models) > 10:
                    logger.info("  ... and %s more", len(cached_models) - 10)
                logger.debug(
                    "🔍 Cache hits breakdown: %d exact + %d fallback = %d total",
                    exact_hits, fallback_hits, len(cached_models)
                )
                logger.info("")

                for model_key, cached in cached_models:
                    self.results.append(cached)

            if new_models:
                logger.info(
                    f"🚀 Starting benchmark for {len(new_models)} new models..."
                )
                models = new_models
            else:
                logger.info("")
                logger.info("💚 All models already cached - using cached results")
                logger.info("📝 No new tests necessary (no new models to test)")
                logger.info("")
                return "no_new_models"
        else:
            if self.model_limit and self.model_limit < len(models):
                logger.info("⚙️ Model limit set: Testing only first %s of %s models",
                        self.model_limit,
                        len(models))
                models = models[: self.model_limit]
            logger.info("🚀 Starting benchmark for %s models...", len(models))

        for model_key in tqdm(models, desc="Benchmarking models"):
            result = self.benchmark_model(model_key)
            if result:
                self.results.append(result)
                newly_tested_models.append(result)

        if newly_tested_models:
            logger.info("📊 Exporting reports for %s newly tested models...",
                    len(newly_tested_models))
            self._export_results_to_files(newly_tested_models)
        else:
            logger.warning("⚠️ No new models tested - no reports generated")

        try:
            subprocess.run(
                ["lms", "unload", "--all"], capture_output=True, text=True, timeout=30
            )
            logger.info("🧹 All models unloaded (cleanup)")
            time.sleep(1)
        except Exception as e:
            logger.warning("⚠️ Error unloading all models: %s", e)

        logger.info("✅ Benchmark completed. %s/%s models successfully tested",
                len(newly_tested_models),
                len(models))
        return "completed"

    def _analyze_best_quantizations(self) -> Dict[str, Dict]:
        """Analyzes best quantization per model based on different criteria"""
        best_by_model = {}

        for result in self.results:
            model_key = result.model_name

            if model_key not in best_by_model:
                best_by_model[model_key] = {
                    "best_speed": None,
                    "best_efficiency": None,
                    "best_ttft": None,
                    "all_quantizations": [],
                }

            best_by_model[model_key]["all_quantizations"].append(result)

            if (
                best_by_model[model_key]["best_speed"] is None
                or result.avg_tokens_per_sec
                > best_by_model[model_key]["best_speed"].avg_tokens_per_sec
            ):
                best_by_model[model_key]["best_speed"] = result

            if (
                best_by_model[model_key]["best_efficiency"] is None
                or result.tokens_per_sec_per_gb
                > best_by_model[model_key]["best_efficiency"].tokens_per_sec_per_gb
            ):
                best_by_model[model_key]["best_efficiency"] = result

            if (
                best_by_model[model_key]["best_ttft"] is None
                or result.avg_ttft < best_by_model[model_key]["best_ttft"].avg_ttft
            ):
                best_by_model[model_key]["best_ttft"] = result

        return best_by_model

    def sort_results(self, rank_by: str = "speed") -> List[BenchmarkResult]:
        """Sorts results by different criteria"""
        if rank_by == "speed":
            return sorted(
                self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True
            )
        elif rank_by == "efficiency":
            return sorted(
                self.results, key=lambda x: x.tokens_per_sec_per_gb, reverse=True
            )
        elif rank_by == "ttft":
            return sorted(self.results, key=lambda x: x.avg_ttft, reverse=False)
        elif rank_by == "vram":

            def get_vram_mb(result):
                try:
                    return (
                        float(result.vram_mb.split()[0])
                        if isinstance(result.vram_mb, str)
                        else float(result.vram_mb)
                    )
                except BaseException:
                    return 999999

            return sorted(self.results, key=get_vram_mb, reverse=False)
        else:
            return sorted(
                self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True
            )

    def calculate_percentile_stats(self) -> Dict[str, Dict]:
        """Calculates P50, P95, P99 statistics for benchmark metrics"""
        if not self.results or len(self.results) < 3:
            return {}

        speeds = [
            r.avg_tokens_per_sec for r in self.results if r.avg_tokens_per_sec > 0
        ]
        ttfts = [r.avg_ttft for r in self.results if r.avg_ttft > 0]
        vram_values = []
        for r in self.results:
            try:
                vram_mb = (
                    float(r.vram_mb.split()[0])
                    if isinstance(r.vram_mb, str)
                    else float(r.vram_mb)
                )
                vram_values.append(vram_mb)
            except BaseException:
                pass

        stats = {}

        if speeds:
            if len(speeds) >= 3:
                quantile_data = quantiles(speeds, n=100, method="inclusive")
                stats["speed"] = {
                    "avg": round(mean(speeds), 2),
                    "median": round(median(speeds), 2),
                    "p95": round(quantile_data[94], 2),
                    "p99": round(quantile_data[98], 2),
                    "min": round(min(speeds), 2),
                    "max": round(max(speeds), 2),
                }

        if ttfts:
            if len(ttfts) >= 3:
                quantile_data = quantiles(ttfts, n=100, method="inclusive")
                stats["ttft"] = {
                    "avg": round(mean(ttfts), 3),
                    "median": round(median(ttfts), 3),
                    "p95": round(quantile_data[94], 3),
                    "p99": round(quantile_data[98], 3),
                    "min": round(min(ttfts), 3),
                    "max": round(max(ttfts), 3),
                }

        if vram_values:
            if len(vram_values) >= 3:
                quantile_data = quantiles(vram_values, n=100, method="inclusive")
                stats["vram"] = {
                    "avg": round(mean(vram_values), 0),
                    "median": round(median(vram_values), 0),
                    "p95": round(quantile_data[94], 0),
                    "p99": round(quantile_data[98], 0),
                    "min": round(min(vram_values), 0),
                    "max": round(max(vram_values), 0),
                }

        return stats

    def generate_quantization_comparison(self) -> Dict[str, Dict]:
        """Generates comparison table Q4 vs Q5 vs Q6 vs Q8 per model"""
        if not self.results:
            return {}

        model_quants = {}
        for result in self.results:
            if result.model_name not in model_quants:
                model_quants[result.model_name] = {}

            quant_level = result.quantization.split("_")[0].lower()
            if quant_level not in model_quants[result.model_name]:
                model_quants[result.model_name][quant_level] = result
            else:
                if (
                    result.avg_tokens_per_sec
                    > model_quants[result.model_name][quant_level].avg_tokens_per_sec
                ):
                    model_quants[result.model_name][quant_level] = result

        comparison = {}
        for model, quants in sorted(model_quants.items()):
            comparison[model] = {"q4": None, "q5": None, "q6": None, "q8": None}
            for q_level in ["q4", "q5", "q6", "q8"]:
                if q_level in quants:
                    r = quants[q_level]
                    comparison[model][q_level] = {
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "efficiency": round(r.tokens_per_sec_per_gb, 2),
                        "vram_mb": r.vram_mb,
                        "ttft": round(r.avg_ttft * 1000, 1),
                    }

        return comparison

    def _generate_best_practices(self) -> List[str]:
        """Generates best-practice recommendations based on hardware and benchmark results"""
        recommendations = []

        if not self.results:
            return recommendations

        gpu_model = self.gpu_monitor.gpu_model or self.gpu_monitor.gpu_type or "Unknown"

        best_speed = max(self.results, key=lambda x: x.avg_tokens_per_sec)
        best_efficiency = max(self.results, key=lambda x: x.tokens_per_sec_per_gb)
        best_ttft = min(
            self.results, key=lambda x: x.avg_ttft if x.avg_ttft > 0 else float("inf")
        )

        best_balance = max(
            self.results,
            key=lambda x: x.avg_tokens_per_sec * 0.6 + x.tokens_per_sec_per_gb * 0.4,
        )

        recommendations.append(f"🖥️  Hardware: {gpu_model} detected")
        recommendations.append("")
        recommendations.append("⚡ Fastest model:")
        recommendations.append(
            f"   → {best_speed.model_name} ({best_speed.quantization})"
        )
        recommendations.append(f"   → {best_speed.avg_tokens_per_sec:.2f} tokens/s")
        recommendations.append("")
        recommendations.append("💎 Most efficient model (tokens/s per GB):")
        recommendations.append(
            f"   → {best_efficiency.model_name} ({best_efficiency.quantization})"
        )
        recommendations.append(
            f"   → {best_efficiency.tokens_per_sec_per_gb:.2f} tokens/s/GB"
        )
        recommendations.append(f"   → Size: {best_efficiency.model_size_gb:.2f} GB")
        recommendations.append("")
        recommendations.append("🚀 Fastest response time (TTFT):")
        recommendations.append(
            f"   → {best_ttft.model_name} ({best_ttft.quantization})"
        )
        recommendations.append(
            f"   → {best_ttft.avg_ttft * 1000:.0f} ms until first token"
        )
        recommendations.append("")
        recommendations.append("⚖️  Beste Balance (Speed + Effizienz):")
        recommendations.append(
            f"   → {best_balance.model_name} ({best_balance.quantization})"
        )
        recommendations.append(
            f"   → {best_balance.avg_tokens_per_sec:.2f} tokens/s, {best_balance.model_size_gb:.2f} GB"
        )
        recommendations.append("")
        recommendations.append("📊 Quantisierungs-Tipps:")
        q4_models = [r for r in self.results if "q4" in r.quantization.lower()]
        q6_models = [r for r in self.results if "q6" in r.quantization.lower()]

        if q4_models and q6_models:
            avg_q4_speed = sum(r.avg_tokens_per_sec for r in q4_models) / len(q4_models)
            avg_q6_speed = sum(r.avg_tokens_per_sec for r in q6_models) / len(q6_models)
            speed_diff = ((avg_q4_speed - avg_q6_speed) / avg_q6_speed) * 100

            recommendations.append(
                f"   → Q4 is on average {abs(speed_diff):.0f}% "
                f"{'faster' if speed_diff > 0 else 'slower'} than Q6"
            )
            recommendations.append(
                f"   → Q4: Faster, less quality | Q6: Slower, better quality"
            )

        recommendations.append("")

        vram_info = []
        for result in sorted(self.results, key=lambda x: x.model_size_gb)[:3]:
            if result.model_size_gb <= 4:
                vram_info.append(
                    f"   → <4 GB VRAM: {result.model_name} ({result.quantization})"
                )
        for result in sorted(self.results, key=lambda x: x.model_size_gb):
            if 4 < result.model_size_gb <= 8:
                vram_info.append(
                    f"   → 4-8 GB VRAM: {result.model_name} ({result.quantization})"
                )
                break
        for result in sorted(self.results, key=lambda x: x.model_size_gb):
            if 8 < result.model_size_gb <= 12:
                vram_info.append(
                    f"   → 8-12 GB VRAM: {result.model_name} ({result.quantization})"
                )
                break

        if vram_info:
            recommendations.append("🎯 VRAM-Empfehlungen:")
            recommendations.extend(vram_info[:3])

        return recommendations

    def load_all_historical_data(self) -> Dict[str, List[Dict]]:
        """Loads all historical benchmark results and groups by model+quantization"""
        trends = {}
        results_dir = RESULTS_DIR

        for json_file in sorted(results_dir.glob("benchmark_results_*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                for item in data:
                    key = f"{item['model_name']}@{item['quantization']}"
                    if key not in trends:
                        trends[key] = []

                    timestamp_str = item.get("timestamp", "2026-01-01 00:00:00")
                    trends[key].append(
                        {
                            "timestamp": timestamp_str,
                            "speed": item["avg_tokens_per_sec"],
                            "ttft": item["avg_ttft"],
                            "vram": item.get("vram_mb", 0),
                        }
                    )
            except Exception as e:
                logger.debug("Error loading %s: %s", json_file, e)

        return trends

    def generate_trend_chart(self) -> Optional[str]:
        """Generates Plotly line chart for performance trends over time"""
        if not PLOTLY_AVAILABLE or not self.previous_results or go is None:
            return None

        try:
            trends = self.load_all_historical_data()
            if not trends:
                return None

            fig = go.Figure()

            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
            color_idx = 0

            for key, history in sorted(trends.items()):
                if len(history) < 2:
                    continue

                history_sorted = sorted(history, key=lambda x: x["timestamp"])

                timestamps = [h["timestamp"] for h in history_sorted]
                speeds = [h["speed"] for h in history_sorted]

                model_name = key.split("@")[0].split("/")[-1][:15]
                quantization = key.split("@")[1][:6]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=speeds,
                        mode="lines+markers",
                        name=f"{model_name} ({quantization})",
                        line=dict(color=colors[color_idx % len(colors)]),
                        marker=dict(size=6),
                    )
                )

                color_idx += 1

            fig.update_layout(
                title="Performance trends over time",
                xaxis_title="Datum",
                yaxis_title="Tokens/s",
                hovermode="x unified",
                height=600,
                template="plotly_white",
            )

            return json.dumps(
                {"data": fig.to_dict()["data"], "layout": fig.to_dict()["layout"]}
            )

        except Exception as e:
            logger.debug("Error creating trend chart: %s", e)
            return None

    def _export_results_to_files(self, results_to_export):
        """Exports given results as JSON, CSV, PDF and HTML"""
        if not results_to_export:
            logger.warning("⚠️ No results to export")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        json_file = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(r) for r in results_to_export], f, indent=2, ensure_ascii=False
            )
        logger.info("📄 JSON results saved: %s", json_file)

        csv_file = RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            if results_to_export:
                writer = csv.DictWriter(
                    f, fieldnames=asdict(results_to_export[0]).keys()
                )
                writer.writeheader()
                for result in results_to_export:
                    writer.writerow(asdict(result))
        logger.info("📊 CSV results saved: %s", csv_file)

        self._export_pdf(timestamp, results_to_export)

        if PLOTLY_AVAILABLE:
            self._export_html(timestamp, results_to_export)

    def export_results(self):
        """Legacy wrapper for direct calls (e.g. --export-only)"""
        self._export_results_to_files(self.results)

    def _export_pdf(self, timestamp: str, results_to_export):
        """Exports given benchmark results as PDF report"""
        try:
            results = results_to_export

            pdf_file = RESULTS_DIR / f"benchmark_results_{timestamp}.pdf"

            doc = SimpleDocTemplate(
                str(pdf_file),
                pagesize=landscape(A4),
                rightMargin=0.5 * inch,
                leftMargin=0.5 * inch,
                topMargin=0.5 * inch,
                bottomMargin=0.5 * inch,
                title="LM Studio Benchmark Results",
            )

            elements = []

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "CustomTitle",
                parent=styles["Heading1"],
                fontSize=24,
                textColor=colors.HexColor("#1f4788"),
                spaceAfter=20,
                alignment=1,
            )
            heading_style = ParagraphStyle(
                "CustomHeading",
                parent=styles["Heading2"],
                fontSize=14,
                textColor=colors.HexColor("#2d5aa8"),
                spaceAfter=12,
                spaceBefore=12,
            )

            elements.append(Paragraph("LM Studio Model Benchmark Report", title_style))
            elements.append(Spacer(1, 12))
            timestamp_display = time.strftime("%d.%m.%Y %H:%M:%S")
            elements.append(
                Paragraph(
                    f"<font size=10>Generated: {timestamp_display}</font>",
                    styles["Normal"],
                )
            )
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Benchmark Summary", heading_style))
            vision_count = sum(1 for r in results if r.has_vision)
            tools_count = sum(1 for r in results if r.has_tools)
            avg_size_gb = (
                sum(r.model_size_gb for r in results) / len(results)
                if self.results
                else 0
            )
            avg_tokens_per_sec = (
                sum(r.avg_tokens_per_sec for r in results) / len(results)
                if self.results
                else 0
            )

            summary_data = [
                ["Metric", "Value"],
                ["Models tested", str(len(results))],
                ["Measurements per model", str(self.num_measurement_runs)],
                [
                    "Standard Prompt",
                    self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt,
                ],
                [
                    "Vision Models",
                    f"{vision_count} ({vision_count * 100 // len(results) if self.results else 0}%)",
                ],
                [
                    "Tool-capable Models",
                    f"{tools_count} ({tools_count * 100 // len(results) if self.results else 0}%)",
                ],
                ["Ø Model Size", f"{avg_size_gb:.2f} GB"],
                ["Ø Speed", f"{avg_tokens_per_sec:.2f} tokens/s"],
            ]
            summary_table = Table(summary_data, colWidths=[3 * inch, 3 * inch])
            summary_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5aa8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f0f0f0")],
                        ),
                    ]
                )
            )
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Benchmark Parameters", heading_style))

            params_data = [
                ["Parameter", "Value"],
                ["Measurements per model", f"{self.num_measurement_runs}"],
                ["Context Length", f"{self.context_length} Tokens"],
                ["Temperature", str(OPTIMIZED_INFERENCE_PARAMS["temperature"])],
                ["Top-K Sampling", str(OPTIMIZED_INFERENCE_PARAMS["top_k_sampling"])],
                ["Top-P Sampling", str(OPTIMIZED_INFERENCE_PARAMS["top_p_sampling"])],
                ["Min-P Sampling", str(OPTIMIZED_INFERENCE_PARAMS["min_p_sampling"])],
                ["Repeat Penalty", str(OPTIMIZED_INFERENCE_PARAMS["repeat_penalty"])],
                ["Max Tokens", str(OPTIMIZED_INFERENCE_PARAMS["max_tokens"])],
                ["GPU-Offload Levels", ", ".join(map(str, GPU_OFFLOAD_LEVELS))],
            ]

            if self._gtt_info and self._gtt_info.get("total", 0) > 0:
                gtt_total = self._gtt_info["total"]
                gtt_used = self._gtt_info["used"]
                gtt_status = "Enabled" if self.use_gtt else "Disabled"
                params_data.append(
                    [
                        "GTT (Shared System RAM)",
                        f"{gtt_status} ({gtt_total:.1f}GB total, {gtt_used:.1f}GB used)",
                    ]
                )

            if self.cli_args.get("limit"):
                params_data.append(["Model limit", str(self.cli_args["limit"])])
            if self.cli_args.get("retest"):
                params_data.append(["Cache ignored", "Yes (--retest)"])
            if self.cli_args.get("only_vision"):
                params_data.append(["Filter", "Vision models only"])
            if self.cli_args.get("only_tools"):
                params_data.append(["Filter", "Tool-capable models only"])
            if self.cli_args.get("include_models"):
                params_data.append(
                    ["Include-Pattern", self.cli_args["include_models"][:30]]
                )
            if self.cli_args.get("exclude_models"):
                params_data.append(
                    ["Exclude-Pattern", self.cli_args["exclude_models"][:30]]
                )
            if self.cli_args.get("enable_profiling"):
                params_data.append(["Hardware Profiling", "Yes (--enable-profiling)"])
                if self.cli_args.get("max_temp"):
                    params_data.append(
                        ["Max. Temperature", f"{self.cli_args['max_temp']}°C"]
                    )
                if self.cli_args.get("max_power"):
                    params_data.append(
                        ["Max. Power Draw", f"{self.cli_args['max_power']}W"]
                    )

            params_table = Table(params_data, colWidths=[3 * inch, 3 * inch])
            params_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5aa8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#e8f0ff")),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f0f5ff")],
                        ),
                    ]
                )
            )
            elements.append(params_table)
            elements.append(Spacer(1, 20))

            elements.append(Paragraph("Detailed Results", heading_style))

            if self.rank_by == "speed":
                sorted_results = sorted(
                    results, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )
            elif self.rank_by == "efficiency":
                sorted_results = sorted(
                    results, key=lambda x: x.tokens_per_sec_per_gb, reverse=True
                )
            elif self.rank_by == "ttft":
                sorted_results = sorted(
                    results, key=lambda x: x.avg_ttft, reverse=False
                )
            elif self.rank_by == "vram":

                def get_vram_mb(result):
                    try:
                        return (
                            float(result.vram_mb.split()[0])
                            if isinstance(result.vram_mb, str)
                            else float(result.vram_mb)
                        )
                    except BaseException:
                        return 999999

                sorted_results = sorted(results, key=get_vram_mb, reverse=False)
            else:
                sorted_results = sorted(
                    results, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )

            rank_labels = {
                "speed": "Speed (Tokens/s)",
                "efficiency": "Efficiency (Tokens/s per GB)",
                "ttft": "Time to First Token (ms)",
                "vram": "VRAM Usage (MB)",
            }
            elements.append(
                Paragraph(
                    f"<font size=9>Sorted by: <b>{rank_labels.get(self.rank_by, 'Speed')}</b></font>",
                    styles["Normal"],
                )
            )
            elements.append(Spacer(1, 10))

            table_data = [
                [
                    "Model",
                    "Param",
                    "Arch",
                    "Size(GB)",
                    "Vision",
                    "Tools",
                    "Quant.",
                    "GPU",
                    "GPU Off.",
                    "Tokens/s",
                    "Δ%",
                    "TTFT (ms)",
                    "Gen.Time (s)",
                ]
            ]
            for result in sorted_results:
                vision_icon = "👁" if result.has_vision else ""
                tools_icon = "🔧" if result.has_tools else ""
                delta_str = (
                    f"{result.speed_delta_pct:+.1f}%"
                    if result.speed_delta_pct is not None
                    else "-"
                )
                table_data.append(
                    [
                        result.model_name[:15],
                        result.params_size[:4],
                        result.architecture[:7],
                        f"{result.model_size_gb:.2f}",
                        vision_icon,
                        tools_icon,
                        result.quantization[:6],
                        result.gpu_type[:5],
                        f"{result.gpu_offload:.1f}",
                        f"{result.avg_tokens_per_sec:.2f}",
                        delta_str,
                        f"{result.avg_ttft * 1000:.1f}" if result.avg_ttft else "N/A",
                        f"{result.avg_gen_time:.2f}",
                    ]
                )

            results_table = Table(
                table_data,
                colWidths=[
                    1.2 * inch,
                    0.55 * inch,
                    0.6 * inch,
                    0.65 * inch,
                    0.45 * inch,
                    0.45 * inch,
                    0.6 * inch,
                    0.5 * inch,
                    0.5 * inch,
                    0.65 * inch,
                    0.45 * inch,
                    0.65 * inch,
                    0.7 * inch,
                ],
            )
            results_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5aa8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 9),
                        ("FONTSIZE", (0, 1), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                        ("TOPPADDING", (0, 0), (-1, 0), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f0f0f0")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            elements.append(results_table)
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Best-of-Quantization Analyse", heading_style))

            best_by_model = {}
            for result in results:
                if result.model_name not in best_by_model:
                    best_by_model[result.model_name] = {
                        "best_speed": None,
                        "best_efficiency": None,
                        "best_ttft": None,
                    }

                if (
                    not best_by_model[result.model_name]["best_speed"]
                    or result.avg_tokens_per_sec
                    > best_by_model[result.model_name]["best_speed"].avg_tokens_per_sec
                ):
                    best_by_model[result.model_name]["best_speed"] = result

                if (
                    not best_by_model[result.model_name]["best_efficiency"]
                    or result.tokens_per_sec_per_gb
                    > best_by_model[result.model_name][
                        "best_efficiency"
                    ].tokens_per_sec_per_gb
                ):
                    best_by_model[result.model_name]["best_efficiency"] = result

                if result.avg_ttft > 0:
                    if (
                        not best_by_model[result.model_name]["best_ttft"]
                        or result.avg_ttft
                        < best_by_model[result.model_name]["best_ttft"].avg_ttft
                    ):
                        best_by_model[result.model_name]["best_ttft"] = result

            quant_data = [["Model", "Best Speed", "Best Efficiency", "Best TTFT"]]
            for model_name, analysis in sorted(best_by_model.items()):
                speed_q = (
                    analysis["best_speed"].quantization
                    if analysis["best_speed"]
                    else "-"
                )
                efficiency_q = (
                    analysis["best_efficiency"].quantization
                    if analysis["best_efficiency"]
                    else "-"
                )
                ttft_q = (
                    analysis["best_ttft"].quantization if analysis["best_ttft"] else "-"
                )
                quant_data.append(
                    [model_name[:20], speed_q[:8], efficiency_q[:8], ttft_q[:8]]
                )

            quant_table = Table(
                quant_data, colWidths=[2.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
            )
            quant_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5aa8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("ALIGN", (0, 0), (0, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        ("FONTSIZE", (0, 1), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f0f0f0")],
                        ),
                    ]
                )
            )
            elements.append(quant_table)
            elements.append(Spacer(1, 20))

            model_quants = {}
            for result in results:
                if result.model_name not in model_quants:
                    model_quants[result.model_name] = {}

                quant_level = result.quantization.split("_")[0].lower()
                if quant_level not in model_quants[result.model_name]:
                    model_quants[result.model_name][quant_level] = result
                else:
                    if (
                        result.avg_tokens_per_sec
                        > model_quants[result.model_name][
                            quant_level
                        ].avg_tokens_per_sec
                    ):
                        model_quants[result.model_name][quant_level] = result

            comp_data = {}
            for model, quants in sorted(model_quants.items()):
                comp_data[model] = {"q4": None, "q5": None, "q6": None, "q8": None}
                for q_level in ["q4", "q5", "q6", "q8"]:
                    if q_level in quants:
                        r = quants[q_level]
                        comp_data[model][q_level] = {
                            "speed": round(r.avg_tokens_per_sec, 2),
                            "efficiency": round(r.tokens_per_sec_per_gb, 2),
                            "vram_mb": r.vram_mb,
                            "ttft": round(r.avg_ttft * 1000, 1),
                        }

            if comp_data:
                elements.append(
                    Paragraph(
                        "Quantisierungs-Vergleich (Q4 vs Q5 vs Q6)", heading_style
                    )
                )

                comp_table_data = [
                    ["Model", "Q4 (t/s)", "Q5 (t/s)", "Q6 (t/s)", "Q8 (t/s)"]
                ]
                for model_name, q_variants in sorted(comp_data.items()):
                    row = [model_name[:15]]
                    for q_level in ["q4", "q5", "q6", "q8"]:
                        if q_variants.get(q_level):
                            row.append(f"{q_variants[q_level]['speed']:.2f}")
                        else:
                            row.append("-")
                    comp_table_data.append(row)

                comp_table = Table(
                    comp_table_data,
                    colWidths=[
                        1.5 * inch,
                        1.2 * inch,
                        1.2 * inch,
                        1.2 * inch,
                        1.2 * inch,
                    ],
                )
                comp_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d7aa8")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("ALIGN", (0, 0), (0, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 9),
                            ("FONTSIZE", (0, 1), (-1, -1), 7),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#f0f0f0")],
                            ),
                        ]
                    )
                )
                elements.append(comp_table)
                elements.append(Spacer(1, 20))

            elements.append(Paragraph("Performance Statistics", heading_style))
            max_tps_result = max(results, key=lambda x: x.avg_tokens_per_sec)
            min_tps_result = min(results, key=lambda x: x.avg_tokens_per_sec)
            avg_tps = sum(r.avg_tokens_per_sec for r in results) / len(results)
            speeds = sorted(
                [r.avg_tokens_per_sec for r in results if r.avg_tokens_per_sec > 0]
            )
            percentile_stats = {}
            if len(speeds) >= 3:
                percentile_stats["speed"] = {
                    "median": speeds[len(speeds) // 2],
                    "p95": speeds[int(len(speeds) * 0.95)],
                    "p99": (
                        speeds[int(len(speeds) * 0.99)]
                        if len(speeds) > 100
                        else speeds[-1]
                    ),
                }

            stats_data = [
                ["Statistic", "Value"],
                [
                    "Fastest Model",
                    f"{max_tps_result.model_name} ({max_tps_result.avg_tokens_per_sec:.2f} tokens/s)",
                ],
                [
                    "Slowest Model",
                    f"{min_tps_result.model_name} ({min_tps_result.avg_tokens_per_sec:.2f} tokens/s)",
                ],
                ["Average Tokens/s", f"{avg_tps:.2f}"],
            ]

            if "speed" in percentile_stats:
                speed_p = percentile_stats["speed"]
                stats_data.extend(
                    [
                        ["Tokens/s - Median", f"{speed_p.get('median', '-'):.2f}"],
                        ["Tokens/s - P95", f"{speed_p.get('p95', '-'):.2f}"],
                        ["Tokens/s - P99", f"{speed_p.get('p99', '-'):.2f}"],
                    ]
                )

            stats_table = Table(stats_data, colWidths=[3 * inch, 3 * inch])
            stats_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5aa8")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 11),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                        ("BACKGROUND", (0, 1), (-1, -1), colors.lightgreen),
                        ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                    ]
                )
            )
            elements.append(stats_table)
            elements.append(Spacer(1, 20))
            elements.append(
                Paragraph("💡 Best Practice Recommendations", heading_style)
            )
            recommendations = self._generate_best_practices()

            if recommendations:
                rec_text = "<br/>".join(recommendations)
                rec_style = ParagraphStyle(
                    "Recommendations",
                    parent=styles["Normal"],
                    fontSize=9,
                    leading=12,
                    leftIndent=10,
                    fontName="Courier",
                )
                elements.append(Paragraph(rec_text, rec_style))
                elements.append(Spacer(1, 20))

            vision_models = [r for r in results if r.has_vision]
            if vision_models:
                elements.append(PageBreak())
                elements.append(
                    Paragraph("👁️  Vision Models (Multimodal)", title_style)
                )
                elements.append(Spacer(1, 12))
                elements.append(
                    Paragraph(
                        f"<font size=10>{len(vision_models)} "
                        f"Vision-capable models found</font>",
                        styles["Normal"],
                    )
                )
                elements.append(Spacer(1, 15))

                vision_sorted = sorted(
                    vision_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )

                vision_data = [
                    [
                        "Model",
                        "Param",
                        "Size(GB)",
                        "Quant.",
                        "Tokens/s",
                        "TTFT (ms)",
                        "Effizienz",
                    ]
                ]
                for r in vision_sorted:
                    vision_data.append(
                        [
                            r.model_name[:25],
                            r.params_size[:6],
                            f"{r.model_size_gb:.2f}",
                            r.quantization[:8],
                            f"{r.avg_tokens_per_sec:.2f}",
                            f"{r.avg_ttft * 1000:.1f}" if r.avg_ttft else "N/A",
                            f"{r.tokens_per_sec_per_gb:.2f}",
                        ]
                    )

                vision_table = Table(
                    vision_data,
                    colWidths=[
                        2 * inch,
                        0.7 * inch,
                        0.8 * inch,
                        0.9 * inch,
                        0.9 * inch,
                        0.9 * inch,
                        0.9 * inch,
                    ],
                )
                vision_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("ALIGN", (0, 0), (0, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("FONTSIZE", (0, 1), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#e8f4ff")],
                            ),
                        ]
                    )
                )
                elements.append(vision_table)
                elements.append(Spacer(1, 15))
                elements.append(Paragraph("Top 3 Vision Models", heading_style))
                top3_text = []
                for i, r in enumerate(vision_sorted[:3], 1):
                    top3_text.append(f"{i}. <b>{r.model_name}</b> ({r.quantization})")
                    top3_text.append(
                        f"   → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB"
                    )
                    top3_text.append("")

                elements.append(Paragraph("<br/>".join(top3_text), styles["Normal"]))

            tool_models = [r for r in results if r.has_tools]
            if tool_models:
                elements.append(PageBreak())
                elements.append(Paragraph("🔧 Tool-Calling Models", title_style))
                elements.append(Spacer(1, 12))
                elements.append(
                    Paragraph(
                        f"<font size=10>{len(tool_models)} "
                        f"Tool-capable models found</font>",
                        styles["Normal"],
                    )
                )
                elements.append(Spacer(1, 15))

                tool_sorted = sorted(
                    tool_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )

                tool_data = [
                    [
                        "Model",
                        "Param",
                        "Size(GB)",
                        "Quant.",
                        "Tokens/s",
                        "TTFT (ms)",
                        "Effizienz",
                    ]
                ]
                for r in tool_sorted:
                    tool_data.append(
                        [
                            r.model_name[:25],
                            r.params_size[:6],
                            f"{r.model_size_gb:.2f}",
                            r.quantization[:8],
                            f"{r.avg_tokens_per_sec:.2f}",
                            f"{r.avg_ttft * 1000:.1f}" if r.avg_ttft else "N/A",
                            f"{r.tokens_per_sec_per_gb:.2f}",
                        ]
                    )

                tool_table = Table(
                    tool_data,
                    colWidths=[
                        2 * inch,
                        0.7 * inch,
                        0.8 * inch,
                        0.9 * inch,
                        0.9 * inch,
                        0.9 * inch,
                        0.9 * inch,
                    ],
                )
                tool_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e27a4a")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("ALIGN", (0, 0), (0, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("FONTSIZE", (0, 1), (-1, -1), 8),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#fff4e8")],
                            ),
                        ]
                    )
                )
                elements.append(tool_table)
                elements.append(Spacer(1, 15))
                elements.append(Paragraph("Top 3 Tool-Calling Models", heading_style))
                top3_text = []
                for i, r in enumerate(tool_sorted[:3], 1):
                    top3_text.append(f"{i}. <b>{r.model_name}</b> ({r.quantization})")
                    top3_text.append(
                        f"   → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB"
                    )
                    top3_text.append("")

                elements.append(Paragraph("<br/>".join(top3_text), styles["Normal"]))

            by_arch = {}
            for r in results:
                arch = r.architecture
                if arch not in by_arch:
                    by_arch[arch] = []
                by_arch[arch].append(r)

            major_archs = {k: v for k, v in by_arch.items() if len(v) >= 2}

            if major_archs:
                elements.append(PageBreak())
                elements.append(Paragraph("🏗️  Models by Architecture", title_style))
                elements.append(Spacer(1, 12))

                for arch_name, arch_models in sorted(
                    major_archs.items(), key=lambda x: -len(x[1])
                ):
                    arch_sorted = sorted(
                        arch_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                    )

                    elements.append(
                        Paragraph(
                            f"<b>{arch_name.upper()}</b> ({len(arch_models)} models)",
                            heading_style,
                        )
                    )
                    elements.append(Spacer(1, 8))

                    arch_data = [["Model", "Param", "Quant.", "Tokens/s", "Size(GB)"]]
                    for r in arch_sorted[:5]:
                        arch_data.append(
                            [
                                r.model_name[:30],
                                r.params_size[:6],
                                r.quantization[:8],
                                f"{r.avg_tokens_per_sec:.2f}",
                                f"{r.model_size_gb:.2f}",
                            ]
                        )

                    arch_table = Table(
                        arch_data,
                        colWidths=[
                            2.5 * inch,
                            0.7 * inch,
                            0.9 * inch,
                            0.9 * inch,
                            0.8 * inch,
                        ],
                    )
                    arch_table.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#6a4ae2"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 9),
                                ("FONTSIZE", (0, 1), (-1, -1), 8),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                                (
                                    "ROWBACKGROUNDS",
                                    (0, 1),
                                    (-1, -1),
                                    [colors.white, colors.HexColor("#f0ecff")],
                                ),
                            ]
                        )
                    )
                    elements.append(arch_table)
                    elements.append(Spacer(1, 15))

            if self.enable_profiling and any(r.temp_celsius_avg for r in results):
                elements.append(PageBreak())
                elements.append(Paragraph("🌡️ Hardware-Profiling Report", title_style))
                elements.append(Spacer(1, 12))
                elements.append(
                    Paragraph("Temperature and Power Analysis", heading_style)
                )

                temps_avg = [r.temp_celsius_avg for r in results if r.temp_celsius_avg]
                powers_avg = [r.power_watts_avg for r in results if r.power_watts_avg]

                profile_summary = [
                    ["Metrik", "Min", "Max", "Durchschnitt"],
                ]

                if temps_avg:
                    profile_summary.append(
                        [
                            "GPU Temperatur (°C)",
                            f"{min(temps_avg):.1f}",
                            f"{max(temps_avg):.1f}",
                            f"{mean(temps_avg):.1f}",
                        ]
                    )

                if powers_avg:
                    profile_summary.append(
                        [
                            "GPU Power-Draw (W)",
                            f"{min(powers_avg):.1f}",
                            f"{max(powers_avg):.1f}",
                            f"{mean(powers_avg):.1f}",
                        ]
                    )

                profile_table = Table(
                    profile_summary,
                    colWidths=[2 * inch, 1.2 * inch, 1.2 * inch, 1.5 * inch],
                )
                profile_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9534f")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("FONTSIZE", (0, 0), (-1, 0), 10),
                            ("FONTSIZE", (0, 1), (-1, -1), 9),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [colors.white, colors.HexColor("#fff0ed")],
                            ),
                        ]
                    )
                )
                elements.append(profile_table)
                elements.append(Spacer(1, 15))
                elements.append(Paragraph("Profiling per Model", heading_style))

                profile_data = [
                    [
                        "Model",
                        "Quant.",
                        "Temp Min",
                        "Temp Max",
                        "Temp Ø (°C)",
                        "Power Min",
                        "Power Max",
                        "Power Ø (W)",
                    ]
                ]
                for r in sorted(
                    results, key=lambda x: x.temp_celsius_avg or 0, reverse=True
                ):
                    if r.temp_celsius_avg or r.power_watts_avg:
                        temp_min = (
                            f"{r.temp_celsius_min:.1f}"
                            if r.temp_celsius_min
                            else "-"
                        )
                        temp_max = (
                            f"{r.temp_celsius_max:.1f}"
                            if r.temp_celsius_max
                            else "-"
                        )
                        temp_avg = (
                            f"{r.temp_celsius_avg:.1f}"
                            if r.temp_celsius_avg
                            else "-"
                        )
                        power_min = (
                            f"{r.power_watts_min:.1f}"
                            if r.power_watts_min
                            else "-"
                        )
                        power_max = (
                            f"{r.power_watts_max:.1f}"
                            if r.power_watts_max
                            else "-"
                        )
                        power_avg = (
                            f"{r.power_watts_avg:.1f}"
                            if r.power_watts_avg
                            else "-"
                        )

                        profile_data.append(
                            [
                                r.model_name[:20],
                                r.quantization[:6],
                                temp_min,
                                temp_max,
                                temp_avg,
                                power_min,
                                power_max,
                                power_avg,
                            ]
                        )

                if len(profile_data) > 1:
                    prof_table = Table(
                        profile_data,
                        colWidths=[
                            1.5 * inch,
                            0.6 * inch,
                            0.7 * inch,
                            0.7 * inch,
                            0.8 * inch,
                            0.7 * inch,
                            0.7 * inch,
                            0.75 * inch,
                        ],
                    )
                    prof_table.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, 0),
                                    colors.HexColor("#d9534f"),
                                ),
                                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                                ("FONTSIZE", (0, 0), (-1, 0), 8),
                                ("FONTSIZE", (0, 1), (-1, -1), 7),
                                ("GRID", (0, 0), (-1, -1), 1, colors.black),
                                (
                                    "ROWBACKGROUNDS",
                                    (0, 1),
                                    (-1, -1),
                                    [colors.white, colors.HexColor("#ffe5e5")],
                                ),
                            ]
                        )
                    )
                    elements.append(prof_table)

            doc.build(elements)
            logger.info("📑 PDF results saved: %s", pdf_file)

        except Exception as e:
            logger.error("❌ Error creating PDF: %s", e)

    def _export_html(self, timestamp: str, results_to_export):
        """Exports given benchmark results as interactive HTML report with Plotly charts"""
        if not PLOTLY_AVAILABLE or go is None:
            logger.warning("⚠️ Plotly not available, skipping HTML export")
            return

        try:
            results = results_to_export

            html_file = RESULTS_DIR / f"benchmark_results_{timestamp}.html"

            template_path = Path(__file__).parent / "report_template.html.template"
            with open(template_path, "r", encoding="utf-8") as f:
                html_template = f.read()

            sorted_results = self.sort_results("speed")

            top_10 = sorted_results[:10]
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=[f"{r.model_name[:20]}\n{r.quantization}" for r in top_10],
                        y=[r.avg_tokens_per_sec for r in top_10],
                        text=[f"{r.avg_tokens_per_sec:.2f}" for r in top_10],
                        textposition="auto",
                        marker_color="#2d5aa8",
                    )
                ]
            )
            fig_bar.update_layout(
                title="Top 10 fastest models",
                xaxis_title="Model + Quantization",
                yaxis_title="Tokens/s",
                hovermode="x unified",
                height=500,
            )

            fig_scatter = go.Figure(
                data=[
                    go.Scatter(
                        x=[r.model_size_gb for r in results],
                        y=[r.avg_tokens_per_sec for r in results],
                        mode="markers",
                        text=[
                            f"{r.model_name}<br>{r.quantization}<br>{r.avg_tokens_per_sec:.2f} t/s"
                            for r in results
                        ],
                        marker=dict(
                            size=[r.avg_tokens_per_sec / 2 for r in results],
                            color=[r.tokens_per_sec_per_gb for r in results],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Effizienz<br>(t/s pro GB)"),
                        ),
                        hovertemplate="<b>%{text}</b><extra></extra>",
                    )
                ]
            )
            fig_scatter.update_layout(
                title="Model Size vs Performance (Bubble size = Speed)",
                xaxis_title="Model Size (GB)",
                yaxis_title="Tokens/s",
                height=500,
                hovermode="closest",
            )

            fig_efficiency = go.Figure(
                data=[
                    go.Scatter(
                        x=[r.tokens_per_sec_per_gb for r in results],
                        y=[r.tokens_per_sec_per_billion_params for r in results],
                        mode="markers",
                        text=[f"{r.model_name} ({r.quantization})" for r in results],
                        marker=dict(
                            size=8,
                            color=[r.avg_tokens_per_sec for r in results],
                            colorscale="RdYlGn",
                            showscale=True,
                            colorbar=dict(title="Speed<br>(tokens/s)"),
                        ),
                        hovertemplate="<b>%{text}</b><br>Per GB: %{x:.2f}<br>Per Billion Params: %{y:.2f}<extra></extra>",
                    )
                ]
            )
            fig_efficiency.update_layout(
                title="Effizienz-Analyse: Tokens/s pro GB vs Tokens/s pro Milliarde Parameter",
                xaxis_title="Tokens/s pro GB",
                yaxis_title="Tokens/s pro Milliarde Parameter",
                height=500,
            )

            vision_count = sum(1 for r in results if r.has_vision)
            tools_count = sum(1 for r in results if r.has_tools)
            avg_size_gb = (
                sum(r.model_size_gb for r in results) / len(results)
                if self.results
                else 0
            )
            avg_tokens_per_sec = (
                sum(r.avg_tokens_per_sec for r in results) / len(results)
                if self.results
                else 0
            )

            summary_stats = {
                "Number of models": len(results),
                "Measurements per Model": self.num_measurement_runs,
                "Standard Prompt": (
                    self.prompt[:50] + "..." if len(self.prompt) > 50 else self.prompt
                ),
                "Fastest": f"{sorted_results[0].model_name[:20]} ({sorted_results[0].avg_tokens_per_sec:.2f} t/s)",
                "Slowest": f"{sorted_results[-1].model_name[:20]} ({sorted_results[-1].avg_tokens_per_sec:.2f} t/s)",
                "Ø Speed": f"{avg_tokens_per_sec:.2f} t/s",
                "Vision Models": f"{vision_count} ({vision_count * 100 // len(results) if self.results else 0}%)",
                "Tool-capable Models": f"{tools_count} ({tools_count * 100 // len(results) if self.results else 0}%)",
                "Ø Model Size": f"{avg_size_gb:.2f} GB",
            }

            summary_boxes = ""
            colors_list = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#00f2fe"]
            for i, (label, value) in enumerate(summary_stats.items()):
                color = colors_list[i % len(colors_list)]
                summary_boxes += f"""
            <div class="summary-box" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
            </div>
"""

            benchmark_params = []
            benchmark_params.append(
                f"<strong>Context Length:</strong> {self.context_length} Tokens"
            )
            benchmark_params.append(
                f"<strong>Temperature:</strong> {OPTIMIZED_INFERENCE_PARAMS['temperature']}"
            )
            benchmark_params.append(
                f"<strong>Top-K Sampling:</strong> {OPTIMIZED_INFERENCE_PARAMS['top_k_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Top-P Sampling:</strong> {OPTIMIZED_INFERENCE_PARAMS['top_p_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Min-P Sampling:</strong> {OPTIMIZED_INFERENCE_PARAMS['min_p_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Repeat Penalty:</strong> {OPTIMIZED_INFERENCE_PARAMS['repeat_penalty']}"
            )
            benchmark_params.append(
                f"<strong>Max Tokens:</strong> {OPTIMIZED_INFERENCE_PARAMS['max_tokens']}"
            )
            benchmark_params.append(
                f"<strong>GPU-Offload Levels:</strong> {', '.join(map(str, GPU_OFFLOAD_LEVELS))}"
            )

            cli_params = []
            cli_params.append(
                f"<strong>Measurements per Model:</strong> {self.cli_args['runs']}"
            )

            if self._gtt_info and self._gtt_info.get("total", 0) > 0:
                gtt_total = self._gtt_info["total"]
                gtt_used = self._gtt_info["used"]
                gtt_status = "✅ Enabled" if self.use_gtt else "❌ Disabled"
                cli_params.append(
                    f"<strong>GTT (Shared System RAM):</strong> {gtt_status} "
                    f"({gtt_total:.1f}GB total, {gtt_used:.1f}GB used)"
                )

            if self.cli_args.get("limit"):
                cli_params.append(
                    f"<strong>Model Limit:</strong> {self.cli_args['limit']}"
                )
            if self.cli_args.get("retest"):
                cli_params.append(f"<strong>Cache:</strong> Ignored (--retest)")
            if self.cli_args.get("only_vision"):
                cli_params.append(f"<strong>Filter:</strong> Vision models only")
            if self.cli_args.get("only_tools"):
                cli_params.append(f"<strong>Filter:</strong> Tool-capable models only")
            if self.cli_args.get("include_models"):
                cli_params.append(
                    f"<strong>Include-Pattern:</strong> {self.cli_args['include_models'][:40]}"
                )
            if self.cli_args.get("exclude_models"):
                cli_params.append(
                    f"<strong>Exclude-Pattern:</strong> {self.cli_args['exclude_models'][:40]}"
                )
            if self.cli_args.get("enable_profiling"):
                cli_params.append(
                    f"<strong>Hardware Profiling:</strong> Yes (--enable-profiling)"
                )
                if self.cli_args.get("max_temp"):
                    cli_params.append(
                        f"<strong>Max. Temperature:</strong> {self.cli_args['max_temp']}°C"
                    )
                if self.cli_args.get("max_power"):
                    cli_params.append(
                        f"<strong>Max. Power Draw:</strong> {self.cli_args['max_power']}W"
                    )

            cli_section = f"""
            <h2>⚙️ Benchmark Parameters</h2>
            <h3>Inference Parameters</h3>
            <p style="line-height: 1.8; color: var(--text-secondary); font-size: 14px;">
                {" | ".join(benchmark_params)}
            </p>
            <h3>CLI Arguments</h3>
            <p style="line-height: 1.8; color: var(--text-secondary); font-size: 14px;">
                {" | ".join(cli_params) if cli_params else "No additional arguments"}
            </p>
"""

            trend_json = self.generate_trend_chart()
            if trend_json:
                trend_section = """
        <h2>📈 Performance Trends Over Time</h2>
        <div class="chart" id="trend-chart"></div>
"""
                trend_data = json.loads(trend_json)
                trend_script = (
                    "        Plotly.newPlot('trend-chart', "
                    f"{json.dumps(trend_data['data'])}, "
                    f"{json.dumps(trend_data['layout'])});"
                )
            else:
                trend_section = ""
                trend_script = ""

            recommendations = self._generate_best_practices()
            best_practices_html = (
                "\n".join(recommendations)
                if recommendations
                else "No recommendations available"
            )

            vision_models = [r for r in results if r.has_vision]
            if vision_models:
                vision_sorted = sorted(
                    vision_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )
                vision_rows = ""
                for r in vision_sorted:
                    vision_rows += f"""
                    <tr>
                        <td>{r.model_name}</td>
                        <td>{r.params_size}</td>
                        <td>{r.model_size_gb:.2f} GB</td>
                        <td>{r.quantization}</td>
                        <td><strong>{r.avg_tokens_per_sec:.2f}</strong></td>
                        <td>{r.avg_ttft * 1000:.1f} ms</td>
                        <td>{r.tokens_per_sec_per_gb:.2f}</td>
                    </tr>"""

                vision_section = f"""
        <h2>👁️ Vision Models (Multimodal)</h2>
        <p>{len(vision_models)} Vision-capable models found</p>
        <table class="category-table">
            <thead class="vision">
                <tr>
                    <th>Model</th>
                    <th>Parameter</th>
                    <th>Size</th>
                    <th>Quantization</th>
                    <th>Tokens/s</th>
                    <th>TTFT</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody>
                {vision_rows}
            </tbody>
        </table>
        <h3>Top 3 Vision Models:</h3>
        <ul>"""
                for i, r in enumerate(vision_sorted[:3], 1):
                    vision_section += f"""
            <li><strong>{r.model_name}</strong> ({r.quantization}) → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB</li>"""
                vision_section += """
        </ul>"""
            else:
                vision_section = ""

            tool_models = [r for r in results if r.has_tools]
            if tool_models:
                tool_sorted = sorted(
                    tool_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                )
                tool_rows = ""
                for r in tool_sorted:
                    tool_rows += f"""
                    <tr>
                        <td>{r.model_name}</td>
                        <td>{r.params_size}</td>
                        <td>{r.model_size_gb:.2f} GB</td>
                        <td>{r.quantization}</td>
                        <td><strong>{r.avg_tokens_per_sec:.2f}</strong></td>
                        <td>{r.avg_ttft * 1000:.1f} ms</td>
                        <td>{r.tokens_per_sec_per_gb:.2f}</td>
                    </tr>"""

                tools_section = f"""
        <h2>🔧 Tool-Calling Models</h2>
        <p>{len(tool_models)} Tool-capable models found</p>
        <table class="category-table">
            <thead class="tools">
                <tr>
                    <th>Model</th>
                    <th>Parameter</th>
                    <th>Size</th>
                    <th>Quantization</th>
                    <th>Tokens/s</th>
                    <th>TTFT</th>
                    <th>Efficiency</th>
                </tr>
            </thead>
            <tbody>
                {tool_rows}
            </tbody>
        </table>
        <h3>Top 3 Tool-Calling Models:</h3>
        <ul>"""
                for i, r in enumerate(tool_sorted[:3], 1):
                    tools_section += f"""
            <li><strong>{r.model_name}</strong> ({r.quantization}) → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB</li>"""
                tools_section += """
        </ul>"""
            else:
                tools_section = ""

            by_arch = {}
            for r in results:
                arch = r.architecture
                if arch not in by_arch:
                    by_arch[arch] = []
                by_arch[arch].append(r)

            major_archs = {k: v for k, v in by_arch.items() if len(v) >= 2}

            if major_archs:
                arch_section = """
        <h2>🏗️ Models by Architecture</h2>"""

                for arch_name, arch_models in sorted(
                    major_archs.items(), key=lambda x: -len(x[1])
                ):
                    arch_sorted = sorted(
                        arch_models, key=lambda x: x.avg_tokens_per_sec, reverse=True
                    )
                    arch_rows = ""
                    for r in arch_sorted[:5]:
                        arch_rows += f"""
                        <tr>
                            <td>{r.model_name}</td>
                            <td>{r.params_size}</td>
                            <td>{r.quantization}</td>
                            <td><strong>{r.avg_tokens_per_sec:.2f}</strong></td>
                            <td>{r.model_size_gb:.2f} GB</td>
                        </tr>"""

                    arch_section += f"""
        <h3>{arch_name.upper()} ({len(arch_models)} models)</h3>
        <table class="category-table">
            <thead class="arch">
                <tr>
                    <th>Model</th>
                    <th>Parameter</th>
                    <th>Quantization</th>
                    <th>Tokens/s</th>
                    <th>Size</th>
                </tr>
            </thead>
            <tbody>
                {arch_rows}
            </tbody>
        </table>"""
            else:
                arch_section = ""

            profiling_section = ""
            if self.enable_profiling and any(r.temp_celsius_avg for r in results):
                temps_avg = [r.temp_celsius_avg for r in results if r.temp_celsius_avg]
                powers_avg = [r.power_watts_avg for r in results if r.power_watts_avg]

                profile_rows = ""
                for r in sorted(
                    results, key=lambda x: x.temp_celsius_avg or 0, reverse=True
                ):
                    if r.temp_celsius_avg or r.power_watts_avg:
                        temp_min = (
                            f"{r.temp_celsius_min:.1f}°C"
                            if r.temp_celsius_min
                            else "-"
                        )
                        temp_max = (
                            f"{r.temp_celsius_max:.1f}°C"
                            if r.temp_celsius_max
                            else "-"
                        )
                        temp_avg = (
                            f"{r.temp_celsius_avg:.1f}°C"
                            if r.temp_celsius_avg
                            else "-"
                        )
                        power_min = (
                            f"{r.power_watts_min:.1f}W"
                            if r.power_watts_min
                            else "-"
                        )
                        power_max = (
                            f"{r.power_watts_max:.1f}W"
                            if r.power_watts_max
                            else "-"
                        )
                        power_avg = (
                            f"{r.power_watts_avg:.1f}W"
                            if r.power_watts_avg
                            else "-"
                        )

                        profile_rows += f"""
                <tr>
                    <td>{r.model_name}</td>
                    <td>{r.quantization}</td>
                    <td>{temp_min}</td>
                    <td>{temp_max}</td>
                    <td><strong>{temp_avg}</strong></td>
                    <td>{power_min}</td>
                    <td>{power_max}</td>
                    <td><strong>{power_avg}</strong></td>
                </tr>"""

                profile_summary = ""
                if temps_avg:
                    profile_summary += f"""
                <div class="summary-box" style="border-left: 4px solid #d9534f;">
                    <h4>🌡️ GPU Temperatur</h4>
                    <p>Min: {min(temps_avg):.1f}°C | Max: {max(temps_avg):.1f}°C | Ø: {mean(temps_avg):.1f}°C</p>
                </div>"""

                if powers_avg:
                    profile_summary += f"""
                <div class="summary-box" style="border-left: 4px solid #ff9800;">
                    <h4>⚡ Power-Draw</h4>
                    <p>Min: {min(powers_avg):.1f}W | Max: {max(powers_avg):.1f}W | Ø: {mean(powers_avg):.1f}W</p>
                </div>"""

                profiling_section = f"""
            <h2>🌡️ Hardware-Profiling</h2>
            <div class="profiling-summary">
                {profile_summary}
            </div>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Quant.</th>
                        <th>Temp Min</th>
                        <th>Temp Max</th>
                        <th>Temp Ø</th>
                        <th>Power Min</th>
                        <th>Power Max</th>
                        <th>Power Ø</th>
                    </tr>
                </thead>
                <tbody>
                    {profile_rows}
                </tbody>
            </table>"""

            html_output = html_template.replace("{{SUMMARY_BOXES}}", summary_boxes)
            html_output = html_output.replace("{{CLI_SECTION}}", cli_section)
            html_output = html_output.replace("{{TREND_SECTION}}", trend_section)
            html_output = html_output.replace("{{BEST_PRACTICES}}", best_practices_html)
            html_output = html_output.replace("{{VISION_SECTION}}", vision_section)
            html_output = html_output.replace("{{TOOLS_SECTION}}", tools_section)
            html_output = html_output.replace("{{ARCH_SECTION}}", arch_section)
            html_output = html_output.replace(
                "{{PROFILING_SECTION}}", profiling_section
            )
            html_output = html_output.replace(
                "{{TIMESTAMP}}", time.strftime("%d.%m.%Y %H:%M:%S")
            )
            html_output = html_output.replace(
                "{{BAR_DATA}}", json.dumps(fig_bar.to_dict()["data"])
            )
            html_output = html_output.replace(
                "{{BAR_LAYOUT}}", json.dumps(fig_bar.to_dict()["layout"])
            )
            html_output = html_output.replace(
                "{{SCATTER_DATA}}", json.dumps(fig_scatter.to_dict()["data"])
            )
            html_output = html_output.replace(
                "{{SCATTER_LAYOUT}}", json.dumps(fig_scatter.to_dict()["layout"])
            )
            html_output = html_output.replace(
                "{{EFFICIENCY_DATA}}", json.dumps(fig_efficiency.to_dict()["data"])
            )
            html_output = html_output.replace(
                "{{EFFICIENCY_LAYOUT}}", json.dumps(fig_efficiency.to_dict()["layout"])
            )
            html_output = html_output.replace("{{TREND_SCRIPT}}", trend_script)

            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_output)

            logger.info("🌐 HTML results saved: %s", html_file)

        except Exception as e:
            logger.error("❌ Error creating HTML: %s", e)


def main():
    """Main function with CLI arguments"""
    global log_filename

    def _expand_short_flag_clusters(cli_args: list[str]) -> list[str]:
        """Expand combined short flags like ``-rd`` to ``-r -d``.

        This only expands flags that do not consume a value directly.
        """
        combinable = {"d", "h"}
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

    current_process = psutil.Process()
    parent_process = current_process.parent()

    is_webapp_subprocess = False
    if parent_process and "python" in parent_process.name().lower():
        try:
            parent_cmdline = " ".join(parent_process.cmdline())
            if "web/app.py" in parent_cmdline:
                is_webapp_subprocess = True
        except BaseException:
            pass

    log_filename = (
        LOGS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    if not is_webapp_subprocess:
        file_handler = logging.FileHandler(log_filename, mode="w")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.addFilter(NoJSONFilter())
        logging.root.addHandler(file_handler)
        logger.info("📝 Benchmark-Log: %s", format_path_for_logs(log_filename))
    else:
        logger.info("📝 Benchmark running as WebApp subprocess - Logging via WebApp")

    latest_link = LOGS_DIR / "benchmark_latest.log"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(log_filename.name)

    normalized_args = _expand_short_flag_clusters(sys.argv[1:])

    preset_manager = PresetManager()
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets and exit",
    )
    preset_parser.add_argument(
        "-p",
        "--preset",
        type=str,
        default="default",
        help="Load benchmark preset by name (default: default)",
    )
    preset_args, remaining_args = preset_parser.parse_known_args(normalized_args)

    if preset_args.list_presets:
        print("\nAvailable presets:\n")
        for preset_name, readonly in preset_manager.list_presets_detailed():
            icon = "[readonly]" if readonly else "[user]"
            print(f"  {icon:<11} {preset_name}")
        print("\nUse: python benchmark.py -p <preset_name>\n")
        return

    try:
        loaded_preset = preset_manager.load_preset(preset_args.preset)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("❌ %s", exc)
        logger.info("Use --list-presets to see all available presets")
        sys.exit(1)

    preset_cli_args = preset_manager.preset_to_cli_args(loaded_preset)
    final_cli_args = preset_cli_args + remaining_args

    parser = argparse.ArgumentParser(
        description="LM Studio Model Benchmark - Tests all locally installed LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                                   # Standard: all models, 3 measurements
  python benchmark.py --list-presets                    # Show available presets and exit
  python benchmark.py -p quick_test                     # Load preset before parsing other args
  python benchmark.py --runs 1                          # Fast: all models, 1 measurement
  python benchmark.py --limit 3 --runs 1                # Test 3 models with 1 measurement
  python benchmark.py --limit 1 --runs 1                # Test 1 model with 1 measurement
  python benchmark.py --runs 2 --context 4096           # 2 measurements, 4096 token context
  python benchmark.py --limit 5 --runs 2 --context 4096 # Test 5 models, more options
        """,
    )

    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=NUM_MEASUREMENT_RUNS,
        help=f"Number of measurements per model quantization (default: {NUM_MEASUREMENT_RUNS})",
    )

    parser.add_argument(
        "--context",
        "-c",
        type=int,
        default=CONTEXT_LENGTH,
        help=f"Context length in tokens (default: {CONTEXT_LENGTH})",
    )

    parser.add_argument(
        "--prompt",
        "-P",
        type=str,
        default=STANDARD_PROMPT,
        help=f'Standard test prompt (default: "{STANDARD_PROMPT}")',
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Maximum number of models to test (e.g. 3 tests only the first 3 models)",
    )

    parser.add_argument(
        "--only-vision",
        action="store_true",
        help="Only test models with vision capability (multimodal)",
    )

    parser.add_argument(
        "--only-tools",
        action="store_true",
        help="Only test models with tool-calling support",
    )

    parser.add_argument(
        "--quants",
        type=str,
        default=None,
        help='Only test specific quantizations (e.g. "q4,q5,q6")',
    )

    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help='Only test specific architectures (e.g. "llama,mistral,gemma")',
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help='Only test specific parameter sizes (e.g. "3B,7B,8B")',
    )

    parser.add_argument(
        "--min-context",
        type=int,
        default=None,
        help="Minimum context length in tokens (e.g. 32000)",
    )

    parser.add_argument(
        "--max-size",
        type=float,
        default=None,
        help="Maximum model size in GB (e.g. 10.0)",
    )

    parser.add_argument(
        "--include-models",
        type=str,
        default=None,
        help='Only models matching the regex pattern (e.g. "llama.*7b" or "qwen|phi")',
    )

    parser.add_argument(
        "--exclude-models",
        type=str,
        default=None,
        help='Exclude models matching the regex pattern (e.g. ".*uncensored.*" or "test|experimental")',
    )

    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help='Compare with previous results (e.g. "20260104_172200.json" or "latest")',
    )

    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["speed", "efficiency", "ttft", "vram"],
        default="speed",
        help="Sort results by: speed (tokens/s), efficiency (tokens/s per GB), ttft (Time to First Token), vram (VRAM usage)",
    )

    parser.add_argument(
        "--retest",
        action="store_true",
        help="Ignore cache and retest all models (overwrites old results in database)",
    )

    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Development mode: Automatically tests the smallest available model with -l 1 -r 1",
    )

    parser.add_argument(
        "--list-cache", action="store_true", help="Show all cached models and exit"
    )

    parser.add_argument(
        "--export-cache",
        type=str,
        default=None,
        metavar="FILE",
        help='Export cache contents as JSON (e.g. "cache_export.json") and exit',
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable hardware profiling: Measures temperature and power draw during benchmark",
    )

    parser.add_argument(
        "--max-temp",
        type=float,
        default=None,
        help="Maximum GPU temperature in °C (warning if exceeded, e.g. 80.0)",
    )

    parser.add_argument(
        "--max-power",
        type=float,
        default=None,
        help="Maximum GPU power draw in watts (warning if exceeded, e.g. 400.0)",
    )

    parser.add_argument(
        "--disable-gtt",
        action="store_true",
        help="Disable GTT (Shared System RAM) on AMD GPUs - use only dedicated VRAM",
    )

    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Generate reports (JSON/CSV/PDF/HTML) from all results in database without running new tests",
    )

    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable DEBUG logging for detailed output",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override: Sampling temperature (e.g. 0.7)",
    )
    parser.add_argument(
        "--top-k",
        "--top-k-sampling",
        dest="top_k_sampling",
        type=int,
        default=None,
        help="Override: Top-K sampling (e.g. 50)",
    )
    parser.add_argument(
        "--top-p",
        "--top-p-sampling",
        dest="top_p_sampling",
        type=float,
        default=None,
        help="Override: Top-P (nucleus sampling, e.g. 0.95)",
    )
    parser.add_argument(
        "--min-p",
        "--min-p-sampling",
        dest="min_p_sampling",
        type=float,
        default=None,
        help="Override: Min-P (Minimum probability threshold, z.B. 0.05)",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=None,
        help="Override: Repeat penalty (e.g. 1.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override: Max output tokens (e.g. 256)",
    )

    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=DEFAULT_LOAD_PARAMS.get("n_gpu_layers", -1),
        help=f"Number of GPU layers (-1=auto/all, 0=CPU only, >0=specific count, default: {DEFAULT_LOAD_PARAMS.get('n_gpu_layers', -1)})",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=DEFAULT_LOAD_PARAMS.get("n_batch", 512),
        help=(
            "Batch size for prompt processing "
            f"(default: {DEFAULT_LOAD_PARAMS.get('n_batch', 512)})"
        ),
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=DEFAULT_LOAD_PARAMS.get("n_threads", -1),
        help=f"Number of CPU threads (-1=auto/all, default: {DEFAULT_LOAD_PARAMS.get('n_threads', -1)})",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        default=DEFAULT_LOAD_PARAMS.get("flash_attention", True),
        help=(
            "Enable Flash Attention (faster attention computation, default: "
            f"{'enabled' if DEFAULT_LOAD_PARAMS.get('flash_attention', True) else 'disabled'})"
        ),
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_false",
        dest="flash_attention",
        help="Disable Flash Attention",
    )
    parser.add_argument(
        "--rope-freq-base",
        type=float,
        default=DEFAULT_LOAD_PARAMS.get("rope_freq_base"),
        help="RoPE frequency base (None=model default, e.g. 10000.0)",
    )
    parser.add_argument(
        "--rope-freq-scale",
        type=float,
        default=DEFAULT_LOAD_PARAMS.get("rope_freq_scale"),
        help="RoPE frequency scaling (None=model default, e.g. 1.0)",
    )
    parser.add_argument(
        "--use-mmap",
        action="store_true",
        default=DEFAULT_LOAD_PARAMS.get("use_mmap", True),
        help=(
            "Enable memory-mapping (faster model load, default: "
            f"{'enabled' if DEFAULT_LOAD_PARAMS.get('use_mmap', True) else 'disabled'})"
        ),
    )
    parser.add_argument(
        "--no-mmap",
        action="store_false",
        dest="use_mmap",
        help="Disable memory-mapping",
    )
    parser.add_argument(
        "--use-mlock",
        action="store_true",
        default=DEFAULT_LOAD_PARAMS.get("use_mlock", False),
        help=(
            "Enable memory-locking (prevents swapping, default: "
            f"{'enabled' if DEFAULT_LOAD_PARAMS.get('use_mlock', False) else 'disabled'})"
        ),
    )
    parser.add_argument(
        "--kv-cache-quant",
        type=str,
        choices=["f32", "f16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"],
        default=DEFAULT_LOAD_PARAMS.get("kv_cache_quant"),
        help="KV-Cache quantization (reduces VRAM, may affect performance, None=model default)",
    )

    parser.add_argument(
        "--use-rest-api",
        action="store_true",
        help="Use LM Studio REST API v1 instead of Python SDK/CLI (enables advanced features like stateful chats, parallel requests)",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="LM Studio API permission token for REST API authentication",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=None,
        help="Max. parallel predictions per model (REST API only, default: 4, requires continuous batching support)",
    )
    parser.add_argument(
        "--unified-kv-cache",
        action="store_true",
        default=None,
        help="Enable unified KV cache (REST API only, optimizes VRAM for parallel requests)",
    )

    args = parser.parse_args(args=final_cli_args)

    logger.info("📦 Loaded preset: %s", preset_args.preset)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logging.root.handlers:
            handler.setLevel(logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("lmstudio").setLevel(logging.DEBUG)
        logging.getLogger("urllib3").setLevel(logging.DEBUG)
        logging.getLogger("websockets").setLevel(logging.DEBUG)
        logger.debug("🐛 DEBUG mode enabled")

    if args.list_cache:
        cache = BenchmarkCache()
        cached = cache.list_cached_models()
        if cached:
            print("\n=== Cached Benchmark Results ===")
            print(
                f"{'Model':<50} {'Quant':<10} {'Params':<8} {'tok/s':<10} {'Date':<12} {'Hash':<10}"
            )
            print("-" * 110)
            for entry in cached:
                print(
                    f"{entry['model_name']:<50} {entry['quantization']:<10} {entry['params_size']:<8} "
                    f"{entry['avg_tokens_per_sec']:<10.2f} {entry['timestamp'][:10]:<12} {entry['params_hash']:<10}"
                )
            print(f"\nTotal: {len(cached)} entries")
        else:
            print("Cache is empty - no results saved")
        return

    if args.export_cache:
        cache = BenchmarkCache()
        output_file = RESULTS_DIR / args.export_cache
        cache.export_to_json(output_file)
        print(f"Cache exported to: {output_file}")
        return

    if args.export_only:
        logger.info("🔄 === Report Regeneration from Database ===")
        cache = BenchmarkCache()
        cached_results = cache.get_all_results()

        if not cached_results:
            logger.error("❌ No results found in database. Run a benchmark first.")
            return

        logger.info("📥 Loading %s results from database...", len(cached_results))

        filter_args = {
            "only_vision": args.only_vision,
            "only_tools": args.only_tools,
            "quants": args.quants,
            "arch": args.arch,
            "params": args.params,
            "min_context": args.min_context,
            "max_size": args.max_size,
            "include_models": args.include_models,
            "exclude_models": args.exclude_models,
        }

        benchmark = LMStudioBenchmark(
            num_runs=1,
            context_length=2048,
            prompt="",
            model_limit=None,
            filter_args=filter_args,
            compare_with=args.compare_with,
            rank_by=args.rank_by,
            use_cache=False,
            enable_profiling=False,
            use_gtt=not args.disable_gtt,
        )

        benchmark.results = cached_results

        if any(filter_args.values()):
            original_count = len(benchmark.results)
            benchmark.results = [
                r for r in benchmark.results if benchmark._matches_filters(r)
            ]
            logger.info(
                f"✔️ After filtering: {len(benchmark.results)}/{original_count} models"
            )

        if not benchmark.results:
            logger.error("❌ No results remaining after filtering")
            return

        if args.compare_with:
            benchmark._load_previous_results()

        logger.info("⚙️ Generating reports for %s models...", len(benchmark.results))
        benchmark.export_results()
        logger.info("✅ Reports successfully generated!")
        return

    if args.dev_mode:
        logger.info("🧪 === Development mode enabled ===")
        args.runs = 1
        args.limit = 1
        all_models = ModelDiscovery.get_installed_models()
        if all_models:
            model_sizes = []
            for model_key in all_models:
                metadata = ModelDiscovery.get_model_metadata(model_key)
                model_sizes.append((model_key, metadata.get("model_size_gb", 999)))

            model_sizes.sort(key=lambda x: x[1])
            smallest = model_sizes[0][0]
            logger.info(
                f"✅ Smallest model selected: {smallest} ({model_sizes[0][1]:.2f} GB)"
            )
            logger.info("⚙️ Configuration: 1 measurement, context %s", args.context)
            logger.info("")
        else:
            logger.error("❌ No models found for dev-mode")
            return

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.context < 256:
        parser.error("--context must be >= 256")
    if len(args.prompt.strip()) == 0:
        parser.error("--prompt must not be empty")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    filter_args = {
        "only_vision": args.only_vision,
        "only_tools": args.only_tools,
        "quants": args.quants,
        "arch": args.arch,
        "params": args.params,
        "min_context": args.min_context,
        "max_size": args.max_size,
        "include_models": args.include_models,
        "exclude_models": args.exclude_models,
    }

    logger.info("🚀 === LM Studio Model Benchmark ===")
    logger.info("💬 Prompt: '%s'", args.prompt)
    logger.info("📏 Context Length: %s Tokens", args.context)
    logger.info("🔢 Measurements per Model: %s (+ %s Warmup)", args.runs, NUM_WARMUP_RUNS)
    if args.limit:
            logger.info("📌 Model Limit: Testing max. %s models", args.limit)
    active_filters = [k for k, v in filter_args.items() if v]
    if active_filters:
        logger.info("🔎 Active filters: %s", ', '.join(active_filters))

    if args.compare_with:
        logger.info("📈 Historical comparison: %s", args.compare_with)

    logger.info(
        "⏱️ Estimated total time (worst-case, uncached): ~%s minutes",
        int(args.runs * 45 * (args.limit or 9) / 9),
    )
    logger.info("")

    inference_overrides = {
        "temperature": args.temperature,
        "top_k_sampling": args.top_k_sampling,
        "top_p_sampling": args.top_p_sampling,
        "min_p_sampling": args.min_p_sampling,
        "repeat_penalty": args.repeat_penalty,
        "max_tokens": args.max_tokens,
    }

    load_params = {
        "n_gpu_layers": args.n_gpu_layers,
        "n_batch": args.n_batch,
        "n_threads": args.n_threads,
        "flash_attention": args.flash_attention,
        "rope_freq_base": args.rope_freq_base,
        "rope_freq_scale": args.rope_freq_scale,
        "use_mmap": args.use_mmap,
        "use_mlock": args.use_mlock,
        "kv_cache_quant": args.kv_cache_quant,
        "n_parallel": args.n_parallel,
        "unified_kv_cache": args.unified_kv_cache,
    }

    if args.api_token:
        lmstudio_config = DEFAULT_CONFIG.get("lmstudio", {})
        lmstudio_config["api_token"] = args.api_token

    benchmark = LMStudioBenchmark(
        num_runs=args.runs,
        context_length=args.context,
        prompt=args.prompt,
        model_limit=args.limit,
        filter_args=filter_args,
        compare_with=args.compare_with,
        rank_by=args.rank_by,
        use_cache=not args.retest,
        enable_profiling=args.enable_profiling,
        max_temp=args.max_temp,
        max_power=args.max_power,
        use_gtt=not args.disable_gtt,
        inference_overrides=inference_overrides,
        load_params=load_params,
        use_rest_api=args.use_rest_api,
    )
    run_status = benchmark.run_all_benchmarks()

    if run_status == "no_new_models":
        logger.info("ℹ️ Benchmark finished with cache only (0 new models tested)")
    elif run_status == "failed":
        logger.error("❌ Benchmark failed before test execution")

    logger.info("🎉 Benchmark completed!")


if __name__ == "__main__":
    main()
