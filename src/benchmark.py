#!/usr/bin/env python3
"""
LM Studio Model Benchmark Tool

Automatic testing of all locally installed LM Studio models and their
quantizations. Measures token/s speed with a standardized prompt.
"""

import subprocess
import json
import csv
import logging
import os
import time
import shutil
import argparse
import sqlite3
import hashlib
import threading
import psutil
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from statistics import quantiles, mean, median
from tqdm import tqdm
import re
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak

from config_loader import BASE_DEFAULT_CONFIG, DEFAULT_CONFIG
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    go = None
    PLOTLY_AVAILABLE = False


# Create logs directory if it does not exist
SCRIPT_DIR = Path(__file__).parent  # src/
PROJECT_ROOT = SCRIPT_DIR.parent    # root
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging - log file is ONLY created when called as __main__
# NOT on import (to not affect WebApp startup)
log_filename = None  # Will be set later when __main__

# Custom filter to filter JSON logs from external libs
class NoJSONFilter(logging.Filter):
    def filter(self, record):
        # Filter JSON logs and other debug events
        if record.getMessage().startswith('{'):
            return False
        return True

# Initialize logging with console only (log file will be added later)
# StreamHandler uses sys.stderr by default - we force sys.stdout for WebApp compatibility
import sys

# Auto-flush for stdout - important for WebApp subprocess communication
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

# Replace sys.stdout with auto-flush version
sys.stdout = AutoFlushStream(sys.stdout)

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        stream_handler  # Explicitly on stdout for WebApp integration
    ]
)
logger = logging.getLogger(__name__)

# Add NoJSONFilter to all handlers
for handler in logging.root.handlers:
    handler.addFilter(NoJSONFilter())

# Mute debug logs from third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("lmstudio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)


# Constants
BASE_STANDARD_PROMPT = "Explain machine learning in 3 sentences"
BASE_CONTEXT_LENGTH = 2048
BASE_NUM_MEASUREMENT_RUNS = 3
GPU_OFFLOAD_LEVELS = [1.0, 0.7, 0.5, 0.3]  # Fallback if no intelligent calculation possible
NUM_WARMUP_RUNS = 1

STANDARD_PROMPT = DEFAULT_CONFIG.get("prompt", BASE_STANDARD_PROMPT)
CONTEXT_LENGTH = int(DEFAULT_CONFIG.get("context_length", BASE_CONTEXT_LENGTH))
NUM_MEASUREMENT_RUNS = int(DEFAULT_CONFIG.get("num_runs", BASE_NUM_MEASUREMENT_RUNS))

# GPU-offload optimization
VRAM_SAFETY_HEADROOM_GB = 1.0  # reserve for system
CONTEXT_VRAM_FACTOR = 0.000002  # ~2KB pro Token
RESULTS_DIR = PROJECT_ROOT / "results"
DATABASE_FILE = RESULTS_DIR / "benchmark_cache.db"
METADATA_DATABASE_FILE = RESULTS_DIR / "model_metadata.db"

# Optimized inference parameters for standardized benchmarks
# (For consistent, reproducible measurements)
BASE_OPTIMIZED_INFERENCE_PARAMS = {
    'temperature': 0.1,        # Low for consistent results (instead of default 0.8)
    'top_k_sampling': 40,      # Sampling from top 40 tokens
    'top_p_sampling': 0.9,     # Nucleus sampling at 90%
    'min_p_sampling': 0.05,    # Minimum probability threshold
    'repeat_penalty': 1.2,     # Slight penalty against repetitions (default 1.1)
    'max_tokens': 256,         # Limited output length for faster measurements
}

BASE_LOAD_PARAMS = dict(BASE_DEFAULT_CONFIG.get('load', {}))

OPTIMIZED_INFERENCE_PARAMS = {
    **BASE_OPTIMIZED_INFERENCE_PARAMS,
    **(DEFAULT_CONFIG.get('inference') or {})
}

DEFAULT_LOAD_PARAMS = {
    **BASE_LOAD_PARAMS,
    **(DEFAULT_CONFIG.get('load') or {})
}


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
    
    # Model metadata
    params_size: str                   # e.g. "3B", "7B"
    architecture: str                  # e.g. "mistral3", "gemma3"
    max_context_length: int            # e.g. 262144
    model_size_gb: float               # e.g. 3.11 (file size in GB)
    has_vision: bool                   # Vision support
    has_tools: bool                    # Tool-calling support
    
    # Efficiency metrics
    tokens_per_sec_per_gb: float       # Tokens/s per GB model size
    tokens_per_sec_per_billion_params: float  # Tokens/s per billion parameters
    
    # Hardware profiling (optional)
    temp_celsius_min: Optional[float] = None   # Min. temperature during benchmark
    temp_celsius_max: Optional[float] = None   # Max. temperature during benchmark
    temp_celsius_avg: Optional[float] = None   # Avg. temperature
    power_watts_min: Optional[float] = None    # Min. power draw
    power_watts_max: Optional[float] = None    # Max. power draw
    power_watts_avg: Optional[float] = None    # Avg. power draw
    
    # GTT (Graphics Translation Table) - shared system RAM for AMD GPUs (optional)
    gtt_enabled: Optional[bool] = None          # GTT usage enabled
    gtt_total_gb: Optional[float] = None        # GTT total available
    gtt_used_gb: Optional[float] = None         # GTT used
    
    # Inference parameters (for reproducibility)
    temperature: Optional[float] = None         # Sampling temperature
    top_k_sampling: Optional[int] = None        # Top-K token filter
    top_p_sampling: Optional[float] = None      # Nucleus sampling threshold
    min_p_sampling: Optional[float] = None      # Min probability threshold
    repeat_penalty: Optional[float] = None      # Penalty against repetitions
    max_tokens: Optional[int] = None            # Max output tokens
    
    # Load-config performance parameters
    n_gpu_layers: Optional[int] = None          # Number of GPU layers (explicit)
    n_batch: Optional[int] = None               # Batch size for prompt processing
    n_threads: Optional[int] = None             # CPU threads for non-GPU ops
    flash_attention: Optional[bool] = None      # Flash attention enabled
    rope_freq_base: Optional[float] = None      # RoPE frequency base
    rope_freq_scale: Optional[float] = None     # RoPE frequency scale
    use_mmap: Optional[bool] = None             # Memory mapping enabled
    use_mlock: Optional[bool] = None            # RAM locking enabled
    kv_cache_quant: Optional[str] = None        # KV cache quantization (none/q4_0/q8_0)
    
    # Run information
    num_runs: Optional[int] = None              # Number of measurements performed
    runs_averaged_from: Optional[int] = None    # Number of successful runs averaged
    warmup_runs: Optional[int] = None           # Number of warmup runs
    run_index: Optional[int] = None             # Index of individual run (0, 1, 2) for statistical analysis
    
    # Version information
    lmstudio_version: Optional[str] = None      # LM Studio version at test time
    nvidia_driver_version: Optional[str] = None # NVIDIA driver version
    rocm_driver_version: Optional[str] = None   # AMD ROCm/driver version
    intel_driver_version: Optional[str] = None  # Intel GPU driver version
    
    # System and configuration information (for reproducibility)
    context_length: Optional[int] = None        # Context length of the benchmark (e.g. 2048)
    prompt_hash: Optional[str] = None           # SHA256 hash of the used prompt
    model_key: Optional[str] = None             # Unique model ID (e.g. qwen/qwen2.5-7b@q3_k_l)
    params_hash: Optional[str] = None           # Hash of inference parameters
    os_name: Optional[str] = None               # Operating system (e.g. Linux, Windows, macOS)
    os_version: Optional[str] = None            # Kernel/OS version (e.g. 6.8.0-45-generic)
    cpu_model: Optional[str] = None             # CPU model (e.g. AMD Ryzen 9 7950X)
    python_version: Optional[str] = None        # Python version (e.g. 3.11.8)
    benchmark_duration_seconds: Optional[float] = None  # Total benchmark duration in seconds
    error_count: Optional[int] = None           # Number of errors during benchmark
    inference_params_hash: Optional[str] = None # Hash of all inference parameters combined
    
    # Historical comparison (optional)
    speed_delta_pct: Optional[float] = None     # Performance change (%) vs. previous benchmark
    prev_timestamp: Optional[str] = None         # Timestamp of previous benchmark


class HardwareMonitor:
    """Real-time monitoring of GPU temperature and power draw"""
    
    def __init__(self, gpu_type: Optional[str], gpu_tool: Optional[str], enabled: bool = False):
        self.gpu_type = gpu_type
        self.gpu_tool = gpu_tool
        self.enabled = enabled
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.temps: List[float] = []
        self.powers: List[float] = []
        self.vrams: List[float] = []  # VRAM usage in GB
        self.gtts: List[float] = []   # GTT usage in GB (system RAM for AMD GPUs)
        self.cpus: List[float] = []   # CPU utilization in %
        self.rams: List[float] = []   # RAM usage in GB
        self.ram_readings: List[float] = []  # For RAM smoothing (rolling avg over 7 measurements)
        self.lock = threading.Lock()
    
    def start(self):
        """Start background monitoring"""
        if not self.enabled:
            logger.info("⚠️ Hardware monitoring disabled (--enable-profiling not set)")
            return
        
        if not self.gpu_tool:
            logger.warning("⚠️ No GPU tools found - hardware monitoring not available")
            return
        
        logger.info(f"🔥 Starting hardware monitoring (GPU: {self.gpu_type}, Tool: {self.gpu_tool})")
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
            'temp_celsius_min': min(temps) if temps else None,
            'temp_celsius_max': max(temps) if temps else None,
            'temp_celsius_avg': mean(temps) if temps else None,
            'power_watts_min': min(powers) if powers else None,
            'power_watts_max': max(powers) if powers else None,
            'power_watts_avg': mean(powers) if powers else None,
            'vram_gb_min': min(vrams) if vrams else None,
            'vram_gb_max': max(vrams) if vrams else None,
            'vram_gb_avg': mean(vrams) if vrams else None,
            'gtt_gb_min': min(gtts) if gtts else None,
            'gtt_gb_max': max(gtts) if gtts else None,
            'gtt_gb_avg': mean(gtts) if gtts else None,
            'cpu_percent_min': min(cpus) if cpus else None,
            'cpu_percent_max': max(cpus) if cpus else None,
            'cpu_percent_avg': mean(cpus) if cpus else None,
            'ram_gb_min': min(rams) if rams else None,
            'ram_gb_max': max(rams) if rams else None,
            'ram_gb_avg': mean(rams) if rams else None,
        }
        return stats
    
    def _monitor_loop(self):
        """Background thread for continuous measurements"""
        logger.info("🔍 Hardware-Monitor-Thread gestartet")
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
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"🌡️ GPU Temp: {temp:.1f}°C")
                    else:
                        logger.debug(f"⚠️ No temperature read (gpu_type={self.gpu_type}, tool={self.gpu_tool})")
                    
                    if power is not None:
                        self.powers.append(power)
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"⚡ GPU Power: {power:.1f}W")
                    else:
                        logger.debug(f"⚠️ No power read (gpu_type={self.gpu_type}, tool={self.gpu_tool})")
                    
                    if vram is not None:
                        self.vrams.append(vram)
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"💾 GPU VRAM: {vram:.1f}GB")
                    else:
                        logger.debug(f"⚠️ No VRAM read (gpu_type={self.gpu_type}, tool={self.gpu_tool})")
                    
                    if gtt is not None:
                        self.gtts.append(gtt)
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"🧠 GPU GTT: {gtt:.1f}GB")
                    else:
                        logger.debug(f"⚠️ No GTT read (gpu_type={self.gpu_type}, tool={self.gpu_tool})")
                    
                    if cpu is not None:
                        self.cpus.append(cpu)
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"🖥️ CPU: {cpu:.1f}%")
                    
                    if ram is not None:
                        self.rams.append(ram)
                        # Logger for log file AND stdout (via AutoFlushStream)
                        logger.info(f"💾 RAM: {ram:.1f}GB")
                
                time.sleep(1)  # Measurements every second
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                time.sleep(2)
        logger.info("🛑 Hardware monitor thread stopped")
    
    def _get_temperature(self) -> Optional[float]:
        """Reads current GPU temperature"""
        try:
            if not self.gpu_tool:
                return None
            
            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [self.gpu_tool, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    temp_str = result.stdout.strip().split('\n')[0]
                    return float(temp_str)
            
            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, '--showtemp'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    # Parse AMD rocm-smi output:
                    # Format: "GPU[0]          : Temperature (Sensor edge) (C): 47.0"
                    import re
                    for line in result.stdout.split('\n'):
                        if 'GPU[' in line and ('(C):' in line or 'c' in line.lower()):
                            try:
                                # Try to extract last number in the line
                                match = re.search(r'[\d.]+\s*$', line.strip())
                                if match:
                                    return float(match.group())
                                # Fallback: old method
                                return float(temp_str)
                            except (ValueError, IndexError):
                                pass
        except (subprocess.TimeoutExpired, Exception):
            pass
        
        return None
    
    def _get_power_draw(self) -> Optional[float]:
        """Reads GPU power draw in watts"""
        try:
            if not self.gpu_tool:
                return None
            
            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [self.gpu_tool, '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    power_str = result.stdout.strip().split('\n')[0]
                    return float(power_str)
            
            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, '--showpower'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    # Parse AMD rocm-smi output:
                    # Format: "GPU[0]          : Current Socket Graphics Package Power (W): 30.076"
                    import re
                    for line in result.stdout.split('\n'):
                        if 'GPU[' in line and ('(W):' in line or 'W' in line):
                            try:
                                # Try to extract last number in the line
                                match = re.search(r'[\d.]+\s*$', line.strip())
                                if match:
                                    return float(match.group())
                                # Fallback: old method
                                power_str = line.split(':')[-1].strip().replace('W', '')
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
                    [self.gpu_tool, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    vram_mb = float(result.stdout.strip().split('\n')[0])
                    return vram_mb / 1024.0  # MB to GB
            
            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, '--showmeminfo', 'vram'],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    # Parse AMD rocm-smi output:
                    # Format: "GPU[0]          : VRAM Total Used Memory (B): 1234567890"
                    import re
                    for line in result.stdout.split('\n'):
                        if 'GPU[' in line and 'Used Memory' in line:
                            match = re.search(r'(\d+)\s*$', line.strip())
                            if match:
                                vram_bytes = float(match.group(1))
                                return vram_bytes / (1024**3)  # Bytes to GB
        except (subprocess.TimeoutExpired, Exception):
            pass
        
        return None
    
    def _get_gtt_usage(self) -> Optional[float]:
        """Reads GTT usage in GB (system RAM for AMD GPUs)"""
        try:
            if self.gpu_type != "AMD" or not self.gpu_tool:
                return None
            
            result = subprocess.run(
                [self.gpu_tool, '--showmeminfo', 'gtt'],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                # Parse AMD rocm-smi output:
                # Format: "GPU[0]          : GTT Total Used Memory (B): 1234567890"
                import re
                for line in result.stdout.split('\n'):
                    if 'GPU[' in line and 'Used Memory' in line:
                        match = re.search(r'(\d+)\s*$', line.strip())
                        if match:
                            gtt_bytes = float(match.group(1))
                            return gtt_bytes / (1024**3)  # Bytes to GB
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
        """Reads system RAM usage in GB with smoothing (rolling avg over 7 measurements)"""
        try:
            mem = psutil.virtual_memory()
            # Use mem.used - that is the most reliable
            current_ram = mem.used / (1024**3)  # Bytes to GB
            
            # Rolling average: keep last 7 readings for strong smoothing
            self.ram_readings.append(current_ram)
            if len(self.ram_readings) > 7:
                self.ram_readings.pop(0)
            
            # Return average of last 7 (or fewer) measurements
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
        
        cursor.execute('''
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
        ''')
        
        # Index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_model_key_hash 
            ON benchmark_results(model_key, inference_params_hash)
        ''')
        
        # Migration: add run_index column if missing
        try:
            cursor.execute("SELECT run_index FROM benchmark_results LIMIT 1")
        except sqlite3.OperationalError:
            logger.info("📦 Migration: Adding run_index column...")
            cursor.execute("ALTER TABLE benchmark_results ADD COLUMN run_index INTEGER")
            conn.commit()
            logger.info("✅ Migration successful")
        
        # Migration: add performance parameter columns
        new_columns = [
            ("n_gpu_layers", "INTEGER"),
            ("n_batch", "INTEGER"),
            ("n_threads", "INTEGER"),
            ("flash_attention", "INTEGER"),
            ("rope_freq_base", "REAL"),
            ("rope_freq_scale", "REAL"),
            ("use_mmap", "INTEGER"),
            ("use_mlock", "INTEGER"),
            ("kv_cache_quant", "TEXT")
        ]
        for col_name, col_type in new_columns:
            try:
                cursor.execute(f"SELECT {col_name} FROM benchmark_results LIMIT 1")
            except sqlite3.OperationalError:
                logger.info(f"📦 Migration: Adding {col_name} column...")
                cursor.execute(f"ALTER TABLE benchmark_results ADD COLUMN {col_name} {col_type}")
                conn.commit()
        
        conn.commit()
        conn.close()
    
    @staticmethod
    def compute_params_hash(prompt: str, context_length: int, inference_params: dict, load_params: Optional[dict] = None) -> str:
        """Calculates hash from all relevant parameters"""
        params_dict = {
            'prompt': prompt,
            'context_length': context_length,
            **inference_params
        }
        if load_params:
            params_dict['load_config'] = load_params
        hash_input = json.dumps(params_dict, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def get_cached_result(self, model_key: str, params_hash: str) -> Optional[BenchmarkResult]:
        """Fetches cached result from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check which columns exist for backward compatibility
        cursor.execute("PRAGMA table_info(benchmark_results)")
        columns = {row[1]: row[0] for row in cursor.fetchall()}  # {column_name: index}
        
        cursor.execute('''
            SELECT * FROM benchmark_results 
            WHERE model_key = ? AND inference_params_hash = ?
            ORDER BY timestamp DESC LIMIT 1
        ''', (model_key, params_hash))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            # Base fields (use fixed positions only for critical fields)
            result_dict = {
                'model_name': row[2],
                'quantization': row[3],
                'gpu_type': row[5],
                'gpu_offload': row[6],
                'vram_mb': row[7],
                'avg_tokens_per_sec': row[8],
                'avg_ttft': row[9],
                'avg_gen_time': row[10],
                'prompt_tokens': row[11],
                'completion_tokens': row[12],
                'timestamp': row[13],
                'params_size': row[14],
                'architecture': row[15],
                'max_context_length': row[16],
                'model_size_gb': row[17],
                'has_vision': bool(row[18]),
                'has_tools': bool(row[19]),
                'tokens_per_sec_per_gb': row[20],
                'tokens_per_sec_per_billion_params': row[21],
                'speed_delta_pct': row[22],
                'prev_timestamp': row[23],
            }
            
            # Optionale neue Felder (Load-Config Parameter)
            if 'n_gpu_layers' in columns:
                result_dict['n_gpu_layers'] = row[columns['n_gpu_layers']]
            if 'n_batch' in columns:
                result_dict['n_batch'] = row[columns['n_batch']]
            if 'n_threads' in columns:
                result_dict['n_threads'] = row[columns['n_threads']]
            if 'flash_attention' in columns:
                val = row[columns['flash_attention']]
                result_dict['flash_attention'] = bool(val) if val is not None else None
            if 'rope_freq_base' in columns:
                result_dict['rope_freq_base'] = row[columns['rope_freq_base']]
            if 'rope_freq_scale' in columns:
                result_dict['rope_freq_scale'] = row[columns['rope_freq_scale']]
            if 'use_mmap' in columns:
                val = row[columns['use_mmap']]
                result_dict['use_mmap'] = bool(val) if val is not None else None
            if 'use_mlock' in columns:
                val = row[columns['use_mlock']]
                result_dict['use_mlock'] = bool(val) if val is not None else None
            if 'kv_cache_quant' in columns:
                result_dict['kv_cache_quant'] = row[columns['kv_cache_quant']]
            
            return BenchmarkResult(**result_dict)
        return None
    
    def save_result(self, result: BenchmarkResult, model_key: str, params_hash: str, 
                   prompt: str, context_length: int):
        """Saves benchmark result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Prepare all values (47 + 9 new = 56 columns)
            values = (
                model_key, result.model_name, result.quantization, result.inference_params_hash,
                result.gpu_type, result.gpu_offload, result.vram_mb, result.avg_tokens_per_sec,
                result.avg_ttft, result.avg_gen_time, result.prompt_tokens, result.completion_tokens,
                result.timestamp, result.params_size, result.architecture, result.max_context_length,
                result.model_size_gb, int(result.has_vision), int(result.has_tools),
                result.tokens_per_sec_per_gb, result.tokens_per_sec_per_billion_params,
                result.speed_delta_pct, result.prev_timestamp, prompt, context_length,
                result.temperature, result.top_k_sampling, result.top_p_sampling,
                result.min_p_sampling, result.repeat_penalty, result.max_tokens,
                result.num_runs, result.runs_averaged_from, result.warmup_runs,
                result.run_index, result.lmstudio_version, result.nvidia_driver_version, result.rocm_driver_version,
                result.intel_driver_version,
                result.prompt_hash, params_hash, result.os_name, result.os_version,
                result.cpu_model, result.python_version, result.benchmark_duration_seconds,
                result.error_count,
                # Neue Load-Config Parameter (9 Spalten)
                result.n_gpu_layers, result.n_batch, result.n_threads,
                int(result.flash_attention) if result.flash_attention is not None else None,
                result.rope_freq_base, result.rope_freq_scale,
                int(result.use_mmap) if result.use_mmap is not None else None,
                int(result.use_mlock) if result.use_mlock is not None else None,
                result.kv_cache_quant
            )
            
            logger.debug(f"📊 INSERT: {len(values)} values for 56 columns")

            placeholders = ", ".join("?" for _ in values)

            cursor.execute(f'''
                INSERT INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, tokens_per_sec_per_gb,
                    tokens_per_sec_per_billion_params, speed_delta_pct, prev_timestamp,
                    prompt, context_length, temperature, top_k_sampling, top_p_sampling,
                    min_p_sampling, repeat_penalty, max_tokens, num_runs, runs_averaged_from,
                    warmup_runs, run_index, lmstudio_version, nvidia_driver_version, rocm_driver_version,
                    intel_driver_version, prompt_hash, params_hash, os_name, os_version,
                    cpu_model, python_version, benchmark_duration_seconds, error_count,
                    n_gpu_layers, n_batch, n_threads, flash_attention, rope_freq_base,
                    rope_freq_scale, use_mmap, use_mlock, kv_cache_quant
                ) VALUES ({placeholders})
            ''', values)
            
            conn.commit()
        except Exception as e:
            logger.error(f"❌ Error saving to cache: {e}")
        finally:
            conn.close()
    
    def get_all_results(self) -> List[BenchmarkResult]:
        """Loads all benchmark results from the database"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Check which columns exist
            cursor.execute("PRAGMA table_info(benchmark_results)")
            columns = {row[1] for row in cursor.fetchall()}
            
            # Basis-Spalten (immer vorhanden)
            base_cols = "model_name, quantization, gpu_type, gpu_offload, vram_mb, " \
                       "avg_tokens_per_sec, avg_ttft, avg_gen_time, prompt_tokens, completion_tokens, " \
                       "timestamp, params_size, architecture, max_context_length, model_size_gb, " \
                       "has_vision, has_tools, tokens_per_sec_per_gb, tokens_per_sec_per_billion_params"
            
            # Optional columns (if available)
            optional_cols = []
            if 'temp_celsius_min' in columns:
                optional_cols.extend(['temp_celsius_min', 'temp_celsius_max', 'temp_celsius_avg'])
            if 'power_watts_min' in columns:
                optional_cols.extend(['power_watts_min', 'power_watts_max', 'power_watts_avg'])
            if 'gtt_enabled' in columns:
                optional_cols.extend(['gtt_enabled', 'gtt_total_gb', 'gtt_used_gb'])
            if 'speed_delta_pct' in columns:
                optional_cols.extend(['speed_delta_pct', 'prev_timestamp'])
            
            # New columns for extended cache storage
            if 'temperature' in columns:
                optional_cols.extend(['temperature', 'top_k_sampling', 'top_p_sampling', 
                                    'min_p_sampling', 'repeat_penalty', 'max_tokens'])
            if 'num_runs' in columns:
                optional_cols.extend(['num_runs', 'runs_averaged_from', 'warmup_runs'])
            if 'lmstudio_version' in columns:
                optional_cols.extend(['lmstudio_version', 'nvidia_driver_version', 
                                    'rocm_driver_version', 'intel_driver_version'])
            if 'prompt_hash' in columns:
                optional_cols.extend(['prompt_hash', 'params_hash', 'os_name', 'os_version',
                                    'cpu_model', 'python_version', 'benchmark_duration_seconds', 'error_count'])
            if 'inference_params_hash' in columns:
                optional_cols.append('inference_params_hash')
            
            select_cols = base_cols + (", " + ", ".join(optional_cols) if optional_cols else "")
            
            cursor.execute(f'''
                SELECT {select_cols}
                FROM benchmark_results
                ORDER BY timestamp DESC
            ''')
            
            results = []
            for row in cursor.fetchall():
                idx = 0
                # Base fields (19 columns)
                result_dict = {
                    'model_name': row[idx], 'quantization': row[idx+1], 'gpu_type': row[idx+2],
                    'gpu_offload': row[idx+3], 'vram_mb': row[idx+4],
                    'avg_tokens_per_sec': row[idx+5], 'avg_ttft': row[idx+6], 'avg_gen_time': row[idx+7],
                    'prompt_tokens': row[idx+8], 'completion_tokens': row[idx+9], 'timestamp': row[idx+10],
                    'params_size': row[idx+11], 'architecture': row[idx+12], 'max_context_length': row[idx+13],
                    'model_size_gb': row[idx+14], 'has_vision': bool(row[idx+15]), 'has_tools': bool(row[idx+16]),
                    'tokens_per_sec_per_gb': row[idx+17], 'tokens_per_sec_per_billion_params': row[idx+18]
                }
                idx = 19
                
                # Optional fields
                if 'temp_celsius_min' in columns:
                    result_dict['temp_celsius_min'] = row[idx]
                    result_dict['temp_celsius_max'] = row[idx+1]
                    result_dict['temp_celsius_avg'] = row[idx+2]
                    idx += 3
                
                if 'power_watts_min' in columns:
                    result_dict['power_watts_min'] = row[idx]
                    result_dict['power_watts_max'] = row[idx+1]
                    result_dict['power_watts_avg'] = row[idx+2]
                    idx += 3
                
                if 'gtt_enabled' in columns:
                    result_dict['gtt_enabled'] = bool(row[idx]) if row[idx] is not None else None
                    result_dict['gtt_total_gb'] = row[idx+1]
                    result_dict['gtt_used_gb'] = row[idx+2]
                    idx += 3
                
                if 'speed_delta_pct' in columns:
                    result_dict['speed_delta_pct'] = row[idx]
                    result_dict['prev_timestamp'] = row[idx+1]
                    idx += 2
                
                # New fields - inference parameters
                if 'temperature' in columns:
                    result_dict['temperature'] = row[idx]
                    result_dict['top_k_sampling'] = row[idx+1]
                    result_dict['top_p_sampling'] = row[idx+2]
                    result_dict['min_p_sampling'] = row[idx+3]
                    result_dict['repeat_penalty'] = row[idx+4]
                    result_dict['max_tokens'] = row[idx+5]
                    idx += 6
                
                # New fields - run information
                if 'num_runs' in columns:
                    result_dict['num_runs'] = row[idx]
                    result_dict['runs_averaged_from'] = row[idx+1]
                    result_dict['warmup_runs'] = row[idx+2]
                    idx += 3
                
                # New fields - version information
                if 'lmstudio_version' in columns:
                    result_dict['lmstudio_version'] = row[idx]
                    result_dict['nvidia_driver_version'] = row[idx+1]
                    result_dict['rocm_driver_version'] = row[idx+2]
                    result_dict['intel_driver_version'] = row[idx+3]
                    idx += 4
                
                # New fields - system and configuration information
                if 'prompt_hash' in columns:
                    result_dict['prompt_hash'] = row[idx]
                    result_dict['params_hash'] = row[idx+1]
                    result_dict['os_name'] = row[idx+2]
                    result_dict['os_version'] = row[idx+3]
                    result_dict['cpu_model'] = row[idx+4]
                    result_dict['python_version'] = row[idx+5]
                    result_dict['benchmark_duration_seconds'] = row[idx+6]
                    result_dict['error_count'] = row[idx+7]
                    idx += 8
                
                # New fields - inference params hash
                if 'inference_params_hash' in columns:
                    result_dict['inference_params_hash'] = row[idx]
                
                result = BenchmarkResult(**result_dict)
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"❌ Error loading all results: {e}")
            return []
        finally:
            conn.close()
    
    def list_cached_models(self) -> List[Dict]:
        """Returns all cached models"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT model_key, model_name, quantization, avg_tokens_per_sec, 
                   timestamp, inference_params_hash, params_size
            FROM benchmark_results
            ORDER BY timestamp DESC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'model_key': row[0],
                'model_name': row[1],
                'quantization': row[2],
                'avg_tokens_per_sec': row[3],
                'timestamp': row[4],
                'params_hash': row[5],
                'params_size': row[6]
            })
        
        conn.close()
        return results
    
    def export_to_json(self, output_file: Path):
        """Exports cache as JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM benchmark_results ORDER BY timestamp DESC')
        
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result_dict = dict(zip(columns, row))
            # Konvertiere Boolean-Werte
            result_dict['has_vision'] = bool(result_dict['has_vision'])
            result_dict['has_tools'] = bool(result_dict['has_tools'])
            results.append(result_dict)
        
        conn.close()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Cache exportiert: {output_file}")


class GPUMonitor:
    """Detects GPU type and measures VRAM usage"""
    
    def __init__(self):
        self.gpu_type: Optional[str] = None
        self.gpu_model: Optional[str] = None  # Full GPU model name
        self.gpu_tool: Optional[str] = None
        self._detect_gpu()
    
    def _find_tool(self, tool_name: str, search_paths: List[str]) -> Optional[str]:
        """Searches for tool in PATH and specific paths"""
        # First search in PATH
        if shutil.which(tool_name):
            return tool_name
        
        # Then search in specific paths
        for path in search_paths:
            full_path = Path(path) / tool_name
            if full_path.exists() and os.access(full_path, os.X_OK):
                return str(full_path)
        
        return None
    
    def _detect_gpu(self):
        """Detects GPU type and finds corresponding monitoring tool"""
        # NVIDIA
        nvidia_paths = ['/usr/bin', '/usr/local/bin', '/usr/local/cuda/bin']
        nvidia_tool = self._find_tool('nvidia-smi', nvidia_paths)
        if nvidia_tool:
            self.gpu_type = "NVIDIA"
            self.gpu_tool = nvidia_tool
            # Fetch full GPU model name
            try:
                result = subprocess.run(
                    [nvidia_tool, '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    self.gpu_model = result.stdout.strip().split('\n')[0]
                else:
                    self.gpu_model = "NVIDIA GPU"
            except Exception:
                self.gpu_model = "NVIDIA GPU"
            logger.info(f"🟢 NVIDIA GPU erkannt: {self.gpu_model}, Tool: {nvidia_tool}")
            return
        
        # AMD - search in ROCm version directories
        amd_paths = ['/usr/bin', '/usr/local/bin', '/opt/rocm/bin']
        # Extend AMD paths with versioned ROCm directories
        import glob
        rocm_versions = glob.glob('/opt/rocm-*/bin')
        amd_paths.extend(rocm_versions)
        
        amd_tool = self._find_tool('rocm-smi', amd_paths)
        if amd_tool:
            self.gpu_type = "AMD"
            self.gpu_tool = amd_tool
            # Fetch AMD GPU model name (comprehensive detection)
            self.gpu_model = self._detect_amd_gpu_model()
            logger.info(f"🔴 AMD GPU erkannt: {self.gpu_model}, Tool: {amd_tool}")
            return
        
        # Intel
        intel_paths = ['/usr/bin', '/usr/local/bin', '/usr/lib/xpu']
        intel_tool = self._find_tool('intel_gpu_top', intel_paths)
        if intel_tool:
            self.gpu_type = "Intel"
            self.gpu_tool = intel_tool
            self.gpu_model = self._detect_intel_gpu_model()
            logger.info(f"🔵 Intel GPU erkannt: {self.gpu_model}, Tool: {intel_tool}")
            return
        
        logger.warning("⚠️ No GPU monitoring tools found. VRAM measurement not available.")
        self.gpu_type = "Unknown"
        self.gpu_model = "Unknown"
    
    def _detect_amd_gpu_model(self) -> str:
        """Detects AMD GPU model name with fallback chain"""
        # AMD Device ID Mapping
        amd_device_mapping = {
            '150e': 'Radeon Graphics',
            '7340': 'Radeon RX 5700 XT',
            '731f': 'Radeon RX 5700',
            '7360': 'Radeon RX 6700 XT',
            '73bf': 'Radeon RX 6600 XT',
            '73df': 'Radeon RX 6600',
            '15c8': 'Radeon RX 7600 XT',
            '5450': 'Radeon RX 6800 XT',
            '5498': 'Radeon RX 6900 XT',
            'gfx906': 'Radeon RX 5700 XT',
            'gfx1103': 'Radeon 890M',  # Phoenix APU
        }
        
        # 1. Try to extract iGPU from CPU info
        try:
            import cpuinfo
            cpu = cpuinfo.get_cpu_info()
            brand = cpu.get('brand_raw', '')
            # Search for Radeon in CPU string (e.g. "AMD Ryzen AI 9 HX 370 with Radeon 890M")
            if 'Radeon' in brand:
                radeon_part = brand.split('Radeon')[1].strip()
                # Extract model (e.g. "890M Graphics" -> "Radeon 890M")
                model = radeon_part.split()[0]
                if model:
                    return f"AMD Radeon {model}"
        except Exception:
            pass
        
        # 2. Try device ID from lspci
        device_id = None
        try:
            result = subprocess.run(
                ['lspci', '-d', '1002:'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if '1002:' in line:
                        parts = line.split('1002:')
                        if len(parts) > 1:
                            device_id = parts[1].split()[0].lower()
                            # If device ID is mapped, use name
                            if device_id in amd_device_mapping:
                                return f"AMD {amd_device_mapping[device_id]}"
                            break
        except Exception:
            pass
        
        # 3. Try gfx code from rocm-smi
        if self.gpu_tool:
            try:
                result = subprocess.run(
                    [self.gpu_tool, '--showproductname'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'GPU[0]' in line:
                            parts = line.split(':')
                            if len(parts) > 1:
                                gfx_code = parts[1].strip()
                                if gfx_code in amd_device_mapping:
                                    return f"AMD {amd_device_mapping[gfx_code]}"
                                return f"AMD {gfx_code}"
            except Exception:
                pass
        
        # Fallback
        if device_id:
            return f"AMD GPU (1002:{device_id})"
        return "AMD GPU"
    
    def _detect_intel_gpu_model(self) -> str:
        """Detects Intel GPU model name"""
        try:
            result = subprocess.run(
                ['lspci', '-d', '8086::0300'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout:
                # Parse first line for GPU model
                line = result.stdout.strip().split('\n')[0]
                if 'Intel' in line:
                    # Extract part after "VGA compatible controller: "
                    parts = line.split(': ')
                    if len(parts) > 1:
                        return parts[1].split('[')[0].strip()
        except Exception:
            pass
        return "Intel GPU"
    
    def get_vram_usage(self) -> str:
        """Misst aktuelle VRAM-Nutzung"""
        if not self.gpu_tool:
            return "N/A"
        
        try:
            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [self.gpu_tool, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            
            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, '--showmeminfo', 'vram'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse AMD rocm-smi output: "GPU[X] : VRAM Total Used Memory (B): XXXXXX"
                    for line in result.stdout.split('\n'):
                        if 'VRAM Total Used Memory' in line:
                            # Format: "GPU[0]          : VRAM Total Used Memory (B): 1674899456"
                            parts = line.split(':')
                            if len(parts) >= 3:
                                # Letzter Teil contains "1674899456"
                                bytes_used = parts[-1].strip()
                                try:
                                    # Konvertiere zu MB
                                    mb_used = int(bytes_used) / (1024 * 1024)
                                    return f"{int(mb_used)}"
                                except ValueError:
                                    pass
            
            elif self.gpu_type == "Intel":
                # Intel GPU Top has no simple VRAM query
                # Here one could use alternative methods
                return "N/A"
        
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"⚠️ VRAM-Messung fehlgeschlagen: {e}")
        
        return "N/A"


class LMStudioServerManager:
    """Manages LM Studio server lifecycle"""
    
    @staticmethod
    def is_server_running() -> bool:
        """Checks whether LM Studio server is running"""
        try:
            result = subprocess.run(
                ['lms', 'server', 'status'],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Server outputs to stderr (not stdout!)
            output = result.stdout + result.stderr
            return result.returncode == 0 and ('running' in output.lower() or 'port' in output.lower())
        except Exception as e:
            logger.error(f"❌ Error checking server status: {e}")
            return False
    
    @staticmethod
    def start_server():
        """Starts LM Studio server"""
        try:
            logger.info("🚀 Starting LM Studio server...")
            subprocess.Popen(
                ['lms', 'server', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait until server is ready
            max_retries = 30
            for i in range(max_retries):
                time.sleep(2)
                if LMStudioServerManager.is_server_running():
                    logger.info("✅ LM Studio Server erfolgreich gestartet")
                    return True
            
            logger.error("❌ Server-StartTimeout nach 60 Sekunden")
            return False
        
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            return False
    
    @staticmethod
    def ensure_server_running():
        """Ensures that the server is running"""
        if not LMStudioServerManager.is_server_running():
            logger.info("⚠️ Server not running, starting server...")
            return LMStudioServerManager.start_server()
        logger.info("✅ Server already running")
        return True


class ModelDiscovery:
    """Finds all locally installed models"""
    
    _metadata_cache: Dict[str, Dict] = {}  # Class-level cache
    
    @staticmethod
    def _get_metadata_cache() -> Dict[str, Dict]:
        """Cache for model metadata (loaded once at startup)"""
        if not ModelDiscovery._metadata_cache:  # Check if cache is empty
            try:
                result = subprocess.run(
                    ['lms', 'ls', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for model_data in data:
                        if model_data.get('type') == 'llm':
                            key = model_data.get('modelKey')
                            ModelDiscovery._metadata_cache[key] = {
                                'architecture': model_data.get('architecture', 'unknown'),
                                'params_size': model_data.get('paramsString', 'unknown'),
                                'max_context_length': model_data.get('maxContextLength', 0),
                                'model_size_gb': round(model_data.get('sizeBytes', 0) / 1024**3, 2),
                                'has_vision': model_data.get('vision', False),
                                'has_tools': model_data.get('trainedForToolUse', False),
                            }
            except Exception as e:
                logger.warning(f"⚠️ Error loading metadata cache: {e}")
        return ModelDiscovery._metadata_cache
    
    @staticmethod
    def get_model_metadata(model_key: str) -> Dict:
        """Fetches metadata for a specific model"""
        cache = ModelDiscovery._get_metadata_cache()
        # Extract model name (before @)
        base_model = model_key.split('@')[0] if '@' in model_key else model_key
        return cache.get(base_model, {
            'architecture': 'unknown',
            'params_size': 'unknown',
            'max_context_length': 0,
            'model_size_gb': 0.0,
            'has_vision': False,
            'has_tools': False,
        })

    @staticmethod
    def get_scraped_metadata(model_key: str) -> Dict:
        """Reads optional scraped metadata from model_metadata.db."""
        if not METADATA_DATABASE_FILE.exists():
            return {}
        base_key = model_key.split('@')[0] if '@' in model_key else model_key
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
            logger.warning(f"⚠️ Could not read scraped metadata: {e}")
            return {}
    
    @staticmethod
    def get_installed_models() -> List[str]:
        """Lists all locally installed models and quantizations"""
        try:
            result = subprocess.run(
                ['lms', 'ls', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Error running 'lms ls': {result.stderr}")
                return []
            
            # Parse JSON Output
            models = []
            import json
            data = json.loads(result.stdout)
            
            for model_data in data:
                # Only LLM models, no embeddings
                if model_data.get('type') == 'llm':
                    # Fetch all variants (quantizations)
                    variants = model_data.get('variants', [])
                    if variants:
                        models.extend(variants)
                    else:
                        # Fallback when no variants
                        models.append(model_data.get('modelKey'))
            
            logger.info(f"🔍 {len(models)} models found")
            if models:
                logger.info("📋 First 5 models:")
                for model in models[:5]:
                    logger.info(f"  • {model}")
            return models
        
        except Exception as e:
            logger.error(f"❌ Error fetching models: {e}")
            return []
    
    @staticmethod
    def filter_models(models: List[str], filter_args: Dict) -> List[str]:
        """Filters models based on CLI arguments"""
        if not filter_args:
            return models
        
        # Load metadata cache once
        metadata_cache = ModelDiscovery._get_metadata_cache()
        filtered = []
        
        # Kompiliere Regex-Pattern falls vorhanden
        include_pattern = None
        exclude_pattern = None
        
        if filter_args.get('include_models'):
            try:
                include_pattern = re.compile(filter_args['include_models'], re.IGNORECASE)
            except re.error as e:
                logger.error(f"❌ Invalid include-models pattern: {e}")
                return []
        
        if filter_args.get('exclude_models'):
            try:
                exclude_pattern = re.compile(filter_args['exclude_models'], re.IGNORECASE)
            except re.error as e:
                logger.error(f"❌ Invalid exclude-models pattern: {e}")
                return []
        
        for model_key in models:
            # Fetch metadata for model
            metadata = ModelDiscovery.get_model_metadata(model_key)
            
            # Filter: include pattern (if set, must match)
            if include_pattern and not include_pattern.search(model_key):
                continue
            
            # Filter: exclude pattern (if set and matches, skip)
            if exclude_pattern and exclude_pattern.search(model_key):
                continue
            
            # Filter: only vision models
            if filter_args.get('only_vision') and not metadata['has_vision']:
                continue
            
            # Filter: only tool models
            if filter_args.get('only_tools') and not metadata['has_tools']:
                continue
            
            # Filter: quantizations (OR conjunction)
            if filter_args.get('quants'):
                quants_list = [q.strip().lower() for q in filter_args['quants'].split(',')]
                # Extract quantization from model_key (e.g. "model@q4_k_m")
                quant = model_key.split('@')[-1].lower() if '@' in model_key else ''
                # Check if quantization is in the list
                if not any(q in quant for q in quants_list):
                    continue
            
            # Filter: architectures (OR conjunction)
            if filter_args.get('arch'):
                arch_list = [a.strip().lower() for a in filter_args['arch'].split(',')]
                if metadata['architecture'].lower() not in arch_list:
                    continue
            
            # Filter: parameter sizes (OR conjunction)
            if filter_args.get('params'):
                params_list = [p.strip().upper() for p in filter_args['params'].split(',')]
                if metadata['params_size'].upper() not in params_list:
                    continue
            
            # Filter: minimum context length
            if filter_args.get('min_context'):
                if metadata['max_context_length'] < filter_args['min_context']:
                    continue
            
            # Filter: maximum file size
            if filter_args.get('max_size'):
                if metadata['model_size_gb'] > filter_args['max_size']:
                    continue
            
            filtered.append(model_key)
        
        logger.info(f"✔️ After filtering: {len(filtered)}/{len(models)} models remaining")
        return filtered



class LMStudioBenchmark:
    """Main benchmark class"""
    
    @staticmethod
    def get_lmstudio_version() -> Optional[str]:
        """Fetches LM Studio version"""
        try:
            result = subprocess.run(['lms', 'version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Format: "lms - LM Studio CLI - v0.0.47\n..."
                for line in result.stdout.split('\n'):
                    if 'v' in line and ('0.0' in line or line.strip().startswith('v')):
                        # Try to extract version (e.g. "v0.0.47")
                        parts = line.split()
                        for part in parts:
                            if part.startswith('v') and len(part) > 1:
                                return part
                # Fallback: first line
                first_line = result.stdout.strip().split('\n')[0]
                return first_line if first_line else None
        except Exception as e:
            logger.debug(f"Error fetching LM Studio version: {e}")
        return None
    
    @staticmethod
    def get_nvidia_driver_version() -> Optional[str]:
        """Fetches NVIDIA driver version"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                return version if version else None
        except Exception:
            logger.debug("NVIDIA driver not available")
        return None
    
    @staticmethod
    def get_rocm_driver_version() -> Optional[str]:
        """Fetches AMD ROCm/driver version"""
        try:
            # Try rocm-smi in various paths
            rocm_paths = ['/usr/bin/rocm-smi', '/usr/local/bin/rocm-smi']
            rocm_paths.extend(glob.glob('/opt/rocm-*/bin/rocm-smi'))
            
            for rocm_smi in rocm_paths:
                try:
                    result = subprocess.run(
                        [rocm_smi, '--version'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        # Format: "ROCM-SMI version: 4.0.0+1a5c7ec\nROCM-SMI-LIB version: 7.8.0"
                        output = result.stdout.strip()
                        lines = output.split('\n')
                        if lines and 'version' in lines[0].lower():
                        # Extract version from first line
                            parts = lines[0].split(':')
                            if len(parts) > 1:
                                return parts[1].strip()
                        return output if output else None
                except Exception:
                    continue
        except Exception:
            logger.debug("ROCm driver not available")
        return None
    
    @staticmethod
    def get_intel_driver_version() -> Optional[str]:
        """Fetches Intel GPU driver version"""
        try:
            result = subprocess.run(
                ['intel_gpu_top', '--help'],
                capture_output=True, text=True, timeout=5
            )
            # Intel_gpu_top shows version in help text
            for line in result.stdout.split('\n'):
                if 'version' in line.lower():
                    # Try to extract version
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'version' in part.lower() and i + 1 < len(parts):
                            return parts[i + 1]
                    return line.strip()
        except Exception:
            logger.debug("Intel GPU driver not available")
        return None
    
    @staticmethod
    def get_os_info() -> Tuple[Optional[str], Optional[str]]:
        """Fetches operating system and kernel version"""
        try:
            import platform
            os_system = platform.system()  # Linux, Windows, Darwin
            
            # On Linux: fetch distribution name
            if os_system == "Linux":
                try:
                    import distro
                    os_name = distro.name()  # e.g. "Pop!_OS", "Ubuntu", "Arch Linux"
                    os_version = distro.version()  # e.g. "22.04"
                    return os_name, os_version
                except Exception:
                    # Fallback to platform.release() (kernel version)
                    os_name = os_system
                    os_version = platform.release()
                    return os_name, os_version
            else:
                # Windows/macOS: use system() and release()
                os_name = os_system
                os_version = platform.release()
                return os_name, os_version
        except Exception:
            logger.debug("OS info not available")
        return None, None
    
    @staticmethod
    def get_cpu_model() -> Optional[str]:
        """Fetches CPU model"""
        try:
            import cpuinfo
            cpu = cpuinfo.get_cpu_info()
            brand = cpu.get('brand_raw', '')
            return brand if brand else None
        except Exception:
            logger.debug("CPU info not available")
        return None
    
    @staticmethod
    def get_python_version() -> Optional[str]:
        """Fetches Python version"""
        try:
            import sys
            return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        except Exception:
            logger.debug("Python version not available")
        return None
    
    def __init__(self, num_runs: int = NUM_MEASUREMENT_RUNS, context_length: int = CONTEXT_LENGTH, prompt: str = STANDARD_PROMPT, model_limit: Optional[int] = None, filter_args: Optional[Dict] = None, compare_with: Optional[str] = None, rank_by: str = 'speed', use_cache: bool = True, enable_profiling: bool = False, max_temp: Optional[float] = None, max_power: Optional[float] = None, use_gtt: bool = True, inference_overrides: Optional[Dict] = None, load_params: Optional[Dict] = None):
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
        self._gtt_info = {}  # Stores GTT information from AMD GPU
        
        # Load parameters with defaults (performance tuning)
        self.load_params = dict(DEFAULT_LOAD_PARAMS)
        if load_params:
            for key, value in load_params.items():
                if value is not None:
                    self.load_params[key] = value
        
        # Cache is ALWAYS initialized (to save results)
        # use_cache only controls LOADING of cache hits
        self.cache = BenchmarkCache()
        self.params_hash = BenchmarkCache.compute_params_hash(
            prompt, context_length, OPTIMIZED_INFERENCE_PARAMS, self.load_params
        )
        
        # Hardware monitor for profiling
        self.hardware_monitor = HardwareMonitor(
            self.gpu_monitor.gpu_type or "Unknown",
            self.gpu_monitor.gpu_tool or "",
            enabled=enable_profiling
        )
        
        # Save CLI arguments for reports
        self.cli_args = {
            'runs': num_runs,
            'context': context_length,
            'prompt': prompt[:50] + '...' if len(prompt) > 50 else prompt,
            'limit': model_limit,
            'only_vision': self.filter_args.get('only_vision', False),
            'only_tools': self.filter_args.get('only_tools', False),
            'quants': self.filter_args.get('quants'),
            'arch': self.filter_args.get('arch'),
            'params': self.filter_args.get('params'),
            'include_models': self.filter_args.get('include_models'),
            'exclude_models': self.filter_args.get('exclude_models'),
            'compare_with': compare_with,
            'rank_by': rank_by,
            'retest': not use_cache,
            'enable_profiling': enable_profiling,
            'max_temp': max_temp,
            'max_power': max_power,
            'use_gtt': use_gtt,
        }
        
        # Capture version information once at benchmark start
        self.system_versions = {
            'lmstudio_version': self.get_lmstudio_version(),
            'nvidia_driver_version': self.get_nvidia_driver_version(),
            'rocm_driver_version': self.get_rocm_driver_version(),
            'intel_driver_version': self.get_intel_driver_version(),
        }
        
        # Debug logging for captured versions
        logger.info(f"📋 Captured version information:")
        for key, value in self.system_versions.items():
            logger.info(f"   • {key}: {value if value else 'N/A'}")
        
        # Capture system information once
        os_name, os_version = self.get_os_info()
        self.system_info = {
            'os_name': os_name,
            'os_version': os_version,
            'cpu_model': self.get_cpu_model(),
            'python_version': self.get_python_version(),
        }
        logger.info(f"💻 System information:")
        logger.info(f"   • OS: {self.system_info['os_name']} {self.system_info['os_version']}")
        logger.info(f"   • CPU: {self.system_info['cpu_model'] or 'N/A'}")
        logger.info(f"   • Python: {self.system_info['python_version']}")
        
        # Save inference parameters, apply overrides if any
        self.inference_params = OPTIMIZED_INFERENCE_PARAMS.copy()
        self.inference_overrides = inference_overrides or {}
        for k, v in (self.inference_overrides or {}).items():
            if v is not None and k in self.inference_params:
                self.inference_params[k] = v
        
        # Save hashes for reproducibility
        self.prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        # Hash with actual inference parameters (incl. overrides) AND load parameters
        self.params_hash = BenchmarkCache.compute_params_hash(
            prompt, context_length, self.inference_params, self.load_params
        )
        
        # Create results directory
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Load previous results if comparison desired
        if self.compare_with:
            self._load_previous_results()
    
    def _get_available_vram_gb(self) -> Optional[float]:
        """Determines available VRAM in GB (incl. GTT when enabled)"""
        try:
            gpu_type = self.gpu_monitor.gpu_type
            gpu_model = self.gpu_monitor.gpu_model or self.gpu_monitor.gpu_type
            gpu_tool = self.gpu_monitor.gpu_tool
            
            if not gpu_tool:
                return None
            
            if gpu_type == "NVIDIA":
                # nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
                result = subprocess.run(
                    [gpu_tool, '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    vram_mb = float(result.stdout.strip().split('\n')[0])
                    return vram_mb / 1024  # MB → GB
            
            elif gpu_type == "AMD":
                # rocm-smi --showmeminfo all (holt VRAM + GTT)
                result = subprocess.run(
                    [gpu_tool, '--showmeminfo', 'all'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    vram_total = vram_used = gtt_total = gtt_used = 0.0
                    
                    for line in result.stdout.split('\n'):
                        if 'VRAM Total Memory (B):' in line:
                            vram_total = float(line.split(':')[-1].strip()) / (1024**3)  # Bytes → GB
                        elif 'VRAM Total Used Memory (B):' in line:
                            vram_used = float(line.split(':')[-1].strip()) / (1024**3)
                        elif 'GTT Total Memory (B):' in line:
                            gtt_total = float(line.split(':')[-1].strip()) / (1024**3)
                        elif 'GTT Total Used Memory (B):' in line:
                            gtt_used = float(line.split(':')[-1].strip()) / (1024**3)
                    
                    vram_free = vram_total - vram_used
                    gtt_free = gtt_total - gtt_used
                    
                    # Save GTT info for later
                    self._gtt_info = {
                        'total': gtt_total,
                        'used': gtt_used,
                        'free': gtt_free
                    }
                    
                    if self.use_gtt and gtt_total > 0:
                        total_available = vram_free + gtt_free
                        logger.info(f"💾 Memory: {vram_free:.1f}GB VRAM + {gtt_free:.1f}GB GTT = {total_available:.1f}GB total")
                        return total_available
                    else:
                        logger.info(f"💾 Memory: {vram_free:.1f}GB VRAM (GTT deaktiviert)")
                        return vram_free
            
            elif gpu_type == "Intel":
                # intel_gpu_top has no direct VRAM query
                # Estimate based on typical Intel iGPU values
                return 8.0  # Conservative estimate for modern iGPUs
            
            return None
        
        except Exception as e:
            logger.debug(f"VRAM-Abfrage fehlgeschlagen: {e}")
            return None
    
    def _predict_optimal_offload(self, model_size_gb: float) -> float:
        """Calculates optimal GPU offload based on VRAM and model size"""
        try:
            available_vram = self._get_available_vram_gb()
            
            if available_vram is None:
                logger.debug("VRAM not available, using default levels")
                return 1.0  # Start with maximum offload
            
            # Estimated VRAM requirement
            # Factor 1.2 for overhead (weights + activations + KV cache)
            estimated_vram = model_size_gb * 1.2
            
            # Context length overhead (~2KB per token)
            estimated_vram += (self.context_length * CONTEXT_VRAM_FACTOR)
            
            # Available VRAM minus safety headroom
            safe_vram = available_vram - VRAM_SAFETY_HEADROOM_GB
            
            if safe_vram <= 0:
                logger.warning(f"⚠️ Insufficient VRAM available: {available_vram:.1f}GB")
                return 0.3  # Minimum offload
            
            # Calculate optimal offload factor
            if estimated_vram <= safe_vram:
                optimal_offload = 1.0  # Fully on GPU
            else:
                optimal_offload = safe_vram / estimated_vram
                optimal_offload = max(0.3, min(1.0, optimal_offload))  # Clamp between 0.3 and 1.0
            
            logger.info(f"📊 VRAM prediction: {available_vram:.1f}GB available, "
                       f"{estimated_vram:.1f}GB estimated → offload {optimal_offload:.2f}")
            
            return round(optimal_offload, 1)  # Round to 1 decimal place
        
        except Exception as e:
            logger.debug(f"Offload prediction failed: {e}")
            return 1.0
    
    def _get_cached_optimal_offload(self, model_key: str, model_size_gb: float) -> Optional[float]:
        """Fetches optimal offload from cache for similar models"""
        if not self.cache:
            return None
        
        try:
            # Search for similar models (same architecture + similar size)
            metadata = ModelDiscovery.get_model_metadata(model_key)
            architecture = metadata.get('architecture', 'unknown')
            
            if architecture == 'unknown':
                return None
            
            # Fetch all cached results with same architecture
            conn = sqlite3.connect(self.cache.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT gpu_offload, model_size_gb 
                FROM benchmark_results 
                WHERE architecture = ? 
                AND model_size_gb BETWEEN ? AND ?
                AND gpu_offload > 0
                ORDER BY timestamp DESC
                LIMIT 5
            """, (architecture, model_size_gb * 0.8, model_size_gb * 1.2))
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                # Average of successful offload levels
                offloads = [r[0] for r in results]
                avg_offload = sum(offloads) / len(offloads)
                logger.info(f"📚 Cache hit: Using avg offload {avg_offload:.2f} "
                           f"for {architecture} (~{model_size_gb:.1f}GB)")
                return round(avg_offload, 1)
            
            return None
        
        except Exception as e:
            logger.debug(f"Cache-Lookup fehlgeschlagen: {e}")
            return None
    
    def _get_smart_offload_levels(self, model_key: str, model_size_gb: float) -> List[float]:
        """Generates intelligent GPU offload levels based on prediction and cache"""
        # 1. Try cache lookup
        cached_offload = self._get_cached_optimal_offload(model_key, model_size_gb)
        if cached_offload:
            # Use cached value + neighbors for robustness
            levels = [cached_offload]
            if cached_offload > 0.5:
                levels.append(round(cached_offload - 0.2, 1))
            if cached_offload < 0.9:
                levels.append(round(cached_offload + 0.1, 1))
            return sorted(set(levels), reverse=True)  # Deduplicate and sort descending
        
        # 2. VRAM-based prediction
        predicted_offload = self._predict_optimal_offload(model_size_gb)
        
        # 3. Generate binary-search-like levels around prediction
        levels = [predicted_offload]
        
        if predicted_offload >= 0.8:
            # Large VRAM → start high, then reduce
            levels.extend([1.0, 0.7, 0.5])
        elif predicted_offload >= 0.5:
            # Medium VRAM → start at prediction
            levels.extend([0.7, 0.5, 0.3])
        else:
            # Little VRAM → start low
            levels.extend([0.5, 0.3, 0.7])
        
        # Deduplicate and sort descending
        return sorted(set(levels), reverse=True)
    
    def _load_previous_results(self):
        """Loads previous benchmark results for comparison"""
        try:
            # Search for matching JSON file
            if not self.compare_with:
                return
            
            if self.compare_with.endswith('.json'):
                json_file = RESULTS_DIR / self.compare_with
            else:
                # Try to parse date (e.g. "20260104" or "latest")
                if self.compare_with.lower() == "latest":
                    # Find newest file
                    json_files = sorted(RESULTS_DIR.glob('benchmark_results_*.json'))
                    if not json_files:
                        logger.warning("⚠️ No previous benchmark files found")
                        return
                    json_file = json_files[-1]
                else:
                    json_file = RESULTS_DIR / f"benchmark_results_{self.compare_with}.json"
            
            if not json_file.exists():
                logger.warning(f"⚠️ File not found: {json_file}")
                return
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.previous_results = [BenchmarkResult(**item) for item in data]
            
            logger.info(
                f"✓ {len(self.previous_results)} previous results loaded from {json_file.name}"
            )
        except Exception as e:
            logger.error(f"❌ Error loading previous results: {e}")
    
    def _matches_filters(self, result: BenchmarkResult) -> bool:
        """Checks whether a BenchmarkResult satisfies the active filters"""
        # Vision-Filter
        if self.filter_args.get('only_vision') and not result.has_vision:
            return False
        
        # Tools-Filter
        if self.filter_args.get('only_tools') and not result.has_tools:
            return False
        
        # Quantisierung-Filter
        if self.filter_args.get('quants'):
            quants = [q.strip().lower() for q in self.filter_args['quants'].split(',')]
            if not any(q in result.quantization.lower() for q in quants):
                return False
        
        # Architektur-Filter
        if self.filter_args.get('arch'):
            archs = [a.strip().lower() for a in self.filter_args['arch'].split(',')]
            if not any(a in result.architecture.lower() for a in archs):
                return False
        
        # Parameter-Filter
        if self.filter_args.get('params'):
            params = [p.strip().upper() for p in self.filter_args['params'].split(',')]
            if result.params_size.upper() not in params:
                return False
        
        # Min Context-Filter
        if self.filter_args.get('min_context'):
            if result.max_context_length < self.filter_args['min_context']:
                return False
        
        # Max Size-Filter
        if self.filter_args.get('max_size'):
            if result.model_size_gb > self.filter_args['max_size']:
                return False
        
        # Include-Pattern (Regex)
        if self.filter_args.get('include_models'):
            import re
            try:
                pattern = re.compile(self.filter_args['include_models'], re.IGNORECASE)
                model_full = f"{result.model_name}@{result.quantization}"
                if not pattern.search(model_full):
                    return False
            except re.error:
                pass
        
        # Exclude-Pattern (Regex)
        if self.filter_args.get('exclude_models'):
            import re
            try:
                pattern = re.compile(self.filter_args['exclude_models'], re.IGNORECASE)
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
        
        # Search for exact match (same model + quantization)
        for prev in self.previous_results:
            if prev.model_name == current.model_name and prev.quantization == current.quantization:
                speed_delta = current.avg_tokens_per_sec - prev.avg_tokens_per_sec
                speed_delta_pct = (speed_delta / prev.avg_tokens_per_sec * 100) if prev.avg_tokens_per_sec > 0 else 0
                
                return {
                    'prev_speed': prev.avg_tokens_per_sec,
                    'current_speed': current.avg_tokens_per_sec,
                    'speed_delta': speed_delta,
                    'speed_delta_pct': speed_delta_pct,
                    'prev_timestamp': prev.timestamp
                }
        
        return None

    
    def benchmark_model(self, model_key: str) -> Optional[BenchmarkResult]:
        """Performs benchmark for a specific model"""
        logger.info(f"🎯 Starting benchmark for {model_key}")
        
        # Start timing
        benchmark_start_time = time.time()
        error_count = 0
        
        # Unload all other models first
        try:
            subprocess.run(
                ['lms', 'unload', '--all'],
                capture_output=True,
                text=True,
                timeout=30
            )
            logger.info("🧹 All models unloaded")
            time.sleep(1)  # Wait until memory is freed
        except Exception as e:
            logger.warning(f"⚠️ Error unloading all models: {e}")
        
        # Parse model name and quantization
        if '@' in model_key:
            model_name, quantization = model_key.split('@', 1)
        else:
            model_name = model_key
            quantization = "unknown"
        
        # Fetch model metadata for intelligent offload calculation
        metadata = ModelDiscovery.get_model_metadata(model_key)
        model_size_gb = metadata.get('model_size_gb', 0)
        
        # Generate intelligent offload levels
        smart_offload_levels = self._get_smart_offload_levels(model_key, model_size_gb)
        logger.info(f"🎯 Intelligente Offload-Levels: {smart_offload_levels}")
        
        # Track which offload level is used (for SDK models)
        # The SDK handles offload automatically, but we save the first level
        # that succeeds for cache purposes
        used_offload = smart_offload_levels[0] if smart_offload_levels else 1.0
        
        try:
            # Warmup
            logger.info(f"🔥 Warmup for {model_key}...")
            for _ in range(NUM_WARMUP_RUNS):
                warmup_result = self._run_inference(model_key)
                if not warmup_result:
                    logger.error(f"❌ Warmup for {model_key} failed")
                    return None
            
            # Start hardware profiling when enabled
            self.hardware_monitor.start()
            
            # Measurements
            logger.info(f"📊 Running {self.num_measurement_runs} measurements...")
            measurements = []
            vram_after = "N/A"  # Initialize default value
            for run in range(self.num_measurement_runs):
                vram_before = self.gpu_monitor.get_vram_usage()
                stats = self._run_inference(model_key)
                vram_after = self.gpu_monitor.get_vram_usage()
                
                if stats:
                    measurements.append(stats)
                    logger.info(f"⚡ Run {run+1}/{self.num_measurement_runs}: {stats['tokens_per_second']:.2f} tokens/s")
                else:
                    logger.warning(f"⚠️ Run {run+1}/{self.num_measurement_runs} fehlgeschlagen")
            
            # Stop hardware profiling and fetch statistics
            profiling_stats = self.hardware_monitor.stop()
            
            # Calculate averages (for reports/logs)
            if measurements:
                result = self._calculate_averages(
                    model_name,
                    quantization,
                    used_offload,  # Verwende vorhergesagten/gecachten Offload-Level
                    vram_after,
                    measurements,
                    model_key
                )
                
                # Add profiling data
                if self.enable_profiling:
                    result.temp_celsius_min = profiling_stats.get('temp_celsius_min')
                    result.temp_celsius_max = profiling_stats.get('temp_celsius_max')
                    result.temp_celsius_avg = profiling_stats.get('temp_celsius_avg')
                    result.power_watts_min = profiling_stats.get('power_watts_min')
                    result.power_watts_max = profiling_stats.get('power_watts_max')
                    result.power_watts_avg = profiling_stats.get('power_watts_avg')
                    
                    # Check limits
                    if self.max_temp and result.temp_celsius_max and result.temp_celsius_max > self.max_temp:
                        logger.warning(
                            f"⚠️ Max. temperature exceeded: "
                            f"{result.temp_celsius_max:.1f}°C > {self.max_temp}°C"
                        )
                    
                    if self.max_power and result.power_watts_max and result.power_watts_max > self.max_power:
                        logger.warning(
                            f"⚠️ Max. power exceeded: "
                            f"{result.power_watts_max:.1f}W > {self.max_power}W"
                        )
                
                # Calculate benchmark duration and populate new fields
                benchmark_end_time = time.time()
                result.benchmark_duration_seconds = round(benchmark_end_time - benchmark_start_time, 2)
                result.error_count = error_count
                result.model_key = model_key
                result.prompt_hash = self.prompt_hash
                result.params_hash = self.params_hash
                result.context_length = self.context_length
                result.os_name = self.system_info.get('os_name')
                result.os_version = self.system_info.get('os_version')
                result.cpu_model = self.system_info.get('cpu_model')
                result.python_version = self.system_info.get('python_version')
                result.inference_params_hash = hashlib.md5(json.dumps(self.inference_params, sort_keys=True).encode()).hexdigest()[:8]
                
                # Save EVERY individual run in cache (for statistical analysis)
                if self.cache:
                    for run_idx, measurement in enumerate(measurements):
                        # Calculate efficiency metrics for this run
                        tps_per_gb = measurement['tokens_per_second'] / result.model_size_gb if result.model_size_gb > 0 else 0.0
                        params_billions = float(result.params_size.replace('B', '')) if result.params_size.endswith('B') and result.params_size[:-1].replace('.', '', 1).isdigit() else 0.0
                        tps_per_billion = measurement['tokens_per_second'] / params_billions if params_billions > 0 else 0.0
                        
                        # Create a result object per run
                        run_result = BenchmarkResult(
                            model_name=model_name,
                            quantization=quantization,
                            gpu_type=result.gpu_type,
                            gpu_offload=used_offload,
                            vram_mb=vram_after,
                            avg_tokens_per_sec=measurement['tokens_per_second'],  # This run, not average
                            avg_ttft=measurement['time_to_first_token'],
                            avg_gen_time=measurement['generation_time'],
                            prompt_tokens=measurement['prompt_tokens'],
                            completion_tokens=measurement['completion_tokens'],
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
                            num_runs=1,  # This is 1 run
                            runs_averaged_from=len(measurements),  # From N runs
                            warmup_runs=NUM_WARMUP_RUNS,
                            run_index=run_idx,  # Index: 0, 1, 2
                            model_key=model_key,
                            prompt_hash=self.prompt_hash,
                            params_hash=self.params_hash,
                            context_length=self.context_length,
                            os_name=result.os_name,
                            os_version=result.os_version,
                            cpu_model=result.cpu_model,
                            python_version=result.python_version,
                            benchmark_duration_seconds=result.benchmark_duration_seconds / len(measurements),  # Anteilige Zeit
                            error_count=error_count,
                            inference_params_hash=result.inference_params_hash,
                            # Load-Config Parameter
                            n_gpu_layers=self.load_params.get('n_gpu_layers'),
                            n_batch=self.load_params.get('n_batch'),
                            n_threads=self.load_params.get('n_threads'),
                            flash_attention=self.load_params.get('flash_attention'),
                            rope_freq_base=self.load_params.get('rope_freq_base'),
                            rope_freq_scale=self.load_params.get('rope_freq_scale'),
                            use_mmap=self.load_params.get('use_mmap'),
                            use_mlock=self.load_params.get('use_mlock'),
                            kv_cache_quant=self.load_params.get('kv_cache_quant')
                        )
                        
                        # Save individual run
                        self.cache.save_result(run_result, model_key, self.params_hash, 
                                              self.prompt, self.context_length)
                
                logger.info(f"✓ {model_key}: {result.avg_tokens_per_sec:.2f} tokens/s (Duration: {result.benchmark_duration_seconds}s)")
                return result
            else:
                logger.error(f"❌ No successful measurements for {model_key}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error benchmarking {model_key}: {e}")
            error_count += 1
            return None
    
    def _load_model(self, model_key: str, gpu_offload: float) -> bool:
        """Loads a model into memory"""
        try:
            cmd = [
                'lms', 'load', model_key,
                '--gpu', str(gpu_offload),
                '--context-length', str(CONTEXT_LENGTH)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"❌ Error loading {model_key}: {e}")
            return False
    
    def _unload_model(self, model_key: str):
        """Unloads a model from memory"""
        try:
            subprocess.run(
                ['lms', 'unload', model_key],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"⚠️ Error unloading {model_key}: {e}")
    
    def _run_inference(self, model_key: str) -> Optional[Dict]:
        """Performs inference and returns stats"""
        try:
            import lmstudio as lms
            
            # Load model via SDK with load parameters
            load_config_params: Dict[str, Any] = {
                'context_length': self.context_length,
            }
            
            # Flash attention (when supported by SDK) - bool type
            flash_attn = self.load_params.get('flash_attention')
            if flash_attn is not None and isinstance(flash_attn, bool):
                load_config_params['flash_attention'] = flash_attn
            
            # Memory Mapping (try_mmap Parameter) - bool type
            use_mmap = self.load_params.get('use_mmap')
            if use_mmap is not None and isinstance(use_mmap, bool):
                load_config_params['try_mmap'] = use_mmap
            
            # KV-Cache quantization (both K and V types) - string literal type
            kv_quant = self.load_params.get('kv_cache_quant')
            if kv_quant and isinstance(kv_quant, str) and kv_quant in ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1']:
                load_config_params['llama_k_cache_quantization_type'] = kv_quant
                load_config_params['llama_v_cache_quantization_type'] = kv_quant
            
            # Note: n_gpu_layers, n_batch, n_threads, rope_freq_base, rope_freq_scale, use_mlock
            # are stored in BenchmarkResult but not directly available as SDK parameters.
            # These parameters affect performance at the llama.cpp level and are set via
            # the CLI or model config, not via the Python SDK.
            
            # INFO-level logging for load config (complete overview)
            logger.info(f"⚙️ Load Config: context={self.context_length}, "
                       f"n_gpu_layers={self.load_params.get('n_gpu_layers')}, "
                       f"n_batch={self.load_params.get('n_batch')}, "
                       f"n_threads={self.load_params.get('n_threads')}, "
                       f"flash_attention={self.load_params.get('flash_attention')}, "
                       f"rope_freq_base={self.load_params.get('rope_freq_base')}, "
                       f"rope_freq_scale={self.load_params.get('rope_freq_scale')}, "
                       f"use_mmap={self.load_params.get('use_mmap')}, "
                       f"use_mlock={self.load_params.get('use_mlock')}, "
                       f"kv_cache_quant={kv_quant}")
            
            model = lms.llm(
                model_key,
                config=lms.LlmLoadModelConfig(**load_config_params)
            )
            
            # Create prediction configuration (with overrides if any)
            prediction_config = lms.LlmPredictionConfig(
                temperature=self.inference_params['temperature'],
                top_k_sampling=self.inference_params['top_k_sampling'],
                top_p_sampling=self.inference_params['top_p_sampling'],
                min_p_sampling=self.inference_params['min_p_sampling'],
                repeat_penalty=self.inference_params['repeat_penalty'],
                max_tokens=self.inference_params['max_tokens']
            )
            
            # INFO-level logging for prediction config (complete overview)
            logger.info(f"⚙️ Inference Config: temp={prediction_config.temperature}, "
                       f"top_k={prediction_config.top_k_sampling}, top_p={prediction_config.top_p_sampling}, "
                       f"min_p={prediction_config.min_p_sampling}, repeat_penalty={prediction_config.repeat_penalty}, "
                       f"max_tokens={prediction_config.max_tokens}")
            
            # Run inference with optimized parameters
            start_time = time.time()
            result = model.respond(self.prompt, config=prediction_config)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Extract stats
            stats = result.stats
            
            tokens_per_sec = 0.0
            if generation_time > 0 and stats.predicted_tokens_count:
                tokens_per_sec = float(stats.predicted_tokens_count) / generation_time
            
            return {
                'tokens_per_second': tokens_per_sec,
                'time_to_first_token': stats.time_to_first_token_sec,
                'generation_time': generation_time,
                'prompt_tokens': stats.prompt_tokens_count or 0,
                'completion_tokens': stats.predicted_tokens_count or 0
            }
        
        except Exception as e:
            logger.error(f"❌ Error during inference with {model_key}: {e}")
            return None
    
    def _calculate_averages(
        self,
        model_name: str,
        quantization: str,
        gpu_offload: float,
        vram_mb: str,
        measurements: List[Dict],
        model_key: str
    ) -> BenchmarkResult:
        """Calculates averages from measurements"""
        avg_tokens_per_sec = sum(m['tokens_per_second'] for m in measurements) / len(measurements)
        avg_ttft = sum(m['time_to_first_token'] for m in measurements) / len(measurements)
        avg_gen_time = sum(m['generation_time'] for m in measurements) / len(measurements)
        prompt_tokens = measurements[0]['prompt_tokens']
        completion_tokens = int(
            sum(m['completion_tokens'] for m in measurements) / len(measurements)
        )
        
        # Fetch metadata
        metadata = ModelDiscovery.get_model_metadata(model_key)
        
        # Calculate efficiency metrics
        model_size_gb = metadata.get('model_size_gb', 0.0)
        params_size_str = metadata.get('params_size', 'unknown')
        
        # tokens/s per GB
        tokens_per_sec_per_gb = (
            round(avg_tokens_per_sec / model_size_gb, 2) if model_size_gb > 0 else 0.0
        )
        
        # tokens/s per billion parameters
        # Extract number from params_size (e.g. "7B" -> 7.0, "8.3B" -> 8.3)
        try:
            params_billion = float(params_size_str.upper().replace('B', '').strip())
            tokens_per_sec_per_billion_params = round(avg_tokens_per_sec / params_billion, 2)
        except (ValueError, AttributeError):
            tokens_per_sec_per_billion_params = 0.0
        
        result = BenchmarkResult(
            model_name=model_name,
            quantization=quantization,
            gpu_type=self.gpu_monitor.gpu_model or self.gpu_monitor.gpu_type or "Unknown",
            gpu_offload=gpu_offload,
            vram_mb=vram_mb,
            avg_tokens_per_sec=round(avg_tokens_per_sec, 2),
            avg_ttft=round(avg_ttft, 3),
            avg_gen_time=round(avg_gen_time, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            params_size=metadata.get('params_size', 'unknown'),
            architecture=metadata.get('architecture', 'unknown'),
            max_context_length=metadata.get('max_context_length', 0),
            model_size_gb=model_size_gb,
            has_vision=metadata.get('has_vision', False),
            has_tools=metadata.get('has_tools', False),
            tokens_per_sec_per_gb=tokens_per_sec_per_gb,
            tokens_per_sec_per_billion_params=tokens_per_sec_per_billion_params,
            # GTT information (AMD GPUs)
            gtt_enabled=self.use_gtt if self._gtt_info else None,
            gtt_total_gb=round(self._gtt_info.get('total', 0), 2) if self._gtt_info else None,
            gtt_used_gb=round(self._gtt_info.get('used', 0), 2) if self._gtt_info else None,
            # Inference parameters (from benchmark configuration)
            temperature=self.inference_params.get('temperature'),
            top_k_sampling=self.inference_params.get('top_k_sampling'),
            top_p_sampling=self.inference_params.get('top_p_sampling'),
            min_p_sampling=self.inference_params.get('min_p_sampling'),
            repeat_penalty=self.inference_params.get('repeat_penalty'),
            max_tokens=self.inference_params.get('max_tokens'),
            # Information about number of runs
            num_runs=self.num_measurement_runs,
            runs_averaged_from=len(measurements),
            warmup_runs=NUM_WARMUP_RUNS,
            # Version information (captured at benchmark start)
            lmstudio_version=self.system_versions.get('lmstudio_version'),
            nvidia_driver_version=self.system_versions.get('nvidia_driver_version'),
            rocm_driver_version=self.system_versions.get('rocm_driver_version'),
            intel_driver_version=self.system_versions.get('intel_driver_version')
        )
        
        # Calculate delta to previous benchmark if available
        delta = self._calculate_delta(result)
        if delta:
            result.speed_delta_pct = delta['speed_delta_pct']
            result.prev_timestamp = delta['prev_timestamp']
        
        return result
    
    def run_all_benchmarks(self):
        """Performs benchmarks for all available models"""
        # Ensure server is running
        if not LMStudioServerManager.ensure_server_running():
            logger.error("❌ Server could not be started, aborting")
            return
        
        # Initialize metadata cache early
        ModelDiscovery._get_metadata_cache()
        
        # Fetch all models
        models = ModelDiscovery.get_installed_models()
        if not models:
            logger.error("❌ No models found")
            return
        
        # Apply filters
        models = ModelDiscovery.filter_models(models, self.filter_args)
        if not models:
            logger.error("❌ No models remaining after filtering")
            return
        
        # Separate lists for newly tested vs. cached models
        newly_tested_models = []  # ONLY for reports
        
        # Check cache and show stats
        if self.cache and self.use_cache:
            cached_models = []
            new_models = []
            
            for model_key in models:
                cached = self.cache.get_cached_result(model_key, self.params_hash)
                if cached:
                    cached_models.append((model_key, cached))
                else:
                    new_models.append(model_key)
            
            # Apply limit ONLY to new models (not to cached ones)
            if self.model_limit and self.model_limit < len(new_models):
                logger.info(
                    f"⚙️ Model limit set: Testing max. {self.model_limit} new models "
                    f"(+ {len(cached_models)} cached)"
                )
                new_models = new_models[:self.model_limit]
            elif self.model_limit:
                logger.info(
                    f"⚙️ Model limit: {len(new_models)} new models + "
                    f"{len(cached_models)} cached = {len(new_models) + len(cached_models)} total"
                )
            
            if cached_models:
                logger.info("")
                logger.info("📦 === Cached models ===")
                logger.info(
                    f"💾 {len(cached_models)} models already tested (loading from cache):"
                )
                for model_key, cached in cached_models[:10]:  # Show max. 10
                    date_part = (
                        cached.timestamp.split('T')[0]
                        if 'T' in cached.timestamp else cached.timestamp[:10]
                    )
                    logger.info(
                        f"  • {model_key}: {cached.avg_tokens_per_sec:.2f} tok/s "
                        f"(last: {date_part})"
                    )
                if len(cached_models) > 10:
                    logger.info(f"  ... and {len(cached_models) - 10} more")
                logger.info("")
                
                # Load cached results into self.results (for internal use)
                # BUT NOT into newly_tested_models (for reports)
                for model_key, cached in cached_models:
                    self.results.append(cached)
            
            if new_models:
                logger.info(f"🚀 Starting benchmark for {len(new_models)} new models...")
                models = new_models
            else:
                logger.info("✅ All models already cached - no new tests needed")
                logger.info("📝 Reports skipped (no new data)")
                return
        else:
            # Apply limit when set (no cache)
            if self.model_limit and self.model_limit < len(models):
                logger.info(
                    f"⚙️ Model limit set: Testing only first {self.model_limit} "
                    f"of {len(models)} models"
                )
                models = models[:self.model_limit]
            logger.info(f"🚀 Starting benchmark for {len(models)} models...")
        
        # Benchmark for each model
        for model_key in tqdm(models, desc="Benchmarking models"):
            result = self.benchmark_model(model_key)
            if result:
                self.results.append(result)
                newly_tested_models.append(result)  # ONLY newly tested!

        
        # Export ONLY newly tested models (not cached ones!)
        if newly_tested_models:
            logger.info(
                f"📊 Exporting reports for {len(newly_tested_models)} newly tested models..."
            )
            self._export_results_to_files(newly_tested_models)
        else:
            logger.warning("⚠️ No new models tested - no reports generated")
        
        # Unload all models at the end (cleanup)
        try:
            subprocess.run(
                ['lms', 'unload', '--all'],
                capture_output=True,
                text=True,
                timeout=30
            )
            logger.info("🧹 All models unloaded (cleanup)")
            time.sleep(1)  # Wait for memory to be freed
        except Exception as e:
            logger.warning(f"⚠️ Error unloading all models: {e}")
        
        logger.info(
            f"✅ Benchmark complete. "
            f"{len(newly_tested_models)}/{len(models)} models successfully tested"
        )
    
    def _analyze_best_quantizations(self) -> Dict[str, Dict]:
        """Analyzes best quantization per model by various criteria"""
        best_by_model = {}
        
        for result in self.results:
            model_key = result.model_name
            
            if model_key not in best_by_model:
                best_by_model[model_key] = {
                    'best_speed': None,
                    'best_efficiency': None,
                    'best_ttft': None,
                    'all_quantizations': []
                }
            
            # Save all quantizations
            best_by_model[model_key]['all_quantizations'].append(result)
            
            # Best speed
            if best_by_model[model_key]['best_speed'] is None or \
               result.avg_tokens_per_sec > best_by_model[model_key]['best_speed'].avg_tokens_per_sec:
                best_by_model[model_key]['best_speed'] = result
            
            # Best efficiency (tokens/s per GB)
            if best_by_model[model_key]['best_efficiency'] is None or \
               result.tokens_per_sec_per_gb > best_by_model[model_key]['best_efficiency'].tokens_per_sec_per_gb:
                best_by_model[model_key]['best_efficiency'] = result
            
            # Best TTFT (time to first token - low = good)
            if best_by_model[model_key]['best_ttft'] is None or \
               result.avg_ttft < best_by_model[model_key]['best_ttft'].avg_ttft:
                best_by_model[model_key]['best_ttft'] = result
        
        return best_by_model
    
    def sort_results(self, rank_by: str = 'speed') -> List[BenchmarkResult]:
        """Sorts results by various criteria"""
        if rank_by == 'speed':
            return sorted(self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True)
        elif rank_by == 'efficiency':
            return sorted(self.results, key=lambda x: x.tokens_per_sec_per_gb, reverse=True)
        elif rank_by == 'ttft':
            return sorted(self.results, key=lambda x: x.avg_ttft, reverse=False)  # Low = good
        elif rank_by == 'vram':
            # Parse VRAM value (e.g. "2048 MB" -> 2048)
            def get_vram_mb(result):
                try:
                    return (
                        float(result.vram_mb.split()[0])
                        if isinstance(result.vram_mb, str) else float(result.vram_mb)
                    )
                except:
                    return 999999
            return sorted(self.results, key=get_vram_mb, reverse=False)  # Low = better
        else:
            return sorted(self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True)  # Default
    
    def calculate_percentile_stats(self) -> Dict[str, Dict]:
        """Calculates P50, P95, P99 statistics for benchmark metrics"""
        if not self.results or len(self.results) < 3:
            return {}
        
        speeds = [r.avg_tokens_per_sec for r in self.results if r.avg_tokens_per_sec > 0]
        ttfts = [r.avg_ttft for r in self.results if r.avg_ttft > 0]
        vram_values = []
        for r in self.results:
            try:
                vram_mb = float(r.vram_mb.split()[0]) if isinstance(r.vram_mb, str) else float(r.vram_mb)
                vram_values.append(vram_mb)
            except:
                pass
        
        stats = {}
        
        # Speed statistics
        if speeds:
            if len(speeds) >= 3:
                quantile_data = quantiles(speeds, n=100, method='inclusive')
                stats['speed'] = {
                    'avg': round(mean(speeds), 2),
                    'median': round(median(speeds), 2),
                    'p95': round(quantile_data[94], 2),
                    'p99': round(quantile_data[98], 2),
                    'min': round(min(speeds), 2),
                    'max': round(max(speeds), 2)
                }
        
        # TTFT statistics
        if ttfts:
            if len(ttfts) >= 3:
                quantile_data = quantiles(ttfts, n=100, method='inclusive')
                stats['ttft'] = {
                    'avg': round(mean(ttfts), 3),
                    'median': round(median(ttfts), 3),
                    'p95': round(quantile_data[94], 3),
                    'p99': round(quantile_data[98], 3),
                    'min': round(min(ttfts), 3),
                    'max': round(max(ttfts), 3)
                }
        
        # VRAM statistics
        if vram_values:
            if len(vram_values) >= 3:
                quantile_data = quantiles(vram_values, n=100, method='inclusive')
                stats['vram'] = {
                    'avg': round(mean(vram_values), 0),
                    'median': round(median(vram_values), 0),
                    'p95': round(quantile_data[94], 0),
                    'p99': round(quantile_data[98], 0),
                    'min': round(min(vram_values), 0),
                    'max': round(max(vram_values), 0)
                }
        
        return stats
    
    def generate_quantization_comparison(self) -> Dict[str, Dict]:
        """Generates comparison table Q4 vs Q5 vs Q6 per model"""
        if not self.results:
            return {}
        
        # Group results by model and quantization
        model_quants = {}
        for result in self.results:
            if result.model_name not in model_quants:
                model_quants[result.model_name] = {}
            
            # Extract quantization level (e.g. "q4_k_m" -> "q4")
            quant_level = result.quantization.split('_')[0].lower()
            if quant_level not in model_quants[result.model_name]:
                model_quants[result.model_name][quant_level] = result
            else:
                # Take best performance (highest tokens/s) if multiple quantizations
                if result.avg_tokens_per_sec > model_quants[result.model_name][quant_level].avg_tokens_per_sec:
                    model_quants[result.model_name][quant_level] = result
        
        # Create comparison table
        comparison = {}
        for model, quants in sorted(model_quants.items()):
            comparison[model] = {
                'q4': None,
                'q5': None, 
                'q6': None,
                'q8': None
            }
            for q_level in ['q4', 'q5', 'q6', 'q8']:
                if q_level in quants:
                    r = quants[q_level]
                    comparison[model][q_level] = {
                        'speed': round(r.avg_tokens_per_sec, 2),
                        'efficiency': round(r.tokens_per_sec_per_gb, 2),
                        'vram_mb': r.vram_mb,
                        'ttft': round(r.avg_ttft * 1000, 1)
                    }
        
        return comparison
    
    def _generate_best_practices(self) -> List[str]:
        """Generates best-practice recommendations based on hardware and benchmark results"""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # GPU type detection
        gpu_model = self.gpu_monitor.gpu_model or self.gpu_monitor.gpu_type or "Unknown"
        
        # Best models by criteria
        best_speed = max(self.results, key=lambda x: x.avg_tokens_per_sec)
        best_efficiency = max(self.results, key=lambda x: x.tokens_per_sec_per_gb)
        best_ttft = min(
            self.results, key=lambda x: x.avg_ttft if x.avg_ttft > 0 else float('inf')
        )
        
        # Find best balance (speed + efficiency)
        best_balance = max(
            self.results,
            key=lambda x: x.avg_tokens_per_sec * 0.6 + x.tokens_per_sec_per_gb * 0.4
        )
        
        # Hardware-specific recommendations
        recommendations.append(f"🖥️  Hardware: {gpu_model} detected")
        recommendations.append("")
        
        # Top recommendation for speed
        recommendations.append(f"⚡ Fastest model:")
        recommendations.append(f"   → {best_speed.model_name} ({best_speed.quantization})")
        recommendations.append(f"   → {best_speed.avg_tokens_per_sec:.2f} tokens/s")
        recommendations.append("")
        
        # Top recommendation for efficiency
        recommendations.append(f"💎 Most efficient model (tokens/s per GB):")
        recommendations.append(f"   → {best_efficiency.model_name} ({best_efficiency.quantization})")
        recommendations.append(f"   → {best_efficiency.tokens_per_sec_per_gb:.2f} tokens/s/GB")
        recommendations.append(f"   → Size: {best_efficiency.model_size_gb:.2f} GB")
        recommendations.append("")
        
        # Top recommendation for TTFT
        recommendations.append(f"🚀 Fastest response time (TTFT):")
        recommendations.append(f"   → {best_ttft.model_name} ({best_ttft.quantization})")
        recommendations.append(f"   → {best_ttft.avg_ttft*1000:.0f} ms to first token")
        recommendations.append("")
        
        # Best balance recommendation
        recommendations.append(f"⚖️  Best balance (speed + efficiency):")
        recommendations.append(f"   → {best_balance.model_name} ({best_balance.quantization})")
        recommendations.append(
            f"   → {best_balance.avg_tokens_per_sec:.2f} tokens/s, "
            f"{best_balance.model_size_gb:.2f} GB"
        )
        recommendations.append("")
        
        # Quantization recommendations
        recommendations.append(f"📊 Quantization tips:")
        q4_models = [r for r in self.results if 'q4' in r.quantization.lower()]
        q6_models = [r for r in self.results if 'q6' in r.quantization.lower()]
        
        if q4_models and q6_models:
            avg_q4_speed = sum(r.avg_tokens_per_sec for r in q4_models) / len(q4_models)
            avg_q6_speed = sum(r.avg_tokens_per_sec for r in q6_models) / len(q6_models)
            speed_diff = ((avg_q4_speed - avg_q6_speed) / avg_q6_speed) * 100
            
            recommendations.append(
                f"   → Q4 is on average {abs(speed_diff):.0f}% "
                f"{'faster' if speed_diff > 0 else 'slower'} than Q6"
            )
            recommendations.append(
                f"   → Q4: faster, less quality | Q6: slower, better quality"
            )
        
        recommendations.append("")
        
        # VRAM-based recommendations
        vram_info = []
        for result in sorted(self.results, key=lambda x: x.model_size_gb)[:3]:
            if result.model_size_gb <= 4:
                vram_info.append(f"   → <4 GB VRAM: {result.model_name} ({result.quantization})")
        for result in sorted(self.results, key=lambda x: x.model_size_gb):
            if 4 < result.model_size_gb <= 8:
                vram_info.append(f"   → 4-8 GB VRAM: {result.model_name} ({result.quantization})")
                break
        for result in sorted(self.results, key=lambda x: x.model_size_gb):
            if 8 < result.model_size_gb <= 12:
                vram_info.append(f"   → 8-12 GB VRAM: {result.model_name} ({result.quantization})")
                break
        
        if vram_info:
            recommendations.append(f"🎯 VRAM recommendations:")
            recommendations.extend(vram_info[:3])  # Max 3 recommendations
        
        return recommendations
    
    def load_all_historical_data(self) -> Dict[str, List[Dict]]:
        """Loads all historical benchmark results and groups by model+quantization"""
        trends = {}
        results_dir = RESULTS_DIR
        
        # Load all JSON files
        for json_file in sorted(results_dir.glob("benchmark_results_*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    
                for item in data:
                    key = f"{item['model_name']}@{item['quantization']}"
                    if key not in trends:
                        trends[key] = []
                    
                    # Extract date from timestamp
                    timestamp_str = item.get('timestamp', '2026-01-01 00:00:00')
                    trends[key].append({
                        'timestamp': timestamp_str,
                        'speed': item['avg_tokens_per_sec'],
                        'ttft': item['avg_ttft'],
                        'vram': item.get('vram_mb', 0)
                    })
            except Exception as e:
                logger.debug(f"Error loading {json_file}: {e}")
        
        return trends
    
    def generate_trend_chart(self) -> Optional[str]:
        """Generates Plotly line chart for performance trends over time"""
        if not PLOTLY_AVAILABLE or not self.previous_results or go is None:
            return None
        
        try:
            trends = self.load_all_historical_data()
            if not trends:
                return None
            
            # Create line chart with trends
            fig = go.Figure()
            
            # Colors for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_idx = 0
            
            for key, history in sorted(trends.items()):
                if len(history) < 2:  # Only trends with at least 2 data points
                    continue
                
                # Sort by timestamp
                history_sorted = sorted(history, key=lambda x: x['timestamp'])
                
                timestamps = [h['timestamp'] for h in history_sorted]
                speeds = [h['speed'] for h in history_sorted]
                
                model_name = key.split('@')[0].split('/')[-1][:15]
                quantization = key.split('@')[1][:6]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=speeds,
                    mode='lines+markers',
                    name=f"{model_name} ({quantization})",
                    line=dict(color=colors[color_idx % len(colors)]),
                    marker=dict(size=6)
                ))
                
                color_idx += 1
            
            fig.update_layout(
                title="Performance trends over time",
                xaxis_title="Date",
                yaxis_title="Tokens/s",
                hovermode='x unified',
                height=600,
                template='plotly_white'
            )
            
            # Return Plotly JSON
            return json.dumps({
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            })
        
        except Exception as e:
            logger.debug(f"Error creating trend chart: {e}")
            return None
    
    def _export_results_to_files(self, results_to_export):
        """Exports given results as JSON, CSV, PDF and HTML"""
        if not results_to_export:
            logger.warning("⚠️ No results to export")
            return
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # JSON Export
        json_file = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results_to_export], f, indent=2, ensure_ascii=False)
        logger.info(f"📄 JSON results saved: {json_file}")
        
        # CSV Export
        csv_file = RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results_to_export:
                writer = csv.DictWriter(f, fieldnames=asdict(results_to_export[0]).keys())
                writer.writeheader()
                for result in results_to_export:
                    writer.writerow(asdict(result))
        logger.info(f"📊 CSV results saved: {csv_file}")
        
        # PDF Export
        self._export_pdf(timestamp, results_to_export)
        
        # HTML Export (optional)
        if PLOTLY_AVAILABLE:
            self._export_html(timestamp, results_to_export)
    
    def export_results(self):
        """Legacy wrapper for direct calls (e.g. --export-only)"""
        self._export_results_to_files(self.results)
    
    def _export_pdf(self, timestamp: str, results_to_export):
        """Exports given benchmark results as PDF report"""
        try:
            # Use results_to_export instead of self.results
            results = results_to_export
            
            pdf_file = RESULTS_DIR / f"benchmark_results_{timestamp}.pdf"
            
            # Create PDF document with landscape format
            doc = SimpleDocTemplate(
                str(pdf_file),
                pagesize=landscape(A4),
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch,
                title="LM Studio Benchmark Results"
            )
            
            # Container for PDF elements
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=20,
                alignment=1  # Center
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2d5aa8'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Title
            elements.append(Paragraph("LM Studio Model Benchmark Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Generated at
            timestamp_display = time.strftime('%d.%m.%Y %H:%M:%S')
            elements.append(
                Paragraph(f"<font size=10>Generated: {timestamp_display}</font>", styles['Normal'])
            )
            elements.append(Spacer(1, 20))
            
            # Summary
            elements.append(Paragraph("Benchmark Summary", heading_style))
            
            # Calculate statistics
            vision_count = sum(1 for r in results if r.has_vision)
            tools_count = sum(1 for r in results if r.has_tools)
            avg_size_gb = sum(r.model_size_gb for r in results) / len(results) if self.results else 0
            avg_tokens_per_sec = (
                sum(r.avg_tokens_per_sec for r in results) / len(results) if self.results else 0
            )
            
            summary_data = [
                ['Metric', 'Value'],
                ['Number of models tested', str(len(results))],
                ['Measurements per model', str(self.num_measurement_runs)],
                ['Standard prompt', self.prompt[:50] + '...' if len(self.prompt) > 50 else self.prompt],
                ['Vision models', f"{vision_count} ({vision_count*100//len(results) if self.results else 0}%)"],
                ['Tool-capable models', f"{tools_count} ({tools_count*100//len(results) if self.results else 0}%)"],
                ['Avg model size', f"{avg_size_gb:.2f} GB"],
                ['Avg speed', f"{avg_tokens_per_sec:.2f} tokens/s"],
            ]
            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Parameter-Tabelle
            elements.append(Paragraph("Benchmark Parameters", heading_style))
            
            params_data = [
                ['Parameter', 'Value'],
                ['Measurements per model', f"{self.num_measurement_runs}"],
                ['Context Length', f"{self.context_length} Tokens"],
                ['Temperature', str(OPTIMIZED_INFERENCE_PARAMS['temperature'])],
                ['Top-K Sampling', str(OPTIMIZED_INFERENCE_PARAMS['top_k_sampling'])],
                ['Top-P Sampling', str(OPTIMIZED_INFERENCE_PARAMS['top_p_sampling'])],
                ['Min-P Sampling', str(OPTIMIZED_INFERENCE_PARAMS['min_p_sampling'])],
                ['Repeat Penalty', str(OPTIMIZED_INFERENCE_PARAMS['repeat_penalty'])],
                ['Max Tokens', str(OPTIMIZED_INFERENCE_PARAMS['max_tokens'])],
                ['GPU-Offload Levels', ', '.join(map(str, GPU_OFFLOAD_LEVELS))],
            ]
            
            # GTT info (AMD-specific)
            if self._gtt_info and self._gtt_info.get('total', 0) > 0:
                gtt_total = self._gtt_info['total']
                gtt_used = self._gtt_info['used']
                gtt_status = "Enabled" if self.use_gtt else "Disabled"
                params_data.append(
                    ['GTT (Shared System RAM)',
                     f"{gtt_status} ({gtt_total:.1f}GB total, {gtt_used:.1f}GB used)"]
                )
            
            # CLI arguments
            if self.cli_args.get('limit'):
                params_data.append(['Model limit', str(self.cli_args['limit'])])
            if self.cli_args.get('retest'):
                params_data.append(['Cache ignored', 'Yes (--retest)'])
            if self.cli_args.get('only_vision'):
                params_data.append(['Filter', 'Only vision models'])
            if self.cli_args.get('only_tools'):
                params_data.append(['Filter', 'Only tool-capable models'])
            if self.cli_args.get('include_models'):
                params_data.append(['Include-Pattern', self.cli_args['include_models'][:30]])
            if self.cli_args.get('exclude_models'):
                params_data.append(['Exclude-Pattern', self.cli_args['exclude_models'][:30]])
            if self.cli_args.get('enable_profiling'):
                params_data.append(['Hardware profiling', 'Yes (--enable-profiling)'])
                if self.cli_args.get('max_temp'):
                    params_data.append(['Max. temperature', f"{self.cli_args['max_temp']}°C"])
                if self.cli_args.get('max_power'):
                    params_data.append(['Max. power draw', f"{self.cli_args['max_power']}W"])
            
            params_table = Table(params_data, colWidths=[3*inch, 3*inch])
            params_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#e8f0ff')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f5ff')])
            ]))
            elements.append(params_table)
            elements.append(Spacer(1, 20))
            
            # Results table
            elements.append(Paragraph("Detailed Results", heading_style))
            
            # Sort results by ranking criterion (ONLY newly tested models!)
            if self.rank_by == 'speed':
                sorted_results = sorted(results, key=lambda x: x.avg_tokens_per_sec, reverse=True)
            elif self.rank_by == 'efficiency':
                sorted_results = sorted(results, key=lambda x: x.tokens_per_sec_per_gb, reverse=True)
            elif self.rank_by == 'ttft':
                sorted_results = sorted(results, key=lambda x: x.avg_ttft, reverse=False)
            elif self.rank_by == 'vram':
                def get_vram_mb(result):
                    try:
                        return (
                            float(result.vram_mb.split()[0])
                            if isinstance(result.vram_mb, str) else float(result.vram_mb)
                        )
                    except:
                        return 999999
                sorted_results = sorted(results, key=get_vram_mb, reverse=False)
            else:
                sorted_results = sorted(results, key=lambda x: x.avg_tokens_per_sec, reverse=True)
            
            # Show ranking criterion
            rank_labels = {
                'speed': 'Speed (Tokens/s)',
                'efficiency': 'Efficiency (Tokens/s per GB)',
                'ttft': 'Time to First Token (ms)',
                'vram': 'VRAM usage (MB)'
            }
            elements.append(Paragraph(
                f"<font size=9>Sorted by: <b>{rank_labels.get(self.rank_by, 'Speed')}</b></font>",
                styles['Normal']
            ))
            elements.append(Spacer(1, 10))
            
            # Create table data
            table_data = [['Model', 'Param', 'Arch', 'Size(GB)', 'Vision', 'Tools', 'Quant.', 'GPU', 'GPU Off.', 'Tokens/s', 'Δ%', 'TTFT (ms)', 'Gen.Time (s)']]
            for result in sorted_results:
                vision_icon = '👁' if result.has_vision else ''
                tools_icon = '🔧' if result.has_tools else ''
                delta_str = f"{result.speed_delta_pct:+.1f}%" if result.speed_delta_pct is not None else "-"
                table_data.append([
                    result.model_name[:15],  # Truncate if needed
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
                    f"{result.avg_ttft*1000:.1f}" if result.avg_ttft else "N/A",
                    f"{result.avg_gen_time:.2f}",
                ])
            
            # Formatiere Tabelle (Landscape mit mehr Spalten)
            results_table = Table(table_data, colWidths=[1.2*inch, 0.55*inch, 0.6*inch, 0.65*inch, 0.45*inch, 0.45*inch, 0.6*inch, 0.5*inch, 0.5*inch, 0.65*inch, 0.45*inch, 0.65*inch, 0.7*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),

                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(results_table)
            elements.append(Spacer(1, 20))
            
            # Best-of-quantization analysis (only for exported models)
            elements.append(Paragraph("Best-of-Quantization Analysis", heading_style))
            
            # Analyze only the exported results
            best_by_model = {}
            for result in results:
                if result.model_name not in best_by_model:
                    best_by_model[result.model_name] = {
                        'best_speed': None,
                        'best_efficiency': None,
                        'best_ttft': None
                    }
                
                # Best Speed
                if not best_by_model[result.model_name]['best_speed'] or \
                   result.avg_tokens_per_sec > best_by_model[result.model_name]['best_speed'].avg_tokens_per_sec:
                    best_by_model[result.model_name]['best_speed'] = result
                
                # Best Efficiency
                if not best_by_model[result.model_name]['best_efficiency'] or \
                   result.tokens_per_sec_per_gb > best_by_model[result.model_name]['best_efficiency'].tokens_per_sec_per_gb:
                    best_by_model[result.model_name]['best_efficiency'] = result
                
                # Best TTFT
                if result.avg_ttft > 0:
                    if not best_by_model[result.model_name]['best_ttft'] or \
                       result.avg_ttft < best_by_model[result.model_name]['best_ttft'].avg_ttft:
                        best_by_model[result.model_name]['best_ttft'] = result
            
            quant_data = [['Model', 'Best Speed', 'Best Efficiency', 'Best TTFT']]
            for model_name, analysis in sorted(best_by_model.items()):
                speed_q = analysis['best_speed'].quantization if analysis['best_speed'] else '-'
                efficiency_q = analysis['best_efficiency'].quantization if analysis['best_efficiency'] else '-'
                ttft_q = analysis['best_ttft'].quantization if analysis['best_ttft'] else '-'
                quant_data.append([
                    model_name[:20],
                    speed_q[:8],
                    efficiency_q[:8],
                    ttft_q[:8]
                ])
            
            quant_table = Table(quant_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            quant_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))
            elements.append(quant_table)
            elements.append(Spacer(1, 20))
            
            # Quantization comparison (Q4 vs Q5 vs Q6) - only for exported models
            model_quants = {}
            for result in results:
                if result.model_name not in model_quants:
                    model_quants[result.model_name] = {}
                
                # Extract quantization level (e.g. "q4_k_m" -> "q4")
                quant_level = result.quantization.split('_')[0].lower()
                if quant_level not in model_quants[result.model_name]:
                    model_quants[result.model_name][quant_level] = result
                else:
                    # Take best performance (highest tokens/s) if multiple quantizations
                    if result.avg_tokens_per_sec > model_quants[result.model_name][quant_level].avg_tokens_per_sec:
                        model_quants[result.model_name][quant_level] = result
            
            # Create comparison table
            comp_data = {}
            for model, quants in sorted(model_quants.items()):
                comp_data[model] = {
                    'q4': None,
                    'q5': None, 
                    'q6': None,
                    'q8': None
                }
                for q_level in ['q4', 'q5', 'q6', 'q8']:
                    if q_level in quants:
                        r = quants[q_level]
                        comp_data[model][q_level] = {
                            'speed': round(r.avg_tokens_per_sec, 2),
                            'efficiency': round(r.tokens_per_sec_per_gb, 2),
                            'vram_mb': r.vram_mb,
                            'ttft': round(r.avg_ttft * 1000, 1)
                        }
            
            if comp_data:
                elements.append(Paragraph("Quantization Comparison (Q4 vs Q5 vs Q6)", heading_style))
                
                # Create comparison table
                comp_table_data = [['Model', 'Q4 (t/s)', 'Q5 (t/s)', 'Q6 (t/s)', 'Q8 (t/s)']]
                for model_name, q_variants in sorted(comp_data.items()):
                    row = [model_name[:15]]
                    for q_level in ['q4', 'q5', 'q6', 'q8']:
                        if q_variants.get(q_level):
                            row.append(f"{q_variants[q_level]['speed']:.2f}")
                        else:
                            row.append('-')
                    comp_table_data.append(row)
                
                comp_table = Table(comp_table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d7aa8')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ]))
                elements.append(comp_table)
                elements.append(Spacer(1, 20))
            
            # Performance statistics
            elements.append(Paragraph("Performance Statistics", heading_style))
            max_tps_result = max(results, key=lambda x: x.avg_tokens_per_sec)
            min_tps_result = min(results, key=lambda x: x.avg_tokens_per_sec)
            avg_tps = sum(r.avg_tokens_per_sec for r in results) / len(results)
            
            # Calculate percentiles (only for exported models)
            speeds = sorted([r.avg_tokens_per_sec for r in results if r.avg_tokens_per_sec > 0])
            percentile_stats = {}
            if len(speeds) >= 3:
                percentile_stats['speed'] = {
                    'median': speeds[len(speeds)//2],
                    'p95': speeds[int(len(speeds) * 0.95)],
                    'p99': speeds[int(len(speeds) * 0.99)] if len(speeds) > 100 else speeds[-1]
                }
            
            stats_data = [
                ['Statistic', 'Value'],
                ['Fastest model', f"{max_tps_result.model_name} ({max_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Slowest model', f"{min_tps_result.model_name} ({min_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Avg tokens/s', f"{avg_tps:.2f}"],
            ]
            
            # Add percentile rows
            if 'speed' in percentile_stats:
                speed_p = percentile_stats['speed']
                stats_data.extend([
                    ['Tokens/s - Median', f"{speed_p.get('median', '-'):.2f}"],
                    ['Tokens/s - P95', f"{speed_p.get('p95', '-'):.2f}"],
                    ['Tokens/s - P99', f"{speed_p.get('p99', '-'):.2f}"],
                ])
            
            stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            elements.append(stats_table)
            elements.append(Spacer(1, 20))
            
            # Best practice recommendations
            elements.append(Paragraph("💡 Best Practice Recommendations", heading_style))
            recommendations = self._generate_best_practices()
            
            if recommendations:
                rec_text = "<br/>".join(recommendations)
                rec_style = ParagraphStyle(
                    'Recommendations',
                    parent=styles['Normal'],
                    fontSize=9,
                    leading=12,
                    leftIndent=10,
                    fontName='Courier'
                )
                elements.append(Paragraph(rec_text, rec_style))
                elements.append(Spacer(1, 20))
            
            # === NEW PAGE: Vision models ===
            vision_models = [r for r in results if r.has_vision]
            if vision_models:
                elements.append(PageBreak())
                elements.append(Paragraph("👁️  Vision Models (Multimodal)", title_style))
                elements.append(Spacer(1, 12))
                
                elements.append(Paragraph(
                    f"<font size=10>{len(vision_models)} vision-capable models found</font>",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 15))
                
                # Sort vision models by speed
                vision_sorted = sorted(vision_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                
                vision_data = [['Model', 'Param', 'Size(GB)', 'Quant.', 'Tokens/s', 'TTFT (ms)', 'Efficiency']]
                for r in vision_sorted:
                    vision_data.append([
                        r.model_name[:25],
                        r.params_size[:6],
                        f"{r.model_size_gb:.2f}",
                        r.quantization[:8],
                        f"{r.avg_tokens_per_sec:.2f}",
                        f"{r.avg_ttft*1000:.1f}" if r.avg_ttft else "N/A",
                        f"{r.tokens_per_sec_per_gb:.2f}"
                    ])
                
                vision_table = Table(vision_data, colWidths=[2*inch, 0.7*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
                vision_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4a90e2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f4ff')]),
                ]))
                elements.append(vision_table)
                elements.append(Spacer(1, 15))
                
                # Top 3 vision models
                elements.append(Paragraph("Top 3 Vision Models", heading_style))
                top3_text = []
                for i, r in enumerate(vision_sorted[:3], 1):
                    top3_text.append(f"{i}. <b>{r.model_name}</b> ({r.quantization})")
                    top3_text.append(f"   → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB")
                    top3_text.append("")
                
                elements.append(Paragraph("<br/>".join(top3_text), styles['Normal']))
            
            # === NEW PAGE: Tool models ===
            tool_models = [r for r in results if r.has_tools]
            if tool_models:
                elements.append(PageBreak())
                elements.append(Paragraph("🔧 Tool-Calling Models", title_style))
                elements.append(Spacer(1, 12))
                
                elements.append(Paragraph(
                    f"<font size=10>{len(tool_models)} tool-capable models found</font>",
                    styles['Normal']
                ))
                elements.append(Spacer(1, 15))
                
                # Sort tool models by speed
                tool_sorted = sorted(tool_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                
                tool_data = [['Model', 'Param', 'Size(GB)', 'Quant.', 'Tokens/s', 'TTFT (ms)', 'Efficiency']]
                for r in tool_sorted:
                    tool_data.append([
                        r.model_name[:25],
                        r.params_size[:6],
                        f"{r.model_size_gb:.2f}",
                        r.quantization[:8],
                        f"{r.avg_tokens_per_sec:.2f}",
                        f"{r.avg_ttft*1000:.1f}" if r.avg_ttft else "N/A",
                        f"{r.tokens_per_sec_per_gb:.2f}"
                    ])
                
                tool_table = Table(tool_data, colWidths=[2*inch, 0.7*inch, 0.8*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
                tool_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e27a4a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff4e8')]),
                ]))
                elements.append(tool_table)
                elements.append(Spacer(1, 15))
                
                # Top 3 tool models
                elements.append(Paragraph("Top 3 Tool-Calling Models", heading_style))
                top3_text = []
                for i, r in enumerate(tool_sorted[:3], 1):
                    top3_text.append(f"{i}. <b>{r.model_name}</b> ({r.quantization})")
                    top3_text.append(f"   → {r.avg_tokens_per_sec:.2f} tokens/s, {r.model_size_gb:.2f} GB")
                    top3_text.append("")
                
                elements.append(Paragraph("<br/>".join(top3_text), styles['Normal']))
            
            # === NEW PAGE: Grouped by architecture ===
            # Group by architecture
            by_arch = {}
            for r in results:
                arch = r.architecture
                if arch not in by_arch:
                    by_arch[arch] = []
                by_arch[arch].append(r)
            
            # Show only architectures with at least 2 models
            major_archs = {k: v for k, v in by_arch.items() if len(v) >= 2}
            
            if major_archs:
                elements.append(PageBreak())
                elements.append(Paragraph("🏗️  Models by Architecture", title_style))
                elements.append(Spacer(1, 12))
                
                for arch_name, arch_models in sorted(major_archs.items(), key=lambda x: -len(x[1])):
                    arch_sorted = sorted(arch_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                    
                    elements.append(Paragraph(
                        f"<b>{arch_name.upper()}</b> ({len(arch_models)} models)", heading_style
                    ))
                    elements.append(Spacer(1, 8))
                    
                    arch_data = [['Model', 'Param', 'Quant.', 'Tokens/s', 'Size(GB)']]
                    for r in arch_sorted[:5]:  # Top 5 per architecture
                        arch_data.append([
                            r.model_name[:30],
                            r.params_size[:6],
                            r.quantization[:8],
                            f"{r.avg_tokens_per_sec:.2f}",
                            f"{r.model_size_gb:.2f}"
                        ])
                    
                    arch_table = Table(arch_data, colWidths=[2.5*inch, 0.7*inch, 0.9*inch, 0.9*inch, 0.8*inch])
                    arch_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6a4ae2')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 9),
                        ('FONTSIZE', (0, 1), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0ecff')]),
                    ]))
                    elements.append(arch_table)
                    elements.append(Spacer(1, 15))
            
            # Hardware profiling page (when enabled)
            if self.enable_profiling and any(r.temp_celsius_avg for r in results):
                elements.append(PageBreak())
                elements.append(Paragraph("🌡️ Hardware Profiling Report", title_style))
                elements.append(Spacer(1, 12))
                
                # Profiling summary
                elements.append(Paragraph("Temperature and power analysis", heading_style))
                
                # Collect profiling data
                temps_avg = [r.temp_celsius_avg for r in results if r.temp_celsius_avg]
                powers_avg = [r.power_watts_avg for r in results if r.power_watts_avg]
                
                profile_summary = [
                    ['Metric', 'Min', 'Max', 'Average'],
                ]
                
                if temps_avg:
                    profile_summary.append([
                        'GPU Temperature (°C)',
                        f"{min(temps_avg):.1f}",
                        f"{max(temps_avg):.1f}",
                        f"{mean(temps_avg):.1f}"
                    ])
                
                if powers_avg:
                    profile_summary.append([
                        'GPU Power Draw (W)',
                        f"{min(powers_avg):.1f}",
                        f"{max(powers_avg):.1f}",
                        f"{mean(powers_avg):.1f}"
                    ])
                
                profile_table = Table(profile_summary, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.5*inch])
                profile_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9534f')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('FONTSIZE', (0, 1), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff0ed')])
                ]))
                elements.append(profile_table)
                elements.append(Spacer(1, 15))
                
                # Detailed profiling table
                elements.append(Paragraph("Profiling per model", heading_style))
                
                profile_data = [['Model', 'Quant.', 'Temp Min', 'Temp Max', 'Temp Avg (°C)', 'Power Min', 'Power Max', 'Power Avg (W)']]
                for r in sorted(results, key=lambda x: x.temp_celsius_avg or 0, reverse=True):
                    if r.temp_celsius_avg or r.power_watts_avg:
                        temp_min = f"{r.temp_celsius_min:.1f}" if r.temp_celsius_min else "-"
                        temp_max = f"{r.temp_celsius_max:.1f}" if r.temp_celsius_max else "-"
                        temp_avg = f"{r.temp_celsius_avg:.1f}" if r.temp_celsius_avg else "-"
                        power_min = f"{r.power_watts_min:.1f}" if r.power_watts_min else "-"
                        power_max = f"{r.power_watts_max:.1f}" if r.power_watts_max else "-"
                        power_avg = f"{r.power_watts_avg:.1f}" if r.power_watts_avg else "-"
                        
                        profile_data.append([
                            r.model_name[:20],
                            r.quantization[:6],
                            temp_min,
                            temp_max,
                            temp_avg,
                            power_min,
                            power_max,
                            power_avg
                        ])
                
                if len(profile_data) > 1:  # Nur wenn Daten vorhanden
                    prof_table = Table(profile_data, colWidths=[1.5*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.75*inch])
                    prof_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d9534f')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 8),
                        ('FONTSIZE', (0, 1), (-1, -1), 7),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ffe5e5')])
                    ]))
                    elements.append(prof_table)
            
            # Create PDF
            doc.build(elements)
            logger.info(f"📑 PDF results saved: {pdf_file}")
        
        except Exception as e:
            logger.error(f"❌ Error creating PDF: {e}")
    
    def _export_html(self, timestamp: str, results_to_export):
        """Exports given benchmark results as interactive HTML report with Plotly charts"""
        if not PLOTLY_AVAILABLE or go is None:
            logger.warning("⚠️ Plotly not available, skipping HTML export")
            return
        
        try:
            # Use results_to_export instead of self.results
            results = results_to_export
            
            html_file = RESULTS_DIR / f"benchmark_results_{timestamp}.html"
            
            # Load HTML template
            template_path = Path(__file__).parent / "report_template.html.template"
            with open(template_path, 'r', encoding='utf-8') as f:
                html_template = f.read()
            
            # Sort results by ranking criterion (but for HTML charts always by speed)
            sorted_results = self.sort_results('speed')  # HTML charts always show top 10 by speed
            
            # Bar chart: Top 10 fastest models
            top_10 = sorted_results[:10]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=[f"{r.model_name[:20]}\n{r.quantization}" for r in top_10],
                    y=[r.avg_tokens_per_sec for r in top_10],
                    text=[f"{r.avg_tokens_per_sec:.2f}" for r in top_10],
                    textposition='auto',
                    marker_color='#2d5aa8'
                )
            ])
            fig_bar.update_layout(
                title="Top 10 fastest models",
                xaxis_title="Model + Quantization",
                yaxis_title="Tokens/s",
                hovermode='x unified',
                height=500
            )
            
            # Scatter plot: model size vs speed
            fig_scatter = go.Figure(data=[
                go.Scatter(
                    x=[r.model_size_gb for r in results],
                    y=[r.avg_tokens_per_sec for r in results],
                    mode='markers',
                    text=[f"{r.model_name}<br>{r.quantization}<br>{r.avg_tokens_per_sec:.2f} t/s" for r in results],
                    marker=dict(
                        size=[r.avg_tokens_per_sec / 2 for r in results],
                        color=[r.tokens_per_sec_per_gb for r in results],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Efficiency<br>(t/s per GB)")
                    ),
                    hovertemplate='<b>%{text}</b><extra></extra>'
                )
            ])
            fig_scatter.update_layout(
                title="Model size vs performance (bubble size = speed)",
                xaxis_title="Model size (GB)",
                yaxis_title="Tokens/s",
                height=500,
                hovermode='closest'
            )
            
            # Scatter plot: parameters vs efficiency
            fig_efficiency = go.Figure(data=[
                go.Scatter(
                    x=[r.tokens_per_sec_per_gb for r in results],
                    y=[r.tokens_per_sec_per_billion_params for r in results],
                    mode='markers',
                    text=[f"{r.model_name} ({r.quantization})" for r in results],
                    marker=dict(
                        size=8,
                        color=[r.avg_tokens_per_sec for r in results],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Speed<br>(tokens/s)")
                    ),
                    hovertemplate='<b>%{text}</b><br>Per GB: %{x:.2f}<br>Per Billion Params: %{y:.2f}<extra></extra>'
                )
            ])
            fig_efficiency.update_layout(
                title="Efficiency analysis: Tokens/s per GB vs tokens/s per billion parameters",
                xaxis_title="Tokens/s per GB",
                yaxis_title="Tokens/s per billion parameters",
                height=500
            )
            
            # Summary table
            vision_count = sum(1 for r in results if r.has_vision)
            tools_count = sum(1 for r in results if r.has_tools)
            avg_size_gb = (
                sum(r.model_size_gb for r in results) / len(results) if self.results else 0
            )
            avg_tokens_per_sec = (
                sum(r.avg_tokens_per_sec for r in results) / len(results) if self.results else 0
            )
            
            summary_stats = {
                'Number of models': len(results),
                'Measurements per model': self.num_measurement_runs,
                'Standard prompt': (
                    self.prompt[:50] + '...' if len(self.prompt) > 50 else self.prompt
                ),
                'Fastest': (
                    f"{sorted_results[0].model_name[:20]}"
                    f" ({sorted_results[0].avg_tokens_per_sec:.2f} t/s)"
                ),
                'Slowest': (
                    f"{sorted_results[-1].model_name[:20]}"
                    f" ({sorted_results[-1].avg_tokens_per_sec:.2f} t/s)"
                ),
                'Avg speed': f"{avg_tokens_per_sec:.2f} t/s",
                'Vision models': (
                    f"{vision_count}"
                    f" ({vision_count*100//len(results) if self.results else 0}%)"
                ),
                'Tool-capable models': (
                    f"{tools_count}"
                    f" ({tools_count*100//len(results) if self.results else 0}%)"
                ),
                'Avg model size': f"{avg_size_gb:.2f} GB",
            }
            
            # Create summary boxes HTML
            summary_boxes = ""
            colors_list = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
            for i, (label, value) in enumerate(summary_stats.items()):
                color = colors_list[i % len(colors_list)]
                summary_boxes += f"""
            <div class="summary-box" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
            </div>
"""
            
            # Benchmark parameter section (complete, same as PDF)
            benchmark_params = []
            benchmark_params.append(
                f"<strong>Context Length:</strong> {self.context_length} Tokens"
            )
            benchmark_params.append(
                f"<strong>Temperature:</strong> {OPTIMIZED_INFERENCE_PARAMS['temperature']}"
            )
            benchmark_params.append(
                f"<strong>Top-K Sampling:</strong> "
                f"{OPTIMIZED_INFERENCE_PARAMS['top_k_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Top-P Sampling:</strong> "
                f"{OPTIMIZED_INFERENCE_PARAMS['top_p_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Min-P Sampling:</strong> "
                f"{OPTIMIZED_INFERENCE_PARAMS['min_p_sampling']}"
            )
            benchmark_params.append(
                f"<strong>Repeat Penalty:</strong> "
                f"{OPTIMIZED_INFERENCE_PARAMS['repeat_penalty']}"
            )
            benchmark_params.append(
                f"<strong>Max Tokens:</strong> {OPTIMIZED_INFERENCE_PARAMS['max_tokens']}"
            )
            benchmark_params.append(
                f"<strong>GPU-Offload Levels:</strong> "
                f"{', '.join(map(str, GPU_OFFLOAD_LEVELS))}"
            )
            
            # CLI arguments
            cli_params = []
            cli_params.append(
                f"<strong>Measurements per model:</strong> {self.cli_args['runs']}"
            )
            
            # GTT info (AMD-specific)
            if self._gtt_info and self._gtt_info.get('total', 0) > 0:
                gtt_total = self._gtt_info['total']
                gtt_used = self._gtt_info['used']
                gtt_status = "✅ Enabled" if self.use_gtt else "❌ Disabled"
                cli_params.append(
                    f"<strong>GTT (Shared System RAM):</strong> {gtt_status}"
                    f" ({gtt_total:.1f}GB total, {gtt_used:.1f}GB used)"
                )
            
            if self.cli_args.get('limit'):
                cli_params.append(f"<strong>Model limit:</strong> {self.cli_args['limit']}")
            if self.cli_args.get('retest'):
                cli_params.append(f"<strong>Cache:</strong> Ignored (--retest)")
            if self.cli_args.get('only_vision'):
                cli_params.append(f"<strong>Filter:</strong> Only vision models")
            if self.cli_args.get('only_tools'):
                cli_params.append(f"<strong>Filter:</strong> Only tool-capable models")
            if self.cli_args.get('include_models'):
                cli_params.append(f"<strong>Include-Pattern:</strong> {self.cli_args['include_models'][:40]}")
            if self.cli_args.get('exclude_models'):
                cli_params.append(f"<strong>Exclude-Pattern:</strong> {self.cli_args['exclude_models'][:40]}")
            if self.cli_args.get('enable_profiling'):
                cli_params.append(f"<strong>Hardware profiling:</strong> Yes (--enable-profiling)")
                if self.cli_args.get('max_temp'):
                    cli_params.append(
                        f"<strong>Max. temperature:</strong> {self.cli_args['max_temp']}°C"
                    )
                if self.cli_args.get('max_power'):
                    cli_params.append(
                        f"<strong>Max. power draw:</strong> {self.cli_args['max_power']}W"
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
            
            # Trend chart if available
            trend_json = self.generate_trend_chart()
            if trend_json:
                trend_section = """
        <h2>📈 Performance Trends Over Time</h2>
        <div class="chart" id="trend-chart"></div>
"""
                trend_data = json.loads(trend_json)
                trend_script = f"        Plotly.newPlot('trend-chart', {json.dumps(trend_data['data'])}, {json.dumps(trend_data['layout'])});"
            else:
                trend_section = ""
                trend_script = ""
            
            # Best practice recommendations
            recommendations = self._generate_best_practices()
            best_practices_html = (
                "\n".join(recommendations) if recommendations else "No recommendations available"
            )
            
            # Vision models section
            vision_models = [r for r in results if r.has_vision]
            if vision_models:
                vision_sorted = sorted(vision_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                vision_rows = ""
                for r in vision_sorted:
                    vision_rows += f"""
                    <tr>
                        <td>{r.model_name}</td>
                        <td>{r.params_size}</td>
                        <td>{r.model_size_gb:.2f} GB</td>
                        <td>{r.quantization}</td>
                        <td><strong>{r.avg_tokens_per_sec:.2f}</strong></td>
                        <td>{r.avg_ttft*1000:.1f} ms</td>
                        <td>{r.tokens_per_sec_per_gb:.2f}</td>
                    </tr>"""
                
                vision_section = f"""
        <h2>👁️ Vision Models (Multimodal)</h2>
        <p>{len(vision_models)} vision-capable models found</p>
        <table class="category-table">
            <thead class="vision">
                <tr>
                    <th>Model</th>
                    <th>Parameters</th>
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
            
            # Tool models section
            tool_models = [r for r in results if r.has_tools]
            if tool_models:
                tool_sorted = sorted(tool_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                tool_rows = ""
                for r in tool_sorted:
                    tool_rows += f"""
                    <tr>
                        <td>{r.model_name}</td>
                        <td>{r.params_size}</td>
                        <td>{r.model_size_gb:.2f} GB</td>
                        <td>{r.quantization}</td>
                        <td><strong>{r.avg_tokens_per_sec:.2f}</strong></td>
                        <td>{r.avg_ttft*1000:.1f} ms</td>
                        <td>{r.tokens_per_sec_per_gb:.2f}</td>
                    </tr>"""
                
                tools_section = f"""
        <h2>🔧 Tool-Calling Models</h2>
        <p>{len(tool_models)} tool-capable models found</p>
        <table class="category-table">
            <thead class="tools">
                <tr>
                    <th>Model</th>
                    <th>Parameters</th>
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
            
            # Architecture grouping
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
                
                for arch_name, arch_models in sorted(major_archs.items(), key=lambda x: -len(x[1])):
                    arch_sorted = sorted(arch_models, key=lambda x: x.avg_tokens_per_sec, reverse=True)
                    arch_rows = ""
                    for r in arch_sorted[:5]:  # Top 5 per architecture
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
                    <th>Parameters</th>
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
            
            # Hardware profiling section
            profiling_section = ""
            if self.enable_profiling and any(r.temp_celsius_avg for r in results):
                temps_avg = [r.temp_celsius_avg for r in results if r.temp_celsius_avg]
                powers_avg = [r.power_watts_avg for r in results if r.power_watts_avg]
                
                profile_rows = ""
                for r in sorted(results, key=lambda x: x.temp_celsius_avg or 0, reverse=True):
                    if r.temp_celsius_avg or r.power_watts_avg:
                        temp_min = f"{r.temp_celsius_min:.1f}°C" if r.temp_celsius_min else "-"
                        temp_max = f"{r.temp_celsius_max:.1f}°C" if r.temp_celsius_max else "-"
                        temp_avg = f"{r.temp_celsius_avg:.1f}°C" if r.temp_celsius_avg else "-"
                        power_min = f"{r.power_watts_min:.1f}W" if r.power_watts_min else "-"
                        power_max = f"{r.power_watts_max:.1f}W" if r.power_watts_max else "-"
                        power_avg = f"{r.power_watts_avg:.1f}W" if r.power_watts_avg else "-"
                        
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
                    <h4>🌡️ GPU Temperature</h4>
                    <p>Min: {min(temps_avg):.1f}°C | Max: {max(temps_avg):.1f}°C | Avg: {mean(temps_avg):.1f}°C</p>
                </div>"""
                
                if powers_avg:
                    profile_summary += f"""
                <div class="summary-box" style="border-left: 4px solid #ff9800;">
                    <h4>⚡ Power Draw</h4>
                    <p>Min: {min(powers_avg):.1f}W | Max: {max(powers_avg):.1f}W | Avg: {mean(powers_avg):.1f}W</p>
                </div>"""
                
                profiling_section = f"""
            <h2>🌡️ Hardware Profiling</h2>
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
                        <th>Temp Avg</th>
                        <th>Power Min</th>
                        <th>Power Max</th>
                        <th>Power Avg</th>
                    </tr>
                </thead>
                <tbody>
                    {profile_rows}
                </tbody>
            </table>"""
            
            # Replace placeholders in template
            html_output = html_template.replace('{{SUMMARY_BOXES}}', summary_boxes)
            html_output = html_output.replace('{{CLI_SECTION}}', cli_section)
            html_output = html_output.replace('{{TREND_SECTION}}', trend_section)
            html_output = html_output.replace('{{BEST_PRACTICES}}', best_practices_html)
            html_output = html_output.replace('{{VISION_SECTION}}', vision_section)
            html_output = html_output.replace('{{TOOLS_SECTION}}', tools_section)
            html_output = html_output.replace('{{ARCH_SECTION}}', arch_section)
            html_output = html_output.replace('{{PROFILING_SECTION}}', profiling_section)
            html_output = html_output.replace('{{TIMESTAMP}}', time.strftime('%d.%m.%Y %H:%M:%S'))
            html_output = html_output.replace('{{BAR_DATA}}', json.dumps(fig_bar.to_dict()['data']))
            html_output = html_output.replace('{{BAR_LAYOUT}}', json.dumps(fig_bar.to_dict()['layout']))
            html_output = html_output.replace('{{SCATTER_DATA}}', json.dumps(fig_scatter.to_dict()['data']))
            html_output = html_output.replace(
                '{{SCATTER_LAYOUT}}', json.dumps(fig_scatter.to_dict()['layout'])
            )
            html_output = html_output.replace(
                '{{EFFICIENCY_DATA}}', json.dumps(fig_efficiency.to_dict()['data'])
            )
            html_output = html_output.replace(
                '{{EFFICIENCY_LAYOUT}}', json.dumps(fig_efficiency.to_dict()['layout'])
            )
            html_output = html_output.replace('{{TREND_SCRIPT}}', trend_script)
            
            # Write HTML file
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_output)
            
            logger.info(f"🌐 HTML results saved: {html_file}")
        
        except Exception as e:
            logger.error(f"❌ Error creating HTML: {e}")


def main():
    """Main function with CLI arguments"""
    global log_filename
    
    # Initialize log file ONLY when called directly (not via WebApp)
    # WebApp already logs via read_output() to its own log file
    # Check if we were started by a parent process (WebApp subprocess)
    import psutil
    current_process = psutil.Process()
    parent_process = current_process.parent()
    
    # If parent is "python" and "web/app.py" in command → no file logging
    # (CLI via run.py should still write a log file)
    is_webapp_subprocess = False
    if parent_process and 'python' in parent_process.name().lower():
        try:
            parent_cmdline = ' '.join(parent_process.cmdline())
            if 'web/app.py' in parent_cmdline:
                is_webapp_subprocess = True
        except:
            pass
    
    if not is_webapp_subprocess:
        # Normal direct call → create log file
        log_filename = LOGS_DIR / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        file_handler.addFilter(NoJSONFilter())
        logging.root.addHandler(file_handler)
        logger.info(f"📝 Benchmark log: {log_filename}")
    else:
        # WebApp subprocess → console logging only (WebApp writes the log file)
        logger.info("📝 Benchmark running as WebApp subprocess - logging via WebApp")
    
    parser = argparse.ArgumentParser(
        description="LM Studio Model Benchmark - Tests all locally installed LLM models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py                                   # Default: all models, 3 measurements
  python benchmark.py --runs 1                          # Fast: all models, 1 measurement
  python benchmark.py --limit 3 --runs 1                # Test 3 models with 1 measurement
  python benchmark.py --limit 1 --runs 1                # Test 1 model with 1 measurement
  python benchmark.py --runs 2 --context 4096           # 2 measurements, 4096 token context
  python benchmark.py --limit 5 --runs 2 --context 4096 # Test 5 models, more options
        """
    )
    
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=NUM_MEASUREMENT_RUNS,
        help=f'Number of measurements per model quantization (default: {NUM_MEASUREMENT_RUNS})'
    )
    
    parser.add_argument(
        '--context', '-c',
        type=int,
        default=CONTEXT_LENGTH,
        help=f'Context length in tokens (default: {CONTEXT_LENGTH})'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=STANDARD_PROMPT,
        help=f'Standard test prompt (default: "{STANDARD_PROMPT}")'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximum number of models to test (e.g. 3 tests only the first 3 models)'
    )
    
    # Extended filter options
    parser.add_argument(
        '--only-vision',
        action='store_true',
        help='Only test models with vision capability (multimodal)'
    )
    
    parser.add_argument(
        '--only-tools',
        action='store_true',
        help='Only test models with tool-calling support'
    )
    
    parser.add_argument(
        '--quants',
        type=str,
        default=None,
        help='Only test specific quantizations (e.g. "q4,q5,q6")'
    )
    
    parser.add_argument(
        '--arch',
        type=str,
        default=None,
        help='Only test specific architectures (e.g. "llama,mistral,gemma")'
    )
    
    parser.add_argument(
        '--params',
        type=str,
        default=None,
        help='Only test specific parameter sizes (e.g. "3B,7B,8B")'
    )
    
    parser.add_argument(
        '--min-context',
        type=int,
        default=None,
        help='Minimum context length in tokens (e.g. 32000)'
    )
    
    parser.add_argument(
        '--max-size',
        type=float,
        default=None,
        help='Maximum model size in GB (e.g. 10.0)'
    )
    
    # Regex-based filters
    parser.add_argument(
        '--include-models',
        type=str,
        default=None,
        help='Only models matching the regex pattern (e.g. "llama.*7b" or "qwen|phi")'
    )
    
    parser.add_argument(
        '--exclude-models',
        type=str,
        default=None,
        help='Exclude models matching the regex pattern (e.g. ".*uncensored.*" or "test|experimental")'
    )
    
    parser.add_argument(
        '--compare-with',
        type=str,
        default=None,
        help='Compare with previous results (e.g. "20260104_172200.json" or "latest")'
    )
    
    parser.add_argument(
        '--rank-by',
        type=str,
        choices=['speed', 'efficiency', 'ttft', 'vram'],
        default='speed',
        help='Sort results by: speed (tokens/s), efficiency (tokens/s per GB), ttft (time to first token), vram (VRAM usage)'
    )
    
    # Cache management
    parser.add_argument(
        '--retest',
        action='store_true',
        help='Ignore cache and retest all models (overwrites old results in the database)'
    )
    
    parser.add_argument(
        '--dev-mode',
        action='store_true',
        help='Development mode: automatically tests the smallest available model with -l 1 -r 1'
    )
    
    parser.add_argument(
        '--list-cache',
        action='store_true',
        help='Show all cached models and exit'
    )
    
    parser.add_argument(
        '--export-cache',
        type=str,
        default=None,
        metavar='FILE',
        help='Export cache contents as JSON (e.g. "cache_export.json") and exit'
    )
    
    # Hardware profiling
    parser.add_argument(
        '--enable-profiling',
        action='store_true',
        help='Enable hardware profiling: measures temperature and power draw during benchmark'
    )
    
    parser.add_argument(
        '--max-temp',
        type=float,
        default=None,
        help='Maximum GPU temperature in °C (warning if exceeded, e.g. 80.0)'
    )
    
    parser.add_argument(
        '--max-power',
        type=float,
        default=None,
        help='Maximum GPU power draw in watts (warning if exceeded, e.g. 400.0)'
    )
    
    # GTT (Graphics Translation Table) - Shared system RAM for AMD GPUs
    parser.add_argument(
        '--disable-gtt',
        action='store_true',
        help='Disable GTT (shared system RAM) for AMD GPUs - use only dedicated VRAM'
    )
    
    # Report regeneration
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Generate reports (JSON/CSV/PDF/HTML) from all results in the database without running new tests'
    )

    # Inference parameter overrides (optional)
    parser.add_argument(
        '--temperature',
        type=float,
        default=None,
        help='Override: sampling temperature (e.g. 0.7)'
    )
    parser.add_argument(
        '--top-k', '--top-k-sampling',
        dest='top_k_sampling',
        type=int,
        default=None,
        help='Override: top-K sampling (e.g. 50)'
    )
    parser.add_argument(
        '--top-p', '--top-p-sampling',
        dest='top_p_sampling',
        type=float,
        default=None,
        help='Override: top-P (nucleus sampling, e.g. 0.95)'
    )
    parser.add_argument(
        '--min-p', '--min-p-sampling',
        dest='min_p_sampling',
        type=float,
        default=None,
        help='Override: min-P (minimum probability threshold, e.g. 0.05)'
    )
    parser.add_argument(
        '--repeat-penalty',
        type=float,
        default=None,
        help='Override: repeat penalty (e.g. 1.2)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=None,
        help='Override: max output tokens (e.g. 256)'
    )
    
    # Load config parameters (performance tuning)
    parser.add_argument(
        '--n-gpu-layers',
        type=int,
        default=DEFAULT_LOAD_PARAMS.get('n_gpu_layers', -1),
        help=(
            f"Number of GPU layers (-1=auto/all, 0=CPU only, >0=specific count, "
            f"default: {DEFAULT_LOAD_PARAMS.get('n_gpu_layers', -1)})"
        )
    )
    parser.add_argument(
        '--n-batch',
        type=int,
        default=DEFAULT_LOAD_PARAMS.get('n_batch', 512),
        help=f"Batch size for prompt processing (default: {DEFAULT_LOAD_PARAMS.get('n_batch', 512)})"
    )
    parser.add_argument(
        '--n-threads',
        type=int,
        default=DEFAULT_LOAD_PARAMS.get('n_threads', -1),
        help=(
            f"Number of CPU threads (-1=auto/all, "
            f"default: {DEFAULT_LOAD_PARAMS.get('n_threads', -1)})"
        )
    )
    parser.add_argument(
        '--flash-attention',
        action='store_true',
        default=DEFAULT_LOAD_PARAMS.get('flash_attention', True),
        help=(
            f"Enable flash attention (faster attention computation, "
            f"default: {'enabled' if DEFAULT_LOAD_PARAMS.get('flash_attention', True) else 'disabled'})"
        )
    )
    parser.add_argument(
        '--no-flash-attention',
        action='store_false',
        dest='flash_attention',
        help='Disable flash attention'
    )
    parser.add_argument(
        '--rope-freq-base',
        type=float,
        default=DEFAULT_LOAD_PARAMS.get('rope_freq_base'),
        help='RoPE frequency base (None=model default, e.g. 10000.0)'
    )
    parser.add_argument(
        '--rope-freq-scale',
        type=float,
        default=DEFAULT_LOAD_PARAMS.get('rope_freq_scale'),
        help='RoPE frequency scale (None=model default, e.g. 1.0)'
    )
    parser.add_argument(
        '--use-mmap',
        action='store_true',
        default=DEFAULT_LOAD_PARAMS.get('use_mmap', True),
        help=(
            f"Enable memory mapping (faster model load, "
            f"default: {'enabled' if DEFAULT_LOAD_PARAMS.get('use_mmap', True) else 'disabled'})"
        )
    )
    parser.add_argument(
        '--no-mmap',
        action='store_false',
        dest='use_mmap',
        help='Disable memory mapping'
    )
    parser.add_argument(
        '--use-mlock',
        action='store_true',
        default=DEFAULT_LOAD_PARAMS.get('use_mlock', False),
        help=(
            f"Enable memory locking (prevents swapping, "
            f"default: {'enabled' if DEFAULT_LOAD_PARAMS.get('use_mlock', False) else 'disabled'})"
        )
    )
    parser.add_argument(
        '--kv-cache-quant',
        type=str,
        choices=['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'iq4_nl', 'q5_0', 'q5_1'],
        default=DEFAULT_LOAD_PARAMS.get('kv_cache_quant'),
        help='KV cache quantization (reduces VRAM, may affect performance, None=model default)'
    )
    
    args = parser.parse_args()
    
    # Cache management commands (exit before benchmark)
    if args.list_cache:
        cache = BenchmarkCache()
        cached = cache.list_cached_models()
        if cached:
            print("\n=== Cached Benchmark Results ===")
            print(f"{'Model':<50} {'Quant':<10} {'Params':<8} {'tok/s':<10} {'Date':<12} {'Hash':<10}")
            print("-" * 110)
            for entry in cached:
                print(f"{entry['model_name']:<50} {entry['quantization']:<10} {entry['params_size']:<8} "
                      f"{entry['avg_tokens_per_sec']:<10.2f} {entry['timestamp'][:10]:<12} {entry['params_hash']:<10}")
            print(f"\nTotal: {len(cached)} entries")
        else:
            print("Cache is empty - no results stored")
        return
    
    if args.export_cache:
        cache = BenchmarkCache()
        output_file = RESULTS_DIR / args.export_cache
        cache.export_to_json(output_file)
        print(f"Cache exported to: {output_file}")
        return
    
    # Export-only: generate reports from DB without new tests
    if args.export_only:
        logger.info("🔄 === Report regeneration from database ===")
        cache = BenchmarkCache()
        cached_results = cache.get_all_results()
        
        if not cached_results:
            logger.error(
                "❌ No results found in database. Run a benchmark first."
            )
            return
        
        logger.info(f"📥 Loading {len(cached_results)} results from database...")
        
        # Create filter dictionary
        filter_args = {
            'only_vision': args.only_vision,
            'only_tools': args.only_tools,
            'quants': args.quants,
            'arch': args.arch,
            'params': args.params,
            'min_context': args.min_context,
            'max_size': args.max_size,
            'include_models': args.include_models,
            'exclude_models': args.exclude_models,
        }
        
        # Create benchmark instance with cached data
        benchmark = LMStudioBenchmark(
            num_runs=1,  # Not used
            context_length=2048,  # Not used
            prompt="",  # Not used
            model_limit=None,
            filter_args=filter_args,
            compare_with=args.compare_with,
            rank_by=args.rank_by,
            use_cache=False,  # No cache checks needed
            enable_profiling=False,
            use_gtt=not args.disable_gtt
        )
        
        # Load all results directly
        benchmark.results = cached_results
        
        # Apply filters if set
        if any(filter_args.values()):
            original_count = len(benchmark.results)
            benchmark.results = [r for r in benchmark.results if benchmark._matches_filters(r)]
            logger.info(
                f"✔️ After filtering: {len(benchmark.results)}/{original_count} models"
            )
        
        if not benchmark.results:
            logger.error("❌ No results remaining after filtering")
            return
        
        # Load previous results for comparison if desired
        if args.compare_with:
            benchmark._load_previous_results()
        
        logger.info(f"⚙️ Generating reports for {len(benchmark.results)} models...")
        benchmark.export_results()
        logger.info("✅ Reports generated successfully!")
        return
    
    # Dev mode: override settings
    if args.dev_mode:
        logger.info("🧪 === Development mode activated ===")
        args.runs = 1
        args.limit = 1
        # Find smallest model
        all_models = ModelDiscovery.get_installed_models()
        if all_models:
            # Fetch metadata and sort by size
            model_sizes = []
            for model_key in all_models:
                metadata = ModelDiscovery.get_model_metadata(model_key)
                model_sizes.append((model_key, metadata.get('model_size_gb', 999)))
            
            model_sizes.sort(key=lambda x: x[1])
            smallest = model_sizes[0][0]
            logger.info(f"✅ Smallest model selected: {smallest} ({model_sizes[0][1]:.2f} GB)")
            logger.info(f"⚙️ Configuration: 1 measurement, context {args.context}")
            logger.info("")
        else:
            logger.error("❌ No models found for dev mode")
            return
    
    # Validation
    if args.runs < 1:
        parser.error('--runs must be >= 1')
    if args.context < 256:
        parser.error('--context must be >= 256')
    if len(args.prompt.strip()) == 0:
        parser.error('--prompt must not be empty')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be >= 1')
    
    # Create filter dictionary from CLI arguments
    filter_args = {
        'only_vision': args.only_vision,
        'only_tools': args.only_tools,
        'quants': args.quants,
        'arch': args.arch,
        'params': args.params,
        'min_context': args.min_context,
        'max_size': args.max_size,
        'include_models': args.include_models,
        'exclude_models': args.exclude_models,
    }
    
    logger.info("🚀 === LM Studio Model Benchmark ===")
    logger.info(f"💬 Prompt: '{args.prompt}'")
    logger.info(f"📏 Context Length: {args.context} Tokens")
    logger.info(f"🔢 Measurements per model: {args.runs} (+ {NUM_WARMUP_RUNS} warmup)")
    if args.limit:
        logger.info(f"📌 Model limit: testing max. {args.limit} models")
    
    # Show active filters
    active_filters = [k for k, v in filter_args.items() if v]
    if active_filters:
        logger.info(f"🔎 Active filters: {', '.join(active_filters)}")
    
    if args.compare_with:
        logger.info(f"📈 Historical comparison: {args.compare_with}")
    
    logger.info(
        f"⏱️ Estimated total time: ~{int(args.runs * 45 * (args.limit or 9) / 9)} minutes"
    )
    logger.info("")
    
    # Collect optional inference overrides
    inference_overrides = {
        'temperature': args.temperature,
        'top_k_sampling': args.top_k_sampling,
        'top_p_sampling': args.top_p_sampling,
        'min_p_sampling': args.min_p_sampling,
        'repeat_penalty': args.repeat_penalty,
        'max_tokens': args.max_tokens,
    }
    
    # Collect load config parameters
    load_params = {
        'n_gpu_layers': args.n_gpu_layers,
        'n_batch': args.n_batch,
        'n_threads': args.n_threads,
        'flash_attention': args.flash_attention,
        'rope_freq_base': args.rope_freq_base,
        'rope_freq_scale': args.rope_freq_scale,
        'use_mmap': args.use_mmap,
        'use_mlock': args.use_mlock,
        'kv_cache_quant': args.kv_cache_quant,
    }

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
        load_params=load_params
    )
    benchmark.run_all_benchmarks()
    
    logger.info("🎉 Benchmark complete!")



if __name__ == "__main__":
    main()
