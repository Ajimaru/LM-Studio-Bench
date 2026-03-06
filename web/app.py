#!/usr/bin/env python3
"""
FastAPI Web Dashboard for LM Studio Benchmark

Controls benchmark.py via subprocess and provides live monitoring via
WebSocket.
"""

import argparse
import asyncio
import json
import logging
import signal
import socket
import subprocess
from subprocess import TimeoutExpired
import sys
import webbrowser
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import sqlite3
import math
import statistics
from contextlib import asynccontextmanager

import re
import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

try:
    from scipy import stats as scipy_stats  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    scipy_stats = None  # type: ignore

SCIPY_AVAILABLE = scipy_stats is not None

# Configure logging - with console and file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebApp startup log file


def setup_webapp_logger():
    """Creates a separate WebApp startup log file"""
    logs_dir = Path(PROJECT_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"webapp_{timestamp}.log"

    # Add file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)

    return log_file

# ============================================================================
# Helper functions
# ============================================================================


def find_free_port() -> int:
    """Finds a free port on the system"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
# Ensure local `src/` is on sys.path so imports like `config_loader` work
sys.path.insert(0, str(SRC_DIR))

# Import config defaults from src/config_loader.py after sys.path is set
from config_loader import DEFAULT_CONFIG
BENCHMARK_SCRIPT = SRC_DIR / "benchmark.py"
TEMPLATES_DIR = Path(__file__).parent / "templates"
RESULTS_DIR = PROJECT_ROOT / "results"
DATABASE_FILE = RESULTS_DIR / "benchmark_cache.db"
METADATA_DATABASE_FILE = RESULTS_DIR / "model_metadata.db"
SCRAPER_SCRIPT = PROJECT_ROOT / "tools" / "scrape_metadata.py"

# Import BenchmarkCache from benchmark.py

sys.path.insert(0, str(SRC_DIR))
try:
    from benchmark import BenchmarkCache, BenchmarkResult  # type: ignore
except ImportError as e:
    logger.error(f"❌ Could not import benchmark.py: {e}")
    BenchmarkCache = None  # type: ignore
    BenchmarkResult = None  # type: ignore

CONFIG_DEFAULTS = DEFAULT_CONFIG
LMSTUDIO_HOST = CONFIG_DEFAULTS.get("lmstudio", {}).get("host", "localhost")
LMSTUDIO_PORTS = CONFIG_DEFAULTS.get("lmstudio", {}).get("ports", [1234, 1235])

# Jinja2 template environment
template_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Global benchmark process management


class BenchmarkManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"  # idle, running, paused, stopped, completed
        self.start_time: Optional[datetime] = None
        self.current_output = ""
        self.connected_clients = set()
        self.benchmark_log_file: Optional[Path] = None
        self.output_queue: Optional[asyncio.Queue[str]] = None
        self.output_task: Optional[asyncio.Task] = None

        # Hardware monitoring data
        self.hardware_history = {
            "temperatures": [],  # Liste von {"timestamp": ISO, "value": float}
            "power": [],
            "vram": [],
            "gtt": [],
            "cpu": [],
            "ram": []
        }
        self.last_hardware_send_time: float = 0

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def _consume_output(self):
        """Continuously reads stdout and puts new chunks into output_queue."""
        if self.output_queue is None:
            self.output_queue = asyncio.Queue()

        logger.info("🔄 Output consumer task started")

        try:
            loop = asyncio.get_event_loop()

            # Read until EOF (even after process ends)
            while True:
                if not self.process or not self.process.stdout:
                    break

                try:
                    # Blocking read in executor (waits until line is available)
                    line = await loop.run_in_executor(
                        None,
                        self.process.stdout.readline
                    )

                    if not line:
                        # EOF reached - process is done
                        break

                    # Write immediately to log file
                    if self.benchmark_log_file:
                        try:
                            with open(
                                self.benchmark_log_file, 'a', encoding='utf-8'
                            ) as f:
                                f.write(line)
                        except Exception as log_error:
                            logger.error(f"❌ Log write error: {log_error}")

                    # Parse hardware metrics
                    self.parse_hardware_metrics(line)

                    # Into queue for WebSocket
                    await self.output_queue.put(line)
                    self.current_output += line

                except Exception as read_error:
                    logger.error(f"❌ Read error: {read_error}")
                    break

            # Set status to completed when process is done
            if not self.is_running():
                self.status = "completed"

            logger.info("🔄 Output consumer task ended (EOF reached)")

            # Completion log
            if self.benchmark_log_file and self.benchmark_log_file.exists():
                logger.info(f"✅ Benchmark-Log: {self.benchmark_log_file}")

        except Exception as e:
            logger.error(f"❌ Error in output consumer: {e}")

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
        return ''.join(chunks)

    async def start_benchmark(self, args: list) -> bool:
        """Starts a new benchmark process"""
        if self.is_running():
            logger.warning("Benchmark already running")
            return False

        try:
            # Reset output buffer/task
            self.output_queue = asyncio.Queue()
            if self.output_task and not self.output_task.done():
                self.output_task.cancel()

            self.process = subprocess.Popen(
                [sys.executable, str(BENCHMARK_SCRIPT)] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT
            )
            self.status = "running"
            self.start_time = datetime.now()
            self.current_output = ""

            # Only set the path - the file is created on first write
            logs_dir = RESULTS_DIR.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = self.start_time.strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_{timestamp_str}.log"
            self.benchmark_log_file = logs_dir / filename

            # Start background task for continuous output reading
            self.output_task = asyncio.create_task(self._consume_output())

            logger.info("✅ Benchmark started with PID %s", self.process.pid)
            logger.info("📝 Benchmark log: %s", self.benchmark_log_file)
            return True
        except Exception as e:
            logger.error(f"❌ Error starting benchmark: {e}")
            self.status = "idle"
            return False

    def pause_benchmark(self) -> bool:
        """Pauses running benchmark"""
        if not self.is_running() or not self.process:
            logger.warning("No running benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGSTOP)
            self.status = "paused"
            logger.info("⏸️ Benchmark paused")
            return True
        except Exception as e:
            logger.error(f"❌ Error pausing: {e}")
            return False

    def resume_benchmark(self) -> bool:
        """Resumes paused benchmark"""
        if self.status != "paused" or not self.process:
            logger.warning("No paused benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGCONT)
            self.status = "running"
            logger.info("▶️ Benchmark resumed")
            return True
        except Exception as e:
            logger.error(f"❌ Error resuming: {e}")
            return False

    def stop_benchmark(self) -> bool:
        """Stops running benchmark"""
        if not self.process:
            logger.warning("No running benchmark")
            return False

        try:
            # Try graceful shutdown with SIGTERM
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
                logger.info("⏹️ Benchmark stopped (SIGTERM)")
            except TimeoutExpired:
                # Fallback to SIGKILL
                self.process.kill()
                self.process.wait()
                logger.warning("⏹️ Benchmark forcefully stopped (SIGKILL)")

            self.status = "stopped"
            self.process = None
            if self.output_task and not self.output_task.done():
                self.output_task.cancel()
            return True
        except Exception as e:
            logger.error(f"❌ Error stopping: {e}")
            return False

    def parse_hardware_metrics(self, output_line: str):
        """Parse hardware metrics from benchmark output"""

        # Pattern for GPU temperature: "🌡️ GPU Temp: 45.5°C"
        temp_pat = r'GPU\s+Temp\s*:\s*(\d+(?:\.\d+)?)°?C'
        temp_match = re.search(
            temp_pat,
            output_line,
            re.IGNORECASE
        )
        if temp_match:
            temp_value = float(temp_match.group(1))
            self.hardware_history["temperatures"].append({
                "timestamp": datetime.now().isoformat(),
                "value": temp_value
            })

        # Pattern for power: "⚡ GPU Power: 150.5W"
        power_pat = r'GPU\s+Power\s*:\s*(\d+(?:\.\d+)?)W'
        power_match = re.search(
            power_pat,
            output_line,
            re.IGNORECASE
        )
        if power_match:
            power_value = float(power_match.group(1))
            self.hardware_history["power"].append({
                "timestamp": datetime.now().isoformat(),
                "value": power_value
            })

        # Pattern for VRAM: "💾 GPU VRAM: 8.5GB"
        vram_pat = r'GPU\s+VRAM\s*:\s*(\d+(?:\.\d+)?)GB'
        vram_match = re.search(
            vram_pat,
            output_line,
            re.IGNORECASE
        )
        if vram_match:
            vram_value = float(vram_match.group(1))
            self.hardware_history["vram"].append({
                "timestamp": datetime.now().isoformat(),
                "value": vram_value
            })

        # Pattern for GTT: "🧠 GPU GTT: 1.5GB"
        gtt_pat = r'GPU\s+GTT\s*:\s*(\d+(?:\.\d+)?)GB'
        gtt_match = re.search(
            gtt_pat,
            output_line,
            re.IGNORECASE
        )
        if gtt_match:
            gtt_value = float(gtt_match.group(1))
            self.hardware_history["gtt"].append({
                "timestamp": datetime.now().isoformat(),
                "value": gtt_value
            })

        # Pattern for CPU: "🖥️ CPU: 45.2%"
        cpu_pat = r'CPU\s*:\s*(\d+(?:\.\d+)?)%'
        cpu_match = re.search(
            cpu_pat,
            output_line,
            re.IGNORECASE
        )
        if cpu_match:
            cpu_value = float(cpu_match.group(1))
            self.hardware_history["cpu"].append({
                "timestamp": datetime.now().isoformat(),
                "value": cpu_value
            })

        # Pattern for RAM: "💾 RAM: 8.5GB" (but NOT "GPU VRAM:")
        # Use negative lookbehind to match only "RAM:" (not "VRAM:")
        ram_pattern = r'(?<![V])RAM\s*:\s*(\d+(?:\.\d+)?)GB'
        ram_match = re.search(ram_pattern, output_line, re.IGNORECASE)
        if ram_match:
            ram_value = float(ram_match.group(1))
            self.hardware_history["ram"].append({
                "timestamp": datetime.now().isoformat(),
                "value": ram_value
            })

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
                        loop.run_in_executor(
                            None,
                            self.process.stdout.readline
                        ),
                        timeout=0.1
                    )

                    if output:
                        lines.append(output)
                    else:
                        # No further line available
                        break
                except asyncio.TimeoutError:
                    # Timeout means no further lines available
                    break

            # Combine all read lines
            if lines:
                combined_output = ''.join(lines)
                self.current_output += combined_output
                return combined_output

            return ""
        except Exception as e:
            logger.error(f"❌ Error reading output: {e}")
            return ""

manager = BenchmarkManager()


async def run_metadata_scraper():
    """Runs the metadata scraper script best-effort."""
    if not SCRAPER_SCRIPT.exists():
        logger.warning(f"⚠️ Scraper script not found: {SCRAPER_SCRIPT}")
        return
    try:
        await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(SCRAPER_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        logger.info("📝 Metadata scraper executed (missing models only)")
    except subprocess.CalledProcessError as scrape_err:
        logger.warning(f"⚠️ Scraper error: {scrape_err.stderr or scrape_err}")
    except Exception as scrape_exc:  # pragma: no cover
        logger.warning(f"⚠️ Scraper execution failed: {scrape_exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    description=(
        "Web dashboard for controlling and monitoring LM Studio benchmarks"
    ),
    lifespan=lifespan
)

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")


# ============================================================================
# Pydantic Models
# ============================================================================

class BenchmarkParams(BaseModel):
    """Parameters for starting a benchmark"""
    runs: Optional[int] = None
    context: Optional[int] = None
    limit: Optional[int] = None
    prompt: Optional[str] = None

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


class InferenceParamSet(BaseModel):
    """Set of inference parameters for an A/B test"""
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
    """Request for creating an A/B experiment"""
    name: str
    model_name: str
    baseline_params: InferenceParamSet
    test_params: InferenceParamSet
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ExperimentResult(BaseModel):
    """A/B test result with statistics"""
    experiment_id: str
    model_name: str
    baseline_data: Dict[str, Any]
    test_data: Dict[str, Any]
    statistical_test: Dict[str, Any]
    winner: str


# ============================================================================
# Statistical Analysis Functions
# ============================================================================

def calculate_hash(params: Dict[str, Any]) -> str:
    """Create SHA256 hash from parameter dictionary"""
    import hashlib
    params_str = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(params_str.encode()).hexdigest()[:16]


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
            "reason": "Insufficient data (min. 2 samples per group)"
        }

    try:
        if SCIPY_AVAILABLE and scipy_stats is not None:
            t_stat, p_value = scipy_stats.ttest_ind(
                baseline_speeds, test_speeds
            )
            return {
                "test_name": "Welch's t-test",
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "alpha": 0.05
            }

        baseline_mean = statistics.mean(baseline_speeds)
        test_mean = statistics.mean(test_speeds)
        baseline_var = (
            statistics.variance(baseline_speeds)
            if len(baseline_speeds) > 1 else 0
        )
        test_var = (
            statistics.variance(test_speeds)
            if len(test_speeds) > 1 else 0
        )

        n1, n2 = len(baseline_speeds), len(test_speeds)
        se = math.sqrt((baseline_var / n1) + (test_var / n2))

        if se == 0:
            return {
                "test_name": "t-test",
                "p_value": None,
                "significant": False,
                "reason": "No variance"
            }

        t_stat = (baseline_mean - test_mean) / se

        p_value = 0.01 if abs(t_stat) > 2.5 else 0.1

        return {
            "test_name": "t-test (approximated)",
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.05,
            "alpha": 0.05
        }

    except Exception as e:
        logger.error(f"Error in t-test: {e}")
        return {
            "test_name": "t-test",
            "error": str(e),
            "significant": False
        }


def match_parameters(row_params: Dict[str, Any], target_params: Dict[str, Optional[Any]]) -> bool:
    """
    Checks whether the parameters of a DB entry match the target parameters.
    Ignores None values in target_params (parameters were not overridden).
    """
    for key, target_value in target_params.items():
        if target_value is None:
            continue  # Parameter not overridden, so skip filtering

        row_value = row_params.get(key)

        if isinstance(target_value, (int, float)) and isinstance(row_value, (int, float)):
            if abs(float(row_value) - float(target_value)) > 0.001:
                return False
        else:
            if row_value != target_value:
                return False

    return True


def calculate_effect_size(baseline_speeds: List[float], test_speeds: List[float]) -> Dict[str, Union[float, str]]:
    """Calculate Cohen's d effect size"""
    if not baseline_speeds or not test_speeds:
        return {"cohens_d": 0.0, "effect_magnitude": "negligible"}

    baseline_mean = statistics.mean(baseline_speeds)
    test_mean = statistics.mean(test_speeds)

    baseline_var = statistics.variance(baseline_speeds) if len(baseline_speeds) > 1 else 0
    test_var = statistics.variance(test_speeds) if len(test_speeds) > 1 else 0

    # Pooled standard deviation
    n1, n2 = len(baseline_speeds), len(test_speeds)
    denom = n1 + n2 - 2

    # Guard against division by zero (when n1 + n2 <= 2)
    if denom <= 0:
        # Fallback: use unpooled standard deviation or 0
        return {"cohens_d": 0.0, "effect_magnitude": "negligible", "reason": "Insufficient data for effect size"}

    pooled_var = ((n1 - 1) * baseline_var + (n2 - 1) * test_var) / denom
    pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 1

    cohens_d = (test_mean - baseline_mean) / pooled_sd if pooled_sd > 0 else 0

    # Magnitude Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "cohens_d": round(cohens_d, 4),
        "effect_magnitude": magnitude
    }


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/")
async def root() -> HTMLResponse:
    """Main page - Dashboard"""
    template = template_env.get_template("dashboard.html.jinja")
    html = template.render(config=CONFIG_DEFAULTS)
    return HTMLResponse(content=html)


@app.get("/api/status")
async def get_status() -> dict:
    """Current benchmark status"""
    return {
        "status": manager.status,
        "running": manager.is_running(),
        "start_time": manager.start_time.isoformat() if manager.start_time else None,
        "uptime_seconds": (
            (datetime.now() - manager.start_time).total_seconds()
            if manager.start_time and manager.is_running() else None
        ),
        "connected_clients": len(manager.connected_clients)
    }


@app.get("/api/lmstudio/health")
async def get_lmstudio_health() -> dict:
    """LM Studio health check - live status without cache"""
    lmstudio_health = {"ok": False, "status": "offline"}
    lmstudio_ports = LMSTUDIO_PORTS
    
    # 1. HTTP API Check
    for port in lmstudio_ports:
        try:
            with httpx.Client(timeout=1.5) as client:
                resp = client.get(f"http://{LMSTUDIO_HOST}:{port}/v1/models")
                if resp.status_code == 200:
                    return {"ok": True, "status": f"online ({LMSTUDIO_HOST}:{port})", "version": None}
        except Exception:
            continue
    
    # 2. CLI Fallback
    try:
        import subprocess
        result = subprocess.run(["lms", "status"], capture_output=True, text=True, timeout=2)
        text = (result.stdout + result.stderr).lower()
        offline_keywords = ["server:  off", "server: off", "off", "not running", "stopped", "offline"]
        online_keywords = ["server:  on", "server: on", "running", "listening", "ready"]
        is_offline = any(kw in text for kw in offline_keywords) or result.returncode != 0
        is_online = any(kw in text for kw in online_keywords) and not is_offline
        if is_online:
            return {"ok": True, "status": "online (cli)", "version": None}
    except Exception:
        pass
    
    return {"ok": False, "status": "offline"}


@app.post("/api/benchmark/start")
async def start_benchmark(params: BenchmarkParams) -> dict:
    """Starts a new benchmark"""
    args = []
    
    # Base parameters
    if params.runs:
        args.extend(["--runs", str(params.runs)])
    if params.context:
        args.extend(["--context", str(params.context)])
    if params.limit:
        args.extend(["--limit", str(params.limit)])
    if params.prompt:
        args.extend(["--prompt", params.prompt])
    
    # Neue Filter-Parameter
    if params.min_context:
        args.extend(["--min-context", str(params.min_context)])
    if params.max_size:
        args.extend(["--max-size", str(params.max_size)])
    if params.quants:
        args.extend(["--quants", params.quants])
    if params.arch:
        args.extend(["--arch", params.arch])
    if params.params:
        args.extend(["--params", params.params])
    if params.rank_by:
        args.extend(["--rank-by", params.rank_by])
    
    # Regex filters
    if params.include_models:
        args.extend(["--include-models", params.include_models])
    if params.exclude_models:
        args.extend(["--exclude-models", params.exclude_models])
    
    # Boolean flags
    if params.only_vision:
        args.append("--only-vision")
    if params.only_tools:
        args.append("--only-tools")
    if params.retest:
        args.append("--retest")
    if params.dev_mode:
        args.append("--dev-mode")
    
    # Hardware profiling
    if params.enable_profiling:
        args.append("--enable-profiling")
    if params.disable_gtt:
        args.append("--disable-gtt")
    
    # Hardware limits
    if params.max_temp:
        args.extend(["--max-temp", str(params.max_temp)])
    if params.max_power:
        args.extend(["--max-power", str(params.max_power)])

    # Inference-Parameter Overrides
    if params.temperature is not None:
        args.extend(["--temperature", str(params.temperature)])
    if params.top_k_sampling is not None:
        args.extend(["--top-k", str(params.top_k_sampling)])
    if params.top_p_sampling is not None:
        args.extend(["--top-p", str(params.top_p_sampling)])
    if params.min_p_sampling is not None:
        args.extend(["--min-p", str(params.min_p_sampling)])
    if params.repeat_penalty is not None:
        args.extend(["--repeat-penalty", str(params.repeat_penalty)])
    if params.max_tokens is not None:
        args.extend(["--max-tokens", str(params.max_tokens)])
    
    # Load-Config Parameter (Performance Tuning)
    if params.n_gpu_layers is not None:
        args.extend(["--n-gpu-layers", str(params.n_gpu_layers)])
    if params.n_batch is not None:
        args.extend(["--n-batch", str(params.n_batch)])
    if params.n_threads is not None:
        args.extend(["--n-threads", str(params.n_threads)])
    if params.flash_attention is not None:
        if params.flash_attention:
            args.append("--flash-attention")
        else:
            args.append("--no-flash-attention")
    if params.rope_freq_base is not None:
        args.extend(["--rope-freq-base", str(params.rope_freq_base)])
    if params.rope_freq_scale is not None:
        args.extend(["--rope-freq-scale", str(params.rope_freq_scale)])
    if params.use_mmap is not None:
        if params.use_mmap:
            args.append("--use-mmap")
        else:
            args.append("--no-mmap")
    if params.use_mlock is not None and params.use_mlock:
        args.append("--use-mlock")
    if params.kv_cache_quant:
        args.extend(["--kv-cache-quant", params.kv_cache_quant])
    
    # Debug: show passed args
    logger.info(f"🔧 Benchmark args: {args}")
    logger.info(f"📊 enable_profiling={params.enable_profiling}, disable_gtt={params.disable_gtt}")
    
    success = await manager.start_benchmark(args)
    return {
        "success": success,
        "status": manager.status,
        "message": "✅ Benchmark started" if success else "❌ Error starting benchmark"
    }


@app.post("/api/benchmark/pause")
async def pause_benchmark() -> dict:
    """Pauses running benchmark"""
    success = manager.pause_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏸️ Paused" if success else "❌ Error"
    }


@app.post("/api/benchmark/resume")
async def resume_benchmark() -> dict:
    """Resumes paused benchmark"""
    success = manager.resume_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "▶️ Resumed" if success else "❌ Error"
    }


@app.post("/api/benchmark/stop")
async def stop_benchmark() -> dict:
    """Stops running benchmark"""
    success = manager.stop_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏹️ Stopped" if success else "❌ Error"
    }


@app.get("/api/results")
async def get_results() -> dict:
    """Returns all cached benchmark results"""
    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache not available",
            "results": []
        }
    
    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()
        
        # Convert BenchmarkResult to dict
        results_data = []
        for result in results:
            # model_key for frontend (for delete button)
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
                "tokens_per_sec_per_billion_params": result.tokens_per_sec_per_billion_params,
                "speed_delta_pct": result.speed_delta_pct,
                "prev_timestamp": result.prev_timestamp
            }
            
            # Optional fields (hardware profiling)
            if hasattr(result, 'temp_celsius_avg') and result.temp_celsius_avg:
                result_dict["temp_celsius_avg"] = result.temp_celsius_avg
            if hasattr(result, 'power_watts_avg') and result.power_watts_avg:
                result_dict["power_watts_avg"] = result.power_watts_avg
            if hasattr(result, 'gtt_enabled'):
                result_dict["gtt_enabled"] = result.gtt_enabled
            
            results_data.append(result_dict)
        
        return {
            "success": True,
            "count": len(results_data),
            "results": results_data
        }
    except Exception as e:
        logger.error(f"❌ Error loading results: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


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
                    "db_size_mb": 0
                }
            }
        
        speeds = [r.avg_tokens_per_sec for r in results]
        fastest = max(results, key=lambda r: r.avg_tokens_per_sec)
        slowest = min(results, key=lambda r: r.avg_tokens_per_sec)
        
        # DB-Größe
        db_size_mb = DATABASE_FILE.stat().st_size / (1024 * 1024) if DATABASE_FILE.exists() else 0
        
        return {
            "success": True,
            "stats": {
                "total_entries": len(results),
                "avg_tokens_per_sec": sum(speeds) / len(speeds),
                "fastest_model": f"{fastest.model_name}@{fastest.quantization}",
                "fastest_speed": fastest.avg_tokens_per_sec,
                "slowest_model": f"{slowest.model_name}@{slowest.quantization}",
                "slowest_speed": slowest.avg_tokens_per_sec,
                "db_size_mb": round(db_size_mb, 2)
            }
        }
    except Exception as e:
        logger.error(f"❌ Error retrieving cache statistics: {e}")
        return {"success": False, "error": str(e)}
        logger.error(f"❌ Error loading cache stats: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/cache/{model_key}")
async def delete_cache_entry(model_key: str) -> dict:
    """Deletes a single cache entry"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}
    
    try:
        import sqlite3
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Check if entry exists
        cursor.execute("SELECT COUNT(*) FROM benchmark_results WHERE model_key = ?", (model_key,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return {"success": False, "error": f"Model {model_key} not found in cache"}
        
        # Delete entry
        cursor.execute("DELETE FROM benchmark_results WHERE model_key = ?", (model_key,))
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        logger.info(f"🗑️ Cache entry deleted: {model_key}")
        return {
            "success": True,
            "message": f"✅ {deleted_count} entry/entries deleted",
            "model_key": model_key
        }
    except Exception as e:
        logger.error(f"❌ Error deleting cache entry: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/cache/clear")
async def clear_cache() -> dict:
    """Clears the entire cache with backup"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}
    
    try:
        import sqlite3
        import shutil
        
        # Create backup before deleting
        backup_dir = RESULTS_DIR / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / f"benchmark_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_backup.db"
        
        try:
            shutil.copy2(DATABASE_FILE, backup_file)
            logger.info(f"💾 Backup created: {backup_file}")
        except Exception as backup_error:
            logger.warning(f"⚠️ Backup error (cache will be cleared anyway): {backup_error}")
            backup_file = None
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Count entries before deleting
        cursor.execute("SELECT COUNT(*) FROM benchmark_results")
        count_before = cursor.fetchone()[0]
        
        # Delete all entries
        cursor.execute("DELETE FROM benchmark_results")
        conn.commit()
        conn.close()
        
        logger.warning(f"⚠️ Cache fully cleared: {count_before} entries deleted")
        logger.warning(f"💾 Backup available at: {backup_file}")
        
        return {
            "success": True,
            "message": f"✅ Cache cleared: {count_before} entries deleted",
            "deleted_count": count_before,
            "backup_file": str(backup_file) if backup_file else None
        }
    except Exception as e:
        logger.error(f"❌ Error clearing cache: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/lmstudio/models")
async def get_lmstudio_models() -> dict:
    """Fetches all locally installed models directly from LM Studio (incl. quantizations)"""
    try:
        import subprocess
        
        # Get list of all models with `lms ls`
        result = subprocess.run(
            ["lms", "ls"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"LM Studio CLI error: {result.stderr}",
                "models": []
            }
        
        # Parse output: lines with model/name (N variants)
        models = []
        lines = result.stdout.strip().split("\n")
        
        for line in lines:
            # Skip header and empty lines
            if not line.strip() or "LLM" in line or "PARAMS" in line or "You have" in line:
                continue
            
            # Parse: "qwen/qwen2.5-vl-7b (4 variants)"
            import re
            match = re.match(r'^([^\s]+(?:/[^\s]+)?)\s+\((\d+)\s+variants?\)', line)
            if match:
                base_model = match.group(1)
                variant_count = int(match.group(2))
                
                # Get all variants for this model
                variants_result = subprocess.run(
                    ["lms", "ls", base_model],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if variants_result.returncode == 0:
                    variant_lines = variants_result.stdout.strip().split("\n")
                    for vline in variant_lines:
                        # Parse: "qwen/qwen2.5-vl-7b@q3_k_l    7B    qwen2vl    5.44 GB"
                        vmatch = re.match(r'^([^\s]+)\s+(\d+(?:\.\d+)?[BMK])\s+(\S+)\s+([\d.]+\s+[GMK]B)', vline.strip())
                        if vmatch:
                            full_name = vmatch.group(1)
                            params = vmatch.group(2)
                            arch = vmatch.group(3)
                            size = vmatch.group(4)
                            
                            models.append({
                                "name": full_name,
                                "base_name": base_model,
                                "params": params,
                                "architecture": arch,
                                "size": size
                            })
        
        return {
            "success": True,
            "models": models,
            "count": len(models)
        }
    
    except TimeoutExpired:
        return {"success": False, "error": "LM Studio CLI Timeout", "models": []}
    except Exception as e:
        logger.error(f"❌ Error fetching LM Studio models: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "models": []}


@app.get("/api/comparison/models")
async def get_comparison_models() -> dict:
    """Returns all models with historical data"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available", "models": []}
    
    try:
        import sqlite3
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get models with entry count
        cursor.execute('''
            SELECT DISTINCT model_name, COUNT(*) as entry_count
            FROM benchmark_results
            GROUP BY model_name
            ORDER BY entry_count DESC, model_name ASC
        ''')
        
        models = []
        for model_name, count in cursor.fetchall():
            # Get newest and oldest entries
            cursor.execute('''
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (model_name,))
            latest = cursor.fetchone()
            
            cursor.execute('''
                SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
                WHERE model_name = ?
                ORDER BY timestamp ASC
                LIMIT 1
            ''', (model_name,))
            oldest = cursor.fetchone()
            
            if latest and oldest:
                delta = ((latest[1] - oldest[1]) / oldest[1] * 100) if oldest[1] > 0 else 0
                models.append({
                    "model_name": model_name,
                    "entry_count": count,
                    "latest_speed": round(latest[1], 2),
                    "latest_timestamp": latest[0],
                    "oldest_timestamp": oldest[0],
                    "speed_delta_pct": round(delta, 2)
                })
        
        conn.close()
        return {"success": True, "models": models}
    except Exception as e:
        logger.error(f"❌ Error fetching comparison models: {e}")
        return {"success": False, "error": str(e), "models": []}


@app.get("/api/comparison/{model_name:path}")
async def get_model_history(model_name: str) -> dict:
    """Returns history for a specific model"""
    # URL-decode model_name if necessary
    from urllib.parse import unquote
    model_name = unquote(model_name)
    
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available", "history": []}
    
    try:
        import sqlite3
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get all entries for the model, sorted by timestamp
        cursor.execute('''
            SELECT 
                timestamp, quantization, avg_tokens_per_sec, avg_ttft, 
                avg_gen_time, gpu_offload, vram_mb, temperature,
                top_k_sampling, top_p_sampling, min_p_sampling, repeat_penalty, max_tokens,
                num_runs, benchmark_duration_seconds, error_count
            FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        ''', (model_name,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
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
                "error_count": row[15]
            })
        
        # Calculate statistics
        if history:
            speeds = [h["speed_tokens_sec"] for h in history]
            stats = {
                "min_speed": round(min(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "total_runs": len(history),
                "first_run": history[0]["timestamp"],
                "last_run": history[-1]["timestamp"],
                "trend": "up" if speeds[-1] > speeds[0] else "down" if speeds[-1] < speeds[0] else "stable"
            }
        else:
            stats = {}
        
        conn.close()
        return {
            "success": True,
            "model_name": model_name,
            "history": history,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"❌ Error fetching model history: {e}")
        return {"success": False, "error": str(e), "history": []}


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
        import sqlite3
        import csv
        from io import StringIO
        
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        # Fallback to query parameters for backwards compatibility
        model_filter = payload.get("model_name", model_name)
        start_filter = payload.get("start_date", start_date)
        end_filter = payload.get("end_date", end_date)
        quant_filters = payload.get("quantizations", []) or []
        if isinstance(quant_filters, str):
            quant_filters = [quant_filters]

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Dynamic filter query
        query = '''
            SELECT timestamp, model_name, quantization, avg_tokens_per_sec, avg_ttft, 
                   avg_gen_time, gpu_offload, vram_mb, temperature, top_k_sampling,
                   top_p_sampling, min_p_sampling, repeat_penalty, max_tokens,
                   num_runs, benchmark_duration_seconds, error_count
            FROM benchmark_results
            WHERE 1=1
        '''
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
            except Exception:
                return value if value is not None else ""
        
        output = StringIO()
        writer = csv.writer(output)
        
        headers = [
            'Timestamp', 'Model', 'Quantization', 'Speed (tok/s)', 'TTFT (ms)',
            'Gen-Time (ms)', 'GPU-Offload', 'VRAM (MB)', 'Temperature', 'Top-K',
            'Top-P', 'Min-P', 'Repeat-Penalty', 'Max-Tokens', 'Num-Runs',
            'Duration (s)', 'Error-Count'
        ]
        writer.writerow(headers)
        
        for row in rows:
            writer.writerow([
                row[0], row[1], row[2], safe_round(row[3], 2), safe_round(row[4], 3),
                safe_round(row[5], 3), safe_round(row[6], 2), row[7], row[8], row[9],
                row[10], row[11], row[12], row[13], row[14],
                safe_round(row[15], 2), row[16]
            ])
        
        csv_content = output.getvalue()
        export_file = RESULTS_DIR / f"comparison_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(export_file, 'w', encoding='utf-8') as f:
            f.write(csv_content)
        
        logger.info(f"📊 CSV Export: {export_file}")
        
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
                "quantizations": quant_filters
            },
            "csv_preview": csv_content.split('\n')[:5]
        }
    except Exception as e:
        logger.error(f"❌ CSV export error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/comparison/export/pdf")
async def export_comparison_pdf(request: Request) -> dict:
    """Exports comparison data as a simple PDF summary"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}

    try:
        import sqlite3
        import statistics
        
        payload = {}
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        model_filter = payload.get("model_name")
        start_filter = payload.get("start_date")
        end_filter = payload.get("end_date")
        quant_filters = payload.get("quantizations", []) or []
        if isinstance(quant_filters, str):
            quant_filters = [quant_filters]

        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = '''
            SELECT timestamp, model_name, quantization, avg_tokens_per_sec, avg_ttft,
                   avg_gen_time, gpu_offload, vram_mb, temperature
            FROM benchmark_results
            WHERE 1=1
        '''
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
            "entries": len(rows)
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

            add_obj("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
            add_obj(f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n")
            pdf_bytes.extend(stream_content)
            pdf_bytes.extend(b"\nendstream\nendobj\n")
            add_obj("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

            xref_offset = len(pdf_bytes)
            pdf_bytes.extend(f"xref\n0 {len(offsets)+1}\n".encode("latin-1"))
            pdf_bytes.extend(b"0000000000 65535 f \n")
            for off in offsets:
                pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
            pdf_bytes.extend(b"trailer\n")
            pdf_bytes.extend(f"<< /Size {len(offsets)+1} /Root 1 0 R >>\n".encode("latin-1"))
            pdf_bytes.extend(b"startxref\n")
            pdf_bytes.extend(f"{xref_offset}\n".encode("latin-1"))
            pdf_bytes.extend(b"%%EOF")
            return bytes(pdf_bytes)

        header_lines = [
            "Historical Comparison Export",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {model_filter or 'All models'}",
            f"Period: {start_filter or '---'} to {end_filter or '---'}",
            f"Quantization: {', '.join(quant_filters) if quant_filters else 'All'}",
            "",
            "Statistics:",
            f"  Min Speed: {stats['min_speed'] if stats['min_speed'] is not None else '-'} tok/s",
            f"  Max Speed: {stats['max_speed'] if stats['max_speed'] is not None else '-'} tok/s",
            f"  Avg Speed: {stats['avg_speed'] if stats['avg_speed'] is not None else '-'} tok/s",
            f"  Runs: {stats['entries']}",
            "",
            "Top 50 Runs:"
        ]

        for row in rows[:50]:
            header_lines.append(
                f"{row[0]} | {row[1]} | {row[2]} | {round(row[3],2) if row[3] is not None else '-'} tok/s | TTFT {round(row[4],3) if row[4] is not None else '-'} | Gen {round(row[5],3) if row[5] is not None else '-'}"
            )

        pdf_bytes = generate_pdf_bytes(header_lines)
        export_file = RESULTS_DIR / f"comparison_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        with open(export_file, 'wb') as f:
            f.write(pdf_bytes)

        logger.info(f"🧾 PDF Export: {export_file}")

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
                "quantizations": quant_filters
            }
        }
    except Exception as e:
        logger.error(f"❌ PDF export error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/comparison/statistics/{model_name:path}")
async def get_advanced_statistics(model_name: str) -> dict:
    """Calculates advanced statistics (volatility, regression, alerts)"""
    # URL-decode model_name if necessary
    from urllib.parse import unquote
    model_name = unquote(model_name)
    
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}
    
    try:
        import sqlite3
        import statistics
        import math
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Get historical data
        cursor.execute('''
            SELECT timestamp, avg_tokens_per_sec FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        ''', (model_name,))
        
        data = cursor.fetchall()
        conn.close()
        
        if not data or len(data) < 2:
            return {"success": False, "error": "Insufficient data for statistical analysis"}
        
        # Extract speed values
        speeds = [row[1] for row in data]
        timestamps = [row[0] for row in data]
        
        # Calculate statistics
        mean = statistics.mean(speeds)
        variance = statistics.variance(speeds) if len(speeds) > 1 else 0
        std_dev = math.sqrt(variance)
        
        # Volatilität (Coefficient of Variation)
        volatility = (std_dev / mean * 100) if mean > 0 else 0
        
        # Linear Regression (y = mx + b)
        n = len(speeds)
        x_values = list(range(n))
        x_mean = statistics.mean(x_values)
        y_mean = mean
        
        numerator = sum((x_values[i] - x_mean) * (speeds[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Forecast for next 3 runs
        forecast = []
        for i in range(n, n + 3):
            predicted_speed = slope * i + intercept
            forecast.append(round(max(0, predicted_speed), 2))
        
        # Z-Score für Anomaly Detection
        z_scores = []
        if std_dev > 0:
            for speed in speeds:
                z = (speed - mean) / std_dev
                z_scores.append(round(z, 2))
        
        # Find anomalies (Z-score > 2 or < -2)
        anomalies = []
        for i, z in enumerate(z_scores):
            if abs(z) > 2:
                anomalies.append({
                    "index": i,
                    "timestamp": timestamps[i],
                    "speed": speeds[i],
                    "z_score": z,
                    "alert": "🔴 ANOMALY" if abs(z) > 2.5 else "🟠 WARNING"
                })
        
        # Performance Alert (Trend-basiert)
        recent_avg = statistics.mean(speeds[-3:]) if len(speeds) >= 3 else speeds[-1]
        overall_avg = mean
        performance_delta = ((recent_avg - overall_avg) / overall_avg * 100) if overall_avg > 0 else 0
        
        alert = ""
        if performance_delta < -10:
            alert = "🔴 PERFORMANCE REGRESSION"
        elif performance_delta > 10:
            alert = "🟢 PERFORMANCE IMPROVEMENT"
        else:
            alert = "⚪ STABLE"
        
        logger.info(f"📈 Advanced stats for {model_name}: σ={std_dev:.2f}, slope={slope:.4f}")
        
        return {
            "success": True,
            "model_name": model_name,
            "basic": {
                "mean": round(mean, 2),
                "median": round(statistics.median(speeds), 2),
                "min": round(min(speeds), 2),
                "max": round(max(speeds), 2)
            },
            "advanced": {
                "std_dev": round(std_dev, 2),
                "variance": round(variance, 2),
                "volatility_pct": round(volatility, 2),
                "coefficient_of_variation": round(std_dev / mean * 100, 2) if mean > 0 else 0
            },
            "trend": {
                "slope": round(slope, 4),
                "intercept": round(intercept, 2),
                "direction": "📈 UPWARD" if slope > 0.01 else "📉 DOWNWARD" if slope < -0.01 else "➡️ FLAT"
            },
            "forecast": {
                "next_3_runs": forecast,
                "confidence": "Medium" if len(speeds) >= 10 else "Low"
            },
            "anomalies": {
                "count": len(anomalies),
                "items": anomalies
            },
            "alert": {
                "status": alert,
                "recent_avg": round(recent_avg, 2),
                "overall_avg": round(overall_avg, 2),
                "delta_pct": round(performance_delta, 2)
            }
        }
    except Exception as e:
        logger.error(f"❌ Advanced statistics error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/output")
async def get_output() -> dict:
    """Returns current output"""
    return {
        "output": manager.current_output,
        "status": manager.status
    }


# ============================================================================
# A/B TESTING ENDPOINTS (PHASE 15)
# ============================================================================

@app.post("/api/experiments/create")
async def create_experiment(request: CreateExperimentRequest) -> dict:
    """Creates a new A/B testing experiment"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}
    
    try:
        import uuid
        
        experiment_id = str(uuid.uuid4())[:8]
        
        # Calculate parameter hashes
        baseline_dict = request.baseline_params.dict(exclude_none=True)
        test_dict = request.test_params.dict(exclude_none=True)
        
        baseline_hash = calculate_hash(baseline_dict)
        test_hash = calculate_hash(test_dict)
        
        logger.info(f"🧪 Experiment created: {experiment_id}")
        logger.info(f"   Model: {request.model_name}")
        logger.info(f"   Baseline: {baseline_dict} (hash: {baseline_hash})")
        logger.info(f"   Test: {test_dict} (hash: {test_hash})")
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": request.model_name,
            "baseline_hash": baseline_hash,
            "test_hash": test_hash,
            "baseline_params": baseline_dict,
            "test_params": test_dict,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Experiment creation error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/experiments/{experiment_id}/comparison")
async def get_experiment_comparison(
    experiment_id: str,
    baseline_hash: str,
    test_hash: str,
    model_name: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> dict:
    """Compares two parameter combinations for a model"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available", "comparison": {}}
    
    try:
        from urllib.parse import unquote
        import re
        
        model_name = unquote(model_name)
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Build query with optional date filters
        query = '''
            SELECT 
                timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                temperature, top_k_sampling, top_p_sampling, min_p_sampling,
                repeat_penalty, max_tokens, num_runs, error_count
            FROM benchmark_results
            WHERE model_name = ?
            ORDER BY timestamp ASC
        '''
        params = [model_name]
        
        if start_date:
            query = query.replace("ORDER BY", f"AND timestamp >= '{start_date}' ORDER BY")
        if end_date:
            query = query.replace("ORDER BY", f"AND timestamp <= '{end_date}' ORDER BY")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {"success": False, "error": "No data for this model", "comparison": {}}
        
        # Group data by parameters
        baseline_data = []
        test_data = []
        
        for row in rows:
            ts, speed, ttft, gen_time, temp, topk, topp, minp, penalty, maxts, runs, errors = row
            
            # Erstelle Parameter-Hash aus aktuellen Werten
            params_dict = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts
            }
            
            current_hash = calculate_hash(params_dict)
            
            # Vergleiche mit baseline_hash und test_hash (Substring-Match für Toleranz)
            if current_hash.startswith(baseline_hash[:8]) or baseline_hash.startswith(current_hash[:8]):
                baseline_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time
                })
            elif current_hash.startswith(test_hash[:8]) or test_hash.startswith(current_hash[:8]):
                test_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time
                })
        
        # Calculate statistics for both groups
        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]
        
        if not baseline_speeds or not test_speeds:
            return {
                "success": False,
                "error": f"Unzureichend Daten: Baseline={len(baseline_speeds)} entries, Test={len(test_speeds)} entries",
                "comparison": {}
            }
        
        # Statistiken
        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": round(statistics.stdev(baseline_speeds), 2) if len(baseline_speeds) > 1 else 0,
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data
        }
        
        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0,
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data
        }
        
        # Statistical test (t-test)
        test_result = perform_ttest(baseline_speeds, test_speeds)
        
        # Effect Size
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)
        
        # Determine winner (robust against division by zero)
        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        if baseline_mean and baseline_mean != 0:
            delta_pct = ((test_mean - baseline_mean) / baseline_mean * 100)
        else:
            # If baseline_mean == 0, define delta as difference relative to 0
            # and mark winner purely by larger mean
            delta_pct = None
            logger.warning("⚠️ Baseline mean is 0 – Delta% will not be calculated")
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            # Without significance: winner only if clearly better mean
            winner = "test" if test_mean > baseline_mean else ("baseline" if baseline_mean > test_mean else "tie")
        
        logger.info(f"🧪 Experiment {experiment_id}: {winner.upper()}")
        logger.info(f"   Baseline: {baseline_stats['mean']} ± {baseline_stats['std_dev']} tok/s")
        logger.info(f"   Test: {test_stats['mean']} ± {test_stats['std_dev']} tok/s")
        delta_str = f"{delta_pct:.1f}%" if isinstance(delta_pct, (int, float)) else "n/a"
        logger.info(f"   Delta: {delta_str} | p-value: {test_result.get('p_value')}")
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None),
                "winner": winner,
                "significant": test_result.get("significant", False),
                "confidence": "High" if test_result.get("p_value", 1) < 0.01 else "Medium" if test_result.get("p_value", 1) < 0.05 else "Low"
            }
        }
    except Exception as e:
        logger.error(f"❌ Experiment comparison error: {e}")
        return {"success": False, "error": str(e), "comparison": {}}


@app.post("/api/experiments/{experiment_id}/comparison")
async def post_experiment_comparison(
    experiment_id: str,
    request: Request
) -> dict:
    """Compares two parameter sets for a model (payload contains params)"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available", "comparison": {}}

    try:
        payload = await request.json()
        model_name = payload.get("model_name")
        baseline_params = payload.get("baseline_params", {})
        test_params = payload.get("test_params", {})
        start_date = payload.get("start_date")
        end_date = payload.get("end_date")

        if not model_name:
            return {"success": False, "error": "model_name missing", "comparison": {}}
        
        # Normalize model_name: remove quantization (@q3_k_l etc.)
        if "@" in model_name:
            model_name = model_name.split("@")[0]

        # DB query with optional date filters
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        query = '''
            SELECT 
                timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                temperature, top_k_sampling, top_p_sampling, min_p_sampling,
                repeat_penalty, max_tokens, num_runs, error_count
            FROM benchmark_results
            WHERE model_name = ?
        '''
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
            return {"success": False, "error": "No data for this model", "comparison": {}}

        baseline_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        for row in rows:
            ts, speed, ttft, gen_time, temp, topk, topp, minp, penalty, maxts, runs, errors = row
            params_dict = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts
            }

            if match_parameters(params_dict, baseline_params):
                baseline_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time
                })
            if match_parameters(params_dict, test_params):
                test_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time
                })

        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]

        if not baseline_speeds or not test_speeds:
            return {
                "success": False,
                "error": f"Unzureichend Daten: Baseline={len(baseline_speeds)} entries, Test={len(test_speeds)} entries",
                "comparison": {}
            }

        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": round(statistics.stdev(baseline_speeds), 2) if len(baseline_speeds) > 1 else 0,
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data
        }

        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0,
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data
        }

        test_result = perform_ttest(baseline_speeds, test_speeds)
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)

        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        delta_pct = None
        if baseline_mean and baseline_mean != 0:
            delta_pct = ((test_mean - baseline_mean) / baseline_mean * 100)
        else:
            logger.warning("⚠️ Baseline mean is 0 – Delta% will not be calculated")
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            winner = "test" if test_mean > baseline_mean else ("baseline" if baseline_mean > test_mean else "tie")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None),
                "winner": winner,
                "significant": test_result.get("significant", False),
                "confidence": "High" if test_result.get("p_value", 1) < 0.01 else "Medium" if test_result.get("p_value", 1) < 0.05 else "Low"
            }
        }
    except Exception as e:
        logger.error(f"❌ Experiment comparison error: {e}")
        return {"success": False, "error": str(e), "comparison": {}}


@app.post("/api/experiments/{experiment_id}/export")
async def export_experiment(
    experiment_id: str,
    request: Request
) -> dict:
    """Exports experiment results as CSV/PDF"""
    try:
        import csv
        from io import StringIO
        
        payload = await request.json()
        export_format = payload.get("format", "csv")  # "csv" or "pdf"
        
        # Get experiment data from payload
        baseline_data = payload.get("baseline", {})
        test_data = payload.get("test", {})
        comparison = payload.get("comparison", {})
        test_result = payload.get("statistical_test", {})
        
        if export_format == "csv":
            # CSV Export
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "Experiment ID", "Type", "Mean (tok/s)", "StdDev", "Min", "Max", "Count"
            ])
            
            # Baseline row
            writer.writerow([
                experiment_id,
                "Baseline",
                baseline_data.get("mean", "-"),
                baseline_data.get("std_dev", "-"),
                baseline_data.get("min", "-"),
                baseline_data.get("max", "-"),
                baseline_data.get("count", 0)
            ])
            
            # Test row
            writer.writerow([
                experiment_id,
                "Test",
                test_data.get("mean", "-"),
                test_data.get("std_dev", "-"),
                test_data.get("min", "-"),
                test_data.get("max", "-"),
                test_data.get("count", 0)
            ])
            
            # Statistical Results
            writer.writerow([])
            writer.writerow(["Statistical Test Results"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["p-value", test_result.get("p_value", "-")])
            writer.writerow(["t-statistic", test_result.get("t_statistic", "-")])
            writer.writerow(["Significant", test_result.get("significant", False)])
            writer.writerow(["Winner", comparison.get("winner", "-")])
            writer.writerow(["Delta %", comparison.get("delta_pct", "-")])
            
            csv_content = output.getvalue()
            export_file = RESULTS_DIR / f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            with open(export_file, 'w', encoding='utf-8') as f:
                f.write(csv_content)
            
            logger.info(f"📊 CSV Experiment Export: {export_file}")
            
            return {
                "success": True,
                "format": "csv",
                "file": str(export_file),
                "url": f"/results/{export_file.name}"
            }
        
        else:  # PDF
            # Simple text PDF (as in comparison export)
            lines = [
                f"Experiment {experiment_id}",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Baseline Results:",
                f"  Mean: {baseline_data.get('mean')} ± {baseline_data.get('std_dev')} tok/s",
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
                f"  Performance Delta: {comparison.get('delta_pct', '-')}%"
            ]
            
            # Generate simple PDF
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

                add_obj("3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
                add_obj(f"4 0 obj\n<< /Length {len(stream_content)} >>\nstream\n")
                pdf_bytes.extend(stream_content)
                pdf_bytes.extend(b"\nendstream\nendobj\n")
                add_obj("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

                xref_offset = len(pdf_bytes)
                pdf_bytes.extend(f"xref\n0 {len(offsets)+1}\n".encode("latin-1"))
                pdf_bytes.extend(b"0000000000 65535 f \n")
                for off in offsets:
                    pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
                pdf_bytes.extend(b"trailer\n")
                pdf_bytes.extend(f"<< /Size {len(offsets)+1} /Root 1 0 R >>\n".encode("latin-1"))
                pdf_bytes.extend(b"startxref\n")
                pdf_bytes.extend(f"{xref_offset}\n".encode("latin-1"))
                pdf_bytes.extend(b"%%EOF")
                return bytes(pdf_bytes)
            
            pdf_bytes = generate_simple_pdf(lines)
            export_file = RESULTS_DIR / f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            with open(export_file, 'wb') as f:
                f.write(pdf_bytes)
            
            logger.info(f"📋 PDF Experiment Export: {export_file}")
            
            return {
                "success": True,
                "format": "pdf",
                "file": str(export_file),
                "url": f"/results/{export_file.name}"
            }
    
    except Exception as e:
        logger.error(f"❌ Experiment export error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/experiments/run")
async def run_experiment(request: Request) -> dict:
    """Actively runs an A/B experiment: two benchmarks with specified parameters"""
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
        
        # Normalize model_name: remove quantization (@q3_k_l etc.)
        # DB only stores "qwen/qwen2.5-vl-7b", not "qwen/qwen2.5-vl-7b@q3_k_l"
        if "@" in model_name:
            model_name = model_name.split("@")[0]
        
        logger.info(f"🎯 Normalized model_name: {model_name}")
        
        # Remove 'name' field from params (only needed for frontend, not for matching)
        baseline_params.pop("name", None)
        test_params.pop("name", None)

        def build_args(param_set: Dict[str, Any]) -> List[str]:
            args: List[str] = []
            # Base parameters
            args.extend(["--runs", str(runs)])
            args.extend(["--context", str(context)])
            args.extend(["--limit", "1"])  # this model only
            args.extend(["--prompt", prompt])
            args.extend(["--include-models", model_name])
            args.append("--retest")  # force new runs instead of cache
            # Inference-Params (best guess mapping)
            if param_set.get("temperature") is not None:
                args.extend(["--temperature", str(param_set["temperature"])])
            if param_set.get("top_k") is not None:
                args.extend(["--top-k", str(param_set["top_k"])])
            if param_set.get("top_p") is not None:
                args.extend(["--top-p", str(param_set["top_p"])])
            if param_set.get("min_p") is not None:
                args.extend(["--min-p", str(param_set["min_p"])])
            if param_set.get("repeat_penalty") is not None:
                args.extend(["--repeat-penalty", str(param_set["repeat_penalty"])])
            if param_set.get("max_tokens") is not None:
                args.extend(["--max-tokens", str(param_set["max_tokens"])])
            # Load-Config Parameter
            if param_set.get("n_gpu_layers") is not None:
                args.extend(["--n-gpu-layers", str(param_set["n_gpu_layers"])])
            if param_set.get("n_batch") is not None:
                args.extend(["--n-batch", str(param_set["n_batch"])])
            if param_set.get("n_threads") is not None:
                args.extend(["--n-threads", str(param_set["n_threads"])])
            if param_set.get("flash_attention") is not None:
                if param_set["flash_attention"]:
                    args.append("--flash-attention")
                else:
                    args.append("--no-flash-attention")
            if param_set.get("rope_freq_base") is not None:
                args.extend(["--rope-freq-base", str(param_set["rope_freq_base"])])
            if param_set.get("rope_freq_scale") is not None:
                args.extend(["--rope-freq-scale", str(param_set["rope_freq_scale"])])
            if param_set.get("use_mmap") is not None:
                if param_set["use_mmap"]:
                    args.append("--use-mmap")
                else:
                    args.append("--no-mmap")
            if param_set.get("use_mlock") is not None and param_set["use_mlock"]:
                args.append("--use-mlock")
            if param_set.get("kv_cache_quant"):
                args.extend(["--kv-cache-quant", param_set["kv_cache_quant"]])
            # Keep profiling active
            args.append("--enable-profiling")
            return args

        async def run_once(args: List[str]) -> bool:
            return await manager.start_benchmark(args)

        # Run baseline
        baseline_args = build_args(baseline_params)
        logger.info(f"🎯 Baseline Args: {baseline_args}")
        baseline_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        baseline_ok = await run_once(baseline_args)
        if not baseline_ok:
            return {"success": False, "error": "Could not start baseline benchmark"}

        # Wait until benchmark is done
        while manager.is_running():
            await asyncio.sleep(1.0)
        # brief buffer time so results are written
        await asyncio.sleep(2.0)

        # Get baseline timestamp after the run
        baseline_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Run test
        test_args = build_args(test_params)
        logger.info(f"🎯 Test Args: {test_args}")
        test_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_ok = await run_once(test_args)
        if not test_ok:
            return {"success": False, "error": "Could not start test benchmark"}

        while manager.is_running():
            await asyncio.sleep(1.0)
        await asyncio.sleep(2.0)

        # Read results from DB - NEW STRATEGY: get the latest entries
        # Since there are now 3 entries per benchmark (run_index 0,1,2),
        # we get the last 20 entries and then filter by parameters
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Get the latest 20 entries for this model (should be 6: 3 baseline + 3 test)
        cursor.execute('''
            SELECT timestamp, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                   temperature, top_k_sampling, top_p_sampling, min_p_sampling, 
                   repeat_penalty, max_tokens, run_index,
                   n_gpu_layers, n_batch, n_threads, flash_attention, 
                   rope_freq_base, rope_freq_scale, use_mmap, use_mlock, kv_cache_quant
            FROM benchmark_results
            WHERE model_name = ? AND timestamp >= ?
            ORDER BY timestamp DESC, run_index DESC
            LIMIT 20
        ''', (model_name, baseline_start_str))
        all_rows = cursor.fetchall()

        conn.close()

        # Filter by parameters - extract both lists from all_rows
        baseline_data: List[Dict[str, Any]] = []
        test_data: List[Dict[str, Any]] = []

        logger.info(f"🔍 Total entries found: {len(all_rows)}")

        for row in all_rows:
            ts, speed, ttft, gen_time, temp, topk, topp, minp, penalty, maxts, run_idx, \
            gpu_layers, batch, threads, flash, rope_base, rope_scale, mmap, mlock, kv_quant = row
            
            row_params = {
                "temperature": temp,
                "top_k": topk,
                "top_p": topp,
                "min_p": minp,
                "repeat_penalty": penalty,
                "max_tokens": maxts,
                # Load-Config Parameter
                "n_gpu_layers": gpu_layers,
                "n_batch": batch,
                "n_threads": threads,
                "flash_attention": bool(flash) if flash is not None else None,
                "rope_freq_base": rope_base,
                "rope_freq_scale": rope_scale,
                "use_mmap": bool(mmap) if mmap is not None else None,
                "use_mlock": bool(mlock) if mlock is not None else None,
                "kv_cache_quant": kv_quant
            }
            
            # Match against baseline params
            if match_parameters(row_params, baseline_params):
                baseline_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time,
                    "run_index": run_idx
                })
                logger.info(f"✅ Baseline Match: run_index={run_idx}, speed={speed}, params={row_params}")
            # Match against test params
            elif match_parameters(row_params, test_params):
                test_data.append({
                    "timestamp": ts,
                    "speed": speed,
                    "ttft": ttft,
                    "gen_time": gen_time,
                    "run_index": run_idx
                })
                logger.info(f"✅ Test Match: run_index={run_idx}, speed={speed}, params={row_params}")
            else:
                logger.info(f"❌ No Match: run_index={run_idx}, params={row_params}")

        logger.info(f"🔍 After filtering: Baseline={len(baseline_data)}, Test={len(test_data)}")

        baseline_speeds = [d["speed"] for d in baseline_data if d["speed"] is not None]
        test_speeds = [d["speed"] for d in test_data if d["speed"] is not None]

        if not baseline_speeds or not test_speeds:
            return {
                "success": False,
                "error": f"Insufficient data after execution: Baseline={len(baseline_speeds)}, Test={len(test_speeds)}",
                "baseline": {"count": len(baseline_speeds)},
                "test": {"count": len(test_speeds)}
            }

        baseline_stats = {
            "count": len(baseline_speeds),
            "mean": round(statistics.mean(baseline_speeds), 2),
            "std_dev": round(statistics.stdev(baseline_speeds), 2) if len(baseline_speeds) > 1 else 0,
            "min": round(min(baseline_speeds), 2),
            "max": round(max(baseline_speeds), 2),
            "data": baseline_data
        }

        test_stats = {
            "count": len(test_speeds),
            "mean": round(statistics.mean(test_speeds), 2),
            "std_dev": round(statistics.stdev(test_speeds), 2) if len(test_speeds) > 1 else 0,
            "min": round(min(test_speeds), 2),
            "max": round(max(test_speeds), 2),
            "data": test_data
        }

        test_result = perform_ttest(baseline_speeds, test_speeds)
        effect_size = calculate_effect_size(baseline_speeds, test_speeds)
        baseline_mean = baseline_stats["mean"]
        test_mean = test_stats["mean"]
        delta_pct = None
        if baseline_mean and baseline_mean != 0:
            delta_pct = ((test_mean - baseline_mean) / baseline_mean * 100)
        else:
            logger.warning("⚠️ Baseline mean is 0 – Delta% will not be calculated (active run)")
        winner = "tie"
        if test_result.get("significant"):
            winner = "test" if test_mean > baseline_mean else "baseline"
        else:
            winner = "test" if test_mean > baseline_mean else ("baseline" if baseline_mean > test_mean else "tie")

        # Prepare results data
        results_data = {
            "success": True,
            "mode": "active",
            "model_name": model_name,
            "baseline": baseline_stats,
            "test": test_stats,
            "statistical_test": test_result,
            "effect_size": effect_size,
            "comparison": {
                "delta_pct": (round(delta_pct, 2) if isinstance(delta_pct, (int, float)) else None),
                "winner": winner,
                "significant": test_result.get("significant", False)
            },
            "experiment_info": {
                "name": experiment_name,
                "baseline_params": baseline_params,
                "test_params": test_params,
                "runs": runs,
                "context": context,
                "prompt": prompt,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Export results to files
        try:
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = PROJECT_ROOT / "results"
            results_dir.mkdir(exist_ok=True)
            
            # JSON Export
            json_file = results_dir / f"ab_test_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            logger.info(f"📄 A/B test JSON saved: {json_file}")
            
            # CSV Export
            csv_file = results_dir / f"ab_test_results_{timestamp}.csv"
            with open(csv_file, 'w', encoding='utf-8') as f:
                f.write(f"Experiment Name: {experiment_name}\n")
                f.write("Metric,Baseline,Test,Delta\n")
                f.write(f"Model,{model_name},{model_name},-\n")
                f.write(f"Mean Speed (tok/s),{baseline_stats['mean']},{test_stats['mean']},{results_data['comparison']['delta_pct']}%\n")
                f.write(f"Min Speed (tok/s),{baseline_stats['min']},{test_stats['min']},-\n")
                f.write(f"Max Speed (tok/s),{baseline_stats['max']},{test_stats['max']},-\n")
                f.write(f"Std Dev,{baseline_stats['std_dev']},{test_stats['std_dev']},-\n")
                f.write(f"Count,{baseline_stats['count']},{test_stats['count']},-\n")
                f.write(f"Temperature,{baseline_params.get('temperature', 'N/A')},{test_params.get('temperature', 'N/A')},-\n")
                f.write(f"Top-K,{baseline_params.get('top_k', 'N/A')},{test_params.get('top_k', 'N/A')},-\n")
                f.write(f"Top-P,{baseline_params.get('top_p', 'N/A')},{test_params.get('top_p', 'N/A')},-\n")
                f.write(f"Winner,-,-,{winner}\n")
                f.write(f"Significant,-,-,{test_result.get('significant', False)}\n")
            logger.info(f"📊 A/B test CSV saved: {csv_file}")
            
            # HTML Export (simple)
            html_file = results_dir / f"ab_test_results_{timestamp}.html"
            html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{experiment_name} - {model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .winner {{ font-weight: bold; color: #27ae60; font-size: 1.2em; }}
        .metric {{ font-weight: 600; }}
        .params {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 {experiment_name}</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Timestamp:</strong> {results_data['experiment_info']['timestamp']}</p>
        
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
            <p>Temperature: {baseline_params.get('temperature', 'N/A')}, Top-K: {baseline_params.get('top_k', 'N/A')}, Top-P: {baseline_params.get('top_p', 'N/A')}</p>
        </div>
        <div class="params">
            <h3>Test</h3>
            <p>Temperature: {test_params.get('temperature', 'N/A')}, Top-K: {test_params.get('top_k', 'N/A')}, Top-P: {test_params.get('top_p', 'N/A')}</p>
        </div>
        
        <h2>📈 Statistical Analysis</h2>
        <p><strong>Winner:</strong> <span class="winner">{winner.upper()}</span></p>
        <p><strong>Statistically Significant:</strong> {test_result.get('significant', False)}</p>
        <p><strong>p-value:</strong> {test_result.get('p_value', 'N/A')}</p>
        <p><strong>Effect Size (Cohen's d):</strong> {effect_size.get('cohens_d', 'N/A')} ({effect_size.get('effect_magnitude', 'N/A')})</p>
    </div>
</body>
</html>"""
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"🌐 A/B test HTML saved: {html_file}")
            
            # PDF Export (if reportlab available)
            try:
                from reportlab.lib.pagesizes import A4, landscape
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                
                pdf_file = results_dir / f"ab_test_results_{timestamp}.pdf"
                doc = SimpleDocTemplate(str(pdf_file), pagesize=landscape(A4))
                elements = []
                styles = getSampleStyleSheet()
                
                # Title
                elements.append(Paragraph(f"<b>{experiment_name}</b>", styles['Title']))
                elements.append(Paragraph(f"Model: {model_name}", styles['Heading2']))
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(f"Timestamp: {results_data['experiment_info']['timestamp']}", styles['Normal']))
                elements.append(Spacer(1, 20))
                
                # Performance Table
                data = [
                    ['Metric', 'Baseline', 'Test', 'Delta'],
                    ['Mean Speed (tok/s)', f"{baseline_stats['mean']}", f"{test_stats['mean']}", f"{results_data['comparison']['delta_pct']}%"],
                    ['Min Speed', f"{baseline_stats['min']}", f"{test_stats['min']}", '-'],
                    ['Max Speed', f"{baseline_stats['max']}", f"{test_stats['max']}", '-'],
                    ['Std Dev', f"±{baseline_stats['std_dev']}", f"±{test_stats['std_dev']}", '-'],
                    ['Count', str(baseline_stats['count']), str(test_stats['count']), '-'],
                ]
                
                table = Table(data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                elements.append(Spacer(1, 20))
                
                # Winner
                elements.append(Paragraph(f"<b>Winner:</b> {winner.upper()}", styles['Heading2']))
                elements.append(Paragraph(f"<b>Significant:</b> {test_result.get('significant', False)}", styles['Normal']))
                
                doc.build(elements)
                logger.info(f"📑 A/B test PDF saved: {pdf_file}")
            except ImportError:
                logger.warning("⚠️ reportlab not installed - PDF export skipped")
            except Exception as pdf_error:
                logger.error(f"❌ PDF export error: {pdf_error}")
            
            results_data['exports'] = {
                'json': str(json_file),
                'csv': str(csv_file),
                'html': str(html_file)
            }
        except Exception as export_error:
            logger.error(f"❌ Export error: {export_error}")
            # Continue anyway - don't fail the entire request
        
        return results_data
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Experiment run error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}

@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> dict:
    """Dashboard statistics for home view"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache not available"}
    
    try:
        import platform
        import sqlite3
        import psutil
        import json
        import subprocess
        from datetime import datetime
        
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()

        # Load capabilities from model_metadata.db
        capabilities_by_model: Dict[str, List[str]] = {}
        distinct_capabilities: set[str] = set()
        try:
            if METADATA_DATABASE_FILE.exists():
                mconn = sqlite3.connect(METADATA_DATABASE_FILE)
                mcur = mconn.cursor()
                mcur.execute("SELECT model_key, capabilities FROM model_metadata WHERE capabilities IS NOT NULL AND TRIM(capabilities) <> ''")
                for mk, caps_json in mcur.fetchall():
                    try:
                        caps = json.loads(caps_json) if caps_json else []
                        if isinstance(caps, list):
                            capabilities_by_model[mk] = caps
                            for c in caps:
                                distinct_capabilities.add(c)
                    except Exception:
                        continue
                mconn.close()
        except Exception as _meta_err:
            # Not critical for dashboard
            pass

        # LM Studio health check - ROBUST via HTTP API ping
        # Primary: HTTP GET on LM Studio API (port 1234 default)
        # Fallback: CLI `lms status` with "Server: OFF" detection
        lmstudio_health = {"ok": False, "status": "offline"}
        lmstudio_ports = LMSTUDIO_PORTS  # LM Studio default ports (NOT 8080 - often other services)
        
        # 1. HTTP API Check (schnellster und zuverlässigster Weg)
        for port in lmstudio_ports:
            try:
                with httpx.Client(timeout=1.5) as client:
                    resp = client.get(f"http://{LMSTUDIO_HOST}:{port}/v1/models")
                    if resp.status_code == 200:
                        lmstudio_health = {"ok": True, "status": f"online ({LMSTUDIO_HOST}:{port})", "version": None}
                        break
            except Exception:
                continue
        
        # 2. Fallback: CLI check when HTTP fails
        if not lmstudio_health["ok"]:
            try:
                result = subprocess.run(["lms", "status"], capture_output=True, text=True, timeout=2)
                text = (result.stdout + result.stderr).lower()
                # Explicit offline markers (incl. "server:  off" and "server: off")
                offline_keywords = ["server:  off", "server: off", "off", "not running", "stopped", "offline", "no server", "not loaded", "error", "failed"]
                is_offline = any(kw in text for kw in offline_keywords) or result.returncode != 0
                # Explicit online markers
                online_keywords = ["server:  on", "server: on", "running", "listening", "ready"]
                is_online = any(kw in text for kw in online_keywords) and not is_offline
                if is_online:
                    lmstudio_health = {"ok": True, "status": "online (cli)", "version": None}
            except Exception:
                pass
        
        # No more process check - too unreliable (autostart scripts give false positives)
        
        # System info (extended with better details)
        system_info = {
            "os": platform.system(),
            "os_version": platform.release(),  # Kernel-Version
            "python_version": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=False),  # Physical cores
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # Try to detect Linux distribution
        if system_info["os"] == "Linux":
            try:
                import distro
                distro_name = distro.name()
                distro_version = distro.version()
                if distro_name:
                    system_info["os"] = f"{distro_name} {distro_version}".strip()
            except:
                # Fallback: try reading /etc/os-release
                try:
                    with open('/etc/os-release', 'r') as f:
                        os_release = {}
                        for line in f:
                            if '=' in line:
                                key, val = line.strip().split('=', 1)
                                os_release[key] = val.strip('"')
                        if 'PRETTY_NAME' in os_release:
                            system_info["os"] = os_release['PRETTY_NAME']
                        elif 'NAME' in os_release:
                            version = os_release.get('VERSION', '')
                            system_info["os"] = f"{os_release['NAME']} {version}".strip()
                except:
                    pass
        
        # Better CPU name via cpuinfo if available
        cpu_gpu_series = None  # For iGPU extraction from CPU string
        try:
            import cpuinfo
            cpu_data = cpuinfo.get_cpu_info()
            if 'brand_raw' in cpu_data and cpu_data['brand_raw']:
                raw_cpu = cpu_data['brand_raw'].replace('®', '').replace('™', '').strip()
                system_info["cpu"] = raw_cpu
 
                # Extract iGPU model from CPU string (e.g. "w/ Radeon 890M")
                if 'Radeon' in raw_cpu:
                    import re
                    radeon_match = re.search(r'Radeon\s+(\d+[A-Za-z]*)', raw_cpu)
                    if radeon_match:
                        cpu_gpu_series = f"AMD Radeon {radeon_match.group(1)}"
        except:
            pass

        # GPU info - improved detection with nvidia-smi/rocm-smi
        gpu_info = None
        try:
            import subprocess
            import glob
            import re

            gpu_type = "Unknown"
            gpu_model = "Unknown"
            vram_total_gb = None
            gtt_total_gb = None

            # Determine GPU type and model from results (if available)
            # The benchmark DB now contains full GPU model names ("AMD Radeon 890M" instead of just "AMD")
            if results:
                gpu_model = results[0].gpu_type  # Use full name from DB
                # Extrahiere GPU-Typ (NVIDIA, AMD, Intel) aus Modellnamen
                if "NVIDIA" in gpu_model or "GeForce" in gpu_model or "RTX" in gpu_model or "GTX" in gpu_model:
                    gpu_type = "NVIDIA"
                elif "AMD" in gpu_model or "Radeon" in gpu_model:
                    gpu_type = "AMD"
                elif "Intel" in gpu_model or "Arc" in gpu_model or "Iris" in gpu_model:
                    gpu_type = "Intel"
                else:
                    gpu_type = gpu_model

            # NVIDIA GPU detection
            try:
                # Get VRAM size
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], timeout=5)
                vram_total_mb = int(output.decode().strip().split('\n')[0])
                vram_total_gb = round(vram_total_mb / 1024, 2)
                gpu_type = "NVIDIA"

                # Get GPU model
                try:
                    model_output = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], timeout=5)
                    gpu_model = model_output.decode().strip().split('\n')[0]
                except Exception:
                    gpu_model = "NVIDIA GPU"
    
            except (TimeoutExpired, FileNotFoundError, ValueError):
                pass

            # AMD GPU detection with rocm-smi
            if not vram_total_gb:
                try:
                    # Search for rocm-smi in various paths
                    rocm_tool = None
                    for path in ['/usr/bin/rocm-smi', '/usr/local/bin/rocm-smi'] + glob.glob('/opt/rocm-*/bin/rocm-smi'):
                        if Path(path).exists():
                            rocm_tool = path
                            break

                    if not rocm_tool:
                        rocm_tool = 'rocm-smi'  # Fallback auf PATH

                    # AMD GPU card series mapping based on device ID
                    amd_device_mapping = {
                        '150e': 'Radeon Graphics',  # Generic iGPU - will be overridden by CPU info
                        '7340': 'Radeon RX 5700 XT',
                        '731f': 'Radeon RX 5700',
                        '7360': 'Radeon RX 6700 XT',
                        '73bf': 'Radeon RX 6600 XT',
                        '73df': 'Radeon RX 6600',
                        '15c8': 'Radeon RX 7600 XT',
                        '5450': 'Radeon RX 6800 XT',
                        '5498': 'Radeon RX 6900 XT',
                        '5780': 'Radeon Pro W6800X',
                        'gfx906': 'Radeon RX 5700 XT (Navi)',
                        'gfx908': 'MI100',
                        'gfx90a': 'MI250',
                    }

                    # Get GPU device ID from lspci or sysfs
                    gpu_series = None
                    device_id = None

                    try:
                        # Try to get device ID from lspci (for dedicated GPUs)
                        lspci_output = subprocess.run(
                            ['lspci', '-d', '1002::0300,1002::0380'],  # VGA/Display Controller
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if lspci_output.returncode == 0 and lspci_output.stdout:
                            # Parse "c7:00.0 0380: 1002:150e (rev c1)"
                            for line in lspci_output.stdout.strip().split('\n'):
                                if '1002:' in line:
                                    # Extract device ID after ':'
                                    parts = line.split('1002:')
                                    if len(parts) > 1:
                                        dev_part = parts[1].split()[0]  # "150e" from "150e (rev c1)"
                                        device_id = dev_part.lower()
                                        # Versuche mit lspci -vv die echte Kartenserie zu holen
                                        lspci_slot = line.split()[0]  # "c7:00.0"
                                        try:
                                            detail_output = subprocess.run(
                                                ['lspci', '-s', lspci_slot, '-v'],
                                                capture_output=True,
                                                text=True,
                                                timeout=5
                                            )
                                            if detail_output.returncode == 0:
                                                detail_text = detail_output.stdout
                                                # Try to extract card name (e.g. from device type strings)
                                                if 'Radeon' in detail_text:
                                                    # Extract card name if available
                                                    for detail_line in detail_text.split('\n'):
                                                        if 'Radeon' in detail_line:
                                                            gpu_series = detail_line.strip()
                                                            break
                                        except:
                                            pass
                                        break
                    except (FileNotFoundError, TimeoutExpired):
                        pass

                    # Fallback: get device ID from /sys
                    if not device_id:
                        try:
                            for gpu_path in Path('/sys/devices').glob('**/pci*/*/0000:c7:00.0/device'):
                                with open(gpu_path, 'r') as f:
                                    dev_id_hex = f.read().strip()  # "0x150e"
                                    device_id = dev_id_hex.replace('0x', '')
                                    break
                        except:
                            pass

                    # Get gfx code from rocm-smi as fallback
                    gfx_code = None
                    try:
                        result = subprocess.run(
                            [rocm_tool, '--showproductname'],
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
                                    break
                    except Exception:
                        pass

                    # Assemble GPU model name with fallback chain
                    # 1. Highest priority: GPU series extracted from CPU string (e.g. "Radeon 890M" from CPU name)
                    if cpu_gpu_series:
                        gpu_model = cpu_gpu_series
                    # 2. lspci card series if available
                    elif gpu_series and 'Radeon' in gpu_series:
                        gpu_model = gpu_series
                    # 3. Device ID Mapping
                    elif device_id and device_id in amd_device_mapping:
                        gpu_model = f"AMD {amd_device_mapping[device_id]}"
                    # 4. GFX-Code Mapping
                    elif gfx_code and gfx_code in amd_device_mapping:
                        gpu_model = f"AMD {amd_device_mapping[gfx_code]}"
                    # 5. Fallback: show device ID
                    elif device_id:
                        gpu_model = f"AMD GPU (Device: 1002:{device_id})"
                    # 6. Fallback: GFX-Code
                    elif gfx_code:
                        gpu_model = f"AMD {gfx_code}"
                    # 7. Fallback: generic
                    else:
                        gpu_model = "AMD GPU"

                    # Get VRAM size
                    result = subprocess.run(
                        [rocm_tool, '--showmeminfo', 'vram'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode == 0:
                        # Parse "GPU[0] : VRAM Total Memory (B): 33550180352"
                        for line in result.stdout.split('\n'):
                            if 'VRAM Total Memory' in line:
                                match = re.search(r':\s*(\d+)\s*$', line.strip())
                                if match:
                                    vram_bytes = int(match.group(1))
                                    vram_total_gb = round(vram_bytes / (1024**3), 2)
                                    gpu_type = "AMD"
                                    break

                    # Get GTT size (AMD specific)
                    if gpu_type == "AMD":
                        result = subprocess.run(
                            [rocm_tool, '--showmeminfo', 'gtt'],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )

                        if result.returncode == 0:
                            # Parse "GPU[0] : GTT Total Memory (B): 98618482688"
                            for line in result.stdout.split('\n'):
                                if 'GTT Total Memory' in line:
                                    match = re.search(r':\s*(\d+)\s*$', line.strip())
                                    if match:
                                        gtt_bytes = int(match.group(1))
                                        gtt_total_gb = round(gtt_bytes / (1024**3), 2)
                                        break

                except (TimeoutExpired, FileNotFoundError):
                    pass

            # Assemble GPU info
            gpu_info = {
                "type": gpu_type,
                "model": gpu_model,
                "vram_gb": vram_total_gb,
                "gtt_gb": gtt_total_gb if gtt_total_gb else system_info["ram_gb"],
                "total_gb": (vram_total_gb + gtt_total_gb) if (vram_total_gb and gtt_total_gb) else (vram_total_gb or system_info["ram_gb"])
            }

        except Exception:
            gpu_info = {
                "type": "Unknown",
                "vram_gb": None,
                "gtt_gb": system_info["ram_gb"],
                "total_gb": system_info["ram_gb"]
            }

        # Cache statistics
        cache_stats = {
            "total_models": len(results),
            "total_runs": len(results),
            "db_size_mb": round(DATABASE_FILE.stat().st_size / (1024 * 1024), 2) if DATABASE_FILE.exists() else 0
        }

        # Performance statistics
        perf_stats = {}
        if results:
            speeds = [r.avg_tokens_per_sec for r in results]
            perf_stats = {
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "min_speed": round(min(speeds), 2)
            }

        # Top 5 fastest models
        top_models = []
        fastest_model = None
        if results:
            sorted_results = sorted(results, key=lambda r: r.avg_tokens_per_sec, reverse=True)
            for i, r in enumerate(sorted_results[:5]):
                base_key = r.model_name.split('@')[0]
                top_models.append({
                    "model_name": r.model_name,
                    "quantization": r.quantization,
                    "speed": round(r.avg_tokens_per_sec, 2),
                    "vram_mb": r.vram_mb,
                    "params_size": r.params_size,
                    "capabilities": capabilities_by_model.get(r.model_name, []),
                    "source_url": f"https://lmstudio.ai/models/{base_key}"
                })
                # Save fastest model
                if i == 0:
                    fastest_model = {
                        "name": r.model_name,
                        "speed": round(r.avg_tokens_per_sec, 2),
                        "capabilities": capabilities_by_model.get(r.model_name, [])
                    }

        # Last 10 benchmark runs (with timestamp)
        recent_runs = []
        last_run_timestamp = None
        if results:
            sorted_by_time = sorted(results, key=lambda r: r.timestamp, reverse=True)
            for i, r in enumerate(sorted_by_time[:10]):
                base_key = r.model_name.split('@')[0]
                recent_runs.append({
                    "model_name": r.model_name,
                    "quantization": r.quantization,
                    "speed": round(r.avg_tokens_per_sec, 2),
                    "timestamp": r.timestamp,
                    "gpu_offload": r.gpu_offload,
                    "capabilities": capabilities_by_model.get(r.model_name, []),
                    "source_url": f"https://lmstudio.ai/models/{base_key}"
                })
                # Save last run timestamp
                if i == 0:
                    last_run_timestamp = r.timestamp

        return {
            "success": True,
            "system_info": system_info,
            "gpu_info": gpu_info,
            "cache_stats": cache_stats,
            "perf_stats": perf_stats,
            "top_models": top_models,
            "recent_runs": recent_runs,
            "fastest_model": fastest_model,
            "capability_catalog": sorted(list(distinct_capabilities)) if distinct_capabilities else [],
            "last_run": last_run_timestamp,
            "lmstudio": lmstudio_health
        }
    except Exception as e:
        logger.error(f"❌ Error loading dashboard stats: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# WebSocket - Live Streaming
# ============================================================================

@app.websocket("/ws/benchmark")
async def websocket_benchmark(websocket: WebSocket):
    """WebSocket for live streaming of benchmark output"""
    await websocket.accept()
    manager.connected_clients.add(websocket)

    heartbeat_count = 0

    try:
        logger.info(f"✅ WebSocket client connected (total: {len(manager.connected_clients)})")

        await websocket.send_json({
            "type": "status",
            "status": manager.status,
            "running": manager.is_running()
        })

        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                logger.debug(f"📨 Client message: {data}")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                logger.info("⚠️ WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket Receive Error: {e}")
                break

            if manager.is_running():
                output = manager.drain_output_queue()
                if output:
                    try:
                        await websocket.send_json({
                            "type": "output",
                            "line": output,
                            "status": manager.status
                        })
                        heartbeat_count = 0
                    except Exception as e:
                        logger.error(f"❌ WebSocket Send Error: {e}")
                        break

                current_time = time.time()
                if current_time - manager.last_hardware_send_time >= 2.0:
                    if manager.hardware_history["temperatures"] or manager.hardware_history["power"] or manager.hardware_history["vram"] or manager.hardware_history["gtt"] or manager.hardware_history["cpu"] or manager.hardware_history["ram"]:
                        try:
                            max_history = 60
                            hardware_data = {
                                "temperatures": manager.hardware_history["temperatures"][-max_history:],
                                "power": manager.hardware_history["power"][-max_history:],
                                "vram": manager.hardware_history["vram"][-max_history:],
                                "gtt": manager.hardware_history["gtt"][-max_history:],
                                "cpu": manager.hardware_history["cpu"][-max_history:],
                                "ram": manager.hardware_history["ram"][-max_history:]
                            }

                            await websocket.send_json({
                                "type": "hardware",
                                "data": hardware_data
                            })
                            manager.last_hardware_send_time = current_time
                        except Exception as e:
                            logger.error(f"❌ WebSocket Hardware Send Error: {e}")
            else:
                heartbeat_count += 1
                if heartbeat_count % 2 == 0:
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "status": manager.status,
                            "running": manager.is_running()
                        })
                    except Exception as e:
                        logger.error(f"❌ WebSocket Heartbeat Error: {e}")
                        break

                if manager.status == "completed":
                    try:
                        await websocket.send_json({
                            "type": "completed",
                            "message": "✅ Benchmark completed"
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Could not send completion message: {e}")
                    finally:
                        manager.status = "idle"

                await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("ℹ️ WebSocket normal disconnect")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}")
    finally:
        manager.connected_clients.discard(websocket)
        logger.info(f"❌ WebSocket client disconnected (total: {len(manager.connected_clients)})")


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
    except Exception as e:
        logger.error(f"Error finding latest results: {e}")
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
        "connected_clients": len(manager.connected_clients)
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # ArgumentParser für Port-Option
    parser = argparse.ArgumentParser(description="FastAPI web dashboard for LM Studio Benchmark")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for web dashboard (default: automatically find a free port)"
    )
    args = parser.parse_args()
    
    # Create WebApp startup log file
    webapp_log_file = setup_webapp_logger()
    
    logger.info("🌐 Starting FastAPI web dashboard...")
    logger.info(f"📝 WebApp log: {webapp_log_file}")
    logger.info(f"📁 Project root: {PROJECT_ROOT}")
    logger.info(f"📄 Benchmark script: {BENCHMARK_SCRIPT}")
    
    if not BENCHMARK_SCRIPT.exists():
        logger.error(f"❌ Benchmark script not found: {BENCHMARK_SCRIPT}")
        sys.exit(1)
    
    # Determine port
    if args.port:
        port = args.port
        logger.info(f"🔧 Using specified port: {port}")
    else:
        port = find_free_port()
        logger.info(f"🎲 Using automatically found free port: {port}")
    
    dashboard_url = f"http://localhost:{port}"
    logger.info(f"🚀 Dashboard verfügbar auf {dashboard_url}")
    logger.info(f"📊 API Docs: {dashboard_url}/docs")
    
    # Öffne Browser in separatem Thread nach kurzer Verzögerung
    def open_browser():
        time.sleep(1.5)  # Warte bis Server bereit ist
        try:
            logger.info(f"🌐 Öffne Browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.warning(f"⚠️ Konnte Browser nicht öffnen: {e}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
