#!/usr/bin/env python3
"""
CLI entrypoint for capability-driven benchmark agent.

Provides command-line interface to run benchmarks on models.
"""

import argparse
import copy
import json
import logging
from pathlib import Path
import platform
import random
import re
import sqlite3
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Optional, SupportsFloat, SupportsIndex, TypedDict

from agents.runner import BenchmarkRunner
from cli.reporting import HTMLReporter, sanitize_report_name
from core.logging_utils import install_level_icons
from core.paths import USER_RESULTS_DIR
from tools.hardware_monitor import GPUMonitor, HardwareMonitor

try:
    import cpuinfo as _cpuinfo
except ModuleNotFoundError:
    CPUINFO: Optional[ModuleType] = None
else:
    CPUINFO = _cpuinfo

try:
    import distro as _distro
except ModuleNotFoundError:
    DISTRO: Optional[ModuleType] = None
else:
    DISTRO = _distro

try:
    import lmstudio as _lmstudio
except ModuleNotFoundError:
    LMSTUDIO_MODULE: Optional[ModuleType] = None
else:
    LMSTUDIO_MODULE = _lmstudio

try:
    import yaml as _yaml
    from yaml import YAMLError as _yaml_error
except ModuleNotFoundError:
    YAML: Optional[ModuleType] = None

    class _FallbackYAMLError(Exception):
        """Fallback YAML error type when PyYAML is unavailable."""

    YamlError: type[Exception] = _FallbackYAMLError
else:
    YAML = _yaml
    YamlError = _yaml_error

REPORTLAB_COLORS = None
REPORTLAB_A4 = None
REPORTLAB_GET_SAMPLE_STYLE_SHEET = None
REPORTLAB_CM = None
REPORTLAB_PARAGRAPH_CLS: Optional[Any] = None
REPORTLAB_SIMPLE_DOC_TEMPLATE_CLS: Optional[Any] = None
REPORTLAB_SPACER_CLS: Optional[Any] = None
REPORTLAB_TABLE_CLS: Optional[Any] = None
REPORTLAB_TABLE_STYLE_CLS: Optional[Any] = None

try:
    from reportlab.lib import colors as _reportlab_colors
    from reportlab.lib.pagesizes import A4 as _reportlab_a4
    from reportlab.lib.styles import (
        getSampleStyleSheet as _reportlab_get_sample_style_sheet,
    )
    from reportlab.lib.units import cm as _reportlab_cm
    from reportlab.platypus import Paragraph as _reportlab_paragraph
    from reportlab.platypus import SimpleDocTemplate as _reportlab_simple_doc_template
    from reportlab.platypus import Spacer as _reportlab_spacer
    from reportlab.platypus import Table as _reportlab_table
    from reportlab.platypus import TableStyle as _reportlab_table_style
except ModuleNotFoundError:
    REPORTLAB_AVAILABLE = False
else:
    REPORTLAB_COLORS = _reportlab_colors
    REPORTLAB_A4 = _reportlab_a4
    REPORTLAB_GET_SAMPLE_STYLE_SHEET = _reportlab_get_sample_style_sheet
    REPORTLAB_CM = _reportlab_cm
    REPORTLAB_PARAGRAPH_CLS = _reportlab_paragraph  # pylint: disable=invalid-name
    REPORTLAB_SIMPLE_DOC_TEMPLATE_CLS = _reportlab_simple_doc_template  # pylint: disable=invalid-name
    REPORTLAB_SPACER_CLS = _reportlab_spacer  # pylint: disable=invalid-name
    REPORTLAB_TABLE_CLS = _reportlab_table  # pylint: disable=invalid-name
    REPORTLAB_TABLE_STYLE_CLS = _reportlab_table_style  # pylint: disable=invalid-name
    REPORTLAB_AVAILABLE = True


BENCHMARK_DB_PATH = USER_RESULTS_DIR / "benchmark_cache.db"


def _get_app_version() -> str:
    """Read app version from VERSION file."""
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        if version_file.exists():
            return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return "unknown"


def _run_command(cmd: list[str], timeout: int = 5) -> Optional[str]:
    """Run a subprocess command and return stdout when successful."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output if output else None


def _get_lmstudio_version() -> Optional[str]:
    """Get LM Studio version string from CLI."""
    output = _run_command(["lms", "version"])
    if not output:
        return None

    match = re.search(r"v\d+\.\d+\.\d+", output)
    if match:
        return match.group(0)

    m2 = re.search(r"CLI commit:\s*([0-9a-fA-F]{6,40})", output)
    if m2:
        commit = m2.group(1)
        pkg_ver = (
            getattr(LMSTUDIO_MODULE, "__version__", None)
            if LMSTUDIO_MODULE is not None
            else None
        )
        if pkg_ver:
            return f"{pkg_ver} (commit:{commit})"
        return f"commit:{commit}"

    for line in output.splitlines():
        stripped = line.strip()
        if stripped and not set(stripped).issubset(set(" _/|\\")):
            return stripped
    return None


def _get_driver_versions() -> dict[str, Optional[str]]:
    """Collect GPU driver versions across NVIDIA/AMD/Intel tools."""
    nvidia = _run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    rocm = _run_command(["rocm-smi", "--version"])
    intel = _run_command(["intel_gpu_top", "--help"])

    if nvidia:
        nvidia = nvidia.splitlines()[0].strip()
    if rocm:
        rocm = rocm.splitlines()[0].strip()
    if intel:
        intel = intel.splitlines()[0].strip()

    return {
        "nvidia_driver_version": nvidia,
        "rocm_driver_version": rocm,
        "intel_driver_version": intel,
    }


def _get_os_info() -> tuple[Optional[str], Optional[str]]:
    """Get operating system name and version."""
    try:
        if platform.system() == "Linux" and DISTRO is not None:
            return DISTRO.name(), DISTRO.version()
        return platform.system(), platform.release()
    except OSError:
        return None, None


def _get_cpu_model() -> Optional[str]:
    """Get CPU model if available."""
    if CPUINFO is None:
        return None
    try:
        info = CPUINFO.get_cpu_info()
        brand = info.get("brand_raw", "")
        return brand if brand else None
    except OSError:
        return None


def _build_classic_metrics(
    config: dict[str, Any],
    report_data: dict[str, Any],
    gpu_monitor: GPUMonitor,
    benchmark_duration_seconds: float,
) -> dict[str, Any]:
    """Map capability run metadata into classic benchmark metric fields."""
    summary = report_data.get("summary", {})
    os_name, os_version = _get_os_info()
    driver_versions = _get_driver_versions()

    metrics: dict[str, Any] = {
        "error_count": len(
            [r for r in report_data.get("results", []) if r.get("error")]
        ),
        "gpu_type": gpu_monitor.gpu_type,
        "gpu_offload": config.get("gpu_offload"),
        "vram_mb": gpu_monitor.get_vram_usage(),
        "context_length": config.get("context_length"),
        "temperature": config.get("temperature"),
        "top_k_sampling": config.get("top_k"),
        "top_p_sampling": config.get("top_p"),
        "min_p_sampling": config.get("min_p"),
        "repeat_penalty": config.get("repeat_penalty"),
        "max_tokens": config.get("max_tokens"),
        "n_gpu_layers": config.get("n_gpu_layers"),
        "n_batch": config.get("n_batch"),
        "n_threads": config.get("n_threads"),
        "flash_attention": config.get("flash_attention"),
        "rope_freq_base": config.get("rope_freq_base"),
        "rope_freq_scale": config.get("rope_freq_scale"),
        "use_mmap": config.get("use_mmap"),
        "use_mlock": config.get("use_mlock"),
        "kv_cache_quant": config.get("kv_cache_quant"),
        "lmstudio_version": _get_lmstudio_version(),
        "app_version": _get_app_version(),
        "os_name": os_name,
        "os_version": os_version,
        "cpu_model": _get_cpu_model(),
        "python_version": platform.python_version(),
        "benchmark_duration_seconds": benchmark_duration_seconds,
        "temp_celsius_min": summary.get("temp_celsius_min"),
        "temp_celsius_max": summary.get("temp_celsius_max"),
        "temp_celsius_avg": summary.get("temp_celsius_avg"),
        "power_watts_min": summary.get("power_watts_min"),
        "power_watts_max": summary.get("power_watts_max"),
        "power_watts_avg": summary.get("power_watts_avg"),
        "vram_gb_min": summary.get("vram_gb_min"),
        "vram_gb_max": summary.get("vram_gb_max"),
        "vram_gb_avg": summary.get("vram_gb_avg"),
        "gtt_gb_min": summary.get("gtt_gb_min"),
        "gtt_gb_max": summary.get("gtt_gb_max"),
        "gtt_gb_avg": summary.get("gtt_gb_avg"),
        "cpu_percent_min": summary.get("cpu_percent_min"),
        "cpu_percent_max": summary.get("cpu_percent_max"),
        "cpu_percent_avg": summary.get("cpu_percent_avg"),
        "ram_gb_min": summary.get("ram_gb_min"),
        "ram_gb_max": summary.get("ram_gb_max"),
        "ram_gb_avg": summary.get("ram_gb_avg"),
    }
    metrics.update(driver_versions)
    return metrics


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    install_level_icons()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(level_icon)s %(message)s",
        force=True,
    )

    httpcore_level = logging.INFO if verbose else logging.WARNING
    for name in ("httpcore", "httpcore.connection", "httpcore.http11"):
        logging.getLogger(name).setLevel(httpcore_level)


def load_config(config_path: Optional[Path]) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    if YAML is None:
        logging.error("PyYAML is not installed, using default configuration")
        return _get_default_config()

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "bench.yaml"

    if not config_path.exists():
        logging.warning(
            "Config file not found: %s, using defaults", config_path
        )
        return _get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = YAML.safe_load(f)
        return config or _get_default_config()
    except (OSError, TypeError, ValueError, YamlError) as error:
        logging.error("Error loading config: %s", error)
        return _get_default_config()


DEFAULT_CONFIG: dict = {
    "context_length": 2048,
    "gpu_offload": 1.0,
    "temperature": 0.1,
    "max_tokens": 256,
    "max_tests_per_capability": 10,
    "use_rest_api": True,
    "data_dir": "tests/data",
    "prompts_dir": "tests/prompts"
}


def _get_default_config() -> dict:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return DEFAULT_CONFIG.copy()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Capability-driven benchmark agent for LLM evaluation"
    )

    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        help="Path to model or model identifier"
    )

    parser.add_argument(
        "--random-models",
        type=int,
        help="Run capability benchmark for N random installed models"
    )

    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run capability benchmark for all installed models"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name (defaults to model path basename)"
    )

    parser.add_argument(
        "--capabilities",
        type=str,
        help=(
            "Comma-separated capabilities: "
            "general_text,reasoning,vision,tooling"
        )
    )

    parser.add_argument(
        "--output-dir",
        type=_sanitize_output_dir,
        default=str(USER_RESULTS_DIR),
        help=(
            "Output directory for results "
            "(default: ~/.local/share/lm-studio-bench/results)"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )

    parser.add_argument(
        "--formats",
        type=str,
        default="json,html,csv,pdf",
        help=(
            "Output formats: json,html,csv,pdf "
            "(default: json,html,csv,pdf)"
        )
    )

    parser.add_argument(
        "--max-tests",
        type=int,
        help="Maximum tests per capability"
    )

    parser.add_argument(
        "--context-length",
        type=int,
        help="Model context length"
    )

    parser.add_argument(
        "--gpu-offload",
        type=float,
        help="GPU offload ratio (0.0 to 1.0)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        help="Generation temperature"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling"
    )

    parser.add_argument(
        "--min-p",
        type=float,
        help="Min-p sampling"
    )

    parser.add_argument(
        "--repeat-penalty",
        type=float,
        help="Repeat penalty"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum generated tokens"
    )

    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        help="Number of GPU layers for load config"
    )

    parser.add_argument(
        "--n-batch",
        type=int,
        help="Batch size for load config"
    )

    parser.add_argument(
        "--n-threads",
        type=int,
        help="Number of threads for load config"
    )

    parser.add_argument(
        "--flash-attention",
        action="store_true",
        help="Enable flash attention"
    )

    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention"
    )

    parser.add_argument(
        "--rope-freq-base",
        type=float,
        help="RoPE frequency base"
    )

    parser.add_argument(
        "--rope-freq-scale",
        type=float,
        help="RoPE frequency scale"
    )

    parser.add_argument(
        "--use-mmap",
        action="store_true",
        help="Enable memory mapping"
    )

    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory mapping"
    )

    parser.add_argument(
        "--use-mlock",
        action="store_true",
        help="Enable mlock"
    )

    parser.add_argument(
        "--kv-cache-quant",
        type=str,
        help="KV cache quantization"
    )

    parser.add_argument(
        "--max-temp",
        type=float,
        help="Max GPU temperature limit"
    )

    parser.add_argument(
        "--max-power",
        type=float,
        help="Max GPU power limit"
    )

    parser.add_argument(
        "--enable-profiling",
        action="store_true",
        help="Enable profiling mode"
    )

    parser.add_argument(
        "--disable-gtt",
        action="store_true",
        help="Disable GTT in capability mode"
    )

    parser.add_argument(
        "--dev-mode",
        action="store_true",
        help="Enable developer mode"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def _sanitize_output_dir(output_dir_arg: str | Path) -> Path:
    """
    Validate and sanitize CLI output directory input.

    Args:
        output_dir_arg: Raw CLI argument for output directory.

    Returns:
        Resolved output directory path.

    Raises:
        ValueError: If the path is empty.
    """
    raw_input = (
        str(output_dir_arg)
        if isinstance(output_dir_arg, Path)
        else output_dir_arg
    )
    if not raw_input or not raw_input.strip():
        raise ValueError("Output directory cannot be empty")

    raw_path = Path(raw_input.strip()).expanduser()
    resolved_path = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (Path.cwd() / raw_path).resolve()
    )

    return resolved_path


def _list_installed_models() -> list[str]:
    """Return installed LM Studio model variants from ``lms ls --json``."""
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

        parsed = json.loads(result.stdout)
        model_names: list[str] = []
        seen: set[str] = set()
        for item in parsed:
            variants = item.get("variants") or []
            if variants:
                for variant in variants:
                    if variant and variant not in seen:
                        model_names.append(variant)
                        seen.add(variant)
                continue

            model_key = item.get("modelKey")
            if model_key and model_key not in seen:
                model_names.append(model_key)
                seen.add(model_key)
        return model_names
    except (
        json.JSONDecodeError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
        ValueError,
    ):
        return []


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """
    Override config with CLI arguments.

    Args:
        config: Base configuration
        args: Parsed CLI arguments

    Returns:
        Updated configuration
    """
    if args.max_tests is not None:
        config["max_tests_per_capability"] = args.max_tests

    if args.context_length is not None:
        config["context_length"] = args.context_length

    if args.gpu_offload is not None:
        config["gpu_offload"] = args.gpu_offload

    if args.temperature is not None:
        config["temperature"] = args.temperature

    if args.top_k is not None:
        config["top_k"] = args.top_k

    if args.top_p is not None:
        config["top_p"] = args.top_p

    if args.min_p is not None:
        config["min_p"] = args.min_p

    if args.repeat_penalty is not None:
        config["repeat_penalty"] = args.repeat_penalty

    if args.max_tokens is not None:
        config["max_tokens"] = args.max_tokens

    if args.n_gpu_layers is not None:
        config["n_gpu_layers"] = args.n_gpu_layers

    if args.n_batch is not None:
        config["n_batch"] = args.n_batch

    if args.n_threads is not None:
        config["n_threads"] = args.n_threads

    if args.flash_attention and args.no_flash_attention:
        raise ValueError(
            "--flash-attention and --no-flash-attention cannot be used together"
        )
    if args.flash_attention:
        config["flash_attention"] = True
    if args.no_flash_attention:
        config["flash_attention"] = False

    if args.rope_freq_base is not None:
        config["rope_freq_base"] = args.rope_freq_base

    if args.rope_freq_scale is not None:
        config["rope_freq_scale"] = args.rope_freq_scale

    if args.use_mmap and args.no_mmap:
        raise ValueError("--use-mmap and --no-mmap cannot be used together")
    if args.use_mmap:
        config["use_mmap"] = True
    if args.no_mmap:
        config["use_mmap"] = False

    if args.use_mlock:
        config["use_mlock"] = True

    if args.kv_cache_quant:
        config["kv_cache_quant"] = args.kv_cache_quant

    if args.max_temp is not None:
        config["max_temp"] = args.max_temp

    if args.max_power is not None:
        config["max_power"] = args.max_power

    if args.enable_profiling:
        config["enable_profiling"] = True

    if args.disable_gtt:
        config["disable_gtt"] = True

    if args.dev_mode:
        config["dev_mode"] = True

    return config


def _write_reports(
    report_data: dict,
    output_dir: Path,
    formats: list[str],
    report_stem: str,
) -> dict[str, Path]:
    """Write benchmark reports to sanitized paths inside output_dir."""
    safe_output_dir = _sanitize_output_dir(str(output_dir))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"{report_stem}_{timestamp}"
    output_files: dict[str, Path] = {}
    safe_output_dir.mkdir(parents=True, exist_ok=True)
    report_data = _enrich_report_data_from_sql(report_data)

    def _to_number(value: object, digits: int = 3) -> object:
        if isinstance(value, (int, float)):
            return round(value, digits)
        return value if value is not None else ""

    def _csv_cell(value: object) -> object:
        if not isinstance(value, str):
            return value
        if value.startswith(("=", "+", "-", "@", "\t")):
            return f"'{value}"
        return value

    def _csv_quote(value: object) -> str:
        text = str(_csv_cell(value))
        if any(char in text for char in [",", '"', "\n", "\r"]):
            text = '"' + text.replace('"', '""') + '"'
        return text

    def _render_simple_pdf(text_lines: list[str]) -> bytes:
        pdf_bytes = bytearray()
        pdf_bytes.extend(b"%PDF-1.4\n")
        offsets: list[int] = []

        def add_obj(content: str) -> None:
            offsets.append(len(pdf_bytes))
            pdf_bytes.extend(content.encode("latin-1", errors="replace"))

        add_obj("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        add_obj("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")

        stream_lines: list[str] = []
        y = 770
        for line in text_lines:
            safe_line = line.replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"BT /F1 10 Tf 50 {y} Td ({safe_line}) Tj ET")
            y -= 12
            if y < 50:
                break
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")

        add_obj(
            "3 0 obj\n"
            "<< /Type /Page /Parent 2 0 R "
            "/MediaBox [0 0 612 792] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>\n"
            "endobj\n"
        )
        add_obj(f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n")
        pdf_bytes.extend(stream)
        pdf_bytes.extend(b"\nendstream\nendobj\n")
        add_obj(
            "5 0 obj\n"
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\n"
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

    def _render_table_pdf(pdf_path: Path, report: dict) -> bool:
        if not REPORTLAB_AVAILABLE:
            return False
        if REPORTLAB_COLORS is None:
            return False
        if REPORTLAB_A4 is None:
            return False
        if REPORTLAB_GET_SAMPLE_STYLE_SHEET is None:
            return False
        if REPORTLAB_CM is None:
            return False
        if REPORTLAB_PARAGRAPH_CLS is None:
            return False
        if REPORTLAB_SIMPLE_DOC_TEMPLATE_CLS is None:
            return False
        if REPORTLAB_SPACER_CLS is None:
            return False
        if REPORTLAB_TABLE_CLS is None:
            return False
        if REPORTLAB_TABLE_STYLE_CLS is None:
            return False

        reportlab_colors = REPORTLAB_COLORS
        page_size = REPORTLAB_A4
        get_styles = REPORTLAB_GET_SAMPLE_STYLE_SHEET
        cm_unit = REPORTLAB_CM
        paragraph_cls = REPORTLAB_PARAGRAPH_CLS
        doc_cls = REPORTLAB_SIMPLE_DOC_TEMPLATE_CLS
        spacer_cls = REPORTLAB_SPACER_CLS
        table_cls = REPORTLAB_TABLE_CLS
        table_style_cls = REPORTLAB_TABLE_STYLE_CLS

        summary = report.get("summary", {}) or {}
        rows = report.get("results", []) or []
        by_capability = summary.get("by_capability", {}) or {}

        doc = doc_cls(
            str(pdf_path),
            pagesize=page_size,
            leftMargin=1.2 * cm_unit,
            rightMargin=1.2 * cm_unit,
            topMargin=1.2 * cm_unit,
            bottomMargin=1.2 * cm_unit,
            title="Capability Benchmark Report",
        )
        styles = get_styles()
        elements: list[Any] = []

        title = f"Capability Benchmark Report: {report.get('model_name', '-')}"
        elements.append(paragraph_cls(title, styles["Title"]))
        elements.append(
            paragraph_cls(
                f"Timestamp: {report.get('timestamp', '-')}",
                styles["Normal"],
            )
        )
        elements.append(spacer_cls(1, 0.35 * cm_unit))

        summary_table_data = [
            ["Metric", "Value"],
            ["Total Tests", str(summary.get("total_tests", 0))],
            ["Successful", str(summary.get("successful_tests", 0))],
            [
                "Success Rate",
                f"{float(summary.get('success_rate') or 0.0) * 100:.1f}%",
            ],
            [
                "Avg Latency",
                f"{_to_number(summary.get('avg_latency_ms'), 3)} ms",
            ],
            [
                "Avg Throughput",
                (
                    f"{_to_number(summary.get('avg_throughput_tokens_per_sec'), 3)} "
                    "tok/s"
                ),
            ],
            ["Avg Quality", str(_to_number(summary.get("avg_quality_score"), 4))],
        ]

        summary_table = table_cls(
            summary_table_data,
            colWidths=[6.0 * cm_unit, 10.8 * cm_unit],
        )
        summary_table.setStyle(
            table_style_cls(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        reportlab_colors.HexColor("#2f6cc4"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), reportlab_colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        reportlab_colors.HexColor("#d6dbe1"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            reportlab_colors.white,
                            reportlab_colors.HexColor("#f6f8fb"),
                        ],
                    ),
                ]
            )
        )
        elements.append(summary_table)
        elements.append(spacer_cls(1, 0.35 * cm_unit))

        if isinstance(by_capability, dict) and by_capability:
            cap_header = [[
                "Capability",
                "Tests",
                "Success",
                "Latency (ms)",
                "Throughput",
                "Quality",
            ]]
            cap_rows = []
            for capability, cap_data in sorted(by_capability.items()):
                if not isinstance(cap_data, dict):
                    continue
                cap_rows.append(
                    [
                        str(capability),
                        str(cap_data.get("test_count", 0)),
                        f"{float(cap_data.get('success_rate') or 0.0) * 100:.1f}%",
                        str(_to_number(cap_data.get("avg_latency_ms"), 2)),
                        str(
                            _to_number(
                                cap_data.get(
                                    "avg_throughput_tokens_per_sec"
                                ),
                                2,
                            )
                        ),
                        str(_to_number(cap_data.get("avg_quality_score"), 3)),
                    ]
                )

            if cap_rows:
                elements.append(
                    paragraph_cls("Capability Breakdown", styles["Heading2"])
                )
                cap_table = table_cls(
                    cap_header + cap_rows,
                    colWidths=[
                        4.2 * cm_unit,
                        1.9 * cm_unit,
                        2.0 * cm_unit,
                        2.5 * cm_unit,
                        2.5 * cm_unit,
                        2.5 * cm_unit,
                    ],
                )
                cap_table.setStyle(
                    table_style_cls(
                        [
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                reportlab_colors.HexColor("#18a999"),
                            ),
                            ("TEXTCOLOR", (0, 0), (-1, 0), reportlab_colors.white),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                            (
                                "GRID",
                                (0, 0),
                                (-1, -1),
                                0.5,
                                reportlab_colors.HexColor("#d6dbe1"),
                            ),
                            (
                                "ROWBACKGROUNDS",
                                (0, 1),
                                (-1, -1),
                                [
                                    reportlab_colors.white,
                                    reportlab_colors.HexColor("#eefaf8"),
                                ],
                            ),
                        ]
                    )
                )
                elements.append(cap_table)
                elements.append(spacer_cls(1, 0.3 * cm_unit))

        if rows:
            elements.append(paragraph_cls("Detailed Results", styles["Heading2"]))
            details = [[
                "ID",
                "Capability",
                "Latency",
                "Throughput",
                "Quality",
                "Status",
            ]]
            for entry in rows[:25]:
                details.append(
                    [
                        str(entry.get("test_id", "-")),
                        str(entry.get("capability", "-")),
                        str(_to_number(entry.get("latency_ms"), 2)),
                        str(_to_number(entry.get("throughput"), 2)),
                        str(_to_number(entry.get("quality_score"), 3)),
                        "error" if entry.get("error") else "success",
                    ]
                )

            details_table = table_cls(
                details,
                colWidths=[
                    2.1 * cm_unit,
                    4.0 * cm_unit,
                    2.3 * cm_unit,
                    2.5 * cm_unit,
                    2.3 * cm_unit,
                    2.2 * cm_unit,
                ],
            )
            details_table.setStyle(
                table_style_cls(
                    [
                        (
                            "BACKGROUND",
                            (0, 0),
                            (-1, 0),
                            reportlab_colors.HexColor("#3f7fbf"),
                        ),
                        ("TEXTCOLOR", (0, 0), (-1, 0), reportlab_colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                        (
                            "GRID",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            reportlab_colors.HexColor("#d6dbe1"),
                        ),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [
                                reportlab_colors.white,
                                reportlab_colors.HexColor("#f7f9fc"),
                            ],
                        ),
                    ]
                )
            )
            elements.append(details_table)

        doc.build(elements)
        return True

    if "json" in formats:
        json_path = safe_output_dir / f"{base_name}.json"
        enriched_data = {
            "schema_version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "report": report_data,
        }
        try:
            json_text = json.dumps(enriched_data, indent=2)
            json_path.write_text(json_text, encoding="utf-8")
            output_files["json"] = json_path
        except (OSError, TypeError, ValueError) as error:
            logging.getLogger(__name__).error(
                "Error generating JSON report: %s",
                error,
            )

    if "html" in formats:
        html_path = safe_output_dir / f"{base_name}.html"
        try:
            html_output = HTMLReporter().render(report_data)
            html_path.write_text(html_output, encoding="utf-8")
            output_files["html"] = html_path
        except (OSError, TypeError, ValueError) as error:
            logging.getLogger(__name__).error(
                "Error generating HTML report: %s",
                error,
            )

    if "csv" in formats:
        csv_path = (safe_output_dir / f"{base_name}.csv").resolve()
        try:
            safe_output_resolved = safe_output_dir.resolve()
            if not str(csv_path).startswith(str(safe_output_resolved)):
                raise ValueError("Path traversal detected in CSV export path")

            rows = report_data.get("results", []) or []
            summary = report_data.get("summary", {}) or {}
            hardware = report_data.get("hardware_profiling", {}) or {}

            csv_lines: list[str] = []

            def _append_csv_row(row: list[object]) -> None:
                csv_lines.append(",".join(_csv_quote(cell) for cell in row))

            _append_csv_row(["Capability Benchmark Report"])
            _append_csv_row(["Model", report_data.get("model_name", "-")])
            _append_csv_row(["Timestamp", report_data.get("timestamp", "-")])
            _append_csv_row(["Total Tests", summary.get("total_tests", 0)])
            _append_csv_row(
                ["Successful Tests", summary.get("successful_tests", 0)]
            )
            _append_csv_row(
                ["Success Rate", _to_number(summary.get("success_rate"), 4)]
            )
            _append_csv_row(
                ["Avg Latency ms", _to_number(summary.get("avg_latency_ms"), 3)]
            )
            _append_csv_row(
                ["Avg Quality", _to_number(summary.get("avg_quality_score"), 4)]
            )
            _append_csv_row(
                [
                    "Avg Throughput tok/s",
                    _to_number(summary.get("avg_throughput_tokens_per_sec"), 3),
                ]
            )
            _append_csv_row(
                ["Avg ROUGE", _to_number(summary.get("avg_rouge"), 4)]
            )
            _append_csv_row(["Avg F1", _to_number(summary.get("avg_f1"), 4)])
            _append_csv_row(
                [
                    "Avg Exact Match",
                    _to_number(summary.get("avg_exact_match"), 4),
                ]
            )
            _append_csv_row(
                ["Avg Accuracy", _to_number(summary.get("avg_accuracy"), 4)]
            )
            _append_csv_row(
                [
                    "Avg Function Call Accuracy",
                    _to_number(summary.get("avg_function_call_accuracy"), 4),
                ]
            )

            if hardware.get("enabled"):
                _append_csv_row([])
                _append_csv_row(["Hardware Profiling", "enabled"])
                _append_csv_row(
                    [
                        "GPU Temp min/max/avg (C)",
                        (
                            f"{_to_number(hardware.get('temp_celsius_min'), 2)} / "
                            f"{_to_number(hardware.get('temp_celsius_max'), 2)} / "
                            f"{_to_number(hardware.get('temp_celsius_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "GPU Power min/max/avg (W)",
                        (
                            f"{_to_number(hardware.get('power_watts_min'), 2)} / "
                            f"{_to_number(hardware.get('power_watts_max'), 2)} / "
                            f"{_to_number(hardware.get('power_watts_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "VRAM min/max/avg (GB)",
                        (
                            f"{_to_number(hardware.get('vram_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('vram_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('vram_gb_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "GTT min/max/avg (GB)",
                        (
                            f"{_to_number(hardware.get('gtt_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('gtt_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('gtt_gb_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "CPU min/max/avg (%)",
                        (
                            f"{_to_number(hardware.get('cpu_percent_min'), 2)} / "
                            f"{_to_number(hardware.get('cpu_percent_max'), 2)} / "
                            f"{_to_number(hardware.get('cpu_percent_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "RAM min/max/avg (GB)",
                        (
                            f"{_to_number(hardware.get('ram_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('ram_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('ram_gb_avg'), 2)}"
                        ),
                    ]
                )
                _append_csv_row(
                    [
                        "Temp Limit Exceeded",
                        str(bool(hardware.get("max_temp_exceeded"))),
                    ]
                )
                _append_csv_row(
                    [
                        "Power Limit Exceeded",
                        str(bool(hardware.get("max_power_exceeded"))),
                    ]
                )

            _append_csv_row([])
            by_capability = summary.get("by_capability", {}) or {}
            if by_capability:
                _append_csv_row(["Capability Summary"])
                _append_csv_row(
                    [
                        "capability",
                        "test_count",
                        "success_rate",
                        "avg_latency_ms",
                        "avg_throughput",
                        "avg_quality_score",
                        "avg_rouge",
                        "avg_f1",
                        "avg_exact_match",
                        "avg_accuracy",
                        "avg_function_call_accuracy",
                    ]
                )
                for capability, cap_data in sorted(by_capability.items()):
                    if not isinstance(cap_data, dict):
                        continue
                    _append_csv_row(
                        [
                            capability,
                            cap_data.get("test_count", 0),
                            _to_number(cap_data.get("success_rate"), 4),
                            _to_number(cap_data.get("avg_latency_ms"), 3),
                            _to_number(
                                cap_data.get("avg_throughput_tokens_per_sec"),
                                3,
                            ),
                            _to_number(cap_data.get("avg_quality_score"), 4),
                            _to_number(cap_data.get("avg_rouge"), 4),
                            _to_number(cap_data.get("avg_f1"), 4),
                            _to_number(cap_data.get("avg_exact_match"), 4),
                            _to_number(cap_data.get("avg_accuracy"), 4),
                            _to_number(
                                cap_data.get("avg_function_call_accuracy"),
                                4,
                            ),
                        ]
                    )

                _append_csv_row([])

            _append_csv_row(
                [
                    "test_id",
                    "test_name",
                    "capability",
                    "latency_ms",
                    "tokens_generated",
                    "throughput",
                    "quality_score",
                    "status",
                    "error",
                ]
            )

            for entry in rows:
                has_error = bool(entry.get("error"))
                _append_csv_row(
                    [
                        entry.get("test_id", ""),
                        entry.get("test_name", ""),
                        entry.get("capability", ""),
                        _to_number(entry.get("latency_ms"), 3),
                        entry.get("tokens_generated", ""),
                        _to_number(entry.get("throughput"), 3),
                        _to_number(entry.get("quality_score"), 4),
                        "error" if has_error else "success",
                        entry.get("error", ""),
                    ]
                )

            csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

            output_files["csv"] = csv_path
        except (OSError, TypeError, ValueError) as error:
            logging.getLogger(__name__).error(
                "Error generating CSV report: %s",
                error,
            )

    if "pdf" in formats:
        pdf_path = safe_output_dir / f"{base_name}.pdf"
        try:
            if _render_table_pdf(pdf_path, report_data):
                output_files["pdf"] = pdf_path
                return output_files

            summary = report_data.get("summary", {}) or {}
            rows = report_data.get("results", []) or []
            hardware = report_data.get("hardware_profiling", {}) or {}
            lines = [
                "Capability Benchmark Report",
                f"Model: {report_data.get('model_name', '-')}",
                f"Timestamp: {report_data.get('timestamp', '-')}",
                f"Total tests: {summary.get('total_tests', 0)}",
                f"Successful tests: {summary.get('successful_tests', 0)}",
                (
                    "Avg latency ms: "
                    f"{_to_number(summary.get('avg_latency_ms'), 3)}"
                ),
                (
                    "Avg quality score: "
                    f"{_to_number(summary.get('avg_quality_score'), 4)}"
                ),
            ]

            if hardware.get("enabled"):
                lines.extend(
                    [
                        "",
                        "Hardware Profiling:",
                        (
                            "GPU Temp min/max/avg (C): "
                            f"{_to_number(hardware.get('temp_celsius_min'), 2)} / "
                            f"{_to_number(hardware.get('temp_celsius_max'), 2)} / "
                            f"{_to_number(hardware.get('temp_celsius_avg'), 2)}"
                        ),
                        (
                            "GPU Power min/max/avg (W): "
                            f"{_to_number(hardware.get('power_watts_min'), 2)} / "
                            f"{_to_number(hardware.get('power_watts_max'), 2)} / "
                            f"{_to_number(hardware.get('power_watts_avg'), 2)}"
                        ),
                        (
                            "VRAM min/max/avg (GB): "
                            f"{_to_number(hardware.get('vram_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('vram_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('vram_gb_avg'), 2)}"
                        ),
                        (
                            "GTT min/max/avg (GB): "
                            f"{_to_number(hardware.get('gtt_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('gtt_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('gtt_gb_avg'), 2)}"
                        ),
                        (
                            "CPU min/max/avg (%): "
                            f"{_to_number(hardware.get('cpu_percent_min'), 2)} / "
                            f"{_to_number(hardware.get('cpu_percent_max'), 2)} / "
                            f"{_to_number(hardware.get('cpu_percent_avg'), 2)}"
                        ),
                        (
                            "RAM min/max/avg (GB): "
                            f"{_to_number(hardware.get('ram_gb_min'), 2)} / "
                            f"{_to_number(hardware.get('ram_gb_max'), 2)} / "
                            f"{_to_number(hardware.get('ram_gb_avg'), 2)}"
                        ),
                        (
                            "Temp limit exceeded: "
                            f"{bool(hardware.get('max_temp_exceeded'))}"
                        ),
                        (
                            "Power limit exceeded: "
                            f"{bool(hardware.get('max_power_exceeded'))}"
                        ),
                    ]
                )

            lines.extend([
                "",
                "Results:",
            ])

            for entry in rows[:40]:
                status = "error" if entry.get("error") else "ok"
                line = (
                    f"{entry.get('test_id', '-')}: "
                    f"{entry.get('capability', '-')} | "
                    f"lat {_to_number(entry.get('latency_ms'), 2)} ms | "
                    f"q {_to_number(entry.get('quality_score'), 3)} | "
                    f"{status}"
                )
                lines.append(line)

            pdf_path.write_bytes(_render_simple_pdf(lines))
            output_files["pdf"] = pdf_path
        except (OSError, TypeError, ValueError) as error:
            logging.getLogger(__name__).error(
                "Error generating PDF report: %s",
                error,
            )

    return output_files


def _to_float_or_none(value: object) -> Optional[float]:
    """Convert numeric-like values to float and keep missing as None."""
    if value is None:
        return None

    if not isinstance(
        value,
        (str, bytes, bytearray, SupportsFloat, SupportsIndex),
    ):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> Optional[float]:
    """Return average for non-empty numeric list."""
    if not values:
        return None
    return sum(values) / len(values)


class _CapabilityBucket(TypedDict):
    """Aggregated numeric metrics per capability."""

    test_count: int
    successful_tests: int
    latency: list[float]
    throughput: list[float]
    quality: list[float]
    rouge: list[float]
    f1: list[float]
    exact_match: list[float]
    accuracy: list[float]
    function_call_accuracy: list[float]


def _enrich_report_data_from_sql(report_data: dict) -> dict:
    """Attach SQL-backed aggregates for compatibility reports.

    The capability benchmark writes one row per test case into
    benchmark_results with source='compatibility'. This helper recomputes
    aggregate metrics from those rows to keep JSON/CSV/HTML/PDF in sync with
    persisted database values.
    """
    model_name = str(report_data.get("model_name") or "").strip()
    if not model_name or not BENCHMARK_DB_PATH.exists():
        return report_data

    expected_tests = int(
        (report_data.get("summary", {}) or {}).get("total_tests")
        or len(report_data.get("results", []) or [])
        or 1
    )

    query = """
        SELECT
            capability,
            avg_gen_time,
            avg_tokens_per_sec,
            quality_score,
            success,
            rouge_score,
            f1_score,
            exact_match_score,
            accuracy_score,
            function_call_accuracy,
            temp_celsius_min,
            temp_celsius_max,
            temp_celsius_avg,
            power_watts_min,
            power_watts_max,
            power_watts_avg,
            vram_gb_min,
            vram_gb_max,
            vram_gb_avg,
            gtt_gb_min,
            gtt_gb_max,
            gtt_gb_avg,
            cpu_percent_min,
            cpu_percent_max,
            cpu_percent_avg,
            ram_gb_min,
            ram_gb_max,
            ram_gb_avg,
            timestamp
        FROM benchmark_results
        WHERE source = 'compatibility'
        AND model_name = ?
        AND capability IS NOT NULL
        AND capability != ''
        AND test_id IS NOT NULL
        AND test_id != ''
        ORDER BY timestamp DESC
        LIMIT ?
    """

    try:
        conn = sqlite3.connect(BENCHMARK_DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (model_name, expected_tests)).fetchall()
        conn.close()
    except sqlite3.Error:
        return report_data

    if not rows:
        return report_data

    ordered_rows = list(reversed(rows))
    enriched = copy.deepcopy(report_data)
    summary = dict(enriched.get("summary", {}) or {})

    latency_values: list[float] = []
    throughput_values: list[float] = []
    quality_values: list[float] = []
    rouge_values: list[float] = []
    f1_values: list[float] = []
    em_values: list[float] = []
    accuracy_values: list[float] = []
    function_call_values: list[float] = []

    by_capability: dict[str, _CapabilityBucket] = {}
    successful_tests = 0

    hardware_collectors: dict[str, list[float]] = {
        "temp_celsius_min": [],
        "temp_celsius_max": [],
        "temp_celsius_avg": [],
        "power_watts_min": [],
        "power_watts_max": [],
        "power_watts_avg": [],
        "vram_gb_min": [],
        "vram_gb_max": [],
        "vram_gb_avg": [],
        "gtt_gb_min": [],
        "gtt_gb_max": [],
        "gtt_gb_avg": [],
        "cpu_percent_min": [],
        "cpu_percent_max": [],
        "cpu_percent_avg": [],
        "ram_gb_min": [],
        "ram_gb_max": [],
        "ram_gb_avg": [],
    }

    for row in ordered_rows:
        capability = str(row["capability"] or "general_text")
        cap_bucket = by_capability.setdefault(
            capability,
            {
                "test_count": 0,
                "successful_tests": 0,
                "latency": [],
                "throughput": [],
                "quality": [],
                "rouge": [],
                "f1": [],
                "exact_match": [],
                "accuracy": [],
                "function_call_accuracy": [],
            },
        )

        cap_bucket["test_count"] += 1

        if int(row["success"] or 0) == 1:
            successful_tests += 1
            cap_bucket["successful_tests"] += 1

        latency = _to_float_or_none(row["avg_gen_time"])
        throughput = _to_float_or_none(row["avg_tokens_per_sec"])
        quality = _to_float_or_none(row["quality_score"])
        rouge = _to_float_or_none(row["rouge_score"])
        f1 = _to_float_or_none(row["f1_score"])
        em = _to_float_or_none(row["exact_match_score"])
        accuracy = _to_float_or_none(row["accuracy_score"])
        fca = _to_float_or_none(row["function_call_accuracy"])

        if latency is not None:
            latency_values.append(latency)
            cap_bucket["latency"].append(latency)
        if throughput is not None:
            throughput_values.append(throughput)
            cap_bucket["throughput"].append(throughput)
        if quality is not None:
            quality_values.append(quality)
            cap_bucket["quality"].append(quality)
        if rouge is not None:
            rouge_values.append(rouge)
            cap_bucket["rouge"].append(rouge)
        if f1 is not None:
            f1_values.append(f1)
            cap_bucket["f1"].append(f1)
        if em is not None:
            em_values.append(em)
            cap_bucket["exact_match"].append(em)
        if accuracy is not None:
            accuracy_values.append(accuracy)
            cap_bucket["accuracy"].append(accuracy)
        if fca is not None:
            function_call_values.append(fca)
            cap_bucket["function_call_accuracy"].append(fca)

        for key, collector in hardware_collectors.items():
            value = _to_float_or_none(row[key])
            if value is not None:
                collector.append(value)

    total_tests = len(ordered_rows)
    summary["total_tests"] = total_tests
    summary["successful_tests"] = successful_tests
    summary["failed_tests"] = max(total_tests - successful_tests, 0)
    summary["success_rate"] = (
        successful_tests / total_tests if total_tests > 0 else 0.0
    )
    summary["avg_latency_ms"] = _avg(latency_values)
    summary["avg_throughput_tokens_per_sec"] = _avg(throughput_values)
    summary["avg_quality_score"] = _avg(quality_values)
    summary["avg_rouge"] = _avg(rouge_values)
    summary["avg_f1"] = _avg(f1_values)
    summary["avg_exact_match"] = _avg(em_values)
    summary["avg_accuracy"] = _avg(accuracy_values)
    summary["avg_function_call_accuracy"] = _avg(function_call_values)

    by_cap_summary: dict[str, dict[str, object]] = {}
    for capability, cap_bucket in by_capability.items():
        count = cap_bucket["test_count"]
        cap_success = cap_bucket["successful_tests"]
        by_cap_summary[capability] = {
            "test_count": count,
            "successful_tests": cap_success,
            "success_rate": cap_success / count if count > 0 else 0.0,
            "avg_latency_ms": _avg(cap_bucket["latency"]),
            "avg_throughput_tokens_per_sec": _avg(cap_bucket["throughput"]),
            "avg_quality_score": _avg(cap_bucket["quality"]),
            "avg_rouge": _avg(cap_bucket["rouge"]),
            "avg_f1": _avg(cap_bucket["f1"]),
            "avg_exact_match": _avg(cap_bucket["exact_match"]),
            "avg_accuracy": _avg(cap_bucket["accuracy"]),
            "avg_function_call_accuracy": _avg(
                cap_bucket["function_call_accuracy"]
            ),
        }
    summary["by_capability"] = by_cap_summary

    for key, values in hardware_collectors.items():
        if values:
            summary[key] = _avg(values)

    hardware = dict(enriched.get("hardware_profiling", {}) or {})
    if any(values for values in hardware_collectors.values()):
        hardware["enabled"] = True
        for key, values in hardware_collectors.items():
            if values:
                if key.endswith("_min"):
                    hardware[key] = min(values)
                elif key.endswith("_max"):
                    hardware[key] = max(values)
                else:
                    hardware[key] = _avg(values)

    enriched["summary"] = summary
    if hardware:
        enriched["hardware_profiling"] = hardware
    enriched["sql_validation"] = {
        "source_table": "benchmark_results",
        "source_filter": "compatibility",
        "rows_used": total_tests,
    }
    return enriched


def main() -> int:
    """
    Main CLI entrypoint.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    setup_logging(args.verbose or args.dev_mode)
    logger = logging.getLogger(__name__)
    total_start = time.perf_counter()

    try:
        config = load_config(args.config)
        config = override_config(config, args)

        safe_output_dir = _sanitize_output_dir(args.output_dir)

        logger.info("🚀 === LM Studio Model Benchmark ===")
        logger.info("🔬 Mode: Capability-driven")
        logger.info("📂 Output directory: %s", safe_output_dir)

        runner = BenchmarkRunner(
            config=config,
            output_dir=safe_output_dir
        )

        gpu_monitor = GPUMonitor()
        hardware_monitor = HardwareMonitor(
            gpu_monitor.gpu_type or "Unknown",
            gpu_monitor.gpu_tool or "",
            enabled=config.get("enable_profiling", False),
        )

        capabilities = None
        if args.capabilities:
            capabilities = [
                c.strip() for c in args.capabilities.split(",")
            ]

        if args.all_models and args.random_models:
            logger.error(
                "--all-models and --random-models cannot be used together"
            )
            return 1

        model_targets: list[str] = []
        if args.all_models:
            installed_models = _list_installed_models()
            if not installed_models:
                logger.error("No installed models found for all-model benchmark")
                return 1
            model_targets = installed_models
            logger.info("⚙️ Model selection: testing all installed models")
        elif args.random_models:
            if args.random_models < 1:
                logger.error("--random-models must be >= 1")
                return 1
            installed_models = _list_installed_models()
            if not installed_models:
                logger.error("No installed models found for random benchmark")
                return 1
            sample_size = min(args.random_models, len(installed_models))
            model_targets = random.sample(installed_models, sample_size)
            logger.info(
                "⚙️ Model limit set: testing random %d of %d model(s)",
                sample_size,
                len(installed_models),
            )
            logger.info("📋 Selected models: %s", ", ".join(model_targets))
        else:
            if not args.model_path:
                logger.error(
                    "model_path is required unless --random-models "
                    "or --all-models is used"
                )
                return 1
            model_targets = [args.model_path]

        formats = [f.strip() for f in args.formats.split(",")]
        logger.info("📊 Models detected: %d total", len(model_targets))
        logger.info(
            "⚙️ Context=%s, GPU Offload=%s, Temp=%s, Max Tests=%s",
            config.get("context_length"),
            config.get("gpu_offload"),
            config.get("temperature"),
            config.get("max_tests_per_capability"),
        )
        logger.info("🚀 Starting benchmark for %d models...", len(model_targets))

        successful_models = 0
        failed_models = 0

        for index, model_target in enumerate(model_targets, start=1):
            model_start = time.perf_counter()
            logger.info(
                "🎯 Starting benchmark for %s (%d/%d)",
                model_target,
                index,
                len(model_targets),
            )
            hardware_monitor.start()

            report_data: Optional[dict] = None
            profiling_stats: Optional[dict] = None

            try:
                report_data = runner.run(
                    model_path=model_target,
                    model_name=args.model_name,
                    capabilities=capabilities,
                    cache_results=False,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                failed_models += 1
                model_duration = time.perf_counter() - model_start
                logger.error(
                    "❌ %s failed after %.2fs: %s",
                    model_target,
                    model_duration,
                    error,
                    exc_info=args.verbose,
                )
                logger.info(
                    "⏭️ Skipping failed model and continuing (%d/%d)",
                    index,
                    len(model_targets),
                )
                continue
            finally:
                profiling_stats = hardware_monitor.stop()
                if profiling_stats:
                    logger.info(
                        "📈 Hardware summary (%s): "
                        "Temp avg=%.2f°C max=%.2f°C | "
                        "Power avg=%.2fW max=%.2fW",
                        model_target,
                        profiling_stats.get("temp_celsius_avg") or 0.0,
                        profiling_stats.get("temp_celsius_max") or 0.0,
                        profiling_stats.get("power_watts_avg") or 0.0,
                        profiling_stats.get("power_watts_max") or 0.0,
                    )

            if report_data is None:
                continue

            model_duration = time.perf_counter() - model_start

            if profiling_stats is not None and config.get("enable_profiling"):
                summary = report_data.setdefault("summary", {})
                summary["temp_celsius_min"] = profiling_stats.get(
                    "temp_celsius_min"
                )
                summary["temp_celsius_max"] = profiling_stats.get(
                    "temp_celsius_max"
                )
                summary["temp_celsius_avg"] = profiling_stats.get(
                    "temp_celsius_avg"
                )
                summary["power_watts_min"] = profiling_stats.get(
                    "power_watts_min"
                )
                summary["power_watts_max"] = profiling_stats.get(
                    "power_watts_max"
                )
                summary["power_watts_avg"] = profiling_stats.get(
                    "power_watts_avg"
                )
                summary["vram_gb_min"] = profiling_stats.get("vram_gb_min")
                summary["vram_gb_max"] = profiling_stats.get("vram_gb_max")
                summary["vram_gb_avg"] = profiling_stats.get("vram_gb_avg")
                summary["gtt_gb_min"] = profiling_stats.get("gtt_gb_min")
                summary["gtt_gb_max"] = profiling_stats.get("gtt_gb_max")
                summary["gtt_gb_avg"] = profiling_stats.get("gtt_gb_avg")
                summary["cpu_percent_min"] = profiling_stats.get(
                    "cpu_percent_min"
                )
                summary["cpu_percent_max"] = profiling_stats.get(
                    "cpu_percent_max"
                )
                summary["cpu_percent_avg"] = profiling_stats.get(
                    "cpu_percent_avg"
                )
                summary["ram_gb_min"] = profiling_stats.get("ram_gb_min")
                summary["ram_gb_max"] = profiling_stats.get("ram_gb_max")
                summary["ram_gb_avg"] = profiling_stats.get("ram_gb_avg")

                max_temp = config.get("max_temp")
                max_power = config.get("max_power")

                if (
                    max_temp
                    and summary.get("temp_celsius_max")
                    and summary["temp_celsius_max"] > max_temp
                ):
                    logger.warning(
                        "⚠️ Max. temperature exceeded: %.2f°C > %.2f°C",
                        summary["temp_celsius_max"],
                        max_temp,
                    )

                if (
                    max_power
                    and summary.get("power_watts_max")
                    and summary["power_watts_max"] > max_power
                ):
                    logger.warning(
                        "⚠️ Max. power exceeded: %.2fW > %.2fW",
                        summary["power_watts_max"],
                        max_power,
                    )

                report_data["hardware_profiling"] = {
                    "enabled": True,
                    "max_temp_limit": max_temp,
                    "max_power_limit": max_power,
                    "max_temp_exceeded": bool(
                        max_temp
                        and summary.get("temp_celsius_max")
                        and summary["temp_celsius_max"] > max_temp
                    ),
                    "max_power_exceeded": bool(
                        max_power
                        and summary.get("power_watts_max")
                        and summary["power_watts_max"] > max_power
                    ),
                } | {k: v for k, v in profiling_stats.items()}

            classic_metrics = _build_classic_metrics(
                config=config,
                report_data=report_data,
                gpu_monitor=gpu_monitor,
                benchmark_duration_seconds=model_duration,
            )
            runner.cache_report(
                report_dict=report_data,
                model_name=report_data.get("model_name", model_target),
                model_path=report_data.get("model_path", model_target),
                classic_metrics=classic_metrics,
            )

            report_stem = sanitize_report_name(
                report_data.get("model_name", model_target)
            )
            logger.info("📊 Exporting reports for %s...", model_target)
            output_files = _write_reports(
                report_data=report_data,
                output_dir=safe_output_dir,
                formats=formats,
                report_stem=report_stem,
            )

            logger.info("🧾 Generated reports for %s:", model_target)
            for fmt, path in output_files.items():
                logger.info("📄 %s: %s", fmt.upper(), path)

            logger.info(
                "✅ %s completed (Duration: %.2fs)",
                model_target,
                model_duration,
            )
            successful_models += 1

        total_duration = time.perf_counter() - total_start
        logger.info(
            "📊 Exporting reports for %d newly tested models...",
            successful_models,
        )
        logger.info(
            "✅ Benchmark completed. %d/%d models successfully tested",
            successful_models,
            len(model_targets),
        )
        if failed_models > 0:
            logger.warning("⚠️ %d model(s) failed and were skipped", failed_models)
        logger.info("🎉 Benchmark completed!")
        logger.info(
            "⏱️ Total capability benchmark duration: %.2fs",
            total_duration,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("🛑 Benchmark interrupted by user")
        return 1

    except (OSError, RuntimeError, TypeError, ValueError) as error:
        logger.error("❌ Benchmark failed: %s", error, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
