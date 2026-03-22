#!/usr/bin/env python3
"""
CLI entrypoint for capability-driven benchmark agent.

Provides command-line interface to run benchmarks on models.
"""

import argparse
import json
import logging
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType
from typing import Optional

from agents.runner import BenchmarkRunner
from cli.reporting import HTMLReporter, sanitize_report_name

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


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
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


def _get_default_config() -> dict:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "context_length": 2048,
        "gpu_offload": 1.0,
        "temperature": 0.1,
        "max_tokens": 256,
        "max_tests_per_capability": 10,
        "use_rest_api": True,
        "data_dir": "tests/data",
        "prompts_dir": "tests/prompts"
    }


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
        default=(Path.cwd() / "output").resolve(),
        help=(
            "Output directory for results inside current workspace "
            "(default: output)"
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
        default="json,html",
        help="Output formats: json,html (default: json,html)"
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
        Resolved output directory path inside current workspace.

    Raises:
        ValueError: If the path is empty or outside the current workspace.
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
    workspace_root = Path.cwd().resolve()

    try:
        resolved_path.relative_to(workspace_root)
    except ValueError as error:
        raise ValueError(
            "Output directory must be inside the current workspace"
        ) from error

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

    return output_files


def main() -> int:
    """
    Main CLI entrypoint.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    total_start = time.perf_counter()

    try:
        config = load_config(args.config)
        config = override_config(config, args)

        safe_output_dir = _sanitize_output_dir(args.output_dir)
        workspace_root = Path.cwd().resolve()
        safe_output_dir = workspace_root / safe_output_dir.relative_to(
            workspace_root
        )

        logger.info("🚀 === LM Studio Model Benchmark ===")
        logger.info("🔬 Mode: Capability-driven")
        logger.info("📂 Output directory: %s", safe_output_dir)

        runner = BenchmarkRunner(
            config=config,
            output_dir=safe_output_dir
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

        for index, model_target in enumerate(model_targets, start=1):
            model_start = time.perf_counter()
            logger.info(
                "🎯 Starting benchmark for %s (%d/%d)",
                model_target,
                index,
                len(model_targets),
            )
            report_data = runner.run(
                model_path=model_target,
                model_name=args.model_name,
                capabilities=capabilities,
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

            logger.info("Generated reports for %s:", model_target)
            for fmt, path in output_files.items():
                logger.info("  %s: %s", fmt.upper(), path)

            model_duration = time.perf_counter() - model_start
            logger.info(
                "✓ %s completed (Duration: %.2fs)",
                model_target,
                model_duration,
            )

        total_duration = time.perf_counter() - total_start
        logger.info(
            "📊 Exporting reports for %d newly tested models...",
            len(model_targets),
        )
        logger.info(
            "✅ Benchmark completed. %d/%d models successfully tested",
            len(model_targets),
            len(model_targets),
        )
        logger.info("🎉 Benchmark completed!")
        logger.info(
            "⏱️ Total capability benchmark duration: %.2fs",
            total_duration,
        )

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1

    except (OSError, RuntimeError, TypeError, ValueError) as error:
        logger.error("Benchmark failed: %s", error, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
