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
from typing import Optional

from agents.runner import BenchmarkRunner
from cli.reporting import generate_reports


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
    import yaml

    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "bench.yaml"

    if not config_path.exists():
        logging.warning(
            "Config file not found: %s, using defaults", config_path
        )
        return _get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or _get_default_config()
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
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
        type=Path,
        default=Path("output"),
        help="Output directory for results (default: output)"
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

        logger.info("🚀 === LM Studio Model Benchmark ===")
        logger.info("🔬 Mode: Capability-driven")
        logger.info("📂 Output directory: %s", args.output_dir)

        runner = BenchmarkRunner(
            config=config,
            output_dir=args.output_dir
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
            logger.info("📊 Exporting reports for %s...", model_target)
            output_files = generate_reports(
                report_data=report_data,
                output_dir=args.output_dir,
                formats=formats,
            )

            logger.info("Generated reports for %s:", model_target)
            for fmt, path in output_files.items():
                logger.info("  %s: %s", fmt.upper(), path)

            model_duration = time.perf_counter() - model_start
            logger.info("✓ %s completed (Duration: %.2fs)", model_target, model_duration)

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
