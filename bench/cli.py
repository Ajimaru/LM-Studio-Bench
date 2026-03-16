#!/usr/bin/env python3
"""
CLI entrypoint for capability-driven benchmark agent.

Provides command-line interface to run benchmarks on models.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from agents.runner import BenchmarkRunner
from bench.reporting import generate_reports


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


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
        config_path = Path(__file__).parent / "config.yaml"

    if not config_path.exists():
        logging.warning(
            f"Config file not found: {config_path}, using defaults"
        )
        return _get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or _get_default_config()
    except Exception as e:
        logging.error(f"Error loading config: {e}")
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
        help="Path to model or model identifier"
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

    try:
        config = load_config(args.config)
        config = override_config(config, args)

        logger.info("Starting capability-driven benchmark")
        logger.info(f"Model: {args.model_path}")
        logger.info(f"Output: {args.output_dir}")

        runner = BenchmarkRunner(
            config=config,
            output_dir=args.output_dir
        )

        capabilities = None
        if args.capabilities:
            capabilities = [
                c.strip() for c in args.capabilities.split(",")
            ]

        report_data = runner.run(
            model_path=args.model_path,
            model_name=args.model_name,
            capabilities=capabilities
        )

        formats = [f.strip() for f in args.formats.split(",")]
        output_files = generate_reports(
            report_data=report_data,
            output_dir=args.output_dir,
            formats=formats
        )

        logger.info("Benchmark completed successfully")
        logger.info("Generated reports:")
        for fmt, path in output_files.items():
            logger.info(f"  {fmt.upper()}: {path}")

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
