"""Shared configuration loader for benchmark and web app."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from core.paths import USER_CONFIG_FILE

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"

BASE_DEFAULT_CONFIG: Dict[str, Any] = {
    "prompt": "Is the sky blue?",
    "context_length": 2048,
    "num_runs": 3,
    "retest": False,
    "enable_profiling": False,
    "lmstudio": {
        "host": "localhost",
        "ports": [1234, 1235],
        "api_token": None,
        "use_rest_api": True,
    },
    "inference": {
        "temperature": 0.1,
        "top_k_sampling": 40,
        "top_p_sampling": 0.9,
        "min_p_sampling": 0.05,
        "repeat_penalty": 1.2,
        "max_tokens": 256,
    },
    "load": {
        "n_gpu_layers": -1,
        "n_batch": 512,
        "n_threads": -1,
        "flash_attention": True,
        "rope_freq_base": None,
        "rope_freq_scale": None,
        "use_mmap": True,
        "use_mlock": False,
        "kv_cache_quant": None,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating base."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _normalize_ports(ports: Any) -> list[int]:
    """Normalize LM Studio port configuration."""
    normalized: list[int] = []
    if isinstance(ports, list):
        for port in ports:
            try:
                port_int = int(port)
            except (TypeError, ValueError):
                continue
            if port_int not in normalized:
                normalized.append(port_int)
    return normalized or list(BASE_DEFAULT_CONFIG["lmstudio"]["ports"])


def load_default_config(config_path: Path | None = None) -> Dict[str, Any]:
    """Load config with fallback chain.

    Load order:
    1. Project defaults (project_root/config/defaults.json)
    2. User overrides (if exists: ~/.config/lm-studio-bench/defaults.json)
    3. Hardcoded defaults (BASE_DEFAULT_CONFIG)
    """
    config = dict(BASE_DEFAULT_CONFIG)
    project_config_path = config_path or PROJECT_CONFIG_PATH

    if project_config_path.exists():
        try:
            with project_config_path.open("r", encoding="utf-8") as f:
                project_config = json.load(f)
            config = _deep_merge(config, project_config)
            logger.info("Loaded project defaults from %s", project_config_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load project config from %s: %s", project_config_path, exc
            )

    if USER_CONFIG_FILE.exists():
        try:
            with USER_CONFIG_FILE.open("r", encoding="utf-8") as f:
                user_config = json.load(f)
            config = _deep_merge(config, user_config)
            logger.info("Loaded user config overrides from %s", USER_CONFIG_FILE)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to load user config from %s: %s", USER_CONFIG_FILE, exc
            )

    lmstudio_cfg = config.get("lmstudio", {}) or {}
    lmstudio_cfg["ports"] = _normalize_ports(lmstudio_cfg.get("ports"))
    lmstudio_cfg.setdefault("host", BASE_DEFAULT_CONFIG["lmstudio"]["host"])
    config["lmstudio"] = lmstudio_cfg

    return config


DEFAULT_CONFIG = load_default_config()
