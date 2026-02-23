"""Shared configuration loader for benchmark and web app."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "defaults.json"

# Built-in fallbacks (used when JSON is missing or incomplete)
BASE_DEFAULT_CONFIG: Dict[str, Any] = {
    "prompt": "Erkläre maschinelles Lernen in 3 Sätzen",
    "context_length": 2048,
    "num_runs": 3,
    "lmstudio": {
        "host": "localhost",
        "ports": [1234, 1235],
        "api_token": None,
        "use_rest_api": False,
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
    """Load defaults from JSON if present, otherwise fall back to built-ins."""
    path = config_path or DEFAULT_CONFIG_PATH
    config = dict(BASE_DEFAULT_CONFIG)

    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                user_config = json.load(f)
            config = _deep_merge(config, user_config)
            logger.info(f"Loaded defaults from {path}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load defaults from {path}: {exc}")

    # Normalize LM Studio ports (always ensure at least one port)
    lmstudio_cfg = config.get("lmstudio", {}) or {}
    lmstudio_cfg["ports"] = _normalize_ports(lmstudio_cfg.get("ports"))
    lmstudio_cfg.setdefault("host", BASE_DEFAULT_CONFIG["lmstudio"]["host"])
    config["lmstudio"] = lmstudio_cfg

    return config


DEFAULT_CONFIG = load_default_config()
