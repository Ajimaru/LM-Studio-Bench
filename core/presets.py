"""Preset management for benchmark CLI and web app."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List

from core.config import DEFAULT_CONFIG
from core.paths import USER_PRESETS_DIR

logger = logging.getLogger(__name__)


PRESET_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,50}$")

LEGACY_KEY_MAP: Dict[str, str] = {
    "context_length": "context",
    "num_runs": "runs",
    "top_k": "top_k_sampling",
    "top_p": "top_p_sampling",
    "min_p": "min_p_sampling",
}


class PresetManager:
    """Manage readonly and user-defined benchmark presets."""

    READONLY_PRESETS = {
        "default",
        "default_classic",
        "default_compatability_test",
        "quick_test",
        "high_quality",
        "resource_limited",
    }

    PREDEFINED_PRESETS: Dict[str, Dict[str, Any]] = {
        "default_classic": {
            "runs": 3,
            "context": 2048,
            "limit": 0,
            "dev_mode": False,
            "min_context": 0,
            "max_size": 0.0,
            "quants": "",
            "arch": "",
            "params": "",
            "rank_by": "speed",
            "only_vision": False,
            "only_tools": False,
            "include_models": "",
            "exclude_models": "",
            "retest": False,
            "enable_profiling": False,
            "disable_gtt": False,
            "max_temp": 0.0,
            "max_power": 0.0,
            "prompt": "Explain machine learning in 3 sentences",
            "temperature": 0.1,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.2,
            "max_tokens": 256,
            "n_gpu_layers": -1,
            "n_batch": 512,
            "n_threads": -1,
            "flash_attention": True,
            "rope_freq_base": None,
            "rope_freq_scale": None,
            "use_mmap": True,
            "use_mlock": False,
            "kv_cache_quant": None,
            "benchmark_mode": "classic",
            "preset_mode": "classic",
            "agent_model": None,
            "agent_capabilities": None,
            "agent_max_tests": None,
        },
        "default_compatability_test": {
            "runs": 1,
            "context": 2048,
            "limit": 0,
            "dev_mode": False,
            "min_context": 0,
            "max_size": 0.0,
            "quants": "",
            "arch": "",
            "params": "",
            "rank_by": "speed",
            "only_vision": False,
            "only_tools": False,
            "include_models": "",
            "exclude_models": "",
            "retest": False,
            "enable_profiling": False,
            "disable_gtt": False,
            "max_temp": 0.0,
            "max_power": 0.0,
            "prompt": "Explain machine learning in 3 sentences",
            "temperature": 0.1,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.2,
            "max_tokens": 256,
            "n_gpu_layers": -1,
            "n_batch": 512,
            "n_threads": -1,
            "flash_attention": True,
            "rope_freq_base": None,
            "rope_freq_scale": None,
            "use_mmap": True,
            "use_mlock": False,
            "kv_cache_quant": None,
            "benchmark_mode": "capability",
            "preset_mode": "capability",
            "agent_model": "qwen2.5-7b-instruct",
            "agent_capabilities": "general_text,reasoning",
            "agent_max_tests": 10,
        },
        "quick_test": {
            "runs": 1,
            "context": 1024,
            "dev_mode": True,
            "enable_profiling": True,
        },
        "high_quality": {
            "runs": 5,
            "context": 8192,
            "enable_profiling": True,
            "retest": True,
        },
        "resource_limited": {
            "runs": 3,
            "context": 2048,
            "max_size": 8.0,
            "n_batch": 256,
            "flash_attention": True,
            "use_mmap": True,
        },
    }

    def __init__(self, presets_dir: Path | None = None) -> None:
        self._presets_dir = presets_dir or USER_PRESETS_DIR
        self._presets_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> List[str]:
        """Return all available preset names."""
        names = ["default", "default_classic", "default_compatability_test"]
        names.extend(sorted(self.PREDEFINED_PRESETS.keys()))

        user_names: List[str] = []
        for path in sorted(self._presets_dir.glob("*.json")):
            name = path.stem
            if name in self.READONLY_PRESETS:
                logger.warning("Ignoring user preset with reserved name: %s", name)
                continue
            user_names.append(name)

        names.extend(user_names)
        return names

    def list_presets_detailed(self) -> List[tuple[str, bool]]:
        """Return presets with readonly status."""
        return [(name, name in self.READONLY_PRESETS) for name in self.list_presets()]

    def load_preset(self, name: str) -> Dict[str, Any]:
        """Load a preset by name."""
        name_to_load = "default_classic" if name == "default" else name

        if name_to_load in self.PREDEFINED_PRESETS:
            logger.info("Loading readonly preset: %s", name_to_load)
            return self._merge_with_default(self.PREDEFINED_PRESETS[name_to_load])

        valid, reason = self.validate_preset_name(name_to_load)
        if not valid:
            raise ValueError(reason)

        preset_path = self._preset_path(name_to_load)
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name_to_load}")

        with preset_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, dict) and "params" in payload:
            preset_data = payload.get("params") or {}
        elif isinstance(payload, dict):
            preset_data = payload
        else:
            raise ValueError(f"Invalid preset format: {name_to_load}")

        if not isinstance(preset_data, dict):
            raise ValueError(f"Invalid preset payload: {name_to_load}")

        logger.info("Loading user preset: %s", name_to_load)
        return self._merge_with_default(preset_data)

    def get_default_preset(self) -> Dict[str, Any]:
        """Build default benchmark preset from shared config."""
        default_cfg = DEFAULT_CONFIG
        inference = default_cfg.get("inference", {}) or {}
        load_cfg = default_cfg.get("load", {}) or {}
        return {
            "runs": int(default_cfg.get("num_runs", 3)),
            "context": int(default_cfg.get("context_length", 2048)),
            "limit": 0,
            "dev_mode": False,
            "min_context": 0,
            "max_size": 0.0,
            "quants": "",
            "arch": "",
            "params": "",
            "rank_by": "speed",
            "only_vision": False,
            "only_tools": False,
            "include_models": "",
            "exclude_models": "",
            "retest": bool(default_cfg.get("retest", False)),
            "enable_profiling": bool(default_cfg.get("enable_profiling", False)),
            "disable_gtt": False,
            "max_temp": 0.0,
            "max_power": 0.0,
            "prompt": default_cfg.get("prompt"),
            "temperature": inference.get("temperature"),
            "top_k_sampling": inference.get("top_k_sampling"),
            "top_p_sampling": inference.get("top_p_sampling"),
            "min_p_sampling": inference.get("min_p_sampling"),
            "repeat_penalty": inference.get("repeat_penalty"),
            "max_tokens": inference.get("max_tokens"),
            "n_gpu_layers": load_cfg.get("n_gpu_layers"),
            "n_batch": load_cfg.get("n_batch"),
            "n_threads": load_cfg.get("n_threads"),
            "flash_attention": load_cfg.get("flash_attention"),
            "rope_freq_base": load_cfg.get("rope_freq_base"),
            "rope_freq_scale": load_cfg.get("rope_freq_scale"),
            "use_mmap": load_cfg.get("use_mmap"),
            "use_mlock": load_cfg.get("use_mlock"),
            "kv_cache_quant": load_cfg.get("kv_cache_quant"),
            "benchmark_mode": "classic",
            "preset_mode": "classic",
            "agent_model": None,
            "agent_capabilities": None,
            "agent_max_tests": None,
        }

    def validate_preset_name(self, name: str) -> tuple[bool, str]:
        """Validate preset name safety constraints."""
        if name in self.READONLY_PRESETS:
            return False, "Readonly preset names cannot be used"
        if not PRESET_NAME_PATTERN.fullmatch(name or ""):
            return False, "Preset name must match [a-zA-Z0-9_-]{1,50}"
        if ".." in name or "/" in name or "\\" in name:
            return False, "Preset name contains invalid path separators"
        return True, ""

    def save_preset(self, name: str, config: Dict[str, Any]) -> None:
        """Save a user preset as JSON in the presets directory."""
        valid, reason = self.validate_preset_name(name)
        if not valid:
            raise ValueError(reason)

        payload = {key: value for key, value in config.items() if value is not None}

        preset_path = self._preset_path(name)
        with preset_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def delete_preset(self, name: str) -> None:
        """Delete a user preset file."""
        valid, reason = self.validate_preset_name(name)
        if not valid:
            raise ValueError(reason)

        preset_path = self._preset_path(name)
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {name}")
        preset_path.unlink()

    def compare_presets(
        self,
        preset_a: Dict[str, Any],
        preset_b: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Return key-wise comparison object for two preset dictionaries."""
        normalized_a = self._merge_with_default(preset_a or {})
        normalized_b = self._merge_with_default(preset_b or {})
        keys = sorted(set(normalized_a.keys()) | set(normalized_b.keys()))
        return {
            key: {
                "preset_a": normalized_a.get(key),
                "preset_b": normalized_b.get(key),
            }
            for key in keys
        }

    def _preset_path(self, name: str) -> Path:
        """Resolve path for user preset name."""
        return self._presets_dir / f"{name}.json"

    def _merge_with_default(self, override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge override values with default preset values."""
        merged = dict(self.get_default_preset())
        normalized_override = self._normalize_legacy_keys(override)
        for key, value in normalized_override.items():
            if value is not None:
                merged[key] = value
        return merged

    def _normalize_legacy_keys(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        """Map legacy preset key names to the current canonical keys."""
        normalized: Dict[str, Any] = {}
        for key, value in (preset or {}).items():
            canonical_key = LEGACY_KEY_MAP.get(key, key)
            normalized[canonical_key] = value
        return normalized

    def preset_to_cli_args(self, preset: Dict[str, Any]) -> List[str]:
        """Convert preset dictionary to CLI argument list."""
        args: List[str] = []

        def add_value(flag: str, value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str) and not value.strip():
                return
            args.extend([flag, str(value)])

        def add_positive_value(flag: str, value: Any) -> None:
            if isinstance(value, (int, float)) and value > 0:
                args.extend([flag, str(value)])

        add_value("--runs", preset.get("runs"))
        add_value("--context", preset.get("context"))
        add_positive_value("--limit", preset.get("limit"))
        add_value("--prompt", preset.get("prompt"))
        add_positive_value("--min-context", preset.get("min_context"))
        max_size = preset.get("max_size")
        if isinstance(max_size, (int, float)) and max_size > 0:
            add_value("--max-size", max_size)
        add_value("--quants", preset.get("quants"))
        add_value("--arch", preset.get("arch"))
        add_value("--params", preset.get("params"))
        add_value("--rank-by", preset.get("rank_by"))
        add_value("--include-models", preset.get("include_models"))
        add_value("--exclude-models", preset.get("exclude_models"))
        add_positive_value("--max-temp", preset.get("max_temp"))
        add_positive_value("--max-power", preset.get("max_power"))

        add_value("--temperature", preset.get("temperature"))
        add_value("--top-k", preset.get("top_k_sampling"))
        add_value("--top-p", preset.get("top_p_sampling"))
        add_value("--min-p", preset.get("min_p_sampling"))
        add_value("--repeat-penalty", preset.get("repeat_penalty"))
        add_value("--max-tokens", preset.get("max_tokens"))

        add_value("--n-gpu-layers", preset.get("n_gpu_layers"))
        add_value("--n-batch", preset.get("n_batch"))
        add_value("--n-threads", preset.get("n_threads"))
        add_value("--rope-freq-base", preset.get("rope_freq_base"))
        add_value("--rope-freq-scale", preset.get("rope_freq_scale"))
        add_value("--kv-cache-quant", preset.get("kv_cache_quant"))

        if preset.get("only_vision"):
            args.append("--only-vision")
        if preset.get("only_tools"):
            args.append("--only-tools")
        if preset.get("retest"):
            args.append("--retest")
        if preset.get("dev_mode"):
            args.append("--dev-mode")
        if preset.get("enable_profiling"):
            args.append("--enable-profiling")
        if preset.get("disable_gtt"):
            args.append("--disable-gtt")

        if preset.get("flash_attention") is True:
            args.append("--flash-attention")
        elif preset.get("flash_attention") is False:
            args.append("--no-flash-attention")

        if preset.get("use_mmap") is True:
            args.append("--use-mmap")
        elif preset.get("use_mmap") is False:
            args.append("--no-mmap")

        if preset.get("use_mlock"):
            args.append("--use-mlock")

        return args
