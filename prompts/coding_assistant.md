# Preset resolution in the `PresetManager`

You are a coding assistant working inside an editor. Below are two files
from the user's repository.

```python
# file: core/version.py
#!/usr/bin/env python3
"""Version checking utilities for LM Studio Benchmark.

Provides functions to read the current version, fetch the latest
release from GitHub, compare versions, and format release URLs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import httpx

LOGGER = logging.getLogger(__name__)


def get_current_version() -> str:
    """Read the current version from VERSION file.

    The VERSION file is located at project root and contains a single
    line with the version string (e.g., "v0.1.0").

    Returns:
        Version string (e.g., "v0.1.0"), or "unknown" if file not found.

    Raises:
        ValueError: If VERSION file is empty or unreadable.
    """
    project_root = Path(__file__).resolve().parent.parent
    version_file = project_root / "VERSION"

    if not version_file.exists():
        LOGGER.warning("VERSION file not found at %s", version_file)
        return "unknown"

    try:
        current = version_file.read_text(encoding="utf-8").strip()
        if not current:
            raise ValueError("VERSION file is empty")
        LOGGER.debug("Current version: %s", current)
        return current
    except (OSError, ValueError) as exc:
        LOGGER.error("Failed to read VERSION file: %s", exc)
        raise ValueError(f"Cannot read VERSION: {exc}") from exc


def fetch_latest_release() -> Optional[dict]:
    """Fetch latest release info from GitHub API.

    Queries https://api.github.com/repos/Ajimaru/LM-Studio-Bench/
    releases/latest and returns the JSON response.

    Returns:
        Dict with keys 'tag_name', 'html_url' on success, None on
        failure.

    Raises:
        None - Errors are logged and None is returned (graceful
        degradation).
    """
    url = (
        "https://api.github.com/repos/Ajimaru/LM-Studio-Bench/releases/latest"
    )

    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            LOGGER.debug("GitHub latest release: %s", data.get("tag_name"))
            return data
    except httpx.HTTPStatusError as exc:
        LOGGER.warning("GitHub API error (%s): %s", exc.response.status_code, exc)
        return None
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        LOGGER.warning("Failed to fetch latest release: %s", exc)
        return None
    except (ValueError, KeyError) as exc:
        LOGGER.error("Failed to parse GitHub API response: %s", exc)
        return None


def compare_versions(current: str, latest: str) -> bool:
    """Check if a newer version is available.

    Compares two semantic version strings (e.g., "v0.1.0" vs
    "v0.2.0"). Returns True if latest > current.

    Args:
        current: Current version string (e.g., "v0.1.0").
        latest: Latest version string (e.g., "v0.2.0").

    Returns:
        True if latest is newer than current, False otherwise.
    """

    def parse_version(v: str) -> tuple:
        """Parse version string to tuple (major, minor, patch)."""
        v_clean = v.lstrip("v").split("-")[0]
        try:
            parts = v_clean.split(".")
            return tuple(int(p) for p in parts[:3])
        except (ValueError, IndexError):
            LOGGER.warning("Invalid version format: %s, assuming older", v)
            return (0, 0, 0)

    current_tuple = parse_version(current)
    latest_tuple = parse_version(latest)

    is_update_available = latest_tuple > current_tuple
    LOGGER.debug(
        "Version comparison: %s (%s) < %s (%s) = %s",
        current,
        current_tuple,
        latest,
        latest_tuple,
        is_update_available,
    )
    return is_update_available


def format_release_url(tag_name: str) -> str:
    """Format a GitHub release URL from tag name.

    Args:
        tag_name: Tag name from GitHub (e.g., "v0.2.0").

    Returns:
        Full GitHub release URL.
    """
    url = f"https://github.com/Ajimaru/LM-Studio-Bench/" f"releases/tag/{tag_name}"
    LOGGER.debug("Release URL: %s", url)
    return url

```

```python
# file: core/presets.py (excerpt)
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

    DEFAULT_ALIAS = "default"
    DEFAULT_PRESET_NAME = "default_classic"

    READONLY_PRESETS = {
        "default_classic",
        "default_compatibility_test",
        "quick_test",
        "high_quality",
        "resource_limited",
        "coding_assistant",
        "legacy_2048",
    }

    # Backwards-compatible alias for the previously misspelled preset name.
    COMPAT_ALIASES: Dict[str, str] = {
        "default_compatability_test": "default_compatibility_test",
    }

    PREDEFINED_PRESETS: Dict[str, Dict[str, Any]] = {
        "default_classic": {
            "runs": 3,
            "context": 8192,
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
            "prompt": DEFAULT_CONFIG.get("prompt", ""),
            "temperature": 0.1,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.2,
            "max_tokens": 512,
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
        "default_compatibility_test": {
            "runs": 1,
            "context": 8192,
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
            "prompt": DEFAULT_CONFIG.get("prompt", ""),
            "temperature": 0.1,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.2,
            "max_tokens": 512,
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
        # Mirrors an IDE assistant workload: long file context in the prompt,
        # a multi-paragraph answer, hardware profiling to catch KV-cache spill.
        "coding_assistant": {
            "runs": 3,
            "context": 16384,
            "max_tokens": 512,
            "enable_profiling": True,
            "retest": True,
            "rank_by": "speed",
        },
        # Preserves the pre-8192 defaults so older cached results stay
        # comparable after the default context length was raised.
        "legacy_2048": {
            "runs": 3,
            "context": 2048,
            "max_tokens": 256,
            "prompt": "Explain machine learning in 3 sentences",
        },
    }

    def __init__(self, presets_dir: Path | None = None) -> None:
        self._presets_dir = presets_dir or USER_PRESETS_DIR
        self._presets_dir.mkdir(parents=True, exist_ok=True)

    def list_presets(self) -> List[str]:
        """Return all available preset names."""
        names = [self.DEFAULT_ALIAS]
        names.extend(sorted(self.PREDEFINED_PRESETS.keys()))

        user_names: List[str] = []
        for path in sorted(self._presets_dir.glob("*.json")):
            name = path.stem
            if name in self.READONLY_PRESETS or self.is_readonly_name(name):
                logger.warning("Ignoring user preset with reserved name: %s", name)
                continue
            user_names.append(name)

        names.extend(user_names)
        return names

    def list_presets_detailed(self) -> List[tuple[str, bool]]:
        """Return presets with readonly status."""
        return [
            (name, self.is_readonly_name(name))
            for name in self.list_presets()
        ]

    def resolve_preset_name(self, name: str) -> str:
        """Resolve public preset aliases to their canonical preset name."""
        if name == self.DEFAULT_ALIAS:
            return self.DEFAULT_PRESET_NAME
        return self.COMPAT_ALIASES.get(name, name)

    def is_readonly_name(self, name: str) -> bool:
        """Return True when a public preset name maps to a readonly preset."""
        return self.resolve_preset_name(name) in self.READONLY_PRESETS

```

Question: Explain how preset resolution handles legacy keys and compatibility
aliases, what happens when an unknown preset name is requested, and which edge
cases the version comparison does not cover. Answer in four sentences.
