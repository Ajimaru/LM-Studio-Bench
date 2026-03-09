"""Shared fixtures and configuration for the test suite."""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
WEB_DIR = PROJECT_ROOT / "web"
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

if "lmstudio" not in sys.modules:
    _mock_lmstudio = MagicMock()
    sys.modules["lmstudio"] = _mock_lmstudio


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Return a temporary config directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


@pytest.fixture
def tmp_presets_dir(tmp_path: Path) -> Path:
    """Return a temporary presets directory."""
    presets_dir = tmp_path / "presets"
    presets_dir.mkdir(parents=True, exist_ok=True)
    return presets_dir


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Return a sample benchmark configuration dictionary."""
    return {
        "prompt": "Test prompt",
        "context_length": 1024,
        "num_runs": 2,
        "retest": False,
        "enable_profiling": False,
        "lmstudio": {
            "host": "localhost",
            "ports": [1234],
            "api_token": None,
            "use_rest_api": True,
        },
        "inference": {
            "temperature": 0.5,
            "top_k_sampling": 20,
            "top_p_sampling": 0.8,
            "min_p_sampling": 0.02,
            "repeat_penalty": 1.1,
            "max_tokens": 128,
        },
        "load": {
            "n_gpu_layers": -1,
            "n_batch": 256,
            "n_threads": -1,
            "flash_attention": True,
            "use_mmap": True,
            "use_mlock": False,
        },
    }


@pytest.fixture
def project_config_file(tmp_path: Path, sample_config: Dict[str, Any]) -> Path:
    """Write a project config file to a temp directory and return its path."""
    config_path = tmp_path / "defaults.json"
    config_path.write_text(json.dumps(sample_config), encoding="utf-8")
    return config_path
