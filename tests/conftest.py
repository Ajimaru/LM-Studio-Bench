"""Shared fixtures and configuration for the test suite."""
import json
from pathlib import Path
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORE_DIR = PROJECT_ROOT / "core"
CLI_DIR = PROJECT_ROOT / "cli"
WEB_DIR = PROJECT_ROOT / "web"
TOOLS_DIR = PROJECT_ROOT / "tools"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))
if str(CLI_DIR) not in sys.path:
    sys.path.insert(0, str(CLI_DIR))
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

if "lmstudio" not in sys.modules:
    _mock_lmstudio = MagicMock()
    sys.modules["lmstudio"] = _mock_lmstudio


# Modules that bound a user directory at import time, with the attribute
# names they use for it. "from core.paths import USER_LOGS_DIR" copies the
# value into the importing module, so patching core.paths alone would not
# reach them.
_DB_ATTRIBUTES = (
    "DATABASE_FILE",
    "METADATA_DATABASE_FILE",
    "BENCHMARK_DB_PATH",
)
_DIR_ATTRIBUTES = (
    "USER_LOGS_DIR",
    "USER_RESULTS_DIR",
    "USER_DATA_DIR",
    "LOGS_DIR",
    "RESULTS_DIR",
)

_USER_DIR_BINDINGS = {
    "core.paths": _DIR_ATTRIBUTES,
    "app": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "web.app": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "run": _DIR_ATTRIBUTES,
    "core.tray": _DIR_ATTRIBUTES,
    "cli.benchmark": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "benchmark": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "cli.main": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "main": _DIR_ATTRIBUTES + _DB_ATTRIBUTES,
    "tools.scrape_metadata": _DIR_ATTRIBUTES,
}


@pytest.fixture(autouse=True)
def isolated_user_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the suite out of the developer's own logs and results.

    Several modules write to ~/.local/share/lm-studio-bench while a test
    merely exercises them: benchmark and tray launcher logs, database
    backups. Nothing there is overwritten, but the files pile up with every
    run. Redirect the bound directories at their importing modules; tests
    that patch these names themselves still win, because their patch is
    applied after this fixture.
    """
    sandbox = tmp_path / "user-dirs"
    logs_dir = sandbox / "logs"
    results_dir = sandbox / "results"
    for directory in (logs_dir, results_dir):
        directory.mkdir(parents=True, exist_ok=True)

    replacements = {
        "USER_LOGS_DIR": logs_dir,
        "LOGS_DIR": logs_dir,
        "USER_RESULTS_DIR": results_dir,
        "RESULTS_DIR": results_dir,
        "USER_DATA_DIR": sandbox,
        # Derived from RESULTS_DIR at import time, so they need redirecting
        # too - otherwise a test still reads the developer's own benchmark
        # history and its outcome depends on what happens to be in there.
        "DATABASE_FILE": results_dir / "benchmark_cache.db",
        "BENCHMARK_DB_PATH": results_dir / "benchmark_cache.db",
        "METADATA_DATABASE_FILE": results_dir / "model_metadata.db",
    }

    for module_name, attributes in _USER_DIR_BINDINGS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute in attributes:
            if hasattr(module, attribute):
                monkeypatch.setattr(module, attribute, replacements[attribute])

    return sandbox


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
def project_config_file(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Path:
    """Write a project config file to a temp directory and return its path."""
    config_path = tmp_path / "defaults.json"
    config_data: Dict[str, Any] = request.getfixturevalue("sample_config")
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    return config_path
