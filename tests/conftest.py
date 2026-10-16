"""Shared fixtures and configuration for the test suite."""
from collections.abc import Iterator
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

_SESSION_HOME: tempfile.TemporaryDirectory[str] | None = None
_OLD_ENV: dict[str, str | None] = {}
_ROOT_LOGGER = logging.getLogger()
_LOGGER_TYPE = logging.Logger
_FILE_HANDLER_TYPE = logging.FileHandler


def _close_test_handlers(directory: Path) -> None:
    """Release only log handles created below an owned test directory."""
    loggers = [_ROOT_LOGGER, *_ROOT_LOGGER.manager.loggerDict.values()]
    for logger in loggers:
        if not isinstance(logger, _LOGGER_TYPE):
            continue
        for handler in list(logger.handlers):
            if not isinstance(handler, _FILE_HANDLER_TYPE):
                continue
            filename = getattr(handler, "baseFilename", None)
            if isinstance(filename, str) and Path(filename).is_relative_to(directory):
                logger.removeHandler(handler)
                handler.close()


def pytest_configure() -> None:
    """Isolate user directories before collection imports project modules."""
    global _SESSION_HOME
    _SESSION_HOME = tempfile.TemporaryDirectory(prefix="lmsbench-tests-")
    home = Path(_SESSION_HOME.name)
    replacements = {
        "XDG_CONFIG_HOME": home / "config",
        "XDG_DATA_HOME": home / "data",
        "XDG_CACHE_HOME": home / "cache",
        "APPDATA": home / "roaming",
        "LOCALAPPDATA": home / "local",
        "SNAP_REAL_HOME": home / "home",
    }
    for key, value in replacements.items():
        _OLD_ENV[key] = os.environ.get(key)
        os.environ[key] = str(value)


def pytest_unconfigure() -> None:
    """Restore environment overrides and remove only this session's data."""
    if _SESSION_HOME is not None:
        _close_test_handlers(Path(_SESSION_HOME.name))
        _SESSION_HOME.cleanup()
    for key, value in _OLD_ENV.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def isolated_native_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native hardware access is opt-in; ordinary unit tests use fixtures."""
    if os.environ.get("LM_STUDIO_NATIVE_SMOKE") == "1":
        return
    import tools.hardware_monitor as hardware

    def disabled_probe(*_args, **_kwargs):
        from tools.windows_gpu import WindowsGpuError

        raise WindowsGpuError("Native GPU probe is disabled in ordinary unit tests")

    if hasattr(hardware, "query_windows_snapshot"):
        monkeypatch.setattr(hardware, "query_windows_snapshot", disabled_probe)

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
def isolated_user_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
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

    yield sandbox
    _close_test_handlers(tmp_path)


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
