"""Shared interpreter, environment and entrypoint handling."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from core.platform_info import IS_WINDOWS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_CONSOLE_EXECUTABLE = "lmstudio-bench.exe"
WINDOWS_GUI_EXECUTABLE = "LM-Studio-Bench.exe"
ENTRYPOINTS = {
    "webapp": ("web", "app.py"),
    "classic": ("cli", "benchmark.py"),
    "capability": ("cli", "main.py"),
    "tray": ("core", "tray.py"),
    "metadata": ("tools", "scrape_metadata.py"),
    "selftest": ("scripts", "windows_selftest.py"),
}


def python_executable(project_root: Path = PROJECT_ROOT) -> str:
    """Return the local virtualenv interpreter, or the current interpreter.

    Args:
        project_root: Directory containing the optional virtual environment.

    Returns:
        Absolute interpreter path.

    Raises:
        FileNotFoundError: If a frozen GUI installation has no console worker.
    """
    if getattr(sys, "frozen", False):
        if Path(sys.executable).name.casefold() == WINDOWS_GUI_EXECUTABLE.casefold():
            worker = Path(sys.executable).with_name(WINDOWS_CONSOLE_EXECUTABLE)
            if not worker.is_file():
                raise FileNotFoundError("The packaged console worker is missing")
            return str(worker)
        return sys.executable
    suffix = ("Scripts", "python.exe") if IS_WINDOWS else ("bin", "python")
    candidate = project_root / ".venv" / Path(*suffix)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return sys.executable


def child_environment(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Build a Python child environment with native separators and UTF-8.

    Args:
        project_root: Import root for source-mode entrypoints.

    Returns:
        A copy of the environment; the parent environment is not changed.
    """
    env = os.environ.copy()
    if getattr(sys, "frozen", False):
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
    else:
        entries = [str(project_root)]
        if env.get("PYTHONPATH"):
            entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(entries)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def entrypoint_command(
    role: str,
    args: list[str],
    *,
    interpreter: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[str]:
    """Build an argv list for one of the application's known entrypoints.

    Args:
        role: Entry in ENTRYPOINTS.
        args: Arguments already validated by the caller.
        interpreter: Optional interpreter selected by the launcher.
        project_root: Source directory containing the entrypoint.

    Returns:
        Arguments for a shell-free subprocess.

    Raises:
        ValueError: If the role is unknown.
        FileNotFoundError: If a source entrypoint is missing.
    """
    if role not in ENTRYPOINTS:
        raise ValueError(f"Unknown application role: {role}")
    executable = interpreter or python_executable(project_root)
    if getattr(sys, "frozen", False):
        return [executable, "--internal-role", role, *args]
    script = project_root.joinpath(*ENTRYPOINTS[role])
    if not script.is_file():
        raise FileNotFoundError(f"Application entrypoint not found: {script}")
    if role == "capability":
        return [executable, "-u", "-m", "cli.main", *args]
    return [executable, str(script), *args]


def configure_console() -> None:
    """Make Windows redirected Python output support the existing log icons."""
    if not IS_WINDOWS:
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = str(getattr(stream, "encoding", "")).lower().replace("-", "")
        if callable(reconfigure) and encoding != "utf8":
            reconfigure(encoding="utf-8", errors="backslashreplace")


def child_creation_flags() -> int:
    """Keep Windows background workers console-free and in their own group."""
    if IS_WINDOWS:
        return 0x08000000 | 0x00000200
    return 0


def code_python_executable() -> str:
    """Return a real interpreter for isolated generated-code tests.

    Returns:
        The current Python in source mode, or the bundled worker interpreter.

    Raises:
        FileNotFoundError: If a frozen build has no real worker interpreter.
    """
    if not getattr(sys, "frozen", False):
        return sys.executable
    bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    interpreter = bundle / "code_runtime" / "python.exe"
    if not interpreter.is_file():
        raise FileNotFoundError("The bundled code-test interpreter is missing")
    return str(interpreter)
