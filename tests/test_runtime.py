"""Tests for source and packaged Python entrypoint handling."""

import os
from pathlib import Path
import sys

import pytest

from core import runtime


@pytest.mark.parametrize(
    ("windows", "suffix"),
    [(True, ("Scripts", "python.exe")), (False, ("bin", "python"))],
)
def test_prefers_the_platform_virtualenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    windows: bool,
    suffix: tuple[str, str],
) -> None:
    """Interpreter lookup uses Scripts on Windows and bin on Unix."""
    monkeypatch.setattr(runtime, "IS_WINDOWS", windows)
    executable = tmp_path / ".venv" / Path(*suffix)
    executable.parent.mkdir(parents=True)
    executable.write_text("interpreter", encoding="utf-8")
    executable.chmod(0o755)
    assert runtime.python_executable(tmp_path) == str(executable)


def test_falls_back_to_current_interpreter(tmp_path: Path) -> None:
    """Missing project environments do not select Store aliases from PATH."""
    assert runtime.python_executable(tmp_path) == sys.executable


def test_environment_uses_native_path_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive colons must not become Python import-path separators."""
    existing = str(tmp_path / "other")
    monkeypatch.setenv("PYTHONPATH", existing)
    result = runtime.child_environment(tmp_path)
    assert result["PYTHONPATH"].split(os.pathsep) == [str(tmp_path), existing]
    assert result["PYTHONIOENCODING"] == "utf-8"
    assert result["PYTHONUTF8"] == "1"
    assert os.environ["PYTHONPATH"] == existing


def test_command_preserves_windows_paths_and_unicode(tmp_path: Path) -> None:
    """The source command is an argv list, with no quoting or shell splitting."""
    script = tmp_path / "cli" / "benchmark.py"
    script.parent.mkdir()
    script.touch()
    args = ["--output-dir", "C:\\Users\\\u00c4nne\\One Drive\\results"]
    command = runtime.entrypoint_command(
        "classic", args, project_root=tmp_path
    )
    assert command == [sys.executable, str(script), *args]


def test_missing_source_entrypoint_fails_explicitly(tmp_path: Path) -> None:
    """Incomplete source checkouts fail before spawning any worker."""
    with pytest.raises(FileNotFoundError, match="entrypoint not found"):
        runtime.entrypoint_command("webapp", [], project_root=tmp_path)


def test_unknown_role_is_rejected() -> None:
    """Only known application modules may be dispatched."""
    with pytest.raises(ValueError, match="Unknown application role"):
        runtime.entrypoint_command("arbitrary.module", [])


def test_frozen_commands_dispatch_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen executable must not receive a Python script as an argument."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    command = runtime.entrypoint_command("classic", ["--limit", "1"])
    assert command == [
        sys.executable, "--internal-role", "classic", "--limit", "1"
    ]


def test_missing_frozen_code_runtime_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code tests must never mistake the packaged launcher for Python."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    with pytest.raises(FileNotFoundError, match="code-test interpreter"):
        runtime.code_python_executable()


def test_frozen_code_runtime_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frozen code tests select the dedicated real interpreter."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    executable = tmp_path / "code_runtime" / "python.exe"
    executable.parent.mkdir()
    executable.touch()
    assert runtime.code_python_executable() == str(executable)
