"""Portable process-control tests, including real owned Windows children."""

from collections.abc import Iterator
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import MagicMock

import psutil
import pytest

from core import process_control
from core.runtime import child_creation_flags, child_environment

Worker = tuple[subprocess.Popen[str], Path]

WORKER = """
from pathlib import Path
import sys
import time
from core.process_control import start_stop_listener

root = Path(sys.argv[1])
start_stop_listener()
try:
    with (root / "heartbeat").open("ab", buffering=0) as output:
        output.write(b".")
        (root / "ready").touch()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            output.write(b".")
            time.sleep(0.02)
except KeyboardInterrupt:
    (root / "cleanup").write_text("cleaned", encoding="utf-8")
"""


@pytest.fixture
def worker(tmp_path: Path) -> Iterator[Worker]:
    """Run one bounded child; always clean up its exact process and pipes."""
    env = child_environment()
    env[process_control.CONTROL_ENV] = "1"
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", WORKER, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        env=env,
        creationflags=child_creation_flags(),
    )
    descendants: list[psutil.Process] = []
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise AssertionError("The test worker did not become ready")
            time.sleep(0.02)
        descendants = psutil.Process(process.pid).children(recursive=True)
        yield process, tmp_path
    finally:
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(descendants, timeout=5)
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


def test_real_pause_resume_and_cooperative_stop(worker: Worker) -> None:
    """Native pause/resume works and a stop reaches the worker's cleanup."""
    process, root = worker
    process_control.pause_process(process)
    time.sleep(0.1)
    size = (root / "heartbeat").stat().st_size
    time.sleep(0.1)
    assert (root / "heartbeat").stat().st_size == size

    process_control.resume_process(process)
    deadline = time.monotonic() + 3
    while (root / "heartbeat").stat().st_size == size:
        assert time.monotonic() < deadline, "The worker did not resume"
        time.sleep(0.02)

    process_control.stop_process(process)
    assert process.returncode == 0
    assert (root / "cleanup").read_text(encoding="utf-8") == "cleaned"


def test_real_stop_resumes_a_paused_worker(worker: Worker) -> None:
    """Stopping a paused process must allow its finally/interrupt cleanup."""
    process, root = worker
    process_control.pause_process(process)

    process_control.stop_process(process, paused=True)

    assert process.returncode == 0
    assert (root / "cleanup").exists()


def test_stop_does_not_kill_another_python_process(worker: Worker) -> None:
    """Stopping one child must not affect an unrelated same-name process."""
    process, _ = worker
    unrelated = subprocess.Popen(
        [
            getattr(sys, "_base_executable", sys.executable),
            "-c", "import time; time.sleep(30)",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=child_creation_flags(),
    )
    try:
        process_control.stop_process(process)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_stop_escalates_after_termination_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that ignores termination is killed by its owned handle."""
    process = MagicMock()
    process.poll.return_value = None
    process.stdin = None
    process.wait.side_effect = [
        subprocess.TimeoutExpired("benchmark", 3), 0
    ]
    owner = MagicMock()
    owner.children.return_value = []
    monkeypatch.setattr(process_control.psutil, "Process", lambda _pid: owner)

    process_control.stop_process(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_finished_process_is_not_signalled() -> None:
    """An already exited child cannot accidentally target a reused PID."""
    process = MagicMock()
    process.poll.return_value = 0

    process_control.stop_process(process)

    process.terminate.assert_not_called()
    process.kill.assert_not_called()
