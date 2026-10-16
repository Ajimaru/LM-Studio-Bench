"""Control owned benchmark processes without assuming POSIX signals."""

from __future__ import annotations

import _thread
import logging
import os
import signal
import subprocess
import sys
import threading

import psutil

from core.platform_info import IS_WINDOWS

LOGGER = logging.getLogger(__name__)
CONTROL_ENV = "LMS_BENCH_CONTROL_STDIN"


def start_stop_listener() -> None:
    """Listen for the dashboard's private stop pipe when explicitly enabled.

    Direct CLI runs do not consume stdin. A stop or closed parent pipe raises
    KeyboardInterrupt in the main thread, allowing existing finally blocks
    to release models. Raw reads avoid a daemon holding a buffered-stdin lock
    during interpreter shutdown.
    """
    if os.environ.get(CONTROL_ENV) != "1":
        return
    if sys.stdin is None:
        raise RuntimeError("Benchmark control stdin is unavailable")
    descriptor = sys.stdin.fileno()

    def listen() -> None:
        pending = b""
        try:
            while True:
                chunk = os.read(descriptor, 64)
                if not chunk:
                    LOGGER.info("Benchmark controller disconnected")
                    _thread.interrupt_main()
                    return
                pending += chunk
                if b"\n" not in pending and len(pending) <= 64:
                    continue
                if pending.strip() == b"stop":
                    LOGGER.info("Benchmark stop requested")
                    _thread.interrupt_main()
                    return
                LOGGER.warning("Ignoring invalid benchmark control message")
                pending = b""
        except OSError as error:
            LOGGER.error("Benchmark control pipe failed: %s", error)
            _thread.interrupt_main()

    threading.Thread(target=listen, name="benchmark-control", daemon=True).start()


def pause_process(process: subprocess.Popen[str]) -> None:
    """Pause an owned benchmark process.

    Args:
        process: Child created by this application.

    Raises:
        OSError: If signalling fails.
        psutil.Error: If the Windows process cannot be suspended.
    """
    if IS_WINDOWS:
        suspended: list[psutil.Process] = []

        def suspend_tree(owner: psutil.Process) -> None:
            # Windows venv python.exe is a redirector: suspending only that
            # launcher leaves the real interpreter running in its child.
            owner.suspend()
            suspended.append(owner)
            for child in owner.children():
                try:
                    suspend_tree(child)
                except psutil.NoSuchProcess:
                    LOGGER.debug("Owned child %s exited during pause", child.pid)

        try:
            suspend_tree(psutil.Process(process.pid))
        except psutil.Error:
            for owner in reversed(suspended):
                try:
                    owner.resume()
                except psutil.NoSuchProcess:
                    LOGGER.debug("Owned child %s already exited", owner.pid)
                except psutil.Error as error:
                    LOGGER.error("Could not roll back process suspension: %s", error)
            raise
    else:
        stop_signal = getattr(signal, "SIGSTOP", None)
        if stop_signal is None:
            raise OSError("Process suspension is unavailable on this platform")
        process.send_signal(stop_signal)


def resume_process(process: subprocess.Popen[str]) -> None:
    """Resume an owned benchmark process.

    Args:
        process: Child created by this application.

    Raises:
        OSError: If signalling fails.
        psutil.Error: If the Windows process cannot be resumed.
    """
    if IS_WINDOWS:
        owner = psutil.Process(process.pid)
        for child in reversed(owner.children(recursive=True)):
            try:
                child.resume()
            except psutil.NoSuchProcess:
                LOGGER.debug("Owned child %s exited during resume", child.pid)
        owner.resume()
    else:
        continue_signal = getattr(signal, "SIGCONT", None)
        if continue_signal is None:
            raise OSError("Process resuming is unavailable on this platform")
        process.send_signal(continue_signal)


def _stop_descendants(descendants: list[psutil.Process]) -> None:
    """Attempt cleanup of every recorded child even if one is inaccessible."""
    for child in descendants:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            LOGGER.debug("Owned child %s already exited", child.pid)
        except psutil.Error as error:
            LOGGER.error("Could not terminate owned child %s: %s", child.pid, error)
    _, remaining = psutil.wait_procs(descendants, timeout=3)
    for child in remaining:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            LOGGER.debug("Owned child %s already exited", child.pid)
        except psutil.Error as error:
            LOGGER.error("Could not kill owned child %s: %s", child.pid, error)
    _, remaining = psutil.wait_procs(remaining, timeout=3)
    if remaining:
        raise RuntimeError("An owned benchmark child did not exit")


def stop_process(
    process: subprocess.Popen[str],
    *,
    paused: bool = False,
    timeout: float = 5,
) -> None:
    """Request cleanup, then terminate only the owned child tree if needed.

    Args:
        process: Benchmark child with an optional private stdin pipe.
        paused: Whether the child must be resumed to handle cleanup.
        timeout: Maximum wait for cooperative shutdown.

    Raises:
        OSError: If process termination or pipe handling fails.
        psutil.Error: If owned descendants cannot be terminated.
        subprocess.TimeoutExpired: If a forcefully killed child does not exit.
        RuntimeError: If a descendant remains alive after bounded cleanup.
    """
    if process.poll() is not None:
        process.wait()
        return
    owner = None
    try:
        owner = psutil.Process(process.pid)
        descendants = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        LOGGER.debug("Benchmark exited before its descendants were inspected")
        descendants = []

    try:
        if paused:
            resume_process(process)
        if process.stdin is not None:
            try:
                process.stdin.write("stop\n")
                process.stdin.flush()
                process.wait(timeout=timeout)
                LOGGER.info("Benchmark stopped after cooperative cleanup")
                return
            except (BrokenPipeError, OSError, ValueError) as error:
                LOGGER.warning("Could not request benchmark cleanup: %s", error)
            except subprocess.TimeoutExpired:
                LOGGER.warning("Benchmark cleanup timed out; terminating child")

        if owner is not None:
            try:
                descendants.extend(
                    child for child in owner.children(recursive=True)
                    if child not in descendants
                )
            except psutil.NoSuchProcess:
                LOGGER.debug("Benchmark exited during its stop request")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            LOGGER.warning("Benchmark did not terminate; killing owned child")
            process.kill()
            process.wait(timeout=3)
    finally:
        try:
            _stop_descendants(descendants)
        finally:
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except (OSError, ValueError) as error:
                    LOGGER.warning("Could not close benchmark control pipe: %s", error)
