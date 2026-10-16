"""Execution of model-generated Python code for the code capability.

Measuring whether a model can write code means running what it wrote. That is
untrusted input, so every execution happens in a throwaway subprocess with:

* a temporary working directory that is deleted afterwards,
* an isolated interpreter (``-I -S``: no user site-packages, no ``PYTHONPATH``,
  no ``sitecustomize``),
* a stripped environment,
* a wall-clock timeout, plus CPU, address-space, file-size and process limits
  where the platform supports them,
* stdin connected to the platform's null device.

On Windows, a job object applies combined CPU and memory limits to the code
process and its descendants before the child is resumed. File-size limits
remain POSIX-only.

This contains accidents — endless loops, memory hogs, a stray ``open(...,'w')``
— and raises the cost of anything deliberate. It is not a security boundary
against a hostile model: a subprocess can still reach the network and read
files the user can read. Run benchmarks against models you are willing to
execute code from.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess  # nosec B404 - sandboxed execution is this module's purpose
import sys
import tempfile
from typing import NamedTuple, Optional

import psutil

from core.platform_info import IS_WINDOWS
from core.runtime import code_python_executable
from core.windows_job import WindowsJob

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no resource module
    resource = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_CPU_SECONDS = 10
MAX_ADDRESS_SPACE_BYTES = 512 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1024 * 1024
MAX_OUTPUT_CHARS = 4000

CODE_FENCE_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


class ExecutionResult(NamedTuple):
    """Outcome of running generated code against its checks.

    Attributes:
        passed: True when every assertion held.
        detail: Short human-readable reason, for reports and logs.
        stdout: Captured standard output, truncated.
        stderr: Captured standard error, truncated.
        timed_out: True when the wall-clock limit was hit.
    """

    passed: bool
    detail: str
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


def extract_code(response: str) -> str:
    """Pull the Python source out of a model response.

    Models wrap code in Markdown fences, sometimes after prose, sometimes not
    at all. The longest fenced block wins; without fences the whole response is
    treated as code.

    Args:
        response: Raw model output.

    Returns:
        Source code, stripped. Empty string when there is nothing to run.
    """
    if not response:
        return ""

    blocks = CODE_FENCE_PATTERN.findall(response)
    if blocks:
        return max(blocks, key=len).strip()

    return response.strip()


def _limit_resources() -> None:  # pragma: no cover - runs in the child process
    """Apply per-process resource limits before exec.

    Best effort: platforms without ``resource`` simply rely on the timeout.
    """
    if sys.platform == "win32" or resource is None:
        return

    limits = [
        (resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS)),
        (resource.RLIMIT_FSIZE, (MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_BYTES)),
        (
            resource.RLIMIT_AS,
            (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES),
        ),
    ]
    for limit_name, values in limits:
        try:
            resource.setrlimit(limit_name, values)
        except (ValueError, OSError):
            continue


def _truncate(text: str) -> str:
    """Cap captured output so reports stay readable."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n…[truncated]"


def _run_windows(
    command: list[str], workdir: str, timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a suspended child only after Windows accepts its resource limits."""
    env = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR")
        if key in os.environ
    }
    env.update({"TEMP": workdir, "TMP": workdir, "HOME": workdir,
                "USERPROFILE": workdir})
    with WindowsJob(MAX_ADDRESS_SPACE_BYTES, MAX_CPU_SECONDS) as job:
        # CREATE_SUSPENDED | CREATE_NO_WINDOW. No model code can run yet.
        with subprocess.Popen(  # nosec B603 - fixed interpreter, no shell
            command,
            cwd=workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
            creationflags=0x00000004 | 0x08000000,
        ) as process:
            try:
                job.assign(process.pid)
                psutil.Process(process.pid).resume()
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                if return_code is None:
                    raise RuntimeError("The code worker has no exit status")
                return subprocess.CompletedProcess(
                    command, return_code, stdout, stderr
                )
            finally:
                try:
                    job.close()
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


def run_code_test(
    code: str,
    check: str,
    timeout: Optional[float] = None,
) -> ExecutionResult:
    """Run generated code followed by its assertions.

    Args:
        code: Source produced by the model.
        check: Assertion snippet appended after the code. It runs in the same
            namespace, so it can call whatever the model defined.
        timeout: Wall-clock limit in seconds.

    Returns:
        ExecutionResult describing what happened.
    """
    source = extract_code(code)
    if not source:
        return ExecutionResult(False, "No code in response")

    effective_timeout = timeout or DEFAULT_TIMEOUT_SECONDS
    script = f"{source}\n\n# --- benchmark checks ---\n{check}\n"

    with tempfile.TemporaryDirectory(prefix="lmsb-code-") as workdir:
        script_path = Path(workdir) / "candidate.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            command = [
                code_python_executable(), "-I", "-S", "-X", "utf8",
                str(script_path),
            ]
            if IS_WINDOWS:
                completed = _run_windows(command, workdir, effective_timeout)
            else:
                completed = subprocess.run(  # nosec B603 - fixed argv, no shell
                    command,
                    cwd=workdir,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=effective_timeout,
                    env={"PATH": "/usr/bin:/bin", "HOME": workdir},
                    preexec_fn=_limit_resources,  # nosec B606
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                False,
                f"Timed out after {effective_timeout:.0f}s",
                timed_out=True,
            )
        except (OSError, RuntimeError, ValueError, psutil.Error) as exc:
            logger.error("Could not execute code test: %s", exc)
            return ExecutionResult(False, f"Could not execute: {exc}")

        stdout = _truncate(completed.stdout or "")
        stderr = _truncate(completed.stderr or "")

        if completed.returncode == 0:
            return ExecutionResult(True, "All checks passed", stdout, stderr)

        last_line = ""
        for line in reversed(stderr.strip().splitlines()):
            if line.strip():
                last_line = line.strip()
                break

        return ExecutionResult(
            False,
            last_line or f"Exited with code {completed.returncode}",
            stdout,
            stderr,
        )
