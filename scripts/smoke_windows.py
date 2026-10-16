"""Exercise a copied Windows bundle with OS-only PATH and isolated user data.

Only the supplied bundle is executed. This never runs an installer, changes
registry settings, launches a browser/tray icon, or starts a benchmark.
All cleanup is restricted to handles and descendants owned by this test.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import psutil

LOGGER = logging.getLogger(__name__)


def create_owned_job() -> Any:
    """Keep every smoke descendant in a bounded, kill-on-close Windows job."""
    original = sys.path[:]
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from core.windows_job import WindowsJob
    finally:
        sys.path[:] = original
    return WindowsJob(4 * 1024**3, 120)


def isolated_environment(workspace: Path) -> dict[str, str]:
    """Create a per-run profile and remove Python/SDK/user PATH dependencies."""
    root = Path(os.environ["SystemRoot"])
    environment = {
        key: value for key, value in os.environ.items()
        if key.upper() in {
            "SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE",
            "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432",
            "NUMBER_OF_PROCESSORS", "PATHEXT",
        }
    }
    environment["PATH"] = os.pathsep.join(
        str(path) for path in (root / "System32", root, root / "System32" / "Wbem")
    )
    directories = {
        "USERPROFILE": workspace / "profile",
        "HOME": workspace / "profile",
        "APPDATA": workspace / "profile" / "Roaming",
        "LOCALAPPDATA": workspace / "profile" / "Local",
        "TEMP": workspace / "scratch",
        "TMP": workspace / "scratch",
    }
    for key, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        environment[key] = str(directory)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    return environment


def run_command(
    command: list[str], environment: dict[str, str], cwd: Path, expected: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Run a bounded packaged command and retain complete failure diagnostics."""
    with create_owned_job() as job:
        with subprocess.Popen(  # nosec B603 - fixed, owned validation process
            command, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004,
        ) as process:
            try:
                job.assign(process.pid)
                psutil.Process(process.pid).resume()
                stdout, stderr = process.communicate(timeout=90)
                result = subprocess.CompletedProcess(
                    command, process.returncode, stdout, stderr
                )
            finally:
                try:
                    job.close()
                finally:
                    stop_owned_tree(process)
    if result.returncode != expected:
        raise RuntimeError(
            f"{command!r} returned {result.returncode}, expected {expected}\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def request(port: int, method: str, path: str) -> tuple[int, bytes]:
    """Make a direct loopback HTTP request, independent of proxy settings."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def stop_owned_tree(process: subprocess.Popen[Any]) -> None:
    """Stop only this test's process and its identified descendants."""
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, survivors = psutil.wait_procs(children, timeout=5)
    for child in survivors:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    psutil.wait_procs(survivors, timeout=5)


def check_dashboard(
    executable: Path, gui: bool, environment: dict[str, str], workspace: Path,
) -> None:
    """Start a managed dashboard, request real HTML, and stop it via its API."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = [] if gui else ["--webapp"]
    args += ["--no-tray", "--no-browser", "--port", str(port)]
    log_path = workspace / ("gui-dashboard.log" if gui else "console-dashboard.log")
    descendants: list[psutil.Process] = []
    with create_owned_job() as job, log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(  # nosec B603 - owned packaged entrypoint
            [str(executable), *args], env=environment, cwd=workspace,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004,
        )
        try:
            job.assign(process.pid)
            psutil.Process(process.pid).resume()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Dashboard exited early; see {log_path}")
                try:
                    status, data = request(port, "GET", "/health")
                    if status == 200:
                        health = json.loads(data)
                        if health.get("benchmark_running") is not False:
                            raise RuntimeError(
                                "Dashboard unexpectedly started a benchmark"
                            )
                        break
                except (OSError, http.client.HTTPException):
                    time.sleep(0.25)
            else:
                raise RuntimeError(f"Dashboard did not become ready; see {log_path}")
            status, html = request(port, "GET", "/")
            if status != 200 or b"LM Studio" not in html:
                raise RuntimeError("Dashboard HTML/template resources are unavailable")
            descendants = psutil.Process(process.pid).children(recursive=True)
            status, data = request(port, "POST", "/api/system/shutdown")
            if status != 200 or json.loads(data).get("success") is not True:
                raise RuntimeError("Managed dashboard refused shutdown")
            if process.wait(timeout=30) != 0:
                raise RuntimeError(f"Dashboard shutdown failed; see {log_path}")
        finally:
            try:
                job.close()
            finally:
                stop_owned_tree(process)
            _, survivors = psutil.wait_procs(descendants, timeout=5)
            for child in survivors:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(survivors, timeout=5)
            if survivors:
                raise RuntimeError("Dashboard left an owned descendant running")


def file_snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Record installation files so smoke checks can detect runtime writes."""
    return {
        str(path.relative_to(directory)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*") if path.is_file()
    }


def smoke(bundle: Path, workspace: Path) -> dict[str, Any]:
    """Copy a bundle, exercise every worker path, and report checked invariants."""
    if sys.platform != "win32":
        raise ValueError("Windows package smoke checks require Windows")
    workspace = workspace.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Smoke workspace must be empty; no existing data is removed")
    workspace.mkdir(parents=True, exist_ok=True)
    copied = workspace / "Portable Grüße 日本語's"
    shutil.copytree(bundle.resolve(), copied)
    before = file_snapshot(copied)
    environment = isolated_environment(workspace)
    console = copied / "lmstudio-bench.exe"
    gui = copied / "LM-Studio-Bench.exe"
    result = run_command([str(console), "--help"], environment, workspace)
    if (
        "CLASSIC BENCHMARK OPTIONS" not in result.stdout
        or "--limit" not in result.stdout
    ):
        raise RuntimeError("Classic CLI help is incomplete")
    result = run_command(
        [str(console), "--internal-role", "capability", "--help"],
        environment, workspace,
    )
    if "--capabilities" not in result.stdout:
        raise RuntimeError("Capability CLI help is incomplete")
    run_command(
        [str(console), "--internal-role", "not-a-real-role"],
        environment, workspace, expected=2,
    )
    result = run_command(
        [str(console), "--internal-role", "metadata", "--no-hf"],
        environment, workspace, expected=1,
    )
    if "Error during scraping" not in result.stdout + result.stderr:
        raise RuntimeError("Missing lms CLI did not produce a graceful scraper error")
    result = run_command(
        [str(console), "--internal-role", "selftest"], environment, workspace,
    )
    code_report = json.loads(result.stdout)
    if not all(code_report["checks"].values()):
        raise RuntimeError(f"Packaged runtime selftest failed: {code_report}")
    native = run_command(
        [str(copied / "_internal" / "native" / "gpu_probe.exe"), "--once"],
        environment, workspace,
    )
    json.loads(native.stdout)
    check_dashboard(console, False, environment, workspace)
    check_dashboard(gui, True, environment, workspace)
    if file_snapshot(copied) != before:
        raise RuntimeError("The application wrote into its installation directory")
    report = {
        "cli_help": True, "capability_help": True, "unknown_role_rejected": True,
        "scraper_offline_failure": True, "native_json": True,
        "console_dashboard_shutdown": True, "gui_dashboard_shutdown": True,
        "no_installation_writes": True, "os_only_path": environment["PATH"],
        "code_runtime": code_report,
        "installer_executed": False,
    }
    (workspace / "smoke-result.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run portable validation without making any machine-wide changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        smoke(args.bundle, args.workspace)
        LOGGER.info("Portable smoke checks passed: %s", args.workspace)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
