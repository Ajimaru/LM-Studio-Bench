#!/usr/bin/env python3
"""
Wrapper script - Entry point for the benchmark tool

Usage:
    ./run.py [args]              - Starts normal benchmark
    ./run.py --webapp            - Starts FastAPI web dashboard
    ./run.py -w                  - Starts FastAPI web dashboard (short form)

Examples:
    ./run.py --limit 5           - Tests 5 new models
    ./run.py --export-only       - Generates reports from cache
    ./run.py --webapp            - Starts web dashboard
    ./run.py -w                  - Starts web dashboard (short form)
"""

import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).parent
os.chdir(project_root)


def _resolve_python_executable() -> str:
    """Resolve Python executable for subprocesses.

    Prefer local virtualenv interpreter when available so subprocesses use
    project dependencies even if run.py was launched by a system Python.
    """
    venv_python = project_root / ".venv" / "bin" / "python"
    if venv_python.exists() and os.access(venv_python, os.X_OK):
        return str(venv_python)
    return sys.executable


python_executable = _resolve_python_executable()


def _tray_python_candidates() -> list[str]:
    """Return Python candidates for tray startup."""
    candidates = [
        python_executable,
        sys.executable,
        "/usr/bin/python3",
        "python3",
    ]
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _find_free_port() -> int:
    """Find an available local TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _extract_port(cli_args: list[str]) -> int | None:
    """Extract --port/-p value from argument list."""
    for index, arg in enumerate(cli_args):
        if arg.startswith("--port="):
            value = arg.split("=", maxsplit=1)[1]
            if value.isdigit():
                return int(value)
        if arg == "--port" and index + 1 < len(cli_args):
            value = cli_args[index + 1]
            if value.isdigit():
                return int(value)
        if arg == "-p" and index + 1 < len(cli_args):
            value = cli_args[index + 1]
            if value.isdigit():
                return int(value)
    return None


def _build_subprocess_env() -> dict[str, str]:
    """Build a sanitized environment for child Python processes."""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    return env


def _start_tray_process(
    tray_dashboard_url: str,
    tray_debug_enabled: bool,
) -> subprocess.Popen | None:
    """Start tray app as background subprocess."""
    tray_script = project_root / "src" / "tray.py"
    if not tray_script.exists():
        print(f"⚠️ Tray script not found: {tray_script}")
        return None

    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log = logs_dir / f"tray_launcher_{timestamp}.log"

    env = _build_subprocess_env()

    for candidate in _tray_python_candidates():
        tray_cmd = [candidate, str(tray_script), "--url", tray_dashboard_url]
        if tray_debug_enabled:
            tray_cmd.append("--debug")

        try:
            with open(launcher_log, "a", encoding="utf-8") as log_handle:
                log_handle.write(f"CMD: {' '.join(tray_cmd)}\n")
                tray_proc = subprocess.Popen(
                    tray_cmd,
                    cwd=project_root,
                    stdout=log_handle,
                    stderr=log_handle,
                    env=env,
                )

            time.sleep(0.6)
            if tray_proc.poll() is None:
                print(f"🧩 Tray started ({tray_dashboard_url})")
                print(f"📝 Tray launcher log: {launcher_log}")
                return tray_proc

            print(
                "⚠️ Tray exited early "
                f"(code {tray_proc.returncode}) with {candidate}"
            )
        except OSError as error:
            print(f"⚠️ Could not launch tray with {candidate}: {error}")

    print(f"📝 Tray launcher log: {launcher_log}")
    try:
        launcher_text = launcher_log.read_text(encoding="utf-8")
        if "symbol lookup error" in launcher_text:
            print("⚠️ Tray failed due to system GTK library mismatch.")
            print(
                "💡 Try launching from a non-Snap terminal/session or "
                "restart VS Code outside Snap."
            )
    except OSError:
        pass
    return None


def _stop_tray_process(tray_proc: subprocess.Popen | None) -> None:
    """Stop tray process gracefully if still running."""
    if tray_proc is None:
        return
    if tray_proc.poll() is not None:
        return

    try:
        tray_proc.terminate()
        tray_proc.wait(timeout=3)
    except (
        subprocess.SubprocessError,
        ProcessLookupError,
        TimeoutError,
        OSError,
    ):
        try:
            tray_proc.kill()
        except (subprocess.SubprocessError, ProcessLookupError, OSError):
            pass


# Show extended help for --help/-h
if "--help" in sys.argv or "-h" in sys.argv:
    print("LM Studio Model Benchmark - Entry Point")
    print("=" * 60)
    print()
    print("📊 BENCHMARK MODES:")
    print()
    print("  1️⃣  CLI Benchmark (Default):")
    print("      ./run.py [benchmark-args]")
    print("      → Runs benchmark directly and shows results")
    print()
    print("  2️⃣  Web Dashboard (Recommended):")
    print("      ./run.py --webapp  (or -w)")
    print("      → Starts modern web interface with live streaming")
    print("      → Automatically opens browser at http://localhost:8080")
    print("      → Features: Live logs, results browser, dark mode")
    print()
    print("=" * 60)
    print()
    print("🌐 WEB DASHBOARD OPTIONS:")
    print()
    print("  --webapp, -w          Starts FastAPI web dashboard")
    print()
    print("=" * 60)
    print()
    print("📋 BENCHMARK OPTIONS (for CLI mode):")
    print()

    benchmark_script = project_root / "src" / "benchmark.py"
    if benchmark_script.exists():
        result = subprocess.run(
            [python_executable, str(benchmark_script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=_build_subprocess_env(),
        )
        lines = result.stdout.split("\n")
        in_options = False
        for line in lines:
            if line.startswith("options:") or line.startswith("  -"):
                in_options = True
            if in_options:
                print(line)

    sys.exit(0)

has_web_flag = "--webapp" in sys.argv or "-w" in sys.argv
debug_enabled = "--debug" in sys.argv or "-d" in sys.argv

tray_process = None

if has_web_flag:
    args = [arg for arg in sys.argv[1:] if arg not in ("--webapp", "-w")]
    web_port = _extract_port(args)
    if web_port is None:
        web_port = _find_free_port()
        args.extend(["--port", str(web_port)])

    dashboard_url = f"http://localhost:{web_port}"
    tray_process = _start_tray_process(dashboard_url, debug_enabled)

    app_script = project_root / "web" / "app.py"
    if not app_script.exists():
        print(f"❌ Error: {app_script} not found")
        print("💡 Tip: Please run the web dashboard setup first")
        sys.exit(1)

    print("🌐 Starting FastAPI web dashboard...")
    try:
        sys.exit(
            subprocess.call(
                [python_executable, str(app_script)] + args,
                env=_build_subprocess_env(),
            )
        )
    finally:
        _stop_tray_process(tray_process)
else:
    benchmark_script = project_root / "src" / "benchmark.py"
    tray_process = _start_tray_process("http://localhost:1234", debug_enabled)

    if not benchmark_script.exists():
        print(f"❌ Error: {benchmark_script} not found")
        _stop_tray_process(tray_process)
        sys.exit(1)

    try:
        sys.exit(
            subprocess.call(
                [python_executable, str(benchmark_script)] + sys.argv[1:],
                env=_build_subprocess_env(),
            )
        )
    finally:
        _stop_tray_process(tray_process)
