#!/usr/bin/env python3
"""
Wrapper script - Entry point for the benchmark tool

Usage:
    ./run.py [args]                 - Starts classic benchmark
    ./run.py --agent MODEL [args]   - Starts capability-driven agent
    ./run.py --webapp               - Starts FastAPI web dashboard
    ./run.py -w                     - Starts FastAPI web dashboard (short form)

Examples:
    ./run.py --limit 5                    - Tests 5 new models (classic)
    ./run.py --export-only                - Generates reports from cache
    ./run.py --agent 'llama-13b'          - Tests model capabilities
    ./run.py --agent 'llama-13b' --capabilities general_text,reasoning
    ./run.py --webapp                     - Starts web dashboard
    ./run.py -w                           - Starts web dashboard (short form)
"""

from datetime import datetime
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess  # nosec B404
import sys
import threading
import time
from typing import TextIO

from core.paths import USER_LOGS_DIR, format_path_for_logs

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


PYTHON_EXECUTABLE = _resolve_python_executable()


def _tray_python_candidates() -> list[str]:
    """Return Python candidates for tray startup."""
    raw_candidates = [
        PYTHON_EXECUTABLE,
        sys.executable,
        "/usr/bin/python3",
        "python3",
    ]
    unique_candidates: list[str] = []
    for candidate in raw_candidates:
        if not candidate:
            continue
        resolved_candidate = candidate
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            resolved_candidate = shutil.which(candidate) or ""
        if not resolved_candidate:
            continue
        if resolved_candidate not in unique_candidates:
            unique_candidates.append(resolved_candidate)
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


def _has_agent_flag(cli_args: list[str]) -> bool:
    """Check if --agent flag is present in CLI arguments."""
    return "--agent" in cli_args or any(
        arg.startswith("--agent=") for arg in cli_args
    )


def _build_subprocess_env() -> dict[str, str]:
    """Build a sanitized environment for child Python processes."""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    root_dir = str(project_root)
    pythonpath_entries = [root_dir]
    existing_path = env.get("PYTHONPATH", "")
    if existing_path:
        pythonpath_entries.append(existing_path)
    env["PYTHONPATH"] = ":".join(pythonpath_entries)

    appdir_candidate = project_root.parents[2]
    app_lib_dir = appdir_candidate / "usr" / "lib"
    app_lib_arch_dir = app_lib_dir / "x86_64-linux-gnu"
    app_gi_arch_dir = app_lib_arch_dir / "girepository-1.0"
    app_gi_dir = app_lib_dir / "girepository-1.0"

    is_appimage_runtime = (
        app_lib_dir.is_dir()
        or app_lib_arch_dir.is_dir()
        or "/.mount_" in str(project_root)
    )

    if not is_appimage_runtime:
        return env

    gi_paths = [
        str(path)
        for path in (app_gi_arch_dir, app_gi_dir)
        if path.is_dir()
    ]
    for system_gi in (
        "/usr/lib/x86_64-linux-gnu/girepository-1.0",
        "/usr/lib/girepository-1.0",
        "/usr/lib64/girepository-1.0",
    ):
        if Path(system_gi).is_dir() and system_gi not in gi_paths:
            gi_paths.append(system_gi)
    if gi_paths:
        existing_gi = env.get("GI_TYPELIB_PATH", "")
        if existing_gi:
            env["GI_TYPELIB_PATH"] = ":".join(gi_paths + [existing_gi])
        else:
            env["GI_TYPELIB_PATH"] = ":".join(gi_paths)

    return env


_ALLOWED_ARG_RE = re.compile(r"^[A-Za-z0-9_./:=,@%+\-]+$")


def _sanitize_cli_args(cli_args: list[str]) -> list[str]:
    """Validate CLI arguments before forwarding to subprocesses."""
    sanitized: list[str] = []
    for arg in cli_args:
        if any(char in arg for char in ("\x00", "\n", "\r")):
            raise ValueError(f"Invalid control character in argument: {arg!r}")
        if not _ALLOWED_ARG_RE.fullmatch(arg):
            raise ValueError(f"Unsupported characters in argument: {arg!r}")
        sanitized.append(arg)
    return sanitized


def _expand_short_flag_clusters(cli_args: list[str]) -> list[str]:
    """Expand supported short-flag clusters (e.g. ``-wd``).

    Only clusters that contain known boolean flags are expanded.
    Other arguments are passed through unchanged.

    Args:
        cli_args: Raw CLI arguments (without argv[0]).

    Returns:
        Normalized argument list.
    """
    combinable_flags = {"w", "d", "h"}
    normalized: list[str] = []

    for arg in cli_args:
        if arg.startswith("--") or not arg.startswith("-"):
            normalized.append(arg)
            continue

        if len(arg) <= 2:
            normalized.append(arg)
            continue

        cluster = arg[1:]
        if all(flag in combinable_flags for flag in cluster):
            normalized.extend(f"-{flag}" for flag in cluster)
        else:
            normalized.append(arg)

    return normalized


def _start_tray_process(
    tray_dashboard_url: str,
    tray_debug_enabled: bool,
) -> subprocess.Popen | None:
    """Start tray app as background subprocess."""
    tray_script = project_root / "core" / "tray.py"
    if not tray_script.exists():
        print(f"⚠️ Tray script not found: {format_path_for_logs(tray_script)}")
        return None

    logs_dir = USER_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log = logs_dir / f"runapp_{timestamp}.log"
    latest_link = logs_dir / "runapp_latest.log"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(launcher_log.name)

    env = _build_subprocess_env()

    for candidate in _tray_python_candidates():
        tray_cmd = [candidate, str(tray_script), "--url", tray_dashboard_url]
        if tray_debug_enabled:
            tray_cmd.append("--debug")

        try:
            start_offset = launcher_log.stat().st_size if launcher_log.exists() else 0
            _append_tray_launcher_log(
                launcher_log,
                f"CMD: {' '.join(tray_cmd)}",
            )
            tray_proc = subprocess.Popen(  # nosec B603
                tray_cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            log_thread = _start_tray_log_thread(tray_proc, launcher_log)

            time.sleep(0.6)
            if tray_proc.poll() is None:
                print(f"🧩 Tray started ({tray_dashboard_url})")
                print(f"📝 Run log: {format_path_for_logs(launcher_log)}")
                return tray_proc

            if log_thread is not None:
                log_thread.join(timeout=1.0)

            failure_excerpt = _read_tray_launcher_excerpt(
                launcher_log,
                start_offset,
            )

            print(
                "⚠️ Tray exited early "
                f"(code {tray_proc.returncode}) "
                f"with {format_path_for_logs(candidate)}"
            )
            failure_summary = _summarize_tray_failure(failure_excerpt)
            if failure_summary:
                print(failure_summary)
        except OSError as error:
            _append_tray_launcher_log(
                launcher_log,
                f"ERROR: Could not launch {candidate}: {error}",
            )
            print(
                "⚠️ Could not launch tray with "
                f"{format_path_for_logs(candidate)}: {error}"
            )

    print(f"📝 Tray launcher log: {format_path_for_logs(launcher_log)}")
    try:
        launcher_text = launcher_log.read_text(encoding="utf-8")
        if "symbol lookup error" in launcher_text:
            print("⚠️ Tray failed due to system GTK library mismatch.")
            print(
                "💡 Try launching from a non-Snap terminal/session or "
                "restart VS Code outside Snap."
            )
        if "ModuleNotFoundError: No module named 'httpx'" in launcher_text:
            print(
                "⚠️ System Python fallback failed because dependency "
                "'httpx' is missing."
            )
    except OSError:
        pass
    return None


def _append_tray_launcher_log(log_path: Path, message: str) -> None:
    """Append a timestamped tray launcher log entry."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"{timestamp} {message.rstrip()}\n")


def _stream_tray_output_to_log(stream: TextIO, log_path: Path) -> None:
    """Write tray subprocess output to the launcher log with timestamps."""
    try:
        for output_line in stream:
            _append_tray_launcher_log(log_path, output_line)
    finally:
        stream.close()


def _start_tray_log_thread(
    tray_proc: subprocess.Popen,
    log_path: Path,
) -> threading.Thread | None:
    """Start background log streaming for tray subprocess output."""
    stream = tray_proc.stdout
    if stream is None:
        return None

    log_thread = threading.Thread(
        target=_stream_tray_output_to_log,
        args=(stream, log_path),
        daemon=True,
    )
    log_thread.start()
    return log_thread


def _read_tray_launcher_excerpt(log_path: Path, start_offset: int) -> str:
    """Read launcher log text appended after a given file offset."""
    try:
        with open(log_path, "r", encoding="utf-8") as log_handle:
            log_handle.seek(start_offset)
            return log_handle.read()
    except OSError:
        return ""


def _summarize_tray_failure(log_text: str) -> str:
    """Summarize known tray launcher failures for console output."""
    if "symbol lookup error" in log_text:
        return (
            "💥 Bundled tray Python failed due to a GLIBC/Snap library "
            "mismatch."
        )
    if "ModuleNotFoundError: No module named 'httpx'" in log_text:
        return (
            "📦 System Python fallback is missing required dependency "
            "'httpx'."
        )
    if "Traceback" in log_text:
        return "🐍 System Python fallback failed with a Python traceback."
    return ""


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


RAW_ARGS = sys.argv[1:]
CLI_ARGS = _expand_short_flag_clusters(RAW_ARGS)


if "--help" in CLI_ARGS or "-h" in CLI_ARGS:
    print("LM Studio Model Benchmark - Entry Point")
    print("=" * 60)
    print()
    print("📊 BENCHMARK MODES:")
    print()
    print("  1️⃣  Classic Benchmark (Default):")
    print("      ./run.py [benchmark-args]")
    print("      → Tests token/s speed on all models")
    print()
    print("  2️⃣  Capability-Driven Agent:")
    print("      ./run.py --agent 'model-id' [agent-args]")
    print("      → Tests model capabilities (general, reasoning, vision, tooling)")
    print(
        "      → Generates JSON, CSV, PDF & HTML reports "
        "with quality/performance metrics"
    )
    print()
    print("  3️⃣  Web Dashboard (Recommended):")
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
    print("🤖 CAPABILITY-DRIVEN AGENT OPTIONS:")
    print()
    print("  --agent MODEL_PATH    Runs capability-driven benchmark")
    print("  --capabilities CAPS   Comma-separated capabilities to test")
    print("                        (general_text,reasoning,vision,tooling)")
    print("  --output-dir DIR      Output directory for results")
    print("  --config FILE         Configuration YAML file")
    print("  --formats FORMATS     Output formats (json,html,csv,pdf)")
    print("  --max-tests N         Maximum tests per capability")
    print("  --context-length N    Model context length")
    print("  --gpu-offload RATIO   GPU offload ratio (0.0-1.0)")
    print("  --temperature TEMP    Generation temperature")
    print("  -v, --verbose         Enable verbose logging")
    print()
    print("=" * 60)
    print()
    print("📋 CLASSIC BENCHMARK OPTIONS:")
    print()

    benchmark_script = project_root / "cli" / "benchmark.py"
    if benchmark_script.exists():
        result = subprocess.run(  # nosec B603
            [PYTHON_EXECUTABLE, str(benchmark_script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=_build_subprocess_env(),
        )
        lines = result.stdout.split("\n")
        IN_OPTIONS = False
        for line in lines:
            if line.startswith("options:") or line.startswith("  -"):
                IN_OPTIONS = True
            if IN_OPTIONS:
                print(line)

    print()
    print("📚 EXAMPLES:")
    print()
    print("  Classic benchmark:")
    print("    ./run.py --limit 5")
    print()
    print("  Capability-driven agent:")
    print("    ./run.py --agent 'llama-13b' --capabilities general_text,reasoning")
    print()
    print("  Web dashboard:")
    print("    ./run.py --webapp")
    print()

    sys.exit(0)

HAS_WEB_FLAG = "--webapp" in CLI_ARGS or "-w" in CLI_ARGS
HAS_AGENT_FLAG = _has_agent_flag(CLI_ARGS)
DEBUG_ENABLED = "--debug" in CLI_ARGS or "-d" in CLI_ARGS

TRAY_PROCESS = None

if HAS_WEB_FLAG:
    args = [arg for arg in CLI_ARGS if arg not in ("--webapp", "-w")]
    web_port = _extract_port(args)
    if web_port is None:
        web_port = _find_free_port()
        args.extend(["--port", str(web_port)])

    try:
        safe_args = _sanitize_cli_args(args)
    except ValueError as error:
        print(f"❌ Invalid CLI arguments: {error}")
        sys.exit(2)

    DASHBOARD_URL = f"http://localhost:{web_port}"
    TRAY_PROCESS = _start_tray_process(DASHBOARD_URL, DEBUG_ENABLED)

    app_script = project_root / "web" / "app.py"
    if not app_script.exists():
        print(f"❌ Error: {app_script} not found")
        print("💡 Tip: Please run the web dashboard setup first")
        sys.exit(1)

    print("🌐 Starting FastAPI web dashboard...")
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(app_script)] + safe_args,
            env=_build_subprocess_env(),
            check=False,
        )
        sys.exit(result.returncode)
    finally:
        _stop_tray_process(TRAY_PROCESS)
elif HAS_AGENT_FLAG:
    AGENT_MODEL = None
    for cli_arg in CLI_ARGS:
        if cli_arg.startswith("--agent="):
            extracted = cli_arg.split("=", 1)[1]
            if not extracted:
                print(
                    "❌ Invalid --agent argument: expected "
                    "--agent MODEL or --agent=MODEL"
                )
                sys.exit(2)
            AGENT_MODEL = extracted

    args = [
        arg
        for arg in CLI_ARGS
        if arg != "--agent" and not arg.startswith("--agent=")
    ]

    if AGENT_MODEL is not None:
        args.insert(0, AGENT_MODEL)
    try:
        safe_args = _sanitize_cli_args(args)
    except ValueError as error:
        print(f"❌ Invalid CLI arguments: {error}")
        sys.exit(2)

    agent_script = project_root / "cli" / "main.py"
    if not agent_script.exists():
        print(f"❌ Error: {agent_script} not found")
        sys.exit(1)

    print("🤖 Starting capability-driven benchmark agent...")
    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, "-m", "cli.main"] + safe_args,
            cwd=project_root,
            env=_build_subprocess_env(),
            check=False,
        )
        sys.exit(result.returncode)
    except (subprocess.SubprocessError, OSError) as error:
        print(f"❌ Error starting agent: {error}")
        sys.exit(1)

else:
    benchmark_script = project_root / "cli" / "benchmark.py"
    TRAY_PROCESS = _start_tray_process("http://localhost:1234", DEBUG_ENABLED)

    if not benchmark_script.exists():
        print(f"❌ Error: {benchmark_script} not found")
        _stop_tray_process(TRAY_PROCESS)
        sys.exit(1)

    try:
        safe_args = _sanitize_cli_args(CLI_ARGS)
    except ValueError as error:
        print(f"❌ Invalid CLI arguments: {error}")
        _stop_tray_process(TRAY_PROCESS)
        sys.exit(2)

    try:
        result = subprocess.run(
            [PYTHON_EXECUTABLE, str(benchmark_script)] + safe_args,
            env=_build_subprocess_env(),
            check=False,
        )
        sys.exit(result.returncode)
    finally:
        _stop_tray_process(TRAY_PROCESS)
