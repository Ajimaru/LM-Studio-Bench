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
import importlib.util
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

from core.logging_utils import update_latest_log
from core.paths import USER_LOGS_DIR, format_path_for_logs
from core.platform_info import IS_LINUX, IS_MACOS, IS_WINDOWS
from core.runtime import (
    child_creation_flags,
    child_environment,
    configure_console,
    entrypoint_command,
    python_executable,
)

project_root = Path(__file__).parent


def _resolve_python_executable() -> str:
    """Resolve Python executable for subprocesses.

    Prefer local virtualenv interpreter when available so subprocesses use
    project dependencies even if run.py was launched by a system Python.
    """
    return python_executable(project_root)


PYTHON_EXECUTABLE = _resolve_python_executable()


def _tray_python_candidates() -> list[str]:
    """Return Python candidates for tray startup."""
    raw_candidates = [
        PYTHON_EXECUTABLE,
        sys.executable,
    ]
    if IS_LINUX:
        raw_candidates.extend(["/usr/bin/python3", "python3"])
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


# GTK/GLib lookup paths that a Snap-confined host (for example the VS Code
# snap) exports into its integrated terminal. Values pointing into /snap make
# GTK load modules built against the snap base, which drags in a mismatched
# glibc and kills the tray with a symbol lookup error.
SNAP_GTK_ENV_VARS = (
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_IM_MODULE_FILE",
    "GIO_MODULE_DIR",
    "GI_TYPELIB_PATH",
    "GSETTINGS_SCHEMA_DIR",
    "GDK_PIXBUF_MODULE_FILE",
    "GDK_PIXBUF_MODULEDIR",
    "LOCPATH",
)


def _is_snap_path(value: str) -> bool:
    """Check if any path entry lives inside a Snap tree."""
    return any(
        "/snap/" in entry
        for entry in value.split(os.pathsep)
        if entry
    )


def _drop_snap_gtk_paths(env: dict[str, str]) -> None:
    """Remove GTK/GLib env vars that resolve into a Snap runtime.

    Snap also exports per-user caches below ``~/snap/<name>/``, so both the
    read-only ``/snap/`` tree and the home-directory variant are matched.
    """
    for name in SNAP_GTK_ENV_VARS:
        value = env.get(name)
        if value and _is_snap_path(value):
            env.pop(name, None)


def _build_subprocess_env() -> dict[str, str]:
    """Build a sanitized environment for child Python processes."""
    env = child_environment(project_root)
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    # macOS equivalents of LD_LIBRARY_PATH/LD_PRELOAD.
    env.pop("DYLD_LIBRARY_PATH", None)
    env.pop("DYLD_INSERT_LIBRARIES", None)
    env.pop("DYLD_FRAMEWORK_PATH", None)
    _drop_snap_gtk_paths(env)
    # AppImage/GI bootstrapping is Linux-only packaging.
    if not IS_LINUX:
        return env

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
            env["GI_TYPELIB_PATH"] = os.pathsep.join(gi_paths + [existing_gi])
        else:
            env["GI_TYPELIB_PATH"] = os.pathsep.join(gi_paths)

    return env


def _run_child_process(argv: list[str], cwd: Path | None = None) -> int:
    """Run a project entrypoint in a child interpreter, return its exit code.

    The three launch paths below (dashboard, agent, classic benchmark) differ
    only in what they append to the argument list, so the call itself lives
    here: one place that owns the sanitized environment, and one place for a
    security audit to look at. Every argv starts with PYTHON_EXECUTABLE and
    carries only arguments that passed _sanitize_cli_args.
    """
    environment = _build_subprocess_env()
    options: dict[str, int] = {}
    if IS_WINDOWS and environment.get("_LM_STUDIO_BENCH_WINDOWLESS") == "1":
        options["creationflags"] = child_creation_flags()
    completed = subprocess.run(  # nosec B603 - fixed argv, sanitized flags
        argv,
        cwd=cwd,
        env=environment,
        check=False,
        **options,
    )
    return completed.returncode


_ALLOWED_ARG_RE = re.compile(r"^[^;&|`$<>!\x00-\x1f\x7f]+$")


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


def _tray_skip_reason() -> str | None:
    """Return why the GTK tray cannot start here, or None if it can.

    The tray relies on PyGObject/GTK, which is packaged for Linux desktops
    only. On other platforms the tray is skipped rather than launched and
    left to fail, so the dashboard and CLI output stays clean.

    On Linux the tray is always attempted: ``_start_tray_process`` tries
    several interpreters, and a system Python may provide ``gi`` even when
    the project virtualenv does not.
    """
    if IS_LINUX:
        return None

    if IS_MACOS:
        return (
            "ℹ️ System tray skipped: the GTK tray is Linux-only. "
            "Benchmarks and the web dashboard are unaffected."
        )

    if IS_WINDOWS:
        try:
            from core.windows_tray import windows_tray_unavailable_reason

            reason = windows_tray_unavailable_reason()
        except (ImportError, OSError, RuntimeError) as error:
            reason = str(error)
        if reason is not None:
            return f"ℹ️ System tray skipped: {reason}"
        return None

    if importlib.util.find_spec("gi") is not None:
        return None

    return (
        "ℹ️ System tray skipped: PyGObject (gi) is not available on this "
        "platform."
    )


def _start_tray_process(
    tray_dashboard_url: str,
    tray_debug_enabled: bool,
) -> subprocess.Popen | None:
    """Start tray app as background subprocess."""
    skip_reason = _tray_skip_reason()
    if skip_reason is not None:
        # Flush so the notice keeps its place when stdout is a pipe or log.
        print(skip_reason, flush=True)
        return None

    tray_script = project_root / "core" / "tray.py"
    if not getattr(sys, "frozen", False) and not tray_script.exists():
        print(f"⚠️ Tray script not found: {format_path_for_logs(tray_script)}")
        return None

    logs_dir = USER_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log = logs_dir / f"runapp_{timestamp}.log"
    latest_link = logs_dir / "runapp_latest.log"
    update_latest_log(launcher_log, latest_link)

    env = _build_subprocess_env()

    for candidate in _tray_python_candidates():
        tray_cmd = entrypoint_command(
            "tray",
            ["--url", tray_dashboard_url],
            interpreter=candidate,
            project_root=project_root,
        )
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
                encoding="utf-8",
                bufsize=1,
                env=env,
                creationflags=child_creation_flags(),
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


def _print_help() -> None:
    """Print wrapper help and the classic benchmark options."""
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
    print("  --no-tray             Skip the optional system tray")
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
    if getattr(sys, "frozen", False) or benchmark_script.exists():
        result = subprocess.run(  # nosec B603
            entrypoint_command(
                "classic", ["--help"],
                interpreter=PYTHON_EXECUTABLE, project_root=project_root,
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
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

def _dispatch_internal_role(role: str, args: list[str]) -> int:
    """Run a fixed packaged entrypoint, preserving its argv and exit behavior."""
    from core.runtime import ENTRYPOINTS

    if role not in ENTRYPOINTS:
        print(f"❌ Unknown application role: {role}", file=sys.stderr)
        return 2

    original_argv = sys.argv
    original_directory = Path.cwd()
    try:
        sys.argv = [sys.executable, *args]
        os.chdir(project_root)
        if role == "webapp":
            from web.app import main as entrypoint
        elif role == "classic":
            from cli.benchmark import main as entrypoint
        elif role == "capability":
            from cli.main import main as entrypoint
        elif role == "tray":
            from core.tray import main as entrypoint
        elif role == "metadata":
            from tools.scrape_metadata import main as entrypoint
        else:
            from scripts.windows_selftest import main as entrypoint
        result = entrypoint()
        return 0 if result is None else int(result)
    finally:
        sys.argv = original_argv
        os.chdir(original_directory)


def main(argv: list[str] | None = None) -> int:
    """Run the requested application mode without import-time subprocesses.

    Args:
        argv: CLI arguments without the executable, or None to use sys.argv.

    Returns:
        The child exit code, 2 for invalid arguments, or 1 for launch failure.
    """
    configure_console()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "--internal-role":
        if len(raw_args) < 2:
            print("❌ --internal-role requires a role", file=sys.stderr)
            return 2
        return _dispatch_internal_role(raw_args[1], raw_args[2:])
    cli_args = _expand_short_flag_clusters(raw_args)
    if "--help" in cli_args or "-h" in cli_args:
        _print_help()
        return 0

    role = "classic"
    no_tray = "--no-tray" in cli_args
    cli_args = [arg for arg in cli_args if arg != "--no-tray"]
    args = cli_args[:]
    dashboard_url = "http://localhost:1234"
    if "--webapp" in cli_args or "-w" in cli_args:
        role = "webapp"
        args = [arg for arg in cli_args if arg not in ("--webapp", "-w")]
        port = _extract_port(args)
        if port is None:
            port = _find_free_port()
            args.extend(["--port", str(port)])
        dashboard_url = f"http://localhost:{port}"
    elif _has_agent_flag(cli_args):
        role = "capability"
        model = None
        for arg in cli_args:
            if arg.startswith("--agent="):
                model = arg.split("=", 1)[1]
                if not model:
                    print("❌ --agent requires a model name or path")
                    return 2
        args = [
            arg for arg in cli_args
            if arg != "--agent" and not arg.startswith("--agent=")
        ]
        if model is not None:
            args.insert(0, model)

    try:
        safe_args = _sanitize_cli_args(args)
    except ValueError as error:
        print(f"❌ Invalid CLI arguments: {error}")
        return 2
    try:
        command = entrypoint_command(
            role, safe_args,
            interpreter=PYTHON_EXECUTABLE,
            project_root=project_root,
        )
    except (OSError, ValueError) as error:
        print(f"❌ Cannot launch application: {error}")
        return 1

    tray_process = None
    try:
        if role != "capability" and not no_tray:
            tray_process = _start_tray_process(
                dashboard_url, "--debug" in cli_args or "-d" in cli_args
            )
        if role == "webapp":
            print("🌐 Starting FastAPI web dashboard...")
        elif role == "capability":
            print("🤖 Starting capability-driven benchmark agent...")
        return _run_child_process(command, cwd=project_root)
    except KeyboardInterrupt:
        print("🛑 Interrupted")
        return 130
    except (subprocess.SubprocessError, OSError) as error:
        print(f"❌ Error starting application: {error}")
        return 1
    finally:
        _stop_tray_process(tray_process)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(main())
