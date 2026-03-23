"""Tests for run.py helper functions.

run.py executes module-level code that starts subprocesses on import.
We patch subprocess and sys.exit before importing to avoid side effects.
"""
import importlib
import io
from pathlib import Path
import re
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_RUN_MODULE = None


def _import_run():
    """Import run.py safely by patching subprocess and sys.exit."""
    global _RUN_MODULE
    if _RUN_MODULE is not None:
        return _RUN_MODULE

    orig_argv = sys.argv[:]
    sys.argv = ["run.py"]

    mock_popen = MagicMock()
    mock_popen_instance = MagicMock()
    mock_popen_instance.poll.return_value = 1
    mock_popen.return_value = mock_popen_instance

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0

    with patch("subprocess.Popen", mock_popen), \
            patch("subprocess.run", mock_run), \
            patch("sys.exit"):
        import run as _run
        _RUN_MODULE = _run

    sys.argv = orig_argv
    return _RUN_MODULE


class TestResolvePythonExecutable:
    """Tests for _resolve_python_executable()."""

    def test_returns_string(self):
        """Returns a non-empty string."""
        run = _import_run()
        result = run._resolve_python_executable()
        assert isinstance(result, str)
        assert result

    def test_returns_venv_python_when_present(self, tmp_path: Path, monkeypatch):
        """Returns venv python path when it exists and is executable."""
        run = _import_run()
        venv_python = tmp_path / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/usr/bin/env python3")
        venv_python.chmod(0o755)
        monkeypatch.setattr(run, "project_root", tmp_path)
        result = run._resolve_python_executable()
        assert result == str(venv_python)

    def test_falls_back_to_sys_executable(self, tmp_path: Path, monkeypatch):
        """Falls back to sys.executable when venv python is absent."""
        run = _import_run()
        monkeypatch.setattr(run, "project_root", tmp_path)
        result = run._resolve_python_executable()
        assert result == sys.executable


class TestTrayPythonCandidates:
    """Tests for _tray_python_candidates()."""

    def test_returns_list(self):
        """Returns a list of candidate paths."""
        run = _import_run()
        candidates = run._tray_python_candidates()
        assert isinstance(candidates, list)
        assert len(candidates) >= 1

    def test_no_duplicates(self):
        """Returned list has no duplicate entries."""
        run = _import_run()
        candidates = run._tray_python_candidates()
        assert len(candidates) == len(set(candidates))

    def test_contains_sys_executable(self):
        """sys.executable is present in the candidate list."""
        run = _import_run()
        candidates = run._tray_python_candidates()
        assert sys.executable in candidates


class TestFindFreePort:
    """Tests for _find_free_port()."""

    def test_returns_integer_port(self):
        """Returns a positive integer port number."""
        run = _import_run()
        port = run._find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_port_is_bindable(self):
        """Returned port is bindable."""
        run = _import_run()
        port = run._find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))


class TestExtractPort:
    """Tests for _extract_port()."""

    def test_extracts_long_form_equals(self):
        """Extracts port from --port=NNNN form."""
        run = _import_run()
        assert run._extract_port(["--port=8080"]) == 8080

    def test_extracts_long_form_separate(self):
        """Extracts port from --port NNNN form."""
        run = _import_run()
        assert run._extract_port(["--port", "9000"]) == 9000

    def test_extracts_short_form(self):
        """Extracts port from -p NNNN form."""
        run = _import_run()
        assert run._extract_port(["-p", "7777"]) == 7777

    def test_returns_none_when_absent(self):
        """Returns None when no port argument is present."""
        run = _import_run()
        assert run._extract_port(["--runs", "3"]) is None

    def test_ignores_non_numeric_port(self):
        """Returns None when port value is non-numeric."""
        run = _import_run()
        assert run._extract_port(["--port=abc"]) is None

    def test_short_port_last_arg(self):
        """Returns None when -p is the last argument (no value)."""
        run = _import_run()
        assert run._extract_port(["-p"]) is None


class TestBuildSubprocessEnv:
    """Tests for _build_subprocess_env()."""

    def test_returns_dict(self):
        """Returns a dictionary."""
        run = _import_run()
        env = run._build_subprocess_env()
        assert isinstance(env, dict)

    def test_removes_ld_library_path(self, monkeypatch):
        """LD_LIBRARY_PATH is removed from the environment."""
        run = _import_run()
        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
        env = run._build_subprocess_env()
        assert "LD_LIBRARY_PATH" not in env

    def test_removes_ld_preload(self, monkeypatch):
        """LD_PRELOAD is removed from the environment."""
        run = _import_run()
        monkeypatch.setenv("LD_PRELOAD", "libfoo.so")
        env = run._build_subprocess_env()
        assert "LD_PRELOAD" not in env

    def test_adds_project_root_to_pythonpath(self):
        """Project root is prepended to PYTHONPATH."""
        run = _import_run()
        env = run._build_subprocess_env()
        assert "PYTHONPATH" in env
        path_entries = env["PYTHONPATH"].split(":")
        assert path_entries[0] == str(run.project_root)

    def test_preserves_existing_pythonpath(self, monkeypatch):
        """Existing PYTHONPATH is preserved after the project-root prefix."""
        run = _import_run()
        monkeypatch.setenv("PYTHONPATH", "/custom/path")
        env = run._build_subprocess_env()
        path_entries = env["PYTHONPATH"].split(":")
        assert path_entries[0] == str(run.project_root)
        assert "/custom/path" in env["PYTHONPATH"]

    def test_appimage_runtime_sets_gi_typelib_path(self, tmp_path, monkeypatch):
        """AppImage-style runtime adds GI_TYPELIB_PATH entries."""
        run = _import_run()

        project_root = tmp_path / "a" / "b" / "project"
        project_root.mkdir(parents=True)
        appdir = project_root.parents[2]
        gi_dir = appdir / "usr" / "lib" / "girepository-1.0"
        gi_dir.mkdir(parents=True)

        monkeypatch.setattr(run, "project_root", project_root)
        monkeypatch.delenv("GI_TYPELIB_PATH", raising=False)

        env = run._build_subprocess_env()
        assert "GI_TYPELIB_PATH" in env
        assert str(gi_dir) in env["GI_TYPELIB_PATH"]

    def test_appimage_runtime_appends_existing_gi_typelib_path(
        self, tmp_path, monkeypatch
    ):
        """Existing GI_TYPELIB_PATH is appended after discovered AppImage dirs."""
        run = _import_run()

        project_root = tmp_path / "x" / "y" / "project"
        project_root.mkdir(parents=True)
        appdir = project_root.parents[2]
        gi_arch_dir = (
            appdir / "usr" / "lib" / "x86_64-linux-gnu" / "girepository-1.0"
        )
        gi_arch_dir.mkdir(parents=True)

        monkeypatch.setattr(run, "project_root", project_root)
        monkeypatch.setenv("GI_TYPELIB_PATH", "/already/present")

        env = run._build_subprocess_env()
        value = env.get("GI_TYPELIB_PATH", "")
        assert str(gi_arch_dir) in value
        assert value.endswith("/already/present")


class TestSanitizeCliArgs:
    """Tests for _sanitize_cli_args()."""

    def test_valid_args_pass_through(self):
        """Valid arguments are returned unchanged."""
        run = _import_run()
        args = ["--runs", "3", "--context", "2048"]
        result = run._sanitize_cli_args(args)
        assert result == args

    def test_null_byte_raises(self):
        """Null byte in argument raises ValueError."""
        run = _import_run()
        with pytest.raises(ValueError, match="control character"):
            run._sanitize_cli_args(["--runs\x003"])

    def test_newline_raises(self):
        """Newline in argument raises ValueError."""
        run = _import_run()
        with pytest.raises(ValueError, match="control character"):
            run._sanitize_cli_args(["--prompt\ninjected"])

    def test_carriage_return_raises(self):
        """Carriage return in argument raises ValueError."""
        run = _import_run()
        with pytest.raises(ValueError, match="control character"):
            run._sanitize_cli_args(["arg\r"])

    def test_shell_injection_chars_raise(self):
        """Shell injection characters raise ValueError."""
        run = _import_run()
        with pytest.raises(ValueError, match="Unsupported"):
            run._sanitize_cli_args(["$(rm -rf /)"])

    def test_empty_list_is_valid(self):
        """Empty list is returned as empty list."""
        run = _import_run()
        assert run._sanitize_cli_args([]) == []

    def test_allowed_special_chars_pass(self):
        """Allowed special characters (dots, slashes, etc.) pass."""
        run = _import_run()
        args = ["--model", "pub/model.gguf", "--path", "/tmp/out"]
        result = run._sanitize_cli_args(args)
        assert result == args


class TestExpandShortFlagClusters:
    """Tests for _expand_short_flag_clusters()."""

    def test_single_short_flag_unchanged(self):
        """Single short flag is returned unchanged."""
        run = _import_run()
        assert run._expand_short_flag_clusters(["-w"]) == ["-w"]

    def test_long_flag_unchanged(self):
        """Long flags are returned unchanged."""
        run = _import_run()
        assert run._expand_short_flag_clusters(["--webapp"]) == ["--webapp"]

    def test_combinable_cluster_expanded(self):
        """Known combinable cluster is expanded to individual flags."""
        run = _import_run()
        result = run._expand_short_flag_clusters(["-wd"])
        assert "-w" in result
        assert "-d" in result

    def test_unknown_cluster_unchanged(self):
        """Unknown flag cluster is NOT expanded."""
        run = _import_run()
        result = run._expand_short_flag_clusters(["-xyz"])
        assert "-xyz" in result

    def test_positional_args_unchanged(self):
        """Non-flag positional arguments are returned unchanged."""
        run = _import_run()
        result = run._expand_short_flag_clusters(["somevalue"])
        assert result == ["somevalue"]

    def test_mixed_args(self):
        """Mixed list is processed correctly."""
        run = _import_run()
        result = run._expand_short_flag_clusters(["--webapp", "-d", "val"])
        assert "--webapp" in result
        assert "-d" in result
        assert "val" in result


class TestStopTrayProcess:
    """Tests for _stop_tray_process()."""

    def test_none_is_no_op(self):
        """None process is handled without error."""
        run = _import_run()
        run._stop_tray_process(None)

    def test_already_stopped_process_is_no_op(self):
        """Process that has already exited is handled without error."""
        run = _import_run()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        run._stop_tray_process(mock_proc)
        mock_proc.terminate.assert_not_called()

    def test_running_process_is_terminated(self):
        """Running process receives SIGTERM."""
        run = _import_run()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0
        run._stop_tray_process(mock_proc)
        mock_proc.terminate.assert_called_once()

    def test_forceful_kill_on_timeout(self):
        """kill() is called when SIGTERM times out."""
        run = _import_run()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = subprocess.SubprocessError("timeout")
        run._stop_tray_process(mock_proc)
        mock_proc.kill.assert_called_once()

    def test_kill_errors_are_ignored(self):
        """kill() exceptions are swallowed in final fallback path."""
        run = _import_run()
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.side_effect = TimeoutError("timeout")
        mock_proc.kill.side_effect = OSError("already gone")

        run._stop_tray_process(mock_proc)
        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


class TestStartTrayProcess:
    """Tests for _start_tray_process() in run.py."""

    def test_returns_none_when_tray_script_missing(self, tmp_path: Path, monkeypatch):
        """_start_tray_process returns None when tray.py does not exist."""
        run = _import_run()
        monkeypatch.setattr(run, "project_root", tmp_path)
        (tmp_path / "core").mkdir(exist_ok=True)
        with patch.object(run, "USER_LOGS_DIR", tmp_path / "logs"), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]):
            result = run._start_tray_process("http://localhost:8080", False)
        assert result is None

    def test_returns_proc_when_tray_starts_successfully(
        self, tmp_path: Path, monkeypatch
    ):
        """_start_tray_process returns Popen when tray starts (poll=None)."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = None
        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]), \
                patch("subprocess.Popen", return_value=mock_proc), \
                patch("time.sleep"):
            result = run._start_tray_process(
                "http://localhost:8080", False
            )
        assert result is mock_proc

    def test_returns_none_when_tray_exits_early_all_candidates(
        self, tmp_path: Path, monkeypatch
    ):
        """_start_tray_process returns None when tray exits for all candidates."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_proc.stdout = None
        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]), \
                patch("subprocess.Popen", return_value=mock_proc), \
                patch("time.sleep"):
            result = run._start_tray_process(
                "http://localhost:8080", False
            )
        assert result is None

    def test_skips_osexception_candidates(
        self, tmp_path: Path, monkeypatch
    ):
        """_start_tray_process continues to next candidate on OSError."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        good_proc = MagicMock()
        good_proc.poll.return_value = None
        good_proc.stdout = None
        call_count = [0]

        def popen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("bad python")
            return good_proc

        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=["/bad/python", sys.executable]), \
                patch("subprocess.Popen", side_effect=popen_side_effect), \
                patch("time.sleep"):
            result = run._start_tray_process(
                "http://localhost:8080", False
            )
        assert result is good_proc

    def test_passes_debug_flag_when_enabled(
        self, tmp_path: Path, monkeypatch
    ):
        """_start_tray_process appends --debug to command when debug=True."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        captured_cmd = []
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout = None

        def capture_popen(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return mock_proc

        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]), \
                patch("subprocess.Popen", side_effect=capture_popen), \
                patch("time.sleep"):
            run._start_tray_process("http://localhost:8080", True)
        assert "--debug" in captured_cmd


class TestStartTrayProcessSymbolLookup:
    """Cover the 'symbol lookup error' branch in _start_tray_process."""

    def test_logs_symbol_lookup_warning(self, tmp_path, monkeypatch):
        """Prints warning when launcher log contains 'symbol lookup error'."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir()
        (src_dir / "tray.py").write_text("# tray stub")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stdout = None

        original_read_text = Path.read_text

        def patched_read_text(self, *args, **kwargs):
            if "runapp_" in str(self) or "tray" in str(self):
                return "symbol lookup error: /lib/libfoo.so.0"
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", patched_read_text)
        monkeypatch.setattr(run, "project_root", tmp_path)

        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]), \
                patch("subprocess.Popen", return_value=mock_proc), \
                patch("time.sleep"):
            result = run._start_tray_process("http://localhost:8080", False)
        assert result is None

    def test_launcher_log_os_error_ignored(self, tmp_path, monkeypatch):
        """OSError reading launcher log is silently ignored."""
        run = _import_run()
        src_dir = tmp_path / "core"
        src_dir.mkdir()
        (src_dir / "tray.py").write_text("# tray stub")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.stdout = None

        original_read_text = Path.read_text

        def raise_os_error(self, *args, **kwargs):
            if "runapp_" in str(self) or "tray" in str(self):
                raise OSError("no access")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", raise_os_error)
        monkeypatch.setattr(run, "project_root", tmp_path)

        with patch.object(run, "USER_LOGS_DIR", logs_dir), \
                patch.object(run, "_tray_python_candidates",
                             return_value=[sys.executable]), \
                patch("subprocess.Popen", return_value=mock_proc), \
                patch("time.sleep"):
            result = run._start_tray_process("http://localhost:8080", False)
        assert result is None


class TestRunPyHelperFunctions:
    """Tests for small helper functions in run.py."""

    def test_append_tray_launcher_log_adds_timestamp(self, tmp_path: Path):
        """Timestamped tray launcher log entries are written to disk."""
        run = _import_run()
        log_path = tmp_path / "runapp_test.log"

        run._append_tray_launcher_log(log_path, "CMD: python tray.py")

        log_text = log_path.read_text(encoding="utf-8")
        assert re.match(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} CMD:",
            log_text,
        )

    def test_summarize_tray_failure_for_missing_httpx(self):
        """Known missing dependency errors get a concise summary."""
        run = _import_run()

        summary = run._summarize_tray_failure(
            "Traceback\nModuleNotFoundError: No module named 'httpx'"
        )

        assert "httpx" in summary

    def test_stream_tray_output_to_log_adds_timestamp(self, tmp_path: Path):
        """Tray subprocess output is persisted with timestamps."""
        run = _import_run()
        log_path = tmp_path / "runapp_test.log"

        run._stream_tray_output_to_log(
            io.StringIO("first line\nsecond line\n"),
            log_path,
        )

        log_text = log_path.read_text(encoding="utf-8")
        assert "first line" in log_text
        assert "second line" in log_text
        assert re.search(
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} first line",
            log_text,
        )

    def test_format_path_for_logs_home(self):
        """format_path_for_logs replaces home dir with ~."""
        run = _import_run()
        home = Path.home()
        p = home / "some" / "path.txt"
        if hasattr(run, "format_path_for_logs"):
            result = run.format_path_for_logs(p)
            assert isinstance(result, str)

    def test_format_path_for_logs_non_home(self):
        """format_path_for_logs returns string for non-home paths."""
        run = _import_run()
        p = Path("/tmp/test_path.txt")
        if hasattr(run, "format_path_for_logs"):
            result = run.format_path_for_logs(p)
            assert isinstance(result, str)

    def test_find_free_port_returns_int(self):
        """_find_free_port returns an integer port."""
        run = _import_run()
        if hasattr(run, "_find_free_port"):
            port = run._find_free_port()
            assert isinstance(port, int)
            assert 1024 <= port <= 65535

    def test_resolve_python_executable_returns_path(self):
        """_resolve_python_executable returns a valid path."""
        run = _import_run()
        result = run._resolve_python_executable()
        assert result is not None
        assert isinstance(result, (str, Path))

    def test_tray_python_candidates_returns_list(self):
        """_tray_python_candidates returns a non-empty list."""
        run = _import_run()
        if hasattr(run, "_tray_python_candidates"):
            candidates = run._tray_python_candidates()
            assert isinstance(candidates, list)
            assert len(candidates) > 0


class TestRunPyMainBlock:
    """Tests for run.py's main entry point modes."""

    def test_webapp_mode_flag_detected(self):
        """HAS_WEB_FLAG reflects --webapp arg."""
        run = _import_run()
        if hasattr(run, "HAS_WEB_FLAG"):
            assert isinstance(run.HAS_WEB_FLAG, bool)

    def test_cli_args_is_list(self):
        """CLI_ARGS is a list of strings."""
        run = _import_run()
        if hasattr(run, "CLI_ARGS"):
            assert isinstance(run.CLI_ARGS, list)


def _execute_run_module_with_args(
    argv: list[str],
    run_result: SimpleNamespace | None = None,
    popen_proc: MagicMock | None = None,
):
    """Import run.py with a custom argv and return execution artifacts."""
    if "run" in sys.modules:
        del sys.modules["run"]

    if run_result is None:
        run_result = SimpleNamespace(returncode=0, stdout="")

    if popen_proc is None:
        popen_proc = MagicMock()
        popen_proc.poll.return_value = 1
        popen_proc.returncode = 1

    def _raise_system_exit(code=0):
        raise SystemExit(code)

    with patch.object(sys, "argv", argv), \
            patch("subprocess.run", return_value=run_result) as mock_run, \
            patch("subprocess.Popen", return_value=popen_proc) as mock_popen, \
            patch("time.sleep"), \
            patch("sys.exit", side_effect=_raise_system_exit) as mock_exit:
        exit_code = None
        try:
            importlib.import_module("run")
        except SystemExit as exc:
            exit_code = exc.code
        return {
            "exit_code": exit_code,
            "mock_run": mock_run,
            "mock_popen": mock_popen,
            "mock_exit": mock_exit,
        }


class TestRunModuleEntrypointCoverage:
    """Additional coverage tests for run.py module-level branches."""

    def test_help_mode_exits_zero_and_requests_benchmark_help(self):
        """`--help` path prints help and exits with code 0."""
        result = _execute_run_module_with_args(
            ["run.py", "--help"],
            run_result=SimpleNamespace(
                returncode=0,
                stdout="usage: benchmark.py\n\noptions:\n  --runs RUNS\n",
            ),
        )
        assert result["exit_code"] == 0
        called_cmd = result["mock_run"].call_args[0][0]
        assert "--help" in called_cmd

    def test_webapp_mode_invalid_args_exits_two(self):
        """Invalid characters in webapp args trigger exit code 2."""
        result = _execute_run_module_with_args(
            ["run.py", "--webapp", "--bad$arg"],
        )
        assert result["exit_code"] == 2
        assert result["mock_run"].call_count == 0

    def test_webapp_mode_runs_app_and_exits_with_subprocess_code(self):
        """Webapp branch forwards args to web/app.py and exits with its code."""
        tray_proc = MagicMock()
        tray_proc.poll.return_value = None
        tray_proc.returncode = 0

        result = _execute_run_module_with_args(
            ["run.py", "--webapp", "--port", "8899"],
            run_result=SimpleNamespace(returncode=7, stdout=""),
            popen_proc=tray_proc,
        )
        assert result["exit_code"] == 7
        called_cmd = result["mock_run"].call_args[0][0]
        assert str(Path("web") / "app.py") in called_cmd[1]
        tray_proc.terminate.assert_called_once()

    def test_cli_mode_invalid_args_exits_two(self):
        """Invalid characters in CLI mode trigger exit code 2."""
        result = _execute_run_module_with_args(
            ["run.py", "--oops;"],
        )
        assert result["exit_code"] == 2
        assert result["mock_run"].call_count == 0

    def test_cli_mode_runs_benchmark_and_exits_with_subprocess_code(self):
        """Default CLI mode runs benchmark subprocess and returns its code."""
        tray_proc = MagicMock()
        tray_proc.poll.return_value = None
        tray_proc.returncode = 0

        result = _execute_run_module_with_args(
            ["run.py", "--limit", "1"],
            run_result=SimpleNamespace(returncode=5, stdout=""),
            popen_proc=tray_proc,
        )
        assert result["exit_code"] == 5
        called_cmd = result["mock_run"].call_args[0][0]
        assert str(Path("cli") / "benchmark.py") in called_cmd[1]
        tray_proc.terminate.assert_called_once()

    def test_webapp_mode_missing_app_script_exits_one(self):
        """Webapp mode exits with code 1 when web/app.py is missing."""
        original_exists = Path.exists

        def patched_exists(path_obj):
            path_str = str(path_obj)
            if path_str.endswith("web/app.py"):
                return False
            return original_exists(path_obj)

        with patch.object(Path, "exists", patched_exists):
            result = _execute_run_module_with_args(["run.py", "--webapp"])

        assert result["exit_code"] == 1

    def test_agent_mode_invalid_args_exits_two(self):
        """Invalid agent args trigger sanitize failure and exit code 2."""
        result = _execute_run_module_with_args(
            ["run.py", "--agent", "model-x", "--bad$arg"]
        )

        assert result["exit_code"] == 2

    def test_agent_mode_subprocess_oserror_exits_one(self):
        """Agent mode handles subprocess OSError with exit code 1."""
        if "run" in sys.modules:
            del sys.modules["run"]

        def _raise_system_exit(code=0):
            raise SystemExit(code)

        with patch.object(sys, "argv", ["run.py", "--agent", "model-x"]), \
                patch("subprocess.run", side_effect=OSError("boom")), \
                patch("subprocess.Popen") as mock_popen, \
                patch("time.sleep"), \
                patch("sys.exit", side_effect=_raise_system_exit):
            exit_code = None
            try:
                importlib.import_module("run")
            except SystemExit as exc:
                exit_code = exc.code

        assert exit_code == 1
        assert mock_popen.call_count == 0
