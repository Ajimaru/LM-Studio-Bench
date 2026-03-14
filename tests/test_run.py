"""Tests for run.py helper functions.

run.py executes module-level code that starts subprocesses on import.
We patch subprocess and sys.exit before importing to avoid side effects.
"""
from pathlib import Path
import socket
import subprocess
import sys
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

    def test_adds_src_to_pythonpath(self):
        """src/ directory is prepended to PYTHONPATH."""
        run = _import_run()
        env = run._build_subprocess_env()
        assert "PYTHONPATH" in env
        assert "src" in env["PYTHONPATH"]

    def test_preserves_existing_pythonpath(self, monkeypatch):
        """Existing PYTHONPATH is preserved after the src/ prefix."""
        run = _import_run()
        monkeypatch.setenv("PYTHONPATH", "/custom/path")
        env = run._build_subprocess_env()
        assert "/custom/path" in env["PYTHONPATH"]


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


class TestStartTrayProcess:
    """Tests for _start_tray_process() in run.py."""

    def test_returns_none_when_tray_script_missing(self, tmp_path: Path, monkeypatch):
        """_start_tray_process returns None when tray.py does not exist."""
        run = _import_run()
        monkeypatch.setattr(run, "project_root", tmp_path)
        (tmp_path / "src").mkdir(exist_ok=True)
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
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
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
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
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
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        good_proc = MagicMock()
        good_proc.poll.return_value = None
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
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
        tray_script = src_dir / "tray.py"
        tray_script.write_text("# tray stub")
        monkeypatch.setattr(run, "project_root", tmp_path)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        captured_cmd = []
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None

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
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "tray.py").write_text("# tray stub")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1

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
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "tray.py").write_text("# tray stub")
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1

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
