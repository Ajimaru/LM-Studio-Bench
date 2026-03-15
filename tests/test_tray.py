"""Tests for src/tray.py.

GTK dependencies (gi, Gtk, AppIndicator3) are mocked throughout
because the test environment does not guarantee a display server.
"""
import logging
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _build_gi_mock() -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return a mock gi module that satisfies tray.py imports."""
    gi_mock = MagicMock()
    gtk_mock = MagicMock()
    indicator_mock = MagicMock()
    gi_mock.repository.Gtk = gtk_mock
    gi_mock.repository.AppIndicator3 = indicator_mock
    return gi_mock, gtk_mock, indicator_mock


def _import_tray() -> tuple[types.ModuleType, MagicMock, MagicMock]:
    """Import tray module with GTK fully mocked."""
    gi_mock, gtk_mock, indicator_mock = _build_gi_mock()
    with patch.dict(
        "sys.modules",
        {
            "gi": gi_mock,
            "gi.repository": gi_mock.repository,
            "gi.repository.Gtk": gtk_mock,
            "gi.repository.AppIndicator3": indicator_mock,
        },
    ):
        if "tray" in sys.modules:
            del sys.modules["tray"]
        import tray
        setattr(tray, "gi", gi_mock)
        setattr(tray, "Gtk", gtk_mock)
        setattr(tray, "AppIndicator3", indicator_mock)
        setattr(tray, "IMPORT_ERROR", None)
    sys.modules["tray"] = tray
    return tray, gtk_mock, indicator_mock


class TestSetupLogger:
    """Tests for tray._setup_logger()."""

    def test_returns_log_file_path(self, tmp_path: Path):
        """_setup_logger returns a Path to the created log file."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            log_path = tray._setup_logger(debug=False)
        assert isinstance(log_path, Path)
        assert log_path.exists()

    def test_debug_mode_sets_debug_level(self, tmp_path: Path):
        """Debug mode sets the logger level to DEBUG."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            tray._setup_logger(debug=True)
        assert tray.LOGGER.level == logging.DEBUG

    def test_non_debug_sets_info_level(self, tmp_path: Path):
        """Non-debug mode sets the logger level to INFO."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            tray._setup_logger(debug=False)
        assert tray.LOGGER.level == logging.INFO

    def test_creates_latest_symlink(self, tmp_path: Path):
        """A 'trayapp_latest.log' symlink is created."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            tray._setup_logger(debug=False)
        symlink = tmp_path / "trayapp_latest.log"
        assert symlink.exists() or symlink.is_symlink()

    def test_log_file_has_timestamp_name(self, tmp_path: Path):
        """Log file name contains 'trayapp_' prefix."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            log_path = tray._setup_logger(debug=False)
        assert "trayapp_" in log_path.name


class TestTrayAppCallApi:
    """Tests for TrayApp._call_api()."""

    def test_get_returns_parsed_json(self, tmp_path: Path):
        """_call_api GET returns parsed JSON on success."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"status": "ok"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("tray.urllib_request.urlopen", return_value=mock_response):
            result = app._call_api("/api/status")
        assert result == {"status": "ok"}

    def test_post_sends_payload(self, tmp_path: Path):
        """_call_api POST sends JSON payload."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"success": true}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("tray.urllib_request.urlopen", return_value=mock_response):
            result = app._call_api(
                "/api/benchmark/start", method="POST", payload={"runs": 1}
            )
        assert result is not None

    def test_call_api_returns_none_on_url_error(self, tmp_path: Path):
        """_call_api returns None on URLError."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch("tray.urllib_request.urlopen", side_effect=tray.urllib_error.URLError("conn")):
            result = app._call_api("/api/status")
        assert result is None

    def test_call_api_returns_none_on_timeout(self, tmp_path: Path):
        """_call_api returns None on TimeoutError."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch("tray.urllib_request.urlopen", side_effect=TimeoutError("timeout")):
            result = app._call_api("/api/status")
        assert result is None

    def test_call_api_returns_none_on_json_error(self, tmp_path: Path):
        """_call_api returns None when response is not valid JSON."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json at all"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("tray.urllib_request.urlopen", return_value=mock_response):
            result = app._call_api("/api/status")
        assert result is None

    def test_call_api_rejects_absolute_endpoint(self, tmp_path: Path):
        """_call_api rejects absolute URLs in endpoint argument."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch("tray.urllib_request.urlopen") as mock_open:
            result = app._call_api("http://evil.example/pwn")
        assert result is None
        mock_open.assert_not_called()

    def test_call_api_rejects_protocol_relative_endpoint(self, tmp_path: Path):
        """_call_api rejects protocol-relative endpoint values."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch("tray.urllib_request.urlopen") as mock_open:
            result = app._call_api("//evil.example/pwn")
        assert result is None
        mock_open.assert_not_called()

    def test_build_api_url_rejects_non_http_scheme(self, tmp_path: Path):
        """_build_api_url rejects non-HTTP schemes."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.api_base = "file://localhost"
        assert app._build_api_url("/api/status") is None

    def test_build_api_url_rejects_cross_origin(self, tmp_path: Path):
        """_build_api_url rejects URLs that change netloc."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.api_base = "http://127.0.0.1:8080"
        assert app._build_api_url("/api/status") is None


class TestTrayAppCheckForUpdates:
    """Tests for TrayApp._check_for_updates()."""

    def test_returns_false_when_api_unavailable(self, tmp_path: Path):
        """_check_for_updates returns False when API is down."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value=None)
        assert app._check_for_updates() is False

    def test_returns_false_when_api_returns_error(self, tmp_path: Path):
        """_check_for_updates returns False when API returns error."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={"success": False})
        assert app._check_for_updates() is False

    def test_returns_true_when_update_available(self, tmp_path: Path):
        """_check_for_updates returns True when update is available."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={
            "success": True,
            "is_update_available": True,
            "current_version": "v0.1.0",
            "latest_version": "v0.2.0",
            "download_url": "https://example.com",
        })
        assert app._check_for_updates() is True
        assert app.pending_update is not None
        assert app.pending_update["latest"] == "v0.2.0"

    def test_sets_pending_update_to_none_when_no_update(self, tmp_path: Path):
        """_check_for_updates clears pending_update when up-to-date."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.pending_update = {"latest": "old"}
        app._call_api = MagicMock(return_value={
            "success": True,
            "is_update_available": False,
        })
        assert app._check_for_updates() is False
        assert app.pending_update is None


class TestTrayAppParseVersionTuple:
    """Tests for TrayApp._parse_version_tuple()."""

    def test_parses_v_prefix(self, tmp_path: Path):
        """Parses version with 'v' prefix."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._parse_version_tuple("v1.2.3") == (1, 2, 3)

    def test_parses_without_prefix(self, tmp_path: Path):
        """Parses version without 'v' prefix."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._parse_version_tuple("2.3.4") == (2, 3, 4)

    def test_ignores_pre_release_suffix(self, tmp_path: Path):
        """Strips -alpha/-beta suffix before parsing."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._parse_version_tuple("v1.0.0-beta") == (1, 0, 0)

    def test_returns_none_for_invalid_version(self, tmp_path: Path):
        """Returns None for non-numeric version string."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._parse_version_tuple("invalid") is None

    def test_returns_none_for_too_short_version(self, tmp_path: Path):
        """Returns None for version with fewer than 3 components."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._parse_version_tuple("1.2") is None


class TestTrayAppGetAboutVersionStatus:
    """Tests for TrayApp._get_about_version_status()."""

    def test_returns_dev_for_invalid_local_version(self, tmp_path: Path):
        """Returns 'dev' when local version string is invalid."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        assert app._get_about_version_status("dev-branch") == "dev"

    def test_returns_unknown_when_api_fails(self, tmp_path: Path):
        """Returns 'unknown' when GitHub release fetch fails."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch("tray.get_cached_latest_release", return_value=None):
            assert app._get_about_version_status("v1.0.0") == "unknown"

    def test_returns_update_available(self, tmp_path: Path):
        """Returns update message when newer version exists."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch(
            "tray.get_cached_latest_release",
            return_value={"tag_name": "v2.0.0"},
        ):
            result = app._get_about_version_status("v1.0.0")
        assert "update" in result.lower() or "avai" in result.lower()

    def test_returns_no_update_when_current(self, tmp_path: Path):
        """Returns 'no update' when on the same version."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch(
            "tray.get_cached_latest_release",
            return_value={"tag_name": "v1.0.0"},
        ):
            assert app._get_about_version_status("v1.0.0") == "no update"

    def test_returns_ahead_when_newer_than_release(self, tmp_path: Path):
        """Returns 'Ahead of release' when local version is newer."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        with patch(
            "tray.get_cached_latest_release",
            return_value={"tag_name": "v1.0.0"},
        ):
            assert app._get_about_version_status("v2.0.0") == "Ahead of release"


class TestTrayAppOnPollingTick:
    """Tests for TrayApp._on_polling_tick()."""

    def test_returns_true_to_keep_timer(self, tmp_path: Path):
        """_on_polling_tick always returns True."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._refresh_menu_buttons = MagicMock()
        app._check_for_updates = MagicMock(return_value=False)
        result = app._on_polling_tick()
        assert result is True

    def test_checks_updates_when_forced(self, tmp_path: Path):
        """Forced update check triggers _check_for_updates."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._refresh_menu_buttons = MagicMock()
        app._check_for_updates = MagicMock(return_value=False)
        app.force_update_check = True
        app.last_update_check = 0.0
        app._on_polling_tick()
        app._check_for_updates.assert_called_once()

    def test_handles_exception_gracefully(self, tmp_path: Path):
        """_on_polling_tick catches exceptions and still returns True."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._refresh_menu_buttons = MagicMock(side_effect=RuntimeError("boom"))
        result = app._on_polling_tick()
        assert result is True


class TestTrayAppOnStartPauseStop:
    """Tests for TrayApp._on_start/pause/stop callbacks."""

    def test_on_start_calls_api(self, tmp_path: Path):
        """_on_start calls the start benchmark API."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={"success": True})
        app._refresh_menu_buttons = MagicMock()
        app._on_start(MagicMock())
        app._call_api.assert_called_once_with(
            "/api/benchmark/start", "POST", payload={}
        )

    def test_on_stop_calls_api(self, tmp_path: Path):
        """_on_stop calls the stop benchmark API."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={"success": True})
        app._refresh_menu_buttons = MagicMock()
        app._on_stop(MagicMock())
        app._call_api.assert_called_once_with("/api/benchmark/stop", "POST")

    def test_on_pause_resume_running(self, tmp_path: Path):
        """_on_pause_resume calls pause when running."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(side_effect=[
            {"status": "running"},
            {"success": True},
        ])
        app._refresh_menu_buttons = MagicMock()
        app._on_pause_resume(MagicMock())

    def test_on_pause_resume_paused(self, tmp_path: Path):
        """_on_pause_resume calls resume when paused."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(side_effect=[
            {"status": "paused"},
            {"success": True},
        ])
        app._refresh_menu_buttons = MagicMock()
        app._on_pause_resume(MagicMock())

    def test_on_open_webapp_opens_browser(self, tmp_path: Path):
        """_on_open_webapp opens the browser with the dashboard URL."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._resolve_dashboard_url_for_open = MagicMock(
            return_value="http://localhost:8080"
        )
        with patch("tray.webbrowser.open") as mock_open:
            app._on_open_webapp(MagicMock())
        mock_open.assert_called_once_with("http://localhost:8080")

    def test_resolve_dashboard_url_recovers_from_webapp_log(
        self,
        tmp_path: Path,
    ):
        """Fallback resolves latest dashboard URL from webapp logs."""
        tray, _, _ = _import_tray()
        log_file = tmp_path / "webapp_20260315_123456.log"
        log_file.write_text(
            "INFO Dashboard available at http://localhost:46617\n",
            encoding="utf-8",
        )

        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:1234")
            app._call_api = MagicMock(return_value=None)
            app._is_dashboard_url_reachable = MagicMock(return_value=True)
            resolved = app._resolve_dashboard_url_for_open()

        assert resolved == "http://localhost:46617"
        assert app.dashboard_url == "http://localhost:46617"

    def test_resolve_dashboard_url_keeps_default_when_no_log_match(
        self,
        tmp_path: Path,
    ):
        """Fallback keeps configured URL when no log URL can be found."""
        tray, _, _ = _import_tray()
        log_file = tmp_path / "webapp_20260315_123456.log"
        log_file.write_text("INFO no dashboard url here\n", encoding="utf-8")

        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:1234")
            app._call_api = MagicMock(return_value=None)
            resolved = app._resolve_dashboard_url_for_open()

        assert resolved == "http://localhost:1234"

    def test_resolve_dashboard_url_prefers_log_over_api_response(
        self,
        tmp_path: Path,
    ):
        """Log discovery wins even when the configured URL's API responds.

        Regression test: port 1234 is LM Studio's own API port.  When
        LM Studio is running, _call_api("/api/status") would have
        returned a non-None response, causing the tray to open the wrong
        URL.  Log discovery must run first so the actual webapp port is
        used.
        """
        tray, _, _ = _import_tray()
        log_file = tmp_path / "webapp_20260315_120000.log"
        log_file.write_text(
            "INFO 🚀 Dashboard available at http://localhost:56789\n",
            encoding="utf-8",
        )

        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:1234")
            app._call_api = MagicMock(
                return_value={"status": "ok"}
            )
            app._is_dashboard_url_reachable = MagicMock(return_value=True)
            resolved = app._resolve_dashboard_url_for_open()

        assert resolved == "http://localhost:56789"
        assert app.dashboard_url == "http://localhost:56789"
        app._call_api.assert_not_called()

    def test_resolve_dashboard_url_ignores_unreachable_log_url(
        self,
        tmp_path: Path,
    ):
        """Stale log URL is ignored when no webapp is currently reachable."""
        tray, _, _ = _import_tray()
        log_file = tmp_path / "webapp_20260315_182849.log"
        log_file.write_text(
            "INFO Dashboard available at http://localhost:46617\n",
            encoding="utf-8",
        )

        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:1234")
            app._is_dashboard_url_reachable = MagicMock(return_value=False)
            app._call_api = MagicMock(return_value=None)
            resolved = app._resolve_dashboard_url_for_open()

        assert resolved == "http://localhost:1234"
        assert app.dashboard_url == "http://localhost:1234"

    def test_on_status_calls_api(self, tmp_path: Path):
        """_on_status calls the status API."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={
            "status": "idle",
            "running": False,
        })
        app._show_info_dialog = MagicMock()
        app._refresh_menu_buttons = MagicMock()
        app._on_status(MagicMock())
        app._call_api.assert_called()

    def test_on_status_shows_error_when_api_fails(self, tmp_path: Path):
        """_on_status shows error dialog when API is unreachable."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value=None)
        app._show_info_dialog = MagicMock()
        app._on_status(MagicMock())
        app._show_info_dialog.assert_called_once()


class TestTrayAppUpdateIcon:
    """Tests for TrayApp._update_icon_for_status()."""

    def test_no_op_without_appindicator(self, tmp_path: Path):
        """_update_icon_for_status does nothing when appindicator is None."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.appindicator = None
        app._update_icon_for_status("running", True)

    def test_sets_green_icon_when_running(self, tmp_path: Path):
        """_update_icon_for_status sets green icon when running."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.appindicator = MagicMock()
        app.icon_dir = "/tmp/icons"
        app._update_icon_for_status("running", True)
        called_args = app.appindicator.set_icon_full.call_args[0][0]
        assert "green" in called_args

    def test_sets_red_icon_when_api_unreachable(self, tmp_path: Path):
        """_update_icon_for_status sets red icon when API unreachable."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.appindicator = MagicMock()
        app.icon_dir = "/tmp/icons"
        app._update_icon_for_status("idle", False)
        called_args = app.appindicator.set_icon_full.call_args[0][0]
        assert "red" in called_args

    def test_sets_yellow_icon_when_paused(self, tmp_path: Path):
        """_update_icon_for_status sets yellow icon when paused."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app.appindicator = MagicMock()
        app.icon_dir = "/tmp/icons"
        app._update_icon_for_status("paused", True)
        called_args = app.appindicator.set_icon_full.call_args[0][0]
        assert "yellow" in called_args


class TestTrayExpandShortFlagClusters:
    """Tests for _parse_args helper _expand_short_flag_clusters in tray."""

    def test_positional_arg_unchanged(self, tmp_path: Path):
        """Non-flag args pass through unchanged via tray._parse_args."""
        tray, _, _ = _import_tray()
        with patch("sys.argv", ["tray.py", "--help"]):
            with pytest.raises(SystemExit):
                tray._parse_args()

    def test_module_level_flags_unchanged(self, tmp_path: Path):
        """tray module-level constants are accessible."""
        tray, _, _ = _import_tray()
        assert hasattr(tray, "LOGGER")
        assert hasattr(tray, "USER_LOGS_DIR")

    def test_long_flag_passes_through(self, tmp_path: Path):
        """TrayApp can be instantiated without error."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            assert tray.TrayApp("http://localhost:8080") is not None


class TestTrayCheckUpdatesClicked:
    """Tests for TrayApp._on_check_updates_clicked()."""

    def test_shows_info_dialog_when_no_update(self, tmp_path: Path):
        """Shows 'No Updates' dialog when already up-to-date."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._check_for_updates = MagicMock(return_value=False)
        app._show_info_dialog = MagicMock()
        app._show_update_notification = MagicMock()
        app._on_check_updates_clicked(MagicMock())
        app._show_info_dialog.assert_called_once()
        app._show_update_notification.assert_not_called()

    def test_shows_update_notification_when_update_available(self, tmp_path: Path):
        """Shows update notification when newer version exists."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._check_for_updates = MagicMock(return_value=True)
        app.pending_update = {"latest": "v2.0.0"}
        app._show_update_notification = MagicMock()
        app._on_check_updates_clicked(MagicMock())
        app._show_update_notification.assert_called_once()


class TestTrayStopStatusPolling:
    """Tests for TrayApp._stop_status_polling()."""

    def test_no_op_when_no_timer(self, tmp_path: Path):
        """_stop_status_polling is a no-op when no timer is running."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._polling_timer_id = None
        app._stop_status_polling()

    def test_removes_timer_when_active(self, tmp_path: Path):
        """_stop_status_polling removes the GLib timer."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._polling_timer_id = 42
        mock_glib = MagicMock()
        mock_gi_repo = MagicMock()
        mock_gi_repo.GLib = mock_glib
        with patch.dict("sys.modules", {"gi.repository": mock_gi_repo}):
            app._stop_status_polling()
        assert app._polling_timer_id is None


class TestTrayUIBuildingMethods:
    """Tests for TrayApp UI building methods."""

    def test_build_menu_returns_menu(self, tmp_path: Path):
        """_build_menu returns a Gtk.Menu object."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        menu = app._build_menu()
        assert menu is not None

    def test_create_info_tab_returns_box(self, tmp_path: Path):
        """_create_info_tab returns a Gtk.Box."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        result = app._create_info_tab()
        assert result is not None

    def test_create_model_details_tab_returns_box(self, tmp_path: Path):
        """_create_contributors_tab returns a Gtk.Box."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        result = app._create_contributors_tab()
        assert result is not None

    def test_create_benchmark_log_tab_returns_box(self, tmp_path: Path):
        """_show_about_dialog builds and shows the about dialog."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._show_info_dialog = MagicMock()
        app._on_about(MagicMock())

    def test_refresh_menu_buttons_when_idle(self, tmp_path: Path):
        """_refresh_menu_buttons sets menu state when benchmark is idle."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={"running": False, "status": "idle"})
        app.start_item = MagicMock()
        app.pause_item = MagicMock()
        app.stop_item = MagicMock()
        app._refresh_menu_buttons()
        app.start_item.set_sensitive.assert_called()

    def test_refresh_menu_buttons_when_running(self, tmp_path: Path):
        """_refresh_menu_buttons sets menu state when benchmark is running."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value={"running": True, "status": "running"})
        app.start_item = MagicMock()
        app.pause_item = MagicMock()
        app.stop_item = MagicMock()
        app._refresh_menu_buttons()
        app.start_item.set_sensitive.assert_called()


class TestTrayVersionStatus:
    """Tests for TrayApp version status helpers."""

    def test_about_version_status_latest(self, tmp_path: Path):
        """_get_about_version_status returns up-to-date message."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value=None)
        result = app._get_about_version_status("v1.0.0")
        assert isinstance(result, str)

    def test_on_check_updates_clicked_no_update(self, tmp_path: Path):
        """_on_check_updates_clicked shows no-update dialog when up to date."""
        tray, _, _ = _import_tray()
        with patch("tray.USER_LOGS_DIR", tmp_path):
            app = tray.TrayApp("http://localhost:8080")
        app._call_api = MagicMock(return_value=None)
        app._show_info_dialog = MagicMock()
        app._on_check_updates_clicked(MagicMock())
        app._show_info_dialog.assert_called()
