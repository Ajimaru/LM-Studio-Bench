#!/usr/bin/env python3
"""System tray app for LM Studio Benchmark controls."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import json
import logging
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Optional, cast
from urllib import error as urllib_error
from urllib.parse import urlparse
from urllib import request as urllib_request
import webbrowser

from user_paths import USER_LOGS_DIR
from version_checker import fetch_latest_release

_LATEST_RELEASE_LOCK = threading.Lock()
_LATEST_RELEASE_STATE: dict[str, Any] = {
    "data": None,
    "fetch_started": False,
}


def _fetch_latest_release_background() -> None:
    """Fetch latest release info in a background thread and cache result."""
    try:
        release_data = fetch_latest_release()
    except (
        OSError,
        RuntimeError,
        ValueError,
        urllib_error.URLError,
        json.JSONDecodeError,
    ) as exc:
        logging.getLogger(__name__).debug(
            "Background fetch_latest_release failed: %s", exc
        )
        return

    with _LATEST_RELEASE_LOCK:
        _LATEST_RELEASE_STATE["data"] = release_data


def _ensure_latest_release_fetch_started() -> None:
    """Start background fetch thread once, if not already running."""
    with _LATEST_RELEASE_LOCK:
        if bool(_LATEST_RELEASE_STATE["fetch_started"]):
            return
        _LATEST_RELEASE_STATE["fetch_started"] = True

    thread = threading.Thread(
        target=_fetch_latest_release_background,
        name="latest-release-fetch",
        daemon=True,
    )
    thread.start()


def get_cached_latest_release() -> Optional[dict[str, Any]]:
    """Return cached latest release data, starting background fetch if needed.

    The first call triggers a non-blocking background fetch and returns the
    current cache value (which may be None). Subsequent calls will return the
    fetched data once available.
    """
    _ensure_latest_release_fetch_started()
    with _LATEST_RELEASE_LOCK:
        return cast(Optional[dict[str, Any]], _LATEST_RELEASE_STATE["data"])


def _prepend_env_paths(var_name: str, paths: list[str]) -> None:
    """Prepend existing directories to colon-separated environment var."""
    valid_paths = [path for path in paths if path and Path(path).is_dir()]
    if not valid_paths:
        return

    current = os.environ.get(var_name, "")
    current_parts = [part for part in current.split(":") if part]
    merged = valid_paths + [part for part in current_parts if part not in valid_paths]
    os.environ[var_name] = ":".join(merged)


def _bootstrap_gi_runtime_paths() -> None:
    """Ensure GI typelibs and shared libs are discoverable in AppImage mode."""
    project_root = Path(__file__).resolve().parent.parent
    appdir_candidate = project_root.parents[2]

    app_lib_dir = appdir_candidate / "usr" / "lib"
    app_arch_lib_dir = app_lib_dir / "x86_64-linux-gnu"

    gi_paths = [
        str(app_arch_lib_dir / "girepository-1.0"),
        str(app_lib_dir / "girepository-1.0"),
        "/usr/lib/x86_64-linux-gnu/girepository-1.0",
        "/usr/lib/girepository-1.0",
        "/usr/lib64/girepository-1.0",
    ]

    _prepend_env_paths("GI_TYPELIB_PATH", gi_paths)


_bootstrap_gi_runtime_paths()


def _import_gi_repository(module_name: str) -> Any:
    """Import a whitelisted GI repository module by name."""
    if module_name == "Gtk":
        return importlib.import_module("gi.repository.Gtk")
    if module_name == "GLib":
        return importlib.import_module("gi.repository.GLib")
    if module_name == "AppIndicator3":
        return importlib.import_module("gi.repository.AppIndicator3")
    if module_name == "AyatanaAppIndicator3":
        return importlib.import_module("gi.repository.AyatanaAppIndicator3")
    raise ValueError(f"Unsupported GI module: {module_name}")


try:
    import gi

    gi.require_version("Gtk", "3.0")
    GTK: Any = _import_gi_repository("Gtk")

    try:
        GLIB: Any = _import_gi_repository("GLib")
    except (ImportError, ModuleNotFoundError, ValueError):
        GLIB = None

    APP_INDICATOR3: Any = None
    for appindicator_module in ("AppIndicator3", "AyatanaAppIndicator3"):
        try:
            gi.require_version(appindicator_module, "0.1")
            APP_INDICATOR3 = _import_gi_repository(appindicator_module)
            break
        except (
            ValueError,
            ImportError,
            ModuleNotFoundError,
            AttributeError,
        ):
            continue
except (ImportError, ValueError, AttributeError) as import_exc:
    gi = None
    GTK = None
    APP_INDICATOR3 = None
    GLIB = None
    IMPORT_ERROR = import_exc
else:
    IMPORT_ERROR = None


LOGGER = logging.getLogger("tray")
_TRAY_STATE: dict[str, Optional[threading.Thread]] = {"thread": None}
_WEBAPP_URL_RE = re.compile(r"Dashboard available at (http://localhost:\d+)")


def _normalize_dashboard_url(dashboard_url: str) -> str:
    """Normalize and validate the dashboard base URL.

    Only absolute same-host HTTP(S) URLs without embedded credentials are
    accepted.
    """
    normalized = dashboard_url.rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Dashboard URL must use http or https")
    if not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Dashboard URL must use a plain host[:port]")
    return normalized


def _setup_logger(debug: bool) -> Path:
    """Configure tray logger with a dedicated log file.

    Args:
        debug: Enables DEBUG level when True.

    Returns:
        Path to created log file.
    """
    logs_dir = USER_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"trayapp_{timestamp}.log"
    latest_link = logs_dir / "trayapp_latest.log"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(log_file.name)

    level = logging.DEBUG if debug else logging.INFO
    LOGGER.setLevel(level)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    LOGGER.handlers.clear()
    LOGGER.addHandler(file_handler)
    return log_file


class TrayApp:
    """Tray application that controls benchmark endpoints."""

    def __init__(self, dashboard_url: str, debug: bool = False) -> None:
        """Initialize tray app state.

        Args:
            dashboard_url: Dashboard base URL.
            debug: Enables debug logging.
        """
        self.dashboard_url = _normalize_dashboard_url(dashboard_url)
        parsed = urlparse(self.dashboard_url)
        self.api_base = f"{parsed.scheme}://{parsed.netloc}"
        self._api_scheme = parsed.scheme
        self._api_netloc = parsed.netloc
        self.debug = debug
        self.menu: Optional[Any] = None
        self.start_item: Optional[Any] = None
        self.pause_item: Optional[Any] = None
        self.stop_item: Optional[Any] = None
        self._polling_timer_id: Optional[int] = None
        self.appindicator: Optional[Any] = None
        self.status_icon: Optional[Any] = None
        self.icon_dir: Optional[Path] = None
        self.last_update_check: float = 0.0
        self.force_update_check: bool = False
        self.pending_update: Optional[dict] = None

    def _build_api_url(self, endpoint: str) -> Optional[str]:
        """Build and validate API URL for endpoint.

        Only same-origin HTTP(S) requests are allowed to reduce SSRF risk.
        """
        if not endpoint.startswith("/") or endpoint.startswith("//"):
            LOGGER.warning("Rejected endpoint format: %s", endpoint)
            return None

        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme or parsed_endpoint.netloc:
            LOGGER.warning("Rejected absolute endpoint: %s", endpoint)
            return None

        url = f"{self.api_base}{endpoint}"
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"}:
            LOGGER.warning("Rejected non-HTTP(S) scheme: %s", parsed_url.scheme)
            return None

        if (
            parsed_url.scheme != self._api_scheme
            or parsed_url.netloc != self._api_netloc
        ):
            LOGGER.warning("Rejected cross-origin API URL: %s", url)
            return None

        return url

    def _set_dashboard_url(self, dashboard_url: str) -> None:
        """Update dashboard URL and derived API base fields."""
        normalized = _normalize_dashboard_url(dashboard_url)
        parsed = urlparse(normalized)
        self.dashboard_url = normalized
        self.api_base = f"{parsed.scheme}://{parsed.netloc}"
        self._api_scheme = parsed.scheme
        self._api_netloc = parsed.netloc

    def _extract_dashboard_url_from_log(
        self,
        log_path: Path,
    ) -> Optional[str]:
        """Extract last dashboard URL from a webapp log file."""
        try:
            content = log_path.read_text(encoding="utf-8")
        except OSError:
            return None

        matches = _WEBAPP_URL_RE.findall(content)
        if not matches:
            return None

        candidate = matches[-1].strip()
        try:
            return _normalize_dashboard_url(candidate)
        except ValueError:
            return None

    def _discover_dashboard_url_from_logs(self) -> Optional[str]:
        """Find the latest dashboard URL from recent webapp logs."""
        latest_link = USER_LOGS_DIR / "webapp_latest.log"
        candidates: list[Path] = []

        if latest_link.exists():
            candidates.append(latest_link)

        try:
            recent_logs = sorted(
                USER_LOGS_DIR.glob("webapp_*.log"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            recent_logs = []

        candidates.extend(recent_logs[:5])

        seen: set[Path] = set()
        for candidate_path in candidates:
            resolved_path = candidate_path.resolve()
            if resolved_path in seen:
                continue
            seen.add(resolved_path)

            dashboard_url = self._extract_dashboard_url_from_log(resolved_path)
            if dashboard_url:
                return dashboard_url
        return None

    def _resolve_dashboard_url_for_open(self) -> str:
        """Resolve dashboard URL for browser action.

        Prefer configured URL when the API responds there. If not,
        try to recover latest active dashboard URL from log files.
        """
        if self._call_api("/api/status") is not None:
            return self.dashboard_url

        discovered_url = self._discover_dashboard_url_from_logs()
        if discovered_url:
            LOGGER.info("Recovered dashboard URL from logs: %s", discovered_url)
            self._set_dashboard_url(discovered_url)
            return discovered_url
        return self.dashboard_url

    def _call_api(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Optional[dict] = None,
    ) -> Optional[dict]:
        """Call a dashboard API endpoint.

        Args:
            endpoint: API path starting with /.
            method: HTTP method.
            payload: Optional JSON payload.

        Returns:
            Parsed JSON as dict when successful, otherwise None.
        """
        url = self._build_api_url(endpoint)
        if not url:
            return None
        try:
            LOGGER.debug("API %s %s payload=%s", method, url, payload)
            data_bytes = b""
            headers = {"Content-Type": "application/json"}
            if payload is not None:
                data_bytes = json.dumps(payload).encode("utf-8")

            req = urllib_request.Request(
                url=url,
                data=data_bytes if method != "GET" else None,
                headers=headers,
                method=method,
            )
            with urllib_request.urlopen(req, timeout=5.0) as response:  # nosec B310
                response_body = response.read().decode("utf-8")
            data = json.loads(response_body) if response_body else {}
            LOGGER.debug("API response (%s): %s", endpoint, data)
            return data
        except (
            urllib_error.URLError,
            ValueError,
            json.JSONDecodeError,
            TimeoutError,
        ) as exc:
            exc_type = type(exc).__name__
            LOGGER.warning("API %s %s failed (%s): %s", method, endpoint, exc_type, exc)
            return None
        except OSError as exc:
            LOGGER.error("Unexpected error on API %s %s: %s", method, endpoint, exc)
            return None

    def _show_info_dialog(self, title: str, message: str) -> None:
        """Show a simple GTK information dialog."""
        dialog = GTK.MessageDialog(
            parent=None,
            modal=True,
            message_type=GTK.MessageType.INFO,
            buttons=GTK.ButtonsType.OK,
            text=title,
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def _show_update_notification(self) -> None:
        """Show update notification with Download button.

        Displays a dialog with current/latest version info and allows
        user to open the GitHub Release URL to download the update.
        """
        if not self.pending_update:
            return

        current = self.pending_update.get("current", "unknown")
        latest = self.pending_update.get("latest", "unknown")
        url = self.pending_update.get("url", "")

        dialog = GTK.MessageDialog(
            parent=None,
            modal=True,
            message_type=GTK.MessageType.QUESTION,
            buttons=GTK.ButtonsType.CANCEL,
            text="🆕 Update Available",
        )
        message = (
            f"A new version of LM Studio Benchmark is available!\n\n"
            f"Current: {current}\n"
            f"Latest: {latest}"
        )
        dialog.format_secondary_text(message)

        dialog.add_button("Download", GTK.ResponseType.OK)
        _btn: Any = cast(
            Any, dialog.get_widget_for_response(GTK.ResponseType.OK)
        )
        if _btn is not None:
            _btn.set_image(
                GTK.Image.new_from_icon_name(
                    "document-save", GTK.IconSize.BUTTON
                )
            )

        response_holder: list[int] = []

        def _on_response(dlg: Any, response_id: int) -> None:
            response_holder.append(response_id)
            dlg.destroy()

        dialog.connect("response", _on_response)
        dialog.run()

        if response_holder and response_holder[0] == GTK.ResponseType.OK and url:
            LOGGER.info("Opening download URL: %s", url)
            webbrowser.open(url)

    def _update_icon_for_status(self, status: str, api_reachable: bool) -> None:
        """Update tray icon based on benchmark status.

        Args:
            status: Benchmark status (running/paused/idle/etc).
            api_reachable: Whether API call succeeded.
        """
        if not self.icon_dir:
            return

        if not api_reachable:
            color = "red"
        elif status == "running":
            color = "green"
        elif status == "paused":
            color = "yellow"
        else:
            color = "gray"

        icon_name = f"lmstudio-bench-tray-{color}"
        if self.appindicator is not None:
            self.appindicator.set_icon_full(icon_name, "")
        elif self.status_icon is not None:
            self.status_icon.set_from_icon_name(icon_name)
        LOGGER.debug("Updated tray icon: %s (status=%s)", color, status)

    def _on_status_icon_popup(
        self,
        _icon: Any,
        button: int,
        activate_time: int,
    ) -> None:
        """Show context menu for Gtk.StatusIcon fallback."""
        if self.menu is None:
            return

        self.menu.popup(
            None,
            None,
            GTK.StatusIcon.position_menu,
            self.status_icon,
            button,
            activate_time,
        )

    def _refresh_menu_buttons(self) -> None:
        """Refresh menu button states based on benchmark status.

        Button states:
        - Idle/Unknown/Error: Start enabled, Pause disabled, Stop disabled
        - Running: Start disabled, Pause enabled, Stop enabled
        - Paused: Start disabled, Resume enabled, Stop enabled

        Design: Start acts as fallback recovery button when benchmark
        is in error state or API is unreachable.
        """
        status_data = self._call_api("/api/status")
        api_reachable = status_data is not None

        if not api_reachable:
            status = "idle"
            LOGGER.warning("API unreachable, treating as idle state")
        else:
            status = str(status_data.get("status", "idle")).lower()

        self._update_icon_for_status(status, api_reachable)

        LOGGER.debug("Updating menu buttons: status=%s, data=%s", status, status_data)

        is_running_or_paused = status in ("running", "paused")
        if self.start_item:
            self.start_item.set_sensitive(not is_running_or_paused)

        if self.pause_item:
            if status == "running":
                self.pause_item.set_label("Pause")
                self.pause_item.set_sensitive(True)
            elif status == "paused":
                self.pause_item.set_label("Resume")
                self.pause_item.set_sensitive(True)
            else:
                self.pause_item.set_label("Pause")
                self.pause_item.set_sensitive(False)

        if self.stop_item:
            self.stop_item.set_sensitive(is_running_or_paused)

    def _on_start(self, _item: Any) -> None:
        """Start benchmark from tray."""
        result = self._call_api("/api/benchmark/start", "POST", payload={})
        if result:
            LOGGER.info("Start action: %s", result.get("message", "ok"))
        self._refresh_menu_buttons()

    def _on_pause_resume(self, _item: Any) -> None:
        """Pause or resume benchmark based on current status."""
        status_data = self._call_api("/api/status")
        if not status_data:
            self._refresh_menu_buttons()
            return

        status = str(status_data.get("status", "")).lower()
        if status == "running":
            result = self._call_api("/api/benchmark/pause", "POST")
            if result:
                LOGGER.info("Pause action: %s", result.get("message", "ok"))
        elif status == "paused":
            result = self._call_api("/api/benchmark/resume", "POST")
            if result:
                LOGGER.info("Resume action: %s", result.get("message", "ok"))
        self._refresh_menu_buttons()

    def _on_stop(self, _item: Any) -> None:
        """Stop benchmark from tray."""
        result = self._call_api("/api/benchmark/stop", "POST")
        if result:
            LOGGER.info("Stop action: %s", result.get("message", "ok"))
        self._refresh_menu_buttons()

    def _on_status(self, _item: Any) -> None:
        """Show current benchmark status."""
        status_data = self._call_api("/api/status")
        if not status_data:
            self._show_info_dialog(
                "Benchmark Status",
                "Status could not be retrieved.",
            )
            return

        status = status_data.get("status", "unknown")
        running = status_data.get("running", False)
        clients = status_data.get("connected_clients", 0)
        message = (
            f"Status: {status}\n"
            f"Running: {running}\n"
            f"Connected Clients: {clients}"
        )
        self._show_info_dialog("Benchmark Status", message)
        self._refresh_menu_buttons()

    def _check_for_updates(self) -> bool:
        """Check for new version and store update info if available.

        Queries /api/system/latest-release endpoint to check for
        updates. If new version is available, stores info in
        self.pending_update and returns True.

        Returns:
            True if update is available, False otherwise.
        """
        update_data = self._call_api("/api/system/latest-release")
        if not update_data:
            LOGGER.debug("Update check failed (API error)")
            return False

        if not update_data.get("success"):
            LOGGER.debug("Update check returned error: %s", update_data.get("message"))
            return False

        is_update_available = update_data.get("is_update_available", False)
        current = update_data.get("current_version", "unknown")
        latest = update_data.get("latest_version", "unknown")
        url = update_data.get("download_url", "")

        LOGGER.info(
            "Update check: current=%s, latest=%s, available=%s",
            current,
            latest,
            is_update_available,
        )

        if is_update_available:
            self.pending_update = {
                "current": current,
                "latest": latest,
                "url": url,
            }
            LOGGER.info("🆕 Update available: %s → %s", current, latest)
            return True

        self.pending_update = None
        LOGGER.debug("No update available")
        return False

    def _on_polling_tick(self) -> bool:
        """Periodic status update callback for GLIB timeout.

        This ensures button states are refreshed every few seconds,
        allowing recovery when benchmark crashes or API becomes
        unavailable and then recovers. Also checks for updates every
        24 hours (or when forced).

        Returns:
            True to keep the timer running, False to stop it.
        """
        try:
            self._refresh_menu_buttons()

            now = time.time()
            time_since_last_check = now - self.last_update_check
            check_interval = 24 * 3600

            if self.force_update_check or time_since_last_check >= check_interval:
                self.last_update_check = now
                self.force_update_check = False
                update_available = self._check_for_updates()
                if update_available and self.pending_update:
                    latest = self.pending_update.get("latest", "?")
                    LOGGER.info(
                        "Update available: %s (use 'Check for Updates')",
                        latest,
                    )
        except (RuntimeError, OSError, AttributeError):
            LOGGER.exception("Error during status polling")
        return True

    def _start_status_polling(self) -> None:
        """Start periodic status polling with 3-second interval.

        This allows the tray to detect status changes without waiting
        for user interaction, and helps recover from benchmark crashes.
        """
        if self._polling_timer_id is not None:
            return

        if GLIB is None:
            LOGGER.warning("GLIB not available, status polling disabled")
            return

        self._polling_timer_id = int(GLIB.timeout_add(3000, self._on_polling_tick))
        LOGGER.debug("Started status polling (interval: 3s)")

    def _on_open_webapp(self, _item: Any) -> None:
        """Open dashboard URL in default browser."""
        dashboard_url = self._resolve_dashboard_url_for_open()
        LOGGER.info("Opening webapp: %s", dashboard_url)
        webbrowser.open(dashboard_url)

    def _on_check_updates_clicked(self, _item: Any) -> None:
        """Handle "Check for Updates" menu item click.

        Forces an immediate update check regardless of the 24h interval.
        Shows update notification if a new version is available.
        """
        LOGGER.info("Manual update check triggered")
        self.force_update_check = True
        self.last_update_check = 0.0
        update_available = self._check_for_updates()

        if update_available and self.pending_update:
            self._show_update_notification()
        else:
            self._show_info_dialog(
                "No Updates Available",
                "You are running the latest version.",
            )

    def _parse_version_tuple(self, version: str) -> Optional[tuple]:
        """Parse semantic version string to a comparable tuple.

        Accepts versions like v1.2.3 or 1.2.3 and ignores suffixes
        like "-beta". Returns None when format is not parseable.

        Args:
            version: Version string to parse.

        Returns:
            Tuple (major, minor, patch) or None.
        """
        cleaned = version.strip()
        if cleaned.startswith("v"):
            cleaned = cleaned[1:]
        cleaned = cleaned.split("-")[0]

        parts = cleaned.split(".")
        if len(parts) < 3:
            return None

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
            return (major, minor, patch)
        except ValueError:
            return None

    def _get_about_version_status(self, local_version: str) -> str:
        """Resolve version status text for About tab.

        Mapping:
        - local version invalid -> dev
        - GitHub version invalid/unavailable -> unknown
        - local == GitHub -> no update
        - local > GitHub -> Ahead of release
        - GitHub > local -> update available

        Args:
            local_version: Local VERSION file value.

        Returns:
            Human-readable status text.
        """
        local_tuple = self._parse_version_tuple(local_version)
        if local_tuple is None:
            return "dev"

        release_data = get_cached_latest_release()
        if not release_data:
            return "unknown"

        github_version = str(release_data.get("tag_name", "")).strip()
        github_tuple = self._parse_version_tuple(github_version)
        if github_tuple is None:
            return "unknown"

        if github_tuple > local_tuple:
            return "update available"
        if local_tuple > github_tuple:
            return "Ahead of release"
        return "no update"

    def _create_info_tab(self) -> Any:
        """Create the Info tab with icon, title, version, etc."""
        box = GTK.Box(orientation=GTK.Orientation.VERTICAL, spacing=15)
        box.set_margin_top(30)
        box.set_margin_bottom(30)
        box.set_margin_start(40)
        box.set_margin_end(40)

        project_root = Path(__file__).resolve().parent.parent

        icon_path = project_root / "assets" / "icons" / "lmstudio-bench.svg"
        if icon_path.exists():
            image = GTK.Image()
            image.set_from_file(str(icon_path))
            image.set_pixel_size(120)
            box.pack_start(image, False, False, 5)

        title_label = GTK.Label()
        title_label.set_markup(
            '<span size="large" weight="bold">LM Studio Model Benchmark</span>'
        )
        title_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(title_label, False, False, 0)

        version_file = project_root / "VERSION"
        version = "v0.1.0"
        if version_file.exists():
            version = version_file.read_text().strip()
        version_status = self._get_about_version_status(version)
        version_label = GTK.Label()
        version_label.set_markup(
            f'<span foreground="#888888">'
            f"{version} ({version_status})"
            f"</span>"
        )
        version_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(version_label, False, False, 5)

        desc_label = GTK.Label()
        desc_label.set_text(
            "Automatic benchmarking tool for all locally installed "
            "LM Studio models. Systematically tests different models "
            "and quantizations to measure and compare "
            "tokens-per-second performance."
        )
        desc_label.set_line_wrap(True)
        desc_label.set_max_width_chars(50)
        desc_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(desc_label, False, False, 10)

        links_box = GTK.Box(orientation=GTK.Orientation.VERTICAL, spacing=5)

        doc_link = GTK.LinkButton(
            uri="https://ajimaru.github.io/LM-Studio-Bench",
            label="Documentation",
        )
        doc_link.set_halign(GTK.Align.CENTER)
        links_box.pack_start(doc_link, False, False, 0)

        github_link = GTK.LinkButton(
            uri="https://github.com/Ajimaru/LM-Studio-Bench",
            label="GitHub Repository",
        )
        github_link.set_halign(GTK.Align.CENTER)
        links_box.pack_start(github_link, False, False, 0)

        box.pack_start(links_box, False, False, 0)

        copyright_label = GTK.Label()
        copyright_label.set_markup(
            '<span foreground="#888888">© 2026 Ajimaru</span>'
        )
        copyright_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(copyright_label, False, False, 5)

        box.show_all()
        return box

    def _create_contributors_tab(self) -> Any:
        """Create the Contributors tab with list of contributors."""
        box = GTK.Box(orientation=GTK.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(30)
        box.set_margin_bottom(30)
        box.set_margin_start(40)
        box.set_margin_end(40)

        project_root = Path(__file__).resolve().parent.parent

        header_label = GTK.Label()
        header_label.set_markup(
            '<span size="large" weight="bold">Contributors</span>'
        )
        header_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(header_label, False, False, 10)

        ajimaru_label = GTK.Label()
        ajimaru_label.set_markup(
            '<b>Ajimaru</b> (<a href="https://github.com/Ajimaru">'
            "@Ajimaru</a>)\n"
            '<span foreground="#888888">Project creator and maintainer</span>'
        )
        ajimaru_label.set_line_wrap(True)
        ajimaru_label.set_justify(GTK.Justification.CENTER)
        box.pack_start(ajimaru_label, False, False, 5)

        authors_file = project_root / "AUTHORS"
        if authors_file.exists():
            content = authors_file.read_text()
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("<!-- Add")
                    or line == "---"
                    or line.startswith("This file lists")
                    or line.startswith("## ")
                    or line == "If you contribute"
                ):
                    continue
                if "Ajimaru" in line and "@Ajimaru" in line:
                    continue
                if line.startswith("- "):
                    contrib = line[2:].strip()
                    if "(@" in contrib:
                        name, handle_part = contrib.split(" ")
                        handle = handle_part.replace("(@", "").replace(")", "").strip()
                        contrib_label = GTK.Label()
                        contrib_label.set_markup(
                            f'{name} (<a href="'
                            f'https://github.com/{handle}">'
                            f"@{handle}</a>)"
                        )
                        contrib_label.set_line_wrap(True)
                        contrib_label.set_justify(GTK.Justification.CENTER)
                        box.pack_start(contrib_label, False, False, 5)
                    else:
                        contrib_label = GTK.Label(label=contrib)
                        contrib_label.set_line_wrap(True)
                        contrib_label.set_justify(GTK.Justification.CENTER)
                        box.pack_start(contrib_label, False, False, 5)

        box.show_all()
        return box

    def _on_about(self, _item: Any) -> None:
        """Show about dialog with two tabs (Info, Contributors)."""
        dialog = GTK.Dialog(
            title="About LM Studio Benchmark",
            parent=None,
            modal=True,
            type=GTK.WindowType.TOPLEVEL,
        )
        dialog.set_default_size(550, 520)
        dialog.set_resizable(False)
        dialog.set_border_width(0)

        notebook = GTK.Notebook()

        info_box = self._create_info_tab()
        info_label = GTK.Label(label="Info")
        info_label.show()
        notebook.append_page(info_box, info_label)

        contributors_box = self._create_contributors_tab()
        contributors_label = GTK.Label(label="Contributors")
        contributors_label.show()
        notebook.append_page(contributors_box, contributors_label)

        content_area: Any = dialog.get_content_area()
        if content_area is not None:
            content_area.add(notebook)

        dialog.add_button("OK", GTK.ResponseType.OK)

        notebook.show_all()

        dialog.run()
        dialog.destroy()

    def _stop_status_polling(self) -> None:
        """Stop any active status polling timer.

        Cleanup method called before quitting.
        """
        if self._polling_timer_id is not None:
            try:
                if GLIB is not None:
                    GLIB.source_remove(self._polling_timer_id)
                self._polling_timer_id = None
                LOGGER.debug("Stopped status polling")
            except (RuntimeError, OSError) as exc:
                LOGGER.warning("Failed to stop polling timer: %s", exc)

    def _on_quit(self, _item: Any) -> None:
        """Stop benchmark and shutdown web dashboard.

        This cleanup handler ensures the benchmark is stopped and the
        web dashboard is shut down gracefully before the tray app exits,
        preventing orphaned processes.
        """
        LOGGER.info("Quit action triggered, initiating shutdown...")

        self._stop_status_polling()

        try:
            LOGGER.info("Calling WebApp shutdown endpoint...")
            result = self._call_api("/api/system/shutdown", "POST")
            if result:
                LOGGER.info("Shutdown response: %s", result.get("message", "ok"))
            else:
                LOGGER.warning("Shutdown endpoint failed, trying stop only...")
                status_data = self._call_api("/api/status")
                if status_data:
                    status = str(status_data.get("status", "")).lower()
                    if status in ("running", "paused"):
                        LOGGER.info("Stopping benchmark...")
                        self._call_api("/api/benchmark/stop", "POST")
        except (OSError, RuntimeError, AttributeError) as exc:
            LOGGER.warning("Error during shutdown: %s", exc)

        LOGGER.info("Benchmark Tray exiting")
        if self.status_icon is not None:
            self.status_icon.set_visible(False)
        if GTK is not None:
            levels = GTK.main_level()
            for _ in range(max(levels, 1)):
                GTK.main_quit()

    def _build_menu(self) -> Any:
        """Build tray menu with benchmark actions."""
        menu = GTK.Menu()

        heading = GTK.MenuItem(label="Benchmark")
        heading.set_sensitive(False)
        menu.append(heading)

        start_item = GTK.MenuItem(label="Start")
        if start_item is None:
            raise RuntimeError("Failed to create Start menu item")
        self.start_item = start_item
        start_item.connect("activate", self._on_start)
        menu.append(start_item)

        pause_item = GTK.MenuItem(label="Pause")
        if pause_item is None:
            raise RuntimeError("Failed to create Pause menu item")
        self.pause_item = pause_item
        pause_item.connect("activate", self._on_pause_resume)
        menu.append(pause_item)

        stop_item = GTK.MenuItem(label="Stop")
        if stop_item is None:
            raise RuntimeError("Failed to create Stop menu item")
        self.stop_item = stop_item
        stop_item.connect("activate", self._on_stop)
        menu.append(stop_item)

        menu.append(GTK.SeparatorMenuItem())

        status_item = GTK.MenuItem(label="Show Status")
        status_item.connect("activate", self._on_status)
        menu.append(status_item)

        open_item = GTK.MenuItem(label="Go to WebApp")
        open_item.connect("activate", self._on_open_webapp)
        menu.append(open_item)

        check_updates_item = GTK.MenuItem(label="Check for Updates")
        check_updates_item.connect("activate", self._on_check_updates_clicked)
        menu.append(check_updates_item)

        menu.append(GTK.SeparatorMenuItem())

        about_item = GTK.MenuItem(label="About")
        about_item.connect("activate", self._on_about)
        menu.append(about_item)

        menu.append(GTK.SeparatorMenuItem())

        quit_item = GTK.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        return menu

    def run(self) -> None:
        """Initialize tray icon and run GTK loop."""
        project_root = Path(__file__).resolve().parent.parent
        self.icon_dir = project_root / "assets" / "icons"

        self.menu = self._build_menu()

        if APP_INDICATOR3 is not None:
            indicator: Any = APP_INDICATOR3.Indicator.new(
                "lm-studio-benchmark",
                "lmstudio-bench-tray-gray",
                APP_INDICATOR3.IndicatorCategory.APPLICATION_STATUS,
            )
            if indicator is None:
                raise RuntimeError("Failed to create AppIndicator3 indicator")
            appindicator: Any = cast(Any, indicator)
            self.appindicator = appindicator
            appindicator.set_icon_theme_path(str(self.icon_dir))
            appindicator.set_status(APP_INDICATOR3.IndicatorStatus.ACTIVE)
            appindicator.set_menu(self.menu)
            LOGGER.info(
                "AppIndicator tray started for dashboard: %s",
                self.dashboard_url,
            )
        else:
            LOGGER.warning(
                "AppIndicator bindings unavailable; using Gtk.StatusIcon "
                "fallback"
            )
            status_icon: Any = GTK.StatusIcon.new_from_icon_name(
                "lmstudio-bench-tray-gray"
            )
            if status_icon is None:
                raise RuntimeError("Failed to create Gtk.StatusIcon fallback")
            si: Any = cast(Any, status_icon)
            self.status_icon = si
            si.set_visible(True)
            si.set_tooltip_text("LM Studio Benchmark")
            si.connect("popup-menu", self._on_status_icon_popup)

        self._refresh_menu_buttons()

        self._start_status_polling()

        GTK.main()


def _run_tray(dashboard_url: str, debug: bool) -> None:
    """Create and run tray app instance."""
    try:
        app = TrayApp(dashboard_url=dashboard_url, debug=debug)
        app.run()
    except (RuntimeError, OSError, ImportError):
        LOGGER.exception("Tray thread crashed")


def start_tray(dashboard_url: str, debug: bool = False) -> bool:
    """Start tray app in a daemon thread.

    Args:
        dashboard_url: Dashboard URL used for API and browser action.
        debug: Enables debug logging for tray app.

    Returns:
        True when startup was initiated, otherwise False.
    """
    log_file = _setup_logger(debug=debug)
    LOGGER.info("Tray log file: %s", log_file)

    if IMPORT_ERROR is not None:
        LOGGER.warning("Tray dependencies unavailable: %s", IMPORT_ERROR)
        return False

    current_thread = _TRAY_STATE["thread"]
    if current_thread and current_thread.is_alive():
        LOGGER.info("Tray already running")
        return True

    tray_thread = threading.Thread(
        target=_run_tray,
        args=(dashboard_url, debug),
        daemon=True,
    )
    _TRAY_STATE["thread"] = tray_thread
    tray_thread.start()
    return True


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone tray execution."""

    def _expand_short_flag_clusters(cli_args: list[str]) -> list[str]:
        """Expand combined short flags like ``-dh`` to ``-d -h``."""
        combinable = {"d", "h"}
        normalized: list[str] = []

        for arg in cli_args:
            if arg.startswith("--") or not arg.startswith("-"):
                normalized.append(arg)
                continue
            if len(arg) <= 2:
                normalized.append(arg)
                continue

            cluster = arg[1:]
            if all(flag in combinable for flag in cluster):
                normalized.extend(f"-{flag}" for flag in cluster)
            else:
                normalized.append(arg)
        return normalized

    parser = argparse.ArgumentParser(description="LM Studio Benchmark Tray")
    parser.add_argument(
        "--url",
        default="http://localhost:1234",
        help="Dashboard URL for API calls and web navigation",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable debug logging for tray app",
    )
    normalized_args = _expand_short_flag_clusters(sys.argv[1:])
    return parser.parse_args(args=normalized_args)


def main() -> int:
    """Run tray app as standalone process."""
    args = _parse_args()
    log_file = _setup_logger(debug=args.debug)
    LOGGER.info("Tray standalone start, log: %s", log_file)

    if IMPORT_ERROR is not None:
        msg = f"Cannot start tray, missing dependencies: {IMPORT_ERROR}"
        LOGGER.error(msg)
        return 1

    try:
        GTK.init(sys.argv)
    except RuntimeError:
        LOGGER.error("No graphical display available for GTK tray")
        return 1

    try:
        app = TrayApp(dashboard_url=args.url, debug=args.debug)
        app.run()
    except (RuntimeError, OSError, ImportError):
        LOGGER.exception("Tray standalone startup failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
