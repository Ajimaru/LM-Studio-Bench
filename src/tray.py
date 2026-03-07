#!/usr/bin/env python3
"""System tray app for LM Studio Benchmark controls."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from typing import Optional
from urllib import error as urllib_error
from urllib.parse import urlparse
from urllib import request as urllib_request
import webbrowser

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("AppIndicator3", "0.1")
    from gi.repository import AppIndicator3, Gtk
except Exception as import_exc:
    gi = None
    Gtk = None
    AppIndicator3 = None
    IMPORT_ERROR = import_exc
else:
    IMPORT_ERROR = None


LOGGER = logging.getLogger("tray")
TRAY_THREAD: Optional[threading.Thread] = None


def _setup_logger(debug: bool) -> Path:
    """Configure tray logger with a dedicated log file.

    Args:
        debug: Enables DEBUG level when True.

    Returns:
        Path to created log file.
    """
    project_root = Path(__file__).resolve().parent.parent
    logs_dir = project_root / "logs"
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
        self.dashboard_url = dashboard_url.rstrip("/")
        parsed = urlparse(self.dashboard_url)
        self.api_base = f"{parsed.scheme}://{parsed.netloc}"
        self.debug = debug
        self.menu: Optional[Gtk.Menu] = None
        self.start_item: Optional[Gtk.MenuItem] = None
        self.pause_item: Optional[Gtk.MenuItem] = None
        self.stop_item: Optional[Gtk.MenuItem] = None
        self._polling_timer_id: Optional[int] = None
        self.appindicator: Optional[AppIndicator3.Indicator] = None
        self.icon_dir: Optional[Path] = None

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
        url = f"{self.api_base}{endpoint}"
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
            with urllib_request.urlopen(req, timeout=5.0) as response:
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
            LOGGER.warning(
                "API %s %s failed (%s): %s", method, endpoint, exc_type, exc
            )
            return None
        except Exception as exc:
            LOGGER.error(
                "Unexpected error on API %s %s: %s", method, endpoint, exc
            )
            return None

    def _show_info_dialog(self, title: str, message: str) -> None:
        """Show a simple GTK information dialog."""
        dialog = Gtk.MessageDialog(
            parent=None,
            modal=True,
            message_type=Gtk.MessageType.INFO,
            buttons=Gtk.ButtonsType.OK,
            text=title,
        )
        dialog.format_secondary_text(message)
        dialog.run()
        dialog.destroy()

    def _update_icon_for_status(
        self, status: str, api_reachable: bool
    ) -> None:
        """Update tray icon based on benchmark status.

        Args:
            status: Benchmark status (running/paused/idle/etc).
            api_reachable: Whether API call succeeded.
        """
        if not self.appindicator or not self.icon_dir:
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
        self.appindicator.set_icon_full(icon_name, "")
        LOGGER.debug("Updated tray icon: %s (status=%s)", color, status)

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

        LOGGER.debug(
            "Updating menu buttons: status=%s, data=%s", status, status_data
        )

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

    def _on_start(self, _item: Gtk.MenuItem) -> None:
        """Start benchmark from tray."""
        result = self._call_api("/api/benchmark/start", "POST", payload={})
        if result:
            LOGGER.info("Start action: %s", result.get("message", "ok"))
        self._refresh_menu_buttons()

    def _on_pause_resume(self, _item: Gtk.MenuItem) -> None:
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

    def _on_stop(self, _item: Gtk.MenuItem) -> None:
        """Stop benchmark from tray."""
        result = self._call_api("/api/benchmark/stop", "POST")
        if result:
            LOGGER.info("Stop action: %s", result.get("message", "ok"))
        self._refresh_menu_buttons()

    def _on_status(self, _item: Gtk.MenuItem) -> None:
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

    def _on_polling_tick(self) -> bool:
        """Periodic status update callback for GLib timeout.

        This ensures button states are refreshed every few seconds,
        allowing recovery when benchmark crashes or API becomes
        unavailable and then recovers.

        Returns:
            True to keep the timer running, False to stop it.
        """
        try:
            self._refresh_menu_buttons()
        except Exception:
            LOGGER.exception("Error during status polling")
        return True

    def _start_status_polling(self) -> None:
        """Start periodic status polling with 3-second interval.

        This allows the tray to detect status changes without waiting
        for user interaction, and helps recover from benchmark crashes.
        """
        if self._polling_timer_id is not None:
            return

        try:
            from gi.repository import GLib
        except ImportError:
            LOGGER.warning("GLib not available, status polling disabled")
            return

        self._polling_timer_id = GLib.timeout_add(
            3000, self._on_polling_tick
        )
        LOGGER.debug("Started status polling (interval: 3s)")

    def _on_open_webapp(self, _item: Gtk.MenuItem) -> None:
        """Open dashboard URL in default browser."""
        LOGGER.info("Opening webapp: %s", self.dashboard_url)
        webbrowser.open(self.dashboard_url)

    def _create_info_tab(self) -> Gtk.Box:
        """Create the Info tab with icon, title, version, etc."""
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=15
        )
        box.set_margin_top(30)
        box.set_margin_bottom(30)
        box.set_margin_start(40)
        box.set_margin_end(40)

        project_root = Path(__file__).resolve().parent.parent

        icon_path = (
            project_root / "assets" / "icons" / "lmstudio-bench.svg"
        )
        if icon_path.exists():
            image = Gtk.Image()
            image.set_from_file(str(icon_path))
            image.set_pixel_size(120)
            box.pack_start(image, False, False, 5)

        title_label = Gtk.Label()
        title_label.set_markup(
            '<span size="large" weight="bold">'
            'LM Studio Model Benchmark</span>'
        )
        title_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(title_label, False, False, 0)

        version_file = project_root / "VERSION"
        version = "v0.1.0"
        if version_file.exists():
            version = version_file.read_text().strip()
        version_label = Gtk.Label()
        version_label.set_markup(
            f'<span foreground="#888888">{version}</span>'
        )
        version_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(version_label, False, False, 5)

        desc_label = Gtk.Label()
        desc_label.set_text(
            "Automatic benchmarking tool for all locally installed "
            "LM Studio models. Systematically tests different models "
            "and quantizations to measure and compare "
            "tokens-per-second performance."
        )
        desc_label.set_line_wrap(True)
        desc_label.set_max_width_chars(50)
        desc_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(desc_label, False, False, 10)

        links_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=5
        )

        doc_link = Gtk.LinkButton(
            uri="https://ajimaru.github.io/LM-Studio-Bench",
            label="Documentation",
        )
        doc_link.set_halign(Gtk.Align.CENTER)
        links_box.pack_start(doc_link, False, False, 0)

        github_link = Gtk.LinkButton(
            uri="https://github.com/Ajimaru/LM-Studio-Bench",
            label="GitHub Repository",
        )
        github_link.set_halign(Gtk.Align.CENTER)
        links_box.pack_start(github_link, False, False, 0)

        box.pack_start(links_box, False, False, 0)

        copyright_label = Gtk.Label()
        copyright_label.set_markup(
            '<span foreground="#888888">© 2026 Ajimaru</span>'
        )
        copyright_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(copyright_label, False, False, 5)

        box.show_all()
        return box

    def _create_contributors_tab(self) -> Gtk.Box:
        """Create the Contributors tab with list of contributors."""
        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12
        )
        box.set_margin_top(30)
        box.set_margin_bottom(30)
        box.set_margin_start(40)
        box.set_margin_end(40)

        project_root = Path(__file__).resolve().parent.parent

        header_label = Gtk.Label()
        header_label.set_markup(
            '<span size="large" weight="bold">Contributors</span>'
        )
        header_label.set_justify(Gtk.Justification.CENTER)
        box.pack_start(header_label, False, False, 10)

        ajimaru_label = Gtk.Label()
        ajimaru_label.set_markup(
            '<b>Ajimaru</b> (<a href="https://github.com/Ajimaru">'
            '@Ajimaru</a>)\n'
            '<span foreground="#888888">Project creator and maintainer</span>'
        )
        ajimaru_label.set_line_wrap(True)
        ajimaru_label.set_justify(Gtk.Justification.CENTER)
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
                        handle = (
                            handle_part.replace("(@", "")
                            .replace(")", "")
                            .strip()
                        )
                        contrib_label = Gtk.Label()
                        contrib_label.set_markup(
                            f'{name} (<a href="'
                            f'https://github.com/{handle}">'
                            f'@{handle}</a>)'
                        )
                        contrib_label.set_line_wrap(True)
                        contrib_label.set_justify(Gtk.Justification.CENTER)
                        box.pack_start(contrib_label, False, False, 5)
                    else:
                        contrib_label = Gtk.Label(label=contrib)
                        contrib_label.set_line_wrap(True)
                        contrib_label.set_justify(Gtk.Justification.CENTER)
                        box.pack_start(contrib_label, False, False, 5)

        box.show_all()
        return box

    def _on_about(self, _item: Gtk.MenuItem) -> None:
        """Show about dialog with two tabs (Info, Contributors)."""
        dialog = Gtk.Dialog(
            title="About LM Studio Benchmark",
            parent=None,
            modal=True,
            type=Gtk.WindowType.TOPLEVEL,
        )
        dialog.set_default_size(550, 520)
        dialog.set_resizable(False)
        dialog.set_border_width(0)

        notebook = Gtk.Notebook()

        info_box = self._create_info_tab()
        info_label = Gtk.Label(label="Info")
        info_label.show()
        notebook.append_page(info_box, info_label)

        contributors_box = self._create_contributors_tab()
        contributors_label = Gtk.Label(label="Contributors")
        contributors_label.show()
        notebook.append_page(contributors_box, contributors_label)

        content_area = dialog.get_content_area()
        content_area.add(notebook)

        dialog.add_button("OK", Gtk.ResponseType.OK)

        notebook.show_all()

        dialog.run()
        dialog.destroy()

    def _stop_status_polling(self) -> None:
        """Stop any active status polling timer.

        Cleanup method called before quitting.
        """
        if self._polling_timer_id is not None:
            try:
                from gi.repository import GLib

                GLib.source_remove(self._polling_timer_id)
                self._polling_timer_id = None
                LOGGER.debug("Stopped status polling")
            except Exception as exc:
                LOGGER.warning("Failed to stop polling timer: %s", exc)

    def _on_quit(self, _item: Gtk.MenuItem) -> None:
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
                LOGGER.info(
                    "Shutdown response: %s", result.get("message", "ok")
                )
            else:
                LOGGER.warning(
                    "Shutdown endpoint failed, trying stop only..."
                )
                status_data = self._call_api("/api/status")
                if status_data:
                    status = str(status_data.get("status", "")).lower()
                    if status in ("running", "paused"):
                        LOGGER.info("Stopping benchmark...")
                        self._call_api("/api/benchmark/stop", "POST")
        except Exception as exc:
            LOGGER.warning("Error during shutdown: %s", exc)

        LOGGER.info("Benchmark Tray exiting")
        Gtk.main_quit()

    def _build_menu(self) -> Gtk.Menu:
        """Build tray menu with benchmark actions."""
        menu = Gtk.Menu()

        heading = Gtk.MenuItem(label="Benchmark")
        heading.set_sensitive(False)
        menu.append(heading)

        self.start_item = Gtk.MenuItem(label="Start")
        self.start_item.connect("activate", self._on_start)
        menu.append(self.start_item)

        self.pause_item = Gtk.MenuItem(label="Pause")
        self.pause_item.connect("activate", self._on_pause_resume)
        menu.append(self.pause_item)

        self.stop_item = Gtk.MenuItem(label="Stop")
        self.stop_item.connect("activate", self._on_stop)
        menu.append(self.stop_item)

        menu.append(Gtk.SeparatorMenuItem())

        status_item = Gtk.MenuItem(label="Show Status")
        status_item.connect("activate", self._on_status)
        menu.append(status_item)

        open_item = Gtk.MenuItem(label="Go to WebApp")
        open_item.connect("activate", self._on_open_webapp)
        menu.append(open_item)

        menu.append(Gtk.SeparatorMenuItem())

        about_item = Gtk.MenuItem(label="About")
        about_item.connect("activate", self._on_about)
        menu.append(about_item)

        menu.append(Gtk.SeparatorMenuItem())

        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        return menu

    def run(self) -> None:
        """Initialize AppIndicator3 and run GTK loop."""
        project_root = Path(__file__).resolve().parent.parent
        self.icon_dir = project_root / "assets" / "icons"

        self.appindicator = AppIndicator3.Indicator.new(
            "lm-studio-benchmark",
            "lmstudio-bench-tray-gray",
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self.appindicator.set_icon_theme_path(str(self.icon_dir))
        self.appindicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)

        self.menu = self._build_menu()
        self.appindicator.set_menu(self.menu)
        self._refresh_menu_buttons()

        LOGGER.info(
            "AppIndicator3 tray started for dashboard: %s",
            self.dashboard_url,
        )

        self._start_status_polling()

        Gtk.main()


def _run_tray(dashboard_url: str, debug: bool) -> None:
    """Create and run tray app instance."""
    try:
        app = TrayApp(dashboard_url=dashboard_url, debug=debug)
        app.run()
    except Exception:
        LOGGER.exception("Tray thread crashed")


def start_tray(dashboard_url: str, debug: bool = False) -> bool:
    """Start tray app in a daemon thread.

    Args:
        dashboard_url: Dashboard URL used for API and browser action.
        debug: Enables debug logging for tray app.

    Returns:
        True when startup was initiated, otherwise False.
    """
    global TRAY_THREAD

    log_file = _setup_logger(debug=debug)
    LOGGER.info("Tray log file: %s", log_file)

    if IMPORT_ERROR is not None:
        LOGGER.warning("Tray dependencies unavailable: %s", IMPORT_ERROR)
        return False

    if TRAY_THREAD and TRAY_THREAD.is_alive():
        LOGGER.info("Tray already running")
        return True

    TRAY_THREAD = threading.Thread(
        target=_run_tray,
        args=(dashboard_url, debug),
        daemon=True,
    )
    TRAY_THREAD.start()
    return True


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone tray execution."""
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
    return parser.parse_args()


def main() -> int:
    """Run tray app as standalone process."""
    args = _parse_args()
    log_file = _setup_logger(debug=args.debug)
    LOGGER.info("Tray standalone start, log: %s", log_file)

    if IMPORT_ERROR is not None:
        msg = f"Cannot start tray, missing dependencies: {IMPORT_ERROR}"
        LOGGER.error(msg)
        return 1

    display_ok, _ = Gtk.init_check()
    if not display_ok:
        LOGGER.error("No graphical display available for GTK tray")
        return 1

    try:
        app = TrayApp(dashboard_url=args.url, debug=args.debug)
        app.run()
    except Exception:
        LOGGER.exception("Tray standalone startup failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
