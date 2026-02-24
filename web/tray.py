#!/usr/bin/env python3
"""GTK3-based system tray helper for LM Studio Benchmark web dashboard.

The tray icon provides quick access to the currently running dashboard URL.
It lives in a separate thread so that the FastAPI server can start normally.

This module is deliberately lightweight: it only depends on GTK/GLib and
AppIndicator3 (or Ayatana) which are provided by the system.  If the
required libraries are not available the functions become no-ops.

The icon menu contains three items today:

* **Open Dashboard** – opens the browser to the given URL
* **Check for updates** – placeholder, currently does nothing
* **About** – placeholder dialog, shows a fixed version string

A "Quit" menu entry stops the GTK main loop when clicked.

The backend code in ``web/app.py`` calls :func:`start_tray` after the
web server is listening; the URL is passed in at that time and recorded
internally so that menu callbacks can access it.
"""
from __future__ import annotations

import logging
import threading
import webbrowser
import subprocess
import sys
import os
import gi

logger = logging.getLogger(__name__)


# DBus-related helpers -------------------------------------------------------

def _has_statusnotifier_watcher() -> bool:
    """Return True if a StatusNotifierWatcher is on the session bus.

    Many desktop environments (GNOME, KDE, etc.) provide a D-Bus service
    at ``org.kde.StatusNotifierWatcher`` when they support AppIndicator
    / KStatusNotifierItem tray icons.  Without it the icon will never
    appear, so we check early and log a helpful message.
    """
    try:
        from gi.repository import Gio  # type: ignore

        proxy = Gio.DBusProxy.new_for_bus_sync(
            Gio.BusType.SESSION,
            Gio.DBusProxyFlags.NONE,
            None,
            "org.kde.StatusNotifierWatcher",
            "/StatusNotifierWatcher",
            "org.kde.StatusNotifierWatcher",
            None,
        )
        # if creation succeeded we have a watcher
        return True
    except Exception:
        return False



def _desktop_notify(message: str) -> None:
    """Show a desktop notification if possible.

    Uses Gio.Notification; failure is logged at debug level.
    """
    try:
        from gi.repository import Gio  # type: ignore

        app = Gio.Application.new("org.lmstudio.tray", Gio.ApplicationFlags.FLAGS_NONE)
        app.register(None)
        notif = Gio.Notification.new("LM Studio Benchmark")
        notif.set_body(message)
        app.send_notification(None, notif)
    except Exception as exc:  # pragma: no cover - optional path
        logger.debug("desktop notification failed: %s", exc)


_dashboard_url: str | None = None


def _build_menu() -> None:
    # warn early if there is no StatusNotifierWatcher on session bus
    if not _has_statusnotifier_watcher():
        msg = (
            "No StatusNotifierWatcher detected on D-Bus; tray icon may be invisible. "
            "Install an AppIndicator/KStatusNotifier extension for your desktop."
        )
        logger.warning(msg)
        _desktop_notify(msg)
    try:
        gi.require_version("Gtk", "3.0")
        # try Ayatana first, since many modern distros ship it
        try:
            gi.require_version("AyatanaAppIndicator3", "0.1")
        except Exception:
            gi.require_version("AppIndicator3", "0.1")
        # Import the required gi repositories after requiring versions
        from gi.repository import Gtk
        # prefer Ayatana indicator if available, fall back to AppIndicator3
        try:
            from gi.repository import AyatanaAppIndicator3 as AppIndicator3  # type: ignore
            indicator_impl = "AyatanaAppIndicator3"
        except Exception:
            from gi.repository import AppIndicator3  # type: ignore
            indicator_impl = "AppIndicator3"
    except (ImportError, ValueError, AttributeError) as exc:  # pragma: no cover - optional dependency
        logger.warning("GTK/AppIndicator3 not available: %s", exc)
        return

    global _dashboard_url
    if not _dashboard_url:
        logger.warning("Dashboard URL not set, skipping tray menu build")
        return

    logger.info("Tray: building indicator using %s", indicator_impl)

    try:
        indicator = AppIndicator3.Indicator.new(
            "lmstudio-bench",
            "applications-system",
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
    except Exception as exc:
        logger.exception("Failed to create indicator: %s", exc)
        return
    indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
    indicator.set_title("LM Studio Benchmark")

    menu = Gtk.Menu()

    def open_cb(_widget):
        logger.info("Tray: Open dashboard requested -> %s", _dashboard_url)
        try:
            webbrowser.open(_dashboard_url)
        except Exception as exc:  # pragma: no cover - best-effort open
            logger.exception("Tray: failed to open browser: %s", exc)

    item_open = Gtk.MenuItem(label="Open dashboard")
    item_open.connect("activate", open_cb)
    menu.append(item_open)

    def check_updates(_widget):
        # placeholder, will be implemented later
        logger.info("Tray: update check requested")

    item_update = Gtk.MenuItem(label="Check for updates")
    item_update.connect("activate", check_updates)
    menu.append(item_update)

    def about_cb(_widget):
        logger.info("Tray: About requested")
        dialog = Gtk.MessageDialog(
            None,
            Gtk.DialogFlags.MODAL,
            Gtk.MessageType.INFO,
            Gtk.ButtonsType.OK,
            "LM Studio Benchmark",
        )
        dialog.format_secondary_text("Version: TBD")
        dialog.run()
        dialog.destroy()

    item_about = Gtk.MenuItem(label="About")
    item_about.connect("activate", about_cb)
    menu.append(item_about)

    menu.append(Gtk.SeparatorMenuItem())

    def _quit_cb(_widget):
        logger.info("Tray: Quit requested, stopping GTK main loop")
        Gtk.main_quit()

    item_quit = Gtk.MenuItem(label="Quit")
    item_quit.connect("activate", _quit_cb)
    menu.append(item_quit)

    menu.show_all()
    indicator.set_menu(menu)
    logger.info("Tray: menu built and indicator set")
    # let the user know via desktop notification in case they don't see icon
    _desktop_notify("LM Studio Benchmark tray icon should now appear in your panel.")

    # Try to set a recognizable icon; fallback to generic name
    try:
        # prefer a full icon name that most DEs provide
        indicator.set_icon_full("applications-system", "LM Studio Benchmark")
    except Exception:
        try:
            indicator.set_icon("applications-system")
        except Exception:
            logger.debug("Could not set indicator icon")

    # Diagnostic env info helpful when the icon doesn't appear
    logger.info("Tray env: DISPLAY=%s WAYLAND_DISPLAY=%s XDG_CURRENT_DESKTOP=%s DBUS_SESSION_BUS_ADDRESS=%s",
                os.environ.get("DISPLAY"),
                os.environ.get("WAYLAND_DISPLAY"),
                os.environ.get("XDG_CURRENT_DESKTOP"),
                os.environ.get("DBUS_SESSION_BUS_ADDRESS"))

def start_tray(dashboard_url: str) -> None:
    """Begin the tray icon in a background thread.

    This function may be called multiple times; the most recent URL is
    used by callbacks.  If GTK/AppIndicator is missing the call is a no-op.
    """
    global _dashboard_url
    _dashboard_url = dashboard_url

    try:
        gi.require_version("Gtk", "3.0")
        gi.require_version("AppIndicator3", "0.1")
        # attempt to import to validate availability
        from gi.repository import Gtk  # type: ignore

        def _run():
            try:
                _build_menu()
                from gi.repository import Gtk  # type: ignore
            except Exception as exc:  # pragma: no cover - runtime guard
                logger.warning("Tray icon unavailable (in-process): %s", exc)
                return
            Gtk.main()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return
    except Exception as exc:  # pragma: no cover - fallback path
        logger.info("In-process GTK3 unavailable, will try helper process: %s", exc)

    # Launch a helper process running this module so it can load GTK3
    try:
        script_path = os.path.abspath(__file__)
        python_exe = sys.executable or "/usr/bin/env python3"
        cmd = [python_exe, script_path, "--dashboard-url", _dashboard_url]
        # Capture helper output to logs/tray_helper.log for diagnosis
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(script_path)), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        helper_log = os.path.join(logs_dir, "tray_helper.log")
        f = open(helper_log, "a", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=f, stderr=f)
        logger.info("Started external tray helper (pid=%s), log=%s", proc.pid, helper_log)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.exception("Failed to start external tray helper: %s", exc)


if __name__ == "__main__":
    # Minimal CLI to run the tray as a standalone helper process. This
    # allows the main web server to spawn the tray when in-process
    # GTK3 initialization isn't possible (e.g. Gtk4 already loaded).
    import argparse

    parser = argparse.ArgumentParser(prog="web.tray")
    parser.add_argument("--dashboard-url", help="URL the tray should open")
    args = parser.parse_args()
    if args.dashboard_url:
        _dashboard_url = args.dashboard_url
    try:
        # Build menu and run GTK main loop in this process.
        _build_menu()
        from gi.repository import Gtk  # type: ignore
        Gtk.main()
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.exception("Tray helper failed: %s", exc)
