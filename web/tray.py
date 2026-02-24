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
import gi

logger = logging.getLogger(__name__)

_dashboard_url: str | None = None


def _build_menu() -> None:
    try:
        gi.require_version("Gtk", "3.0")
        gi.require_version("AppIndicator3", "0.1")
        # Import the required gi repositories after requiring versions
        from gi.repository import Gtk, AppIndicator3
    except (ImportError, ValueError, AttributeError) as exc:  # pragma: no cover - optional dependency
        logger.warning("GTK/AppIndicator3 not available: %s", exc)
        return

    global _dashboard_url
    if not _dashboard_url:
        logger.warning("Dashboard URL not set, skipping tray menu build")
        return

    indicator = AppIndicator3.Indicator.new(
        "lmstudio-bench",
        "applications-system",
        AppIndicator3.IndicatorCategory.APPLICATION_STATUS,
    )
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


def start_tray(dashboard_url: str) -> None:
    """Begin the tray icon in a background thread.

    This function may be called multiple times; the most recent URL is
    used by callbacks.  If GTK/AppIndicator is missing the call is a no-op.
    """
    global _dashboard_url
    _dashboard_url = dashboard_url

    def _run():
        try:
            # build menu first so indicator exists even if main loop
            # never runs (e.g. immediate failure)
            _build_menu()
            from gi.repository import Gtk
        except Exception as exc:  # pragma: no cover
            logger.warning("Tray icon unavailable: %s", exc)
            return
        Gtk.main()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
