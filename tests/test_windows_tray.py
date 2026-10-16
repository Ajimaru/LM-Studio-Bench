"""Native tray tests with no live GUI, HTTP requests, or model workers."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]


class MenuItem:
    """Minimal pystray menu double with evaluated labels/enabled state."""

    def __init__(
        self, text: Any, action: Any, enabled: Any = True, default: bool = False
    ) -> None:
        self.label = text
        self.action = action
        self.sensitivity = enabled
        self.default = default

    @property
    def text(self) -> str:
        """Return the dynamic label just as pystray does."""
        return self.label(self) if callable(self.label) else self.label

    @property
    def enabled(self) -> bool:
        """Evaluate cached menu state without network activity."""
        return (
            self.sensitivity(self)
            if callable(self.sensitivity)
            else self.sensitivity
        )


class Menu(tuple):
    """Minimal pystray container."""

    SEPARATOR = object()

    def __new__(cls, *items: Any) -> Menu:
        return super().__new__(cls, items)


class Icon:
    """Native event-loop double with bounded lifecycle."""

    def __init__(self, name: str, image: Any, title: str, menu: Menu) -> None:
        self.name = name
        self.icon = image
        self.title = title
        self.menu = menu
        self.visible = False
        self.stopped = threading.Event()
        self.started = threading.Event()
        self.update_menu = MagicMock()

    def run(self, setup: Callable[[Icon], None]) -> None:
        """Run until explicitly stopped; never leave a background test hanging."""
        setup(self)
        self.started.set()
        if not self.stopped.wait(5):
            raise RuntimeError("Test icon was not stopped")

    def stop(self) -> None:
        """Stop only this fake icon."""
        self.visible = False
        self.stopped.set()


class Dialogs:
    """Nonblocking native dialog recorder."""

    def __init__(self) -> None:
        self.show = MagicMock()
        self.dismiss = MagicMock()
        self.close = MagicMock(return_value=True)


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load against the current tray module, after GTK tests may have reloaded it."""
    module = importlib.import_module("core.windows_tray")
    module = importlib.reload(module)
    monkeypatch.setattr(
        module, "_load_backend",
        lambda: (SimpleNamespace(Menu=Menu, MenuItem=MenuItem), SimpleNamespace(Icon=Icon)),
    )
    monkeypatch.setattr(
        module, "_create_status_icons",
        lambda: {color: MagicMock(name=color) for color in ("gray", "green", "yellow", "red")},
    )
    monkeypatch.setattr(module, "_NativeDialogs", Dialogs)
    return module


def make_app(native: Any) -> Any:
    """Create a controller with fake UI state without running a worker."""
    app = native.WindowsTrayApp("http://localhost:8080")
    icons = getattr(native, "_create_status_icons")()
    setattr(app, "_icons", icons)
    setattr(app, "_dialogs", Dialogs())
    menu = getattr(app, "_build_native_menu")(
        SimpleNamespace(Menu=Menu, MenuItem=MenuItem)
    )
    setattr(app, "_icon", Icon("test", icons["gray"], "test", menu))
    return app


def menu_item(app: Any, text: str) -> MenuItem:
    """Find an action in the fake root menu."""
    return next(
        entry for entry in getattr(app, "_icon").menu
        if isinstance(entry, MenuItem) and entry.text == text
    )


@pytest.mark.parametrize(
    ("status", "reachable", "color", "start", "pause"),
    [
        ("idle", True, "gray", True, False),
        ("running", True, "green", False, True),
        ("paused", True, "yellow", False, True),
        ("error", True, "red", True, False),
        ("idle", False, "red", True, False),
    ],
)
def test_status_controls_use_shared_colors(
    native, status, reachable, color, start, pause
):
    """Every state maps to the same status color and controls as the GTK tray."""
    app = make_app(native)
    app._call_api = MagicMock(
        return_value={"status": status} if reachable else None
    )
    getattr(app, "_refresh_menu_buttons")()
    assert getattr(app, "_icon").icon is getattr(app, "_icons")[color]
    assert menu_item(app, "Start").enabled is start
    assert menu_item(app, "Resume" if status == "paused" else "Pause").enabled is pause
    assert menu_item(app, "Stop").enabled is pause
    app._call_api.assert_called_once_with("/api/status")


def test_callbacks_only_enqueue_and_keep_all_controls(native):
    """Native callbacks/menu getters never execute HTTP or a modal dialog."""
    app = make_app(native)
    app._call_api = MagicMock()
    app._show_info_dialog = MagicMock()
    for label in ("Start", "Pause", "Stop", "Show Status", "Check for Updates"):
        menu_item(app, label).action(None, None)
    assert getattr(app, "_commands").qsize() == 5
    app._call_api.assert_not_called()
    app._show_info_dialog.assert_not_called()
    assert menu_item(app, "Open Dashboard").default
    about = menu_item(app, "About").action
    assert [entry.text for entry in about] == [
        "Info and Contributors", "Documentation", "GitHub Repository"
    ]
    assert menu_item(app, "Quit").enabled


def test_controller_actions_are_inherited(native):
    """Windows does not maintain a second set of API/status/update logic."""
    tray = importlib.import_module("core.tray")
    for method in (
        "_call_api", "_on_start", "_on_pause_resume", "_on_stop", "_on_status",
        "_on_open_webapp", "_on_check_updates_clicked", "_on_polling_tick",
        "_shutdown_dashboard",
    ):
        assert getattr(native.WindowsTrayApp, method) is getattr(tray.TrayApp, method)


def test_failed_shutdown_requires_explicit_tray_only_confirmation(native):
    """Failure remains visible and does not silently close the tray."""
    app = make_app(native)
    app._call_api = MagicMock(
        return_value={"success": False, "message": "Worker cleanup failed"}
    )
    app.request_quit()
    getattr(app, "_on_quit")(None)
    icon = getattr(app, "_icon")
    assert not icon.stopped.is_set()
    title, text, answer = getattr(app, "_dialogs").show.call_args.args
    assert title == "Shutdown Failed"
    assert "Worker cleanup failed" in text
    assert "may still be running" in text
    answer(False)
    assert not getattr(app, "_closing").is_set()
    assert not icon.stopped.is_set()
    app.request_quit()
    getattr(app, "_on_quit")(None)
    getattr(app, "_dialogs").show.call_args.args[2](True)
    assert icon.stopped.is_set()


def test_about_preserves_version_and_contributors(native):
    """Native About retains bundled metadata without importing GTK."""
    app = make_app(native)
    app._get_about_version_status = MagicMock(return_value="no update")
    getattr(app, "_on_about")(None)
    title, message = getattr(app, "_dialogs").show.call_args.args
    assert title == "About LM Studio Benchmark"
    assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in message
    assert "Ajimaru" in message
    assert "kjake" in message
    assert "macOS support" in message
    assert native.DOCUMENTATION_URL in message


def test_update_download_confirmation_is_nonblocking(native, monkeypatch):
    """The native question queues browser work only after explicit consent."""
    app = make_app(native)
    app.pending_update = {
        "current": "v1.0.0", "latest": "v1.1.0",
        "url": "https://github.com/Ajimaru/LM-Studio-Bench/releases",
    }
    browser = MagicMock()
    monkeypatch.setattr(native.webbrowser, "open", browser)
    getattr(app, "_show_update_notification")()
    answer = getattr(app, "_dialogs").show.call_args.args[2]
    answer(False)
    assert getattr(app, "_commands").empty()
    answer(True)
    browser.assert_not_called()
    getattr(app, "_commands").get_nowait()()
    browser.assert_called_once_with(app.pending_update["url"])


def test_quit_is_responsive_during_slow_http_and_joins_workers(native):
    """HTTP cleanup stays off the native event loop and owned workers stop."""
    app = native.WindowsTrayApp("http://localhost:8080")
    request_started = threading.Event()
    release_request = threading.Event()
    finished = threading.Event()
    request_threads = []
    failures = []

    def api(endpoint, *_args, **_kwargs):
        request_threads.append(threading.get_ident())
        if endpoint == "/api/status":
            request_started.set()
            if not release_request.wait(3):
                raise RuntimeError("Test HTTP response was not released")
            return {"status": "running"}
        return {"success": True}

    def run():
        try:
            app.run()
        except Exception as exc:
            failures.append(exc)
        finally:
            finished.set()

    app._call_api = api
    ui_thread = threading.Thread(target=run)
    ui_thread.start()
    try:
        assert request_started.wait(2)
        app.request_quit()
        assert not finished.is_set()
        assert getattr(app, "_quit_requested").is_set()
        release_request.set()
        assert finished.wait(3)
    finally:
        release_request.set()
        app.stop()
        ui_thread.join(5)
    assert not failures
    assert not ui_thread.is_alive()
    assert not getattr(app, "_worker").is_alive()
    getattr(app, "_dialogs").close.assert_called_once()
    assert all(ident != threading.get_ident() for ident in request_threads)
    assert all(ident != ui_thread.ident for ident in request_threads)
    for image in getattr(app, "_icons").values():
        image.close.assert_called_once()


def test_startup_failure_disposes_native_resources(native, monkeypatch):
    """A failed native event loop still closes images and dialog workers."""
    app = native.WindowsTrayApp("http://localhost:8080")
    monkeypatch.setattr(Icon, "run", MagicMock(side_effect=OSError("no desktop")))
    with pytest.raises(OSError, match="no desktop"):
        app.run()
    getattr(app, "_dialogs").close.assert_called_once()
    for image in getattr(app, "_icons").values():
        image.close.assert_called_once()


def test_icon_renderer_matches_existing_svg_colors():
    """Pillow reuses checked-in icon geometry/colors, without generated files."""
    module = importlib.import_module("core.windows_tray")
    images = getattr(module, "_create_status_icons")()
    try:
        for color, image in images.items():
            path = ROOT / "assets" / "icons" / f"lmstudio-bench-tray-{color}.svg"
            circles = ET.parse(path).findall("{http://www.w3.org/2000/svg}circle")
            expected = getattr(module, "_STATUS_COLORS")[color]
            assert (circles[1].get("fill"), circles[1].get("stroke")) == expected
            assert image.size == (48, 48)
            assert image.mode == "RGBA"
            assert image.getpixel((24, 8))[:3] == (74, 144, 226)
    finally:
        for image in images.values():
            image.close()


def test_gui_imports_are_lazy(monkeypatch):
    """Importing either controller module cannot require a GUI dependency."""
    import_ = importlib.import_module
    forbidden = {"gi", "pystray", "PIL", "tkinter"}

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            pytest.fail(f"Eager GUI import: {name}")
        return import_(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", guarded)
    tray = importlib.reload(import_("core.tray"))
    module = importlib.reload(import_("core.windows_tray"))
    module.WindowsTrayApp("http://localhost:8080")
    assert tray.GTK is None


def test_factory_selects_windows_without_gtk(native, monkeypatch):
    """The parent launcher can use one public platform-aware factory."""
    tray = importlib.import_module("core.tray")
    monkeypatch.setattr(tray, "IS_MACOS", False)
    monkeypatch.setattr(tray, "IS_LINUX", False)
    monkeypatch.setattr(tray, "IS_WINDOWS", True)
    monkeypatch.setattr(native, "windows_tray_unavailable_reason", lambda: None)
    app = tray.create_tray_app("http://localhost:8080")
    assert isinstance(app, native.WindowsTrayApp)


def test_macos_remains_tray_free(monkeypatch):
    """Installing GUI libraries on macOS does not accidentally enable a tray."""
    tray = importlib.import_module("core.tray")
    monkeypatch.setattr(tray, "IS_MACOS", True)
    monkeypatch.setattr(tray, "GTK", MagicMock())
    assert "macOS" in tray.tray_unavailable_reason()
    with pytest.raises(RuntimeError, match="macOS"):
        tray.create_tray_app("http://localhost:8080")


def test_windows_arm64_is_not_claimed(native, monkeypatch):
    """Native host architecture, not just pointer width, gates support."""
    monkeypatch.setattr(native.platform, "system", lambda: "Windows")
    monkeypatch.setenv("PROCESSOR_ARCHITEW6432", "ARM64")
    assert "x64" in native.windows_tray_unavailable_reason()
