"""Lazy native Windows presentation for the shared tray controller."""

from __future__ import annotations

from collections.abc import Callable
import importlib
import os
import platform
import queue
import struct
import threading
import time
from typing import Any
import webbrowser

from core.tray import (
    APP_DESCRIPTION,
    CLEANUP_API_TIMEOUT,
    DOCUMENTATION_URL,
    LOGGER,
    REPOSITORY_URL,
    VERSION_RECHECK_ATTEMPTS,
    VERSION_RECHECK_INTERVAL_MS,
    VERSION_STATUS_PENDING,
    TrayApp,
    status_icon_color,
)

_STATUS_COLORS = {
    "gray": ("#8c8c8c", "#4d4d4d"),
    "green": ("#3bbb00", "#007111"),
    "yellow": ("#FFD700", "#CC9900"),
    "red": ("#ff0000ca", "#690000"),
}


def _load_backend() -> tuple[Any, Any]:
    """Import only the Windows pystray backend, never GTK or tkinter."""
    menu_module = importlib.import_module("pystray")
    win32_module = importlib.import_module("pystray._win32")
    importlib.import_module("PIL.Image")
    return menu_module, win32_module


def windows_tray_unavailable_reason() -> str | None:
    """Explain missing Windows x64 tray support without creating an icon."""
    native_arch = os.environ.get(
        "PROCESSOR_ARCHITEW6432",
        os.environ.get("PROCESSOR_ARCHITECTURE", platform.machine()),
    )
    if (
        platform.system() != "Windows"
        or native_arch.lower() not in {"amd64", "x86_64"}
        or struct.calcsize("P") != 8
    ):
        return "The Windows tray requires Windows 10/11 x64 and 64-bit Python."
    if int(platform.version().split(".")[0]) < 10:
        return "The Windows tray requires Windows 10/11 x64."
    try:
        _load_backend()
    except (ImportError, OSError, RuntimeError) as exc:
        return f"Windows tray requires pystray and Pillow: {exc}"
    return None


def _create_status_icons() -> dict[str, Any]:
    """Rasterize the existing status SVG geometry/colors without Cairo."""
    image = importlib.import_module("PIL.Image")
    draw_module = importlib.import_module("PIL.ImageDraw")
    fonts = importlib.import_module("PIL.ImageFont")
    scale = 4
    try:
        font = fonts.truetype("arialbd.ttf", 20 * scale)
    except OSError as exc:
        LOGGER.debug("Arial unavailable; using Pillow's bundled font: %s", exc)
        font = fonts.load_default(size=20 * scale)
    icons = {}
    for color, (fill, stroke) in _STATUS_COLORS.items():
        canvas = image.new("RGBA", (48 * scale, 48 * scale))
        draw = draw_module.Draw(canvas)
        draw.ellipse(
            (8, 8, 184, 184), fill="#4A90E2", outline="#2E5C8A", width=8
        )
        draw.text((96, 128), "LM", font=font, fill="white", anchor="ms")
        draw.ellipse((128, 16, 176, 64), fill=fill, outline=stroke, width=4)
        draw.line((152, 28, 152, 52), fill=stroke, width=6)
        draw.line((140, 40, 164, 40), fill=stroke, width=6)
        icons[color] = canvas.resize((48, 48), image.Resampling.LANCZOS)
        canvas.close()
    return icons


class _NativeDialogs:
    """Own message boxes on a separate thread so controls never wait for them."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._window_callback = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )
        self._user32.MessageBoxW.argtypes = [
            wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT
        ]
        self._user32.MessageBoxW.restype = ctypes.c_int
        self._user32.EnumThreadWindows.argtypes = [
            wintypes.DWORD, self._window_callback, wintypes.LPARAM
        ]
        self._user32.EnumThreadWindows.restype = wintypes.BOOL
        self._user32.PostMessageW.argtypes = [
            wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        ]
        self._user32.PostMessageW.restype = wintypes.BOOL
        self._closed = threading.Event()
        self._messages: queue.Queue[
            tuple[str, str, Callable[[bool], None] | None]
        ] = queue.Queue(maxsize=8)
        self._thread = threading.Thread(
            target=self._run, name="tray-dialogs", daemon=True
        )
        self._thread.start()

    def show(
        self,
        title: str,
        message: str,
        answer: Callable[[bool], None] | None = None,
    ) -> None:
        """Queue an information box or a Yes/No question without blocking."""
        if self._closed.is_set():
            return
        try:
            self._messages.put_nowait((title, message, answer))
        except queue.Full:
            LOGGER.warning("Tray dialog queue is full: %s", title)
            if answer is not None:
                answer(False)

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                title, message, answer = self._messages.get(timeout=0.1)
            except queue.Empty:
                continue
            accepted = False
            try:
                # Questions default to No; Cancel/WM_CLOSE also decline.
                flags = 0x10000 | (0x133 if answer else 0x40)
                result = self._user32.MessageBoxW(None, message, title, flags)
                if result == 0:
                    raise self._ctypes.WinError(self._ctypes.get_last_error())
                accepted = result == 6  # IDYES
            except OSError:
                LOGGER.exception("Could not display native tray dialog: %s", title)
            finally:
                self._messages.task_done()
            if answer is not None and not self._closed.is_set():
                try:
                    answer(accepted)
                except Exception:
                    LOGGER.exception("Native tray dialog callback failed")

    def dismiss(self) -> None:
        """Dismiss only windows owned by this instance's dialog thread."""
        while True:
            try:
                self._messages.get_nowait()
                self._messages.task_done()
            except queue.Empty:
                break
        thread_id = self._thread.native_id
        if thread_id is None or not self._thread.is_alive():
            return

        @self._window_callback
        def close_window(hwnd: Any, _param: int) -> bool:
            if not self._user32.PostMessageW(hwnd, 0x0010, 0, 0):  # WM_CLOSE
                LOGGER.warning("Could not close owned tray dialog: %s", hwnd)
            return True

        self._user32.EnumThreadWindows(thread_id, close_window, 0)

    def close(self) -> bool:
        """Close owned dialogs and join their worker within three seconds."""
        self._closed.set()
        deadline = time.monotonic() + 3
        while self._thread.is_alive() and time.monotonic() < deadline:
            self.dismiss()
            self._thread.join(0.05)
        if self._thread.is_alive():
            LOGGER.error("Native tray dialog worker did not stop")
            return False
        return True


class WindowsTrayApp(TrayApp):
    """Run shared API/status/update actions behind a native, responsive tray."""

    def __init__(self, dashboard_url: str, debug: bool = False) -> None:
        """Initialize controller state without importing GUI dependencies."""
        super().__init__(dashboard_url, debug)
        self._icon: Any = None
        self._icons: dict[str, Any] = {}
        self._dialogs: _NativeDialogs | None = None
        self._commands: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=16)
        self._stopped = threading.Event()
        self._closing = threading.Event()
        self._quit_requested = threading.Event()
        self._worker: threading.Thread | None = None
        self._status = "idle"

    def _wait_for_close(self, seconds: float) -> bool:
        return self._closing.wait(seconds) or self._stopped.is_set()

    def _enqueue(self, action: Callable[[Any], None]) -> None:
        if self._closing.is_set() or self._stopped.is_set():
            LOGGER.debug("Ignoring tray action while closing")
            return
        try:
            self._commands.put_nowait(lambda: action(None))
        except queue.Full:
            LOGGER.warning("Tray action queue is full")
            self._show_info_dialog("Tray Busy", "Please wait for the current action.")

    def _callback(self, action: Callable[[Any], None]) -> Callable[..., None]:
        def invoke(_icon: Any, _item: Any) -> None:
            self._enqueue(action)

        return invoke

    def _build_native_menu(self, pystray: Any) -> Any:
        item = pystray.MenuItem

        def ready(_item: Any) -> bool:
            return not self._closing.is_set() and not self._stopped.is_set()

        def active(menu_item: Any) -> bool:
            return ready(menu_item) and self._status in {"running", "paused"}

        return pystray.Menu(
            item("Benchmark", None, enabled=False),
            item(
                "Start", self._callback(self._on_start),
                enabled=lambda entry: ready(entry) and not active(entry),
            ),
            item(
                lambda _entry: "Resume" if self._status == "paused" else "Pause",
                self._callback(self._on_pause_resume), enabled=active,
            ),
            item("Stop", self._callback(self._on_stop), enabled=active),
            pystray.Menu.SEPARATOR,
            item("Show Status", self._callback(self._on_status), enabled=ready),
            item(
                "Open Dashboard", self._callback(self._on_open_webapp),
                enabled=ready, default=True,
            ),
            item(
                "Check for Updates", self._callback(self._on_check_updates_clicked),
                enabled=ready,
            ),
            item(
                "About",
                pystray.Menu(
                    item("Info and Contributors", self._callback(self._on_about)),
                    item(
                        "Documentation",
                        self._callback(
                            lambda _item: webbrowser.open(DOCUMENTATION_URL)
                        ),
                    ),
                    item(
                        "GitHub Repository",
                        self._callback(lambda _item: webbrowser.open(REPOSITORY_URL)),
                    ),
                ),
                enabled=ready,
            ),
            pystray.Menu.SEPARATOR,
            item(
                "Quit", lambda _icon, _item: self.request_quit(), enabled=ready
            ),
        )

    def _apply_menu_status(self, status: str) -> None:
        self._status = status
        if self._icon is not None and not self._stopped.is_set():
            self._icon.update_menu()

    def _update_icon_for_status(self, status: str, api_reachable: bool) -> None:
        if self._icon is None or self._stopped.is_set():
            return
        self._icon.icon = self._icons[status_icon_color(status, api_reachable)]
        detail = status if api_reachable else "dashboard unavailable"
        self._icon.title = f"LM Studio Benchmark - {detail}"

    def _show_info_dialog(self, title: str, message: str) -> None:
        if self._dialogs is not None:
            self._dialogs.show(title, message)
        else:
            LOGGER.error("%s: %s (native dialog unavailable)", title, message)

    def _show_update_notification(self) -> None:
        if not self.pending_update or self._dialogs is None:
            return
        update = dict(self.pending_update)

        def open_release(accepted: bool) -> None:
            if accepted and update.get("url"):
                self._enqueue(lambda _item: webbrowser.open(update["url"]))

        self._dialogs.show(
            "Update Available",
            f"Current: {update.get('current', 'unknown')}\n"
            f"Latest: {update.get('latest', 'unknown')}\n\n"
            "Open the release download page?",
            open_release,
        )

    def _on_about(self, _item: Any) -> None:
        version = self._read_version()
        status = self._get_about_version_status(version)
        for _ in range(VERSION_RECHECK_ATTEMPTS):
            if status != VERSION_STATUS_PENDING:
                break
            if self._wait_for_close(VERSION_RECHECK_INTERVAL_MS / 1000):
                return
            status = self._get_about_version_status(version)
        if status == VERSION_STATUS_PENDING:
            status = "unknown"
        contributors = []
        for name, handle, role in self._read_contributors():
            text = f"{name} (@{handle})" if handle else name
            contributors.append(f"{text} - {role}" if role else text)
        self._show_info_dialog(
            "About LM Studio Benchmark",
            f"LM Studio Model Benchmark\n{version} ({status})\n\n"
            f"{APP_DESCRIPTION}\n\nContributors\n"
            + "\n".join(contributors)
            + f"\n\nDocumentation: {DOCUMENTATION_URL}"
            + f"\nGitHub: {REPOSITORY_URL}\n\n\u00a9 2026 Ajimaru",
        )

    def request_quit(self) -> None:
        """Queue cooperative dashboard shutdown; safe on the native UI thread."""
        if not self._closing.is_set() and not self._stopped.is_set():
            self._closing.set()
            self._quit_requested.set()

    def _on_quit(self, _item: Any) -> None:
        if self._dialogs is not None:
            self._dialogs.dismiss()
        if self._shutdown_dashboard():
            self.stop()
            return
        self._update_icon_for_status("error", True)
        if self._dialogs is None:
            self._closing.clear()
            return

        def finish(accepted: bool) -> None:
            if accepted:
                LOGGER.warning("Exiting tray only; dashboard cleanup is unconfirmed")
                self.stop()
            else:
                self._closing.clear()

        self._dialogs.show(
            "Shutdown Failed",
            f"{self._shutdown_error}\n\n"
            "The dashboard or benchmark may still be running. "
            "No process has been force-killed.\n\n"
            "Quit only the tray without confirming dashboard cleanup?\n"
            "Choose No to keep the tray and retry.",
            finish,
        )

    def _work(self) -> None:
        next_poll = time.monotonic()
        while not self._stopped.is_set():
            try:
                if self._quit_requested.is_set():
                    self._quit_requested.clear()
                    self._on_quit(None)
                    continue
                if not self._closing.is_set() and time.monotonic() >= next_poll:
                    self._on_polling_tick()
                    next_poll = time.monotonic() + 3
                try:
                    action = self._commands.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if not self._closing.is_set() and not self._stopped.is_set():
                        action()
                finally:
                    self._commands.task_done()
            except Exception:
                LOGGER.exception("Windows tray worker failed")
                self._closing.clear()
                self._show_info_dialog(
                    "Tray Action Failed",
                    "The action failed. Check the tray log and try again.",
                )
                next_poll = time.monotonic() + 3

    def _start_status_polling(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(
                target=self._work, name="tray-controller", daemon=True
            )
            self._worker.start()

    def stop(self) -> None:
        """Stop this tray only; use request_quit for dashboard cleanup too."""
        self._closing.set()
        self._stopped.set()
        if self._icon is not None:
            self._icon.stop()

    def run(self) -> None:
        """Run native events and always dispose of owned workers and icons."""
        pystray, win32 = _load_backend()
        self._icons = _create_status_icons()
        cleanup_failed = False
        try:
            self._dialogs = _NativeDialogs()
            self._icon = win32.Icon(
                "lm-studio-benchmark", self._icons["gray"],
                "LM Studio Benchmark", self._build_native_menu(pystray),
            )

            def setup(icon: Any) -> None:
                if not self._stopped.is_set():
                    icon.visible = True
                    self._start_status_polling()

            LOGGER.info("Native Windows tray started for %s", self.dashboard_url)
            self._icon.run(setup=setup)
        finally:
            self.stop()
            if self._dialogs is not None and not self._dialogs.close():
                cleanup_failed = True
            if self._worker is not None:
                self._worker.join(CLEANUP_API_TIMEOUT + 10)
                if self._worker.is_alive():
                    LOGGER.error("Windows tray controller did not stop")
                    cleanup_failed = True
            for icon in self._icons.values():
                icon.close()
            if cleanup_failed:
                raise RuntimeError("Windows tray cleanup was incomplete; see tray log")
