"""Versioned PyInstaller spec, copied to build/ with a .spec suffix by the builder."""

import json
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

# These names are supplied by PyInstaller when it evaluates a spec.
# ruff: noqa: F821

root = Path(os.environ["LMSB_BUILD_ROOT"])
stage = Path(os.environ["LMSB_BUILD_STAGE"])
inputs = json.loads((stage / "freeze-inputs.json").read_text(encoding="utf-8"))
datas = [(item[0], item[1]) for item in inputs["datas"]]
binaries = [(inputs["native_probe"], "native")]
hiddenimports = [
    "cli.benchmark",
    "cli.main",
    "core.tray",
    "core.windows_tray",
    "tools.scrape_metadata",
    "web.app",
    "scripts.windows_selftest",
    "pystray._win32",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]
sdk_data, sdk_binaries, sdk_imports = collect_all("lmstudio")
datas += sdk_data + collect_data_files("plotly") + collect_data_files("reportlab")
binaries += sdk_binaries
hiddenimports += sdk_imports
excludes = [
    "gi", "cairo", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
    "IPython", "pytest", "pystray._appindicator", "pystray._gtk",
    "pystray._darwin", "pystray._xorg",
]
common = dict(
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
console = Analysis([str(root / "run.py")], **common)
gui = Analysis([str(root / "scripts" / "windows_gui.py")], **common)
console_exe = EXE(
    PYZ(console.pure),
    console.scripts,
    [],
    exclude_binaries=True,
    name="lmstudio-bench",
    console=True,
    debug=False,
    strip=False,
    upx=False,
    icon=str(stage / "lmstudio-bench.ico"),
    version=str(stage / "version-info.txt"),
)
gui_exe = EXE(
    PYZ(gui.pure),
    gui.scripts,
    [],
    exclude_binaries=True,
    name="LM-Studio-Bench",
    console=False,
    debug=False,
    strip=False,
    upx=False,
    icon=str(stage / "lmstudio-bench.ico"),
    version=str(stage / "version-info.txt"),
)
COLLECT(
    console_exe,
    gui_exe,
    console.binaries,
    console.datas,
    gui.binaries,
    gui.datas,
    strip=False,
    upx=False,
    name="LM-Studio-Bench",
)
