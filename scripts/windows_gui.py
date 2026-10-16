"""Windowless Windows launcher; opening it never starts a benchmark."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import subprocess
import sys

from core.paths import USER_LOGS_DIR
from core.runtime import (
    PROJECT_ROOT,
    child_creation_flags,
    child_environment,
    python_executable,
)


def main(argv: list[str] | None = None) -> int:
    """Open the managed dashboard and tray using the console worker.

    Args:
        argv: Dashboard options, without the executable name.

    Returns:
        The dashboard launcher's exit code, or one on launch failure.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if getattr(sys, "frozen", False):
        command = [python_executable(), "--webapp", *args]
    else:
        command = [
            python_executable(), str(PROJECT_ROOT / "run.py"), "--webapp", *args
        ]
    log_file = USER_LOGS_DIR / (
        f"windows_launcher_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    environment = child_environment()
    environment["_LM_STUDIO_BENCH_WINDOWLESS"] = "1"
    try:
        with log_file.open("a", encoding="utf-8") as log:
            result = subprocess.run(  # nosec B603 - fixed app, shell-free argv
                command,
                cwd=Path(sys.executable).parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=child_creation_flags(),
                check=False,
            )
        return result.returncode
    except OSError:
        logging.exception("Could not launch dashboard; see %s", log_file)
        return 1


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    sys.exit(main())
