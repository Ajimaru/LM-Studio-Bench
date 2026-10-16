"""Tests for portable latest-log references."""

import os
from pathlib import Path

import pytest

from core import logging_utils


@pytest.fixture(autouse=True)
def windows_references(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the portable Windows implementation on every test host."""
    monkeypatch.setattr(logging_utils, "IS_WINDOWS", True)


def test_writes_a_filename_pointer(tmp_path: Path) -> None:
    """A Windows reference is a regular UTF-8 file, not a symlink."""
    log = tmp_path / "benchmark_20260101_120000.log"
    log.write_text("test\n", encoding="utf-8")
    latest = tmp_path / "benchmark_latest.log"

    logging_utils.update_latest_log(log, latest)

    pointer = latest.with_suffix(".txt")
    assert pointer.read_text(encoding="utf-8") == f"{log.name}\n"
    assert not pointer.is_symlink()
    assert not latest.exists()
    assert logging_utils.resolve_latest_log(latest) == log
    assert not list(tmp_path.glob("*.tmp"))


def test_replaces_the_previous_pointer(tmp_path: Path) -> None:
    """Repeated launches update the reference without modifying old logs."""
    latest = tmp_path / "webapp_latest.log"
    first = tmp_path / "webapp_first.log"
    second = tmp_path / "webapp_second.log"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    logging_utils.update_latest_log(first, latest)
    logging_utils.update_latest_log(second, latest)

    assert logging_utils.resolve_latest_log(latest) == second
    assert first.read_text(encoding="utf-8") == "first"


def test_failed_replace_preserves_old_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Locked references are reported and do not prevent timestamped logs."""
    latest = tmp_path / "webapp_latest.log"
    first = tmp_path / "webapp_first.log"
    second = tmp_path / "webapp_second.log"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    logging_utils.update_latest_log(first, latest)

    def locked_replace(_self: Path, _target: Path) -> None:
        raise PermissionError("reference is locked")

    monkeypatch.setattr(Path, "replace", locked_replace)
    logging_utils.update_latest_log(second, latest)

    assert "Could not update latest-log reference" in caplog.text
    assert logging_utils.resolve_latest_log(latest) == first
    assert second.read_text(encoding="utf-8") == "second"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "value",
    [
        "../outside.log",
        r"..\outside.log",
        r"C:\outside.log",
        "/outside.log",
        "bad\x00.log",
        "bad\nname.log",
        "",
        "not-a-log.txt",
    ],
)
def test_rejects_invalid_pointer_contents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, value: str
) -> None:
    """A pointer cannot make log discovery read a different directory."""
    latest = tmp_path / "webapp_latest.log"
    latest.with_suffix(".txt").write_text(value, encoding="utf-8")

    assert logging_utils.resolve_latest_log(latest) is None
    assert "Could not read latest-log reference" in caplog.text


def test_reports_invalid_utf8(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt references are reported instead of crashing log discovery."""
    latest = tmp_path / "webapp_latest.log"
    latest.with_suffix(".txt").write_bytes(b"\xff")

    assert logging_utils.resolve_latest_log(latest) is None
    assert "Could not read latest-log reference" in caplog.text


def test_reports_missing_target(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A removed log does not resolve to an apparently usable path."""
    latest = tmp_path / "webapp_latest.log"
    latest.with_suffix(".txt").write_text("missing.log", encoding="utf-8")

    assert logging_utils.resolve_latest_log(latest) is None
    assert "missing log" in caplog.text


def test_reads_legacy_log(tmp_path: Path) -> None:
    """Existing latest-log files remain readable after an upgrade."""
    latest = tmp_path / "webapp_latest.log"
    latest.write_text("legacy log", encoding="utf-8")

    assert logging_utils.resolve_latest_log(latest) == latest.resolve()


def test_no_reference_returns_none(tmp_path: Path) -> None:
    """First-run log discovery does not require any references to exist."""
    assert logging_utils.resolve_latest_log(tmp_path / "latest.log") is None


def test_requires_the_same_log_directory(tmp_path: Path) -> None:
    """Only same-directory references can safely contain a bare filename."""
    with pytest.raises(ValueError, match="share the log directory"):
        logging_utils.update_latest_log(
            tmp_path / "log.log", tmp_path / "other" / "latest.log"
        )


@pytest.mark.skipif(os.name == "nt", reason="Native Unix symlink integration")
def test_preserves_unix_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unix still has the established readable latest-log symlink."""
    monkeypatch.setattr(logging_utils, "IS_WINDOWS", False)
    log = tmp_path / "webapp_test.log"
    log.write_text("log", encoding="utf-8")
    latest = tmp_path / "webapp_latest.log"

    logging_utils.update_latest_log(log, latest)

    assert latest.is_symlink()
    assert latest.read_text(encoding="utf-8") == "log"
    assert logging_utils.resolve_latest_log(latest) == log.resolve()
