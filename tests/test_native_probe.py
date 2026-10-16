"""Isolated native GPU protocol checks; no live sensor values are assumed."""

import ctypes
import json
import math
import os
import queue
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"temp_celsius", "power_watts", "vram_gb", "gtt_gb"}


@pytest.fixture(scope="module")
def native_binaries() -> tuple[Path, Path]:
    """Find the opt-in native build without installing tools or SDKs."""
    directory = ROOT / "build" / "windows_gpu" / "bin"
    probe = directory / "gpu_probe.exe"
    unit = directory / "gpu_probe_tests.exe"
    if os.name != "nt" or not probe.is_file() or not unit.is_file():
        pytest.skip("Build native/gpu_probe with BUILD_TESTING first.")
    return probe, unit


def test_native_units(native_binaries: tuple[Path, Path]) -> None:
    """Exercise identity, tagged SDK values, errors, units and counter resets."""
    result = subprocess.run(
        [str(native_binaries[1])],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_native_json_fixture(native_binaries: tuple[Path, Path]) -> None:
    """Validate the full JSON schema, UTF-8, nulls, valid zero and large VRAM."""
    result = subprocess.run(
        [str(native_binaries[1]), "--fixture"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    )
    assert len(result.stdout.splitlines()) == 1
    frame = json.loads(result.stdout)
    assert frame["protocol_version"] == 1
    assert frame["timestamp_unix_ms"] == 1700000000123
    assert frame["errors"]
    assert len({device["id"] for device in frame["devices"]}) == 2
    for device in frame["devices"]:
        for section in ("metrics", "status", "messages", "sources"):
            assert set(device[section]) == METRICS
        for name, value in device["metrics"].items():
            status = device["status"][name]
            assert status in {"ok", "unsupported", "unavailable", "error"}
            assert isinstance(device["messages"][name], str)
            assert isinstance(device["sources"][name], str)
            if status == "ok":
                assert isinstance(value, (int, float))
                assert math.isfinite(value)
            else:
                assert value is None
                assert device["messages"][name]
    first, second = frame["devices"]
    assert first["name"] == 'GPU "ä"\\test\n'
    assert first["metrics"]["temp_celsius"] == 0
    assert first["status"]["temp_celsius"] == "ok"
    assert first["vram_total_gb"] == 32
    assert first["driver_version"] is None
    assert first["integrated"] is None
    assert all(
        first[field] is None for field in ("pci_bus", "pci_device", "pci_function")
    )
    assert first["status"]["vram_gb"] == "error"
    assert second["metrics"]["power_watts"] == 12.5
    assert second["metrics"]["gtt_gb"] == 1.25
    assert second["integrated"] is True
    assert (second["pci_bus"], second["pci_device"], second["pci_function"]) == (
        0,
        2,
        0,
    )
    assert all("pci_domain" not in device for device in frame["devices"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--unknown"],
        ["--once", "--watch"],
        ["--watch", "--interval-ms"],
        ["--watch", "--interval-ms", "0"],
        ["--watch", "--interval-ms", "-1"],
        ["--watch", "--interval-ms", "NaN"],
        ["--watch", "--interval-ms", "60001"],
        ["--once", "--interval-ms", "1000"],
    ],
)
def test_native_cli_rejects_invalid_arguments(
    native_binaries: tuple[Path, Path], arguments: list[str]
) -> None:
    """Reject malformed CLI input before touching any vendor driver."""
    result = subprocess.run(
        [str(native_binaries[0]), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr


@pytest.mark.skipif(
    os.environ.get("LM_STUDIO_NATIVE_SMOKE") != "1",
    reason="Set LM_STUDIO_NATIVE_SMOKE=1 to exercise installed GPU drivers.",
)
def test_native_live_once(native_binaries: tuple[Path, Path]) -> None:
    """Smoke-test one real frame without assuming vendor sensor support."""
    started_ms = time.time_ns() // 1_000_000
    result = subprocess.run(
        [str(native_binaries[0]), "--once"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
        check=True,
    )
    assert len(result.stdout.splitlines()) == 1
    frame = json.loads(result.stdout)
    assert frame["protocol_version"] == 1
    assert isinstance(frame["timestamp_unix_ms"], int)
    assert started_ms <= frame["timestamp_unix_ms"] <= time.time_ns() // 1_000_000
    assert isinstance(frame["errors"], list)
    for device in frame["devices"]:
        assert device["integrated"] is None or isinstance(
            device["integrated"], bool
        )
        address = [
            device[field] for field in ("pci_bus", "pci_device", "pci_function")
        ]
        assert all(value is None for value in address) or all(
            isinstance(value, int) and 0 <= value <= maximum
            for value, maximum in zip(address, (255, 31, 7), strict=True)
        )
        assert "pci_domain" not in device
        if address[0] is None:
            assert any(
                device["id"] in error and "PCI address" in error
                for error in frame["errors"]
            )
        assert set(device["metrics"]) == METRICS
        for name, value in device["metrics"].items():
            if device["status"][name] == "ok":
                assert isinstance(value, (int, float)) and math.isfinite(value)
            else:
                assert value is None
                assert device["messages"][name]


@pytest.mark.skipif(
    os.environ.get("LM_STUDIO_NATIVE_SMOKE") != "1",
    reason="Set LM_STUDIO_NATIVE_SMOKE=1 to exercise installed GPU drivers.",
)
@pytest.mark.parametrize("shutdown", ["stop", "eof", "suspend"])
def test_native_live_watch(
    native_binaries: tuple[Path, Path], shutdown: str
) -> None:
    """Verify flushed JSON lines and bounded stop/EOF without a console."""
    process = subprocess.Popen(
        [
            str(native_binaries[0]),
            "--watch",
            "--interval-ms",
            "100" if shutdown == "suspend" else "1000",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    lines: queue.Queue[str] = queue.Queue()
    errors: list[str] = []
    drain_output = threading.Event()
    if shutdown != "suspend":
        drain_output.set()

    def read_frames() -> None:
        """Continuously drain protocol output, including shutdown output."""
        for line in process.stdout:
            lines.put(line)
            drain_output.wait()

    def read_errors() -> None:
        """Avoid stderr pipe backpressure while testing shutdown."""
        errors.extend(process.stderr)

    readers = [
        threading.Thread(target=read_frames),
        threading.Thread(target=read_errors),
    ]
    for reader in readers:
        reader.start()
    try:
        first = json.loads(lines.get(timeout=20))
        resumed_ms = None
        if shutdown == "suspend":
            # This test launches gpu_probe.exe directly, not the Python venv
            # redirector. Suspend only this owned native child and its threads.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ntdll = ctypes.WinDLL("ntdll")
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            for operation in (ntdll.NtSuspendProcess, ntdll.NtResumeProcess):
                operation.argtypes = [wintypes.HANDLE]
                operation.restype = wintypes.LONG
            handle = kernel32.OpenProcess(0x0800, False, process.pid)
            assert handle, ctypes.WinError(ctypes.get_last_error())
            suspended = False
            try:
                # Leave stdout unread so a complete pre-pause frame is
                # buffered and the producer can be blocked in its write.
                time.sleep(0.5)
                status = ntdll.NtSuspendProcess(handle)
                assert status == 0, f"NtSuspendProcess: {status:#x}"
                suspended = True
                time.sleep(16)
                resumed_ms = time.time_ns() // 1_000_000
                status = ntdll.NtResumeProcess(handle)
                assert status == 0, f"NtResumeProcess: {status:#x}"
                suspended = False
            finally:
                if suspended:
                    ntdll.NtResumeProcess(handle)
                kernel32.CloseHandle(handle)
            drain_output.set()
        second = json.loads(lines.get(timeout=5))
        assert first["protocol_version"] == second["protocol_version"] == 1
        assert second["sequence"] == first["sequence"] + 1
        assert second["timestamp_unix_ms"] > first["timestamp_unix_ms"]
        assert second["timestamp_unix_ms"] <= time.time_ns() // 1_000_000
        assert {d["id"] for d in first["devices"]} == {
            d["id"] for d in second["devices"]
        }
        if resumed_ms is not None:
            assert first["timestamp_unix_ms"] < resumed_ms
            assert second["timestamp_unix_ms"] < resumed_ms
            latest = second
            deadline = time.monotonic() + 5
            while latest["timestamp_unix_ms"] < resumed_ms:
                remaining = deadline - time.monotonic()
                assert remaining > 0, "No fresh post-resume snapshot arrived."
                latest = json.loads(lines.get(timeout=remaining))
            assert latest["sequence"] > second["sequence"]
        if shutdown != "eof":
            process.stdin.write("stop\n")
            process.stdin.flush()
        else:
            process.stdin.close()
        process.wait(timeout=5)
        assert process.returncode == 0, "".join(errors)
    finally:
        drain_output.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if not process.stdin.closed:
            process.stdin.close()
        for reader in readers:
            reader.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
    assert all(not reader.is_alive() for reader in readers)
