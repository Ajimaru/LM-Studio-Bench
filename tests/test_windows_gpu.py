"""Deterministic tests for the native Windows telemetry protocol and bridge."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from tools import windows_gpu as win


class _FakeWatch:
    """A controlled native-pipe test double, never a hardware implementation."""

    def __init__(self, frame: str, ignore_stop: bool = False) -> None:
        self.done = threading.Event()
        self.frame = frame
        self.sent = False
        self.killed = False
        self.ignore_stop = ignore_stop
        self.stdout = self
        self.stderr = io.StringIO("")
        self.stdin = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.done.set()

    def readline(self, _size: int) -> str:
        if not self.sent:
            self.sent = True
            return self.frame + "\n"
        self.done.wait(5)
        return ""

    def write(self, value: str) -> None:
        if value == "stop\n" and not self.ignore_stop:
            self.done.set()

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.done.set()

    def poll(self) -> int | None:
        return (1 if self.killed else 0) if self.done.is_set() else None

    def wait(self, timeout: float = 1) -> int:
        if self.ignore_stop and not self.killed:
            raise subprocess.TimeoutExpired("fake-probe", timeout)
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired("fake-probe", timeout)
        return 1 if self.killed else 0

    def kill(self) -> None:
        self.killed = True
        self.done.set()


def device_data(vendor: str = "AMD", index: int = 0) -> dict[str, Any]:
    """Build a realistic protocol fixture without claiming physical readings."""
    return {
        "id": f"luid:00000000:{index + 1:08x}",
        "index": index,
        "vendor": vendor,
        "name": f"{vendor} test adapter",
        "vram_total_gb": 8.0,
        "shared_total_gb": 16.0,
        "driver_version": "test-driver",
        "integrated": False,
        "pci_bus": index + 1,
        "pci_device": 0,
        "pci_function": 0,
        "metrics": {
            "temp_celsius": 50.0,
            "power_watts": 75.0,
            "vram_gb": 2.0,
            "gtt_gb": 0.0,
        },
        "status": {sensor: "ok" for sensor in win.SENSOR_NAMES},
        "messages": {sensor: "" for sensor in win.SENSOR_NAMES},
        "sources": {sensor: "fixture" for sensor in win.SENSOR_NAMES},
    }


def snapshot_data(
    devices: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Wrap device fixtures in the actual versioned envelope."""
    return {
        "protocol_version": 1,
        "sequence": 0,
        "timestamp_unix_ms": int(time.time() * 1000),
        "devices": [device_data()] if devices is None else devices,
        "errors": [],
    }


def test_amd_all_metrics_survive_protocol_validation() -> None:
    """ADLX/Windows readings retain their values and units end to end."""
    payload = snapshot_data()
    sample = win.parse_snapshot(json.dumps(payload))
    adapter = sample.primary_device()
    assert adapter is not None
    assert adapter.vendor == "AMD"
    assert adapter.metrics.temp_celsius == 50
    assert adapter.metrics.power_watts == 75
    assert adapter.metrics.vram_gb == 2
    assert adapter.metrics.gtt_gb == 0
    assert adapter.status.gtt_gb == "ok"


def test_unsupported_amd_sensor_stays_missing() -> None:
    """A support-query refusal is not converted into a zero measurement."""
    payload = snapshot_data()
    device = payload["devices"][0]
    device["metrics"]["power_watts"] = None
    device["status"]["power_watts"] = "unsupported"
    device["messages"]["power_watts"] = "ADLX support query returned false"
    sample = win.parse_snapshot(json.dumps(payload))
    assert sample.devices[0].metrics.power_watts is None
    assert sample.devices[0].status.power_watts == "unsupported"


def test_driver_errors_are_not_unsupported_sensors() -> None:
    """Loader/query failures retain a different status from physical support."""
    payload = snapshot_data()
    device = payload["devices"][0]
    device["metrics"]["temp_celsius"] = None
    device["status"]["temp_celsius"] = "error"
    device["messages"]["temp_celsius"] = "ADLX initialization failed"
    sample = win.parse_snapshot(json.dumps(payload))
    assert sample.devices[0].status.temp_celsius == "error"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, "12", True])
def test_rejects_invalid_memory_values(value: object) -> None:
    """Malformed readings cannot reach averages, exports or charts."""
    payload = snapshot_data()
    payload["devices"][0]["metrics"]["vram_gb"] = value
    with pytest.raises(win.WindowsGpuError, match="Invalid native GPU response"):
        win.parse_snapshot(json.dumps(payload))


@pytest.mark.parametrize("version", [2, "1", True])
def test_rejects_incompatible_protocol(version: object) -> None:
    """Unsupported or incorrectly typed protocol versions fail explicitly."""
    payload = snapshot_data()
    payload["protocol_version"] = version
    with pytest.raises(win.WindowsGpuError):
        win.parse_snapshot(json.dumps(payload))


def test_rejects_success_without_a_value() -> None:
    """An ok status without a reading is a protocol error, not a fallback."""
    payload = snapshot_data()
    payload["devices"][0]["metrics"]["power_watts"] = None
    with pytest.raises(win.WindowsGpuError, match="Inconsistent"):
        win.parse_snapshot(json.dumps(payload))


def test_duplicate_ids_are_rejected() -> None:
    """Adapters cannot be merged or overwritten by an ambiguous identity."""
    payload = snapshot_data()
    payload["devices"].append(deepcopy(payload["devices"][0]))
    with pytest.raises(win.WindowsGpuError, match="Duplicate"):
        win.parse_snapshot(json.dumps(payload))


def test_primary_selection_preserves_vendor_priority() -> None:
    """A hybrid laptop selects NVIDIA without using names or row order."""
    sample = win.parse_snapshot(json.dumps(snapshot_data([
        device_data("Intel", 0), device_data("NVIDIA", 1), device_data("AMD", 2)
    ])))
    primary = sample.primary_device()
    assert primary is not None and primary.vendor == "NVIDIA"


def test_nvidia_pci_matching_ignores_order_and_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PCI identity, not a similarly named GPU, selects the NVIDIA reading."""
    payload = snapshot_data([
        device_data("NVIDIA", 0), device_data("NVIDIA", 1)
    ])
    output = (
        "00000000:02:00.0, 0, Same name, 61, 110, 4096, 8192, 555.0\n"
        "00000000:01:00.0, 1, Same name, 42, 55, 1024, 4096, 555.0\n"
    )
    monkeypatch.setattr(
        win.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, output, ""
        ),
    )
    result = win.merge_nvidia(win.parse_snapshot(json.dumps(payload)), "nvidia-smi")
    assert result.devices[0].metrics.temp_celsius == 42
    assert result.devices[0].metrics.vram_gb == 1
    assert result.devices[0].vram_total_gb == 4
    assert result.devices[1].metrics.temp_celsius == 61
    assert result.devices[1].metrics.vram_gb == 4
    primary = result.primary_device()
    assert primary is not None
    assert primary.id == result.devices[1].id


def test_ambiguous_nvidia_domains_do_not_guess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown PCI domain cannot silently pick between two matching GPUs."""
    output = (
        "00000000:01:00.0, 0, GPU, 40, 50, 1024, 4096, 555.0\n"
        "00000001:01:00.0, 1, GPU, 70, 150, 2048, 8192, 555.0\n"
    )
    monkeypatch.setattr(
        win.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    sample = win.parse_snapshot(json.dumps(snapshot_data([device_data("NVIDIA")])))
    result = win.merge_nvidia(sample, "nvidia-smi")
    assert result.devices[0].metrics.temp_celsius is None
    assert result.devices[0].status.temp_celsius == "error"
    assert "uniquely associated" in result.devices[0].messages["temp_celsius"]


def test_missing_nvidia_does_not_hide_amd_readings() -> None:
    """Vendor failures affect only their own device, not the entire frame."""
    sample = win.parse_snapshot(json.dumps(snapshot_data([
        device_data("AMD"), device_data("NVIDIA", 1)
    ])))
    result = win.merge_nvidia(sample, None)
    assert result.devices[0].metrics.power_watts == 75
    assert result.devices[1].metrics.power_watts is None
    assert result.devices[1].status.power_watts == "unavailable"
    assert result.errors


def test_probe_exit_124_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watchdog's failure is not an empty successful inventory."""
    monkeypatch.setattr(win, "probe_path", lambda: Path("gpu_probe.exe"))
    monkeypatch.setattr(
        win.subprocess, "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 124, "", "watchdog timeout"
        ),
    )
    with pytest.raises(win.WindowsGpuError, match="124"):
        win.query_snapshot()


def test_probe_timeout_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blocked driver calls remain bounded independently of the native watchdog."""
    monkeypatch.setattr(win, "probe_path", lambda: Path("gpu_probe.exe"))
    monkeypatch.setattr(
        win.subprocess, "run",
        MagicMock(side_effect=subprocess.TimeoutExpired("gpu_probe", 20)),
    )
    with pytest.raises(win.WindowsGpuError, match="probe failed"):
        win.query_snapshot()


def test_stale_and_future_frames_are_not_current_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suspended parent must not replay buffered measurements as current."""
    sampler = win.WindowsGpuSampler()
    payload = snapshot_data()
    snapshot = win.parse_snapshot(json.dumps(payload))
    monkeypatch.setattr(sampler, "_raw", snapshot)
    now = snapshot.timestamp_unix_ms / 1000
    monkeypatch.setattr(win.time, "time", lambda: now)
    assert sampler.raw_snapshot() is snapshot
    monkeypatch.setattr(win.time, "time", lambda: now + 16)
    assert sampler.raw_snapshot() is None
    monkeypatch.setattr(win.time, "time", lambda: now - 2)
    assert sampler.raw_snapshot() is None


def test_amd_detection_never_uses_linux_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated AMD adapter is used without ROCm, lspci or sysfs probes."""
    import tools.hardware_monitor as hw

    snapshot = win.parse_snapshot(json.dumps(snapshot_data()))
    monkeypatch.setattr(hw, "IS_WINDOWS", True)
    monkeypatch.setattr(hw, "IS_MACOS", False)
    monkeypatch.setattr(hw, "query_windows_snapshot", lambda _tool: snapshot)
    vendor_command = MagicMock(side_effect=AssertionError("Linux probe executed"))
    monkeypatch.setattr(hw, "_run_tool", vendor_command)

    monitor = hw.GPUMonitor()

    assert monitor.gpu_type == "AMD"
    assert monitor.gpu_tool == hw.WINDOWS_NATIVE_TOOL
    assert monitor.native_device_id == snapshot.devices[0].id
    assert monitor.get_total_vram_gb() == 8
    assert monitor.get_vram_usage() == "2048"
    vendor_command.assert_not_called()


def test_amd_shared_budget_is_limited_by_free_system_ram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared-memory aperture is not mistaken for additional physical RAM."""
    import tools.hardware_monitor as hw

    snapshot = win.parse_snapshot(json.dumps(snapshot_data()))
    monkeypatch.setattr(hw, "IS_WINDOWS", True)
    monkeypatch.setattr(hw, "query_windows_snapshot", lambda _tool: snapshot)
    memory = MagicMock()
    memory.available = 3 * 1024**3
    monkeypatch.setattr(hw.psutil, "virtual_memory", lambda: memory)
    monitor = hw.GPUMonitor()

    assert monitor.get_available_windows_memory(use_shared=False) == 6
    assert monitor.get_available_windows_memory(use_shared=True) == 9
    assert monitor.shared_memory_info == {"total": 16, "used": 0, "free": 16}


def test_nvidia_offload_keeps_the_dedicated_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Showing shared WDDM usage must not assume CUDA can use its full budget."""
    import tools.hardware_monitor as hw

    snapshot = win.parse_snapshot(json.dumps(snapshot_data([device_data("NVIDIA")])))
    monkeypatch.setattr(hw, "IS_WINDOWS", True)
    monkeypatch.setattr(hw, "query_windows_snapshot", lambda _tool: snapshot)
    monitor = hw.GPUMonitor()
    assert monitor.get_available_windows_memory(use_shared=True) == 6


def test_amd_measurements_reach_statistics_sqlite_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real backend-shaped AMD data flows through existing result storage."""
    from cli.benchmark import BenchmarkCache, BenchmarkResult
    import tools.hardware_monitor as hw

    payload = snapshot_data()
    payload["devices"][0]["metrics"]["power_watts"] = None
    payload["devices"][0]["status"]["power_watts"] = "unsupported"
    snapshot = win.parse_snapshot(json.dumps(payload))
    monkeypatch.setattr(hw, "IS_WINDOWS", True)
    monkeypatch.setattr(hw, "IS_MACOS", False)
    monkeypatch.setattr(hw, "query_windows_snapshot", lambda _tool: snapshot)
    sampler = MagicMock(spec=win.WindowsGpuSampler)
    sampler.running = True
    sampler.sample.return_value = snapshot
    sampler.latest_sample.return_value = snapshot
    monkeypatch.setattr(hw, "WindowsGpuSampler", lambda _tool: sampler)
    gpu = hw.GPUMonitor()
    monitor = hw.HardwareMonitor(
        gpu.gpu_type, gpu.gpu_tool, enabled=True, device_id=gpu.native_device_id
    )
    gpu.attach_windows_sampler(monitor.windows_sampler)
    monkeypatch.setattr(monitor, "_get_cpu_usage", lambda: 10.0)
    monkeypatch.setattr(monitor, "_get_ram_usage", lambda: 1.0)
    monkeypatch.setattr(
        hw.time, "sleep", lambda _seconds: setattr(monitor, "monitoring", False)
    )
    monitor.monitoring = True
    getattr(monitor, "_monitor_loop")()
    stats = monitor.stop()
    per_gpu_json = monitor.per_gpu_metrics_json()
    assert per_gpu_json is not None
    assert stats["temp_celsius_avg"] == 50
    assert stats["power_watts_avg"] is None
    assert stats["gtt_gb_avg"] == 0
    result = BenchmarkResult(
        model_name="amd-fixture", quantization="Q4", gpu_type=gpu.gpu_model,
        gpu_offload=1, vram_mb=gpu.get_vram_usage(),
        avg_tokens_per_sec=10, avg_ttft=0.1, avg_gen_time=1,
        prompt_tokens=10, completion_tokens=10, timestamp="2026-01-01T00:00:00",
        params_size="1B", architecture="test", max_context_length=2048,
        model_size_gb=1, has_vision=False, has_tools=False,
        tokens_per_sec_per_gb=10, tokens_per_sec_per_billion_params=10,
        inference_params_hash="native-fixture",
        temp_celsius_avg=stats["temp_celsius_avg"],
        power_watts_avg=stats["power_watts_avg"],
        gtt_gb_avg=stats["gtt_gb_avg"],
        gpu_count=stats["gpu_count"], gpu_metrics_json=per_gpu_json,
    )
    cache = BenchmarkCache(tmp_path / "results.db")
    cache.save_result(result, "amd-fixture", "params", "prompt", 2048)
    stored = cache.get_all_results()
    assert len(stored) == 1
    assert stored[0].temp_celsius_avg == 50
    assert stored[0].power_watts_avg is None
    assert stored[0].gtt_gb_avg == 0
    metadata = json.loads(stored[0].gpu_metrics_json)[0]["metadata"]
    assert metadata["sensor_status"]["power_watts"] == "unsupported"
    export = tmp_path / "results.json"
    cache.export_to_json(export)
    data = json.loads(export.read_text(encoding="utf-8"))
    assert data[0]["gtt_gb_avg"] == 0
    assert data[0]["power_watts_avg"] is None


def test_watch_reader_drains_frames_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python reader validates an AMD-shaped frame and reaps its own child."""
    process = _FakeWatch(json.dumps(snapshot_data()))
    monkeypatch.setattr(win, "probe_path", lambda: Path("gpu_probe.exe"))
    monkeypatch.setattr(win.subprocess, "Popen", lambda *_args, **_kwargs: process)
    sampler = win.WindowsGpuSampler()
    try:
        sampler.start()
        sampler.start()
        assert sampler.wait_ready(timeout=2)
        snapshot = sampler.sample()
        assert snapshot is not None
        assert snapshot.devices[0].vendor == "AMD"
        assert sampler.latest_sample() is not None
    finally:
        sampler.stop()
    assert not sampler.running
    assert not process.killed
    assert sampler.latest_sample() is None


def test_watch_rejects_malformed_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken stream invalidates the current sample and reports the failure."""
    process = _FakeWatch('{"invalid": true}')
    monkeypatch.setattr(win, "probe_path", lambda: Path("gpu_probe.exe"))
    monkeypatch.setattr(win.subprocess, "Popen", lambda *_args, **_kwargs: process)
    sampler = win.WindowsGpuSampler()
    try:
        sampler.start()
        assert not sampler.wait_ready(timeout=2)
        assert sampler.error is not None
        assert "Invalid native GPU response" in sampler.error
        assert sampler.sample() is None
    finally:
        sampler.stop()
    assert process.killed


def test_watch_force_stops_only_its_unresponsive_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked stop handshake escalates to the held process, not its name."""
    process = _FakeWatch(json.dumps(snapshot_data()), ignore_stop=True)
    monkeypatch.setattr(win, "probe_path", lambda: Path("gpu_probe.exe"))
    monkeypatch.setattr(win.subprocess, "Popen", lambda *_args, **_kwargs: process)
    sampler = win.WindowsGpuSampler()
    sampler.start()
    try:
        assert sampler.wait_ready(timeout=2)
    finally:
        sampler.stop()
    assert process.killed
    assert not sampler.running


def test_watch_missing_executable_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing native components do not masquerade as unsupported hardware."""
    def missing() -> Path:
        raise win.WindowsGpuError("Native component missing")

    monkeypatch.setattr(win, "probe_path", missing)
    sampler = win.WindowsGpuSampler()
    try:
        sampler.start()
        assert not sampler.wait_ready(timeout=2)
        assert sampler.error == "Native component missing"
    finally:
        sampler.stop()
