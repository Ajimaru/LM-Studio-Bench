"""Tests for tools/gpu_devices.py - per-device GPU sampling."""

import json
from unittest.mock import patch

from tools.gpu_devices import (
    GpuAggregate,
    GpuSample,
    detect_device_count,
    device_names,
    sample_devices,
    serialize_aggregates,
)

ROCM_TEMP = """
GPU[0]		: Temperature (Sensor edge) (C): 62.0
GPU[1]		: Temperature (Sensor edge) (C): 45.0
"""

ROCM_POWER = """
GPU[0]		: Average Graphics Package Power (W): 104.0
GPU[1]		: Average Graphics Package Power (W): 24.0
"""

ROCM_VRAM = """
GPU[0]		: VRAM Total Memory (B): 8573157376
GPU[0]		: VRAM Total Used Memory (B): 6356467712
GPU[1]		: VRAM Total Memory (B): 17179869184
GPU[1]		: VRAM Total Used Memory (B): 294842368
"""

ROCM_GTT = """
GPU[0]		: GTT Total Memory (B): 40752824320
GPU[0]		: GTT Total Used Memory (B): 794568704
GPU[1]		: GTT Total Memory (B): 40752824320
GPU[1]		: GTT Total Used Memory (B): 109166592
"""

ROCM_NAMES = """
GPU[0]		: Card Series: 		AMD Radeon RX 7600M XT
GPU[1]		: Card Series: 		AMD Radeon Graphics
"""

NVIDIA_CSV = (
    "0, NVIDIA GeForce RTX 4090, 71, 320.15, 18000\n"
    "1, NVIDIA GeForce RTX 3060, 55, 120.00, 4096\n"
)


def _rocm_outputs(command):
    """Return the canned rocm-smi output for a given command."""
    if "--showtemp" in command:
        return ROCM_TEMP
    if "--showpower" in command:
        return ROCM_POWER
    if "vram" in command:
        return ROCM_VRAM
    if "gtt" in command:
        return ROCM_GTT
    if "--showproductname" in command:
        return ROCM_NAMES
    return ""


class TestSampleRocm:
    """Sampling AMD devices through rocm-smi."""

    def test_samples_every_device(self):
        """Both GPUs are returned with their own readings."""
        with patch("tools.gpu_devices._run", side_effect=_rocm_outputs):
            samples = sample_devices("AMD", "rocm-smi")

        assert [s.index for s in samples] == [0, 1]
        assert samples[0].temp_celsius == 62.0
        assert samples[1].temp_celsius == 45.0
        assert samples[0].power_watts == 104.0
        assert round(samples[0].vram_gb, 2) == 5.92
        assert round(samples[1].vram_gb, 2) == 0.27
        assert round(samples[0].gtt_gb, 2) == 0.74

    def test_does_not_confuse_total_with_used(self):
        """"Total Memory" lines must not be read as usage."""
        with patch("tools.gpu_devices._run", side_effect=_rocm_outputs):
            samples = sample_devices("AMD", "rocm-smi")

        # 8573157376 B total vs 6356467712 B used - picking the wrong line
        # would report 7.98 GB here.
        assert round(samples[0].vram_gb, 2) == 5.92

    def test_device_names(self):
        """Card series names are mapped to their index."""
        with patch("tools.gpu_devices._run", side_effect=_rocm_outputs):
            names = device_names("AMD", "rocm-smi")
        assert names == {
            0: "AMD Radeon RX 7600M XT",
            1: "AMD Radeon Graphics",
        }

    def test_device_count(self):
        """Counting devices does not need a separate query path."""
        with patch("tools.gpu_devices._run", side_effect=_rocm_outputs):
            assert detect_device_count("AMD", "rocm-smi") == 2

    def test_tool_failure_yields_no_samples(self):
        """A failing vendor tool degrades to "no per-device data"."""
        with patch("tools.gpu_devices._run", return_value=None):
            assert sample_devices("AMD", "rocm-smi") == []


class TestSampleNvidia:
    """Sampling NVIDIA devices through nvidia-smi."""

    def test_parses_csv(self):
        """Index, name, temperature, power and memory are read."""
        with patch("tools.gpu_devices._run", return_value=NVIDIA_CSV):
            samples = sample_devices("NVIDIA", "nvidia-smi")

        assert len(samples) == 2
        assert samples[0].name == "NVIDIA GeForce RTX 4090"
        assert samples[0].temp_celsius == 71.0
        assert samples[0].power_watts == 320.15
        assert round(samples[0].vram_gb, 2) == 17.58
        assert samples[1].index == 1

    def test_malformed_rows_are_skipped(self):
        """A truncated line must not abort the whole sample."""
        with patch("tools.gpu_devices._run", return_value="broken\n" + NVIDIA_CSV):
            samples = sample_devices("NVIDIA", "nvidia-smi")
        assert len(samples) == 2


class TestUnsupportedBackends:
    """Backends without a device index stay on the aggregate readers."""

    def test_sysfs_returns_nothing(self):
        """sysfs has no stable per-device index here."""
        assert sample_devices("AMD", "sysfs") == []

    def test_apple_returns_nothing(self):
        """Apple Silicon has one GPU by definition."""
        assert sample_devices("Apple", "macmon") == []

    def test_missing_tool_returns_nothing(self):
        """Without a tool there is nothing to query."""
        assert sample_devices("AMD", None) == []


class TestAggregation:
    """Folding samples into min/max/avg per device."""

    def test_aggregates_metrics(self):
        """Each metric keeps its own min, max and average."""
        aggregate = GpuAggregate(index=0, name="Test GPU")
        for temp in (60.0, 70.0, 65.0):
            aggregate.add(GpuSample(index=0, temp_celsius=temp))

        data = aggregate.as_dict()
        assert data["samples"] == 3
        assert data["temp_celsius_min"] == 60.0
        assert data["temp_celsius_max"] == 70.0
        assert data["temp_celsius_avg"] == 65.0

    def test_missing_metrics_are_omitted(self):
        """A metric the tool never reported does not appear as zero."""
        aggregate = GpuAggregate(index=0)
        aggregate.add(GpuSample(index=0, temp_celsius=50.0))
        data = aggregate.as_dict()
        assert "power_watts_max" not in data

    def test_name_is_filled_from_first_sample_that_has_one(self):
        """The name may only arrive with a later sample."""
        aggregate = GpuAggregate(index=1)
        aggregate.add(GpuSample(index=1, temp_celsius=40.0))
        aggregate.add(GpuSample(index=1, name="Late Name", temp_celsius=41.0))
        assert aggregate.as_dict()["name"] == "Late Name"

    def test_serialization_round_trip(self):
        """Aggregates survive the trip through the TEXT column."""
        aggregate = GpuAggregate(index=0, name="GPU A")
        aggregate.add(GpuSample(index=0, vram_gb=5.9, temp_celsius=60.0))
        payload = serialize_aggregates([aggregate])
        decoded = json.loads(payload)
        assert decoded[0]["index"] == 0
        assert decoded[0]["vram_gb_max"] == 5.9

    def test_empty_aggregates_serialize_to_none(self):
        """Nothing measured means nothing stored, not "[]"."""
        assert serialize_aggregates([]) is None
        assert serialize_aggregates([GpuAggregate(index=0)]) is None
