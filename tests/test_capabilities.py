"""
Unit tests for capability detection module.
"""

import json
from pathlib import Path

from agents.capabilities import (
    Capability,
    CapabilityDetectionResult,
    CapabilityDetector,
    get_capability_tests
)


class TestCapabilityDetector:
    """
    Tests for CapabilityDetector class.
    """

    def test_detect_from_flags_valid(self):
        """
        Test detecting capabilities from valid CLI flags.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_flags("general_text,reasoning")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected GENERAL_TEXT capability")

        if Capability.REASONING not in result.capabilities:
            raise AssertionError("Expected REASONING capability")

        if result.source != "cli_flags":
            raise AssertionError(f"Expected cli_flags source, got {result.source}")

        if result.confidence != 1.0:
            raise AssertionError(f"Expected confidence 1.0, got {result.confidence}")

    def test_detect_from_flags_invalid(self):
        """
        Test detecting capabilities from invalid flags.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_flags("invalid_capability")

        if result is not None:
            raise AssertionError("Expected None for invalid capability")

    def test_detect_from_flags_none(self):
        """
        Test detecting capabilities with None input.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_flags(None)

        if result is not None:
            raise AssertionError("Expected None for None input")

    def test_detect_from_metadata_with_capabilities(self, tmp_path):
        """
        Test detecting from metadata file with capabilities field.
        """
        detector = CapabilityDetector()

        metadata = {
            "name": "test-model",
            "capabilities": ["vision", "tooling"]
        }

        metadata_file = tmp_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        result = detector.detect_from_metadata(metadata_file)

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION capability")

        if Capability.TOOLING not in result.capabilities:
            raise AssertionError("Expected TOOLING capability")

    def test_detect_from_metadata_with_modalities(self, tmp_path):
        """
        Test detecting from metadata with modalities field.
        """
        detector = CapabilityDetector()

        metadata = {
            "name": "test-model",
            "modalities": ["vision", "text"]
        }

        metadata_file = tmp_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        result = detector.detect_from_metadata(metadata_file)

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION capability")

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected GENERAL_TEXT capability")

    def test_detect_from_metadata_not_found(self):
        """
        Test detecting from non-existent metadata file.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_metadata(Path("/nonexistent/metadata.json"))

        if result is not None:
            raise AssertionError("Expected None for nonexistent file")

    def test_detect_from_model_name_vision(self):
        """
        Test detecting vision capability from model name.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("llava-vision-13b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION capability")

        if result.source != "model_name_heuristics":
            raise AssertionError(
                f"Expected model_name_heuristics source, got {result.source}"
            )

    def test_detect_from_model_name_reasoning(self):
        """
        Test detecting reasoning capability from model name.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("deepthink-reasoning-7b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.REASONING not in result.capabilities:
            raise AssertionError("Expected REASONING capability")

    def test_detect_from_model_name_tooling(self):
        """
        Test detecting tooling capability from model name.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("agent-toolformer-13b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.TOOLING not in result.capabilities:
            raise AssertionError("Expected TOOLING capability")

    def test_detect_from_model_name_default(self):
        """
        Test detecting default capability from generic model name.
        """
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("generic-model-7b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected GENERAL_TEXT capability")

    def test_detect_priority_flags_over_metadata(self, tmp_path):
        """
        Test that CLI flags take priority over metadata.
        """
        detector = CapabilityDetector()

        metadata = {
            "name": "test-model",
            "capabilities": ["vision"]
        }

        metadata_file = tmp_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        result = detector.detect(
            model_name="test-model",
            metadata_path=metadata_file,
            capabilities_str="reasoning"
        )

        if Capability.REASONING not in result.capabilities:
            raise AssertionError("Expected REASONING from flags")

        if Capability.VISION in result.capabilities:
            raise AssertionError("Should not have VISION from metadata")

        if result.source != "cli_flags":
            raise AssertionError(f"Expected cli_flags source, got {result.source}")

    def test_detect_default_capability(self):
        """
        Test fallback to default capability.
        """
        detector = CapabilityDetector()

        result = detector.detect()

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected default GENERAL_TEXT capability")

        if result.source != "default":
            raise AssertionError(f"Expected default source, got {result.source}")

        if result.confidence != 1.0:
            raise AssertionError(f"Expected confidence 1.0, got {result.confidence}")


def test_get_capability_tests():
    """
    Test getting test types for capabilities.
    """
    general_tests = get_capability_tests(Capability.GENERAL_TEXT)
    if not general_tests:
        raise AssertionError("Expected tests for GENERAL_TEXT")

    if "text_generation" not in general_tests:
        raise AssertionError("Expected text_generation test")

    reasoning_tests = get_capability_tests(Capability.REASONING)
    if not reasoning_tests:
        raise AssertionError("Expected tests for REASONING")

    if "logical_reasoning" not in reasoning_tests:
        raise AssertionError("Expected logical_reasoning test")

    vision_tests = get_capability_tests(Capability.VISION)
    if not vision_tests:
        raise AssertionError("Expected tests for VISION")

    if "vqa" not in vision_tests:
        raise AssertionError("Expected vqa test")

    tooling_tests = get_capability_tests(Capability.TOOLING)
    if not tooling_tests:
        raise AssertionError("Expected tests for TOOLING")

    if "function_calling" not in tooling_tests:
        raise AssertionError("Expected function_calling test")
