"""
Unit tests for capability detection module.
"""

import json
from pathlib import Path
import sqlite3
from typing import cast
from unittest.mock import patch

from agents.capabilities import (
    Capability,
    CapabilityDetectionResult,
    CapabilityDetector,
    get_capability_tests,
)


class TestCapabilityDetector:
    """
    Tests for CapabilityDetector class.
    """

    def test_detect_from_flags_valid(self):
        """Test detecting capabilities from valid CLI flags."""
        detector = CapabilityDetector()

        result = detector.detect_from_flags("general_text,reasoning")

        if result is None:
            raise AssertionError("Expected result, got None")

        if not isinstance(result, CapabilityDetectionResult):
            raise AssertionError(
                "Expected CapabilityDetectionResult instance"
            )

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected GENERAL_TEXT capability")

        if Capability.REASONING not in result.capabilities:
            raise AssertionError("Expected REASONING capability")

        if result.source != "cli_flags":
            raise AssertionError(f"Expected cli_flags source, got {result.source}")

        if result.confidence != 1.0:
            raise AssertionError(f"Expected confidence 1.0, got {result.confidence}")

    def test_detect_from_flags_invalid(self):
        """Test detecting capabilities from invalid flags."""
        detector = CapabilityDetector()

        result = detector.detect_from_flags("invalid_capability")

        if result is not None:
            raise AssertionError("Expected None for invalid capability")

    def test_detect_from_flags_none(self):
        """Test detecting capabilities with None input."""
        detector = CapabilityDetector()

        result = detector.detect_from_flags(None)

        if result is not None:
            raise AssertionError("Expected None for None input")

    def test_detect_from_metadata_with_capabilities(self, tmp_path):
        """Test detecting from metadata file with capabilities field."""
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
        """Test detecting from metadata with modalities field."""
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
        """Test detecting from non-existent metadata file."""
        detector = CapabilityDetector()

        result = detector.detect_from_metadata(Path("/nonexistent/metadata.json"))

        if result is not None:
            raise AssertionError("Expected None for nonexistent file")

    def test_detect_from_model_name_vision(self):
        """Test detecting vision capability from model name."""
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
        """Test detecting reasoning capability from model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("deepthink-reasoning-7b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.REASONING not in result.capabilities:
            raise AssertionError("Expected REASONING capability")

    def test_detect_from_model_name_tooling(self):
        """Test detecting tooling capability from model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("agent-toolformer-13b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.TOOLING not in result.capabilities:
            raise AssertionError("Expected TOOLING capability")

    def test_detect_from_model_name_default(self):
        """Test detecting default capability from generic model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("generic-model-7b")

        if result is None:
            raise AssertionError("Expected result, got None")

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected GENERAL_TEXT capability")

    def test_detect_priority_flags_over_metadata(self, tmp_path):
        """Test that CLI flags take priority over metadata."""
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
        """Test fallback to default capability."""
        detector = CapabilityDetector()

        result = detector.detect()

        if Capability.GENERAL_TEXT not in result.capabilities:
            raise AssertionError("Expected default GENERAL_TEXT capability")

        if result.source != "default":
            raise AssertionError(f"Expected default source, got {result.source}")

        if result.confidence != 1.0:
            raise AssertionError(f"Expected confidence 1.0, got {result.confidence}")

    def test_detect_from_metadata_db_found(self, tmp_path):
        """Test detecting from metadata database when model exists."""
        detector = CapabilityDetector()

        db_path = tmp_path / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE model_metadata (
                model_key TEXT,
                display_name TEXT,
                vision INTEGER,
                tool_use INTEGER,
                capabilities TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO model_metadata VALUES (
                'llava-7b',
                'LLaVA 7B',
                1,
                0,
                '["vision"]'
            )
        """)

        conn.commit()
        conn.close()

        result = detector.detect_from_metadata_db(
            db_path, "llava-7b"
        )

        if result is None:
            raise AssertionError("Expected result from metadata DB")

        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION capability from DB")

        if result.source != "metadata_db":
            raise AssertionError(f"Expected metadata_db source, got {result.source}")

    def test_detect_from_metadata_db_not_found(self, tmp_path):
        """Test detecting from metadata database when model not found."""
        detector = CapabilityDetector()

        db_path = tmp_path / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE model_metadata (
                model_key TEXT,
                display_name TEXT,
                vision INTEGER,
                tool_use INTEGER,
                capabilities TEXT
            )
        """)

        conn.commit()
        conn.close()

        result = detector.detect_from_metadata_db(
            db_path, "nonexistent-model"
        )

        if result is not None:
            raise AssertionError("Expected None for nonexistent model")

    def test_detect_from_metadata_db_not_exists(self):
        """Test detecting from nonexistent metadata database."""
        detector = CapabilityDetector()

        result = detector.detect_from_metadata_db(
            Path("/nonexistent/metadata.db"), "model"
        )

        if result is not None:
            raise AssertionError("Expected None for nonexistent database")

    def test_detect_from_metadata_db_with_tooling(self, tmp_path):
        """Test detecting tooling capability from metadata DB."""
        detector = CapabilityDetector()

        db_path = tmp_path / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE model_metadata (
                model_key TEXT,
                display_name TEXT,
                vision INTEGER,
                tool_use INTEGER,
                capabilities TEXT
            )
        """)

        cursor.execute("""
            INSERT INTO model_metadata VALUES (
                'agent-7b',
                'Agent Model',
                0,
                1,
                NULL
            )
        """)

        conn.commit()
        conn.close()

        result = detector.detect_from_metadata_db(
            db_path, "agent-7b"
        )

        if result is None:
            raise AssertionError("Expected result from metadata DB")

        if Capability.TOOLING not in result.capabilities:
            raise AssertionError("Expected TOOLING capability from DB")

    def test_map_metadata_capability_reasoning(self):
        """Test mapping metadata capability names to REASONING."""
        detector = CapabilityDetector()

        for cap_name in ["reasoning", "cot", "chain_of_thought"]:
            result = detector._map_metadata_capability(cap_name)
            if result != Capability.REASONING:
                raise AssertionError(f"Expected REASONING for {cap_name}")

    def test_map_metadata_capability_vision(self):
        """Test mapping metadata capability names to VISION."""
        detector = CapabilityDetector()

        for cap_name in ["vision", "image", "multimodal"]:
            result = detector._map_metadata_capability(cap_name)
            if result != Capability.VISION:
                raise AssertionError(f"Expected VISION for {cap_name}")

    def test_map_metadata_capability_tooling(self):
        """Test mapping metadata capability names to TOOLING."""
        detector = CapabilityDetector()

        for cap_name in ["tooling", "tool_use", "tool-use", "function_calling"]:
            result = detector._map_metadata_capability(cap_name)
            if result != Capability.TOOLING:
                raise AssertionError(f"Expected TOOLING for {cap_name}")

    def test_map_metadata_capability_unknown(self):
        """Test mapping unknown capability names returns None."""
        detector = CapabilityDetector()

        result = detector._map_metadata_capability("unknown_capability")
        if result is not None:
            raise AssertionError("Expected None for unknown capability")

    def test_capabilities_from_metadata_function_calling(self):
        """Test extracting function_calling flag from metadata."""
        detector = CapabilityDetector()

        metadata = {"function_calling": True}
        result = detector._capabilities_from_metadata(metadata)

        if Capability.TOOLING not in result:
            raise AssertionError("Expected TOOLING from function_calling flag")

    def test_map_metadata_capability_general_text_aliases(self):
        """General-text aliases map to Capability.GENERAL_TEXT."""
        detector = CapabilityDetector()

        for cap_name in ["general_text", "chat", "coding", "math"]:
            mapped = detector._map_metadata_capability(cap_name)
            if mapped != Capability.GENERAL_TEXT:
                raise AssertionError(f"Expected GENERAL_TEXT for {cap_name}")

    def test_capabilities_from_metadata_skips_non_string_caps(self):
        """Non-string capability list entries are ignored safely."""
        detector = CapabilityDetector()

        metadata = {"capabilities": ["vision", 123, None]}
        result = detector._capabilities_from_metadata(metadata)

        if Capability.VISION not in result:
            raise AssertionError("Expected VISION from string capability")

    def test_capabilities_from_metadata_reasoning_from_name(self):
        """Reasoning keyword in model name adds reasoning capability."""
        detector = CapabilityDetector()

        metadata = {"name": "my-chain-of-thought-model"}
        result = detector._capabilities_from_metadata(metadata)

        if Capability.REASONING not in result:
            raise AssertionError("Expected REASONING from metadata name")

    def test_detect_from_metadata_db_no_candidates(self, tmp_path):
        """DB detection returns None when no candidate names are provided."""
        detector = CapabilityDetector()
        db_path = tmp_path / "metadata.db"
        db_path.write_text("", encoding="utf-8")

        result = detector.detect_from_metadata_db(db_path, None, None)
        if result is not None:
            raise AssertionError("Expected None without candidate model names")

    def test_detect_from_metadata_db_invalid_caps_json(self, tmp_path):
        """Invalid capabilities JSON in DB row is handled gracefully."""
        detector = CapabilityDetector()

        db_path = tmp_path / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE model_metadata (
                model_key TEXT,
                display_name TEXT,
                vision INTEGER,
                tool_use INTEGER,
                capabilities TEXT
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO model_metadata VALUES (
                'model-key',
                'Model Display',
                1,
                0,
                '{not-json}'
            )
            """
        )
        conn.commit()
        conn.close()

        result = detector.detect_from_metadata_db(db_path, "model-key")
        if result is None:
            raise AssertionError("Expected result despite invalid JSON payload")
        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION from DB vision flag")

    def test_detect_from_metadata_invalid_json_file(self, tmp_path):
        """Invalid metadata JSON file returns None."""
        detector = CapabilityDetector()
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text("{invalid json", encoding="utf-8")

        result = detector.detect_from_metadata(metadata_file)
        if result is not None:
            raise AssertionError("Expected None for invalid metadata JSON")

    def test_detect_from_metadata_os_error(self, tmp_path):
        """I/O errors while reading metadata return None."""
        detector = CapabilityDetector()
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text("{}", encoding="utf-8")

        with patch("builtins.open", side_effect=OSError("read failed")):
            result = detector.detect_from_metadata(metadata_file)
        if result is not None:
            raise AssertionError("Expected None for unreadable metadata path")

    def test_detect_from_model_name_empty_returns_none(self):
        """Empty model names return None in heuristic detector."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("")
        if result is not None:
            raise AssertionError("Expected None for empty model name")

    def test_detect_prefers_metadata_db_when_flags_absent(self, tmp_path):
        """detect() returns metadata_db result before heuristics."""
        detector = CapabilityDetector()

        db_path = tmp_path / "metadata.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE model_metadata (
                model_key TEXT,
                display_name TEXT,
                vision INTEGER,
                tool_use INTEGER,
                capabilities TEXT
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO model_metadata VALUES (
                'llava-7b',
                'LLaVA 7B',
                1,
                0,
                '["vision"]'
            )
            """
        )
        conn.commit()
        conn.close()

        result = detector.detect(
            model_name="llava-7b",
            metadata_db_path=db_path,
            metadata_path=None,
            capabilities_str=None,
        )
        if result.source != "metadata_db":
            raise AssertionError(f"Expected metadata_db source, got {result.source}")

    def test_detect_prefers_metadata_file_when_db_missing(self, tmp_path):
        """detect() falls back to metadata file when DB has no match."""
        detector = CapabilityDetector()
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(
            json.dumps({"capabilities": ["tooling"]}),
            encoding="utf-8",
        )

        result = detector.detect(
            model_name="x",
            metadata_db_path=tmp_path / "no.db",
            metadata_path=metadata_file,
            capabilities_str=None,
        )
        if result.source != "metadata_file":
            raise AssertionError(
                f"Expected metadata_file source, got {result.source}"
            )

    def test_detect_prefers_model_name_when_no_other_sources(self):
        """detect() uses model-name heuristics before default fallback."""
        detector = CapabilityDetector()

        result = detector.detect(model_name="vision-model-x")
        if result.source != "model_name_heuristics":
            raise AssertionError(
                f"Expected heuristics source, got {result.source}"
            )
        if Capability.VISION not in result.capabilities:
            raise AssertionError("Expected VISION from model name")


def test_get_capability_tests():
    """Test getting test types for capabilities."""
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


def test_get_capability_tests_unknown_defaults_to_general():
    """Unknown capabilities fall back to a general test list."""
    tests = get_capability_tests(cast(Capability, "unknown"))
    if tests != ["general"]:
        raise AssertionError("Expected ['general'] fallback for unknown capability")
