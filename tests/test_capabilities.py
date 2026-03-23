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

        assert result is not None, "Expected result, got None"

        assert isinstance(result, CapabilityDetectionResult), "Expected CapabilityDetectionResult instance"

        assert Capability.GENERAL_TEXT in result.capabilities, "Expected GENERAL_TEXT capability"

        assert Capability.REASONING in result.capabilities, "Expected REASONING capability"

        assert result.source == "cli_flags", f"Expected cli_flags source, got {result.source}"

        assert result.confidence == 1.0, f"Expected confidence 1.0, got {result.confidence}"

    def test_detect_from_flags_invalid(self):
        """Test detecting capabilities from invalid flags."""
        detector = CapabilityDetector()

        result = detector.detect_from_flags("invalid_capability")

        assert result is None, "Expected None for invalid capability"

    def test_detect_from_flags_none(self):
        """Test detecting capabilities with None input."""
        detector = CapabilityDetector()

        result = detector.detect_from_flags(None)

        assert result is None, "Expected None for None input"

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

        assert result is not None, "Expected result, got None"

        assert Capability.VISION in result.capabilities, "Expected VISION capability"

        assert Capability.TOOLING in result.capabilities, "Expected TOOLING capability"

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

        assert result is not None, "Expected result, got None"

        assert Capability.VISION in result.capabilities, "Expected VISION capability"

        assert Capability.GENERAL_TEXT in result.capabilities, "Expected GENERAL_TEXT capability"

    def test_detect_from_metadata_not_found(self):
        """Test detecting from non-existent metadata file."""
        detector = CapabilityDetector()

        result = detector.detect_from_metadata(Path("/nonexistent/metadata.json"))

        assert result is None, "Expected None for nonexistent file"

    def test_detect_from_model_name_vision(self):
        """Test detecting vision capability from model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("llava-vision-13b")

        assert result is not None, "Expected result, got None"

        assert Capability.VISION in result.capabilities, "Expected VISION capability"

        assert result.source == "model_name_heuristics", (
            f"Expected model_name_heuristics source, got {result.source}"
        )

    def test_detect_from_model_name_reasoning(self):
        """Test detecting reasoning capability from model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("deepthink-reasoning-7b")

        assert result is not None, "Expected result, got None"

        assert Capability.REASONING in result.capabilities, "Expected REASONING capability"

    def test_detect_from_model_name_tooling(self):
        """Test detecting tooling capability from model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("agent-toolformer-13b")

        assert result is not None, "Expected result, got None"

        assert Capability.TOOLING in result.capabilities, "Expected TOOLING capability"

    def test_detect_from_model_name_default(self):
        """Test detecting default capability from generic model name."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("generic-model-7b")

        assert result is not None, "Expected result, got None"

        assert Capability.GENERAL_TEXT in result.capabilities, "Expected GENERAL_TEXT capability"

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

        assert Capability.REASONING in result.capabilities, "Expected REASONING from flags"

        assert Capability.VISION not in result.capabilities, "Should not have VISION from metadata"

        assert result.source == "cli_flags", f"Expected cli_flags source, got {result.source}"

    def test_detect_default_capability(self):
        """Test fallback to default capability."""
        detector = CapabilityDetector()

        result = detector.detect()

        assert Capability.GENERAL_TEXT in result.capabilities, "Expected default GENERAL_TEXT capability"

        assert result.source == "default", f"Expected default source, got {result.source}"

        assert result.confidence == 1.0, f"Expected confidence 1.0, got {result.confidence}"

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

        assert result is not None, "Expected result from metadata DB"

        assert Capability.VISION in result.capabilities, "Expected VISION capability from DB"

        assert result.source == "metadata_db", f"Expected metadata_db source, got {result.source}"

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

        assert result is None, "Expected None for nonexistent model"

    def test_detect_from_metadata_db_not_exists(self):
        """Test detecting from nonexistent metadata database."""
        detector = CapabilityDetector()

        result = detector.detect_from_metadata_db(
            Path("/nonexistent/metadata.db"), "model"
        )

        assert result is None, "Expected None for nonexistent database"

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

        assert result is not None, "Expected result from metadata DB"

        assert Capability.TOOLING in result.capabilities, "Expected TOOLING capability from DB"

    def test_map_metadata_capability_reasoning(self):
        """Test mapping metadata capability names to REASONING."""
        detector = CapabilityDetector()

        for cap_name in ["reasoning", "cot", "chain_of_thought"]:
            result = detector._map_metadata_capability(cap_name)
            assert result == Capability.REASONING, f"Expected REASONING for {cap_name}"

    def test_map_metadata_capability_vision(self):
        """Test mapping metadata capability names to VISION."""
        detector = CapabilityDetector()

        for cap_name in ["vision", "image", "multimodal"]:
            result = detector._map_metadata_capability(cap_name)
            assert result == Capability.VISION, f"Expected VISION for {cap_name}"

    def test_map_metadata_capability_tooling(self):
        """Test mapping metadata capability names to TOOLING."""
        detector = CapabilityDetector()

        for cap_name in ["tooling", "tool_use", "tool-use", "function_calling"]:
            result = detector._map_metadata_capability(cap_name)
            assert result == Capability.TOOLING, f"Expected TOOLING for {cap_name}"

    def test_map_metadata_capability_unknown(self):
        """Test mapping unknown capability names returns None."""
        detector = CapabilityDetector()

        result = detector._map_metadata_capability("unknown_capability")
        assert result is None, "Expected None for unknown capability"

    def test_capabilities_from_metadata_function_calling(self):
        """Test extracting function_calling flag from metadata."""
        detector = CapabilityDetector()

        metadata = {"function_calling": True}
        result = detector._capabilities_from_metadata(metadata)

        assert Capability.TOOLING in result, "Expected TOOLING from function_calling flag"

    def test_map_metadata_capability_general_text_aliases(self):
        """General-text aliases map to Capability.GENERAL_TEXT."""
        detector = CapabilityDetector()

        for cap_name in ["general_text", "chat", "coding", "math"]:
            mapped = detector._map_metadata_capability(cap_name)
            assert mapped == Capability.GENERAL_TEXT, (
                f"Expected GENERAL_TEXT for {cap_name}"
            )

    def test_capabilities_from_metadata_skips_non_string_caps(self):
        """Non-string capability list entries are ignored safely."""
        detector = CapabilityDetector()

        metadata = {"capabilities": ["vision", 123, None]}
        result = detector._capabilities_from_metadata(metadata)

        assert Capability.VISION in result, "Expected VISION from string capability"

    def test_capabilities_from_metadata_reasoning_from_name(self):
        """Reasoning keyword in model name adds reasoning capability."""
        detector = CapabilityDetector()

        metadata = {"name": "my-chain-of-thought-model"}
        result = detector._capabilities_from_metadata(metadata)

        assert Capability.REASONING in result, "Expected REASONING from metadata name"

    def test_detect_from_metadata_db_no_candidates(self, tmp_path):
        """DB detection returns None when no candidate names are provided."""
        detector = CapabilityDetector()
        db_path = tmp_path / "metadata.db"
        db_path.write_text("", encoding="utf-8")

        result = detector.detect_from_metadata_db(db_path, None, None)
        assert result is None, "Expected None without candidate model names"

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
        assert result is not None, "Expected result despite invalid JSON payload"
        assert Capability.VISION in result.capabilities, "Expected VISION from DB vision flag"

    def test_detect_from_metadata_invalid_json_file(self, tmp_path):
        """Invalid metadata JSON file returns None."""
        detector = CapabilityDetector()
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text("{invalid json", encoding="utf-8")

        result = detector.detect_from_metadata(metadata_file)
        assert result is None, "Expected None for invalid metadata JSON"

    def test_detect_from_metadata_os_error(self, tmp_path):
        """I/O errors while reading metadata return None."""
        detector = CapabilityDetector()
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text("{}", encoding="utf-8")

        with patch("builtins.open", side_effect=OSError("read failed")):
            result = detector.detect_from_metadata(metadata_file)
        assert result is None, "Expected None for unreadable metadata path"

    def test_detect_from_model_name_empty_returns_none(self):
        """Empty model names return None in heuristic detector."""
        detector = CapabilityDetector()

        result = detector.detect_from_model_name("")
        assert result is None, "Expected None for empty model name"

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
        assert result.source == "metadata_db", f"Expected metadata_db source, got {result.source}"

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
        assert result.source == "metadata_file", (
            f"Expected metadata_file source, got {result.source}"
        )

    def test_detect_prefers_model_name_when_no_other_sources(self):
        """detect() uses model-name heuristics before default fallback."""
        detector = CapabilityDetector()

        result = detector.detect(model_name="vision-model-x")
        assert result.source == "model_name_heuristics", (
            f"Expected heuristics source, got {result.source}"
        )
        assert Capability.VISION in result.capabilities, "Expected VISION from model name"


def test_get_capability_tests():
    """Test getting test types for capabilities."""
    general_tests = get_capability_tests(Capability.GENERAL_TEXT)
    assert general_tests, "Expected tests for GENERAL_TEXT"

    assert "text_generation" in general_tests, "Expected text_generation test"

    reasoning_tests = get_capability_tests(Capability.REASONING)
    assert reasoning_tests, "Expected tests for REASONING"

    assert "logical_reasoning" in reasoning_tests, "Expected logical_reasoning test"

    vision_tests = get_capability_tests(Capability.VISION)
    assert vision_tests, "Expected tests for VISION"

    assert "vqa" in vision_tests, "Expected vqa test"

    tooling_tests = get_capability_tests(Capability.TOOLING)
    assert tooling_tests, "Expected tests for TOOLING"

    assert "function_calling" in tooling_tests, "Expected function_calling test"


def test_get_capability_tests_unknown_defaults_to_general():
    """Unknown capabilities fall back to a general test list."""
    tests = get_capability_tests(cast(Capability, "unknown"))
    assert tests == ["general"], "Expected ['general'] fallback for unknown capability"
