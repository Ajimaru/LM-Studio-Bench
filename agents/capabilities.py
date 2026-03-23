"""
Capability detection module for LM Studio Bench.

Detects model capabilities from metadata or user-specified flags.
Supports: general_text, reasoning, vision, tooling.
"""

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class Capability(str, Enum):
    """
    Enumeration of supported model capabilities.
    """
    GENERAL_TEXT = "general_text"
    REASONING = "reasoning"
    VISION = "vision"
    TOOLING = "tooling"


@dataclass
class CapabilityDetectionResult:
    """
    Result of capability detection for a model.

    Attributes:
        capabilities: Set of detected capabilities
        source: Source of detection (metadata, flags, default)
        confidence: Confidence level (0.0 to 1.0)
        metadata: Additional metadata used for detection
    """
    capabilities: Set[Capability]
    source: str
    confidence: float
    metadata: Optional[Dict] = None


class CapabilityDetector:
    """
    Detects model capabilities from various sources.
    """

    def __init__(self, default_capability: Capability = Capability.GENERAL_TEXT):
        """
        Initialize capability detector.

        Args:
            default_capability: Default capability if none detected
        """
        self.default_capability = default_capability

    def _map_metadata_capability(self, capability_name: str) -> Optional[Capability]:
        """Map free-form capability names to benchmark capabilities."""
        cap = capability_name.strip().lower()
        if cap in {"reasoning", "cot", "chain_of_thought"}:
            return Capability.REASONING
        if cap in {"vision", "image", "multimodal"}:
            return Capability.VISION
        if cap in {
            "tooling",
            "tool_use",
            "tool-use",
            "tools",
            "function_calling",
            "function-calling",
        }:
            return Capability.TOOLING
        if cap in {
            "general_text",
            "chat",
            "coding",
            "creative",
            "math",
        }:
            return Capability.GENERAL_TEXT
        return None

    def _capabilities_from_metadata(self, metadata: Dict) -> Set[Capability]:
        """Extract benchmark capabilities from a metadata dictionary."""
        capabilities: Set[Capability] = set()

        caps_field = metadata.get("capabilities")
        if isinstance(caps_field, list):
            for cap_name in caps_field:
                if not isinstance(cap_name, str):
                    continue
                mapped = self._map_metadata_capability(cap_name)
                if mapped is not None:
                    capabilities.add(mapped)

        modalities = metadata.get("modalities")
        if isinstance(modalities, list):
            lowered_modalities = {
                str(modality).strip().lower() for modality in modalities
            }
            if "vision" in lowered_modalities or "image" in lowered_modalities:
                capabilities.add(Capability.VISION)
            if "text" in lowered_modalities:
                capabilities.add(Capability.GENERAL_TEXT)

        if metadata.get("function_calling"):
            capabilities.add(Capability.TOOLING)

        if metadata.get("tools") or metadata.get("tool_use"):
            capabilities.add(Capability.TOOLING)

        model_name = str(metadata.get("name", "")).lower()
        if any(
            kw in model_name for kw in ["reasoning", "cot", "chain-of-thought"]
        ):
            capabilities.add(Capability.REASONING)

        return capabilities

    def detect_from_metadata_db(
        self,
        metadata_db_path: Path,
        model_name: Optional[str],
        model_path: Optional[str] = None,
    ) -> Optional[CapabilityDetectionResult]:
        """Detect capabilities from the scraped metadata SQLite database."""
        if not metadata_db_path.exists():
            logger.debug("Metadata DB not found: %s", metadata_db_path)
            return None

        candidates: List[str] = []
        for value in [model_name, model_path]:
            if not value:
                continue
            text = str(value).strip()
            if text and text not in candidates:
                candidates.append(text)
            path_name = Path(text).name
            if path_name and path_name not in candidates:
                candidates.append(path_name)

        if not candidates:
            return None

        try:
            conn = sqlite3.connect(metadata_db_path)
            conn.row_factory = sqlite3.Row

            row: Optional[sqlite3.Row] = None
            for candidate in candidates:
                query = (
                    "SELECT model_key, display_name, vision, tool_use, capabilities "
                    "FROM model_metadata "
                    "WHERE lower(model_key)=lower(?) OR lower(display_name)=lower(?)"
                )
                row = conn.execute(query, (candidate, candidate)).fetchone()
                if row:
                    break

            conn.close()

            if not row:
                return None

            metadata: Dict = {
                "model_key": row["model_key"],
                "display_name": row["display_name"],
                "vision": bool(row["vision"]),
                "tool_use": bool(row["tool_use"]),
            }

            raw_caps = row["capabilities"]
            if raw_caps:
                try:
                    metadata["capabilities"] = json.loads(raw_caps)
                except json.JSONDecodeError:
                    logger.debug("Invalid capabilities JSON in metadata DB row")

            capabilities = self._capabilities_from_metadata(metadata)
            if metadata["vision"]:
                capabilities.add(Capability.VISION)
            if metadata["tool_use"]:
                capabilities.add(Capability.TOOLING)

            if not capabilities:
                return None

            return CapabilityDetectionResult(
                capabilities=capabilities,
                source="metadata_db",
                confidence=0.95,
                metadata={
                    "db_path": str(metadata_db_path),
                    "row": metadata,
                },
            )

        except sqlite3.Error as exc:
            logger.debug("Could not read metadata DB %s: %s", metadata_db_path, exc)
            return None

    def detect_from_flags(
        self,
        capabilities_str: Optional[str]
    ) -> Optional[CapabilityDetectionResult]:
        """
        Detect capabilities from CLI flags string.

        Args:
            capabilities_str: Comma-separated capability names

        Returns:
            CapabilityDetectionResult or None if invalid
        """
        if not capabilities_str:
            return None

        cap_names = [c.strip().lower() for c in capabilities_str.split(",")]
        capabilities: Set[Capability] = set()

        for name in cap_names:
            try:
                capabilities.add(Capability(name))
            except ValueError:
                logger.warning(
                    "Invalid capability name: %s, skipping",
                    name,
                )
                continue

        if not capabilities:
            return None

        return CapabilityDetectionResult(
            capabilities=capabilities,
            source="cli_flags",
            confidence=1.0,
            metadata={"raw_input": capabilities_str}
        )

    def detect_from_metadata(
        self,
        metadata_path: Path
    ) -> Optional[CapabilityDetectionResult]:
        """
        Detect capabilities from model metadata file.

        Args:
            metadata_path: Path to metadata JSON file

        Returns:
            CapabilityDetectionResult or None if unavailable
        """
        if not metadata_path.exists():
            logger.debug("Metadata file not found: %s", metadata_path)
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            capabilities = self._capabilities_from_metadata(metadata)
            confidence = 1.0 if metadata.get("capabilities") else 0.7

            if not capabilities:
                return None

            return CapabilityDetectionResult(
                capabilities=capabilities,
                source="metadata_file",
                confidence=confidence,
                metadata=metadata
            )

        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in metadata file: %s", e)
            return None
        except OSError as e:
            logger.error("Error reading metadata: %s", e)
            return None

    def detect_from_model_name(
        self,
        model_name: str
    ) -> Optional[CapabilityDetectionResult]:
        """
        Detect capabilities from model name heuristics.

        Args:
            model_name: Name of the model

        Returns:
            CapabilityDetectionResult or None if no heuristics match
        """
        if not model_name:
            return None

        model_lower = model_name.lower()
        capabilities = set()
        confidence = 0.5

        vision_keywords = [
            "vision", "vlm", "multimodal", "llava", "clip",
            "image", "visual", "bakllava", "cogvlm"
        ]
        if any(kw in model_lower for kw in vision_keywords):
            capabilities.add(Capability.VISION)

        reasoning_keywords = [
            "reasoning", "cot", "thought", "think", "o1",
            "deepthink", "reflect"
        ]
        if any(kw in model_lower for kw in reasoning_keywords):
            capabilities.add(Capability.REASONING)

        tooling_keywords = [
            "tool", "function", "agent", "gorilla", "toolformer"
        ]
        if any(kw in model_lower for kw in tooling_keywords):
            capabilities.add(Capability.TOOLING)

        if not capabilities:
            capabilities.add(Capability.GENERAL_TEXT)

        return CapabilityDetectionResult(
            capabilities=capabilities,
            source="model_name_heuristics",
            confidence=confidence,
            metadata={"model_name": model_name}
        )

    def detect(
        self,
        model_name: Optional[str] = None,
        metadata_path: Optional[Path] = None,
        capabilities_str: Optional[str] = None,
        metadata_db_path: Optional[Path] = None,
        model_path: Optional[str] = None,
    ) -> CapabilityDetectionResult:
        """
        Detect capabilities using all available sources.

        Priority order:
        1. CLI flags (if provided)
        2. Metadata database (if exists)
        3. Metadata file (if exists)
        4. Model name heuristics
        5. Default capability

        Args:
            model_name: Name of the model
            metadata_path: Path to metadata file
            capabilities_str: Comma-separated capability flags
            metadata_db_path: Path to metadata SQLite database
            model_path: Model path or model identifier

        Returns:
            CapabilityDetectionResult with detected capabilities
        """
        result = self.detect_from_flags(capabilities_str)
        if result and result.capabilities:
            return result

        if metadata_db_path:
            result = self.detect_from_metadata_db(
                metadata_db_path=metadata_db_path,
                model_name=model_name,
                model_path=model_path,
            )
            if result and result.capabilities:
                return result

        if metadata_path:
            result = self.detect_from_metadata(metadata_path)
            if result and result.capabilities:
                return result

        if model_name:
            result = self.detect_from_model_name(model_name)
            if result and result.capabilities:
                return result

        return CapabilityDetectionResult(
            capabilities={self.default_capability},
            source="default",
            confidence=1.0,
            metadata=None
        )


def get_capability_tests(capability: Capability) -> List[str]:
    """
    Get list of test types for a given capability.

    Args:
        capability: The capability to get tests for

    Returns:
        List of test type identifiers
    """
    test_map = {
        Capability.GENERAL_TEXT: [
            "text_generation",
            "summarization",
            "question_answering",
            "classification"
        ],
        Capability.REASONING: [
            "logical_reasoning",
            "math_problem_solving",
            "code_reasoning",
            "chain_of_thought"
        ],
        Capability.VISION: [
            "image_captioning",
            "vqa",
            "ocr",
            "visual_reasoning"
        ],
        Capability.TOOLING: [
            "function_calling",
            "api_interaction",
            "tool_selection",
            "parameter_extraction"
        ]
    }
    return test_map.get(capability, ["general"])
