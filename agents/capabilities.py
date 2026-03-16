"""
Capability detection module for LM Studio Bench.

Detects model capabilities from metadata or user-specified flags.
Supports: general_text, reasoning, vision, tooling.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
import logging

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

        try:
            cap_names = [c.strip().lower() for c in capabilities_str.split(",")]
            capabilities = set()

            for name in cap_names:
                try:
                    capabilities.add(Capability(name))
                except ValueError:
                    logger.warning(
                        f"Invalid capability name: {name}, skipping"
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

        except Exception as e:
            logger.error(f"Error parsing capability flags: {e}")
            return None

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
            logger.debug(f"Metadata file not found: {metadata_path}")
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            capabilities = set()
            confidence = 0.7

            if "capabilities" in metadata:
                for cap in metadata["capabilities"]:
                    try:
                        capabilities.add(Capability(cap.lower()))
                    except ValueError:
                        logger.warning(
                            f"Unknown capability in metadata: {cap}"
                        )
                        continue
                confidence = 1.0

            if "modalities" in metadata:
                modalities = metadata["modalities"]
                if isinstance(modalities, list):
                    if "vision" in modalities or "image" in modalities:
                        capabilities.add(Capability.VISION)
                    if "text" in modalities:
                        capabilities.add(Capability.GENERAL_TEXT)

            if "function_calling" in metadata:
                if metadata["function_calling"]:
                    capabilities.add(Capability.TOOLING)

            if "tools" in metadata or "tool_use" in metadata:
                capabilities.add(Capability.TOOLING)

            model_name = metadata.get("name", "").lower()
            if any(
                kw in model_name
                for kw in ["reasoning", "cot", "chain-of-thought"]
            ):
                capabilities.add(Capability.REASONING)

            if not capabilities:
                return None

            return CapabilityDetectionResult(
                capabilities=capabilities,
                source="metadata_file",
                confidence=confidence,
                metadata=metadata
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
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
        capabilities_str: Optional[str] = None
    ) -> CapabilityDetectionResult:
        """
        Detect capabilities using all available sources.

        Priority order:
        1. CLI flags (if provided)
        2. Metadata file (if exists)
        3. Model name heuristics
        4. Default capability

        Args:
            model_name: Name of the model
            metadata_path: Path to metadata file
            capabilities_str: Comma-separated capability flags

        Returns:
            CapabilityDetectionResult with detected capabilities
        """
        result = self.detect_from_flags(capabilities_str)
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
