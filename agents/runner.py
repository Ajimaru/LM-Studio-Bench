"""
Test runner for capability-driven benchmarks.

Loads test cases, orchestrates benchmark execution, and manages outputs.
"""

import json
import logging
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional

from agents.benchmark import (
    BenchmarkAgent,
    LMStudioAdapter,
    ModelAdapter,
    TestCase,
)
from agents.cache import AgentCache
from agents.capabilities import Capability, CapabilityDetector
from core.paths import USER_RESULTS_DIR

logger = logging.getLogger(__name__)


class TestLoader:
    """
    Loads test cases from data files and prompt templates.
    """

    def __init__(self, data_dir: Path, prompts_dir: Path):
        """
        Initialize test loader.

        Args:
            data_dir: Directory containing test data
            prompts_dir: Directory containing prompt templates
        """
        self.data_dir = Path(data_dir)
        self.prompts_dir = Path(prompts_dir)

    def load_prompt_template(self, template_name: str) -> Optional[str]:
        """
        Load prompt template from file.

        Args:
            template_name: Name of template file

        Returns:
            Template content or None if not found
        """
        template_path = self.prompts_dir / template_name
        if not template_path.exists():
            logger.warning("Template not found: %s", template_path)
            return None

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as err:
            logger.error("Error loading template %s: %s", template_name, err)
            return None

    def load_test_data(self, data_file: str) -> List[Dict]:
        """
        Load test data from JSON file.

        Args:
            data_file: Name of data file

        Returns:
            List of test data dictionaries
        """
        data_path = self.data_dir / "text" / data_file
        if not data_path.exists():
            logger.warning("Data file not found: %s", data_path)
            return []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
            logger.error("Error loading data %s: %s", data_file, err)
            return []

    def create_test_cases(
        self,
        capability: Capability,
        max_tests: Optional[int] = None
    ) -> List[TestCase]:
        """
        Create test cases for a capability.

        Args:
            capability: Capability to create tests for
            max_tests: Maximum number of tests (None for all)

        Returns:
            List of test cases
        """
        test_cases = []

        if capability == Capability.GENERAL_TEXT:
            test_cases.extend(self._create_qa_tests())
        elif capability == Capability.REASONING:
            test_cases.extend(self._create_reasoning_tests())
        elif capability == Capability.VISION:
            test_cases.extend(self._create_vision_tests())
        elif capability == Capability.TOOLING:
            test_cases.extend(self._create_tooling_tests())

        if max_tests is not None:
            test_cases = test_cases[:max_tests]

        return test_cases

    def _create_qa_tests(self) -> List[TestCase]:
        """
        Create question answering test cases.

        Returns:
            List of QA test cases
        """
        data = self.load_test_data("qa_samples.json")
        if not data:
            return []

        test_cases = []
        for item in data:
            test_case = TestCase(
                id=item["id"],
                capability=Capability.GENERAL_TEXT,
                prompt=item["prompt"],
                reference=item["reference"],
                metadata={"category": item.get("category", "general")}
            )
            test_cases.append(test_case)

        return test_cases

    def _create_reasoning_tests(self) -> List[TestCase]:
        """
        Create reasoning test cases.

        Returns:
            List of reasoning test cases
        """
        data = self.load_test_data("reasoning_samples.json")
        if not data:
            return []

        test_cases = []
        for item in data:
            test_case = TestCase(
                id=item["id"],
                capability=Capability.REASONING,
                prompt=item["prompt"],
                reference=item["reference"],
                metadata={
                    "category": item.get("category", "general"),
                    "reasoning": item.get("reasoning")
                }
            )
            test_cases.append(test_case)

        return test_cases

    def _create_vision_tests(self) -> List[TestCase]:
        """
        Create vision test cases.

        Returns:
            List of vision test cases
        """
        images_dir = self.data_dir / "images"
        if not images_dir.exists():
            logger.info(
                "Vision tests skipped: images directory not found: %s",
                images_dir,
            )
            return []

        vqa_template = self.load_prompt_template("vision_vqa.md")
        if not vqa_template:
            vqa_template = (
                "Answer the question about the image.\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )

        candidates = [
            (
                "vision_001",
                images_dir / "test_image_001.jpg",
                "What animal is shown in the image?",
                "cat",
            ),
            (
                "vision_002",
                images_dir / "test_image_002.jpg",
                "Name one object type visible in the scene.",
                "car",
            ),
            (
                "vision_003",
                images_dir / "test_image_003.jpg",
                "What word is written on the sign?",
                "stop",
            ),
        ]

        test_cases: List[TestCase] = []
        for case_id, image_path, question, reference in candidates:
            if not image_path.exists():
                logger.info("Vision image not found, skipping: %s", image_path)
                continue

            prompt = vqa_template.replace("{question}", question)
            test_cases.append(
                TestCase(
                    id=case_id,
                    capability=Capability.VISION,
                    prompt=prompt,
                    reference=reference,
                    image_path=str(image_path),
                    metadata={"category": "vision_vqa"},
                )
            )

        if not test_cases:
            logger.info("Vision tests skipped: no local image test cases available")

        return test_cases

    def _create_tooling_tests(self) -> List[TestCase]:
        """
        Create tooling test cases.

        Returns:
            List of tooling test cases
        """
        data = self.load_test_data("tooling_samples.json")
        if not data:
            return []

        template = self.load_prompt_template("tooling_function_call.md")
        if not template:
            logger.warning("Tooling template not found, using default")
            template = "{task}"

        test_cases = []
        for item in data:
            prompt = template.replace("{task}", item["task"])

            reference = json.dumps({
                "function": item["expected_function"],
                "parameters": item["expected_parameters"]
            })

            test_case = TestCase(
                id=item["id"],
                capability=Capability.TOOLING,
                prompt=prompt,
                reference=reference,
                metadata={"category": item.get("category", "function_calling")}
            )
            test_cases.append(test_case)

        return test_cases


class BenchmarkRunner:
    """
    Orchestrates benchmark execution across models and capabilities.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: Path
    ):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        data_dir = Path(config.get("data_dir", "tests/data"))
        prompts_dir = Path(config.get("prompts_dir", "tests/prompts"))

        self.test_loader = TestLoader(data_dir, prompts_dir)
        self.capability_detector = CapabilityDetector()
        self.cache = AgentCache()
        self.metadata_db_path = USER_RESULTS_DIR / "model_metadata.db"
        self.metadata_json_dir = USER_RESULTS_DIR / "metadata"

    def run(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        adapter: Optional[ModelAdapter] = None,
        capabilities: Optional[List[str]] = None
    ) -> Dict:
        """
        Run benchmark on a model.

        Args:
            model_path: Path to model or model identifier
            model_name: Optional model name (defaults to path)
            adapter: Optional pre-configured adapter
            capabilities: Optional list of capability strings

        Returns:
            Benchmark report dictionary
        """
        if model_name is None:
            model_name = Path(model_path).name

        logger.info("Running benchmark for: %s", model_name)

        detected_caps = self._detect_capabilities(
            model_path,
            model_name,
            capabilities
        )

        logger.info("Detected capabilities: %s", detected_caps)

        test_cases = self._load_test_cases(detected_caps)

        logger.info("Loaded %s test cases", len(test_cases))

        if adapter is None:
            adapter = LMStudioAdapter(
                use_rest_api=self.config.get("use_rest_api", True)
            )

        try:
            load_config = self._get_load_config()
            adapter.load(model_path, **load_config)

            agent = BenchmarkAgent(
                adapter=adapter,
                output_dir=self.output_dir,
                capability_detector=self.capability_detector
            )

            report = agent.run_benchmark(
                test_cases=test_cases,
                model_name=model_name,
                model_path=model_path,
                config=self.config,
                detected_capabilities=detected_caps,
            )

            report_dict = self._report_to_dict(report)
            self._cache_results(report_dict, model_name, model_path)

            return report_dict

        finally:
            if adapter.is_loaded():
                adapter.unload()

    def _detect_capabilities(
        self,
        model_path: str,
        model_name: str,
        capabilities_str: Optional[List[str]]
    ) -> List[Capability]:
        """
        Detect model capabilities.

        Args:
            model_path: Model path
            model_name: Model name
            capabilities_str: Optional capability strings

        Returns:
            List of detected capabilities
        """
        caps_str = ",".join(capabilities_str) if capabilities_str else None
        metadata_path = self._resolve_metadata_json_path(model_path, model_name)

        result = self.capability_detector.detect(
            model_name=model_name,
            metadata_path=metadata_path,
            capabilities_str=caps_str,
            metadata_db_path=self.metadata_db_path,
            model_path=model_path,
        )

        return sorted(result.capabilities, key=lambda cap: cap.value)

    def _resolve_metadata_json_path(
        self,
        model_path: str,
        model_name: str,
    ) -> Optional[Path]:
        """Resolve metadata JSON fallback path for the given model.

        Preferred location is the user results directory, with backward
        compatibility for legacy ``<model_path>/metadata.json`` layout.
        """
        model_id = model_name.strip() or model_path.strip()
        safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)

        user_metadata_path = (
            self.metadata_json_dir / safe_model_id / "metadata.json"
        )
        if user_metadata_path.exists():
            return user_metadata_path

        legacy_metadata_path = Path(model_path) / "metadata.json"
        if legacy_metadata_path.exists():
            return legacy_metadata_path

        return None

    def _load_test_cases(
        self,
        capabilities: List[Capability]
    ) -> List[TestCase]:
        """
        Load test cases for capabilities.

        Args:
            capabilities: List of capabilities

        Returns:
            List of test cases
        """
        max_tests_per_cap = self.config.get("max_tests_per_capability", 10)

        all_test_cases = []
        for cap in capabilities:
            test_cases = self.test_loader.create_test_cases(
                cap,
                max_tests=max_tests_per_cap
            )
            all_test_cases.extend(test_cases)

        return all_test_cases

    def _get_load_config(self) -> Dict:
        """
        Get model loading configuration.

        Returns:
            Load configuration dictionary
        """
        return {
            "context_length": self.config.get("context_length", 2048),
            "gpu_offload": self.config.get("gpu_offload", 1.0),
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 256)
        }

    def _report_to_dict(self, report) -> Dict:
        """
        Convert benchmark report to dictionary.

        Args:
            report: BenchmarkReport object

        Returns:
            Report as dictionary
        """
        return {
            "model_name": report.model_name,
            "model_path": report.model_path,
            "capabilities": report.capabilities,
            "timestamp": report.timestamp,
            "results": report.results,
            "summary": report.summary,
            "config": report.config,
            "raw_outputs_dir": report.raw_outputs_dir
        }

    def _cache_results(
        self,
        report_dict: Dict,
        model_name: str,
        model_path: str
    ) -> None:
        """
        Cache benchmark results to SQLite database.

        Args:
            report_dict: Report dictionary
            model_name: Model name
            model_path: Model path
        """
        try:
            results = report_dict.get("results", [])
            summary = report_dict.get("summary", {})

            for result in results:
                capability = result.get("capability")
                if not capability:
                    continue

                test_id = result.get("test_id", "")
                test_name = result.get("test_name", test_id)
                error_message = result.get("error")

                self.cache.save_test_result(
                    model_name=model_name,
                    model_path=model_path,
                    capability=capability,
                    test_id=test_id,
                    test_name=test_name,
                    latency_ms=result.get("latency_ms", 0),
                    tokens_generated=result.get("tokens_generated"),
                    throughput_tokens_per_sec=result.get("throughput"),
                    quality_score=result.get("quality_score", 0),
                    rouge_score=result.get("rouge_score"),
                    f1_score=result.get("f1_score"),
                    exact_match_score=result.get("exact_match_score"),
                    accuracy_score=result.get("accuracy_score"),
                    function_call_accuracy=result.get("function_call_accuracy"),
                    raw_output=result.get("raw_output", ""),
                    reference_output=result.get("reference_output", ""),
                    error_message=error_message,
                    success=error_message is None,
                )

            capability_summary = summary.get("by_capability", {})
            for capability, cap_data in capability_summary.items():
                if not isinstance(cap_data, dict):
                    continue

                cap_results = [
                    item for item in results if item.get("capability") == capability
                ]
                total_tests = cap_data.get("test_count", len(cap_results))
                successful_tests = sum(
                    1 for item in cap_results if not item.get("error")
                )
                failed_tests = max(total_tests - successful_tests, 0)

                self.cache.save_summary(
                    model_name=model_name,
                    model_path=model_path,
                    capability=capability,
                    total_tests=total_tests,
                    successful_tests=successful_tests,
                    failed_tests=failed_tests,
                    success_rate=cap_data.get("success_rate", 0),
                    avg_latency_ms=0,
                    avg_throughput=None,
                    avg_quality_score=cap_data.get("avg_quality_score", 0),
                    avg_rouge=None,
                    avg_f1=None,
                    avg_exact_match=None,
                    avg_accuracy=None,
                )

            logger.info("Benchmark results cached for %s", model_name)

        except (sqlite3.Error, TypeError, ValueError, KeyError) as err:
            logger.error("Error caching results for %s: %s", model_name, err)
