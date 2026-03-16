"""
Test runner for capability-driven benchmarks.

Loads test cases, orchestrates benchmark execution, and manages outputs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging

from agents.bench_agent import BenchmarkAgent, ModelAdapter, TestCase
from agents.capabilities import Capability, CapabilityDetector

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
            logger.warning(f"Template not found: {template_path}")
            return None

        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading template {template_name}: {e}")
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
            logger.warning(f"Data file not found: {data_path}")
            return []

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading data {data_file}: {e}")
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
        logger.info(
            "Vision tests require image files. "
            "Skipping if images not available."
        )
        return []

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

        logger.info(f"Running benchmark for: {model_name}")

        detected_caps = self._detect_capabilities(
            model_path,
            model_name,
            capabilities
        )

        logger.info(f"Detected capabilities: {detected_caps}")

        test_cases = self._load_test_cases(detected_caps)

        logger.info(f"Loaded {len(test_cases)} test cases")

        if adapter is None:
            from agents.bench_agent import LMStudioAdapter
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
                config=self.config
            )

            return self._report_to_dict(report)

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

        result = self.capability_detector.detect(
            model_name=model_name,
            metadata_path=Path(model_path) / "metadata.json",
            capabilities_str=caps_str
        )

        return list(result.capabilities)

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
