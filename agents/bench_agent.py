"""
Benchmark agent core for capability-driven evaluation.

Manages model inference, metric computation, and result collection.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import logging
import time

from agents.capabilities import Capability, CapabilityDetector
from bench.metrics import (
    AccuracyMetric,
    ExactMatchMetric,
    F1Metric,
    FunctionCallMetric,
    MetricResult,
    RougeMetric,
    aggregate_metrics
)

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """
    A single test case for evaluation.

    Attributes:
        id: Unique test identifier
        capability: Capability being tested
        prompt: Input prompt text
        reference: Reference answer(s)
        image_path: Optional image path for vision tests
        metadata: Additional metadata
    """
    id: str
    capability: Capability
    prompt: str
    reference: Union[str, List[str]]
    image_path: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class InferenceResult:
    """
    Result of a single model inference.

    Attributes:
        test_id: Test case identifier
        prompt: Input prompt
        response: Model response
        timestamp_start: Start timestamp
        timestamp_end: End timestamp
        latency_ms: Latency in milliseconds
        tokens_generated: Number of tokens generated
        throughput: Tokens per second
        raw_output: Raw model output
        tool_calls: Tool calls made (for tooling capability)
        error: Error message if inference failed
    """
    test_id: str
    prompt: str
    response: str
    timestamp_start: float
    timestamp_end: float
    latency_ms: float
    tokens_generated: Optional[int] = None
    throughput: Optional[float] = None
    raw_output: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """
    Result of evaluating a model on a test case.

    Attributes:
        test_id: Test case identifier
        capability: Capability being tested
        inference: Inference result
        metrics: Computed metrics
        quality_score: Aggregated quality score
    """
    test_id: str
    capability: Capability
    inference: InferenceResult
    metrics: List[MetricResult]
    quality_score: float


@dataclass
class BenchmarkReport:
    """
    Complete benchmark report for a model.

    Attributes:
        model_name: Name of the model
        model_path: Path to model
        capabilities: Detected capabilities
        timestamp: Benchmark timestamp
        results: Evaluation results
        summary: Summary statistics
        config: Benchmark configuration
        raw_outputs_dir: Directory with raw outputs
    """
    model_name: str
    model_path: str
    capabilities: List[str]
    timestamp: str
    results: List[Dict]
    summary: Dict
    config: Dict
    raw_outputs_dir: Optional[str] = None


class ModelAdapter:
    """
    Abstract adapter for model inference.

    Subclasses implement specific model loading and inference logic.
    """

    def load(self, model_path: str, **kwargs: Any) -> None:
        """
        Load model from path.

        Args:
            model_path: Path to model
            **kwargs: Additional loading arguments
        """
        raise NotImplementedError("Subclasses must implement load")

    def unload(self) -> None:
        """
        Unload model and free resources.
        """
        raise NotImplementedError("Subclasses must implement unload")

    def infer(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any
    ) -> InferenceResult:
        """
        Run inference on prompt.

        Args:
            prompt: Input prompt
            image_path: Optional image for vision models
            **kwargs: Additional inference arguments

        Returns:
            InferenceResult with response and metrics
        """
        raise NotImplementedError("Subclasses must implement infer")

    def is_loaded(self) -> bool:
        """
        Check if model is loaded.

        Returns:
            True if model is loaded
        """
        raise NotImplementedError("Subclasses must implement is_loaded")


class LMStudioAdapter(ModelAdapter):
    """
    Adapter for LM Studio models.

    Uses LM Studio SDK or REST API for inference.
    """

    def __init__(self, use_rest_api: bool = True):
        """
        Initialize LM Studio adapter.

        Args:
            use_rest_api: Whether to use REST API instead of SDK
        """
        self.use_rest_api = use_rest_api
        self.model = None
        self.model_path = None
        self._rest_instance_id: Optional[str] = None

        if not use_rest_api:
            try:
                import lmstudio
                self.lms = lmstudio
            except ImportError:
                logger.error("lmstudio package not available")
                raise

    def load(self, model_path: str, **kwargs: Any) -> None:
        """
        Load LM Studio model.

        Args:
            model_path: Model identifier or path
            **kwargs: Additional arguments (context_length, gpu_offload, etc)
        """
        self.model_path = model_path

        if self.use_rest_api:
            from src.rest_client import LMStudioRESTClient

            context_length = kwargs.get("context_length", 2048)
            gpu_offload = kwargs.get("gpu_offload", 1.0)

            self.rest_client = LMStudioRESTClient()
            instance_id = self.rest_client.load_model(
                model_path,
                context_length=context_length,
                gpu_offload=gpu_offload
            )

            if not instance_id:
                raise RuntimeError(f"Failed to load model: {model_path}")

            self._rest_instance_id = instance_id
            logger.info(f"Loaded model via REST API: {model_path}")
        else:
            from lmstudio import LlmLoadModelConfig

            config = LlmLoadModelConfig(
                context_length=kwargs.get("context_length", 2048),
                gpu_offload=kwargs.get("gpu_offload", 1.0)
            )

            self.model = self.lms.llm(model_path, config=config)
            logger.info(f"Loaded model via SDK: {model_path}")

    def unload(self) -> None:
        """
        Unload model and free resources.
        """
        if self.use_rest_api and hasattr(self, "rest_client"):
            try:
                if self._rest_instance_id is not None:
                    self.rest_client.unload_model(self._rest_instance_id)
                    logger.info("Unloaded model via REST API")
                else:
                    logger.warning(
                        "No REST instance id set; skipping unload_model call"
                    )
            except Exception as e:
                logger.warning(f"Error unloading model: {e}")
            finally:
                self._rest_instance_id = None
        else:
            self.model = None
            logger.info("Unloaded model via SDK")

        self.model_path = None

    def infer(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        **kwargs: Any
    ) -> InferenceResult:
        """
        Run inference using LM Studio.

        Args:
            prompt: Input prompt
            image_path: Optional image path (for multimodal models)
            **kwargs: Additional inference arguments

        Returns:
            InferenceResult with response and metrics
        """
        test_id = kwargs.get("test_id", "unknown")
        start_time = time.time()
        timestamp_start = start_time

        try:
            if self.use_rest_api:
                messages = [{"role": "user", "content": prompt}]
                response = self.rest_client.chat_stream(messages)

                response_text = ""
                tokens_generated = 0

                if isinstance(response, dict):
                    # REST client returns a dict, e.g. {"text": ..., "stats": {...}}
                    response_text = (
                        response.get("text")
                        or response.get("content")
                        or ""
                    )
                    stats = response.get("stats")
                    if isinstance(stats, dict):
                        completion_tokens = stats.get("completion_tokens")
                        if completion_tokens is not None:
                            tokens_generated = completion_tokens
                    elif stats is not None:
                        # Handle ChatStats-like objects with tokens_out attribute
                        tokens_generated = getattr(
                            stats,
                            "tokens_out",
                            getattr(stats, "completion_tokens", 0),
                        )
                elif hasattr(response, "stats"):
                    # Fallback for object-style responses with stats/text attrs
                    stats_obj = response.stats
                    tokens_generated = getattr(
                        stats_obj,
                        "tokens_out",
                        getattr(stats_obj, "completion_tokens", 0),
                    )
                    if hasattr(response, "content"):
                        response_text = response.content
                    elif hasattr(response, "text"):
                        response_text = response.text
                    else:
                        response_text = ""
                else:
                    # Last-resort fallback: stringify unknown response type
                    response_text = str(response)

            else:
                from lmstudio import PredictionConfig

                config = PredictionConfig(
                    temperature=kwargs.get("temperature", 0.1),
                    max_tokens=kwargs.get("max_tokens", 256)
                )

                response = self.model.respond(prompt, config=config)
                response_text = response.content
                tokens_generated = getattr(
                    response,
                    "completion_tokens",
                    None
                )

            end_time = time.time()
            timestamp_end = end_time
            latency_ms = (end_time - start_time) * 1000

            throughput = None
            if tokens_generated and latency_ms > 0:
                throughput = (tokens_generated / latency_ms) * 1000

            return InferenceResult(
                test_id=test_id,
                prompt=prompt,
                response=response_text,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                latency_ms=latency_ms,
                tokens_generated=tokens_generated,
                throughput=throughput,
                raw_output=response_text,
                tool_calls=None,
                error=None
            )

        except Exception as e:
            end_time = time.time()
            error_msg = f"Inference error: {str(e)}"
            logger.error(error_msg)

            return InferenceResult(
                test_id=test_id,
                prompt=prompt,
                response="",
                timestamp_start=timestamp_start,
                timestamp_end=end_time,
                latency_ms=(end_time - start_time) * 1000,
                tokens_generated=0,
                throughput=None,
                raw_output=None,
                tool_calls=None,
                error=error_msg
            )

    def is_loaded(self) -> bool:
        """
        Check if model is currently loaded.

        Returns:
            True if model is loaded
        """
        if self.use_rest_api:
            return hasattr(self, "rest_client") and self.model_path is not None
        return self.model is not None


class BenchmarkAgent:
    """
    Main benchmark agent for capability-driven evaluation.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        output_dir: Path,
        capability_detector: Optional[CapabilityDetector] = None
    ):
        """
        Initialize benchmark agent.

        Args:
            adapter: Model adapter for inference
            output_dir: Directory for output files
            capability_detector: Optional capability detector
        """
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.output_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        self.capability_detector = (
            capability_detector or CapabilityDetector()
        )

        self.metrics_map = {
            Capability.GENERAL_TEXT: [
                RougeMetric("rouge-1"),
                RougeMetric("rouge-l"),
                F1Metric()
            ],
            Capability.REASONING: [
                ExactMatchMetric(),
                F1Metric(),
                AccuracyMetric()
            ],
            Capability.VISION: [
                AccuracyMetric(),
                RougeMetric("rouge-l")
            ],
            Capability.TOOLING: [
                FunctionCallMetric(),
                AccuracyMetric(extract_answer=False)
            ]
        }

    def evaluate_test_case(
        self,
        test_case: TestCase
    ) -> EvaluationResult:
        """
        Evaluate model on a single test case.

        Args:
            test_case: Test case to evaluate

        Returns:
            EvaluationResult with metrics
        """
        inference = self.adapter.infer(
            prompt=test_case.prompt,
            image_path=test_case.image_path,
            test_id=test_case.id
        )

        self._save_raw_output(test_case, inference)

        if inference.error:
            logger.error(
                f"Inference failed for {test_case.id}: {inference.error}"
            )
            return EvaluationResult(
                test_id=test_case.id,
                capability=test_case.capability,
                inference=inference,
                metrics=[],
                quality_score=0.0
            )

        metrics = self._compute_metrics(
            test_case.capability,
            inference.response,
            test_case.reference
        )

        quality_score = aggregate_metrics(metrics)

        return EvaluationResult(
            test_id=test_case.id,
            capability=test_case.capability,
            inference=inference,
            metrics=metrics,
            quality_score=quality_score
        )

    def _compute_metrics(
        self,
        capability: Capability,
        prediction: str,
        reference: Union[str, List[str]]
    ) -> List[MetricResult]:
        """
        Compute metrics for a capability.

        Args:
            capability: Capability being tested
            prediction: Model prediction
            reference: Reference answer(s)

        Returns:
            List of computed metrics
        """
        metrics = self.metrics_map.get(capability, [])
        results = []

        for metric in metrics:
            try:
                result = metric.compute(prediction, reference)
                results.append(result)
            except Exception as e:
                logger.error(f"Error computing {metric.name}: {e}")
                continue

        return results

    def _save_raw_output(
        self,
        test_case: TestCase,
        inference: InferenceResult
    ) -> None:
        """
        Save raw inference output to file.

        Args:
            test_case: Test case
            inference: Inference result
        """
        try:
            output_file = self.raw_dir / f"{test_case.id}.json"
            data = {
                "test_id": test_case.id,
                "capability": test_case.capability.value,
                "prompt": test_case.prompt,
                "response": inference.response,
                "latency_ms": inference.latency_ms,
                "tokens_generated": inference.tokens_generated,
                "throughput": inference.throughput,
                "timestamp": inference.timestamp_start,
                "error": inference.error
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving raw output: {e}")

    def run_benchmark(
        self,
        test_cases: List[TestCase],
        model_name: str,
        model_path: str,
        config: Optional[Dict] = None
    ) -> BenchmarkReport:
        """
        Run complete benchmark on test cases.

        Args:
            test_cases: List of test cases.
            model_name: Name of the model.
            model_path: Path to model.
            config: Optional configuration dict.
            detected_capabilities: Optional list of capabilities
                detected for this model. When provided, this list
                is used for the report's capabilities field, even
                if some capabilities have zero test cases.

        Returns:
            BenchmarkReport with all results.
        """
        logger.info(
            f"Starting benchmark for {model_name} "
            f"with {len(test_cases)} test cases"
        )

        results = []
        capability_results: Dict[str, List[Any]] = {}

        for test_case in test_cases:
            logger.info(f"Evaluating test case: {test_case.id}")

            eval_result = self.evaluate_test_case(test_case)
            results.append(eval_result)

            cap = test_case.capability.value
            if cap not in capability_results:
                capability_results[cap] = []
            capability_results[cap].append(eval_result)

        summary = self._compute_summary(results, capability_results)

        if detected_capabilities:
            detected_caps_set = {
                c.value if isinstance(c, Capability) else str(c)
                for c in detected_capabilities
            }
        else:
            detected_caps_set = {
                tc.capability.value for tc in test_cases
            }

        detected_caps = sorted(detected_caps_set)

        return BenchmarkReport(
            model_name=model_name,
            model_path=model_path,
            capabilities=detected_caps,
            timestamp=datetime.now().isoformat(),
            results=[self._serialize_result(r) for r in results],
            summary=summary,
            config=config or {},
            raw_outputs_dir=str(self.raw_dir)
        )

    def _compute_summary(
        self,
        results: List[EvaluationResult],
        capability_results: Dict[str, List[EvaluationResult]]
    ) -> Dict:
        """
        Compute summary statistics.

        Args:
            results: All evaluation results
            capability_results: Results grouped by capability

        Returns:
            Summary statistics dictionary
        """
        total_tests = len(results)
        successful_tests = sum(
            1 for r in results if not r.inference.error
        )

        avg_latency = sum(
            r.inference.latency_ms for r in results
        ) / total_tests if total_tests > 0 else 0

        avg_quality = sum(
            r.quality_score for r in results
        ) / total_tests if total_tests > 0 else 0

        avg_throughput_values = [
            r.inference.throughput for r in results
            if r.inference.throughput is not None
        ]
        avg_throughput = (
            sum(avg_throughput_values) / len(avg_throughput_values)
            if avg_throughput_values else None
        )

        capability_summaries = {}
        for cap, cap_results in capability_results.items():
            cap_count = len(cap_results)
            cap_quality = sum(
                r.quality_score for r in cap_results
            ) / cap_count if cap_count > 0 else 0

            capability_summaries[cap] = {
                "test_count": cap_count,
                "avg_quality_score": cap_quality,
                "success_rate": sum(
                    1 for r in cap_results if not r.inference.error
                ) / cap_count if cap_count > 0 else 0
            }

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": (
                successful_tests / total_tests if total_tests > 0 else 0
            ),
            "avg_latency_ms": avg_latency,
            "avg_quality_score": avg_quality,
            "avg_throughput_tokens_per_sec": avg_throughput,
            "by_capability": capability_summaries
        }

    def _serialize_result(
        self,
        result: EvaluationResult
    ) -> Dict:
        """
        Serialize evaluation result to dictionary.

        Args:
            result: Evaluation result

        Returns:
            Serialized result dictionary
        """
        return {
            "test_id": result.test_id,
            "capability": result.capability.value,
            "latency_ms": result.inference.latency_ms,
            "tokens_generated": result.inference.tokens_generated,
            "throughput": result.inference.throughput,
            "quality_score": result.quality_score,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "normalized": m.normalized
                }
                for m in result.metrics
            ],
            "error": result.inference.error
        }
