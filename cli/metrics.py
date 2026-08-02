"""
Metrics computation module for capability-driven benchmarks.

Implements quality metrics for different capabilities:
- General text: ROUGE
- Reasoning: Exact Match (EM), F1, accuracy
- Tooling: Function call accuracy, parameter accuracy
"""

from collections import Counter
from dataclasses import dataclass
import json
import logging
import re
import string
from typing import Any, Dict, List, Optional, Union

from core.code_exec import run_code_test

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """
    Result of a metric computation.

    Attributes:
        name: Metric name
        value: Computed value
        max_value: Maximum possible value
        normalized: Normalized value (0.0 to 1.0)
        metadata: Additional metadata about computation
    """
    name: str
    value: float
    max_value: float
    normalized: float
    metadata: Optional[Dict] = None


class BaseMetric:
    """
    Base class for all metrics.
    """

    def __init__(self, name: str):
        """
        Initialize metric.

        Args:
            name: Name of the metric
        """
        self.name = name

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute metric between prediction and reference.

        Args:
            prediction: Model prediction
            reference: Reference answer(s)
            **kwargs: Additional arguments

        Returns:
            MetricResult with computed score
        """
        raise NotImplementedError("Subclasses must implement compute")

    def normalize(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join(text.split())
        return text


class ExactMatchMetric(BaseMetric):
    """
    Exact match metric for reasoning tasks.
    """

    def __init__(self, normalize_text: bool = True):
        """
        Initialize exact match metric.

        Args:
            normalize_text: Whether to normalize text before comparison
        """
        super().__init__("exact_match")
        self.normalize_text = normalize_text

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute exact match score.

        Args:
            prediction: Model prediction
            reference: Reference answer(s)
            **kwargs: Additional arguments

        Returns:
            MetricResult with 1.0 for match, 0.0 otherwise
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        pred_text = prediction
        if self.normalize_text:
            pred_text = self.normalize(pred_text)

        for ref in references:
            ref_text = ref
            if self.normalize_text:
                ref_text = self.normalize(ref_text)

            if pred_text == ref_text:
                return MetricResult(
                    name=self.name,
                    value=1.0,
                    max_value=1.0,
                    normalized=1.0,
                    metadata={"matched": True, "matched_reference": ref}
                )

        return MetricResult(
            name=self.name,
            value=0.0,
            max_value=1.0,
            normalized=0.0,
            metadata={"matched": False}
        )


class F1Metric(BaseMetric):
    """
    Token-level F1 metric for reasoning tasks.
    """

    def __init__(self):
        """
        Initialize F1 metric.
        """
        super().__init__("f1")

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute token-level F1 score.

        Args:
            prediction: Model prediction
            reference: Reference answer(s)
            **kwargs: Additional arguments

        Returns:
            MetricResult with F1 score
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        pred_tokens = self.normalize(prediction).split()
        pred_counts = Counter(pred_tokens)
        total_pred = sum(pred_counts.values())

        max_f1 = 0.0
        best_ref = None

        for ref in references:
            ref_tokens = self.normalize(ref).split()
            ref_counts = Counter(ref_tokens)
            total_ref = sum(ref_counts.values())

            if not total_pred or not total_ref:
                f1 = 0.0
            else:
                common_counts = pred_counts & ref_counts
                overlap = sum(common_counts.values())
                if overlap == 0:
                    f1 = 0.0
                else:
                    precision = overlap / total_pred
                    recall = overlap / total_ref
                    f1 = (
                        2 * precision * recall / (precision + recall)
                    )

            if f1 > max_f1:
                max_f1 = f1
                best_ref = ref

        return MetricResult(
            name=self.name,
            value=max_f1,
            max_value=1.0,
            normalized=max_f1,
            metadata={"best_matching_reference": best_ref}
        )


class RougeMetric(BaseMetric):
    """
    ROUGE metric for text generation tasks.
    Simplified implementation without external dependencies.
    """

    def __init__(self, rouge_type: str = "rouge-1"):
        """
        Initialize ROUGE metric.

        Args:
            rouge_type: Type of ROUGE (rouge-1, rouge-2, rouge-l)
        """
        super().__init__(rouge_type)
        self.rouge_type = rouge_type

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """
        Get n-grams from token list.

        Args:
            tokens: List of tokens
            n: N-gram size

        Returns:
            List of n-gram tuples
        """
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def _rouge_n(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str],
        n: int
    ) -> float:
        """
        Compute ROUGE-N score.

        Args:
            pred_tokens: Prediction tokens
            ref_tokens: Reference tokens
            n: N-gram size

        Returns:
            ROUGE-N score
        """
        if len(pred_tokens) < n or len(ref_tokens) < n:
            return 0.0

        pred_ngrams = set(self._get_ngrams(pred_tokens, n))
        ref_ngrams = set(self._get_ngrams(ref_tokens, n))

        if not ref_ngrams:
            return 0.0

        overlap = pred_ngrams & ref_ngrams
        return len(overlap) / len(ref_ngrams)

    def _rouge_l(
        self,
        pred_tokens: List[str],
        ref_tokens: List[str]
    ) -> float:
        """
        Compute ROUGE-L (longest common subsequence) score.

        Args:
            pred_tokens: Prediction tokens
            ref_tokens: Reference tokens

        Returns:
            ROUGE-L score
        """
        m, n = len(pred_tokens), len(ref_tokens)
        if m == 0 or n == 0:
            return 0.0

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_tokens[i - 1] == ref_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        if n == 0:
            return 0.0
        return lcs_length / n

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute ROUGE score.

        Args:
            prediction: Model prediction
            reference: Reference answer(s)
            **kwargs: Additional arguments

        Returns:
            MetricResult with ROUGE score
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        pred_tokens = self.normalize(prediction).split()

        max_score = 0.0
        best_ref = None

        for ref in references:
            ref_tokens = self.normalize(ref).split()

            if self.rouge_type == "rouge-1":
                score = self._rouge_n(pred_tokens, ref_tokens, 1)
            elif self.rouge_type == "rouge-2":
                score = self._rouge_n(pred_tokens, ref_tokens, 2)
            elif self.rouge_type == "rouge-l":
                score = self._rouge_l(pred_tokens, ref_tokens)
            else:
                score = 0.0

            if score > max_score:
                max_score = score
                best_ref = ref

        return MetricResult(
            name=self.name,
            value=max_score,
            max_value=1.0,
            normalized=max_score,
            metadata={"best_matching_reference": best_ref}
        )


class AccuracyMetric(BaseMetric):
    """
    Accuracy metric for classification and reasoning tasks.
    """

    def __init__(self, extract_answer: bool = True):
        """
        Initialize accuracy metric.

        Args:
            extract_answer: Whether to extract answer from response
        """
        super().__init__("accuracy")
        self.extract_answer = extract_answer

    def _answer_candidates(self, text: str) -> List[str]:
        """Collect plausible final answers from a free-form response.

        Models answer in whatever shape they like: a bare token, a sentence
        ending in the value, a chain of thought that revisits intermediate
        results. Rather than betting on one pattern, gather several candidates
        and let the comparison decide. Everything is taken from the *end* of
        the response, because that is where a conclusion lives; the beginning
        usually restates the question.

        Args:
            text: Raw model response.

        Returns:
            Candidate answers, most specific first.
        """
        candidates: List[str] = []

        def add(value: Optional[str]) -> None:
            if value:
                cleaned = value.strip().strip("*").strip(" .:,;")
                if cleaned and cleaned not in candidates:
                    candidates.append(cleaned)

        boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
        if boxed:
            add(boxed[-1])

        marker_pattern = (
            r"(?:final answer|the answer is|answer|result|equals|therefore)"
            r"\b[^\n]*?"
        )
        marker_hits = re.findall(marker_pattern + r"(-?\d+(?:\.\d+)?)", text,
                                 re.IGNORECASE)
        if marker_hits:
            add(marker_hits[-1])

        short_marker = re.findall(
            r"(?:final answer|the answer is|answer)\s*(?:is|:|=)\s*"
            r"\**([^\n*.]{1,40})",
            text,
            re.IGNORECASE,
        )
        if short_marker:
            add(short_marker[-1])

        numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
        if numbers:
            add(numbers[-1])

        lines = [line.strip() for line in text.strip().splitlines()
                 if line.strip()]
        if lines:
            add(lines[-1])

        first_token = text.strip().split()
        if first_token:
            add(first_token[0])

        add(text)
        return candidates

    def _extract_answer(self, text: str) -> str:
        """
        Extract the most likely answer from text.

        Args:
            text: Text to extract answer from

        Returns:
            Extracted answer, or the whole text when nothing stands out
        """
        candidates = self._answer_candidates(text)
        return candidates[0] if candidates else text.strip()

    def _ends_with_reference(self, text: str, ref_text: str) -> bool:
        """Whether the reference appears as a whole word near the end.

        Scoped to the closing sentences so an intermediate value mentioned
        halfway through a derivation does not count as the answer.
        """
        tail = " ".join(text.strip().splitlines()[-3:])
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", tail)
                     if part.strip()]
        window = " ".join(sentences[-2:]) if sentences else tail
        pattern = r"(?<![\w.])" + re.escape(ref_text) + r"(?![\w.])"
        return bool(re.search(pattern, self.normalize(window)))

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute accuracy score.

        Args:
            prediction: Model prediction
            reference: Reference answer(s)
            **kwargs: Additional arguments

        Returns:
            MetricResult with 1.0 for correct, 0.0 otherwise
        """
        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        if self.extract_answer:
            candidates = [
                self.normalize(candidate)
                for candidate in self._answer_candidates(prediction)
            ]
        else:
            candidates = [self.normalize(prediction)]

        for ref in references:
            ref_text = self.normalize(ref)
            if not ref_text:
                continue

            matched_by = None
            if ref_text in candidates:
                matched_by = "candidate"
            elif self.extract_answer and self._ends_with_reference(
                prediction, ref_text
            ):
                # "…the average speed is 60 mph." states the answer inside a
                # sentence. Requiring a bare token would fail every model that
                # writes in prose.
                matched_by = "closing_sentence"

            if matched_by:
                return MetricResult(
                    name=self.name,
                    value=1.0,
                    max_value=1.0,
                    normalized=1.0,
                    metadata={
                        "correct": True,
                        "matched_reference": ref,
                        "matched_by": matched_by,
                    }
                )

        return MetricResult(
            name=self.name,
            value=0.0,
            max_value=1.0,
            normalized=0.0,
            metadata={"correct": False}
        )


class FunctionCallMetric(BaseMetric):
    """
    Function call accuracy metric for tooling tasks.
    """

    def __init__(self):
        """
        Initialize function call metric.
        """
        super().__init__("function_call_accuracy")

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Compute function call accuracy.

        Args:
            prediction: Model prediction (JSON string)
            reference: Reference function call (JSON string)
            **kwargs: Additional arguments

        Returns:
            MetricResult with accuracy score
        """
        try:
            if isinstance(prediction, str):
                pred_data = json.loads(prediction)
            else:
                pred_data = prediction
        except (json.JSONDecodeError, TypeError):
            return MetricResult(
                name=self.name,
                value=0.0,
                max_value=1.0,
                normalized=0.0,
                metadata={"error": "Invalid prediction JSON"}
            )

        if not isinstance(pred_data, dict):
            return MetricResult(
                name=self.name,
                value=0.0,
                max_value=1.0,
                normalized=0.0,
                metadata={"error": "Prediction JSON must be an object"},
            )

        if isinstance(reference, str):
            references = [reference]
        else:
            references = reference

        best_score = 0.0
        best_metadata: Optional[Dict[str, Any]] = None

        for ref in references:
            try:
                if isinstance(ref, str):
                    ref_data = json.loads(ref)
                else:
                    ref_data = ref
            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(ref_data, dict):
                continue

            function_match = (
                pred_data.get("function") == ref_data.get("function")
            )

            if not function_match:
                continue

            pred_params = pred_data.get("parameters", {})
            ref_params = ref_data.get("parameters", {})

            if not isinstance(pred_params, dict):
                pred_params = {}
            if not isinstance(ref_params, dict):
                ref_params = {}

            if not ref_params:
                param_score = 1.0 if function_match else 0.0
            else:
                matching_params = sum(
                    1 for k, v in ref_params.items()
                    if pred_params.get(k) == v
                )
                param_score = matching_params / len(ref_params)

            total_score = 0.5 * (
                1.0 if function_match else 0.0
            ) + 0.5 * param_score

            if total_score > best_score:
                best_score = total_score
                best_metadata = {
                    "function_match": function_match,
                    "param_accuracy": param_score,
                }

        if best_metadata is not None:
            return MetricResult(
                name=self.name,
                value=best_score,
                max_value=1.0,
                normalized=best_score,
                metadata=best_metadata,
            )

        return MetricResult(
            name=self.name,
            value=0.0,
            max_value=1.0,
            normalized=0.0,
            metadata={"function_match": False, "param_accuracy": 0.0}
        )


class CodeExecutionMetric(BaseMetric):
    """Runs generated code against assertions and scores pass/fail.

    Unlike the text metrics this one does not compare strings: the reference
    is a snippet of checks, and the score is whether they hold. Wording,
    formatting and commentary are irrelevant, which is the point — code is
    correct or it is not.
    """

    def __init__(self, timeout: Optional[float] = None):
        """
        Initialize code execution metric.

        Args:
            timeout: Wall-clock limit per execution in seconds.
        """
        super().__init__("code_execution")
        self.timeout = timeout

    def compute(
        self,
        prediction: str,
        reference: Union[str, List[str]],
        **kwargs: Any
    ) -> MetricResult:
        """
        Execute the prediction and evaluate the reference checks.

        Args:
            prediction: Model response containing code.
            reference: Assertion snippet, or a list of them (all must pass).
            **kwargs: Additional arguments.

        Returns:
            MetricResult with 1.0 when all checks pass, 0.0 otherwise.
        """
        checks = [reference] if isinstance(reference, str) else list(reference)
        if not checks:
            return MetricResult(
                name=self.name,
                value=0.0,
                max_value=1.0,
                normalized=0.0,
                metadata={"error": "No checks provided"},
            )

        details: List[str] = []
        timed_out = False
        for check in checks:
            result = run_code_test(prediction, check, timeout=self.timeout)
            details.append(result.detail)
            timed_out = timed_out or result.timed_out
            if not result.passed:
                return MetricResult(
                    name=self.name,
                    value=0.0,
                    max_value=1.0,
                    normalized=0.0,
                    metadata={
                        "passed": False,
                        "detail": result.detail,
                        "timed_out": result.timed_out,
                        "stderr": result.stderr,
                    },
                )

        return MetricResult(
            name=self.name,
            value=1.0,
            max_value=1.0,
            normalized=1.0,
            metadata={
                "passed": True,
                "detail": "; ".join(details),
                "timed_out": timed_out,
                "checks": len(checks),
            },
        )


def aggregate_metrics(
    results: List[MetricResult],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Aggregate multiple metric results into a single score.

    Args:
        results: List of metric results
        weights: Optional weights for each metric

    Returns:
        Aggregated score (0.0 to 1.0)
    """
    if not results:
        return 0.0

    if weights is None:
        weights = {r.name: 1.0 for r in results}

    total_weight = sum(
        weights.get(r.name, 1.0) for r in results
    )

    if total_weight == 0:
        return 0.0

    weighted_sum = sum(
        r.normalized * weights.get(r.name, 1.0)
        for r in results
    )

    return weighted_sum / total_weight
