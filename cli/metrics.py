"""
Metrics computation module for capability-driven benchmarks.

Implements quality metrics for different capabilities:
- General text: ROUGE, BERTScore
- Reasoning: Exact Match (EM), F1, accuracy
- Vision: VQA accuracy, CIDEr
- Tooling: Function call accuracy, parameter accuracy
"""

from collections import Counter
from dataclasses import dataclass
import json
import logging
import re
import string
from typing import Any, Dict, List, Optional, Union

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

    def _extract_answer(self, text: str) -> str:
        """
        Extract answer from text.

        Args:
            text: Text to extract answer from

        Returns:
            Extracted answer
        """
        patterns = [
            r"(?:answer|final answer|result)\s*(?:is|:)\s*([^\n.]+)",
            r"(?:the answer is|equals)\s+([^\n.]+)",
            r"^([A-D])\b",
            r"\b(\d+)\b"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return text.strip()

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

        pred_text = prediction
        if self.extract_answer:
            pred_text = self._extract_answer(pred_text)

        pred_text = self.normalize(pred_text)

        for ref in references:
            ref_text = ref
            if self.extract_answer:
                ref_text = self._extract_answer(ref_text)
            ref_text = self.normalize(ref_text)

            if pred_text == ref_text:
                return MetricResult(
                    name=self.name,
                    value=1.0,
                    max_value=1.0,
                    normalized=1.0,
                    metadata={
                        "correct": True,
                        "matched_reference": ref
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
