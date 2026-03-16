"""
Unit tests for metrics computation module.
"""

import json
import pytest

from bench.metrics import (
    AccuracyMetric,
    ExactMatchMetric,
    F1Metric,
    FunctionCallMetric,
    MetricResult,
    RougeMetric,
    aggregate_metrics
)


class TestExactMatchMetric:
    """
    Tests for ExactMatchMetric.
    """

    def test_exact_match_success(self):
        """
        Test exact match with matching strings.
        """
        metric = ExactMatchMetric()
        result = metric.compute("Paris", "Paris")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")

        if result.normalized != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.normalized}")

    def test_exact_match_failure(self):
        """
        Test exact match with non-matching strings.
        """
        metric = ExactMatchMetric()
        result = metric.compute("London", "Paris")

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0, got {result.value}")

        if result.normalized != 0.0:
            raise AssertionError(f"Expected 0.0, got {result.normalized}")

    def test_exact_match_normalization(self):
        """
        Test exact match with normalization.
        """
        metric = ExactMatchMetric(normalize_text=True)
        result = metric.compute("Paris!", "paris")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0 with normalization, got {result.value}")

    def test_exact_match_multiple_references(self):
        """
        Test exact match with multiple reference answers.
        """
        metric = ExactMatchMetric()
        result = metric.compute("London", ["Paris", "London", "Berlin"])

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")


class TestF1Metric:
    """
    Tests for F1Metric.
    """

    def test_f1_perfect_match(self):
        """
        Test F1 with perfect token match.
        """
        metric = F1Metric()
        result = metric.compute("The cat sat", "The cat sat")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")

    def test_f1_partial_match(self):
        """
        Test F1 with partial token overlap.
        """
        metric = F1Metric()
        result = metric.compute("The cat sat on mat", "The dog sat on rug")

        if result.value <= 0.0 or result.value >= 1.0:
            raise AssertionError(f"Expected partial match, got {result.value}")

    def test_f1_no_match(self):
        """
        Test F1 with no token overlap.
        """
        metric = F1Metric()
        result = metric.compute("completely different", "entirely unrelated")

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0, got {result.value}")


class TestRougeMetric:
    """
    Tests for RougeMetric.
    """

    def test_rouge1_perfect_match(self):
        """
        Test ROUGE-1 with perfect match.
        """
        metric = RougeMetric("rouge-1")
        result = metric.compute("The cat sat", "The cat sat")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")

    def test_rouge1_partial_match(self):
        """
        Test ROUGE-1 with partial match.
        """
        metric = RougeMetric("rouge-1")
        result = metric.compute("The cat", "The dog")

        if result.value <= 0.0 or result.value >= 1.0:
            raise AssertionError(f"Expected partial match, got {result.value}")

    def test_rouge2(self):
        """
        Test ROUGE-2 metric.
        """
        metric = RougeMetric("rouge-2")
        result = metric.compute("The big cat sat", "The big dog ran")

        if result.name != "rouge-2":
            raise AssertionError(f"Expected rouge-2, got {result.name}")

    def test_rougel_lcs(self):
        """
        Test ROUGE-L (longest common subsequence).
        """
        metric = RougeMetric("rouge-l")
        result = metric.compute("The cat sat on the mat", "The cat on mat")

        if result.value <= 0.0 or result.value > 1.0:
            raise AssertionError(f"Expected valid score, got {result.value}")

    def test_rouge_empty_prediction(self):
        """
        Test ROUGE with empty prediction.
        """
        metric = RougeMetric("rouge-1")
        result = metric.compute("", "The cat sat")

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0 for empty prediction, got {result.value}")


class TestAccuracyMetric:
    """
    Tests for AccuracyMetric.
    """

    def test_accuracy_correct(self):
        """
        Test accuracy with correct answer.
        """
        metric = AccuracyMetric()
        result = metric.compute("Paris", "Paris")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")

    def test_accuracy_incorrect(self):
        """
        Test accuracy with incorrect answer.
        """
        metric = AccuracyMetric()
        result = metric.compute("London", "Paris")

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0, got {result.value}")

    def test_accuracy_extract_answer(self):
        """
        Test accuracy with answer extraction.
        """
        metric = AccuracyMetric(extract_answer=True)
        result = metric.compute(
            "The answer is: Paris",
            "Paris"
        )

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0 with extraction, got {result.value}")

    def test_accuracy_multiple_choice(self):
        """
        Test accuracy with multiple choice format.
        """
        metric = AccuracyMetric(extract_answer=True)
        result = metric.compute("A", "A")

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")


class TestFunctionCallMetric:
    """
    Tests for FunctionCallMetric.
    """

    def test_function_call_perfect_match(self):
        """
        Test function call with perfect match.
        """
        metric = FunctionCallMetric()

        prediction = json.dumps({
            "function": "get_weather",
            "parameters": {"location": "New York"}
        })

        reference = json.dumps({
            "function": "get_weather",
            "parameters": {"location": "New York"}
        })

        result = metric.compute(prediction, reference)

        if result.value != 1.0:
            raise AssertionError(f"Expected 1.0, got {result.value}")

    def test_function_call_wrong_function(self):
        """
        Test function call with wrong function.
        """
        metric = FunctionCallMetric()

        prediction = json.dumps({
            "function": "calculate",
            "parameters": {"expression": "2+2"}
        })

        reference = json.dumps({
            "function": "get_weather",
            "parameters": {"location": "New York"}
        })

        result = metric.compute(prediction, reference)

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0, got {result.value}")

    def test_function_call_partial_params(self):
        """
        Test function call with partial parameter match.
        """
        metric = FunctionCallMetric()

        prediction = json.dumps({
            "function": "get_weather",
            "parameters": {"location": "New York", "extra": "value"}
        })

        reference = json.dumps({
            "function": "get_weather",
            "parameters": {"location": "New York"}
        })

        result = metric.compute(prediction, reference)

        if result.value <= 0.5:
            raise AssertionError(
                f"Expected score > 0.5 for correct function, got {result.value}"
            )

    def test_function_call_invalid_json(self):
        """
        Test function call with invalid JSON.
        """
        metric = FunctionCallMetric()

        result = metric.compute("not json", '{"function": "test"}')

        if result.value != 0.0:
            raise AssertionError(f"Expected 0.0 for invalid JSON, got {result.value}")


def test_aggregate_metrics():
    """
    Test metric aggregation.
    """
    metrics = [
        MetricResult("metric1", 0.8, 1.0, 0.8),
        MetricResult("metric2", 0.6, 1.0, 0.6),
        MetricResult("metric3", 1.0, 1.0, 1.0)
    ]

    score = aggregate_metrics(metrics)

    expected = (0.8 + 0.6 + 1.0) / 3
    if abs(score - expected) > 0.01:
        raise AssertionError(f"Expected {expected}, got {score}")


def test_aggregate_metrics_with_weights():
    """
    Test metric aggregation with custom weights.
    """
    metrics = [
        MetricResult("metric1", 0.8, 1.0, 0.8),
        MetricResult("metric2", 0.6, 1.0, 0.6)
    ]

    weights = {
        "metric1": 0.7,
        "metric2": 0.3
    }

    score = aggregate_metrics(metrics, weights)

    expected = (0.8 * 0.7 + 0.6 * 0.3) / (0.7 + 0.3)
    if abs(score - expected) > 0.01:
        raise AssertionError(f"Expected {expected}, got {score}")


def test_aggregate_metrics_empty():
    """
    Test metric aggregation with empty list.
    """
    score = aggregate_metrics([])

    if score != 0.0:
        raise AssertionError(f"Expected 0.0 for empty metrics, got {score}")
