"""
Unit tests for metrics computation module.
"""

import json

from cli.metrics import (
    AccuracyMetric,
    ExactMatchMetric,
    F1Metric,
    FunctionCallMetric,
    MetricResult,
    RougeMetric,
    aggregate_metrics,
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


class TestAccuracyAnswerExtraction:
    """Accuracy must credit correct answers regardless of how they are worded.

    These cases are taken from real model responses that the previous
    first-number/whole-text matching scored as wrong.
    """

    def test_yes_no_answer_with_explanation(self):
        """A correct yes/no answer followed by reasoning counts."""
        metric = AccuracyMetric()
        response = (
            "Yes, if all A are B, and all B are C, then it logically follows "
            "that all A are also C. This is basic syllogistic reasoning."
        )
        assert metric.compute(response, "Yes").normalized == 1.0

    def test_answer_stated_in_closing_sentence(self):
        """'…the average speed is 60 mph.' is a correct answer."""
        metric = AccuracyMetric()
        response = (
            "Average Speed = Distance / Time\n"
            "Average Speed = 120 miles / 2 hours\n\n"
            "Therefore, the train's average speed is 60 mph."
        )
        assert metric.compute(response, "60").normalized == 1.0

    def test_question_restatement_does_not_count(self):
        """A number copied from the question is not an answer."""
        metric = AccuracyMetric()
        response = "The train went 120 miles in 2 hours. I am not sure."
        assert metric.compute(response, "60").normalized == 0.0

    def test_wrong_yes_no_answer(self):
        """A contradicting answer stays wrong."""
        metric = AccuracyMetric()
        assert metric.compute("No, that does not follow.", "Yes").normalized == 0.0

    def test_step_by_step_solution(self):
        """The final assignment wins over intermediate values."""
        metric = AccuracyMetric()
        response = "3x + 5 = 20\n3x = 15\nx = 5\n\nTherefore x = 5."
        assert metric.compute(response, "5").normalized == 1.0

    def test_boxed_answer(self):
        """LaTeX-style boxed answers are recognized."""
        metric = AccuracyMetric()
        assert metric.compute("So we get \\boxed{42}.", "42").normalized == 1.0

    def test_extract_answer_returns_best_candidate(self):
        """_extract_answer still returns a single string for callers."""
        metric = AccuracyMetric()
        assert metric._extract_answer("The answer is 7.") == "7"


class TestMetricWeighting:
    """aggregate_metrics must honour configured weights."""

    def test_weights_change_the_score(self):
        """A dominant metric pulls the aggregate towards itself."""
        from cli.metrics import aggregate_metrics

        results = [
            MetricResult("accuracy", 1.0, 1.0, 1.0),
            MetricResult("exact_match", 0.0, 1.0, 0.0),
            MetricResult("f1", 0.0, 1.0, 0.0),
        ]
        unweighted = aggregate_metrics(results)
        weighted = aggregate_metrics(
            results, {"accuracy": 0.8, "exact_match": 0.1, "f1": 0.1}
        )
        assert round(unweighted, 3) == 0.333
        assert round(weighted, 3) == 0.8


class TestShippedReasoningSamples:
    """The reasoning references must be extractable from real answers.

    A reference that no phrasing can match would score every model at zero
    without anyone noticing, which is exactly the failure mode these tests
    were written after.
    """

    @staticmethod
    def _samples():
        """Load the shipped reasoning test cases."""
        import json
        from pathlib import Path
        return json.loads(
            Path("tests/data/text/reasoning_samples.json").read_text(
                encoding="utf-8"
            )
        )

    def test_every_reference_is_recognized_in_verbose_answers(self):
        """A correct answer wrapped in reasoning must score 1.0."""
        metric = AccuracyMetric()
        templates = [
            "Let me work through this.\n\n{r}\n\nTherefore, the answer is {a}.",
            "{r}\n\nSo the final answer is {a}",
            "Step by step:\n{r}\nAnswer: {a}",
        ]
        for item in self._samples():
            for template in templates:
                response = template.format(
                    r=item["reasoning"], a=item["reference"]
                )
                score = metric.compute(response, item["reference"]).normalized
                assert score == 1.0, (
                    f"{item['id']} not recognized in: {response[:80]}"
                )

    def test_wrong_answers_are_not_credited(self):
        """Stating a different value must not score."""
        metric = AccuracyMetric()
        for item in self._samples():
            response = "After thinking about it, the answer is 999999."
            if item["reference"] == "999999":
                continue
            assert metric.compute(response, item["reference"]).normalized == 0.0

    def test_samples_are_not_trivial(self):
        """Guard the set against sliding back to one-liners."""
        samples = self._samples()
        assert len(samples) >= 8
        categories = {item.get("category") for item in samples}
        assert {"trap", "multi_step", "logic"} <= categories
