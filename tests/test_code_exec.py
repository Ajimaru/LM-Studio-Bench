"""Tests for core/code_exec.py - sandboxed execution of generated code."""

import pytest

from core.code_exec import ExecutionResult, extract_code, run_code_test


class TestExtractCode:
    """Tests for extract_code()."""

    def test_extracts_fenced_python_block(self):
        """A ```python block is preferred over surrounding prose."""
        response = "Sure!\n```python\ndef f():\n    return 1\n```\nDone."
        assert extract_code(response) == "def f():\n    return 1"

    def test_extracts_unlabelled_fence(self):
        """Fences without a language tag still count."""
        assert extract_code("```\nx = 1\n```") == "x = 1"

    def test_picks_longest_block(self):
        """Models often show a short example next to the real answer."""
        response = "```python\nprint(1)\n```\n```python\ndef f():\n    return 2\n```"
        assert extract_code(response) == "def f():\n    return 2"

    def test_falls_back_to_whole_response(self):
        """Bare code without fences is used as-is."""
        assert extract_code("def f():\n    return 1") == "def f():\n    return 1"

    def test_empty_response(self):
        """Nothing in, nothing out."""
        assert extract_code("") == ""


class TestRunCodeTest:
    """Tests for run_code_test()."""

    def test_passing_checks(self):
        """Correct code with holding assertions passes."""
        result = run_code_test("def add(a, b):\n    return a + b",
                               "assert add(2, 3) == 5")
        assert result.passed is True
        assert result.timed_out is False

    def test_failing_assertion(self):
        """A wrong implementation is reported with its error."""
        result = run_code_test("def add(a, b):\n    return a - b",
                               "assert add(2, 3) == 5")
        assert result.passed is False
        assert "AssertionError" in result.detail

    def test_syntax_error(self):
        """Unparsable code fails rather than raising."""
        result = run_code_test("def broken(:", "assert True")
        assert result.passed is False

    def test_missing_function(self):
        """Checks calling an undefined name fail cleanly."""
        result = run_code_test("x = 1", "assert add(1, 2) == 3")
        assert result.passed is False
        assert "NameError" in result.detail

    def test_empty_response_is_not_passed(self):
        """An empty answer must never count as success."""
        result = run_code_test("", "assert True")
        assert result.passed is False
        assert result.detail == "No code in response"

    def test_infinite_loop_times_out(self):
        """Runaway code is killed by the wall-clock limit."""
        result = run_code_test("while True:\n    pass", "assert True",
                               timeout=2)
        assert result.passed is False
        assert result.timed_out is True

    def test_memory_hog_is_capped(self):
        """Excessive allocation hits the address-space limit."""
        result = run_code_test("x = bytearray(2 * 1024 ** 3)", "assert True",
                               timeout=15)
        assert result.passed is False

    def test_code_runs_in_temporary_directory(self, tmp_path):
        """Writes land in the sandbox, not in the project."""
        code = "open('side_effect.txt', 'w').write('x')"
        result = run_code_test(code, "assert True")
        assert result.passed is True
        assert not (tmp_path / "side_effect.txt").exists()

    def test_multiple_checks_all_must_pass(self):
        """A list of checks fails as soon as one does."""
        code = "def f(x):\n    return x if x > 0 else None"
        result = run_code_test(code, "assert f(1) == 1")
        assert result.passed is True
        failing = run_code_test(code, "assert f(-1) == 0")
        assert failing.passed is False

    def test_stdout_is_captured(self):
        """Printed output is available for reports."""
        result = run_code_test("print('hello')", "assert True")
        assert result.passed is True
        assert "hello" in result.stdout

    def test_returns_execution_result(self):
        """The public type is stable for callers."""
        assert isinstance(run_code_test("x = 1", "assert True"),
                          ExecutionResult)


class TestCodeExecutionMetric:
    """Tests for the metric wrapper around run_code_test()."""

    def test_scores_one_when_all_checks_pass(self):
        """Working code scores full marks."""
        from cli.metrics import CodeExecutionMetric
        metric = CodeExecutionMetric(timeout=10)
        result = metric.compute(
            "```python\ndef add(a, b):\n    return a + b\n```",
            ["assert add(1, 1) == 2", "assert add(0, 0) == 0"],
        )
        assert result.normalized == 1.0
        assert result.metadata["checks"] == 2

    def test_scores_zero_on_first_failing_check(self):
        """One broken edge case is enough to fail."""
        from cli.metrics import CodeExecutionMetric
        metric = CodeExecutionMetric(timeout=10)
        result = metric.compute(
            "def add(a, b):\n    return a + b",
            ["assert add(1, 1) == 2", "assert add(1, 1) == 3"],
        )
        assert result.normalized == 0.0
        assert result.metadata["passed"] is False

    def test_prose_answer_scores_zero(self):
        """An explanation without code is not a solution."""
        from cli.metrics import CodeExecutionMetric
        metric = CodeExecutionMetric(timeout=10)
        result = metric.compute(
            "You should iterate over the list and compare each element.",
            "assert add(1, 1) == 2",
        )
        assert result.normalized == 0.0

    def test_no_checks_scores_zero(self):
        """Without assertions there is nothing to verify."""
        from cli.metrics import CodeExecutionMetric
        metric = CodeExecutionMetric()
        result = metric.compute("def f(): pass", [])
        assert result.normalized == 0.0


@pytest.mark.parametrize("task_id", [
    "code_001", "code_002", "code_003", "code_004", "code_005",
    "code_006", "code_007",
])
def test_shipped_code_samples_have_runnable_checks(task_id):
    """Every shipped task must be solvable, or the benchmark is broken.

    Guards against a check that can never pass (typo, wrong expectation):
    a known-good reference implementation is run against the real checks.
    """
    import json
    from pathlib import Path

    samples = json.loads(
        Path("tests/data/text/code_samples.json").read_text(encoding="utf-8")
    )
    sample = next(item for item in samples if item["id"] == task_id)

    reference_solutions = {
        "code_001": (
            "def merge_intervals(intervals):\n"
            "    if not intervals:\n"
            "        return []\n"
            "    ordered = sorted(intervals, key=lambda pair: pair[0])\n"
            "    merged = [list(ordered[0])]\n"
            "    for start, end in ordered[1:]:\n"
            "        if start <= merged[-1][1]:\n"
            "            merged[-1][1] = max(merged[-1][1], end)\n"
            "        else:\n"
            "            merged.append([start, end])\n"
            "    return merged\n"
        ),
        "code_002": (
            "import re\n"
            "def parse_duration(text):\n"
            "    match = re.fullmatch(\n"
            "        r'(?:(\\d+)h)?(?:(\\d+)m)?(?:(\\d+)s)?', text or ''\n"
            "    )\n"
            "    if not match or not any(match.groups()):\n"
            "        raise ValueError(text)\n"
            "    hours, minutes, seconds = (\n"
            "        int(value) if value else 0 for value in match.groups()\n"
            "    )\n"
            "    return hours * 3600 + minutes * 60 + seconds\n"
        ),
        "code_003": (
            "def second_largest(numbers):\n"
            "    unique = sorted(set(numbers))\n"
            "    if len(unique) < 2:\n"
            "        return None\n"
            "    return unique[-2]\n"
        ),
        "code_004": (
            "def group_by_key(records, key):\n"
            "    grouped = {}\n"
            "    for record in records:\n"
            "        if key not in record:\n"
            "            continue\n"
            "        grouped.setdefault(record[key], []).append(record)\n"
            "    return grouped\n"
        ),
        "code_006": (
            "class RateLimiter:\n"
            "    def __init__(self, max_calls, per_seconds):\n"
            "        self.max_calls = max_calls\n"
            "        self.per_seconds = per_seconds\n"
            "        self.calls = []\n"
            "    def allow(self, timestamp):\n"
            "        self.calls = [\n"
            "            t for t in self.calls\n"
            "            if timestamp - t < self.per_seconds\n"
            "        ]\n"
            "        if len(self.calls) < self.max_calls:\n"
            "            self.calls.append(timestamp)\n"
            "            return True\n"
            "        return False\n"
        ),
        "code_007": (
            "def flatten(value):\n"
            "    out = []\n"
            "    def walk(item):\n"
            "        if isinstance(item, (list, tuple)):\n"
            "            for sub in item:\n"
            "                walk(sub)\n"
            "        else:\n"
            "            out.append(item)\n"
            "    walk(value)\n"
            "    return out\n"
        ),
        "code_005": (
            "def truncate_words(text, limit):\n"
            "    if len(text) <= limit:\n"
            "        return text\n"
            "    kept = []\n"
            "    for word in text.split():\n"
            "        candidate = ' '.join(kept + [word])\n"
            "        if len(candidate) + 1 > limit:\n"
            "            break\n"
            "        kept.append(word)\n"
            "    return ' '.join(kept) + '…'\n"
        ),
    }

    solution = reference_solutions[task_id]
    for check in sample["checks"]:
        result = run_code_test(solution, check, timeout=15)
        assert result.passed, f"{task_id}: {result.detail}\n{check}"
