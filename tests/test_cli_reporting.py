"""Tests for cli/reporting.py - report generation."""

from cli.reporting import HTMLReporter, JSONReporter, sanitize_report_name


class TestSanitizeReportName:
    """Tests for sanitize_report_name function."""

    def test_empty_name_returns_default(self):
        """Empty model name returns 'model' default."""
        result = sanitize_report_name("")
        assert result == "model"

    def test_whitespace_only_returns_default(self):
        """Whitespace-only name returns 'model' default."""
        result = sanitize_report_name("   ")
        assert result == "model"

    def test_none_input_returns_default(self):
        """None input is converted to string."""
        result = sanitize_report_name(None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_simple_alphanumeric_unchanged(self):
        """Simple alphanumeric names pass through unchanged."""
        result = sanitize_report_name("llama2-7b")
        assert result == "llama2-7b"

    def test_removes_forward_slashes(self):
        """Forward slashes are replaced with underscores."""
        result = sanitize_report_name("models/llama/7b")
        assert "/" not in result
        assert "_" in result

    def test_removes_backslashes(self):
        """Backslashes are replaced with underscores."""
        result = sanitize_report_name("models\\llama\\7b")
        assert "\\" not in result

    def test_removes_invalid_chars(self):
        """Invalid filename characters are replaced."""
        result = sanitize_report_name("model@#$%^&*()")
        assert "@" not in result
        assert "#" not in result
        assert " " not in result

    def test_strips_leading_dots(self):
        """Leading dots are stripped."""
        result = sanitize_report_name("...model")
        assert not result.startswith(".")

    def test_strips_trailing_underscores(self):
        """Trailing underscores are stripped."""
        result = sanitize_report_name("model___")
        assert not result.endswith("_")

    def test_preserves_dots_in_middle(self):
        """Dots in the middle of name are preserved."""
        result = sanitize_report_name("model.v2.0")
        assert ".v2.0" in result or "v2_0" in result

    def test_handles_special_slash_patterns(self):
        """Paths with multiple slashes are handled."""
        result = sanitize_report_name("/home/user/models/llama")
        assert "/" not in result


class TestJSONReporter:
    """Tests for JSONReporter class."""

    def test_initializer_sets_schema_version(self):
        """JSONReporter stores schema version."""
        reporter = JSONReporter(schema_version="2.0")
        assert reporter.schema_version == "2.0"

    def test_initializer_default_schema(self):
        """JSONReporter defaults to schema 1.0."""
        reporter = JSONReporter()
        assert reporter.schema_version == "1.0"

    def test_generate_creates_json_file(self, tmp_path):
        """generate() creates a JSON file."""
        reporter = JSONReporter()
        output_file = tmp_path / "report.json"
        data = {"model": "test", "results": []}

        success = reporter.generate(data, output_file)

        assert success is True
        assert output_file.exists()

    def test_generate_returns_false_on_io_error(self, tmp_path):
        """generate() returns False when write fails."""
        reporter = JSONReporter()
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        read_only_dir.chmod(0o444)

        output_file = read_only_dir / "report.json"
        data = {"test": "data"}

        try:
            success = reporter.generate(data, output_file)
        finally:
            read_only_dir.chmod(0o755)

        assert success is False


class TestHTMLReporter:
    """Tests for HTMLReporter class."""

    def test_render_returns_html_string(self):
        """render() returns HTML content as string."""
        reporter = HTMLReporter()
        report_data = {
            "model_name": "Test Model",
            "timestamp": "2026-03-22T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        html = reporter.render(report_data)

        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "Test Model" in html

    def test_render_includes_model_name(self):
        """Rendered HTML includes model name."""
        reporter = HTMLReporter()
        report_data = {
            "model_name": "MyAwesomeModel",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        html = reporter.render(report_data)

        assert "MyAwesomeModel" in html

    def test_generate_creates_html_file(self, tmp_path):
        """generate() creates an HTML file."""
        reporter = HTMLReporter()
        output_file = tmp_path / "report.html"
        data = {
            "model_name": "test",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        success = reporter.generate(data, output_file)

        assert success is True
        assert output_file.exists()
        assert output_file.read_text().startswith("<!DOCTYPE")

    def test_generate_html_for_model_with_summary(self, tmp_path):
        """generate() includes summary metrics in HTML."""
        reporter = HTMLReporter()
        output_file = tmp_path / "report.html"
        data = {
            "model_name": "llama",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {
                "total_tests": 10,
                "successful_tests": 9,
                "avg_latency_ms": 100.5
            },
            "results": [],
            "capabilities": ["reasoning"]
        }

        success = reporter.generate(data, output_file)

        assert success is True
        html_content = output_file.read_text()
        assert "llama" in html_content.lower()

    def test_html_summary_with_throughput(self):
        """_html_summary_section includes throughput when present."""
        reporter = HTMLReporter()
        summary = {
            "total_tests": 5,
            "successful_tests": 5,
            "success_rate": 1.0,
            "avg_latency_ms": 50.0,
            "avg_quality_score": 0.95,
            "avg_throughput_tokens_per_sec": 120.5
        }

        result = reporter._html_summary_section(
            summary, ["reasoning"], "2026-01-01T00:00:00"
        )

        assert "120.5" in result
        assert "tokens/sec" in result

    def test_html_summary_without_throughput(self):
        """_html_summary_section handles missing throughput gracefully."""
        reporter = HTMLReporter()
        summary = {
            "total_tests": 5,
            "successful_tests": 5,
            "success_rate": 1.0,
            "avg_latency_ms": 50.0,
            "avg_quality_score": 0.95,
            "avg_throughput_tokens_per_sec": None
        }

        result = reporter._html_summary_section(
            summary, ["reasoning"], "2026-01-01T00:00:00"
        )

        assert "tokens/sec" not in result
        assert "reasoning" in result

    def test_html_results_empty(self):
        """_html_results_section handles empty results."""
        reporter = HTMLReporter()

        result = reporter._html_results_section([])

        assert "No results available" in result

    def test_html_results_with_data(self):
        """_html_results_section includes test results."""
        reporter = HTMLReporter()
        results = [
            {
                "test_id": "test1",
                "capability": "reasoning",
                "latency_ms": 100.0,
                "tokens_generated": 50,
                "throughput": 125.5,
                "quality_score": 0.85,
                "error": None
            },
            {
                "test_id": "test2",
                "capability": "coding",
                "latency_ms": 150.0,
                "tokens_generated": None,
                "throughput": None,
                "quality_score": 0.65,
                "error": "timeout"
            }
        ]

        result = reporter._html_results_section(results)

        assert "test1" in result
        assert "test2" in result
        assert "Success" in result
        assert "Error" in result

    def test_score_class_high(self):
        """_get_score_class returns high class for score >= 0.7."""
        reporter = HTMLReporter()

        assert reporter._get_score_class(0.7) == "score-high"
        assert reporter._get_score_class(0.95) == "score-high"
        assert reporter._get_score_class(1.0) == "score-high"

    def test_score_class_medium(self):
        """_get_score_class returns medium class for 0.4 <= score < 0.7."""
        reporter = HTMLReporter()

        assert reporter._get_score_class(0.4) == "score-medium"
        assert reporter._get_score_class(0.5) == "score-medium"
        assert reporter._get_score_class(0.69) == "score-medium"

    def test_score_class_low(self):
        """_get_score_class returns low class for score < 0.4."""
        reporter = HTMLReporter()

        assert reporter._get_score_class(0.0) == "score-low"
        assert reporter._get_score_class(0.39) == "score-low"
        assert reporter._get_score_class(0.2) == "score-low"

    def test_html_capability_breakdown_empty(self):
        """_html_capability_breakdown handles empty breakdown."""
        reporter = HTMLReporter()
        summary = {"by_capability": {}}

        result = reporter._html_capability_breakdown(summary)

        assert result == ""

    def test_html_capability_breakdown_with_data(self):
        """_html_capability_breakdown includes capability stats."""
        reporter = HTMLReporter()
        summary = {
            "by_capability": {
                "reasoning": {
                    "test_count": 10,
                    "avg_quality_score": 0.85,
                    "success_rate": 0.9
                },
                "coding": {
                    "test_count": 5,
                    "avg_quality_score": 0.75,
                    "success_rate": 0.8
                }
            }
        }

        result = reporter._html_capability_breakdown(summary)

        assert "Reasoning" in result
        assert "Coding" in result
        assert "10" in result
        assert "90.0%" in result

    def test_html_footer(self):
        """_html_footer returns valid HTML footer."""
        reporter = HTMLReporter()

        footer = reporter._html_footer()

        assert "</html>" in footer
        assert "</body>" in footer
        assert "</div>" in footer


class TestExportReports:
    """Tests for report export functions."""

    def test_export_reports_json_only(self, tmp_path):
        """export_reports creates JSON file when json format requested."""
        from cli.reporting import export_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-01-01T00:00:00"
        }

        outputs = export_reports(
            report_data, tmp_path, formats=["json"]
        )

        assert "json" in outputs
        assert outputs["json"].exists()
        assert outputs["json"].suffix == ".json"

    def test_export_reports_html_only(self, tmp_path):
        """export_reports creates HTML file when html format requested."""
        from cli.reporting import export_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = export_reports(
            report_data, tmp_path, formats=["html"]
        )

        assert "html" in outputs
        assert outputs["html"].exists()
        assert outputs["html"].suffix == ".html"

    def test_export_reports_both_formats(self, tmp_path):
        """export_reports creates both formats when requested."""
        from cli.reporting import export_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = export_reports(
            report_data, tmp_path, formats=["json", "html"]
        )

        assert len(outputs) == 2
        assert "json" in outputs
        assert "html" in outputs

    def test_export_reports_default_formats(self, tmp_path):
        """export_reports defaults to both json and html formats."""
        from cli.reporting import export_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = export_reports(report_data, tmp_path)

        assert len(outputs) == 2
        assert "json" in outputs
        assert "html" in outputs

    def test_export_reports_custom_report_stem(self, tmp_path):
        """export_reports uses custom report_stem in filenames."""
        from cli.reporting import export_reports

        report_data = {
            "model_name": "test-model",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = export_reports(
            report_data,
            tmp_path,
            formats=["json"],
            report_stem="custom-name"
        )

        json_file = outputs["json"]
        assert "custom-name" in json_file.name

    def test_export_reports_creates_directory(self, tmp_path):
        """export_reports creates output directory if not exists."""
        from cli.reporting import export_reports

        nested_dir = tmp_path / "deep" / "nested" / "path"
        report_data = {
            "model_name": "test",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = export_reports(
            report_data, nested_dir, formats=["json"]
        )

        assert nested_dir.exists()
        assert "json" in outputs

    def test_generate_reports_backward_compatible(self, tmp_path):
        """generate_reports wrapper calls export_reports correctly."""
        from cli.reporting import generate_reports

        report_data = {
            "model_name": "test",
            "timestamp": "2026-01-01T00:00:00",
            "summary": {},
            "results": [],
            "capabilities": []
        }

        outputs = generate_reports(
            report_data, tmp_path, formats=["json"]
        )

        assert "json" in outputs
        assert outputs["json"].exists()
