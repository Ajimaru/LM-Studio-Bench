"""
Reporting module for benchmark results.

Generates JSON and HTML reports from benchmark data.
"""

from datetime import datetime
from html import escape
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONReporter:
    """
    Generates JSON reports from benchmark data.
    """

    def __init__(self, schema_version: str = "1.0"):
        """
        Initialize JSON reporter.

        Args:
            schema_version: Report schema version
        """
        self.schema_version = schema_version

    def generate(
        self,
        report_data: Dict,
        output_path: Path
    ) -> bool:
        """
        Generate JSON report.

        Args:
            report_data: Benchmark report data
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            enriched_data = self._enrich_report(report_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(enriched_data, f, indent=2)

            logger.info(f"JSON report saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            return False

    def _enrich_report(self, report_data: Dict) -> Dict:
        """
        Enrich report with metadata.

        Args:
            report_data: Raw report data

        Returns:
            Enriched report data
        """
        return {
            "schema_version": self.schema_version,
            "generated_at": datetime.now().isoformat(),
            "report": report_data
        }


class HTMLReporter:
    """
    Generates HTML reports with visualizations.
    """

    def __init__(self):
        """
        Initialize HTML reporter.
        """
        pass

    def generate(
        self,
        report_data: Dict,
        output_path: Path
    ) -> bool:
        """
        Generate HTML report.

        Args:
            report_data: Benchmark report data
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            html = self._generate_html(report_data)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info(f"HTML report saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            return False

    def _generate_html(self, report_data: Dict) -> str:
        """
        Generate HTML content.

        Args:
            report_data: Report data

        Returns:
            HTML string
        """
        model_name = report_data.get("model_name", "Unknown Model")
        timestamp = report_data.get("timestamp", "")
        summary = report_data.get("summary", {})
        results = report_data.get("results", [])
        capabilities = report_data.get("capabilities", [])

        html_parts = [
            self._html_header(model_name),
            self._html_summary_section(summary, capabilities, timestamp),
            self._html_results_section(results),
            self._html_capability_breakdown(summary, results),
            self._html_footer()
        ]

        return "\n".join(html_parts)

    def _html_header(self, model_name: str) -> str:
        """
        Generate HTML header.

        Args:
            model_name: Name of the model

        Returns:
            HTML header string
        """
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Report: {escape(model_name)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 1.2em;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .score {{
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .score-high {{
            background: #d4edda;
            color: #155724;
        }}
        .score-medium {{
            background: #fff3cd;
            color: #856404;
        }}
        .score-low {{
            background: #f8d7da;
            color: #721c24;
        }}
        .capability-section {{
            margin: 20px 0;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Capability-Driven Benchmark Report</h1>
        <h2>Model: {escape(model_name)}</h2>
"""

    def _html_summary_section(
        self,
        summary: Dict,
        capabilities: List[str],
        timestamp: str
    ) -> str:
        """
        Generate summary section.

        Args:
            summary: Summary statistics
            capabilities: List of capabilities
            timestamp: Benchmark timestamp

        Returns:
            HTML summary section
        """
        total_tests = summary.get("total_tests", 0)
        successful = summary.get("successful_tests", 0)
        success_rate = summary.get("success_rate", 0) * 100
        avg_latency = summary.get("avg_latency_ms", 0)
        avg_quality = summary.get("avg_quality_score", 0)
        avg_throughput = summary.get("avg_throughput_tokens_per_sec")

        throughput_html = ""
        if avg_throughput is not None:
            throughput_html = (
                f'<div class="metric">'
                f'<span class="metric-label">Avg Throughput:</span> '
                f'<span class="metric-value">{avg_throughput:.2f} '
                f'tokens/sec</span></div>'
            )

        caps_str = ", ".join(capabilities) if capabilities else "general_text"

        return f"""
        <div class="timestamp">Generated: {timestamp}</div>
        <h3>Summary</h3>
        <div class="metric">
            <span class="metric-label">Capabilities:</span>
            <span class="metric-value">{caps_str}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Tests:</span>
            <span class="metric-value">{total_tests}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Successful:</span>
            <span class="metric-value">{successful}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Success Rate:</span>
            <span class="metric-value">{success_rate:.1f}%</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Latency:</span>
            <span class="metric-value">{avg_latency:.2f} ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Quality:</span>
            <span class="metric-value">{avg_quality:.3f}</span>
        </div>
        {throughput_html}
"""

    def _html_results_section(self, results: List[Dict]) -> str:
        """
        Generate results table section.

        Args:
            results: List of result dictionaries

        Returns:
            HTML results section
        """
        if not results:
            return "<h3>Results</h3><p>No results available.</p>"

        rows = []
        for result in results:
            test_id = result.get("test_id", "")
            capability = result.get("capability", "")
            latency = result.get("latency_ms", 0)
            tokens = result.get("tokens_generated") or "N/A"
            throughput = result.get("throughput")
            quality = result.get("quality_score", 0)
            error = result.get("error")

            throughput_str = (
                f"{throughput:.2f}" if throughput is not None else "N/A"
            )

            score_class = self._get_score_class(quality)
            status = (
                '<span class="error">Error</span>' if error
                else '<span class="success">Success</span>'
            )

            safe_test_id = escape(str(test_id), quote=True)
            safe_capability = escape(str(capability), quote=True)
            safe_tokens = escape(str(tokens), quote=True)
            safe_throughput = escape(str(throughput_str), quote=True)

            rows.append(f"""
            <tr>
                <td>{safe_test_id}</td>
                <td>{safe_capability}</td>
                <td>{status}</td>
                <td>{latency:.2f}</td>
                <td>{safe_tokens}</td>
                <td>{safe_throughput}</td>
                <td><span class="score {score_class}">{quality:.3f}</span></td>
            </tr>
            """)

        return f"""
        <h3>Test Results</h3>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Capability</th>
                    <th>Status</th>
                    <th>Latency (ms)</th>
                    <th>Tokens</th>
                    <th>Throughput</th>
                    <th>Quality</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
"""

    def _html_capability_breakdown(
        self,
        summary: Dict,
        results: List[Dict]
    ) -> str:
        """
        Generate capability breakdown section.

        Args:
            summary: Summary statistics
            results: Result list

        Returns:
            HTML capability breakdown
        """
        by_cap = summary.get("by_capability", {})
        if not by_cap:
            return ""

        sections = []
        for cap, stats in by_cap.items():
            test_count = stats.get("test_count", 0)
            avg_quality = stats.get("avg_quality_score", 0)
            success_rate = stats.get("success_rate", 0) * 100

            sections.append(f"""
            <div class="capability-section">
                <h4>{cap.replace("_", " ").title()}</h4>
                <div class="metric">
                    <span class="metric-label">Tests:</span>
                    <span class="metric-value">{test_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{success_rate:.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Quality:</span>
                    <span class="metric-value">{avg_quality:.3f}</span>
                </div>
            </div>
            """)

        return f"""
        <h3>Capability Breakdown</h3>
        {"".join(sections)}
"""

    def _html_footer(self) -> str:
        """
        Generate HTML footer.

        Returns:
            HTML footer string
        """
        return """
    </div>
</body>
</html>
"""

    def _get_score_class(self, score: float) -> str:
        """
        Get CSS class for score.

        Args:
            score: Quality score

        Returns:
            CSS class name
        """
        if score >= 0.7:
            return "score-high"
        if score >= 0.4:
            return "score-medium"
        return "score-low"


def generate_reports(
    report_data: Dict,
    output_dir: Path,
    formats: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Generate benchmark reports in multiple formats.

    Args:
        report_data: Benchmark report data
        output_dir: Output directory
        formats: List of formats (json, html) - defaults to all

    Returns:
        Dictionary mapping format to output path
    """
    if formats is None:
        formats = ["json", "html"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = report_data.get("model_name", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_name}_{timestamp}"

    outputs = {}

    if "json" in formats:
        json_path = output_dir / f"{base_name}.json"
        reporter = JSONReporter()
        if reporter.generate(report_data, json_path):
            outputs["json"] = json_path

    if "html" in formats:
        html_path = output_dir / f"{base_name}.html"
        reporter = HTMLReporter()
        if reporter.generate(report_data, html_path):
            outputs["html"] = html_path

    return outputs
