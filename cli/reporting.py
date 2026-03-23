"""
Reporting module for benchmark results.

Generates JSON and HTML reports from benchmark data.
"""

from datetime import datetime
from html import escape
import json
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_INVALID_REPORT_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_report_name(model_name: object) -> str:
    """Return a safe file stem for generated report files."""
    raw_name = str(model_name).strip()
    if not raw_name:
        return "model"

    safe_name = raw_name.replace("/", "_").replace("\\", "_")
    safe_name = _INVALID_REPORT_NAME_RE.sub("_", safe_name)
    safe_name = safe_name.strip("._")
    return safe_name or "model"


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

            logger.info("JSON report saved to: %s", output_path)
            return True

        except (OSError, TypeError, ValueError) as e:
            logger.error("Error generating JSON report: %s", e)
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
            html = self.render(report_data)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            logger.info("HTML report saved to: %s", output_path)
            return True

        except (OSError, TypeError, ValueError) as e:
            logger.error("Error generating HTML report: %s", e)
            return False

    def render(self, report_data: Dict) -> str:
        """Render report data as HTML without writing it to disk."""
        return self._generate_html(report_data)

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
            self._html_kpi_cards(summary),
            self._html_capability_bars(summary),
            self._html_results_section(results),
            self._html_capability_breakdown(summary),
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
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 16px 0 24px;
        }}
        .kpi-card {{
            background: linear-gradient(135deg, #1f4a89 0%, #2f6cc4 100%);
            color: #fff;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
        }}
        .kpi-title {{
            font-size: 0.8rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .kpi-value {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 4px;
        }}
        .cap-bars {{
            margin: 20px 0;
            padding: 14px;
            border-radius: 8px;
            background: #f9fafb;
            border: 1px solid #e7eaee;
        }}
        .cap-row {{
            margin: 12px 0;
        }}
        .cap-head {{
            display: flex;
            justify-content: space-between;
            font-size: 0.92rem;
            margin-bottom: 6px;
            color: #34495e;
        }}
        .bar-wrap {{
            height: 10px;
            border-radius: 999px;
            background: #e5ebf2;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #00a36c 0%, #2ecc71 100%);
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

        hardware_html = ""
        temp_avg = summary.get("temp_celsius_avg")
        power_avg = summary.get("power_watts_avg")
        vram_avg = summary.get("vram_gb_avg")
        cpu_avg = summary.get("cpu_percent_avg")
        ram_avg = summary.get("ram_gb_avg")

        if any(
            value is not None
            for value in [
                temp_avg,
                power_avg,
                vram_avg,
                cpu_avg,
                ram_avg,
            ]
        ):
            hardware_html = """
        <h3>Hardware Profiling</h3>
"""
            if temp_avg is not None:
                hardware_html += (
                    '<div class="metric">'
                    '<span class="metric-label">GPU Temp Avg:</span> '
                    f'<span class="metric-value">{temp_avg:.2f} C</span></div>'
                )
            if power_avg is not None:
                hardware_html += (
                    '<div class="metric">'
                    '<span class="metric-label">GPU Power Avg:</span> '
                    f'<span class="metric-value">{power_avg:.2f} W</span></div>'
                )
            if vram_avg is not None:
                hardware_html += (
                    '<div class="metric">'
                    '<span class="metric-label">VRAM Avg:</span> '
                    f'<span class="metric-value">{vram_avg:.2f} GB</span></div>'
                )
            if cpu_avg is not None:
                hardware_html += (
                    '<div class="metric">'
                    '<span class="metric-label">CPU Avg:</span> '
                    f'<span class="metric-value">{cpu_avg:.2f}%</span></div>'
                )
            if ram_avg is not None:
                hardware_html += (
                    '<div class="metric">'
                    '<span class="metric-label">RAM Avg:</span> '
                    f'<span class="metric-value">{ram_avg:.2f} GB</span></div>'
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
            <span class="metric-value">{avg_latency / 1000:.2f} s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Quality:</span>
            <span class="metric-value">{avg_quality:.3f}</span>
        </div>
        {throughput_html}
        {hardware_html}
"""

    def _html_kpi_cards(self, summary: Dict) -> str:
        """Generate compact KPI cards for key aggregate metrics."""
        total_tests = summary.get("total_tests") or 0
        successful = summary.get("successful_tests") or 0
        success_rate = (summary.get("success_rate") or 0) * 100
        avg_latency_ms = summary.get("avg_latency_ms")
        avg_quality = summary.get("avg_quality_score")
        avg_throughput = summary.get("avg_throughput_tokens_per_sec")

        def _fmt(value: object, suffix: str = "") -> str:
            if isinstance(value, (int, float)):
                return f"{value:.2f}{suffix}"
            return f"N/A{suffix}"

        cards = [
            ("Total Tests", str(total_tests)),
            ("Successful", str(successful)),
            ("Success Rate", f"{success_rate:.1f}%"),
            ("Avg Latency", _fmt(avg_latency_ms, " ms")),
            ("Avg Throughput", _fmt(avg_throughput, " tok/s")),
            ("Avg Quality", _fmt(avg_quality)),
        ]

        card_html = []
        for title, value in cards:
            card_html.append(
                "<div class='kpi-card'>"
                f"<div class='kpi-title'>{escape(title)}</div>"
                f"<div class='kpi-value'>{escape(value)}</div>"
                "</div>"
            )

        return (
            "<h3>Key Performance Indicators</h3>"
            f"<div class='kpi-grid'>{''.join(card_html)}</div>"
        )

    def _html_capability_bars(self, summary: Dict) -> str:
        """Generate horizontal capability success bars for fast comparison."""
        by_capability = summary.get("by_capability") or {}
        if not isinstance(by_capability, dict) or not by_capability:
            return ""

        rows = []
        for capability, cap_stats in by_capability.items():
            if not isinstance(cap_stats, dict):
                continue
            success_rate = float(cap_stats.get("success_rate") or 0.0)
            success_percent = max(0.0, min(100.0, success_rate * 100.0))
            test_count = int(cap_stats.get("test_count") or 0)
            label = capability.replace("_", " ").title()
            rows.append(
                "<div class='cap-row'>"
                "<div class='cap-head'>"
                f"<span>{escape(label)} ({test_count} tests)</span>"
                f"<span>{success_percent:.1f}%</span>"
                "</div>"
                "<div class='bar-wrap'>"
                f"<div class='bar-fill' style='width: {success_percent:.1f}%;'></div>"
                "</div>"
                "</div>"
            )

        if not rows:
            return ""

        return (
            "<h3>Capability Success Overview</h3>"
            f"<div class='cap-bars'>{''.join(rows)}</div>"
        )

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
                <td>{latency / 1000:.2f}</td>
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
                    <th>Latency (s)</th>
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
        summary: Dict
    ) -> str:
        """
        Generate capability breakdown section.

        Args:
            summary: Summary statistics

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


def _convert_latency_ms_to_seconds(report_data: Dict) -> Dict:
    """
    Convert latency values from ms to seconds in report data.

    Args:
        report_data: Report data with latency in milliseconds

    Returns:
        Report data with latency converted to seconds
    """
    result = report_data.copy()

    if "summary" in result:
        summary = result["summary"].copy()
        if "avg_latency_ms" in summary:
            summary["avg_latency_ms"] = round(summary["avg_latency_ms"] / 1000, 2)
        result["summary"] = summary

    if "results" in result:
        results = []
        for item in result["results"]:
            item_copy = item.copy()
            if "latency_ms" in item_copy:
                item_copy["latency_ms"] = round(item_copy["latency_ms"] / 1000, 2)
            results.append(item_copy)
        result["results"] = results

    return result


def export_reports(
    report_data: Dict,
    output_dir: Path,
    formats: Optional[List[str]] = None,
    report_stem: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Generate benchmark reports in multiple formats.

    Args:
        report_data: Benchmark report data
        output_dir: Output directory
        formats: List of formats (json, html) - defaults to all
        report_stem: Optional sanitized file stem for report filenames

    Returns:
        Dictionary mapping format to output path
    """
    if formats is None:
        formats = ["json", "html"]

    report_data = _convert_latency_ms_to_seconds(report_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = report_data.get("model_name", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = sanitize_report_name(report_stem or model_name)
    base_name = f"{safe_model_name}_{timestamp}"

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


def generate_reports(
    report_data: Dict,
    output_dir: Path,
    formats: Optional[List[str]] = None,
    report_stem: Optional[str] = None,
) -> Dict[str, Path]:
    """Backward-compatible wrapper for report generation."""
    return export_reports(
        report_data=report_data,
        output_dir=output_dir,
        formats=formats,
        report_stem=report_stem,
    )
