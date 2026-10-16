"""Fixed, offline frozen-runtime checks; never evaluates caller-supplied code."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys

from core.code_exec import run_code_test
from core.runtime import PROJECT_ROOT, code_python_executable, configure_console


def reporting_checks() -> dict[str, bool]:
    """Exercise dynamically loaded PDF fonts, Plotly JS and scientific extensions."""
    import plotly.graph_objects as graph
    import reportlab
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas
    from scipy.stats import ttest_ind

    font = Path(reportlab.__file__).parent / "fonts" / "Vera.ttf"
    pdfmetrics.registerFont(TTFont("BundleVera", str(font)))
    buffer = io.BytesIO()
    canvas = Canvas(buffer)
    canvas.setFont("BundleVera", 12)
    canvas.drawString(20, 20, "LM Studio Bench – Grüße")
    canvas.save()
    html = graph.Figure(data=graph.Bar(x=["a", "b"], y=[1, 2])).to_html(
        include_plotlyjs=True
    )
    statistic = ttest_ind([1, 2, 3], [4, 5, 6])
    return {
        "pdf_and_fonts": buffer.getvalue().startswith(b"%PDF-"),
        "plotly_embedded_js": "Plotly.newPlot" in html and len(html) > 100_000,
        "scipy_extensions": bool(0 < statistic.pvalue < 0.05),
    }


def main(argv: list[str] | None = None) -> int:
    """Check packaged resources and real isolated Python execution.

    Args:
        argv: Command-line arguments; only argparse's help option is accepted.

    Returns:
        Zero if every fixed check passes; one otherwise.
    """
    configure_console()
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    required = (
        "VERSION",
        "AUTHORS",
        "LICENSE",
        "config/defaults.json",
        "config/bench.yaml",
        "prompts/coding_assistant.md",
        "tests/data/text/code_samples.json",
        "tests/prompts/code_implementation.md",
        "web/templates/dashboard.html.jinja",
        "cli/report_template.html.template",
        "assets/icons/lmstudio-bench.svg",
    )
    checks = {
        "resources": all(PROJECT_ROOT.joinpath(*item.split("/")).is_file()
                         for item in required)
    }
    checks.update(reporting_checks())
    interpreter = code_python_executable()
    version = subprocess.run(  # nosec B603 - real bundled Python, fixed program
        [
            interpreter, "-I", "-S", "-X", "utf8", "-c",
            "import json,sys; print(json.dumps(list(sys.version_info[:3])))",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=10,
        check=True,
    )
    checks["version_matched_python"] = (
        json.loads(version.stdout) == list(sys.version_info[:3])
    )
    if getattr(sys, "frozen", False):
        checks["separate_interpreter"] = (
            Path(interpreter).resolve() != Path(sys.executable).resolve()
        )
    cases = {
        "pass": (
            "import hashlib, json, math, sqlite3, ssl\n"
            "def add(a, b): return a + b\nprint('Grüße 日本語 ✓')",
            "assert add(2, 3) == 5",
            5,
        ),
        "fail": ("answer = 1", "assert answer == 2", 5),
        "timeout": ("while True: pass", "", 0.5),
        "memory": ("data = bytearray(2 * 1024**3)", "", 5),
    }
    results = {
        name: run_code_test(code, check, timeout)._asdict()
        for name, (code, check, timeout) in cases.items()
    }
    checks.update(
        code_pass=results["pass"]["passed"],
        unicode_output="Grüße 日本語 ✓" in results["pass"]["stdout"],
        code_fail=(
            not results["fail"]["passed"]
            and "AssertionError" in results["fail"]["detail"]
        ),
        code_timeout=results["timeout"]["timed_out"],
        code_memory=(
            not results["memory"]["passed"]
            and "MemoryError" in results["memory"]["detail"]
        ),
    )
    print(json.dumps({"checks": checks, "code_tests": results}, ensure_ascii=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
