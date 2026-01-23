#!/usr/bin/env python3
"""
Wrapper script - Entry point for the benchmark tool

Usage:
    ./run.py [args]              - Starts normal benchmark
    ./run.py --webapp            - Starts FastAPI web dashboard
    ./run.py -w                  - Starts FastAPI web dashboard (short form)

Examples:
    ./run.py --limit 5           - Tests 5 new models
    ./run.py --export-only       - Generates reports from cache
    ./run.py --webapp            - Starts web dashboard
    ./run.py -w                  - Starts web dashboard (short form)
"""

import sys
import os
import subprocess
from pathlib import Path

# Set working directory to project root
project_root = Path(__file__).parent
os.chdir(project_root)


# Show extended help for --help/-h
if "--help" in sys.argv or "-h" in sys.argv:
    print("LM Studio Model Benchmark - Entry Point")
    print("=" * 60)
    print()
    print("📊 BENCHMARK MODES:")
    print()
    print("  1️⃣  CLI Benchmark (Default):")
    print("      ./run.py [benchmark-args]")
    print("      → Runs benchmark directly and shows results")
    print()
    print("  2️⃣  Web Dashboard (Recommended):")
    print("      ./run.py --webapp  (or -w)")
    print("      → Starts modern web interface with live streaming")
    print("      → Automatically opens browser at http://localhost:8080")
    print("      → Features: Live logs, results browser, dark mode")
    print()
    print("=" * 60)
    print()
    print("🌐 WEB DASHBOARD OPTIONS:")
    print()
    print("  --webapp, -w          Starts FastAPI web dashboard")
    print()
    print("=" * 60)
    print()
    print("📋 BENCHMARK OPTIONS (for CLI mode):")
    print()

    # Show benchmark.py help
    benchmark_script = project_root / "src" / "benchmark.py"
    if benchmark_script.exists():
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--help"],
            capture_output=True,
            text=True,
            check=False
        )
        # Skip the first line (usage) from benchmark.py
        lines = result.stdout.split('\n')
        IN_OPTIONS = False
        for line in lines:
            if line.startswith('options:') or line.startswith('  -'):
                IN_OPTIONS = True
            if IN_OPTIONS:
                print(line)

    sys.exit(0)

# Check for --webapp or -w flag
has_web_flag = "--webapp" in sys.argv or "-w" in sys.argv

if has_web_flag:
    # Remove --webapp/-w from argv for clean passing
    args = [arg for arg in sys.argv[1:] if arg not in ("--webapp", "-w")]

    # Start web dashboard via app.py
    web_dir = project_root / "web"
    app_script = web_dir / "app.py"

    if not app_script.exists():
        print(f"❌ Error: {app_script} not found")
        print("💡 Tip: Please run the web dashboard setup first")
        sys.exit(1)

    print("🌐 Starting FastAPI web dashboard...")
    sys.exit(subprocess.call([sys.executable, str(app_script)] + args))
else:
    # Start normal benchmark via benchmark.py
    src_dir = project_root / "src"
    benchmark_script = src_dir / "benchmark.py"

    if not benchmark_script.exists():
        print(f"❌ Error: {benchmark_script} not found")
        sys.exit(1)

    # Pass all arguments through
    sys.exit(
        subprocess.call(
            [sys.executable, str(benchmark_script)] + sys.argv[1:]
        )
    )
