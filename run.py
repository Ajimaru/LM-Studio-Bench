#!/usr/bin/env python3
"""
Wrapper script - startet benchmark.py aus src/ Ordner
"""

import sys
import subprocess
from pathlib import Path

# Starte benchmark.py aus src/ Ordner
src_dir = Path(__file__).parent / "src"
benchmark_script = src_dir / "benchmark.py"

if not benchmark_script.exists():
    print(f"❌ Fehler: {benchmark_script} nicht gefunden")
    sys.exit(1)

# Alle Argumente weitergeben
sys.exit(subprocess.call([sys.executable, str(benchmark_script)] + sys.argv[1:]))
