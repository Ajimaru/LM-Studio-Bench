#!/usr/bin/env python3
"""
Wrapper script - startet benchmark.py aus src/ Ordner
"""

import sys
import os
import subprocess
from pathlib import Path

# Setze Working Directory zur Root des Projekts
project_root = Path(__file__).parent
os.chdir(project_root)

# Starte benchmark.py aus src/ Ordner mit richtigen Pfaden
src_dir = project_root / "src"
benchmark_script = src_dir / "benchmark.py"

if not benchmark_script.exists():
    print(f"❌ Fehler: {benchmark_script} nicht gefunden")
    sys.exit(1)

# Alle Argumente weitergeben
sys.exit(subprocess.call([sys.executable, str(benchmark_script)] + sys.argv[1:]))
