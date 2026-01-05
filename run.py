#!/usr/bin/env python3
"""
Wrapper script - Einstiegspunkt für das Benchmark-Tool

Verwendung:
  ./run.py [args]              - Startet normalen Benchmark
  ./run.py --web               - Startet FastAPI Web-Dashboard (automatischer freier Port)
  ./run.py -w                  - Startet FastAPI Web-Dashboard (automatischer freier Port)
  
Beispiele:
  ./run.py --limit 5           - Testet 5 neue Modelle
  ./run.py --export-only       - Generiert Reports aus Cache
  ./run.py --web               - Startet Web-Dashboard auf zufälligem freien Port
  ./run.py --web --port 9000   - Startet Web-Dashboard auf Port 9000
  ./run.py -w -p 8888          - Web-Dashboard auf Port 8888
"""

import sys
import os
import subprocess
from pathlib import Path

# Setze Working Directory zur Root des Projekts
project_root = Path(__file__).parent
os.chdir(project_root)

# Prüfe auf --web oder -w Flag
has_web_flag = "--web" in sys.argv or "-w" in sys.argv

if has_web_flag:
    # Entferne --web/-w aus argv für saubere Übergabe
    args = [arg for arg in sys.argv[1:] if arg not in ("--web", "-w")]
    
    # Starte Web-Dashboard via app.py
    web_dir = project_root / "web"
    app_script = web_dir / "app.py"
    
    if not app_script.exists():
        print(f"❌ Fehler: {app_script} nicht gefunden")
        print("💡 Tipp: Führe zuerst die Web-Dashboard-Setup durch")
        sys.exit(1)
    
    print("🌐 Starte FastAPI Web-Dashboard...")
    sys.exit(subprocess.call([sys.executable, str(app_script)] + args))
else:
    # Starte normalen Benchmark via benchmark.py
    src_dir = project_root / "src"
    benchmark_script = src_dir / "benchmark.py"
    
    if not benchmark_script.exists():
        print(f"❌ Fehler: {benchmark_script} nicht gefunden")
        sys.exit(1)
    
    # Alle Argumente weitergeben
    sys.exit(subprocess.call([sys.executable, str(benchmark_script)] + sys.argv[1:]))
