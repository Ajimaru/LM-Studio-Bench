#!/usr/bin/env python3
"""
Wrapper script - Einstiegspunkt für das Benchmark-Tool

Verwendung:
  ./run.py [args]              - Startet normalen Benchmark
  ./run.py --webapp            - Startet FastAPI Web-Dashboard
  ./run.py -w                  - Startet FastAPI Web-Dashboard (Kurzform)
  
Beispiele:
  ./run.py --limit 5           - Testet 5 neue Modelle
  ./run.py --export-only       - Generiert Reports aus Cache
  ./run.py --webapp            - Startet Web-Dashboard
  ./run.py -w                  - Startet Web-Dashboard (Kurzform)
"""

import sys
import os
import subprocess
from pathlib import Path

# Setze Working Directory zur Root des Projekts
project_root = Path(__file__).parent
os.chdir(project_root)

# Zeige erweiterte Hilfe bei --help/-h
if "--help" in sys.argv or "-h" in sys.argv:
    print("LM Studio Model Benchmark - Einstiegspunkt")
    print("=" * 60)
    print()
    print("📊 BENCHMARK-MODI:")
    print()
    print("  1️⃣  CLI-Benchmark (Standard):")
    print("      ./run.py [benchmark-args]")
    print("      → Führt Benchmark direkt aus und zeigt Ergebnisse")
    print()
    print("  2️⃣  Web-Dashboard (Empfohlen):")
    print("      ./run.py --webapp  (oder -w)")
    print("      → Startet modernes Web-Interface mit Live-Streaming")
    print("      → Öffnet automatisch Browser auf http://localhost:8080")
    print("      → Features: Live-Logs, Results-Browser, Dark Mode")
    print()
    print("=" * 60)
    print()
    print("🌐 WEB-DASHBOARD OPTIONEN:")
    print()
    print("  --webapp, -w          Startet FastAPI Web-Dashboard")
    print()
    print("=" * 60)
    print()
    print("📋 BENCHMARK OPTIONEN (für CLI-Modus):")
    print()
    
    # Zeige benchmark.py Hilfe
    benchmark_script = project_root / "src" / "benchmark.py"
    if benchmark_script.exists():
        result = subprocess.run(
            [sys.executable, str(benchmark_script), "--help"],
            capture_output=True,
            text=True
        )
        # Überspringe die erste Zeile (usage) von benchmark.py
        lines = result.stdout.split('\n')
        in_options = False
        for line in lines:
            if line.startswith('options:') or line.startswith('  -'):
                in_options = True
            if in_options:
                print(line)
    
    sys.exit(0)

# Prüfe auf --webapp oder -w Flag
has_web_flag = "--webapp" in sys.argv or "-w" in sys.argv

if has_web_flag:
    # Entferne --webapp/-w aus argv für saubere Übergabe
    args = [arg for arg in sys.argv[1:] if arg not in ("--webapp", "-w")]
    
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
