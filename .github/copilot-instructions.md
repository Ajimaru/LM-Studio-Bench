# GitHub Copilot Instructions - LM Studio Benchmark

## Projekt-Kontext

Dies ist ein Python-Benchmark-Tool für LM Studio mit modernem Web-Dashboard, das automatisch alle lokal installierten LLM-Modelle und deren Quantisierungen testet. Das Ziel ist es, Token/s-Geschwindigkeiten zu messen und zu vergleichen.

## Coding Guidelines

### Allgemein
- Python 3.8+ kompatibel
- Type Hints verwenden wo möglich
- Docstrings für Klassen und wichtige Funktionen
- Fehlerbehandlung mit aussagekräftigen Error Messages
- Logging statt print() für wichtige Events

### Projekt-Spezifisch
- **GPU-Handling**: Detaillierte Erkennung (NVIDIA nvidia-smi, AMD rocm-smi, Intel)
  - NVIDIA: `nvidia-smi --query-gpu=name` + `--query-gpu=memory.total`
  - AMD: `rocm-smi --showproductname` (gfx-Code) + `lspci` (Device-ID-Mapping)
  - iGPU-Extraktion: Radeon-Modelle aus CPU-String (z.B. "Radeon 890M")
- **Healthcheck**: HTTP API (Ports 1234/1235) + CLI Fallback (`lms status`) + 5s Polling
- **System-Info**: Linux-Distro (distro.name()), Kernel (platform.release()), CPU-Modell (cpuinfo)
- **VRAM-Management**: Progressive GPU-Offload-Reduktion (1.0 → 0.7 → 0.5 → 0.3)
- **Error Recovery**: Niemals gesamten Benchmark abbrechen, nur einzelnes Modell überspringen
- **Progress Feedback**: tqdm für alle längeren Operationen
- **Resource Cleanup**: Modelle immer entladen nach Test
- **Web-UI**: FastAPI Backend + Jinja2 Templates + Plotly Charts

### LM Studio SDK & Tools
- Verwende `lmstudio` Python SDK
- Server-Management via subprocess (`lms` CLI)
- Nutze `stats` aus Response für Metriken
- Context Length auf 2048 begrenzen (VRAM-Optimierung)
- GPU-Tools nutzen: nvidia-smi, rocm-smi, lspci

### Dateistruktur

```text
project-root/
├── run.py              # Wrapper-Script (entry point)
├── README.md           # Hauptdokumentation
├── requirements.txt    # Dependencies
├── .gitignore          # Git-Ausschlüsse
├── src/
│   ├── benchmark.py    # Haupt-Anwendung (~1,900 Zeilen)
│   └── report_template.html.template  # HTML-Report-Template
├── web/
│   ├── app.py          # FastAPI Backend (~1,400 Zeilen)
│   └── templates/
│       └── dashboard.html.jinja  # Dashboard UI (~2,600 Zeilen)
├── docs/               # Öffentliche Dokumentation
│   ├── QUICKSTART.md   # Schnelleinstieg
│   ├── HARDWARE_MONITORING_GUIDE.md
│   └── LLM_METADATA_GUIDE.md
├── development/        # Interne Notizen (in .gitignore)
│   └── FEATURES.md
├── results/            # Benchmark-Ergebnisse
│   ├── benchmark_results_*.json
│   ├── benchmark_results_*.csv
│   ├── benchmark_results_*.pdf
│   ├── benchmark_results_*.html
│   └── benchmark_cache.db  # SQLite-Cache
├── logs/               # Logs mit Datum
│   ├── webapp_YYYYMMDD_HHMMSS.log  # Dashboard Logs
│   └── benchmark_YYYYMMDD_HHMMSS.log  # Benchmark-Logs
└── .vscode/            # VSCode-Einstellungen
    └── settings.json
```

### Output-Format
- **JSON**: Strukturiert, 21 Felder (11 Metriken + 6 Metadaten + 2 Effizienz + 2 Delta)
- **CSV**: Tabellarisch, Excel/Sheets-kompatibel
- **PDF**: Landscape A4 mit Tabellen, Diagrammen und Analyse
- **HTML**: Interaktive Plotly-Diagramme (Bar, Scatter, Trend-Charts)
- Felder: model_name, quantization, gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec, tokens_per_sec_per_gb, speed_delta_pct, etc.

### Test-Konfiguration
- **Prompt**: "Erkläre maschinelles Lernen in 3 Sätzen"
- **Warmup**: 1 Durchlauf (verwerfen)
- **Messungen**: 3 Durchläufe (Durchschnitt)
- **Context**: 2048 Tokens
- **VRAM-Limit**: 12GB optimiert

## Web-Dashboard (FastAPI)

### Backend (web/app.py)
- **GPU-Detection**: Umfassende Erkennung mit fallback-Kette
- **System-Info**: Linux-Distro, Kernel, CPU-Modell, RAM
- **Healthcheck**: `/api/lmstudio/health` mit 5s Polling
- **Dashboard-Stats**: `/api/dashboard/stats` mit System- und GPU-Info
- **Benchmark-Control**: Start/Pause/Resume/Stop via REST API
- **WebSocket**: Live-Streaming von Benchmark-Output

### Frontend (web/templates/dashboard.html.jinja)
- **Responsive Design**: 2-Spalten Layout (Hardware | Benchmark)
- **27 Themes**: Light, Dark, Ocean Blue, Gruvbox, Dracula, Nord, etc.
- **Benchmark-Form**: Web-Formular mit Tooltip-Erklärungen für alle CLI-Argumente
- **Filter-Optionen**: Quantisierung, Architektur, Parametergröße, Context-Length
- **Ranking**: Nach Speed, Effizienz, TTFT oder VRAM
- **Hardware-Limits**: Max. GPU-Temp, Max. Power-Draw
- **Live-Charts**: GPU Temp, Power, VRAM, GTT, CPU, System-RAM (6 interaktive Plotly-Charts)

### Tooltip-System
- Fragezeichen-Icons neben jedem Label
- Hover zeigt kurze Erklärung
- Konsistente dunkle Hintergrund-Farbe (rgba(0,0,0,0.9)) für alle Themes

## Best Practices
- Subprocess-Calls mit Timeout versehen (z.B. `timeout=5`)
- GPU-Monitoring kann fehlschlagen → Graceful degradation
- Alle Pfade OS-agnostisch (pathlib.Path)
- CSV mit UTF-8 encoding
- JSON mit indent=2 für Lesbarkeit
- Type Hints vollständig (Pylance type-safe)
- Plotly-Availability prüfen für HTML/PDF exports
- Error Recovery: Niemals ganzen Benchmark abbrechen, einzelne Modelle überspringen
- GPU-Erkennung: Immer fallback-Ketten verwenden (HTTP → CLI → Parse sysfs)
- Healthcheck: Nicht blockierend, 5s Polling, keine Fehler wenn LM Studio offline

## Neue Dependencies
- `httpx`: HTTP-Client für Healthcheck
- `distro`: Linux-Distro-Erkennung
- `py-cpuinfo`: CPU-Modell-Extraktion
- `fastapi`: Web-Framework (existiert bereits)
- `jinja2`: Template-Rendering (existiert bereits)

## Implementierungsplan
- Interne Roadmap: siehe `development/FEATURES.md`

## Troubleshooting
- Prüfe LM Studio Installation mit `lms --help`
- Nutze Log-Dateien in `logs/` für Debugging (separate Logs für Webapp und Benchmark)
- Nutze LMStudio logs in `~/.lmstudio/server-logs/` für tiefere Fehleranalyse
- Bei Plotly-Fehlern: HTML/PDF fallback auf Text-Ausgabe
- GPU-Probleme: Prüfe VRAM mit `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), `intel_gpu_top` (Intel)
- Healthcheck-Fehler: Überprüfe ob LM Studio auf Port 1234 oder 1235 läuft
- Device-ID-Mapping: Nutze `lspci -d 1002:` für AMD GPU-Erkennung