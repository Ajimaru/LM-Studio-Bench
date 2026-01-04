# GitHub Copilot Instructions - LM Studio Benchmark

## Projekt-Kontext

Dies ist ein Python-Benchmark-Tool für LM Studio, das automatisch alle lokal installierten LLM-Modelle und deren Quantisierungen testet. Das Ziel ist es, Token/s-Geschwindigkeiten zu messen und zu vergleichen.

## Coding Guidelines

### Allgemein
- Python 3.8+ kompatibel
- Type Hints verwenden wo möglich
- Docstrings für Klassen und wichtige Funktionen
- Fehlerbehandlung mit aussagekräftigen Error Messages
- Logging statt print() für wichtige Events

### Projekt-Spezifisch
- **GPU-Handling**: Immer Fallbacks implementieren (NVIDIA/AMD/Intel)
- **VRAM-Management**: Progressive GPU-Offload-Reduktion (1.0 → 0.7 → 0.5 → 0.3)
- **Error Recovery**: Niemals gesamten Benchmark abbrechen, nur einzelnes Modell überspringen
- **Progress Feedback**: tqdm für alle längeren Operationen
- **Resource Cleanup**: Modelle immer entladen nach Test

### LM Studio SDK
- Verwende `lmstudio` Python SDK
- Server-Management via subprocess (`lms` CLI)
- Nutze `stats` aus Response für Metriken
- Context Length auf 2048 begrenzen (VRAM-Optimierung)

### Dateistruktur

```text
project-root/
├── run.py              # Wrapper-Script (entry point)
├── README.md           # Hauptdokumentation
├── requirements.txt    # Dependencies
├── .gitignore          # Git-Ausschlüsse
├── src/
│   └── benchmark.py    # Haupt-Anwendung (1,676 Zeilen)
├── docs/               # Öffentliche Dokumentation
│   ├── QUICKSTART.md   # Schnelleinstieg
│   └── LLM_METADATA_GUIDE.md
├── development/        # Interne Notizen (in .gitignore)
│   ├── PLAN.md
│   └── FEATURES.md
├── results/            # Benchmark-Ergebnisse (JSON/CSV/PDF/HTML)
└── logs/               # Error-Logs mit Datum (error_YYYY-MM-DD.log)
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

## Best Practices
- Subprocess-Calls mit Timeout versehen
- GPU-Monitoring kann fehlschlagen → Graceful degradation
- Alle Pfade OS-agnostisch (pathlib.Path)
- CSV mit UTF-8 encoding
- JSON mit indent=2 für Lesbarkeit
- Type Hints vollständig (Pylance type-safe)
- Plotly-Availability prüfen für HTML/PDF exports
- Error Recovery: Niemals ganzen Benchmark abbrechen, einzelne Modelle überspringen

## Implementierungsplan
- Interne Roadmap: siehe `development/PLAN.md`
- Öffentliche Features: siehe `development/FEATURES.md`

## Troubleshooting
- Prüfe LM Studio Installation mit `lms --help`
- Nutze Log-Dateien in `logs/` für Debugging (Format: error_YYYY-MM-DD.log)
- Nutze LMStudio logs in `~/.lmstudio/server-logs/` für tiefere Fehleranalyse
- Bei Plotly-Fehlern: HTML/PDF fallback auf Text-Ausgabe
- GPU-Probleme: Prüfe VRAM mit `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), `intel_gpu_top` (Intel)