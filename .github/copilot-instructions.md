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
```
benchmark.py          # Haupt-Script
requirements.txt      # Dependencies
results/             # Benchmark-Ergebnisse (JSON/CSV)
errors.log           # Fehler-Log
PLAN.md              # Implementierungsplan
README.md            # Dokumentation
```

### Output-Format
- JSON: Strukturiert, maschinell lesbar
- CSV: Tabellarisch, Excel/Sheets-kompatibel
- Felder: model_name, quantization, gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec, avg_ttft, avg_gen_time, prompt_tokens, completion_tokens

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
