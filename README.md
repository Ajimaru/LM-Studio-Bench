# LM Studio Model Benchmark

Automatisches Benchmark-Tool für alle lokal installierten LM Studio Modelle. Testet systematisch verschiedene Modelle und Quantisierungen, um Token/s-Geschwindigkeiten zu messen und zu vergleichen.

## Features

- 🌐 **Web Dashboard**: Moderne FastAPI-basierte Web-UI mit Live-Streaming, Dark Mode und interaktivem Ergebnisse-Browser
- ✅ **Automatische Modell-Discovery**: Findet alle lokal installierten Modelle und Quantisierungen
- ✅ **GPU-Detection**: Erkennt NVIDIA, AMD und Intel GPUs automatisch
- ✅ **6 Live Hardware-Charts**: GPU Temp, Power, VRAM, GTT (AMD) + System CPU & RAM
- ✅ **VRAM-Monitoring**: Misst VRAM-Nutzung während des Benchmarks
- ✅ **GTT Support (AMD)**: Nutzt Shared System RAM zusätzlich zu VRAM (z.B. 2GB VRAM + 46GB GTT = 48GB)
- ✅ **System Profiling**: CPU-Auslastung und RAM-Nutzung mit `--enable-profiling`
- ✅ **Hardware-Profiling**: Optionales Monitoring von GPU-Temperatur und Power-Draw (NVIDIA/AMD/Intel)
- ✅ **Progressive GPU-Offload**: Versucht automatisch verschiedene GPU-Offload-Levels (1.0 → 0.7 → 0.5 → 0.3)
- ✅ **Server-Management**: Startet LM Studio Server automatisch falls nötig
- ✅ **Standardisierte Tests**: Verwendet denselben Prompt für alle Modelle
- ✅ **Statistische Auswertung**: Warmup + mehrere Messungen für genaue Ergebnisse
- ✅ **SQLite-Cache**: Automatisches Caching von Benchmark-Ergebnissen (überspringt bereits getestete Modelle)
- ✅ **Intelligente Limits**: Mit `-l 5` werden 5 NEUE Modelle getestet + alle gecachten Modelle geladen
- ✅ **Dev-Mode**: Automatische Auswahl des kleinsten Modells für schnelle Tests während Entwicklung
- ✅ **Saubere Logging**: Emoji-Icons, formatierte Modell-Listen, gefilterte Third-Party-Debug-Logs, separate Log-Dateien für WebApp und Benchmarks
- ✅ **Export**:
  - JSON, CSV (Excel/Sheets-kompatibel)
  - PDF (Multi-Page mit Best-Practice-Empfehlungen, Vision/Tool/Architektur-Seiten, optionale Plotly-Charts)
  - HTML (interaktive Plotly-Charts, Best-Practices, Vision/Tool/Architektur-Tabellen, Dark Mode)
- ✅ **Instant Report Regeneration**: `--export-only` generiert Reports in <1s aus gecachten Daten
- ✅ **Umfangreiche Metadaten**: Parametergröße, Architektur, Context-Länge, Dateigröße, Vision- und Tool-Support

## Systemanforderungen

- **OS**: Linux (primär), Windows, macOS
- **Python**: 3.8 oder höher
- **GPU**: ~12GB VRAM empfohlen (NVIDIA/AMD/Intel)
- **Software**: [LM Studio](https://lmstudio.ai/) lokal installiert mit `lms` CLI verfügbar

## Installation

1. **Repository klonen**:

   ```bash
   git clone <repository-url>
   cd local-llm-bench
   ```

2. **Virtuelle Umgebung erstellen und aktivieren**:

   ```bash
   # Virtuelle Umgebung erstellen
   python3 -m venv .venv

   # Aktivieren (Linux/macOS)
   source .venv/bin/activate

   # Aktivieren (Windows)
   # .venv\Scripts\activate
   ```

3. **Python-Dependencies installieren**:

   ```bash
   pip install -r requirements.txt
   ```

4. **LM Studio CLI prüfen**:

   ```bash
   lms --help
   ```

## Nutzung

### 🌐 Web Dashboard (Empfohlen)

Starte die moderne Web-UI mit Live-Streaming und interaktivem Ergebnisse-Browser:

```bash
# Stelle sicher dass die virtuelle Umgebung aktiviert ist
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows

# Web-Dashboard starten (öffnet automatisch Browser)
./run.py --webapp
```

**Dashboard-Features:**

- 📊 **Live-Streaming**: Benchmark-Output in Echtzeit via WebSocket
- 🎨 **Dark Mode**: Modernes dunkles Design (per Default)
- 📁 **Ergebnisse-Browser**: Durchsuche alle gecachten Benchmark-Ergebnisse
- 🔧 **Flexible Konfiguration**: Alle CLI-Parameter als Web-Formular
- 📱 **Responsive Design**: Funktioniert auf Desktop und Tablet
- 🌐 **Netzwerk-Zugriff**: Öffne Dashboard auf anderen Geräten (<http://your-ip:8080>)
- 📝 **Separate Logs**: `logs/webapp_*.log` (Dashboard) und `logs/benchmark_*.log` (Benchmark-Läufe)

### 💻 Kommandozeile (CLI)

Alternativ kannst du Benchmarks direkt über die Kommandozeile starten:

```bash
./run.py
```

Das Script wird:

1. LM Studio Server prüfen/starten
2. Alle installierten Modelle finden
3. Jeden mit standardisiertem Prompt testen
4. Ergebnisse in `results/` speichern

### CLI-Optionen

#### Basis-Parameter

```bash
./run.py --runs 1           # Anzahl Messungen pro Modell
./run.py --context 4096     # Context Length in Tokens
./run.py --prompt "..."      # Custom Prompt
./run.py --limit 5          # Max. 5 NEUE Modelle testen (+ alle gecachten)
```

### Hardware-Profiling

```bash
# Aktiviere GPU-Monitoring (Temperatur + Power-Draw)
./run.py --enable-profiling

# Mit Sicherheitslimits
./run.py --enable-profiling --max-temp 85 --max-power 350

# AMD GTT (Shared System RAM)
./run.py --disable-gtt  # Nur VRAM nutzen (Standard: GTT aktiviert)
```

### Erweiterte Filter

```bash
# Nur bestimmte Quantisierungen
./run.py --quants q4,q5

# Nur bestimmte Architekturen
./run.py --arch llama,mistral

# Nur bestimmte Parametergrößen
./run.py --params 3B,7B

# Nur Vision-Modelle
./run.py --only-vision

# Nur Tool-fähige Modelle
./run.py --only-tools

# Minimale Context-Length
./run.py --min-context 32000

# Maximale Modellgröße
./run.py --max-size 10.0

# Regex-Filter: Include (nur Modelle die matchen)
./run.py --include-models "qwen|phi"       # Qwen oder Phi
./run.py --include-models "llama.*7b"      # Llama 7B Modelle
./run.py --include-models ".*q4.*"         # Alle Q4 Quantisierungen

# Regex-Filter: Exclude (schließe Modelle aus)
./run.py --exclude-models "uncensored"     # Keine uncensored Modelle
./run.py --exclude-models "q2|q3"          # Keine Q2/Q3 Quantisierungen
./run.py --exclude-models ".*vision.*"     # Keine Vision-Modelle

# Filter kombinieren (AND-Verknüpfung)
./run.py --include-models "llama" --exclude-models "q2" --only-tools
./run.py --only-vision --params 7B --max-size 12
```

### Cache-Verwaltung

```bash
# Cache nutzen (Standard - überspringt bereits getestete Modelle)
./run.py --limit 5

# Cache ignorieren und alle neu testen
./run.py --retest --limit 5

# Entwicklungs-Modus (kleinstes Modell, 1 Run)
./run.py --dev-mode

# Zeige alle gecachten Ergebnisse
./run.py --list-cache

# Exportiere Cache als JSON
./run.py --export-cache my_cache.json

# Generiere Reports aus Datenbank (ohne neue Tests)
./run.py --export-only                  # Alle gecachten Modelle
./run.py --export-only --params 7B      # Nur 7B Modelle
./run.py --export-only --quants q4      # Nur Q4 Quantisierungen
./run.py --export-only --compare-with latest  # Mit historischem Vergleich
```

### Standard-Einstellungen

- **Prompt**: "Erkläre maschinelles Lernen in 3 Sätzen"
- **Context Length**: 2048 Tokens
- **Warmup**: 1 Durchlauf
- **Messungen**: 3 Durchläufe
- **GPU-Offload**: Automatisch (1.0 → 0.7 → 0.5 → 0.3)

### Optimierte Inference-Parameter

Für standardisierte und reproduzierbare Benchmarks werden optimierte Sampling-Parameter verwendet:

| Parameter          | Wert | Grund                                                     |
|--------------------|------|-----------------------------------------------------------|
| **Temperatur**     | 0.1  | Niedrig für konsistente, deterministische Ergebnisse      |
| **Top-K Sampling** | 40   | Sampling aus top 40 Tokens                                |
| **Top-P Sampling** | 0.9  | Nucleus-Sampling bei 90% kumulativer Wahrscheinlichkeit   |
| **Min-P Sampling** | 0.05 | Minimum-Wahrscheinlichkeits-Schwelle                      |
| **Repeat Penalty** | 1.2  | Reduziert Wiederholungen (default 1.1)                    |
| **Max Tokens**     | 256  | Begrenzte Output-Länge für schnellere Tests               |

Diese werden automatisch in `_run_inference()` über das Python SDK angewendet und können in der `OPTIMIZED_INFERENCE_PARAMS` Konstante angepasst werden (siehe [benchmark.py](benchmark.py) Zeile ~47).

## Output

### Log-Dateien

Das Tool verwendet separate Log-Dateien für verschiedene Komponenten:

```text
logs/
├── webapp_20260105_112201.log       # Web-Dashboard Logs (nur wenn --webapp verwendet)
└── benchmark_20260105_113045.log    # Benchmark-Ausführungs-Logs
```

- **WebApp-Logs** (`webapp_*.log`): FastAPI Server, WebSocket-Events, HTTP-Requests
- **Benchmark-Logs** (`benchmark_*.log`): Modell-Tests, VRAM-Monitoring, Fehler
- Logs werden nur erstellt wenn die jeweilige Komponente aktiv ist
- Timestamps im Format `YYYYMMDD_HHMMSS`

### Ergebnis-Dateien

**Benchmark-Reports** werden im Verzeichnis `results/` gespeichert:

- `benchmark_results_YYYYMMDD_HHMMSS.json` - Strukturierte Daten (für Automatisierung)
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - Tabelle (Excel/Sheets-kompatibel)
- `benchmark_results_YYYYMMDD_HHMMSS.pdf` - Formatierter Report (zum Teilen/Archivieren)
- `benchmark_results_YYYYMMDD_HHMMSS.html` - Interaktive Plotly-Charts
- `benchmark_cache.db` - SQLite-Datenbank mit allen Benchmark-Ergebnissen (automatisches Caching)

**Wichtig:** Reports enthalten nur die **neu getesteten Modelle** aus dem aktuellen Benchmark-Lauf. Der interaktive Ergebnisse-Browser im Web-Dashboard zeigt alle gecachten Ergebnisse für historische Vergleiche.

### PDF-Report

Der PDF-Report wird im **Landscape A4-Format** erstellt und enthält:

- **Summary**: Benchmark-Konfiguration (Modellanzahl, Context Length, Prompt)
- **Detaillierte Tabelle**: Alle Metriken inkl. Metadaten (Parametergröße, Architektur, Dateigröße)
- **Visuelle Indikatoren**: Emoji-Icons für Vision-Fähigkeit (👁) und Tool-Support (🔧)
- **Performance-Statistiken**: Schnellstes/Langsamstes Modell, Durchschnittswerte

Perfekt zum Teilen von Benchmark-Ergebnissen oder zum Archivieren!

### Beispiel CSV-Output

```csv
model_name,quantization,gpu_type,gpu_offload,vram_mb,avg_tokens_per_sec,avg_ttft,avg_gen_time,prompt_tokens,completion_tokens,timestamp,params_size,architecture,max_context_length,model_size_gb,has_vision,has_tools,tokens_per_sec_per_gb,tokens_per_sec_per_billion_params
llama-3.2-3b-instruct,q4_k_m,NVIDIA,1.0,2048,51.43,0.111,0.954,10,49,2026-01-04 10:30:45,3B,llama,8192,1.92,False,False,26.79,17.14
qwen2.5-7b-instruct,q5_k_m,NVIDIA,0.7,4512,38.76,0.145,1.287,10,49,2026-01-04 10:35:12,7B,qwen,131072,4.38,False,True,8.85,5.54
```

### Logs

- **Console**: Echtzeit-Fortschritt mit tqdm Progress-Bar und Emoji-Icons
  - 🚀 Start/Launch
  - 🔍 Detection/Discovery
  - 📊 Data/Statistics
  - 💾 Storage/Memory
  - ✅ Success/Completion
  - 🎯 Optimization
  - ⚙️ Configuration
  - ⏱️ Time/Performance
  - Weitere Icons für spezifische Operationen
- **Per-Run Logs**: `logs/benchmark_YYYYMMDD_HHMMSS.log` - Separate Log-Datei für jeden Run
- **Gefilterte Logs**: Third-Party-Bibliotheken (httpx, lmstudio, urllib3, websockets) auf WARNING-Level begrenzt
- **JSON-Filtering**: Websocket-Debug-Events automatisch gefiltert

## Gemessene Metriken

| Metrik | Beschreibung |
| ------ | ------------ |
| **avg_tokens_per_sec** | Durchschnittliche Token-Generierungsgeschwindigkeit |
| **avg_ttft** | Time to First Token - Latenz bis zum ersten generierten Token |
| **avg_gen_time** | Gesamtzeit für die Antwort-Generierung |
| **vram_mb** | VRAM-Nutzung während Inferenz (falls messbar) |
| **prompt_tokens** | Anzahl Input-Tokens |
| **completion_tokens** | Anzahl generierter Tokens |
| **params_size** | Parametergröße des Modells (z.B. "3B", "7B") |
| **architecture** | Modell-Architektur (z.B. "mistral3", "gemma3") |
| **max_context_length** | Max. Context-Länge des Modells in Tokens |
| **model_size_gb** | Dateigröße des Modells in GB (auf 2 Dezimalstellen) |
| **has_vision** | Vision-Fähigkeit (Multimodal: Text + Bilder) |
| **has_tools** | Tool-Calling-Support (Function/Tool-Use) |
| **tokens_per_sec_per_gb** | Effizienz: Tokens/s pro GB Modellgröße |
| **tokens_per_sec_per_billion_params** | Effizienz: Tokens/s pro Milliarde Parameter |
| **temp_celsius_min/max/avg** | GPU-Temperatur während Benchmark (°C) - nur mit --enable-profiling |
| **power_watts_min/max/avg** | GPU-Stromverbrauch während Benchmark (W) - nur mit --enable-profiling |

## Fehlerbehebung

### "lms: command not found"

LM Studio CLI ist nicht im PATH. Installiere/konfiguriere LM Studio:

```bash
# Prüfe Installation
which lms
```

### "Keine Modelle gefunden"

Stelle sicher dass Modelle in LM Studio heruntergeladen sind:

```bash
lms ls
```

### "GPU-Monitoring nicht verfügbar"

GPU-Tools fehlen. Installiere je nach GPU:

**NVIDIA**:

```bash
sudo apt install nvidia-utils
```

**AMD**:

```bash
# ROCm installieren
```

**Intel**:

```bash
sudo apt install intel-gpu-tools
```

### Modell lädt nicht (VRAM-Fehler)

Das Script versucht automatisch niedrigere GPU-Offload-Levels. Bei 12GB VRAM:

- ✅ 3B Modelle mit Q5_K_M
- ✅ 7B Modelle mit Q4_K_M
- ⚠️ 13B Modelle mit Q3_K_M (möglicherweise)
- ❌ 32B+ Modelle (nicht empfohlen)

## Anpassung

### Eigene Prompts

Ändere in [benchmark.py](benchmark.py):

```python
STANDARD_PROMPT = "Dein eigener Test-Prompt"
```

### Mehr/Weniger Durchläufe

```python
NUM_MEASUREMENT_RUNS = 5  # Standard: 3
```

### Context Length

```python
CONTEXT_LENGTH = 4096  # Standard: 2048
```

## Projekt-Struktur

```text
local-llm-bench/
├── benchmark.py              # Haupt-Script
├── requirements.txt          # Python-Dependencies
├── results/                  # Benchmark-Ergebnisse
│   ├── benchmark_results_*.json
│   └── benchmark_results_*.csv
├── errors.log                # Fehler-Log
├── PLAN.md                   # Implementierungsplan
├── README.md                 # Diese Datei
└── .github/
    └── copilot-instructions.md
```

## Technische Details

### GPU-Detection

Das Tool sucht GPU-Monitoring-Tools in:

- Standard PATH
- `/usr/bin`
- `/usr/local/bin`
- `/opt/rocm/bin` (AMD)
- `/usr/lib/xpu` (Intel)

### GPU-Offload-Strategie

Bei Ladefehlern wird automatisch reduziert:

1. 🟢 `gpuOffload: 1.0` (100% GPU)
2. 🟡 `gpuOffload: 0.7` (70% GPU)
3. 🟠 `gpuOffload: 0.5` (50% GPU)
4. 🔴 `gpuOffload: 0.3` (30% GPU)
5. ❌ Fehler → Überspringen + Loggen

### Benchmark-Ablauf

Für jedes Modell:

1. **Laden**: Mit optimalem GPU-Offload
2. **Warmup**: 1x Inferenz (verwerfen)
3. **Messung**: 3x Inferenz
4. **Stats**: Durchschnitt berechnen
5. **Entladen**: Modell aus Speicher entfernen
6. **Nächstes Modell**

## Contributing

Contributions sind willkommen! Bitte:

1. Fork das Repository
2. Erstelle einen Feature-Branch
3. Committe deine Änderungen
4. Push zum Branch
5. Öffne einen Pull Request

## Lizenz

MIT License - siehe LICENSE-Datei

## Web-Dashboard (Phase 13)

## 🌐 Web Dashboard

Modernes FastAPI-basiertes Web-Interface zur Steuerung und Live-Überwachung von Benchmarks.

### Verwendung

```bash
# Dashboard starten (öffnet automatisch Browser)
./run.py --webapp

# Oder kurze Syntax
./run.py -w
```

Das Dashboard ist standardmäßig verfügbar unter: <http://localhost:8080>

### Dashboard Features

- 🌐 **Modernes Web-UI**: Responsives Dashboard mit Dark Mode (Default)
- ⚡ **Live-Streaming**: WebSocket für Echtzeit-Terminal-Ausgabe
- 🎮 **Benchmark-Kontrolle**: Start/Stop über Web-Interface
- ⚙️ **Parameter-Konfiguration**: Alle CLI-Parameter über GUI
- 📊 **Ergebnisse-Browser**: Durchsuche alle gecachten Benchmark-Ergebnisse
- 📥 **Export-Funktionen**: Download JSON/CSV/PDF/HTML Reports
- 🌐 **Netzwerk-Zugriff**: Erreichbar von anderen Geräten im Netzwerk
- 📝 **Separate Logs**: `logs/webapp_*.log` und `logs/benchmark_*.log`
- 🎨 **Dark Mode**: Modernes dunkles Design mit Toggle-Option
- 🔌 **REST API**: Vollständige API für Automatisierung (`/docs` für OpenAPI)

### REST API Endpoints

- `GET /` - Dashboard UI
- `GET /api/status` - Benchmark-Status
- `GET /api/output` - Terminal-Ausgabe
- `POST /api/benchmark/start` - Benchmark starten
- `POST /api/benchmark/pause` - Pausieren
- `POST /api/benchmark/resume` - Fortsetzen
- `POST /api/benchmark/stop` - Stoppen
- `WS /ws/benchmark` - WebSocket Live-Streaming

### Parameter im Dashboard

Alle Benchmark-Parameter können über das Dashboard konfiguriert werden:

- Runs (Messungen pro Modell)
- Context Length
- Modell-Limit
- Custom Prompt
- Include/Exclude Regex-Patterns
- Vision/Tools/Retest/DevMode Flags

---

## Support

Bei Problemen:

1. Prüfe `logs/` für Error-Logs
2. Stelle sicher dass LM Studio läuft
3. Öffne ein Issue mit Logs und System-Info

---

**Hinweis**: Dieses Tool ist für Entwicklung/Testing gedacht. Für Produktions-Deployments siehe LM Studio Dokumentation.
