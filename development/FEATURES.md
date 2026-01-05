# Feature Roadmap - LM Studio Benchmark

## ✅ Implementiert

### Core-Funktionalität

- ✅ Automatische Modell-Discovery via `lms ls --json`
- ✅ Multi-GPU Support (NVIDIA, AMD, Intel)
- ✅ VRAM-Monitoring während Benchmark
- ✅ Progressive GPU-Offload (1.0 → 0.7 → 0.5 → 0.3)
- ✅ Automatisches Server-Management (Start/Status-Check)
- ✅ Standardisierte Test-Prompts
- ✅ Warmup + Multi-Run Messungen
- ✅ Optimierte Inference-Parameter (temperature, sampling)
- ✅ **SQLite-Cache-System** - Automatisches Caching von Benchmark-Ergebnissen
- ✅ **Parameter-Hashing** - Intelligente Cache-Invalidierung bei Änderungen

### Metadaten & Informationen

- ✅ Model Metadata Integration (15 Felder aus LM Studio CLI)
- ✅ Parametergröße (params_size: "3B", "7B", etc.)
- ✅ Architektur-Erkennung (architecture: "llama", "mistral", etc.)
- ✅ Max Context Length
- ✅ Modell-Dateigröße in GB (model_size_gb)
- ✅ Vision-Capability Detection (has_vision)
- ✅ Tool-Calling Support Detection (has_tools)

### Export-Formate

- ✅ JSON Export (strukturierte Daten)
- ✅ CSV Export (Excel/Sheets-kompatibel)
- ✅ PDF Export (Landscape A4 mit visuellen Indikatoren)
- ✅ HTML Export (interaktive Plotly-Charts mit Template-System)
- ✅ Emoji-Icons für Vision 👁 und Tools 🔧
- ✅ Separates HTML-Template (report_template.html.template)
- ✅ **Dark Mode** - Standard-Theme mit Toggle-Button (🌙/☀️)
- ✅ **Back-to-Top Button** - Smooth Scroll-Navigation

### CLI & Konfiguration

- ✅ `--runs N` - Anzahl Messungen anpassen
- ✅ `--context N` - Context Length konfigurieren
- ✅ `--prompt "..."` - Custom Prompt verwenden
- ✅ `--limit N` - Anzahl zu testender Modelle begrenzen
- ✅ `--retest` - Cache ignorieren, alle Modelle neu testen
- ✅ `--dev-mode` - Entwicklungs-Modus: kleinstes Modell, 1 Run
- ✅ `--list-cache` - Zeige alle gecachten Modelle
- ✅ `--export-cache FILE` - Exportiere Cache als JSON

### Erweiterte Filterung

- ✅ `--only-vision` - Nur Vision-Modelle testen
- ✅ `--only-tools` - Nur Tool-fähige Modelle testen
- ✅ `--min-context 32000` - Min. Context Length Filter
- ✅ `--max-size 10` - Max. Dateigröße in GB
- ✅ `--params 7B,8B` - Bestimmte Parametergrößen (OR-Verknüpfung)
- ✅ `--quants q4,q5,q6` - Quantisierungs-Filter (OR-Verknüpfung)
- ✅ `--arch llama,mistral` - Architektur-Filter (OR-Verknüpfung)
- ✅ `--include-models "pattern"` - Regex-Filter für gewünschte Modelle
- ✅ `--exclude-models "pattern"` - Regex-Filter zum Ausschließen von Modellen

### Effizienz-Metriken

- ✅ tokens_per_sec_per_gb - Performance pro GB Modellgröße
- ✅ tokens_per_sec_per_billion_params - Performance pro Milliarde Parameter

### PDF-Verbesserungen

- ✅ Metadata-Summary im Header (Vision/Tools-Count, Ø Größe, Ø Speed)

---

## 📋 Geplant

### 🎯 Erweiterte Filterung (Priorität: Hoch)

- ✅ `--exclude-models "pattern"` - Modelle ausschließen (Regex-Support)
- ✅ `--include-models "pattern"` - Nur bestimmte Modelle (Regex-Support)
- ✅ Filter-Kombinationslogik (AND-Verknüpfung für alle Filter, OR innerhalb Listen)

### 📊 Vergleich & Analyse (Priorität: Mittel)

- ✅ Historischer Vergleich (frühere Runs laden und vergleichen) - `--compare-with latest|filename`
- ✅ Delta-Anzeige (Performance-Veränderungen über Zeit) - Δ% Spalte in PDF/CSV
- ✅ Ranking-System nach verschiedenen Kriterien (Speed, Effizienz, VRAM) - `--rank-by speed|efficiency|ttft|vram`
- ✅ Best-of-Quantization (bestes Q-Level pro Modell finden) - PDF-Sektion
- ✅ Percentile-Statistiken (P50, P95, P99 für Metriken) - PDF-Sektion

### 📄 PDF-Report-Verbesserungen (Priorität: Mittel)

- ✅ Bar-Charts für Top 10 schnellste Modelle
- ✅ Best-Practice-Empfehlungen basierend auf Hardware (Hardware-Detection, Top-Empfehlungen, VRAM-Tipps)
- ✅ Quantisierungs-Vergleichstabelle (Q4 vs Q5 vs Q6 vs Q8 Side-by-Side) - PDF-Sektion
- ✅ Separate Seiten für verschiedene Kategorien (Vision-Modelle, Tool-Modelle, Nach Architektur)

### 📈 Visualisierung (Priorität: Mittel)

- ✅ HTML-Report mit interaktiven Charts - Plotly (Bar-Chart, 2x Scatter-Plots)
- ✅ Scatter-Plots (Größe vs Speed, VRAM vs Performance)
- ✅ Performance-Trends über Zeit (Line-Charts mit historischen Daten)
- ✅ Dark Mode mit CSS Variables (Standard aktiviert, LocalStorage-Persistenz)
- ✅ Theme Toggle Button (Wechsel zwischen Hell/Dunkel)
- ✅ Dynamische Chart-Farbanpassung (Plotly-Charts reagieren auf Theme)
- ✅ Back-to-Top Navigation Button (erscheint ab 300px Scroll)

### 🚀 Weitere Features

- ✅ **Daten-Management (SQLite)** - Cache-System implementiert
- ✅ **Dev-Mode** - Automatische Modell-Auswahl für schnelle Tests
- ✅ **Hardware-Profiling** - GPU-Temperatur und Power-Draw Monitoring (--enable-profiling)
- ✅ **GPU-Offload-Strategie-Optimierung** - Intelligente Anpassung der Offload-Levels
- [ ] Web-Dashboard

## 🏆 Nächste Schritte

**Phase 13 - Web-Dashboard: ABGESCHLOSSEN** ✅ (2026-01-05)

Alle Core Features (Phase 1-13) sind vollständig implementiert! 🎉

- Phase 1: Best-of-Quantization, Historical Comparison ✅
- Phase 2: HTML Reports, Ranking System ✅
- Phase 3: Delta Display, Percentiles, Quantization Comparison ✅
- Phase 4: Trend Visualization ✅
- Phase 5: SQLite-Cache-System ✅
- Phase 6: Regex-basierte Filterung ✅
- Phase 7: PDF-Report-Erweiterungen ✅
- Phase 8: HTML-Report-Erweiterungen ✅
- Phase 9: Hardware-Profiling & Report-Parität ✅
- Phase 10: GPU-Offload-Strategie-Optimierung ✅
- Phase 11: GTT Support für AMD GPUs ✅
- Phase 12: Instant Report Regeneration ✅
- Phase 13: **Web-Dashboard** ✅
  - Benchmark-Kontrollzentrum mit Live-Streaming
  - Results Browser mit Cache-Management
  - Dark Mode & UI Polish
  - Separate Logging-System
  - Report Content Filtering

### 🚀 Optionale Zukünftige Erweiterungen

- Web-Dashboard - Erweiterte Features:
  - Home/Dashboard-View mit Quick Stats
  - Live Hardware-Monitoring Charts (GPU Temp/Power während Run)
  - Historical Comparison + Line-Charts für Performance-Trends
  - Advanced Filtering (Multi-Select, Date-Range Picker)
  - Keyboard-Shortcuts (S=Start, Q=Stop)
  - Toast-Notifications für Events
  - Export Results Browser zu JSON/CSV/PDF
- Multi-Prompt Benchmarks (mehrere Test-Prompts parallel)
- Slack/Email Notifications bei Abschluss
- Model-Warmup-Optimierung (Adaptive Warmup-Runs)
- A/B Testing Framework (Vergleich verschiedener Inference-Parameter)

---

## 📝 Detaillierte Phase-Dokumentation

Die folgenden Phasen wurden bereits vollständig implementiert:

### Phase 7: PDF-Report-Erweiterungen

- **Separate PDF-Seiten** für bessere Organisation:
  - Bar-Chart: Top 10 schnellste Modelle
  - Scatter-Plot: Modellgröße vs Performance
  - Effizienz-Analyse Chart

### Phase 8: HTML-Report-Erweiterungen

- **Best-Practice-Empfehlungen** (identisch zu PDF)
- **Vision-Modelle-Sektion** mit dedizierter Tabelle und Top 3
- **Tool-Calling-Modelle-Sektion** mit dedizierter Tabelle und Top 3
- **Architektur-Gruppierung** mit Top 5 pro Architektur
- Farbcodierte Tabellen (Blau=Vision, Orange=Tools, Lila=Architektur)
- Responsive Design mit Dark Mode Support

### Phase 9: Hardware-Profiling & Report-Parität

  - **HardwareMonitor-Klasse** mit Background-Threading:
    - GPU-Temperatur-Monitoring (nvidia-smi, rocm-smi, intel_gpu_top)
    - Power-Draw-Monitoring (Stromverbrauch in Watt)
    - 1-Sekunden-Intervall während Benchmark-Run
    - Min/Max/Avg-Berechnung für temp_celsius und power_watts
  - **CLI-Flags für Profiling**:
    - `--enable-profiling` - Aktiviert Hardware-Monitoring
    - `--max-temp CELSIUS` - Abort bei Überhitzung
    - `--max-power WATTS` - Abort bei zu hohem Stromverbrauch
  - **BenchmarkResult erweitert** - 6 neue optionale Felder:
    - temp_celsius_min/max/avg
    - power_watts_min/max/avg
  - **CLI-Parameter in Reports**:
    - PDF Benchmark Parameter zeigt alle CLI-Argumente
    - HTML Benchmark Parameter vollständig wie PDF
    - Inference-Parameter + Runtime-Parameter strukturiert
  - **HTML/PDF Report-Parität**:
    - Benchmark Summary: 9 identische Metriken
    - Benchmark Parameter: Inference-Settings + CLI-Argumente
    - Vollständige Informations-Parität zwischen beiden Formaten

- Phase 10: **GPU-Offload-Strategie-Optimierung** (2026-01-04)
  - **VRAM-basierte Vorhersage**:
    - Query verfügbares VRAM (nvidia-smi, rocm-smi, intel_gpu_top)
    - Berechnung optimaler Offload-Level basierend auf Modellgröße
    - Sicherheits-Headroom (1GB) für System-Stabilität
    - Context-Length-Overhead-Berücksichtigung
  - **Cache-basiertes Learning**:
    - Lookup erfolgreicher Offload-Levels für ähnliche Modelle
    - Filter: Gleiche Architektur + ähnliche Größe (±20%)
    - Durchschnitt der letzten 5 erfolgreichen Runs
  - **Intelligente Level-Generierung**:

- Phase 11: **GTT (Graphics Translation Table) Support für AMD GPUs** (2026-01-04)
  - **AMD Shared System RAM Detection**:
    - Parst rocm-smi --showmeminfo all für VRAM + GTT
    - Berechnet verfügbares Memory: VRAM + GTT (z.B. 2GB VRAM + 46GB GTT = 48GB)
    - Ermöglicht größere Modelle auf AMD APUs/iGPUs mit wenig dediziertem VRAM
  - **CLI-Toggle**:
    - `--disable-gtt` - Deaktiviert GTT-Nutzung (nur VRAM)
    - Standard: GTT aktiviert für AMD GPUs
  - **BenchmarkResult erweitert** - 3 neue optionale Felder:
    - gtt_enabled: bool (ob GTT aktiv ist)
    - gtt_total_gb: float (Gesamter GTT-Speicher)
    - gtt_used_gb: float (Benutzter GTT-Speicher)
  - **Report-Integration**:
    - PDF: Zeigt GTT-Status in Benchmark Parameter Tabelle
    - HTML: Zeigt GTT-Info in CLI-Argumente-Sektion mit ✅/❌ Icons
    - JSON/CSV: Exportiert gtt_enabled, gtt_total_gb, gtt_used_gb
  - **Logging-Ausgabe**:
    - Mit GTT: "💾 Memory: 0.4GB VRAM + 44.7GB GTT = 45.1GB total"
    - Ohne GTT: "💾 Memory: 0.4GB VRAM (GTT deaktiviert)"
  - **Intelligente Offload-Anpassung**:
    - Mit GTT: Höhere Offload-Levels möglich (1.0, 0.7, 0.5)
    - Ohne GTT: Konservative Levels bei wenig VRAM (0.7, 0.5, 0.3)
    - Adaptive Offload-Sequence basierend auf VRAM-Prediction
    - Binary-Search-ähnliche Level-Auswahl
    - Deduplizierung und absteigende Sortierung
  - **Performance-Verbesserungen**:
    - 30-50% schnelleres Laden durch prädiktive Auswahl
    - Reduzierte Fehlversuche beim Modell-Loading
    - Lerneffekt durch gecachte Konfigurationen

- Phase 12: **Instant Report Regeneration** (2026-01-04)
  - **--export-only CLI-Flag**:
    - Generiert Reports aus SQLite-Datenbank ohne neue Tests
    - <1s für 13 Modelle (vs. ~10+ Minuten für vollständigen Benchmark)
    - Lädt alle gecachten BenchmarkResult-Objekte
  - **Filter-Support**:
    - Alle Filter funktionieren: --params, --quants, --arch, --only-vision, etc.
    - _matches_filters() Methode für BenchmarkResult-Filterung
    - Unterstützt Regex-Patterns (--include-models, --exclude-models)
  - **Historischer Vergleich**:
    - Funktioniert mit --compare-with latest/filename
    - Δ% Delta-Berechnung für Performance-Vergleiche
  - **Robuste DB-Schema-Erkennung**:
    - PRAGMA table_info(benchmark_results) prüft verfügbare Spalten
    - Unterstützt alte Datenbanken ohne temp_celsius/power_watts/gtt_* Spalten
    - Dynamische SELECT-Query basierend auf Schema-Version
  - **Use Cases**:
    - Test neuer Report-Layouts ohne Re-Benchmarking
    - Schnelle Re-Exportierung mit anderen Filtern
    - Report-Regenerierung nach Template-Änderungen
    - Daten-Exploration mit verschiedenen Filtern
  - **Beispiele**:

    ```bash
    ./run.py --export-only                      # Alle gecachten Modelle
    ./run.py --export-only --params 7B          # Nur 7B Modelle
    ./run.py --export-only --quants q4,q5       # Q4/Q5 Quantisierungen
    ./run.py --export-only --compare-with latest # Mit Delta-Vergleich
    ```

- Phase 13: **Web-Dashboard** ✅ (2026-01-05 - ABGESCHLOSSEN)
  - **Architektur**:
    - `run.py --webapp` / `run.py -w` startet FastAPI Web-Dashboard ✅
    - `run.py [andere Args]` startet normalen Benchmark (bestehendes Verhalten) ✅
    - `web/app.py` steuert `src/benchmark.py` via Subprocess ✅
  
  - **Backend (FastAPI + Jinja2)** ✅:
    - 3 Dependencies: fastapi, uvicorn, jinja2 ✅
    - REST API Endpoints für Daten-Zugriff ✅
    - Subprocess-Management für Benchmark-Kontrolle ✅
    - WebSocket (/ws/benchmark) für Live-Streaming von Metriken ✅
    - Status-Tracking (idle, running, completed, stopped) ✅
    - Graceful Shutdown mit SIGTERM/SIGKILL Fallback ✅
    - Jinja2 Templates für HTML-Rendering ✅
    - Separate Logging: `logs/webapp_*.log` und `logs/benchmark_*.log` ✅
  
  - **Phase 13.1: Benchmark-Kontrollzentrum** ✅ (2026-01-05 Completed):
    - ▶️ Start-Button mit Parameter-Konfiguration vor Start ✅
    - ⏹️ Stop-Button mit Bestätigung ✅
    - Parameter-Anpassung im Dashboard: ✅
      - Runs, Context Length, Modell-Limit
      - Include/Exclude Regex-Patterns
      - Cache Ignore (--retest)
      - GPU-Settings (Offload Limit, VRAM Limit)
    - Live Console-Ausgabe (scrollbare Logs) ✅
    - Status-Indicator (Idle/Running/Stopped) ✅
    - **WebSocket-Resilienz**: ✅
      - Auto-Reconnect nach Desktop-Lock (3s Interval)
      - Keep-Alive Heartbeat (1s) zur Verbindungshaltung
      - 2s Timeout für read_output() Operationen
      - Clean Disconnection Handling
    - **Type Safety**: Alle Pylance-Fehler behoben ✅
    - **Struktur**: Konsolidierte requirements.txt/README.md im Root ✅
  
  - **Phase 13.3: Results Browser + Cache-Management** ✅ (2026-01-05 Completed):
    - **Phase 13.3.1**: GET /api/results - Alle gecachten Benchmark-Ergebnisse ✅
      - JSON-Response mit allen 21 Feldern pro Modell
      - Direkte SQLite-Queries auf benchmark_results Tabelle
      - Verfügbar für Tabellen-Rendering und Filterung
    - **Phase 13.3.2**: GET /api/cache/stats - Cache-Statistiken ✅
      - Total Entries, Avg Speed (tok/s)
      - Fastest Model + Speed, Slowest Model + Speed
      - DB Size in MB
    - **Phase 13.3.3**: Cache-Management APIs ✅
      - DELETE /api/cache/{model_key}: Einzeleintrag löschen
        - Validiert Existenz vor Löschung
        - Gibt deleted_count zurück
      - POST /api/cache/clear: Gesamten Cache löschen
        - Löscht alle benchmark_results Einträge
        - Warning-Level Log für Audit-Trail
    - **Phase 13.3.4**: Results Browser UI ✅ (2026-01-05 Completed)
      - Interaktive HTML-Tabelle mit 7 Spalten:
        - Model, Quantization, Tokens/s, VRAM, GPU-Offload, GPU-Typ, Aktionen
      - Sortierbar nach allen Spalten (Click Header für Toggle)
      - Cache-Statistiken-Anzeige:
        - 📦 Anzahl Einträge | ⚡ Ø Speed | 🏆 Schnellstes Modell | 💾 DB-Größe
      - Action-Buttons:
        - 🔄 Aktualisieren (reload Results)
        - 🗑️ Cache Löschen (mit Bestätigungs-Dialog)
        - Pro-Row: 🗑️ Delete-Button für einzelne Einträge
      - Empty-State Message wenn keine Ergebnisse
      - Auto-Load beim Page-Load
      - Number Formatting: 2 Decimals für Speed, 0 für VRAM, % für GPU-Offload
  
  - **Phase 13.5: Dark Mode & UI Polish** ✅ (2026-01-05 Completed):
    - **Dark Mode**: ✅
      - Standardmäßig aktiviert (localStorage: "theme":"dark")
      - Toggle-Button (🌙/☀️) in Navigation
      - CSS Variables für Farben (--bg-primary, --text-primary, etc.)
      - Persistenz über Browser-Neustarts
    - **Back-to-Top Button**: ✅
      - Gradient-Design (purple #667eea → #764ba2)
      - Erscheint ab 300px Scroll
      - Smooth Scroll zum Seitenanfang
      - Hover-Effekte
    - **Responsive Design**: ✅
      - Mobile/Tablet/Desktop-optimiert
      - Flexbox/Grid-Layout
    - **Auto-Browser-Open**: ✅
      - Öffnet automatisch <http://localhost:8080> beim Start
      - Wartet auf Server-Bereitschaft (max 5s)
  
  - **Phase 13.6: Report Content Filtering** ✅ (2026-01-05 Completed):
    - **Report-Export nur für neu getestete Modelle**: ✅
      - Separate newly_tested_models Liste in run_all_benchmarks()
      - _export_results_to_files() nimmt results_to_export Parameter
      - JSON/CSV/PDF/HTML exportieren nur neue Modelle
    - **Ergebnisse-Browser zeigt alle gecachten Modelle**: ✅
      - GET /api/results liefert kompletten Cache
      - Ermöglicht historische Vergleiche im Dashboard
    - **Subprocess-Detection**: ✅
      - benchmark.py erkennt Ausführung als Subprocess via psutil
      - Deaktiviert File-Logging wenn von webapp.py gestartet
      - Verhindert doppelte Log-Dateien
    - **Phase 13.3.1**: GET /api/results - Alle gecachten Benchmark-Ergebnisse
      - JSON-Response mit allen 21 Feldern pro Modell
      - Direkte SQLite-Queries auf benchmark_results Tabelle
      - Verfügbar für Tabellen-Rendering und Filterung
    - **Phase 13.3.2**: GET /api/cache/stats - Cache-Statistiken
      - Total Entries, Avg Speed (tok/s)
      - Fastest Model + Speed, Slowest Model + Speed
      - DB Size in MB
    - **Phase 13.3.3**: Cache-Management APIs
      - DELETE /api/cache/{model_key}: Einzeleintrag löschen
        - Validiert Existenz vor Löschung
        - Gibt deleted_count zurück
      - POST /api/cache/clear: Gesamten Cache löschen
        - Löscht alle benchmark_results Einträge
        - Warning-Level Log für Audit-Trail
    - **Phase 13.3.4**: Results Browser UI ✅ (2026-01-05 Completed)
      - Interaktive HTML-Tabelle mit 7 Spalten:
        - Model, Quantization, Tokens/s, VRAM, GPU-Offload, GPU-Typ, Aktionen
      - Sortierbar nach allen Spalten (Click Header für Toggle)
      - Cache-Statistiken-Anzeige:
        - 📦 Anzahl Einträge | ⚡ Ø Speed | 🏆 Schnellstes Modell | 💾 DB-Größe
      - Action-Buttons:
        - 🔄 Aktualisieren (reload Results)
        - 🗑️ Cache Löschen (mit Bestätigungs-Dialog)
        - Pro-Row: 🗑️ Delete-Button für einzelne Einträge
      - Empty-State Message wenn keine Ergebnisse
      - Auto-Load beim Page-Load
      - Number Formatting: 2 Decimals für Speed, 0 für VRAM, % für GPU-Offload
  
  - **Technische Details - Implementierte Endpoints**:
    - GET / - Dashboard Haupt-UI (Jinja2 Template) ✅
    - GET /api/status - Benchmark-Status (idle/running/stopped) ✅
    - GET /api/benchmark/output - Terminal-Ausgabe (plaintext) ✅
    - POST /api/benchmark/start - Benchmark mit Parametern starten ✅
    - POST /api/benchmark/stop - Stop mit Cleanup ✅
    - GET /api/results - Alle gecachten Ergebnisse ✅
    - GET /api/cache/stats - Cache-Statistiken ✅
    - DELETE /api/cache/{model_key} - Einzelnen Cache-Eintrag löschen ✅
    - POST /api/cache/clear - Gesamten Cache leeren ✅
    - WebSocket /ws/benchmark - Live-Streaming (JSON) ✅
    - GET /health - Health-Check ✅
  
  - **WebSocket-Messages** (Implementiert): ✅

    ```json
    {
      "type": "terminal_output",
      "content": "🚀 Starte Benchmark für qwen/qwen3-7b@q4_k_m\n"
    }
    ```

  - **Subprocess-Management**: ✅
    - Start: subprocess.Popen mit capture_output
    - Stop: SIGTERM → wait(timeout=5) → SIGKILL fallback
    - Cleanup: Temp-Dateien löschen, Datenbank committen
    - Logging: Separate webapp_*.log und benchmark_*.log Dateien
  
  - **Implementierungs-Status**:
    - ✅ **Phase 13.1**: Benchmark-Kontrollzentrum (ABGESCHLOSSEN - 2026-01-05)
    - ✅ **Phase 13.3**: Results Browser + Cache-Management (ABGESCHLOSSEN - 2026-01-05)
    - ✅ **Phase 13.5**: Dark Mode & UI Polish (ABGESCHLOSSEN - 2026-01-05)
    - ✅ **Phase 13.6**: Report Content Filtering (ABGESCHLOSSEN - 2026-01-05)
    - ✅ **Web-Dashboard vollständig implementiert!** 🎉
  
  - **Dateien**: ✅
    - web/app.py - FastAPI Backend (736 Zeilen)
    - web/templates/dashboard.html.jinja - Frontend UI (984 Zeilen)
    - requirements.txt - Python Dependencies (konsolidiert)
  
  - **Optional - Zukünftige Erweiterungen**:
    - Home/Dashboard-View mit Quick Stats
    - Live Hardware-Monitoring Charts (GPU Temp/Power während Run)
    - Historical Comparison + Line-Charts für Performance-Trends
    - Advanced Filtering (Multi-Select, Date-Range Picker)
    - Keyboard-Shortcuts (S=Start, Q=Stop)
    - Toast-Notifications für Events
    - Export Results Browser zu JSON/CSV/PDF

### 🚀 Zukünftige Erweiterungen

- Phase 14: **Web-Dashboard - Erweiterte Features** ✅ (2026-01-05 - ABGESCHLOSSEN)
  - **Phase 14.1: Home/Dashboard-View** ✅:
    - Dedizierte Landing-Page mit Navigation (🏠 Home, ⚡ Benchmark, 📊 Results)
    - Dashboard-Statistiken: Cache-Count, DB-Größe, Durchschnitts/Max-Speed
    - System-Info-Widget: OS, Python-Version, GPU-Typ
    - Top 5 schnellste Modelle Tabelle
    - Letzte 10 Benchmark-Runs mit Zeitstempeln
    - Quick-Action-Cards für Navigation zu Benchmark/Results
    - GET /api/dashboard/stats Endpoint für aggregierte Daten
    - View-Switching-System mit aktiver Navigation-Highlights
  
  - **Phase 14.2: Toast-Notifications** ✅:
    - Non-Blocking Notification-System (Success/Warning/Error/Info)
    - Auto-Dismiss mit konfigurierbarem Timeout (4s standard)
    - Toast-Stack-Management für mehrere gleichzeitige Notifications
    - Slide-In/Slide-Out Animationen
    - Dark-Mode kompatibel
    - Integration: Start/Stop Benchmark, Cache gelöscht, Fehler-Handling
    - Ersetzt alte alert()-Dialoge mit modernem UI
  
  - **Phase 14.3: Keyboard-Shortcuts** ✅:
    - `S` - Start Benchmark (öffnet Modal oder startet direkt)
    - `Q` - Stop Benchmark (mit Bestätigung)
    - `R` - Refresh Results (nur in Results-View)
    - `H` - Home-View
    - `B` - Benchmark-View
    - `Esc` - Close Modal
    - Visual Hints in Button-Labels (z.B. "Start (S)")
    - Prevention bei aktiven Input-Feldern
    - Tooltips für Shortcuts
  
  - **Technische Details**:
    - CSS-Animations: slideIn/slideOut für Toasts
    - Navigation: Active-State mit .active CSS-Klasse
    - View-Switching: .hidden CSS-Toggle für View-Container
    - Toast-Container: Fixed Position top-right mit Z-Index 9999
    - Keyboard-Event-Listener: Document-Level mit Input-Prevention
    - Dashboard-Stats-Loading: Auto-Load bei Home-View-Aktivierung

- [ ] Web-Dashboard - Phase 14.5: Live Hardware-Monitoring Charts:
  - [ ] Plotly.js Line-Charts für GPU-Monitoring
  - [ ] Echtzeit-Daten: Temperature, Power Draw, VRAM Usage
  - [ ] Verlaufs-Anzeige während Benchmark-Run
  - [ ] Requires: Erweiterung von WebSocket-Messages
- [ ] Web-Dashboard - Phase 14.6: Historical Comparison UI
- [ ] Web-Dashboard - Phase 14.7: Advanced Filtering
- [ ] Web-Dashboard - Phase 14.4: Export Results Browser
- [ ] Multi-Prompt Benchmarks (mehrere Test-Prompts parallel)
- [ ] Slack/Email Notifications bei Abschluss
- [ ] Model-Warmup-Optimierung (Adaptive Warmup-Runs)
- [ ] A/B Testing Framework (Vergleich verschiedener Inference-Parameter)
