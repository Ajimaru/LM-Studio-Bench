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
- [ ] Web-Dashboard
- [ ] Automatisierung (Scheduler, CI/CD, Notifications)
- [ ] Hardware-Profiling (Temperatur, Stromverbrauch)
- [ ] GPU-Offload-Strategie-Optimierung

## 🏆 Nächste Schritte

**Abgeschlossen**: Alle Core Features (Phase 1-5) ✅

- Phase 1: Best-of-Quantization, Historical Comparison
- Phase 2: HTML Reports, Ranking System
- Phase 3: Delta Display, Percentiles, Quantization Comparison
- Phase 4: Trend Visualization
- Phase 5: **SQLite-Cache-System** (2026-01-04)
  - Automatisches Caching aller Benchmark-Ergebnisse
  - Cache-Verwaltungs-CLI (--list-cache, --export-cache, --retest)
  - Dev-Mode mit automatischer Modell-Auswahl
  - Parameter-basierte Cache-Invalidierung
  - Performance: 0.6s statt 2+ Minuten für gecachte Modelle
  - Metadata-Cache-Fix (unknown → korrekte Werte)
  - HTML-Template-Extraktion (bessere Wartbarkeit)
  - **Dark Mode & Navigation** (2026-01-04):
    - Dark Mode als Standard-Theme (CSS Variables)
    - Toggle-Button für Theme-Wechsel (LocalStorage)
    - Dynamische Plotly-Chart-Farben
    - Back-to-Top Button mit Scroll-Detection

- Phase 6: **Regex-basierte Filterung** (2026-01-04)
  - `--include-models "pattern"` - Regex für gewünschte Modelle
  - `--exclude-models "pattern"` - Regex zum Ausschließen
  - AND-Verknüpfung zwischen verschiedenen Filtern
  - OR-Verknüpfung innerhalb von Listen (--quants, --arch, --params)
  - Case-insensitive Pattern Matching
  - Fehlerbehandlung bei ungültigen Regex-Patterns

- Phase 7: **PDF-Report-Erweiterungen** (2026-01-04)
  - **Best-Practice-Empfehlungen** - Intelligente Analyse-basierte Tipps:
    - Hardware-spezifische Empfehlungen (GPU-Typ)
    - Top-Modelle nach Kriterien (Speed, Effizienz, TTFT, Balance)
    - Quantisierungs-Vergleich (Q4 vs Q6 Performance-Unterschiede)
    - VRAM-basierte Empfehlungen (<4GB, 4-8GB, 8-12GB)
  - **Separate PDF-Seiten** für bessere Organisation:
    - Vision-Modelle (Multimodal) mit Top 3
    - Tool-Calling Modelle mit Top 3
    - Gruppierung nach Architektur (Top 5 pro Architektur)
    - Farbcodierte Tabellen für verschiedene Kategorien
  - **Plotly-Charts im PDF** (optional mit kaleido):
    - Bar-Chart: Top 10 schnellste Modelle
    - Scatter-Plot: Modellgröße vs Performance
    - Effizienz-Analyse Chart

- Phase 8: **HTML-Report-Erweiterungen** (2026-01-04)
  - **Best-Practice-Empfehlungen** (identisch zu PDF)
  - **Vision-Modelle-Sektion** mit dedizierter Tabelle und Top 3
  - **Tool-Calling-Modelle-Sektion** mit dedizierter Tabelle und Top 3
  - **Architektur-Gruppierung** mit Top 5 pro Architektur
  - Farbcodierte Tabellen (Blau=Vision, Orange=Tools, Lila=Architektur)
  - Responsive Design mit Dark Mode Support

- Phase 9: **Web-Dashboard** (Geplant)
  - **Backend (Flask/FastAPI)**:
    - REST API Endpoints für Daten-Zugriff
    - Subprocess-Management für Benchmark-Kontrolle
    - WebSocket für Live-Streaming von Metriken
    - Status-Tracking (idle, running, paused, completed, stopped)
    - Graceful Shutdown mit SIGTERM/SIGKILL Fallback
  
  - **Benchmark-Kontrollzentrum**:
    - ▶️ Start-Button mit Parameter-Konfiguration vor Start
    - ⏸️ Pause/Resume für längere Benchmarks
    - ⏹️ Stop-Button mit Bestätigung
    - Parameter-Anpassung im Dashboard:
      - Runs, Context Length, Modell-Limit
      - Include/Exclude Regex-Patterns
      - Cache Ignore (--retest)
      - GPU-Settings (Offload Limit, VRAM Limit)
    - Live Console-Ausgabe (scrollbare Logs)
    - Status-Indicator (Idle/Running/Paused/Stopped)
  
  - **Home/Dashboard-View**:
    - Quick Stats: Durchschn. Speed, schnellstes Modell, VRAM, Cache-Status
    - Server-Status: Online/Offline, verfügbare Modelle
    - Gecachte Modelle: Anzahl, Last-Used Datum
    - Quick-Actions: Start Benchmark, View Results, Manage Cache
  
  - **Live Benchmark Monitor** (während aktiver Tests):
    - Progress-Bar: X/Y Modelle komplett
    - Aktuelles Modell + aktuelle Run-Nummer
    - Live-Metriken: Speed (tok/s), Temp (°C), VRAM (MB/GB)
    - Top 5 bisher beste Modelle mit Echtzeit-Updates
    - ETA bis Abschluss (geschätzt)
    - Pause/Resume/Stop Buttons aktiv während Run
  
  - **Results Browser**:
    - Interaktive HTML-Tabelle mit allen gecachten Ergebnissen
    - Sortierbar nach: Speed, VRAM, Effizienz, Params, Quantization
    - Filterbar nach: Architektur, Parameter-Größe, Quantisierung, Vision/Tools
    - Multi-Select für Modell-Vergleich (2-3 Modelle nebeneinander)
    - Detail-View pro Modell: Historische Runs, Charts, Metadaten
    - Export: JSON/CSV/PDF für gefilterte Daten
  
  - **Historischer Vergleich**:
    - Line-Charts für Performance-Trends über Zeit
    - Auswählbare Modelle zum Vergleich
    - Metriken: Speed, VRAM-Verbrauch, Effizienz (tok/s/GB)
    - Date-Range Picker (Letzte Woche, Monat, Custom)
    - Delta-Anzeige: Δ% Veränderung seit letztem Run
  
  - **Cache-Management Interface**:
    - Zeige alle gecachten Einträge mit Details
    - Sortierbar nach: Model Name, Last-Used, Size, Speed
    - Delete einzelner Cache-Einträge
    - Bulk-Delete mit Filter
    - Cache komplett leeren mit Bestätigung
    - Export Cache als JSON
    - Cache-Größe und Speicherplatz
  
  - **UI/UX**:
    - Dark Mode (CSS Variables, toggle Button)
    - Responsive Design (Mobile/Tablet/Desktop)
    - WebSocket vs Polling Fallback
    - Status-Badges mit Farben (Grün=OK, Orange=Warning, Rot=Error)
    - Toast-Notifications für wichtige Events
    - Keyboard-Shortcuts (S=Start, P=Pause, Q=Stop)
  
  - **Technische Details**:
    - **Endpoints**:
      - GET /api/stats - Quick Stats (Speed, Models, Cache-Status)
      - GET /api/results - Alle gecachten Ergebnisse
      - GET /api/models - Liste verfügbarer Modelle
      - GET /api/models/{id} - Detail eines Modells
      - POST /api/benchmark/start - Benchmark mit Parametern starten
      - POST /api/benchmark/pause - Pause
      - POST /api/benchmark/resume - Resume
      - POST /api/benchmark/stop - Stop mit Cleanup
      - GET /api/benchmark/status - Aktueller Status
      - WebSocket /ws/benchmark - Live-Streaming
      - POST /api/cache/export - Cache exportieren
      - DELETE /api/cache/{id} - Eintrag löschen
    
    - **WebSocket-Messages** (Live):
    
      ```json
      {
        "type": "benchmark_progress",
        "current_model": "qwen/qwen3-7b@q4_k_m",
        "current_run": 2,
        "total_runs": 3,
        "models_completed": 8,
        "total_models": 30,
        "speed_tok_per_sec": 13.45,
        "vram_mb": 7200,
        "temp_celsius": 65,
        "eta_seconds": 1200,
        "message": "Run 2/3 abgeschlossen..."
      }
      ```
    
    - **Subprocess-Management**:
      - Start: subprocess.Popen mit capture_output
      - Pause: signal.SIGSTOP
      - Resume: signal.SIGCONT
      - Stop: SIGTERM → wait(timeout=5) → SIGKILL fallback
      - Cleanup: Temp-Dateien löschen, Datenbank committen
  
  - **Implementierungs-Phasen**:
    - **Phase 9.1**: Flask-Backend mit REST API (GET /api/*)
    - **Phase 9.2**: Benchmark Control (POST /api/benchmark/start|stop)
    - **Phase 9.3**: WebSocket Live-Streaming
    - **Phase 9.4**: Frontend Dashboard (Home + Control Panel)
    - **Phase 9.5**: Results Browser + Historischer Vergleich
    - **Phase 9.6**: Cache-Management UI
    - **Phase 9.7**: Error Handling + Robustheit

**Optional**: Erweiterte Features

- Multi-Prompt Benchmarks (mehrere Test-Prompts parallel)
- Hardware-Profiling (Temperatur, Stromverbrauch, Power-Draw)
- GPU-Offload-Strategie-Optimierung (Intelligente Offset-Anpassung)
- Slack/Email Notifications bei Abschluss
