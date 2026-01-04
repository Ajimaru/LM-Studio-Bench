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
- ✅ Emoji-Icons für Vision 👁 und Tools 🔧

### CLI & Konfiguration

- ✅ `--runs N` - Anzahl Messungen anpassen
- ✅ `--context N` - Context Length konfigurieren
- ✅ `--prompt "..."` - Custom Prompt verwenden
- ✅ `--limit N` - Anzahl zu testender Modelle begrenzen

### Erweiterte Filterung

- ✅ `--only-vision` - Nur Vision-Modelle testen
- ✅ `--only-tools` - Nur Tool-fähige Modelle testen
- ✅ `--min-context 32000` - Min. Context Length Filter
- ✅ `--max-size 10` - Max. Dateigröße in GB
- ✅ `--params 7B,8B` - Bestimmte Parametergrößen
- ✅ `--quants q4,q5,q6` - Quantisierungs-Filter
- ✅ `--arch llama,mistral` - Architektur-Filter

### Effizienz-Metriken

- ✅ tokens_per_sec_per_gb - Performance pro GB Modellgröße
- ✅ tokens_per_sec_per_billion_params - Performance pro Milliarde Parameter

### PDF-Verbesserungen

- ✅ Metadata-Summary im Header (Vision/Tools-Count, Ø Größe, Ø Speed)

---

## 📋 Geplant

### 🎯 Erweiterte Filterung (Priorität: Hoch)

- [ ] `--exclude-models "pattern"` - Modelle ausschließen (Regex-Support)
- [ ] `--include-models "pattern"` - Nur bestimmte Modelle (Regex-Support)
- [ ] Filter-Kombinationslogik verbessern (OR-Verknüpfung ermöglichen)

### 📊 Vergleich & Analyse (Priorität: Mittel)

- ✅ Historischer Vergleich (frühere Runs laden und vergleichen) - `--compare-with latest|filename`
- ✅ Delta-Anzeige (Performance-Veränderungen über Zeit) - Δ% Spalte in PDF/CSV
- ✅ Ranking-System nach verschiedenen Kriterien (Speed, Effizienz, VRAM) - `--rank-by speed|efficiency|ttft|vram`
- ✅ Best-of-Quantization (bestes Q-Level pro Modell finden) - PDF-Sektion
- ✅ Percentile-Statistiken (P50, P95, P99 für Metriken) - PDF-Sektion

### 📄 PDF-Report-Verbesserungen (Priorität: Mittel)

- ✅ Bar-Charts für Top 10 schnellste Modelle
- [ ] Best-Practice-Empfehlungen basierend auf Hardware
- ✅ Quantisierungs-Vergleichstabelle (Q4 vs Q5 vs Q6 vs Q8 Side-by-Side) - PDF-Sektion
- [ ] Separate Seiten für verschiedene Kategorien

### 📈 Visualisierung (Priorität: Mittel)

- ✅ HTML-Report mit interaktiven Charts - Plotly (Bar-Chart, 2x Scatter-Plots)
- ✅ Scatter-Plots (Größe vs Speed, VRAM vs Performance)
- ✅ Performance-Trends über Zeit (Line-Charts mit historischen Daten)

### 🚀 Weitere Features

- Benchmark-Modi (Fast-Mode, Multi-Prompt, Stress-Test)
- Web-Dashboard
- Automatisierung (Scheduler, CI/CD, Notifications)
- Hardware-Profiling (Temperatur, Stromverbrauch)
- Daten-Management (SQLite)

## 🏆 Nächste Schritte

**Abgeschlossen**: Alle Core Features (Phase 1-4) ✅

- Phase 1: Best-of-Quantization, Historical Comparison
- Phase 2: HTML Reports, Ranking System  
- Phase 3: Delta Display, Percentiles, Quantization Comparison
- Phase 4: Trend Visualization

**Optional**: Erweiterte Features

- Regex-basierte Filter (--include/--exclude-models)
- Best-Practice-Empfehlungen basierend auf Hardware
- Web-Dashboard
- Performance Optimierungen
