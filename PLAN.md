# LM Studio Modell-Benchmark - Implementierungsplan

## Überblick

Python-Skript zum automatischen Testen aller lokal installierten LM Studio Modelle und deren Quantisierungen. Misst Token/s-Geschwindigkeit mit standardisiertem Prompt "Erkläre maschinelles Lernen in 3 Sätzen" unter konstanten Bedingungen (12GB VRAM-optimiert).

## Implementierungsschritte

### 1. Projekt-Setup

- ✅ Git-Repository initialisiert
- ✅ `.gitignore` für Python-Projekte
- ✅ `PLAN.md` mit Implementierungsplan
- ✅ `.github/copilot-instructions.md` mit Projekt-Kontext

### 2. Dependencies

- `lmstudio` - Python SDK für LM Studio
- `tqdm` - Progress-Bar für Benchmark-Fortschritt
- `psutil` - System-Monitoring

### 3. GPU-Detection

- Systemweite Suche nach GPU-Tools:
  - NVIDIA: `nvidia-smi`
  - AMD: `rocm-smi`
  - Intel: `intel_gpu_top`
- Suchpfade: PATH, `/usr/bin`, `/usr/local/bin`, `/opt/rocm/bin`, `/usr/lib/xpu`
- GPU-Monitor-Klasse für VRAM-Auslese je nach Hersteller
- Fallback: "N/A" wenn keine Tools gefunden

### 4. Server-Management

- Status-Check mit `lms server status`
- Automatischer Start mit `lms server start` (Standard-Port)
- Warten bis Server bereit

### 5. Modell-Discovery

- `lms ls` via subprocess ausführen
- Output parsen für alle Modelle + Quantisierungen
- Liste mit Model-Keys erstellen

### 6. Benchmark-Loop

- Iteration über alle Modell-Quantisierungen mit tqdm
- VRAM-Handling:
  - Start mit `gpuOffload: 1.0`
  - Bei Fehler reduzieren: 0.7 → 0.5 → 0.3
  - VRAM-Nutzung während Inferenz messen
- Modell nach jedem Test entladen

### 7. Messung

- **Warmup**: 1 Durchlauf (verwerfen)
- **Messungen**: 3 Durchläufe pro Modell-Quantisierung
- **Prompt**: "Erkläre maschinelles Lernen in 3 Sätzen"
- **Stats erfassen**:
  - Tokens/s (Durchschnitt)
  - Time to First Token (TTFT)
  - Generation Time
  - Prompt Tokens
  - Completion Tokens

### 8. Error-Handling & Export

- `errors.log` für Modelle die nicht laden
- Export in `results/`:
  - JSON-Format (strukturiert)
  - CSV-Format (Tabellarisch)
- **Felder**: Modellname, Quantisierung, GPU-Typ, GPU-Offload, VRAM Usage (MB), Avg Tokens/s, Avg TTFT, Avg Generation Time, Prompt/Completion Tokens

### 9. Dokumentation

- `README.md` mit:
  - Projekt-Beschreibung
  - Installationsanweisungen
  - Nutzung
  - Beispiel-Output
  - Systemanforderungen

## Systemanforderungen

- **GPU**: 12GB VRAM (NVIDIA/AMD/Intel)
- **Software**: LM Studio lokal installiert
- **Python**: 3.8+
- **OS**: Linux (bevorzugt), Windows, macOS

## Technische Details

### Standard-Prompt

```text
Erkläre maschinelles Lernen in 3 Sätzen
```

### Optimierte Inference-Parameter

Für standardisierte und reproduzierbare Benchmarks werden die folgenden Parameter verwendet (via `LlmPredictionConfig` im SDK):

| Parameter | Wert | Grund |
| --------- | ---- | ----- |
| **temperature** | 0.1 | Niedrig für deterministische, konsistente Ergebnisse (statt default 0.8 für zufälligere Ausgabe) |
| **top_k_sampling** | 40 | Sampling aus den top 40 Token-Kandidaten |
| **top_p_sampling** | 0.9 | Nucleus-Sampling bei 90% kumulativer Wahrscheinlichkeit |
| **min_p_sampling** | 0.05 | Minimum-Wahrscheinlichkeits-Schwelle zur Filterung niedriger Wahrscheinlichkeits-Token |
| **repeat_penalty** | 1.2 | Leichte Strafe gegen wiederholte Tokens (default 1.1) für variablere Ausgabe |
| **max_tokens** | 256 | Begrenzte Output-Länge für schnellere und konsistentere Messungen |

Diese Parameter ermöglichen:
- **Reproduzierbarkeit**: Gleiche Eingabe → gleiche Ausgabe über mehrere Durchläufe
- **Fairness**: Alle Modelle werden mit denselben Sampling-Strategien gemessen
- **Performance**: Maximale Generierungsgeschwindigkeit durch limitierte Output-Länge
- **Konsistenz**: Reduzierung von Wiederholungen und erzeugten Fehler-Token

### GPU-Offload-Strategie

1. Versuche `gpuOffload: 1.0` (100%)
2. Bei Fehler: `gpuOffload: 0.7` (70%)
3. Bei Fehler: `gpuOffload: 0.5` (50%)
4. Bei Fehler: `gpuOffload: 0.3` (30%)
5. Bei Fehler: Logge in `errors.log`

### Context Length

- Standard: 2048 Tokens (VRAM-optimiert für 12GB)

### Anzahl Durchläufe

- 1x Warmup (verwerfen)
- 3x Messung (Durchschnitt berechnen)
