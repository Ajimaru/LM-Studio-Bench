# LM Studio CLI - Verfügbare LLM-Metadaten mit GPU-Analyse

## 📋 Schnelle Referenz

### Haupt-Befehle zur Metadaten-Abfrage

```bash
lms ls --json           # Alle heruntergeladenen Modelle mit Metadaten
lms ps --json           # Aktuell geladene Modelle
lms status              # Server-Status + Modell-Größe
lms version             # LM Studio Version
```

## 🎯 GPU-Unterstützung und Hardware-Anforderungen

### Automatische GPU-Erkennung im Benchmark

Das Benchmark-System erkennt automatisch all Ihre GPUs und deren Spezifikationen:

**NVIDIA GPUs:**

- Automatische Erkennung via `nvidia-smi`
- VRAM-Größe erfasst für Offload-Optimierung
- Temperatur und Leistung werden monitort

**AMD GPUs (rocm-smi):**

- Detaillierte Device-ID-Mapping für GPU-Modell-Namen
- VRAM- und GTT-Speicher werden separat erfasst
- rocm-smi Suchpfade: `/usr/bin`, `/usr/local/bin`, `/opt/rocm-*/bin/`

**iGPU-Erkennung:**

- Radeon iGPUs werden aus CPU-String extrahiert
- Regex-Muster: `Radeon\s+(\d+[A-Za-z]*)`
- Zeigt z.B. "Radeon 890M (Ryzen 9 7950X3D)" separat an

## 📊 Vollständige Metadaten-Felder (15 Felder pro Modell)

### Kategorie 1: Modell-Identifikation (5 Felder)

| Feld | Typ | Beispiel | Beschreibung |
| --- | --- | --- | --- |
| `type` | string | "llm" | Modelltyp (llm, embedding) |
| `modelKey` | string | "mistralai/ministral-3-3b" | Eindeutige Modell-ID |
| `displayName` | string | "Ministral 3 3B" | Anzeigename |
| `publisher` | string | "mistralai" | Modell-Publisher/Entwickler |
| `path` | string | "mistralai/ministral-3-3b" | Lokaler Speicher-Pfad |

### Kategorie 2: Technische Spezifikationen (4 Felder)

| Feld | Typ | Beispiel | Beschreibung |
| --- | --- | --- | --- |
| `architecture` | string | "mistral3", "gemma3", "llama" | Modell-Architektur |
| `format` | string | "gguf" | Dateiformat (GGUF, etc.) |
| `paramsString` | string | "3B", "7B", "13B" | Parameter-Größe |
| `sizeBytes` | number | 2986817071 | Größe in Bytes |

### Kategorie 3: Modell-Fähigkeiten (3 Felder)

| Feld | Typ | Beispiel | Beschreibung |
| --- | --- | --- | --- |
| `vision` | boolean | true / false | Kann Bilder verarbeiten? |
| `trainedForToolUse` | boolean | true / false | Unterstützt Tool-Calling/Funktions-Aufrufe? |
| `maxContextLength` | number | 131072, 262144 | Max. Context-Länge in Tokens |

### Kategorie 4: Quantisierung & Varianten (4 Felder)

| Feld | Typ | Beispiel | Beschreibung |
| --- | --- | --- | --- |
| `quantization.name` | string | "Q4_K_M", "Q8_0", "F16" | Quantisierungs-Methode |
| `quantization.bits` | number | 4, 8, 16 | Bits pro Gewicht |
| `variants` | array | ["@q4_k_m", "@q8_0"] | Alle verfügbaren Quantisierungen |
| `selectedVariant` | string | "mistralai/ministral-3-3b@q4_k_m" | Aktuell ausgewählte Variante |

## 🔍 Praktische Beispiele mit Ihren Modellen

### Beispiel 1: Vision-Modelle auflisten

```bash
lms ls --json | jq '.[] | select(.vision == true) | {displayName, paramsString, maxContextLength}'
```

**Ausgabe:**

```text
  • Gemma 3 4B (4B) - 131072 Tokens
  • Ministral 3 3B (3B) - 262144 Tokens
  • Qwen3 Vl 8B (8B) - 262144 Tokens
```

### Beispiel 2: Nur Tool-Calling Modelle

```bash
lms ls --json | jq '.[] | select(.trainedForToolUse == true) | .displayName'
```

### Beispiel 3: Modelle nach Größe sortieren

```bash
lms ls --json | jq 'sort_by(.sizeBytes) | .[] | {displayName, sizeGB: (.sizeBytes/1024/1024/1024|round*100/100)}'
```

### Beispiel 4: Modelle mit großem Context (≥128k Tokens)

```bash
lms ls --json | jq '.[] | select(.maxContextLength >= 131072) | {modelKey, maxContextLength}'
```

### Beispiel 5: Modell-Architektur Verteilung

```bash
lms ls --json | jq -r '.[] | .architecture' | sort | uniq -c
```

## 🐍 Python SDK Zugriff

### SDK-Methoden zur Metadaten-Abfrage

```python
import lmstudio

# 1. Alle heruntergeladenen Modelle abrufen
models = lmstudio.list_downloaded_models()
for model in models:
    print(f"Model: {model.model_key}")
    print(f"  Größe: {model.info.sizeBytes / 1024**3:.2f} GB")
    print(f"  Vision: {model.info.vision}")
    print(f"  Max Context: {model.info.maxContextLength} Tokens")
    print(f"  Architektur: {model.info.architecture}")
    print()

# 2. Aktuell geladene Modelle
loaded_models = lmstudio.list_loaded_models()
for llm in loaded_models:
    print(f"Geladen: {llm.identifier}")

# 3. Modelle filtern
vision_models = [m for m in models if m.info.vision]
print(f"Vision-Modelle: {len(vision_models)}")

# 4. Nach Größe sortieren
large_models = sorted(models, key=lambda m: m.info.sizeBytes, reverse=True)[:3]
for model in large_models:
    print(f"{model.info.displayName}: {model.info.sizeBytes / 1024**3:.2f} GB")
```

## 💡 Häufige Anwendungsfälle

### Use Case 1: Schnelle Performance-Tests

Filter nur kleine Modelle < 1GB für schnelle Benchmarks:

```bash
lms ls --json | jq '.[] | select(.sizeBytes < 1000000000) | .modelKey'
```

### Use Case 2: Langtext-Verarbeitung

Modelle mit großem Context für Dokumentanalyse:

```bash
lms ls --json | jq '.[] | select(.maxContextLength >= 100000) | .displayName'
```

### Use Case 3: Bildverarbeitung

Multi-Modal Modelle für Vision-Tasks:

```bash
lms ls --json | jq '.[] | select(.vision == true) | .modelKey'
```

### Use Case 4: Tool-Integration

Modelle mit Function-Calling für Agent-Systeme:

```bash
lms ls --json | jq '.[] | select(.trainedForToolUse == true) | .displayName'
```

### Use Case 5: Quantisierungs-Vergleich

Alle verfügbaren Quantisierungen eines Modells:

```bash
lms ls "google/gemma-3-1b" --json | jq '.variants[]'
```

## 🎯 Benchmarking mit Metadaten

Integration in Benchmark-Skripte:

```python
import subprocess
import json

# Lade Modell-Metadaten
result = subprocess.run(['lms', 'ls', '--json'], capture_output=True, text=True)
models = json.loads(result.stdout)

# Filtere für Benchmark
benchmark_candidates = [
    m for m in models
    if m['sizeBytes'] < 5e9  # < 5GB
    and m['vision'] == False  # Nur Text
]

print(f"Benchmark-Kandidaten: {len(benchmark_candidates)}")
for model in benchmark_candidates:
    print(f"  - {model['displayName']} ({model['paramsString']})")
```

## 📝 Tipps & Tricks

### Größe konvertieren

```bash
# Bytes zu GB
python3 -c "print(f'{2986817071/1024**3:.2f} GB')"  # Output: 2.78 GB
```

### JSON Pretty-Print

```bash
lms ls --json | jq '.' | less
```

### Schnelle Statistiken

```bash
# Durchschnittliche Modellgröße
lms ls --json | jq '[.[].sizeBytes] | add / length / 1024 / 1024 / 1024'

# Größtes Modell
lms ls --json | jq 'max_by(.sizeBytes) | .displayName'

# Modelle pro Architektur
lms ls --json | jq 'group_by(.architecture) | map({architecture: .[0].architecture, count: length})'
```

## 🔗 Verwandte Befehle

```bash
lms status              # Server-Status (zeigt auch geladene Modelle)
lms version             # LM Studio Version
lms load <model>        # Modell laden
lms unload --all        # Alle Modelle entladen
```

## Troubleshooting

### Keine Ausgabe bei `lms ls --json`

- Sicherstellen dass LM Studio Server läuft: `lms server start`
- Port-Konflikt prüfen

### jq nicht installiert

- Installation: `sudo apt install jq` (Linux) oder `brew install jq` (macOS)
- Alternative: Python Parsing verwenden

### Unbegrenzte Ausgabe

- Nutze `| head -n 5` zum Begrenzen
- Oder pipe zu `less` für paging: `| less`
