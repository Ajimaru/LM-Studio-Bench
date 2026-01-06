# Hardware-Monitoring Live-Charts - Anleitung

## ✅ Status: Vollständig Implementiert mit GPU-Detection

Das Hardware-Monitoring ist jetzt vollständig funktionsfähig mit stabilen Live-Charts für alle Metriken sowie verbesserter GPU-Modell-Erkennung.

## 📊 Implementierte Metriken

### GPU-Erkennung & Modell-Informationen

Das System erkennt automatisch alle installierten GPUs:

1. **NVIDIA-GPUs**
   - Erkennung: `nvidia-smi --query-gpu=name`
   - VRAM: `nvidia-smi --query-gpu=memory.total`
   - Temperatur: `nvidia-smi --query-gpu=temperature.gpu`
   - Leistung: `nvidia-smi --query-gpu=power.draw`

2. **AMD-GPUs**
   - rocm-smi Erkennung: `rocm-smi --showproductname`
   - Device ID-Mapping: `lspci -d 1002:{device_id}`
   - Beispiel: `1002:150e` → "Radeon Graphics (Ryzen 9 7950X3D)"
   - rocm-smi-Suchpfad: `/usr/bin`, `/usr/local/bin`, `/opt/rocm-*/bin/`
   - VRAM: `rocm-smi --showmeminfo vram`
   - GTT: `rocm-smi --showmeminfo gtt`
   - Temperatur: `rocm-smi --showtemp`

3. **iGPU-Erkennung**
   - Extraktion aus CPU-String: Regex `r'Radeon\s+(\d+[A-Za-z]*)'`
   - Zeigt integrierte Radeon-Grafik separat an
   - Verhindert Redundanz mit dedizierten GPUs

### GPU-Metriken

1. **🌡️ GPU Temperatur** (°C) - Rot
   - NVIDIA: `nvidia-smi --query-gpu=temperature.gpu`
   - AMD: `rocm-smi --showtemp`
   - Intel: `intel-gpu-top` (wenn verfügbar)

2. **⚡ GPU Leistung** (W) - Blau
   - NVIDIA: `nvidia-smi --query-gpu=power.draw`
   - AMD: `rocm-smi` (Current Socket Graphics Package Power)
   - Intel: Alternative Messmethoden

3. **💾 GPU VRAM-Auslastung** (GB) - Grün
   - NVIDIA: `nvidia-smi --query-gpu=memory.used`
   - AMD: `rocm-smi --showmeminfo vram` (in Bytes)

4. **🧠 GPU GTT-Auslastung** (GB) - Violett
   - AMD nur: `rocm-smi --showmeminfo gtt`
   - System RAM das als VRAM genutzt wird
   - Beispiel: 2GB VRAM + 46GB GTT = 48GB effektiv

### System-Metriken (mit --enable-profiling)

1. **🖥️ CPU-Auslastung** (%) - Orange
   - `psutil.cpu_percent(interval=0.1)`
   - 0-100% Range
   - System-weit, nicht pro Prozess

2. **💾 System-RAM-Auslastung** (GB) - Cyan
   - `psutil.virtual_memory().used`
   - **Smoothing**: Gleitendes Mittel über 3 Messungen
   - Verhindert Spikes durch Cache/Buffer-Schwankungen
   - Sehr stabile Kurven

## 🔧 Aktivierung

Hardware-Monitoring wird automatisch aktiviert mit:

```bash
# WebApp mit Hardware-Monitoring
./run.py --webapp

# CLI mit Hardware-Monitoring
./run.py --enable-profiling

# Nur mit bestimmten Modellen
./run.py --limit 2 --enable-profiling
```

## 📝 Logger-Ausgaben

Wenn `--enable-profiling` aktiv ist, gibt das Benchmark jede Sekunde Metriken aus:

```text
🌡️ GPU Temp: 45.3°C
⚡ GPU Power: 125.5W
💾 GPU VRAM: 8.2GB
🧠 GPU GTT: 0.0GB
🖥️ CPU: 35.2%
💾 RAM: 18.5GB
```

Diese Ausgaben werden:

- ✅ In `logs/benchmark_YYYYMMDD_HHMMSS.log` gespeichert
- ✅ In der WebApp im Terminal angezeigt
- ✅ Als Charts visualisiert

## 🎯 Datenfluss

```
Backend (benchmark.py)
  ↓
HardwareMonitor._monitor_loop()
  ├─ _get_temperature()
  ├─ _get_power_draw()
  ├─ _get_vram_usage()
  ├─ _get_gtt_usage()
  ├─ _get_cpu_usage()
  └─ _get_ram_usage()
       ↓
logger.info() → stdout + log file
       ↓
WebApp Backend (app.py)
  ├─ _consume_output() Task (blocking readline)
  ├─ parse_hardware_metrics() (Regex patterns)
  └─ hardware_history dict
       ↓
WebSocket
  └─ Sendet alle 2 Sekunden (letzte 60 Einträge)
       ↓
Frontend (dashboard.html.jinja)
  └─ 6 Plotly.js Charts mit Live-Updates
```

## 🐛 Fixes und Optimierungen

### Fix 1: rocm-smi 7.0.1 Format-Änderung

**Problem**: rocm-smi änderte sein Output-Format
**Lösung**: Regex-Parser extrakt letzte Zahl aus Zeile

```python
match = re.search(r'[\d.]+\s*$', line.strip())
```

### Fix 2: Logger-Routing

**Problem**: Hardware-Daten erschienen nicht in Log-Dateien
**Lösung**: `print()` → `logger.info()` für stdout + file

### Fix 3: WebApp Output-Streaming

**Problem**: WebApp zeigte nur 10% der Hardware-Daten
**Lösung**: `asyncio.wait_for()` → blocking `readline()` im executor

### Fix 4: RAM-Monitoring Spikes

**Problem**: RAM-Chart sprang zwischen 1.8GB und 28.3GB
**Lösung**: Gleitendes Mittel über 3 Messungen → sehr stabile Kurve

### Fix 5: Laufzeit-Counter stoppt nicht

**Problem**: Laufzeit-Counter lief nach Benchmark-Ende weiter
**Lösung**: `clearInterval(uptimeInterval)` bei completion

### Fix 6: WebApp Initialization Race Conditions

**Problem**: Links waren nicht interaktiv, Light Mode beim Start
**Lösung**: 3x DOMContentLoaded Events → 1x konsolidiertes Event

## 📊 Chart-Eigenschaften

Alle Charts aktualisieren sich alle 2 Sekunden mit:

- **Min/Max/Avg Statistiken** - Echtzeit-Berechnung
- **Letzte 60 Datenpunkte** - Ca. 2 Minuten Geschichte
- **Responsive Design** - Passt sich Fenster an
- **Dark Mode** - Standard für alle Charts
- **Hover-Tooltips** - Zeige exakte Werte beim Hover
