# Hardware-Monitoring Live-Charts - Anleitung

## 🐛 Bugfix: Hardware-Monitoring anzeigen

### Problem
Das Hardware-Monitoring (GPU-Temperatur, Power-Draw, VRAM-Auslastung) wurde nicht live in der WebApp angezeigt, obwohl es im Backend korrekt funktionierte.

### Ursache
Die HardwareMonitor-Klasse sammelte die Daten in Background-Threads, gab sie aber nicht auf `stdout` aus. Die WebApp konnte daher keine Daten zum Parsen finden.

### Lösung

#### 1. Backend: Hardware-Metriken auf stdout drucken
In `src/benchmark.py` wurde die `_monitor_loop()` Methode aktualisiert:

```python
def _monitor_loop(self):
    """Background-Thread für kontinuierliche Messungen"""
    while self.monitoring:
        try:
            temp = self._get_temperature()
            power = self._get_power_draw()
            
            with self.lock:
                if temp is not None:
                    self.temps.append(temp)
                    # 🆕 Drucke auf stdout für Live-Monitoring in WebApp
                    print(f"🌡️ GPU Temp: {temp:.1f}°C", flush=True)
                if power is not None:
                    self.powers.append(power)
                    # 🆕 Drucke auf stdout für Live-Monitoring in WebApp
                    print(f"⚡ GPU Power: {power:.1f}W", flush=True)
            
            time.sleep(1)  # Messungen jede Sekunde
        except Exception as e:
            logger.debug(f"Monitoring-Fehler: {e}")
            time.sleep(2)
```

**Key Points:**
- `flush=True` stellt sicher, dass die Ausgabe sofort verfügbar ist
- Emoji-Präfixe machen die Ausgabe leicht erkennbar
- Jede Sekunde wird eine neue Messung gedruckt

#### 2. WebApp: Regex-Patterns aktualisiert
In `web/app.py` wurde die `parse_hardware_metrics()` Methode aktualisiert:

```python
def parse_hardware_metrics(self, output_line: str):
    """Parse Hardware-Metriken aus Benchmark-Output"""
    import re
    
    # Pattern für GPU-Temperatur: "🌡️ GPU Temp: 45.5°C"
    temp_match = re.search(r'GPU\s+Temp\s*:\s*(\d+(?:\.\d+)?)°?C', output_line, re.IGNORECASE)
    if temp_match:
        temp_value = float(temp_match.group(1))
        self.hardware_history["temperatures"].append({
            "timestamp": datetime.now().isoformat(),
            "value": temp_value
        })
    
    # Pattern für Power: "⚡ GPU Power: 150.5W"
    power_match = re.search(r'GPU\s+Power\s*:\s*(\d+(?:\.\d+)?)W', output_line, re.IGNORECASE)
    if power_match:
        power_value = float(power_match.group(1))
        self.hardware_history["power"].append({
            "timestamp": datetime.now().isoformat(),
            "value": power_value
        })
```

**Key Points:**
- Regex-Patterns trimmen die Emojis und Formate
- `re.IGNORECASE` erlaubt flexible Schreibweisen
- Werte werden mit Timestamps in Listen gespeichert

#### 3. WebSocket-Streaming
Die WebSocket-Verbindung sendet die Hardware-Metriken alle 2 Sekunden an den Frontend:

```python
# In websocket_benchmark() Funktion
if current_time - manager.last_hardware_send_time >= 2.0:
    hardware_data = {
        "temperatures": manager.hardware_history["temperatures"][-60:],
        "power": manager.hardware_history["power"][-60:],
        "vram": manager.hardware_history["vram"][-60:]
    }
    await websocket.send_json({"type": "hardware", "data": hardware_data})
    manager.last_hardware_send_time = current_time
```

**Features:**
- Drosselung: Nur alle 2 Sekunden senden (nicht jede Messung)
- History: Letzte 60 Einträge pro Metrik (~2 Minuten)
- Effizient: Nur neue Daten senden

#### 4. Frontend: Plotly.js Charts
In `web/templates/dashboard.html.jinja` werden die Daten in Echtzeit-Charts visualisiert:

```javascript
function updateTemperatureChart(data) {
    const trace = {
        x: data.map(d => new Date(d.timestamp).toLocaleTimeString()),
        y: data.map(d => d.value),
        type: 'scatter',
        mode: 'lines+markers',
        line: {color: '#ef4444', width: 2},  // Red for temperature
        fill: 'tozeroy'
    };
    Plotly.newPlot('chart-temperature', [trace], layout, {responsive: true});
}
```

**Features:**
- Live-Updates alle 2 Sekunden
- Min/Max/Avg Stats automatisch berechnet
- Dark-Mode unterstützt
- Responsive Design

---

## 🧪 Testing

### 1. Syntax-Validierung
```bash
# Prüfe Python-Dateien auf Syntax-Fehler
./.venv/bin/python -m py_compile web/app.py src/benchmark.py
```

### 2. Regex-Pattern Validation
```python
import re

# Test-Daten
test_lines = [
    "🌡️ GPU Temp: 45.5°C",
    "⚡ GPU Power: 150.5W",
]

# Test Temperature
temp_match = re.search(r'GPU\s+Temp\s*:\s*(\d+(?:\.\d+)?)°?C', test_lines[0])
print(f"Temp: {float(temp_match.group(1))}°C")  # 45.5

# Test Power
power_match = re.search(r'GPU\s+Power\s*:\s*(\d+(?:\.\d+)?)W', test_lines[1])
print(f"Power: {float(power_match.group(1))}W")  # 150.5
```

### 3. WebApp Test
```bash
# Starte WebApp
python run.py --webapp

# Im Browser öffnen: http://localhost:<PORT>

# Benchmark starten mit Hardware-Monitoring:
1. Navigiere zu "Benchmark" View
2. Starte ein Benchmark ("Start Benchmark" Button)
3. Beobachte die 3 Hardware-Charts:
   - 🌡️ GPU Temperatur (Rot)
   - ⚡ GPU Leistung (Orange)
   - 💾 GPU VRAM (Grün)
```

---

## 📊 Output-Beispiele

### Backend (Console)
```
[11:23:45] ⚙️ Starting benchmark for model: llama-7b-q4...
[11:23:48] 🌡️ GPU Temp: 45.5°C
[11:23:48] ⚡ GPU Power: 150.5W
[11:23:49] 🌡️ GPU Temp: 46.2°C
[11:23:49] ⚡ GPU Power: 155.0W
[11:23:50] 🌡️ GPU Temp: 48.1°C
[11:23:50] ⚡ GPU Power: 160.5W
⚡ Run 1/3: 45.23 tokens/s
```

### WebSocket (Browser DevTools)
```json
{
  "type": "hardware",
  "data": {
    "temperatures": [
      {"timestamp": "2026-01-05T11:23:48.123456", "value": 45.5},
      {"timestamp": "2026-01-05T11:23:49.456789", "value": 46.2},
      {"timestamp": "2026-01-05T11:23:50.789012", "value": 48.1}
    ],
    "power": [
      {"timestamp": "2026-01-05T11:23:48.234567", "value": 150.5},
      {"timestamp": "2026-01-05T11:23:49.567890", "value": 155.0},
      {"timestamp": "2026-01-05T11:23:50.890123", "value": 160.5}
    ],
    "vram": []
  }
}
```

---

## ⚠️ Bekannte Limitierungen

1. **GPU-Tool Abhängigkeit**: Hardware-Monitoring benötigt:
   - NVIDIA: `nvidia-smi` Command
   - AMD: `rocm-smi` Command
   - Ohne GPU-Tool: Keine Temperatur/Power-Messungen

2. **VRAM-Support**: Aktuell wird VRAM nicht vom Backend gemessen (nur Temperatur + Power)

3. **Timing**: Hardware-Messungen erfolgen jede 1 Sekunde, WebSocket-Updates alle 2 Sekunden

---

## 🔧 Troubleshooting

### Charts zeigen immer noch keine Daten?

**Prüfen Sie:**
1. Ist `--enable-profiling` aktiviert? (Optional, aber empfohlen)
2. Sind die GPU-Tools (`nvidia-smi` / `rocm-smi`) installiert?
3. Öffnen Sie Browser DevTools (F12) und prüfen Sie WebSocket-Messages unter Network/WS

**Debug-Tipps:**
- Schauen Sie in die Console-Ausgabe der WebApp
- Suchen Sie nach `GPU Temp:` oder `GPU Power:` Zeilen
- Wenn keine Ausgabe: Hardware-Monitor ist möglicherweise deaktiviert

### Regex-Fehler beim Parsen?

**Lösung:**
- Prüfen Sie das exakte Output-Format mit `print(output_line)`
- Testen Sie die Regex mit einem Online-Tool
- Aktualisieren Sie die Regex-Patterns in `parse_hardware_metrics()`

---

## 📝 Commits

```
🐛 Fix: Hardware-Monitoring Live-Ausgabe (408ab62)
  - HardwareMonitor._monitor_loop() gibt Temperature und Power auf stdout aus
  - Regex-Patterns in WebApp angepasst
  - Live-Metriken werden während Benchmark-Ausführung angezeigt

📝 Update: Phase 14.5 Bugfix-Dokumentation (3f09fae)
  - Hardware-Output-Formate mit Emoji aktualisiert
  - Bugfix dokumentiert
```

---

## 🎯 Nächste Schritte (Phase 14.6+)

- [ ] VRAM-Monitoring (GPU Memory Usage)
- [ ] Historische Daten-Speicherung (Zwischen Sessions)
- [ ] Erweiterte Statistiken (P50, P95, P99)
- [ ] Export der Monitoring-Daten (CSV/JSON)
- [ ] Alerting bei Temperatur-Grenzen
