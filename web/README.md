# Web-Dashboard für LM Studio Benchmark

FastAPI + Jinja2 basiertes Web-Dashboard zur Steuerung und Überwachung von Benchmark-Runs.

## Features

- 🌐 **Web-Interface**: Modernes Dashboard mit Dark Mode
- ⚡ **Live-Streaming**: WebSocket für Echtzeit-Terminal-Ausgabe
- 🎮 **Benchmark-Kontrolle**: Start, Pause, Resume, Stop
- ⚙️ **Parameter-Konfiguration**: Alle CLI-Parameter über GUI
- 📊 **Status-Überwachung**: Live-Updates und Uptime-Anzeige
- 🔌 **REST API**: Vollständige API für Automatisierung

## Installation

```bash
# Dependencies installieren
pip install -r requirements.txt

# Oder einzeln:
pip install fastapi uvicorn jinja2
```

## Verwendung

```bash
# Über run.py (empfohlen)
./run.py --web
./run.py -w

# Oder direkt
python web/app.py
```

Dashboard ist dann verfügbar unter: **http://localhost:8000**

## Architektur

```
web/
├── app.py                    # FastAPI Backend mit Subprocess-Management
├── requirements.txt          # Dependencies (3 packages)
├── templates/
│   └── dashboard.html.jinja  # Jinja2 Template mit interaktivem UI
└── __init__.py
```

## API Endpoints

### REST API

- `GET /` - Dashboard UI
- `GET /api/status` - Benchmark-Status
- `GET /api/output` - Aktuelle Terminal-Ausgabe
- `POST /api/benchmark/start` - Benchmark starten
- `POST /api/benchmark/pause` - Pausieren
- `POST /api/benchmark/resume` - Fortsetzen
- `POST /api/benchmark/stop` - Stoppen
- `GET /health` - Health-Check

### WebSocket

- `WS /ws/benchmark` - Live-Streaming von Benchmark-Output

## Subprocess-Management

Das Dashboard steuert `src/benchmark.py` via Python `subprocess`:

- **Start**: `subprocess.Popen()` mit Pipe-Ausgabe
- **Pause**: `signal.SIGSTOP`
- **Resume**: `signal.SIGCONT`
- **Stop**: `SIGTERM` → `wait(timeout=5)` → `SIGKILL` fallback

## Benchmark-Parameter

Alle Parameter aus `run.py` können über das Dashboard konfiguriert werden:

- `--runs N` - Anzahl Messungen
- `--context N` - Context Length
- `--limit N` - Modell-Limit
- `--prompt "..."` - Custom Prompt
- `--only-vision` - Nur Vision-Modelle
- `--only-tools` - Nur Tool-Modelle
- `--include-models "pattern"` - Regex-Filter
- `--exclude-models "pattern"` - Regex-Ausschluss
- `--retest` - Cache ignorieren
- `--dev-mode` - Dev-Mode

## UI Features

### Status-Anzeige
- Live-Status Badge (Idle, Running, Paused, Completed, Stopped)
- Uptime-Zähler
- Connected Clients Counter

### Control Panel
- ▶️ Start-Button (öffnet Parameter-Modal)
- ⏸️ Pause-Button
- ▶️ Resume-Button
- ⏹️ Stop-Button (mit Bestätigung)

### Live-Terminal
- Scrollbare Terminal-Ausgabe
- 400px Höhe mit Auto-Scroll
- Grüner Text auf dunklem Hintergrund

### Theme
- Dark Mode Toggle (🌙)
- LocalStorage Persistenz
- CSS Variables für konsistente Styling

## Development

### Logging

```python
import logging
logger = logging.getLogger(__name__)
logger.info("ℹ️ Info Message")
logger.warning("⚠️ Warning Message")
logger.error("❌ Error Message")
```

### Pydantic Model erweitern

```python
class CustomParams(BaseModel):
    new_param: Optional[str] = None
```

### WebSocket Messages

```json
{
  "type": "output|status|completed",
  "line": "...",
  "status": "running|paused|...",
  "message": "..."
}
```

## Troubleshooting

### Port 8000 bereits in Verwendung
```bash
# Anderen Port verwenden (später konfigurierbar)
python web/app.py --port 9000
```

### WebSocket Connection Failed
- Browser Console prüfen (F12 → Console)
- Firewall prüft ob WebSocket erlaubt
- Falls hinter Proxy: WebSocket-Support überprüfen

### Benchmark startet nicht
- `python run.py --dev-mode` testen (kürzester Benchmark)
- Logs in `/logs/` überprüfen
- `benchmark.py` Pfad validieren

## Performance

- Startup: < 1 Sekunde
- Live-Streaming Latenz: < 100ms
- Dashboard Rendering: < 50ms
- Memory-Overhead: ~50MB

## Sicherheit (Zukünftig)

- [ ] Authentication/Authorization
- [ ] HTTPS/WSS Support
- [ ] CORS Configuration
- [ ] Rate Limiting
- [ ] Input Validation

## Lizenz

Siehe Parent-Projekt
