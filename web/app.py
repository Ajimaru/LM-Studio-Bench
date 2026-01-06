#!/usr/bin/env python3
"""
FastAPI Web-Dashboard für LM Studio Benchmark

Steuert benchmark.py via Subprocess und bietet Live-Monitoring über WebSocket.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import webbrowser
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

# Logging konfigurieren - mit Console und File Handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WebApp Startup Log-Datei
def setup_webapp_logger():
    """Erstellt separate WebApp Startup Log-Datei"""
    logs_dir = Path(PROJECT_ROOT) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f"webapp_{timestamp}.log"
    
    # File Handler hinzufügen
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(file_handler)
    
    return log_file

# ============================================================================
# Hilfsfunktionen
# ============================================================================

def find_free_port() -> int:
    """Sucht einen freien Port auf dem System"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# Pfade
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
BENCHMARK_SCRIPT = SRC_DIR / "benchmark.py"
TEMPLATES_DIR = Path(__file__).parent / "templates"
RESULTS_DIR = PROJECT_ROOT / "results"
DATABASE_FILE = RESULTS_DIR / "benchmark_cache.db"

# Importiere BenchmarkCache aus benchmark.py
import sys
sys.path.insert(0, str(SRC_DIR))
try:
    from benchmark import BenchmarkCache, BenchmarkResult  # type: ignore
except ImportError as e:
    logger.error(f"❌ Konnte benchmark.py nicht importieren: {e}")
    BenchmarkCache = None  # type: ignore
    BenchmarkResult = None  # type: ignore

# FastAPI App initialisieren
app = FastAPI(
    title="LM Studio Benchmark Dashboard",
    description="Web-Dashboard zur Steuerung und Überwachung von LM Studio Benchmarks"
)

# Mount Results-Verzeichnis für statische Dateien (PDF, HTML, JSON, CSV)
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Jinja2 Template Environment
template_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

# Globale Benchmark-Prozess-Verwaltung
class BenchmarkManager:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"  # idle, running, paused, stopped, completed
        self.start_time: Optional[datetime] = None
        self.current_output = ""
        self.connected_clients = set()
        self.benchmark_log_file: Optional[Path] = None  # Log-Datei nur für aktive Benchmarks
        self.output_queue: Optional[asyncio.Queue[str]] = None  # Puffer für neue Output-Chunks
        self.output_task: Optional[asyncio.Task] = None
        
        # Hardware-Monitoring Daten
        self.hardware_history = {
            "temperatures": [],  # Liste von {"timestamp": ISO, "value": float}
            "power": [],
            "vram": [],
            "gtt": [],  # GTT-Nutzung (System RAM für AMD GPUs)
            "cpu": [],  # CPU-Auslastung in %
            "ram": []   # RAM-Nutzung in GB
        }
        self.last_hardware_send_time: float = 0  # Zur Drosselung von WebSocket-Updates

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def _consume_output(self):
        """Liest kontinuierlich stdout und legt neue Chunks in output_queue."""
        if self.output_queue is None:
            self.output_queue = asyncio.Queue()

        logger.info("🔄 Output-Consumer-Task gestartet")
        
        try:
            loop = asyncio.get_event_loop()
            
            # Lese bis EOF (auch nach Prozessende)
            while True:
                if not self.process or not self.process.stdout:
                    break
                
                try:
                    # Blockierendes Lesen in Executor (wartet bis Zeile da ist)
                    line = await loop.run_in_executor(
                        None,
                        self.process.stdout.readline
                    )
                    
                    if not line:
                        # EOF erreicht - Prozess ist fertig
                        break
                    
                    # Schreibe sofort in Log-Datei
                    if self.benchmark_log_file:
                        try:
                            with open(self.benchmark_log_file, 'a', encoding='utf-8') as f:
                                f.write(line)
                        except Exception as log_error:
                            logger.error(f"❌ Log-Write-Fehler: {log_error}")
                    
                    # Parse Hardware-Metriken
                    self.parse_hardware_metrics(line)
                    
                    # In Queue für WebSocket
                    await self.output_queue.put(line)
                    self.current_output += line
                    
                except Exception as read_error:
                    logger.error(f"❌ Read-Fehler: {read_error}")
                    break
            
            # Setze Status auf completed wenn Prozess fertig
            if not self.is_running():
                self.status = "completed"
            
            logger.info("🔄 Output-Consumer-Task beendet (EOF erreicht)")
            
            # Completion-Log
            if self.benchmark_log_file and self.benchmark_log_file.exists():
                logger.info(f"✅ Benchmark-Log: {self.benchmark_log_file}")
                
        except Exception as e:
            logger.error(f"❌ Fehler im Output-Consumer: {e}")

    def drain_output_queue(self) -> str:
        """Holt alle aktuell verfügbaren Output-Chunks ohne zu blockieren."""
        if not self.output_queue:
            return ""
        chunks: List[str] = []
        try:
            while True:
                chunks.append(self.output_queue.get_nowait())
        except asyncio.QueueEmpty:
            pass
        return ''.join(chunks)

    async def start_benchmark(self, args: list) -> bool:
        """Startet neuen Benchmark-Prozess"""
        if self.is_running():
            logger.warning("Benchmark läuft bereits")
            return False

        try:
            # Output-Puffer/Task zurücksetzen
            self.output_queue = asyncio.Queue()
            if self.output_task and not self.output_task.done():
                self.output_task.cancel()

            self.process = subprocess.Popen(
                [sys.executable, str(BENCHMARK_SCRIPT)] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT
            )
            self.status = "running"
            self.start_time = datetime.now()
            self.current_output = ""
            
            # Setze nur den Pfad - die Datei wird erst beim ersten Write erstellt
            logs_dir = RESULTS_DIR.parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self.benchmark_log_file = logs_dir / f"benchmark_{self.start_time.strftime('%Y%m%d_%H%M%S')}.log"
            
            # Starte Hintergrund-Task zum kontinuierlichen Lesen des Outputs
            self.output_task = asyncio.create_task(self._consume_output())

            logger.info(f"✅ Benchmark gestartet mit PID {self.process.pid}")
            logger.info(f"📝 Benchmark-Log wird geschrieben nach: {self.benchmark_log_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Starten des Benchmarks: {e}")
            self.status = "idle"
            return False

    def pause_benchmark(self) -> bool:
        """Pausiert laufenden Benchmark"""
        if not self.is_running() or not self.process:
            logger.warning("Kein laufender Benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGSTOP)
            self.status = "paused"
            logger.info("⏸️ Benchmark pausiert")
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Pausieren: {e}")
            return False

    def resume_benchmark(self) -> bool:
        """Setzt pausiertes Benchmark fort"""
        if self.status != "paused" or not self.process:
            logger.warning("Kein pausiertes Benchmark")
            return False
        try:
            self.process.send_signal(signal.SIGCONT)
            self.status = "running"
            logger.info("▶️ Benchmark fortgesetzt")
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Fortsetzen: {e}")
            return False

    def stop_benchmark(self) -> bool:
        """Stoppt laufenden Benchmark"""
        if not self.process:
            logger.warning("Kein laufender Benchmark")
            return False

        try:
            # Versuche graceful shutdown mit SIGTERM
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
                logger.info("⏹️ Benchmark beendet (SIGTERM)")
            except subprocess.TimeoutExpired:
                # Fallback zu SIGKILL
                self.process.kill()
                self.process.wait()
                logger.warning("⏹️ Benchmark erzwungen beendet (SIGKILL)")
            
            self.status = "stopped"
            self.process = None
            if self.output_task and not self.output_task.done():
                self.output_task.cancel()
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Stoppen: {e}")
            return False

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
        
        # Pattern für VRAM: "💾 GPU VRAM: 8.5GB"
        vram_match = re.search(r'GPU\s+VRAM\s*:\s*(\d+(?:\.\d+)?)GB', output_line, re.IGNORECASE)
        if vram_match:
            vram_value = float(vram_match.group(1))
            self.hardware_history["vram"].append({
                "timestamp": datetime.now().isoformat(),
                "value": vram_value
            })
        
        # Pattern für GTT: "🧠 GPU GTT: 1.5GB"
        gtt_match = re.search(r'GPU\s+GTT\s*:\s*(\d+(?:\.\d+)?)GB', output_line, re.IGNORECASE)
        if gtt_match:
            gtt_value = float(gtt_match.group(1))
            self.hardware_history["gtt"].append({
                "timestamp": datetime.now().isoformat(),
                "value": gtt_value
            })
        
        # Pattern für CPU: "🖥️ CPU: 45.2%"
        cpu_match = re.search(r'CPU\s*:\s*(\d+(?:\.\d+)?)%', output_line, re.IGNORECASE)
        if cpu_match:
            cpu_value = float(cpu_match.group(1))
            self.hardware_history["cpu"].append({
                "timestamp": datetime.now().isoformat(),
                "value": cpu_value
            })
        
        # Pattern für RAM: "💾 RAM: 8.5GB" (aber NICHT "GPU VRAM:")
        # Verwende negative Lookbehind, um nur "RAM:" zu matchen (nicht "VRAM:")
        ram_match = re.search(r'(?<![V])RAM\s*:\s*(\d+(?:\.\d+)?)GB', output_line, re.IGNORECASE)
        if ram_match:
            ram_value = float(ram_match.group(1))
            self.hardware_history["ram"].append({
                "timestamp": datetime.now().isoformat(),
                "value": ram_value
            })

    async def read_output(self) -> str:
        """Liest ALLE verfügbaren Zeilen aus Prozess ohne Blocking"""
        if not self.process or not self.process.stdout:
            return ""

        try:
            # Lese ALLE verfügbaren Zeilen (nicht nur eine!)
            loop = asyncio.get_event_loop()
            lines = []
            
            # Lese so viele Zeilen wie verfügbar sind (nicht-blockierend)
            while True:
                try:
                    # Versuche eine Zeile zu lesen mit kurzem Timeout
                    output = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            self.process.stdout.readline
                        ),
                        timeout=0.1  # Kurzer Timeout (100ms) pro Zeile
                    )
                    
                    if output:
                        lines.append(output)
                    else:
                        # Keine weitere Zeile verfügbar
                        break
                except asyncio.TimeoutError:
                    # Timeout bedeutet keine weiteren Zeilen verfügbar
                    break
            
            # Kombiniere alle gelesenen Zeilen
            if lines:
                combined_output = ''.join(lines)
                self.current_output += combined_output
                return combined_output
            
            return ""
        except Exception as e:
            logger.error(f"❌ Fehler beim Lesen des Outputs: {e}")
            return ""


# Globale Manager-Instanz
manager = BenchmarkManager()


# ============================================================================
# Pydantic Models
# ============================================================================

class BenchmarkParams(BaseModel):
    """Parameter für Benchmark-Start"""
    runs: Optional[int] = None
    context: Optional[int] = None
    limit: Optional[int] = None
    prompt: Optional[str] = None
    only_vision: bool = False
    only_tools: bool = False
    include_models: Optional[str] = None
    exclude_models: Optional[str] = None
    retest: bool = False
    dev_mode: bool = False
    enable_profiling: bool = True  # Hardware-Monitoring aktivieren (Standard: ON)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root() -> HTMLResponse:
    """Hauptseite - Dashboard"""
    template = template_env.get_template("dashboard.html.jinja")
    html = template.render()
    return HTMLResponse(content=html)


@app.get("/api/status")
async def get_status() -> dict:
    """Aktueller Status des Benchmarks"""
    return {
        "status": manager.status,
        "running": manager.is_running(),
        "start_time": manager.start_time.isoformat() if manager.start_time else None,
        "uptime_seconds": (
            (datetime.now() - manager.start_time).total_seconds()
            if manager.start_time and manager.is_running() else None
        ),
        "connected_clients": len(manager.connected_clients)
    }


@app.post("/api/benchmark/start")
async def start_benchmark(params: BenchmarkParams) -> dict:
    """Startet neuen Benchmark"""
    args = []
    
    if params.runs:
        args.extend(["--runs", str(params.runs)])
    if params.context:
        args.extend(["--context", str(params.context)])
    if params.limit:
        args.extend(["--limit", str(params.limit)])
    if params.prompt:
        args.extend(["--prompt", params.prompt])
    if params.only_vision:
        args.append("--only-vision")
    if params.only_tools:
        args.append("--only-tools")
    if params.include_models:
        args.extend(["--include-models", params.include_models])
    if params.exclude_models:
        args.extend(["--exclude-models", params.exclude_models])
    if params.retest:
        args.append("--retest")
    if params.dev_mode:
        args.append("--dev-mode")
    # Hardware-Profiling aktivieren (für Live-Monitoring Charts)
    if params.enable_profiling:
        args.append("--enable-profiling")
    
    # Debug: Zeige übergebene Args
    logger.info(f"🔧 Benchmark-Args: {args}")
    logger.info(f"📊 enable_profiling={params.enable_profiling}")
    
    success = await manager.start_benchmark(args)
    return {
        "success": success,
        "status": manager.status,
        "message": "✅ Benchmark gestartet" if success else "❌ Fehler beim Starten"
    }


@app.post("/api/benchmark/pause")
async def pause_benchmark() -> dict:
    """Pausiert laufenden Benchmark"""
    success = manager.pause_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏸️ Pausiert" if success else "❌ Fehler"
    }


@app.post("/api/benchmark/resume")
async def resume_benchmark() -> dict:
    """Setzt pausiertes Benchmark fort"""
    success = manager.resume_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "▶️ Fortgesetzt" if success else "❌ Fehler"
    }


@app.post("/api/benchmark/stop")
async def stop_benchmark() -> dict:
    """Stoppt laufenden Benchmark"""
    success = manager.stop_benchmark()
    return {
        "success": success,
        "status": manager.status,
        "message": "⏹️ Beendet" if success else "❌ Fehler"
    }


@app.get("/api/results")
async def get_results() -> dict:
    """Gibt alle gecachten Benchmark-Ergebnisse zurück"""
    if not BenchmarkCache:
        return {
            "success": False,
            "error": "BenchmarkCache nicht verfügbar",
            "results": []
        }
    
    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()
        
        # Konvertiere BenchmarkResult zu Dict
        results_data = []
        for result in results:
            # model_key für Frontend (für Delete-Button)
            model_key = f"{result.model_name}@{result.quantization}"
            
            result_dict = {
                "model_key": model_key,
                "model_name": result.model_name,
                "quantization": result.quantization,
                "gpu_type": result.gpu_type,
                "gpu_offload": result.gpu_offload,
                "vram_mb": result.vram_mb,
                "avg_tokens_per_sec": result.avg_tokens_per_sec,
                "avg_ttft": result.avg_ttft,
                "avg_gen_time": result.avg_gen_time,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "timestamp": result.timestamp,
                "params_size": result.params_size,
                "architecture": result.architecture,
                "max_context_length": result.max_context_length,
                "model_size_gb": result.model_size_gb,
                "has_vision": result.has_vision,
                "has_tools": result.has_tools,
                "tokens_per_sec_per_gb": result.tokens_per_sec_per_gb,
                "tokens_per_sec_per_billion_params": result.tokens_per_sec_per_billion_params,
                "speed_delta_pct": result.speed_delta_pct,
                "prev_timestamp": result.prev_timestamp
            }
            
            # Optionale Felder (Hardware-Profiling)
            if hasattr(result, 'temp_celsius_avg') and result.temp_celsius_avg:
                result_dict["temp_celsius_avg"] = result.temp_celsius_avg
            if hasattr(result, 'power_watts_avg') and result.power_watts_avg:
                result_dict["power_watts_avg"] = result.power_watts_avg
            if hasattr(result, 'gtt_enabled'):
                result_dict["gtt_enabled"] = result.gtt_enabled
            
            results_data.append(result_dict)
        
        return {
            "success": True,
            "count": len(results_data),
            "results": results_data
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Ergebnisse: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


@app.get("/api/cache/stats")
async def get_cache_stats() -> dict:
    """Gibt Cache-Statistiken zurück"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache nicht verfügbar"}
    
    try:
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()
        
        if not results:
            return {
                "success": True,
                "stats": {
                    "total_entries": 0,
                    "avg_tokens_per_sec": 0,
                    "fastest_model": "Keine Daten",
                    "fastest_speed": 0,
                    "slowest_model": "Keine Daten",
                    "slowest_speed": 0,
                    "db_size_mb": 0
                }
            }
        
        speeds = [r.avg_tokens_per_sec for r in results]
        fastest = max(results, key=lambda r: r.avg_tokens_per_sec)
        slowest = min(results, key=lambda r: r.avg_tokens_per_sec)
        
        # DB-Größe
        db_size_mb = DATABASE_FILE.stat().st_size / (1024 * 1024) if DATABASE_FILE.exists() else 0
        
        return {
            "success": True,
            "stats": {
                "total_entries": len(results),
                "avg_tokens_per_sec": sum(speeds) / len(speeds),
                "fastest_model": f"{fastest.model_name}@{fastest.quantization}",
                "fastest_speed": fastest.avg_tokens_per_sec,
                "slowest_model": f"{slowest.model_name}@{slowest.quantization}",
                "slowest_speed": slowest.avg_tokens_per_sec,
                "db_size_mb": round(db_size_mb, 2)
            }
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Abrufen von Cache-Statistiken: {e}")
        return {"success": False, "error": str(e)}
        logger.error(f"❌ Fehler beim Laden der Cache-Stats: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/cache/{model_key}")
async def delete_cache_entry(model_key: str) -> dict:
    """Löscht einen einzelnen Cache-Eintrag"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache nicht verfügbar"}
    
    try:
        import sqlite3
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Prüfe ob Eintrag existiert
        cursor.execute("SELECT COUNT(*) FROM benchmark_results WHERE model_key = ?", (model_key,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            conn.close()
            return {"success": False, "error": f"Model {model_key} nicht im Cache gefunden"}
        
        # Lösche Eintrag
        cursor.execute("DELETE FROM benchmark_results WHERE model_key = ?", (model_key,))
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        logger.info(f"🗑️ Cache-Eintrag gelöscht: {model_key}")
        return {
            "success": True,
            "message": f"✅ {deleted_count} Eintrag(e) gelöscht",
            "model_key": model_key
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Löschen des Cache-Eintrags: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/cache/clear")
async def clear_cache() -> dict:
    """Leert den gesamten Cache mit Backup"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache nicht verfügbar"}
    
    try:
        import sqlite3
        import shutil
        
        # Erstelle Backup vor dem Löschen
        backup_dir = RESULTS_DIR / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_file = backup_dir / f"benchmark_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}_backup.db"
        
        try:
            shutil.copy2(DATABASE_FILE, backup_file)
            logger.info(f"💾 Backup erstellt: {backup_file}")
        except Exception as backup_error:
            logger.warning(f"⚠️ Backup-Fehler (Cache wird trotzdem geleert): {backup_error}")
            backup_file = None
        
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        
        # Zähle Einträge vor dem Löschen
        cursor.execute("SELECT COUNT(*) FROM benchmark_results")
        count_before = cursor.fetchone()[0]
        
        # Lösche alle Einträge
        cursor.execute("DELETE FROM benchmark_results")
        conn.commit()
        conn.close()
        
        logger.warning(f"⚠️ Cache komplett geleert: {count_before} Einträge gelöscht")
        logger.warning(f"💾 Backup verfügbar unter: {backup_file}")
        
        return {
            "success": True,
            "message": f"✅ Cache geleert: {count_before} Einträge gelöscht",
            "deleted_count": count_before,
            "backup_file": str(backup_file) if backup_file else None
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Leeren des Cache: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/output")
async def get_output() -> dict:
    """Gibt aktuellen Output"""
    return {
        "output": manager.current_output,
        "status": manager.status
    }


@app.get("/api/dashboard/stats")
async def get_dashboard_stats() -> dict:
    """Dashboard-Statistiken für Home-View"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache nicht verfügbar"}
    
    try:
        import platform
        import sqlite3
        import psutil
        import json
        import subprocess
        from datetime import datetime
        
        cache = BenchmarkCache(DATABASE_FILE)
        results = cache.get_all_results()

        # LM Studio Healthcheck (CLI, tolerant gegenüber fehlendem --json)
        lmstudio_health = {"ok": False, "status": "unknown"}
        try:
            # Prefer JSON output if available
            try:
                output = subprocess.check_output(["lms", "status", "--json"], timeout=3)
                status_json = json.loads(output.decode()) if output else None
                if isinstance(status_json, dict):
                    lmstudio_health = {
                        "ok": True,
                        "status": status_json.get("status", "online"),
                        "version": status_json.get("version")
                    }
                else:
                    lmstudio_health = {"ok": True, "status": "online"}
            except Exception:
                # Fallback: parse plain text output
                output = subprocess.check_output(["lms", "status"], timeout=3, text=True)
                text = output.lower() if output else ""
                is_online = "online" in text or "running" in text or "server" in text
                lmstudio_health = {
                    "ok": is_online,
                    "status": output.strip() if output else "unknown",
                    "version": None
                }
        except Exception as health_err:
            lmstudio_health = {
                "ok": False,
                "status": str(health_err)
            }
        
        # System-Info (erweitert)
        system_info = {
            "os": platform.system(),
            "python_version": platform.python_version(),
            "cpu": platform.processor() or platform.machine(),
            "cpu_cores": psutil.cpu_count(logical=False),  # Physical cores
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        
        # GPU-Info aus erstem Result (wenn verfügbar)
        gpu_info = None
        if results:
            try:
                import subprocess
                vram_info = {"vram_gb": None, "gtt_gb": None, "total_gb": None}
                
                # Versuche NVIDIA GPU Info zu ermitteln
                if results[0].gpu_type == "NVIDIA":
                    try:
                        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], timeout=5)
                        vram_total_mb = int(output.decode().strip().split('\n')[0])
                        vram_info["vram_gb"] = round(vram_total_mb / 1024, 2)
                    except:
                        vram_info["vram_gb"] = None
                
                gpu_info = {
                    "type": results[0].gpu_type,
                    "vram_gb": vram_info["vram_gb"],
                    "gtt_gb": system_info["ram_gb"],  # System RAM als GTT
                    "total_gb": vram_info["vram_gb"] + system_info["ram_gb"] if vram_info["vram_gb"] else system_info["ram_gb"]
                }
            except:
                gpu_info = {
                    "type": results[0].gpu_type,
                    "vram_gb": None,
                    "gtt_gb": system_info["ram_gb"],
                    "total_gb": None
                }
        
        # Cache-Statistiken
        cache_stats = {
            "total_models": len(results),
            "total_runs": len(results),
            "db_size_mb": round(DATABASE_FILE.stat().st_size / (1024 * 1024), 2) if DATABASE_FILE.exists() else 0
        }
        
        # Performance-Statistiken
        perf_stats = {}
        if results:
            speeds = [r.avg_tokens_per_sec for r in results]
            perf_stats = {
                "avg_speed": round(sum(speeds) / len(speeds), 2),
                "max_speed": round(max(speeds), 2),
                "min_speed": round(min(speeds), 2)
            }
        
        # Top 5 Schnellste Modelle
        top_models = []
        fastest_model = None
        if results:
            sorted_results = sorted(results, key=lambda r: r.avg_tokens_per_sec, reverse=True)
            for i, r in enumerate(sorted_results[:5]):
                top_models.append({
                    "model_name": r.model_name,
                    "quantization": r.quantization,
                    "speed": round(r.avg_tokens_per_sec, 2),
                    "vram_mb": r.vram_mb,
                    "params_size": r.params_size
                })
                # Speichere schnellstes Modell
                if i == 0:
                    fastest_model = {
                        "name": r.model_name,
                        "speed": round(r.avg_tokens_per_sec, 2)
                    }
        
        # Letzte 10 Benchmark-Runs (mit Timestamp)
        recent_runs = []
        last_run_timestamp = None
        if results:
            sorted_by_time = sorted(results, key=lambda r: r.timestamp, reverse=True)
            for i, r in enumerate(sorted_by_time[:10]):
                recent_runs.append({
                    "model_name": r.model_name,
                    "quantization": r.quantization,
                    "speed": round(r.avg_tokens_per_sec, 2),
                    "timestamp": r.timestamp,
                    "gpu_offload": r.gpu_offload
                })
                # Speichere letzten Run Timestamp
                if i == 0:
                    last_run_timestamp = r.timestamp
        
        return {
            "success": True,
            "system_info": system_info,
            "gpu_info": gpu_info,
            "cache_stats": cache_stats,
            "perf_stats": perf_stats,
            "top_models": top_models,
            "recent_runs": recent_runs,
            "fastest_model": fastest_model,
            "last_run": last_run_timestamp,
            "lmstudio": lmstudio_health
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Dashboard-Stats: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# WebSocket - Live Streaming
# ============================================================================

@app.websocket("/ws/benchmark")
async def websocket_benchmark(websocket: WebSocket):
    """WebSocket für Live-Streaming von Benchmark-Output"""
    await websocket.accept()
    manager.connected_clients.add(websocket)
    
    heartbeat_count = 0
    
    try:
        logger.info(f"✅ WebSocket Client verbunden (Total: {len(manager.connected_clients)})")
        
        # Sende initialen Status
        await websocket.send_json({
            "type": "status",
            "status": manager.status,
            "running": manager.is_running()
        })
        
        # Stream Output während Benchmark läuft
        while True:
            # Prüfe auf Client-Nachrichten (mit kurzen Timeout)
            try:
                # Client kann commands schicken (zukünftige Erweiterung)
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                logger.debug(f"📨 Client message: {data}")
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                logger.info("⚠️ WebSocket Client hat Verbindung getrennt")
                break
            except Exception as e:
                logger.error(f"❌ WebSocket Receive Error: {e}")
                break
            
            # Lese Output wenn Benchmark läuft
            if manager.is_running():
                output = manager.drain_output_queue()
                if output:
                    try:
                        await websocket.send_json({
                            "type": "output",
                            "line": output,
                            "status": manager.status
                        })
                        heartbeat_count = 0  # Reset heartbeat counter
                    except Exception as e:
                        logger.error(f"❌ WebSocket Send Error: {e}")
                        break
                
                # Sende Hardware-Monitoring-Daten periodisch (alle 2s)
                current_time = time.time()
                if current_time - manager.last_hardware_send_time >= 2.0:
                    if manager.hardware_history["temperatures"] or manager.hardware_history["power"] or manager.hardware_history["vram"] or manager.hardware_history["gtt"] or manager.hardware_history["cpu"] or manager.hardware_history["ram"]:
                        try:
                            # Behalte nur letzte 60 Einträge pro Metrik (2 Minuten bei 2s Intervall)
                            max_history = 60
                            hardware_data = {
                                "temperatures": manager.hardware_history["temperatures"][-max_history:],
                                "power": manager.hardware_history["power"][-max_history:],
                                "vram": manager.hardware_history["vram"][-max_history:],
                                "gtt": manager.hardware_history["gtt"][-max_history:],
                                "cpu": manager.hardware_history["cpu"][-max_history:],
                                "ram": manager.hardware_history["ram"][-max_history:]
                            }
                            
                            await websocket.send_json({
                                "type": "hardware",
                                "data": hardware_data
                            })
                            manager.last_hardware_send_time = current_time
                        except Exception as e:
                            logger.error(f"❌ WebSocket Hardware Send Error: {e}")
            else:
                # Keep-Alive Heartbeat: Sende Status periodisch
                heartbeat_count += 1
                if heartbeat_count % 2 == 0:  # Alle 1 Sekunde (0.5s * 2)
                    try:
                        await websocket.send_json({
                            "type": "status",
                            "status": manager.status,
                            "running": manager.is_running()
                        })
                    except Exception as e:
                        logger.error(f"❌ WebSocket Heartbeat Error: {e}")
                        break
                
                # Beende Streaming wenn Benchmark komplett
                if manager.status == "completed":
                    try:
                        await websocket.send_json({
                            "type": "completed",
                            "message": "✅ Benchmark abgeschlossen"
                        })
                    except Exception as e:
                        logger.warning(f"⚠️ Konnte Completion-Message nicht senden: {e}")
                    break
                
                await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        logger.info("ℹ️ WebSocket normaler Disconnect")
    except Exception as e:
        logger.error(f"❌ WebSocket Error: {e}")
    finally:
        manager.connected_clients.discard(websocket)
        logger.info(f"❌ WebSocket Client getrennt (Total: {len(manager.connected_clients)})")


# ============================================================================
# Latest Results Export
# ============================================================================

@app.get("/api/latest-results")
async def get_latest_results() -> dict:
    """Findet die neuesten Benchmark-Ergebnisse"""
    try:
        results_dir = RESULTS_DIR  # /home/robby/Temp/local-llm-bench/results/
        
        # Finde alle benchmark_results_*.json Dateien
        json_files = list(results_dir.glob("benchmark_results_*.json"))
        
        if not json_files:
            return {"latest": None}
        
        # Sortiere nach Modifizierungszeit, neueste zuerst
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        # Gib nur den Dateinamen zurück (ohne Pfad, weil /results/ schon gemountet ist)
        return {"latest": latest_file.name}
    except Exception as e:
        logger.error(f"Fehler beim Finden der neuesten Ergebnisse: {e}")
        return {"latest": None}


# ============================================================================
# Health Check
# ============================================================================

@app.get("/health")
async def health_check() -> dict:
    """Health Check Endpoint"""
    return {
        "status": "ok",
        "benchmark_running": manager.is_running(),
        "connected_clients": len(manager.connected_clients)
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup beim Shutdown"""
    if manager.is_running():
        logger.info("🛑 Benchmark bei Shutdown beenden...")
        manager.stop_benchmark()
        await asyncio.sleep(1)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # ArgumentParser für Port-Option
    parser = argparse.ArgumentParser(description="FastAPI Web-Dashboard für LM Studio Benchmark")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port für Web-Dashboard (Standard: automatisch freien Port suchen)"
    )
    args = parser.parse_args()
    
    # Erstelle WebApp Startup Log-Datei
    webapp_log_file = setup_webapp_logger()
    
    logger.info("🌐 Starte FastAPI Web-Dashboard...")
    logger.info(f"📝 WebApp-Log: {webapp_log_file}")
    logger.info(f"📁 Projekt-Root: {PROJECT_ROOT}")
    logger.info(f"📄 Benchmark-Script: {BENCHMARK_SCRIPT}")
    
    if not BENCHMARK_SCRIPT.exists():
        logger.error(f"❌ Benchmark-Script nicht gefunden: {BENCHMARK_SCRIPT}")
        sys.exit(1)
    
    # Port bestimmen
    if args.port:
        port = args.port
        logger.info(f"🔧 Verwende angegebenen Port: {port}")
    else:
        port = find_free_port()
        logger.info(f"🎲 Nutze automatisch gefundenen freien Port: {port}")
    
    dashboard_url = f"http://localhost:{port}"
    logger.info(f"🚀 Dashboard verfügbar auf {dashboard_url}")
    logger.info(f"📊 API Docs: {dashboard_url}/docs")
    
    # Öffne Browser in separatem Thread nach kurzer Verzögerung
    def open_browser():
        time.sleep(1.5)  # Warte bis Server bereit ist
        try:
            logger.info(f"🌐 Öffne Browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.warning(f"⚠️ Konnte Browser nicht öffnen: {e}")
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
