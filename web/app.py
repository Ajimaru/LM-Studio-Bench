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
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    from benchmark import BenchmarkCache, BenchmarkResult
except ImportError as e:
    logger.error(f"❌ Konnte benchmark.py nicht importieren: {e}")
    BenchmarkCache = None  # type: ignore
    BenchmarkResult = None  # type: ignore

# FastAPI App initialisieren
app = FastAPI(
    title="LM Studio Benchmark Dashboard",
    description="Web-Dashboard zur Steuerung und Überwachung von LM Studio Benchmarks"
)

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

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def start_benchmark(self, args: list) -> bool:
        """Startet neuen Benchmark-Prozess"""
        if self.is_running():
            logger.warning("Benchmark läuft bereits")
            return False

        try:
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
            logger.info(f"✅ Benchmark gestartet mit PID {self.process.pid}")
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
            return True
        except Exception as e:
            logger.error(f"❌ Fehler beim Stoppen: {e}")
            return False

    async def read_output(self) -> str:
        """Liest verfügbaren Output aus Prozess mit Timeout"""
        if not self.process or not self.process.stdout:
            return ""

        try:
            # Nicht-blockierendes Lesen mit Timeout
            loop = asyncio.get_event_loop()
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self.process.stdout.readline
                ),
                timeout=2.0  # 2 Sekunden Timeout
            )
            
            if output:
                self.current_output += output
                return output
            elif not self.is_running():
                self.status = "completed"
            
            return ""
        except asyncio.TimeoutError:
            # Timeout ist normal - bedeutet kein Output verfügbar
            if not self.is_running():
                self.status = "completed"
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
        )
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
            result_dict = {
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
                "total_models": 0,
                "avg_speed": 0,
                "fastest_model": None,
                "slowest_model": None,
                "db_size_mb": 0
            }
        
        speeds = [r.avg_tokens_per_sec for r in results]
        fastest = max(results, key=lambda r: r.avg_tokens_per_sec)
        slowest = min(results, key=lambda r: r.avg_tokens_per_sec)
        
        # DB-Größe
        db_size_mb = DATABASE_FILE.stat().st_size / (1024 * 1024) if DATABASE_FILE.exists() else 0
        
        return {
            "success": True,
            "total_models": len(results),
            "avg_speed": sum(speeds) / len(speeds),
            "fastest_model": {
                "name": f"{fastest.model_name}@{fastest.quantization}",
                "speed": fastest.avg_tokens_per_sec
            },
            "slowest_model": {
                "name": f"{slowest.model_name}@{slowest.quantization}",
                "speed": slowest.avg_tokens_per_sec
            },
            "db_size_mb": round(db_size_mb, 2)
        }
    except Exception as e:
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
    """Leert den gesamten Cache"""
    if not BenchmarkCache:
        return {"success": False, "error": "BenchmarkCache nicht verfügbar"}
    
    try:
        import sqlite3
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
        return {
            "success": True,
            "message": f"✅ Cache geleert: {count_before} Einträge gelöscht",
            "deleted_count": count_before
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
                output = await manager.read_output()
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
    
    logger.info("🌐 Starte FastAPI Web-Dashboard...")
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
    
    logger.info(f"🚀 Dashboard verfügbar auf http://localhost:{port}")
    logger.info(f"📊 API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
