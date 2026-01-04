#!/usr/bin/env python3
"""
LM Studio Model Benchmark Tool

Automatisches Testen aller lokal installierten LM Studio Modelle und deren
Quantisierungen. Misst Token/s-Geschwindigkeit mit standardisiertem Prompt.
"""

import subprocess
import json
import csv
import logging
import os
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import psutil
from tqdm import tqdm


# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Konstanten
STANDARD_PROMPT = "Erkläre maschinelles Lernen in 3 Sätzen"
CONTEXT_LENGTH = 2048
GPU_OFFLOAD_LEVELS = [1.0, 0.7, 0.5, 0.3]
NUM_WARMUP_RUNS = 1
NUM_MEASUREMENT_RUNS = 3
RESULTS_DIR = Path("results")


@dataclass
class BenchmarkResult:
    """Ergebnisse eines einzelnen Benchmark-Durchlaufs"""
    model_name: str
    quantization: str
    gpu_type: str
    gpu_offload: float
    vram_mb: str
    avg_tokens_per_sec: float
    avg_ttft: float
    avg_gen_time: float
    prompt_tokens: int
    completion_tokens: int
    timestamp: str


class GPUMonitor:
    """Erkennt GPU-Typ und misst VRAM-Nutzung"""
    
    def __init__(self):
        self.gpu_type: Optional[str] = None
        self.gpu_tool: Optional[str] = None
        self._detect_gpu()
    
    def _find_tool(self, tool_name: str, search_paths: List[str]) -> Optional[str]:
        """Sucht Tool in PATH und spezifischen Pfaden"""
        # Zuerst in PATH suchen
        if shutil.which(tool_name):
            return tool_name
        
        # Dann in spezifischen Pfaden suchen
        for path in search_paths:
            full_path = Path(path) / tool_name
            if full_path.exists() and os.access(full_path, os.X_OK):
                return str(full_path)
        
        return None
    
    def _detect_gpu(self):
        """Erkennt GPU-Typ und findet entsprechendes Monitoring-Tool"""
        # NVIDIA
        nvidia_paths = ['/usr/bin', '/usr/local/bin', '/usr/local/cuda/bin']
        nvidia_tool = self._find_tool('nvidia-smi', nvidia_paths)
        if nvidia_tool:
            self.gpu_type = "NVIDIA"
            self.gpu_tool = nvidia_tool
            logger.info(f"NVIDIA GPU erkannt, Tool: {nvidia_tool}")
            return
        
        # AMD
        amd_paths = ['/usr/bin', '/usr/local/bin', '/opt/rocm/bin']
        amd_tool = self._find_tool('rocm-smi', amd_paths)
        if amd_tool:
            self.gpu_type = "AMD"
            self.gpu_tool = amd_tool
            logger.info(f"AMD GPU erkannt, Tool: {amd_tool}")
            return
        
        # Intel
        intel_paths = ['/usr/bin', '/usr/local/bin', '/usr/lib/xpu']
        intel_tool = self._find_tool('intel_gpu_top', intel_paths)
        if intel_tool:
            self.gpu_type = "Intel"
            self.gpu_tool = intel_tool
            logger.info(f"Intel GPU erkannt, Tool: {intel_tool}")
            return
        
        logger.warning("Keine GPU-Monitoring-Tools gefunden. VRAM-Messung nicht verfügbar.")
        self.gpu_type = "Unknown"
    
    def get_vram_usage(self) -> str:
        """Misst aktuelle VRAM-Nutzung"""
        if not self.gpu_tool:
            return "N/A"
        
        try:
            if self.gpu_type == "NVIDIA":
                result = subprocess.run(
                    [self.gpu_tool, '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            
            elif self.gpu_type == "AMD":
                result = subprocess.run(
                    [self.gpu_tool, '--showmeminfo', 'vram'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # Parse AMD output (format kann variieren)
                    for line in result.stdout.split('\n'):
                        if 'Used' in line or 'used' in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                return parts[1]
            
            elif self.gpu_type == "Intel":
                # Intel GPU Top hat kein einfaches VRAM-Query
                # Hier könnte man alternative Methoden verwenden
                return "N/A"
        
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"VRAM-Messung fehlgeschlagen: {e}")
        
        return "N/A"


class LMStudioServerManager:
    """Verwaltet LM Studio Server-Lifecycle"""
    
    @staticmethod
    def is_server_running() -> bool:
        """Prüft ob LM Studio Server läuft"""
        try:
            result = subprocess.run(
                ['lms', 'server', 'status'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0 and 'running' in result.stdout.lower()
        except Exception as e:
            logger.error(f"Fehler bei Server-Status-Prüfung: {e}")
            return False
    
    @staticmethod
    def start_server():
        """Startet LM Studio Server"""
        try:
            logger.info("Starte LM Studio Server...")
            subprocess.Popen(
                ['lms', 'server', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Warte bis Server bereit ist
            max_retries = 30
            for i in range(max_retries):
                time.sleep(2)
                if LMStudioServerManager.is_server_running():
                    logger.info("LM Studio Server erfolgreich gestartet")
                    return True
            
            logger.error("Server-StartTimeout nach 60 Sekunden")
            return False
        
        except Exception as e:
            logger.error(f"Fehler beim Starten des Servers: {e}")
            return False
    
    @staticmethod
    def ensure_server_running():
        """Stellt sicher dass Server läuft"""
        if not LMStudioServerManager.is_server_running():
            logger.info("Server läuft nicht, starte Server...")
            return LMStudioServerManager.start_server()
        logger.info("Server läuft bereits")
        return True


class ModelDiscovery:
    """Findet alle lokal installierten Modelle"""
    
    @staticmethod
    def get_installed_models() -> List[str]:
        """Listet alle lokal installierten Modelle und Quantisierungen auf"""
        try:
            result = subprocess.run(
                ['lms', 'ls'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Fehler bei 'lms ls': {result.stderr}")
                return []
            
            # Parse Output
            models = []
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Model'):
                    # Extrahiere Model-Key (kann Format haben wie: model-name@quant)
                    parts = line.split()
                    if parts:
                        model_key = parts[0]
                        models.append(model_key)
            
            logger.info(f"{len(models)} Modelle gefunden: {models}")
            return models
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Modelle: {e}")
            return []


class LMStudioBenchmark:
    """Haupt-Benchmark-Klasse"""
    
    def __init__(self):
        self.gpu_monitor = GPUMonitor()
        self.results: List[BenchmarkResult] = []
        
        # Erstelle Results-Verzeichnis
        RESULTS_DIR.mkdir(exist_ok=True)
    
    def benchmark_model(self, model_key: str) -> Optional[BenchmarkResult]:
        """Führt Benchmark für ein spezifisches Modell durch"""
        logger.info(f"Starte Benchmark für {model_key}")
        
        # Parse Model-Name und Quantisierung
        if '@' in model_key:
            model_name, quantization = model_key.split('@', 1)
        else:
            model_name = model_key
            quantization = "unknown"
        
        # Versuche Modell mit verschiedenen GPU-Offload-Levels zu laden
        for gpu_offload in GPU_OFFLOAD_LEVELS:
            try:
                logger.info(f"Versuche {model_key} mit GPU-Offload {gpu_offload}")
                
                # Lade Modell
                if not self._load_model(model_key, gpu_offload):
                    continue
                
                # Warmup
                logger.info(f"Warmup für {model_key}...")
                for _ in range(NUM_WARMUP_RUNS):
                    self._run_inference(model_key)
                
                # Messungen
                logger.info(f"Führe {NUM_MEASUREMENT_RUNS} Messungen durch...")
                measurements = []
                for _ in range(NUM_MEASUREMENT_RUNS):
                    vram_before = self.gpu_monitor.get_vram_usage()
                    stats = self._run_inference(model_key)
                    vram_after = self.gpu_monitor.get_vram_usage()
                    
                    if stats:
                        measurements.append(stats)
                
                # Berechne Durchschnitte
                if measurements:
                    result = self._calculate_averages(
                        model_name,
                        quantization,
                        gpu_offload,
                        vram_after,
                        measurements
                    )
                    
                    # Entlade Modell
                    self._unload_model(model_key)
                    
                    return result
                
            except Exception as e:
                logger.warning(f"Fehler mit GPU-Offload {gpu_offload}: {e}")
                continue
        
        # Wenn alle Versuche fehlschlagen
        logger.error(f"Konnte {model_key} nicht laden, überspringe")
        return None
    
    def _load_model(self, model_key: str, gpu_offload: float) -> bool:
        """Lädt ein Modell in den Speicher"""
        try:
            cmd = [
                'lms', 'load', model_key,
                '--gpu', str(gpu_offload),
                '--context-length', str(CONTEXT_LENGTH)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            return result.returncode == 0
        
        except Exception as e:
            logger.error(f"Fehler beim Laden von {model_key}: {e}")
            return False
    
    def _unload_model(self, model_key: str):
        """Entlädt ein Modell aus dem Speicher"""
        try:
            subprocess.run(
                ['lms', 'unload', model_key],
                capture_output=True,
                timeout=30
            )
        except Exception as e:
            logger.warning(f"Fehler beim Entladen von {model_key}: {e}")
    
    def _run_inference(self, model_key: str) -> Optional[Dict]:
        """Führt Inferenz durch und gibt Stats zurück"""
        try:
            # Hier müsste die tatsächliche LM Studio Python SDK Integration erfolgen
            # Da lmstudio SDK möglicherweise nicht verfügbar ist, verwenden wir
            # eine Placeholder-Implementierung
            
            # TODO: Implementiere mit lmstudio SDK:
            # import lmstudio as lms
            # model = lms.llm(model_key)
            # result = model.respond(STANDARD_PROMPT)
            # return result.stats
            
            logger.warning("LM Studio SDK Integration fehlt noch - Placeholder verwendet")
            time.sleep(1)  # Simuliere Inferenz
            
            return {
                'tokens_per_second': 50.0,
                'time_to_first_token': 0.1,
                'generation_time': 1.0,
                'prompt_tokens': 10,
                'completion_tokens': 50
            }
        
        except Exception as e:
            logger.error(f"Fehler bei Inferenz: {e}")
            return None
    
    def _calculate_averages(
        self,
        model_name: str,
        quantization: str,
        gpu_offload: float,
        vram_mb: str,
        measurements: List[Dict]
    ) -> BenchmarkResult:
        """Berechnet Durchschnittswerte aus Messungen"""
        avg_tokens_per_sec = sum(m['tokens_per_second'] for m in measurements) / len(measurements)
        avg_ttft = sum(m['time_to_first_token'] for m in measurements) / len(measurements)
        avg_gen_time = sum(m['generation_time'] for m in measurements) / len(measurements)
        prompt_tokens = measurements[0]['prompt_tokens']
        completion_tokens = int(sum(m['completion_tokens'] for m in measurements) / len(measurements))
        
        return BenchmarkResult(
            model_name=model_name,
            quantization=quantization,
            gpu_type=self.gpu_monitor.gpu_type,
            gpu_offload=gpu_offload,
            vram_mb=vram_mb,
            avg_tokens_per_sec=round(avg_tokens_per_sec, 2),
            avg_ttft=round(avg_ttft, 3),
            avg_gen_time=round(avg_gen_time, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
        )
    
    def run_all_benchmarks(self):
        """Führt Benchmarks für alle verfügbaren Modelle durch"""
        # Stelle sicher dass Server läuft
        if not LMStudioServerManager.ensure_server_running():
            logger.error("Server konnte nicht gestartet werden, breche ab")
            return
        
        # Hole alle Modelle
        models = ModelDiscovery.get_installed_models()
        if not models:
            logger.error("Keine Modelle gefunden")
            return
        
        logger.info(f"Starte Benchmark für {len(models)} Modelle...")
        
        # Benchmark für jedes Modell
        for model_key in tqdm(models, desc="Benchmarking Modelle"):
            result = self.benchmark_model(model_key)
            if result:
                self.results.append(result)
        
        # Exportiere Ergebnisse
        self.export_results()
        
        logger.info(f"Benchmark abgeschlossen. {len(self.results)}/{len(models)} Modelle erfolgreich getestet")
    
    def export_results(self):
        """Exportiert Ergebnisse als JSON und CSV"""
        if not self.results:
            logger.warning("Keine Ergebnisse zum Exportieren")
            return
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # JSON Export
        json_file = RESULTS_DIR / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, ensure_ascii=False)
        logger.info(f"JSON-Ergebnisse gespeichert: {json_file}")
        
        # CSV Export
        csv_file = RESULTS_DIR / f"benchmark_results_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
        logger.info(f"CSV-Ergebnisse gespeichert: {csv_file}")


def main():
    """Hauptfunktion"""
    logger.info("=== LM Studio Model Benchmark ===")
    logger.info(f"Standard-Prompt: '{STANDARD_PROMPT}'")
    logger.info(f"Context Length: {CONTEXT_LENGTH}")
    logger.info(f"Messungen pro Modell: {NUM_MEASUREMENT_RUNS} (+ {NUM_WARMUP_RUNS} Warmup)")
    
    benchmark = LMStudioBenchmark()
    benchmark.run_all_benchmarks()
    
    logger.info("Benchmark abgeschlossen!")


if __name__ == "__main__":
    main()
