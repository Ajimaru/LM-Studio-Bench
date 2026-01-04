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
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil
from statistics import quantiles, mean, median
from tqdm import tqdm
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    go = None
    PLOTLY_AVAILABLE = False


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

# Optimierte Inference-Parameter für standardisierte Benchmarks
# (Für konsistente, reproduzierbare Messungen)
OPTIMIZED_INFERENCE_PARAMS = {
    'temperature': 0.1,        # Niedrig für konsistente Ergebnisse (statt default 0.8)
    'top_k_sampling': 40,      # Sampling aus top 40 Tokens
    'top_p_sampling': 0.9,     # Nucleus sampling bei 90%
    'min_p_sampling': 0.05,    # Minimum probability threshold
    'repeat_penalty': 1.2,     # Leichte Strafe gegen Wiederholungen (default 1.1)
    'max_tokens': 256,         # Begrenzte Output-Länge für schnellere Messungen
}


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
    
    # Modell-Metadaten
    params_size: str                   # z.B. "3B", "7B"
    architecture: str                  # z.B. "mistral3", "gemma3"
    max_context_length: int            # z.B. 262144
    model_size_gb: float               # z.B. 3.11 (Dateigröße in GB)
    has_vision: bool                   # Vision-Support
    has_tools: bool                    # Tool-Calling Support
    
    # Effizienz-Metriken
    tokens_per_sec_per_gb: float       # Tokens/s pro GB Modellgröße
    tokens_per_sec_per_billion_params: float  # Tokens/s pro Milliarde Parameter
    
    # Historischer Vergleich (optional)
    speed_delta_pct: Optional[float] = None     # Performance-Veränderung (%) vs. vorheriger Benchmark
    prev_timestamp: Optional[str] = None         # Zeitstempel des vorherigen Benchmarks


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
        
        # AMD - Suche in ROCm Versionsverzeichnissen
        amd_paths = ['/usr/bin', '/usr/local/bin', '/opt/rocm/bin']
        # Erweitere AMD Pfade mit versionierten ROCm Verzeichnissen
        import glob
        rocm_versions = glob.glob('/opt/rocm-*/bin')
        amd_paths.extend(rocm_versions)
        
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
                    # Parse AMD rocm-smi output: "GPU[X] : VRAM Total Used Memory (B): XXXXXX"
                    for line in result.stdout.split('\n'):
                        if 'VRAM Total Used Memory' in line:
                            # Format: "GPU[0]          : VRAM Total Used Memory (B): 1674899456"
                            parts = line.split(':')
                            if len(parts) >= 3:
                                # Letzter Teil contains "1674899456"
                                bytes_used = parts[-1].strip()
                                try:
                                    # Konvertiere zu MB
                                    mb_used = int(bytes_used) / (1024 * 1024)
                                    return f"{int(mb_used)}"
                                except ValueError:
                                    pass
            
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
            # Server gibt Ausgabe in stderr aus (nicht stdout!)
            output = result.stdout + result.stderr
            return result.returncode == 0 and ('running' in output.lower() or 'port' in output.lower())
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
    
    _metadata_cache: Dict[str, Dict] = {}  # Class-level cache
    
    @staticmethod
    def _get_metadata_cache() -> Dict[str, Dict]:
        """Cache für Modell-Metadaten (geladen einmal am Anfang)"""
        if not hasattr(ModelDiscovery, '_metadata_cache'):
            ModelDiscovery._metadata_cache = {}
            try:
                result = subprocess.run(
                    ['lms', 'ls', '--json'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    for model_data in data:
                        if model_data.get('type') == 'llm':
                            key = model_data.get('modelKey')
                            ModelDiscovery._metadata_cache[key] = {
                                'architecture': model_data.get('architecture', 'unknown'),
                                'params_size': model_data.get('paramsString', 'unknown'),
                                'max_context_length': model_data.get('maxContextLength', 0),
                                'model_size_gb': round(model_data.get('sizeBytes', 0) / 1024**3, 2),
                                'has_vision': model_data.get('vision', False),
                                'has_tools': model_data.get('trainedForToolUse', False),
                            }
            except Exception as e:
                logger.warning(f"Fehler beim Laden von Metadaten-Cache: {e}")
        return ModelDiscovery._metadata_cache
    
    @staticmethod
    def get_model_metadata(model_key: str) -> Dict:
        """Hole Metadaten für ein bestimmtes Modell"""
        cache = ModelDiscovery._get_metadata_cache()
        # Extrahiere Model-Name (vor @)
        base_model = model_key.split('@')[0] if '@' in model_key else model_key
        return cache.get(base_model, {
            'architecture': 'unknown',
            'params_size': 'unknown',
            'max_context_length': 0,
            'model_size_gb': 0.0,
            'has_vision': False,
            'has_tools': False,
        })
    
    @staticmethod
    def get_installed_models() -> List[str]:
        """Listet alle lokal installierten Modelle und Quantisierungen auf"""
        try:
            result = subprocess.run(
                ['lms', 'ls', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.error(f"Fehler bei 'lms ls': {result.stderr}")
                return []
            
            # Parse JSON Output
            models = []
            import json
            data = json.loads(result.stdout)
            
            for model_data in data:
                # Nur LLM-Modelle, keine Embeddings
                if model_data.get('type') == 'llm':
                    # Hole alle Varianten (Quantisierungen)
                    variants = model_data.get('variants', [])
                    if variants:
                        models.extend(variants)
                    else:
                        # Fallback wenn keine Varianten
                        models.append(model_data.get('modelKey'))
            
            logger.info(f"{len(models)} Modelle gefunden")
            if models:
                logger.info(f"Erste 5 Modelle: {models[:5]}")
            return models
        
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Modelle: {e}")
            return []
    
    @staticmethod
    def filter_models(models: List[str], filter_args: Dict) -> List[str]:
        """Filtert Modelle basierend auf CLI-Argumenten"""
        if not filter_args:
            return models
        
        # Lade Metadaten-Cache einmal
        metadata_cache = ModelDiscovery._get_metadata_cache()
        filtered = []
        
        for model_key in models:
            # Hole Metadaten für Modell
            metadata = ModelDiscovery.get_model_metadata(model_key)
            
            # Filter: nur Vision-Modelle
            if filter_args.get('only_vision') and not metadata['has_vision']:
                continue
            
            # Filter: nur Tool-Modelle
            if filter_args.get('only_tools') and not metadata['has_tools']:
                continue
            
            # Filter: Quantisierungen
            if filter_args.get('quants'):
                quants_list = [q.strip().lower() for q in filter_args['quants'].split(',')]
                # Extrahiere Quantisierung aus model_key (z.B. "model@q4_k_m")
                quant = model_key.split('@')[-1].lower() if '@' in model_key else ''
                # Prüfe ob Quantisierung in der Liste ist
                if not any(q in quant for q in quants_list):
                    continue
            
            # Filter: Architekturen
            if filter_args.get('arch'):
                arch_list = [a.strip().lower() for a in filter_args['arch'].split(',')]
                if metadata['architecture'].lower() not in arch_list:
                    continue
            
            # Filter: Parametergrößen
            if filter_args.get('params'):
                params_list = [p.strip().upper() for p in filter_args['params'].split(',')]
                if metadata['params_size'].upper() not in params_list:
                    continue
            
            # Filter: Minimale Context-Length
            if filter_args.get('min_context'):
                if metadata['max_context_length'] < filter_args['min_context']:
                    continue
            
            # Filter: Maximale Dateigröße
            if filter_args.get('max_size'):
                if metadata['model_size_gb'] > filter_args['max_size']:
                    continue
            
            filtered.append(model_key)
        
        logger.info(f"Nach Filterung: {len(filtered)}/{len(models)} Modelle übrig")
        return filtered



class LMStudioBenchmark:
    """Haupt-Benchmark-Klasse"""
    
    def __init__(self, num_runs: int = 3, context_length: int = 2048, prompt: str = "Erkläre maschinelles Lernen in 3 Sätzen", model_limit: Optional[int] = None, filter_args: Optional[Dict] = None, compare_with: Optional[str] = None, rank_by: str = 'speed'):
        self.gpu_monitor = GPUMonitor()
        self.results: List[BenchmarkResult] = []
        self.num_measurement_runs = num_runs
        self.context_length = context_length
        self.prompt = prompt
        self.model_limit = model_limit
        self.filter_args = filter_args or {}
        self.compare_with = compare_with
        self.rank_by = rank_by
        self.previous_results: List[BenchmarkResult] = []
        
        # Erstelle Results-Verzeichnis
        RESULTS_DIR.mkdir(exist_ok=True)
        
        # Lade frühere Ergebnisse wenn Vergleich gewünscht
        if self.compare_with:
            self._load_previous_results()
    
    def _load_previous_results(self):
        """Lädt frühere Benchmark-Ergebnisse zum Vergleich"""
        try:
            # Suche nach passenden JSON-Datei
            if not self.compare_with:
                return
            
            if self.compare_with.endswith('.json'):
                json_file = RESULTS_DIR / self.compare_with
            else:
                # Versuche Datum zu parsen (z.B. "20260104" oder "latest")
                if self.compare_with.lower() == "latest":
                    # Finde neueste Datei
                    json_files = sorted(RESULTS_DIR.glob('benchmark_results_*.json'))
                    if not json_files:
                        logger.warning("Keine früheren Benchmark-Dateien gefunden")
                        return
                    json_file = json_files[-1]
                else:
                    json_file = RESULTS_DIR / f"benchmark_results_{self.compare_with}.json"
            
            if not json_file.exists():
                logger.warning(f"Datei nicht gefunden: {json_file}")
                return
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.previous_results = [BenchmarkResult(**item) for item in data]
            
            logger.info(f"✓ {len(self.previous_results)} frühere Ergebnisse geladen aus {json_file.name}")
        except Exception as e:
            logger.error(f"Fehler beim Laden von frühen Ergebnissen: {e}")
    
    def _calculate_delta(self, current: BenchmarkResult) -> Optional[Dict]:
        """Berechnet Delta zu früherem Benchmark für selbes Modell+Quantisierung"""
        if not self.previous_results:
            return None
        
        # Suche nach exaktem Match (selbes Modell + Quantisierung)
        for prev in self.previous_results:
            if prev.model_name == current.model_name and prev.quantization == current.quantization:
                speed_delta = current.avg_tokens_per_sec - prev.avg_tokens_per_sec
                speed_delta_pct = (speed_delta / prev.avg_tokens_per_sec * 100) if prev.avg_tokens_per_sec > 0 else 0
                
                return {
                    'prev_speed': prev.avg_tokens_per_sec,
                    'current_speed': current.avg_tokens_per_sec,
                    'speed_delta': speed_delta,
                    'speed_delta_pct': speed_delta_pct,
                    'prev_timestamp': prev.timestamp
                }
        
        return None

    
    def benchmark_model(self, model_key: str) -> Optional[BenchmarkResult]:
        """Führt Benchmark für ein spezifisches Modell durch"""
        logger.info(f"Starte Benchmark für {model_key}")
        
        # Entlade alle anderen Modelle zuerst
        try:
            subprocess.run(
                ['lms', 'unload', '--all'],
                capture_output=True,
                text=True,
                timeout=30
            )
            logger.info("Alle Modelle entladen")
            time.sleep(1)  # Warte bis Speicher freigegeben
        except Exception as e:
            logger.warning(f"Fehler beim Entladen aller Modelle: {e}")
        
        # Parse Model-Name und Quantisierung
        if '@' in model_key:
            model_name, quantization = model_key.split('@', 1)
        else:
            model_name = model_key
            quantization = "unknown"
        
        try:
            # Warmup
            logger.info(f"Warmup für {model_key}...")
            for _ in range(NUM_WARMUP_RUNS):
                warmup_result = self._run_inference(model_key)
                if not warmup_result:
                    logger.error(f"Warmup für {model_key} fehlgeschlagen")
                    return None
            
            # Messungen
            logger.info(f"Führe {self.num_measurement_runs} Messungen durch...")
            measurements = []
            vram_after = "N/A"  # Initialisiere Standard-Wert
            for run in range(self.num_measurement_runs):
                vram_before = self.gpu_monitor.get_vram_usage()
                stats = self._run_inference(model_key)
                vram_after = self.gpu_monitor.get_vram_usage()
                
                if stats:
                    measurements.append(stats)
                    logger.info(f"Run {run+1}/{self.num_measurement_runs}: {stats['tokens_per_second']:.2f} tokens/s")
                else:
                    logger.warning(f"Run {run+1}/{self.num_measurement_runs} fehlgeschlagen")
            
            # Berechne Durchschnitte
            if measurements:
                result = self._calculate_averages(
                    model_name,
                    quantization,
                    1.0,  # SDK handhabt GPU-Offload automatisch
                    vram_after,
                    measurements,
                    model_key
                )
                
                logger.info(f"✓ {model_key}: {result.avg_tokens_per_sec:.2f} tokens/s")
                return result
            else:
                logger.error(f"Keine erfolgreichen Messungen für {model_key}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler beim Benchmarking von {model_key}: {e}")
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
            import lmstudio as lms
            
            # Lade Modell über SDK mit Kontextlängen-Konfiguration
            model = lms.llm(
                model_key,
                config=lms.LlmLoadModelConfig(
                    context_length=self.context_length
                )
            )
            
            # Erstelle optimierte Prediction-Konfiguration
            prediction_config = lms.LlmPredictionConfig(
                temperature=OPTIMIZED_INFERENCE_PARAMS['temperature'],
                top_k_sampling=OPTIMIZED_INFERENCE_PARAMS['top_k_sampling'],
                top_p_sampling=OPTIMIZED_INFERENCE_PARAMS['top_p_sampling'],
                min_p_sampling=OPTIMIZED_INFERENCE_PARAMS['min_p_sampling'],
                repeat_penalty=OPTIMIZED_INFERENCE_PARAMS['repeat_penalty'],
                max_tokens=OPTIMIZED_INFERENCE_PARAMS['max_tokens']
            )
            
            # Führe Inferenz durch mit optimierten Parametern
            start_time = time.time()
            result = model.respond(self.prompt, config=prediction_config)
            end_time = time.time()
            
            generation_time = end_time - start_time
            
            # Extrahiere Stats
            stats = result.stats
            
            tokens_per_sec = 0.0
            if generation_time > 0 and stats.predicted_tokens_count:
                tokens_per_sec = float(stats.predicted_tokens_count) / generation_time
            
            return {
                'tokens_per_second': tokens_per_sec,
                'time_to_first_token': stats.time_to_first_token_sec,
                'generation_time': generation_time,
                'prompt_tokens': stats.prompt_tokens_count or 0,
                'completion_tokens': stats.predicted_tokens_count or 0
            }
        
        except Exception as e:
            logger.error(f"Fehler bei Inferenz mit {model_key}: {e}")
            return None
    
    def _calculate_averages(
        self,
        model_name: str,
        quantization: str,
        gpu_offload: float,
        vram_mb: str,
        measurements: List[Dict],
        model_key: str
    ) -> BenchmarkResult:
        """Berechnet Durchschnittswerte aus Messungen"""
        avg_tokens_per_sec = sum(m['tokens_per_second'] for m in measurements) / len(measurements)
        avg_ttft = sum(m['time_to_first_token'] for m in measurements) / len(measurements)
        avg_gen_time = sum(m['generation_time'] for m in measurements) / len(measurements)
        prompt_tokens = measurements[0]['prompt_tokens']
        completion_tokens = int(sum(m['completion_tokens'] for m in measurements) / len(measurements))
        
        # Hole Metadaten
        metadata = ModelDiscovery.get_model_metadata(model_key)
        
        # Berechne Effizienz-Metriken
        model_size_gb = metadata.get('model_size_gb', 0.0)
        params_size_str = metadata.get('params_size', 'unknown')
        
        # tokens/s pro GB
        tokens_per_sec_per_gb = round(avg_tokens_per_sec / model_size_gb, 2) if model_size_gb > 0 else 0.0
        
        # tokens/s pro Milliarde Parameter
        # Extrahiere Zahl aus params_size (z.B. "7B" -> 7.0, "8.3B" -> 8.3)
        try:
            params_billion = float(params_size_str.upper().replace('B', '').strip())
            tokens_per_sec_per_billion_params = round(avg_tokens_per_sec / params_billion, 2)
        except (ValueError, AttributeError):
            tokens_per_sec_per_billion_params = 0.0
        
        result = BenchmarkResult(
            model_name=model_name,
            quantization=quantization,
            gpu_type=self.gpu_monitor.gpu_type or "Unknown",
            gpu_offload=gpu_offload,
            vram_mb=vram_mb,
            avg_tokens_per_sec=round(avg_tokens_per_sec, 2),
            avg_ttft=round(avg_ttft, 3),
            avg_gen_time=round(avg_gen_time, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            params_size=metadata.get('params_size', 'unknown'),
            architecture=metadata.get('architecture', 'unknown'),
            max_context_length=metadata.get('max_context_length', 0),
            model_size_gb=model_size_gb,
            has_vision=metadata.get('has_vision', False),
            has_tools=metadata.get('has_tools', False),
            tokens_per_sec_per_gb=tokens_per_sec_per_gb,
            tokens_per_sec_per_billion_params=tokens_per_sec_per_billion_params
        )
        
        # Berechne Delta zu vorherigem Benchmark falls vorhanden
        delta = self._calculate_delta(result)
        if delta:
            result.speed_delta_pct = delta['speed_delta_pct']
            result.prev_timestamp = delta['prev_timestamp']
        
        return result
    
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
        
        # Wende Filter an
        models = ModelDiscovery.filter_models(models, self.filter_args)
        if not models:
            logger.error("Keine Modelle nach Filterung übrig")
            return
        
        # Wende Limit an wenn gesetzt
        if self.model_limit and self.model_limit < len(models):
            logger.info(f"Modell-Limit gesetzt: Testet nur erste {self.model_limit} von {len(models)} Modellen")
            models = models[:self.model_limit]
        
        logger.info(f"Starte Benchmark für {len(models)} Modelle...")
        
        # Benchmark für jedes Modell
        for model_key in tqdm(models, desc="Benchmarking Modelle"):
            result = self.benchmark_model(model_key)
            if result:
                self.results.append(result)

        
        # Exportiere Ergebnisse
        self.export_results()
        
        logger.info(f"Benchmark abgeschlossen. {len(self.results)}/{len(models)} Modelle erfolgreich getestet")
    
    def _analyze_best_quantizations(self) -> Dict[str, Dict]:
        """Analysiert beste Quantisierung pro Modell nach verschiedenen Kriterien"""
        best_by_model = {}
        
        for result in self.results:
            model_key = result.model_name
            
            if model_key not in best_by_model:
                best_by_model[model_key] = {
                    'best_speed': None,
                    'best_efficiency': None,
                    'best_ttft': None,
                    'all_quantizations': []
                }
            
            # Speichere alle Quantisierungen
            best_by_model[model_key]['all_quantizations'].append(result)
            
            # Bestes Speed
            if best_by_model[model_key]['best_speed'] is None or \
               result.avg_tokens_per_sec > best_by_model[model_key]['best_speed'].avg_tokens_per_sec:
                best_by_model[model_key]['best_speed'] = result
            
            # Beste Effizienz (tokens/s per GB)
            if best_by_model[model_key]['best_efficiency'] is None or \
               result.tokens_per_sec_per_gb > best_by_model[model_key]['best_efficiency'].tokens_per_sec_per_gb:
                best_by_model[model_key]['best_efficiency'] = result
            
            # Bestes TTFT (Time to First Token - niedrig = gut)
            if best_by_model[model_key]['best_ttft'] is None or \
               result.avg_ttft < best_by_model[model_key]['best_ttft'].avg_ttft:
                best_by_model[model_key]['best_ttft'] = result
        
        return best_by_model
    
    def sort_results(self, rank_by: str = 'speed') -> List[BenchmarkResult]:
        """Sortiert Ergebnisse nach verschiedenen Kriterien"""
        if rank_by == 'speed':
            return sorted(self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True)
        elif rank_by == 'efficiency':
            return sorted(self.results, key=lambda x: x.tokens_per_sec_per_gb, reverse=True)
        elif rank_by == 'ttft':
            return sorted(self.results, key=lambda x: x.avg_ttft, reverse=False)  # Niedrig = gut
        elif rank_by == 'vram':
            # Parse VRAM-Wert (z.B. "2048 MB" -> 2048)
            def get_vram_mb(result):
                try:
                    return float(result.vram_mb.split()[0]) if isinstance(result.vram_mb, str) else float(result.vram_mb)
                except:
                    return 999999
            return sorted(self.results, key=get_vram_mb, reverse=False)  # Niedrig = besser
        else:
            return sorted(self.results, key=lambda x: x.avg_tokens_per_sec, reverse=True)  # Default
    
    def calculate_percentile_stats(self) -> Dict[str, Dict]:
        """Berechnet P50, P95, P99 Statistiken für Benchmark-Metriken"""
        if not self.results or len(self.results) < 3:
            return {}
        
        speeds = [r.avg_tokens_per_sec for r in self.results if r.avg_tokens_per_sec > 0]
        ttfts = [r.avg_ttft for r in self.results if r.avg_ttft > 0]
        vram_values = []
        for r in self.results:
            try:
                vram_mb = float(r.vram_mb.split()[0]) if isinstance(r.vram_mb, str) else float(r.vram_mb)
                vram_values.append(vram_mb)
            except:
                pass
        
        stats = {}
        
        # Speed-Statistiken
        if speeds:
            if len(speeds) >= 3:
                quantile_data = quantiles(speeds, n=100, method='inclusive')
                stats['speed'] = {
                    'avg': round(mean(speeds), 2),
                    'median': round(median(speeds), 2),
                    'p95': round(quantile_data[94], 2),
                    'p99': round(quantile_data[98], 2),
                    'min': round(min(speeds), 2),
                    'max': round(max(speeds), 2)
                }
        
        # TTFT-Statistiken
        if ttfts:
            if len(ttfts) >= 3:
                quantile_data = quantiles(ttfts, n=100, method='inclusive')
                stats['ttft'] = {
                    'avg': round(mean(ttfts), 3),
                    'median': round(median(ttfts), 3),
                    'p95': round(quantile_data[94], 3),
                    'p99': round(quantile_data[98], 3),
                    'min': round(min(ttfts), 3),
                    'max': round(max(ttfts), 3)
                }
        
        # VRAM-Statistiken
        if vram_values:
            if len(vram_values) >= 3:
                quantile_data = quantiles(vram_values, n=100, method='inclusive')
                stats['vram'] = {
                    'avg': round(mean(vram_values), 0),
                    'median': round(median(vram_values), 0),
                    'p95': round(quantile_data[94], 0),
                    'p99': round(quantile_data[98], 0),
                    'min': round(min(vram_values), 0),
                    'max': round(max(vram_values), 0)
                }
        
        return stats
    
    def generate_quantization_comparison(self) -> Dict[str, Dict]:
        """Generiert Vergleichstabelle Q4 vs Q5 vs Q6 pro Modell"""
        if not self.results:
            return {}
        
        # Gruppiere Ergebnisse nach Modell und Quantisierung
        model_quants = {}
        for result in self.results:
            if result.model_name not in model_quants:
                model_quants[result.model_name] = {}
            
            # Extrahiere Quantisierungs-Level (z.B. "q4_k_m" -> "q4")
            quant_level = result.quantization.split('_')[0].lower()
            if quant_level not in model_quants[result.model_name]:
                model_quants[result.model_name][quant_level] = result
            else:
                # Nimm die beste Performance (höchste Tokens/s) falls mehrere Quantisierungen
                if result.avg_tokens_per_sec > model_quants[result.model_name][quant_level].avg_tokens_per_sec:
                    model_quants[result.model_name][quant_level] = result
        
        # Erstelle Vergleich-Tabelle
        comparison = {}
        for model, quants in sorted(model_quants.items()):
            comparison[model] = {
                'q4': None,
                'q5': None, 
                'q6': None,
                'q8': None
            }
            for q_level in ['q4', 'q5', 'q6', 'q8']:
                if q_level in quants:
                    r = quants[q_level]
                    comparison[model][q_level] = {
                        'speed': round(r.avg_tokens_per_sec, 2),
                        'efficiency': round(r.tokens_per_sec_per_gb, 2),
                        'vram_mb': r.vram_mb,
                        'ttft': round(r.avg_ttft * 1000, 1)
                    }
        
        return comparison
    
    def load_all_historical_data(self) -> Dict[str, List[Dict]]:
        """Lädt alle historischen Benchmark-Ergebnisse und gruppiert nach Modell+Quantisierung"""
        trends = {}
        results_dir = RESULTS_DIR
        
        # Lade alle JSON-Dateien
        for json_file in sorted(results_dir.glob("benchmark_results_*.json")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    
                for item in data:
                    key = f"{item['model_name']}@{item['quantization']}"
                    if key not in trends:
                        trends[key] = []
                    
                    # Extrahiere Datum aus Timestamp
                    timestamp_str = item.get('timestamp', '2026-01-01 00:00:00')
                    trends[key].append({
                        'timestamp': timestamp_str,
                        'speed': item['avg_tokens_per_sec'],
                        'ttft': item['avg_ttft'],
                        'vram': item.get('vram_mb', 0)
                    })
            except Exception as e:
                logger.debug(f"Fehler beim Laden von {json_file}: {e}")
        
        return trends
    
    def generate_trend_chart(self) -> Optional[str]:
        """Generiert Plotly Line-Chart für Performance-Trends über Zeit"""
        if not PLOTLY_AVAILABLE or not self.previous_results or go is None:
            return None
        
        try:
            trends = self.load_all_historical_data()
            if not trends:
                return None
            
            # Erstelle Line-Chart mit Trends
            fig = go.Figure()
            
            # Farben für verschiedene Modelle
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            color_idx = 0
            
            for key, history in sorted(trends.items()):
                if len(history) < 2:  # Nur Trends mit mindestens 2 Datenpunkten
                    continue
                
                # Sortiere nach Timestamp
                history_sorted = sorted(history, key=lambda x: x['timestamp'])
                
                timestamps = [h['timestamp'] for h in history_sorted]
                speeds = [h['speed'] for h in history_sorted]
                
                model_name = key.split('@')[0].split('/')[-1][:15]
                quantization = key.split('@')[1][:6]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=speeds,
                    mode='lines+markers',
                    name=f"{model_name} ({quantization})",
                    line=dict(color=colors[color_idx % len(colors)]),
                    marker=dict(size=6)
                ))
                
                color_idx += 1
            
            fig.update_layout(
                title="Performance-Trends über Zeit",
                xaxis_title="Datum",
                yaxis_title="Tokens/s",
                hovermode='x unified',
                height=600,
                template='plotly_white'
            )
            
            # Gebe Plotly JSON zurück
            return json.dumps({
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            })
        
        except Exception as e:
            logger.debug(f"Fehler beim Erstellen von Trend-Chart: {e}")
            return None
    
    def export_results(self):
        """Exportiert Ergebnisse als JSON, CSV, PDF und HTML"""
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
        
        # PDF Export
        self._export_pdf(timestamp)
        
        # HTML Export (optional)
        if PLOTLY_AVAILABLE:
            self._export_html(timestamp)
    
    def _export_pdf(self, timestamp: str):
        """Exportiert Benchmark-Ergebnisse als PDF-Report"""
        try:
            pdf_file = RESULTS_DIR / f"benchmark_results_{timestamp}.pdf"
            
            # Erstelle PDF-Dokument mit Landscape-Format
            doc = SimpleDocTemplate(
                str(pdf_file),
                pagesize=landscape(A4),
                rightMargin=0.5*inch,
                leftMargin=0.5*inch,
                topMargin=0.5*inch,
                bottomMargin=0.5*inch,
                title="LM Studio Benchmark Results"
            )
            
            # Container für PDF-Elemente
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f4788'),
                spaceAfter=20,
                alignment=1  # Center
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#2d5aa8'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Titel
            elements.append(Paragraph("LM Studio Model Benchmark Report", title_style))
            elements.append(Spacer(1, 12))
            
            # Generiert am
            timestamp_display = time.strftime('%d.%m.%Y %H:%M:%S')
            elements.append(Paragraph(f"<font size=10>Generiert: {timestamp_display}</font>", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Zusammenfassung
            elements.append(Paragraph("Benchmark Summary", heading_style))
            
            # Berechne Statistiken
            vision_count = sum(1 for r in self.results if r.has_vision)
            tools_count = sum(1 for r in self.results if r.has_tools)
            avg_size_gb = sum(r.model_size_gb for r in self.results) / len(self.results) if self.results else 0
            avg_tokens_per_sec = sum(r.avg_tokens_per_sec for r in self.results) / len(self.results) if self.results else 0
            
            summary_data = [
                ['Metrik', 'Wert'],
                ['Anzahl Modelle getestet', str(len(self.results))],
                ['Messungen pro Modell', str(self.num_measurement_runs)],
                ['Context Length', f"{self.context_length} Tokens"],
                ['Standard-Prompt', self.prompt[:50] + '...' if len(self.prompt) > 50 else self.prompt],
                ['Vision-Modelle', f"{vision_count} ({vision_count*100//len(self.results) if self.results else 0}%)"],
                ['Tool-fähige Modelle', f"{tools_count} ({tools_count*100//len(self.results) if self.results else 0}%)"],
                ['Ø Modellgröße', f"{avg_size_gb:.2f} GB"],
                ['Ø Geschwindigkeit', f"{avg_tokens_per_sec:.2f} tokens/s"],
            ]
            summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')])
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Ergebnisse-Tabelle
            elements.append(Paragraph("Detaillierte Ergebnisse", heading_style))
            
            # Sortiere Ergebnisse nach Ranking-Kriterium
            sorted_results = self.sort_results(self.rank_by)
            
            # Zeige Ranking-Kriterium
            rank_labels = {
                'speed': 'Geschwindigkeit (Tokens/s)',
                'efficiency': 'Effizienz (Tokens/s pro GB)',
                'ttft': 'Time to First Token (ms)',
                'vram': 'VRAM-Nutzung (MB)'
            }
            elements.append(Paragraph(f"<font size=9>Sortiert nach: <b>{rank_labels.get(self.rank_by, 'Speed')}</b></font>", styles['Normal']))
            elements.append(Spacer(1, 10))
            
            # Erstelle Tabellen-Daten
            table_data = [['Modell', 'Param', 'Arch', 'Size(GB)', 'Vision', 'Tools', 'Quant.', 'GPU', 'Tokens/s', 'Δ%', 'TTFT (ms)', 'Gen.Zeit (s)']]
            for result in sorted_results:
                vision_icon = '👁' if result.has_vision else ''
                tools_icon = '🔧' if result.has_tools else ''
                delta_str = f"{result.speed_delta_pct:+.1f}%" if result.speed_delta_pct is not None else "-"
                table_data.append([
                    result.model_name[:15],  # Kürzen bei Bedarf
                    result.params_size[:4],
                    result.architecture[:7],
                    f"{result.model_size_gb:.2f}",
                    vision_icon,
                    tools_icon,
                    result.quantization[:6],
                    result.gpu_type[:5],
                    f"{result.avg_tokens_per_sec:.2f}",
                    delta_str,
                    f"{result.avg_ttft*1000:.1f}" if result.avg_ttft else "N/A",
                    f"{result.avg_gen_time:.2f}",
                ])
            
            # Formatiere Tabelle (Landscape mit mehr Spalten)
            results_table = Table(table_data, colWidths=[1.2*inch, 0.55*inch, 0.6*inch, 0.65*inch, 0.45*inch, 0.45*inch, 0.6*inch, 0.5*inch, 0.65*inch, 0.45*inch, 0.65*inch, 0.7*inch])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),

                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(results_table)
            elements.append(Spacer(1, 20))
            
            # Best-of-Quantization Analyse
            elements.append(Paragraph("Best-of-Quantization Analyse", heading_style))
            best_quants = self._analyze_best_quantizations()
            
            quant_data = [['Modell', 'Best Speed', 'Best Effizienz', 'Best TTFT']]
            for model_name, analysis in sorted(best_quants.items()):
                speed_q = analysis['best_speed'].quantization if analysis['best_speed'] else '-'
                efficiency_q = analysis['best_efficiency'].quantization if analysis['best_efficiency'] else '-'
                ttft_q = analysis['best_ttft'].quantization if analysis['best_ttft'] else '-'
                quant_data.append([
                    model_name[:20],
                    speed_q[:8],
                    efficiency_q[:8],
                    ttft_q[:8]
                ])
            
            quant_table = Table(quant_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            quant_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))
            elements.append(quant_table)
            elements.append(Spacer(1, 20))
            
            # Quantisierungs-Vergleich (Q4 vs Q5 vs Q6)
            comp_data = self.generate_quantization_comparison()
            if comp_data:
                elements.append(Paragraph("Quantisierungs-Vergleich (Q4 vs Q5 vs Q6)", heading_style))
                
                # Erstelle Vergleichs-Tabelle
                comp_table_data = [['Modell', 'Q4 (t/s)', 'Q5 (t/s)', 'Q6 (t/s)', 'Q8 (t/s)']]
                for model_name, q_variants in sorted(comp_data.items()):
                    row = [model_name[:15]]
                    for q_level in ['q4', 'q5', 'q6', 'q8']:
                        if q_variants.get(q_level):
                            row.append(f"{q_variants[q_level]['speed']:.2f}")
                        else:
                            row.append('-')
                    comp_table_data.append(row)
                
                comp_table = Table(comp_table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
                comp_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d7aa8')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ]))
                elements.append(comp_table)
                elements.append(Spacer(1, 20))
            
            # Performance-Statistiken
            elements.append(Paragraph("Performance-Statistiken", heading_style))
            max_tps_result = max(self.results, key=lambda x: x.avg_tokens_per_sec)
            min_tps_result = min(self.results, key=lambda x: x.avg_tokens_per_sec)
            avg_tps = sum(r.avg_tokens_per_sec for r in self.results) / len(self.results)
            
            # Percentile berechnen
            percentile_stats = self.calculate_percentile_stats()
            
            stats_data = [
                ['Statistik', 'Wert'],
                ['Schnellstes Modell', f"{max_tps_result.model_name} ({max_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Langsamstes Modell', f"{min_tps_result.model_name} ({min_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Durchschnitt Tokens/s', f"{avg_tps:.2f}"],
            ]
            
            # Percentile-Zeilen hinzufügen
            if 'speed' in percentile_stats:
                speed_p = percentile_stats['speed']
                stats_data.extend([
                    ['', ''],  # Leerzeile
                    ['Tokens/s - Median', f"{speed_p.get('median', '-'):.2f}"],
                    ['Tokens/s - P95', f"{speed_p.get('p95', '-'):.2f}"],
                    ['Tokens/s - P99', f"{speed_p.get('p99', '-'):.2f}"],
                ])
            
            stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2d5aa8')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            elements.append(stats_table)
            
            # Erstelle PDF
            doc.build(elements)
            logger.info(f"PDF-Ergebnisse gespeichert: {pdf_file}")
        
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der PDF: {e}")
    
    def _export_html(self, timestamp: str):
        """Exportiert Benchmark-Ergebnisse als interaktiven HTML-Report mit Plotly Charts"""
        if not PLOTLY_AVAILABLE or go is None:
            logger.warning("Plotly nicht verfügbar, überspringe HTML-Export")
            return
        
        try:
            html_file = RESULTS_DIR / f"benchmark_results_{timestamp}.html"
            
            # Sortiere Ergebnisse nach Ranking-Kriterium (aber für HTML Charts immer nach Speed)
            sorted_results = self.sort_results('speed')  # HTML Charts zeigen immer Top 10 nach Speed
            
            # Bar-Chart: Top 10 schnellste Modelle
            top_10 = sorted_results[:10]
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=[f"{r.model_name[:20]}\n{r.quantization}" for r in top_10],
                    y=[r.avg_tokens_per_sec for r in top_10],
                    text=[f"{r.avg_tokens_per_sec:.2f}" for r in top_10],
                    textposition='auto',
                    marker_color='#2d5aa8'
                )
            ])
            fig_bar.update_layout(
                title="Top 10 schnellste Modelle",
                xaxis_title="Modell + Quantisierung",
                yaxis_title="Tokens/s",
                hovermode='x unified',
                height=500
            )
            
            # Scatter-Plot: Modellgröße vs Speed
            fig_scatter = go.Figure(data=[
                go.Scatter(
                    x=[r.model_size_gb for r in self.results],
                    y=[r.avg_tokens_per_sec for r in self.results],
                    mode='markers',
                    text=[f"{r.model_name}<br>{r.quantization}<br>{r.avg_tokens_per_sec:.2f} t/s" for r in self.results],
                    marker=dict(
                        size=[r.avg_tokens_per_sec / 2 for r in self.results],
                        color=[r.tokens_per_sec_per_gb for r in self.results],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Effizienz<br>(t/s pro GB)")
                    ),
                    hovertemplate='<b>%{text}</b><extra></extra>'
                )
            ])
            fig_scatter.update_layout(
                title="Modellgröße vs Performance (Größe der Bubble = Speed)",
                xaxis_title="Modellgröße (GB)",
                yaxis_title="Tokens/s",
                height=500,
                hovermode='closest'
            )
            
            # Scatter-Plot: Parameter vs Effizienz
            fig_efficiency = go.Figure(data=[
                go.Scatter(
                    x=[r.tokens_per_sec_per_gb for r in self.results],
                    y=[r.tokens_per_sec_per_billion_params for r in self.results],
                    mode='markers',
                    text=[f"{r.model_name} ({r.quantization})" for r in self.results],
                    marker=dict(
                        size=8,
                        color=[r.avg_tokens_per_sec for r in self.results],
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="Speed<br>(tokens/s)")
                    ),
                    hovertemplate='<b>%{text}</b><br>Per GB: %{x:.2f}<br>Per Billion Params: %{y:.2f}<extra></extra>'
                )
            ])
            fig_efficiency.update_layout(
                title="Effizienz-Analyse: Tokens/s pro GB vs Tokens/s pro Milliarde Parameter",
                xaxis_title="Tokens/s pro GB",
                yaxis_title="Tokens/s pro Milliarde Parameter",
                height=500
            )
            
            # Summary-Tabelle
            summary_stats = {
                'Anzahl Modelle': len(self.results),
                'Schnellstes': f"{sorted_results[0].model_name} ({sorted_results[0].avg_tokens_per_sec:.2f} t/s)",
                'Langsamster': f"{sorted_results[-1].model_name} ({sorted_results[-1].avg_tokens_per_sec:.2f} t/s)",
                'Ø Geschwindigkeit': f"{sum(r.avg_tokens_per_sec for r in self.results) / len(self.results):.2f} t/s",
                'Vision-Modelle': f"{sum(1 for r in self.results if r.has_vision)}",
                'Tool-fähige Modelle': f"{sum(1 for r in self.results if r.has_tools)}",
                'Ø Modellgröße': f"{sum(r.model_size_gb for r in self.results) / len(self.results):.2f} GB",
            }
            
            # Erstelle HTML mit allen Charts
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"""
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LM Studio Benchmark Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1f4788;
            text-align: center;
            border-bottom: 3px solid #2d5aa8;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2d5aa8;
            margin-top: 30px;
            border-left: 4px solid #2d5aa8;
            padding-left: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-label {{
            font-size: 12px;
            text-transform: uppercase;
            opacity: 0.9;
            margin-bottom: 5px;
        }}
        .summary-value {{
            font-size: 18px;
            font-weight: bold;
        }}
        .chart {{
            margin: 30px 0;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            background-color: #fafafa;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 30px;
            border-top: 1px solid #e0e0e0;
            padding-top: 15px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 LM Studio Model Benchmark Report</h1>
        
        <div class="summary">
""")
                
                # Summary-Boxen
                colors_list = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']
                for i, (label, value) in enumerate(summary_stats.items()):
                    color = colors_list[i % len(colors_list)]
                    f.write(f"""
            <div class="summary-box" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
                <div class="summary-label">{label}</div>
                <div class="summary-value">{value}</div>
            </div>
""")
                
                f.write("""
        </div>
        
        <h2>Performance Rankings</h2>
        <div class="chart" id="bar-chart"></div>
        
        <h2>Modellgröße vs Performance</h2>
        <div class="chart" id="scatter-chart"></div>
        
        <h2>Effizienz-Analyse</h2>
        <div class="chart" id="efficiency-chart"></div>
        """)
                
                # Trend-Chart wenn vorhanden
                trend_json = self.generate_trend_chart()
                if trend_json:
                    f.write("""
        <h2>📈 Performance-Trends über Zeit</h2>
        <div class="chart" id="trend-chart"></div>
        """)
                
                f.write(f"""
        <div class="timestamp">
            Generiert: {time.strftime('%d.%m.%Y %H:%M:%S')}
        </div>
    </div>
    
    <script>
        Plotly.newPlot('bar-chart', {json.dumps(fig_bar.to_dict()['data'])}, {json.dumps(fig_bar.to_dict()['layout'])});
        Plotly.newPlot('scatter-chart', {json.dumps(fig_scatter.to_dict()['data'])}, {json.dumps(fig_scatter.to_dict()['layout'])});
        Plotly.newPlot('efficiency-chart', {json.dumps(fig_efficiency.to_dict()['data'])}, {json.dumps(fig_efficiency.to_dict()['layout'])});
""")
                
                # Trend-Chart Script
                if trend_json:
                    trend_data = json.loads(trend_json)
                    f.write(f"""        Plotly.newPlot('trend-chart', {json.dumps(trend_data['data'])}, {json.dumps(trend_data['layout'])});
""")
                
                f.write("""    </script>
</body>
</html>
""")
            
            logger.info(f"HTML-Ergebnisse gespeichert: {html_file}")
        
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der HTML: {e}")


def main():
    """Hauptfunktion mit CLI-Argumenten"""
    parser = argparse.ArgumentParser(
        description="LM Studio Model Benchmark - Testet alle lokal installierten LLM-Modelle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python benchmark.py                              # Standard: alle Modelle, 3 Messungen
  python benchmark.py --runs 1                     # Schnell: alle Modelle, 1 Messung
  python benchmark.py --limit 3 --runs 1           # Test 3 Modelle mit 1 Messung (~15 Min)
  python benchmark.py --limit 1 --runs 1           # Test 1 Modell mit 1 Messung (~5 Min)
  python benchmark.py --runs 2 --context 4096      # 2 Messungen, 4096 Token Context
  python benchmark.py --limit 5 --runs 2 --context 4096  # Test 5 Modelle, weitere Optionen
        """
    )
    
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=NUM_MEASUREMENT_RUNS,
        help=f'Anzahl der Messungen pro Modell-Quantisierung (Standard: {NUM_MEASUREMENT_RUNS})'
    )
    
    parser.add_argument(
        '--context', '-c',
        type=int,
        default=CONTEXT_LENGTH,
        help=f'Kontextlänge in Tokens (Standard: {CONTEXT_LENGTH})'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default=STANDARD_PROMPT,
        help=f'Standard-Test-Prompt (Standard: "{STANDARD_PROMPT}")'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Maximale Anzahl von Modellen zum Testen (z.B. 3 testet nur die ersten 3 Modelle)'
    )
    
    # Erweiterte Filter-Optionen
    parser.add_argument(
        '--only-vision',
        action='store_true',
        help='Nur Modelle mit Vision-Fähigkeit (Multimodal) testen'
    )
    
    parser.add_argument(
        '--only-tools',
        action='store_true',
        help='Nur Modelle mit Tool-Calling-Support testen'
    )
    
    parser.add_argument(
        '--quants',
        type=str,
        default=None,
        help='Nur bestimmte Quantisierungen testen (z.B. "q4,q5,q6")'
    )
    
    parser.add_argument(
        '--arch',
        type=str,
        default=None,
        help='Nur bestimmte Architekturen testen (z.B. "llama,mistral,gemma")'
    )
    
    parser.add_argument(
        '--params',
        type=str,
        default=None,
        help='Nur bestimmte Parametergrößen testen (z.B. "3B,7B,8B")'
    )
    
    parser.add_argument(
        '--min-context',
        type=int,
        default=None,
        help='Minimale Context-Length in Tokens (z.B. 32000)'
    )
    
    parser.add_argument(
        '--max-size',
        type=float,
        default=None,
        help='Maximale Modellgröße in GB (z.B. 10.0)'
    )
    
    parser.add_argument(
        '--compare-with',
        type=str,
        default=None,
        help='Vergleiche mit früheren Ergebnissen (z.B. "20260104_172200.json" oder "latest")'
    )
    
    parser.add_argument(
        '--rank-by',
        type=str,
        choices=['speed', 'efficiency', 'ttft', 'vram'],
        default='speed',
        help='Sortiere Ergebnisse nach: speed (tokens/s), efficiency (tokens/s pro GB), ttft (Time to First Token), vram (VRAM-Nutzung)'
    )
    
    args = parser.parse_args()
    
    # Validierung
    if args.runs < 1:
        parser.error('--runs muss >= 1 sein')
    if args.context < 256:
        parser.error('--context muss >= 256 sein')
    if len(args.prompt.strip()) == 0:
        parser.error('--prompt darf nicht leer sein')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit muss >= 1 sein')
    
    # Erstelle Filter-Dictionary aus CLI-Argumenten
    filter_args = {
        'only_vision': args.only_vision,
        'only_tools': args.only_tools,
        'quants': args.quants,
        'arch': args.arch,
        'params': args.params,
        'min_context': args.min_context,
        'max_size': args.max_size,
    }
    
    logger.info("=== LM Studio Model Benchmark ===")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Context Length: {args.context} Tokens")
    logger.info(f"Messungen pro Modell: {args.runs} (+ {NUM_WARMUP_RUNS} Warmup)")
    if args.limit:
        logger.info(f"Modell-Limit: Testet max. {args.limit} Modelle")
    
    # Zeige aktive Filter
    active_filters = [k for k, v in filter_args.items() if v]
    if active_filters:
        logger.info(f"Aktive Filter: {', '.join(active_filters)}")
    
    if args.compare_with:
        logger.info(f"Historischer Vergleich: {args.compare_with}")
    
    logger.info(f"Geschätzte Gesamtzeit: ~{int(args.runs * 45 * (args.limit or 9) / 9)} Minuten")
    logger.info("")
    
    benchmark = LMStudioBenchmark(
        num_runs=args.runs,
        context_length=args.context,
        prompt=args.prompt,
        model_limit=args.limit,
        filter_args=filter_args,
        compare_with=args.compare_with,
        rank_by=args.rank_by
    )
    benchmark.run_all_benchmarks()
    
    logger.info("Benchmark abgeschlossen!")



if __name__ == "__main__":
    main()
