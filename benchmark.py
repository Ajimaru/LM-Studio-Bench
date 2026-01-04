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
import psutil
from tqdm import tqdm
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape


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


class LMStudioBenchmark:
    """Haupt-Benchmark-Klasse"""
    
    def __init__(self, num_runs: int = 3, context_length: int = 2048, prompt: str = "Erkläre maschinelles Lernen in 3 Sätzen", model_limit: Optional[int] = None):
        self.gpu_monitor = GPUMonitor()
        self.results: List[BenchmarkResult] = []
        self.num_measurement_runs = num_runs
        self.context_length = context_length
        self.prompt = prompt
        self.model_limit = model_limit
        
        # Erstelle Results-Verzeichnis
        RESULTS_DIR.mkdir(exist_ok=True)
    
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
        
        return BenchmarkResult(
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
            model_size_gb=metadata.get('model_size_gb', 0.0),
            has_vision=metadata.get('has_vision', False),
            has_tools=metadata.get('has_tools', False)
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
    
    def export_results(self):
        """Exportiert Ergebnisse als JSON, CSV und PDF"""
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
            summary_data = [
                ['Metrik', 'Wert'],
                ['Anzahl Modelle getestet', str(len(self.results))],
                ['Messungen pro Modell', str(self.num_measurement_runs)],
                ['Context Length', f"{self.context_length} Tokens"],
                ['Standard-Prompt', self.prompt[:50] + '...' if len(self.prompt) > 50 else self.prompt],
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
            
            # Sortiere Ergebnisse nach Tokens/s (absteigend)
            sorted_results = sorted(
                self.results,
                key=lambda x: x.avg_tokens_per_sec,
                reverse=True
            )
            
            # Erstelle Tabellen-Daten
            table_data = [['Modell', 'Param', 'Arch', 'Size(GB)', 'Vision', 'Tools', 'Quant.', 'GPU', 'Tokens/s', 'TTFT (ms)', 'Gen.Zeit (s)']]
            for result in sorted_results:
                vision_icon = '👁' if result.has_vision else ''
                tools_icon = '🔧' if result.has_tools else ''
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
                    f"{result.avg_ttft*1000:.1f}" if result.avg_ttft else "N/A",
                    f"{result.avg_gen_time:.2f}",
                ])
            
            # Formatiere Tabelle (Landscape mit mehr Spalten)
            results_table = Table(table_data, colWidths=[1.2*inch, 0.55*inch, 0.6*inch, 0.65*inch, 0.45*inch, 0.45*inch, 0.6*inch, 0.5*inch, 0.65*inch, 0.65*inch, 0.7*inch])
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
            # Performance-Statistiken
            elements.append(Paragraph("Performance-Statistiken", heading_style))
            max_tps_result = max(self.results, key=lambda x: x.avg_tokens_per_sec)
            min_tps_result = min(self.results, key=lambda x: x.avg_tokens_per_sec)
            avg_tps = sum(r.avg_tokens_per_sec for r in self.results) / len(self.results)
            
            stats_data = [
                ['Statistik', 'Wert'],
                ['Schnellstes Modell', f"{max_tps_result.model_name} ({max_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Langsamstes Modell', f"{min_tps_result.model_name} ({min_tps_result.avg_tokens_per_sec:.2f} tokens/s)"],
                ['Durchschnitt Tokens/s', f"{avg_tps:.2f}"],
            ]
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
    
    logger.info("=== LM Studio Model Benchmark ===")
    logger.info(f"Prompt: '{args.prompt}'")
    logger.info(f"Context Length: {args.context} Tokens")
    logger.info(f"Messungen pro Modell: {args.runs} (+ {NUM_WARMUP_RUNS} Warmup)")
    if args.limit:
        logger.info(f"Modell-Limit: Testet max. {args.limit} Modelle")
    logger.info(f"Geschätzte Gesamtzeit: ~{int(args.runs * 45 * (args.limit or 9) / 9)} Minuten")
    logger.info("")
    
    benchmark = LMStudioBenchmark(
        num_runs=args.runs,
        context_length=args.context,
        prompt=args.prompt,
        model_limit=args.limit
    )
    benchmark.run_all_benchmarks()
    
    logger.info("Benchmark abgeschlossen!")


if __name__ == "__main__":
    main()
