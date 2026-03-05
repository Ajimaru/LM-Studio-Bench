---
description: Configuration for Copilot's behavior when interacting with the LM Studio Benchmark project
applyTo: '**'
---

# GitHub Copilot Instructions - LM Studio Benchmark

## Project Context

This is a Python benchmark tool for LM Studio with a modern web dashboard that automatically tests all locally installed LLM models and their quantizations. The goal is to measure and compare token/s speeds.

## Coding Guidelines

### General
- Compatible with Python 3.10+
- Use logging instead of print() for important events

### Project-Specific
- **GPU Handling**: Detailed detection (NVIDIA nvidia-smi, AMD rocm-smi, Intel)
  - NVIDIA: `nvidia-smi --query-gpu=name` + `--query-gpu=memory.total`
  - AMD: `rocm-smi --showproductname` (gfx code) + `lspci` (device ID mapping)
  - iGPU extraction: Radeon models from CPU string (e.g., "Radeon 890M")
- **Healthcheck**: HTTP API (ports 1234/1235) + CLI fallback (`lms status`) + 5s polling
- **System Info**: Linux distro (`distro.name()`), kernel (`platform.release()`), CPU model (`cpuinfo`)
- **VRAM Management**: Progressive GPU offload reduction (1.0 → 0.7 → 0.5 → 0.3)
- **Error Recovery**: Never abort the entire benchmark, only skip individual models
- **Progress Feedback**: Use tqdm for all longer operations
- **Resource Cleanup**: Always unload models after test
- **Web UI**: FastAPI backend + Jinja2 templates + Plotly charts

### LM Studio SDK & Tools
- Use `lmstudio` Python SDK
- Server management via subprocess (`lms` CLI)
- Use `stats` from response for metrics
- Limit context length to 2048 (VRAM optimization)
- Use GPU tools: nvidia-smi, rocm-smi, lspci

### Directory Structure

```text
project-root/
├── run.py              # Wrapper script (entry point)
├── README.md           # Main documentation
├── requirements.txt    # Dependencies
├── .gitignore          # Git exclusions
├── src/
│   ├── benchmark.py    # Main application (~1,900 lines)
│   └── report_template.html.template  # HTML report template
├── web/
│   ├── app.py          # FastAPI backend (~1,400 lines)
│   └── templates/
│       └── dashboard.html.jinja  # Dashboard UI (~2,600 lines)
├── docs/               # Public documentation
│   ├── QUICKSTART.md   # Quickstart
│   ├── HARDWARE_MONITORING_GUIDE.md
│   └── LLM_METADATA_GUIDE.md
├── development/        # Internal notes (in .gitignore)
│   └── FEATURES.md
├── results/            # Benchmark results
│   ├── benchmark_results_*.json
│   ├── benchmark_results_*.csv
│   ├── benchmark_results_*.pdf
│   ├── benchmark_results_*.html
│   └── benchmark_cache.db  # SQLite cache
├── logs/               # Logs with date
│   ├── webapp_YYYYMMDD_HHMMSS.log  # Dashboard logs
│   └── benchmark_YYYYMMDD_HHMMSS.log  # Benchmark logs
└── .vscode/            # VSCode settings
    └── settings.json
```

### Output Format
- **JSON**: Structured, 21 fields (11 metrics + 6 metadata + 2 efficiency + 2 delta)
- **CSV**: Tabular, Excel/Sheets compatible
- **PDF**: Landscape A4 with tables, charts, and analysis
- **HTML**: Interactive Plotly charts (bar, scatter, trend charts)
- Fields: model_name, quantization, gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec, tokens_per_sec_per_gb, speed_delta_pct, etc.

### Test Configuration
- **Prompt**: "Explain machine learning in 3 sentences"
- **Warmup**: 1 run (discard)
- **Measurements**: 3 runs (average)
- **Context**: 2048 tokens
- **VRAM Limit**: 12GB optimized

## Web Dashboard (FastAPI)

### Backend (web/app.py)
- **GPU Detection**: Comprehensive detection with fallback chain
- **System Info**: Linux distro, kernel, CPU model, RAM
- **Healthcheck**: `/api/lmstudio/health` with 5s polling
- **Dashboard Stats**: `/api/dashboard/stats` with system and GPU info
- **Benchmark Control**: Start/Pause/Resume/Stop via REST API
- **WebSocket**: Live streaming of benchmark output

### Frontend (web/templates/dashboard.html.jinja)
- **Responsive Design**: 2-column layout (hardware | benchmark)
- **27 Themes**: Light, Dark, Ocean Blue, Gruvbox, Dracula, Nord, etc.
- **Benchmark Form**: Web form with tooltip explanations for all CLI arguments
- **Filter Options**: Quantization, architecture, parameter size, context length
- **Ranking**: By speed, efficiency, TTFT or VRAM
- **Hardware Limits**: Max GPU temp, max power draw
- **Live Charts**: GPU temp, power, VRAM, GTT, CPU, system RAM (6 interactive Plotly charts)

### Tooltip System
- Question mark icons next to each label
- Hover shows short explanation
- Consistent dark background color (rgba(0,0,0,0.9)) for all themes

## Best Practices
- Subprocess calls with timeout (e.g., `timeout=5`)
- GPU monitoring may fail → graceful degradation
- All paths OS-agnostic (`pathlib.Path`)
- CSV with UTF-8 encoding
- JSON with indent=2 for readability
- Check Plotly availability for HTML/PDF exports
- Error recovery: Never abort entire benchmark, skip individual models
- GPU detection: Always use fallback chains (HTTP → CLI → parse sysfs)
- Healthcheck: Non-blocking, 5s polling, no error if LM Studio offline

## New Dependencies
- `httpx`: HTTP client for healthcheck
- `distro`: Linux distro detection
- `py-cpuinfo`: CPU model extraction
- `fastapi`: Web framework (already present)
- `jinja2`: Template rendering (already present)

## Implementation Plan
- Internal roadmap: see `development/FEATURES.md`

## Troubleshooting
- Check LM Studio installation with `lms --help`
- Use log files in `logs/` for debugging (separate logs for webapp and benchmark)
- Use LMStudio logs in `~/.lmstudio/server-logs/` for deeper error analysis
- For Plotly errors: HTML/PDF fallback to text output
- GPU issues: Check VRAM with `nvidia-smi` (NVIDIA), `rocm-smi` (AMD), `intel_gpu_top` (Intel)
- Healthcheck errors: Check if LM Studio is running on port 1234 or 1235
- Device ID mapping: Use `lspci -d 1002:` for AMD GPU detection
