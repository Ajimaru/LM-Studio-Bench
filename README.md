# LM Studio Model Benchmark

![alt text](assets/logo.svg)

---

Automatic benchmarking tool for all locally installed LM Studio models. Systematically tests different models and quantizations to measure and compare tokens-per-second performance.

## Features

- 🌐 **Web Dashboard**: Modern FastAPI-based web UI with live streaming, dark mode and an interactive results browser
- 🤖 **Automatic Model Discovery**: Finds all locally installed models and quantizations
- 🎮 **GPU Detection**: Detailed GPU detection for NVIDIA, AMD and Intel GPUs
  - NVIDIA: GPU model via `nvidia-smi --query-gpu=name`
  - AMD: GPU series via `lspci` device-ID mapping, `rocm-smi`, or gfx code
  - iGPU extraction from CPU string (e.g. "Radeon 890M")
- 📊 **6 Live Hardware Charts**: GPU temperature, power, VRAM, GTT (AMD) plus system CPU & RAM
- 💾 **VRAM Monitoring**: Measures VRAM usage during benchmarks
- 🧠 **GTT Support (AMD)**: Uses shared system RAM in addition to VRAM (e.g. 2GB VRAM + 46GB GTT = 48GB)
- 🖥️ **System Profiling**: CPU and RAM usage with `--enable-profiling`
- 🌡️ **Hardware Profiling**: Optional monitoring of GPU temperature and power draw (NVIDIA/AMD/Intel)
- 🔄 **Progressive GPU Offload**: Automatically tries different GPU offload levels (1.0 → 0.7 → 0.5 → 0.3)
- 🖥️ **Server Management**: Starts LM Studio server automatically if needed
- ❤️‍🩹 **Live Healthcheck**: Real-time LM Studio status (HTTP API + CLI fallback, 5s polling)
- 📝 **Standardized Tests**: Uses the same prompt for all models
- 📈 **Statistical Evaluation**: Warmup + multiple measurements for accurate results
- 🗄️ **SQLite Cache**: Automatically caches benchmark results (skips already-tested models)
- 🧠 **Smart Limits**: `-l 5` runs 5 NEW models plus all cached results
- ⚡ **Dev Mode**: Picks the smallest model for quick tests during development
- 🧹 **Clean Logging**: Emojis, formatted model lists, filtered third-party debug logs, separate log files for webapp and benchmarks
- 📤 **Export**:
  - 🗂️ JSON, CSV (Excel/Sheets compatible)
  - 📄 PDF (multi-page with best-practice recommendations, vision/tool/architecture pages, optional Plotly charts)
  - 🌐 HTML (interactive Plotly charts, best-practices, vision/tool/architecture tables, dark mode)
- ⚡ **Instant Report Regeneration**: `--export-only` generates reports from cached data in <1s
- 🏷️ **Extensive Metadata**: parameter size, architecture, context length, file size, vision and tool support
- **🎨 27 Themes**: Light, Dark, Ocean Blue, Deep Slate, Mint Green, Speed Red, Neon Purple, Solarized Dark/Light, Gruvbox, Dracula, Nord, Monokai, Paper, Terminal Green, OLED, Forest, Sunset, Cyberpunk, Pastel, Sepia, 80s, 90s, Hacker/Matrix, Hardware
- **📊 Live Hardware Monitoring**: 6 interactive charts (GPU temp, power, VRAM, GTT, CPU, system RAM) with stats
- **📥 Export Buttons**: Quick access to the latest HTML/PDF/JSON/CSV benchmark results
- **🏠 Dashboard Home**:
  - System info (OS, kernel, CPU, GPU with detailed model names)
  - LM Studio healthcheck status
  - Top 5 fastest models
  - Last 10 benchmark runs
- **📋 Advanced Benchmark Configuration in Web UI**:
  - All CLI arguments available
  - Tooltip explanations for all options
  - Filters by quantization, architecture, parameter size, context length
  - Sort by speed, efficiency, TTFT or VRAM
  - Hardware limits (max GPU temp, max power)
  - GTT options (AMD GPUs)

## System Requirements

- **OS**: Linux (primary), macOS
- **Python**: 3.8 or newer
- **GPU**: ~12GB VRAM recommended (NVIDIA/AMD/Intel)
- **Software**: [LM Studio](https://lmstudio.ai/) installed locally with the `lms` CLI available

## Installation

**1. Clone the repository**:

  ```bash
  git clone <repository-url>
  cd local-llm-bench
  ```

**2. Create and activate a virtual environment**:

  ```bash
  # Create virtual environment
  python3 -m venv .venv

  # Activate (Linux/macOS)
  source .venv/bin/activate

  # Activate (Windows)
  .venv\Scripts\activate
  ```

**3. Install Python dependencies**:

  ```bash
  pip install -r requirements.txt
  ```

**4. Check LM Studio CLI**:

  ```bash
  lms --help
  ```

## Usage

### 🌐 Web Dashboard (Recommended)

Start the modern web UI with live streaming and an interactive results browser:

```bash
# Start the web dashboard (opens browser automatically)
./run.py --webapp
```

**Dashboard features:**

- 📊 **Live streaming**: Benchmark output in real time via WebSocket
- 🎨 **27 themes**: Light, Dark, Ocean Blue, Deep Slate, Mint Green, Speed Red, Neon Purple, Solarized, Gruvbox, Dracula, Nord, Monokai, Paper, Terminal, OLED, Forest, Sunset, Cyberpunk, Pastel, Sepia, 80s, 90s, Hacker, Hardware + High Contrast
- 📁 **Results browser**: Browse all cached benchmark results in a sortable table
- 📥 **Export buttons**: Quick access to the latest HTML/PDF/JSON/CSV results
- 💻 **Hardware monitoring**: 6 live charts (GPU temp, power, VRAM, GTT, CPU, RAM) with min/avg/max stats
- 🏠 **Home statistics**:
  - System info (OS, kernel, CPU, GPU with model names)
  - LM Studio healthcheck status
  - Top 5 fastest models
  - Last 10 benchmark runs
- 🔧 **Flexible configuration**:
  - All CLI parameters available as a web form with tooltips
  - Filters by quantization, architecture, parameter size
  - Sort by speed, efficiency, TTFT or VRAM
  - Hardware limits (max GPU temp, max power)
- 📱 **Responsive design**: Works on desktop and tablet
- 🌐 **Network access**: Open the dashboard from other devices (`http://your-ip:8080`)

1. Check/start LM Studio server
2. Discover all installed models
3. Test each model with a standardized prompt
4. Save results into the `results/` directory

### CLI Options

#### Basic parameters

```bash
./run.py --runs 1           # Number of measurements per model
./run.py --context 4096     # Context length in tokens
./run.py --prompt "..."     # Custom prompt
./run.py --limit 5          # Test up to 5 NEW models (+ all cached results)
```

### Hardware profiling

```bash
# Enable GPU monitoring (temperature + power draw)
./run.py --enable-profiling

# With safety limits
./run.py --enable-profiling --max-temp 85 --max-power 350

# AMD GTT (shared system RAM)
./run.py --disable-gtt  # Use VRAM only (default: GTT enabled)
```

### Advanced filters

```bash
# Specific quantizations only
./run.py --quants q4,q5

# Specific architectures only
./run.py --arch llama,mistral

# Specific parameter sizes only
./run.py --params 3B,7B

# Vision models only
./run.py --only-vision

# Tool-capable models only
./run.py --only-tools

# Minimum context length
./run.py --min-context 32000

# Maximum model size (GB)
./run.py --max-size 10.0

# Regex filter: include (only models that match)
./run.py --include-models "qwen|phi"       # Qwen or Phi
./run.py --include-models "llama.*7b"      # Llama 7B models
./run.py --include-models ".*q4.*"         # All Q4 quantizations

# Regex filter: exclude (exclude models)
./run.py --exclude-models "uncensored"     # No uncensored models
./run.py --exclude-models "q2|q3"          # No Q2/Q3 quantizations
./run.py --exclude-models ".*vision.*"     # No vision models

# Combine filters (AND semantics)
./run.py --include-models "llama" --exclude-models "q2" --only-tools
./run.py --only-vision --params 7B --max-size 12
```

### Cache management

```bash
# Use cache (default - skips already-tested models)
./run.py --limit 5

# Ignore cache and retest everything
./run.py --retest --limit 5

# Development mode (smallest model, 1 run)
./run.py --dev-mode

# Show all cached results
./run.py --list-cache

# Export cache as JSON
./run.py --export-cache my_cache.json

# Generate reports from the database (no new tests)
./run.py --export-only                  # All cached models
./run.py --export-only --params 7B      # Only 7B models
./run.py --export-only --quants q4      # Only Q4 quantizations
./run.py --export-only --compare-with latest  # With historical comparison
```

### Default settings

- **Prompt**: "Explain machine learning in 3 sentences"
- **Context length**: 2048 tokens
- **Warmup**: 1 run
- **Measurements**: 3 runs
- **GPU offload**: automatic (1.0 → 0.7 → 0.5 → 0.3)

### Optimized inference parameters

For standardized and reproducible benchmarks the following sampling parameters are used:

| Parameter           | Value | Reason                                                     |
|---------------------|-------|----------------------------------------------------------- |
| **Temperature**     | 0.1   | Low for consistent, near-deterministic outputs             |
| **Top-K Sampling**  | 40    | Sample from top 40 tokens                                  |
| **Top-P Sampling**  | 0.9   | Nucleus sampling with 90% cumulative probability           |
| **Min-P Sampling**  | 0.05  | Minimum probability threshold                              |
| **Repeat Penalty**  | 1.2   | Reduces repetitions (default 1.1)                          |
| **Max Tokens**      | 256   | Bounded output length for faster tests                     |

Diese werden automatisch in `_run_inference()` über das Python SDK angewendet und können in der `OPTIMIZED_INFERENCE_PARAMS` Konstante angepasst werden (siehe [benchmark.py](benchmark.py) Zeile ~47).

## Output

### Log files

The tool uses separate log files for different components:

```text
logs/
├── webapp_20260105_112201.log       # Web dashboard logs (only when --webapp is used)
└── benchmark_20260105_113045.log    # Benchmark run logs
```

- **WebApp logs** (`webapp_*.log`): FastAPI server, WebSocket events, HTTP requests
- **Benchmark logs** (`benchmark_*.log`): model tests, VRAM monitoring, errors
- Logs are only created when the corresponding component is active
- Timestamps use the format `YYYYMMDD_HHMMSS`

### Result files

Benchmark reports are stored in the `results/` directory:

- `benchmark_results_YYYYMMDD_HHMMSS.json` - structured data (for automation)
- `benchmark_results_YYYYMMDD_HHMMSS.csv` - tabular data (Excel/Sheets compatible)
- `benchmark_results_YYYYMMDD_HHMMSS.pdf` - formatted report (share/archive)
- `benchmark_results_YYYYMMDD_HHMMSS.html` - interactive Plotly charts
- `benchmark_cache.db` - SQLite database with all benchmark results (automatic caching)

**Note:** Reports contain only the **newly tested models** from the current run. The interactive results browser in the web dashboard shows all cached results for historical comparisons.

### PDF report

The PDF report is generated in **A4 landscape** and includes:

- **Summary**: benchmark configuration (number of models, context length, prompt)
- **Detailed table**: all metrics including metadata (parameter size, architecture, file size)
- **Visual indicators**: emoji icons for vision capability (👁) and tool support (🔧)
- **Performance stats**: fastest/slowest model, averages

Great for sharing benchmark results or archiving them.

### Example CSV output

```csv
model_name,quantization,gpu_type,gpu_offload,vram_mb,avg_tokens_per_sec,avg_ttft,avg_gen_time,prompt_tokens,completion_tokens,timestamp,params_size,architecture,max_context_length,model_size_gb,has_vision,has_tools,tokens_per_sec_per_gb,tokens_per_sec_per_billion_params
llama-3.2-3b-instruct,q4_k_m,NVIDIA,1.0,2048,51.43,0.111,0.954,10,49,2026-01-04 10:30:45,3B,llama,8192,1.92,False,False,26.79,17.14
qwen2.5-7b-instruct,q5_k_m,NVIDIA,0.7,4512,38.76,0.145,1.287,10,49,2026-01-04 10:35:12,7B,qwen,131072,4.38,False,True,8.85,5.54
```

### Logs

- **Console**: real-time progress with a `tqdm` progress bar and emoji icons
  - 🚀 Start/Launch
  - 🔍 Detection/Discovery
  - 📊 Data/Statistics
  - 💾 Storage/Memory
  - ✅ Success/Completion
  - 🎯 Optimization
  - ⚙️ Configuration
  - ⏱️ Time/Performance
  - Additional icons for specific operations
- **Per-run logs**: `logs/benchmark_YYYYMMDD_HHMMSS.log` - separate log file per run
- **Filtered logs**: third-party libraries (httpx, lmstudio, urllib3, websockets) are limited to WARNING level
- **JSON filtering**: WebSocket debug events are automatically filtered

## Measured metrics

| Metric | Description |
| ------ | ----------- |
| **avg_tokens_per_sec** | Average token generation speed |
| **avg_ttft** | Time to First Token - latency until the first generated token |
| **avg_gen_time** | Total time for generating the response |
| **vram_mb** | VRAM usage during inference (if measurable) |
| **prompt_tokens** | Number of input tokens |
| **completion_tokens** | Number of generated tokens |
| **params_size** | Model parameter size (e.g. "3B", "7B") |
| **architecture** | Model architecture (e.g. "mistral3", "gemma3") |
| **max_context_length** | Maximum context length of the model in tokens |
| **model_size_gb** | Model file size in GB (rounded to 2 decimals) |
| **has_vision** | Vision capability (multimodal: text + images) |
| **has_tools** | Tool-calling support (function/tool use) |
| **tokens_per_sec_per_gb** | Efficiency: tokens/s per GB of model size |
| **tokens_per_sec_per_billion_params** | Efficiency: tokens/s per billion parameters |
| **temp_celsius_min/max/avg** | GPU temperature during the benchmark (°C) - only with `--enable-profiling` |
| **power_watts_min/max/avg** | GPU power draw during the benchmark (W) - only with `--enable-profiling` |

## Troubleshooting

### "lms: command not found"

The LM Studio CLI is not in your PATH. Install or configure LM Studio:

```bash
# Check installation
which lms
```

### "No models found"

Ensure models are downloaded in LM Studio:

```bash
lms ls
```

### "GPU monitoring not available"

GPU tooling is missing. Install the appropriate tools for your GPU:

**NVIDIA**:

```bash
sudo apt install nvidia-utils
```

**AMD**:

```bash
sudo apt install rocm-dkms rocm-smi
```

**Intel**:

```bash
sudo apt install intel-gpu-tools
```

### Model fails to load (VRAM error)

The script will automatically try lower GPU offload levels. With ~12GB VRAM:

- ✅ 3B models with Q5_K_M
- ✅ 7B models with Q4_K_M
- ⚠️ 13B models with Q3_K_M (possible)
- ❌ 32B+ models (not recommended)

## Customization

### Custom Prompts

Change in [benchmark.py](benchmark.py):

```python
STANDARD_PROMPT = "Your custom test prompt"
```

### More/Fewer Runs

```python
NUM_MEASUREMENT_RUNS = 5  # Default: 3
```

### Context Length

```python
CONTEXT_LENGTH = 4096  # Default: 2048
```

## Project Structure

```text
local-llm-bench/
├── benchmark.py              # Main script
├── requirements.txt          # Python dependencies
├── results/                  # Benchmark results
│   ├── benchmark_results_*.json
│   └── benchmark_results_*.csv
├── errors.log                # Error log
├── PLAN.md                   # Implementation plan
├── README.md                 # This file
└── .github/
  └── copilot-instructions.md
```

## Technical Details

### GPU Detection

The tool searches for GPU monitoring tools in:

- Standard PATH
- `/usr/bin`
- `/usr/local/bin`
- `/opt/rocm/bin` (AMD)
- `/usr/lib/xpu` (Intel)

### GPU Offload Strategy

If loading fails, offload is automatically reduced:

1. 🟢 `gpuOffload: 1.0` (100% GPU)
2. 🟡 `gpuOffload: 0.7` (70% GPU)
3. 🟠 `gpuOffload: 0.5` (50% GPU)
4. 🔴 `gpuOffload: 0.3` (30% GPU)
5. ❌ Error → Skip model + log

### Benchmark Procedure

For each model:

1. **Load**: With optimal GPU offload
2. **Warmup**: 1x inference (discard result)
3. **Measurement**: 3x inference
4. **Stats**: Calculate average
5. **Unload**: Remove model from memory
6. **Next model**

### Custom prompts

Change in [benchmark.py](benchmark.py):

```python
STANDARD_PROMPT = "Your custom test prompt"
```

### More/less runs

```python
NUM_MEASUREMENT_RUNS = 5  # Default: 3
```

### Context length

```python
CONTEXT_LENGTH = 4096  # Default: 2048
```

## Project structure

```text
# Oder kurze Syntax
./run.py -w
```

The dashboard is available by default at: <http://localhost:8080>

### Dashboard Features

- 🌐 **Modern Web UI**: Responsive dashboard with dark mode (default)
- ⚡ **Live Streaming**: WebSocket for real-time terminal output
- 🎮 **Benchmark Control**: Start/stop via web interface
- ⚙️ **Parameter Configuration**: All CLI parameters available via GUI
- 📊 **Results Browser**: Browse all cached benchmark results
- 📥 **Export Functions**: Download JSON/CSV/PDF/HTML reports
- 🌐 **Network Access**: Reachable from other devices on your network
- 📝 **Separate Logs**: `logs/webapp_*.log` and `logs/benchmark_*.log`
- 🎨 **Dark Mode**: Modern dark design with toggle option
- 🔌 **REST API**: Full API for automation (`/docs` for OpenAPI)

### REST API Endpoints

- `GET /` - Dashboard UI
- `GET /api/status` - Benchmark status
- `GET /api/output` - Terminal output
- `POST /api/benchmark/start` - Start benchmark
- `POST /api/benchmark/pause` - Pause
- `POST /api/benchmark/resume` - Resume
- `POST /api/benchmark/stop` - Stop
- `WS /ws/benchmark` - WebSocket live streaming

### Dashboard Parameters

All benchmark parameters can be configured via the dashboard:

- Runs (measurements per model)
- Context length
- Model limit
- Custom prompt
- Include/exclude regex patterns
- Vision/tools/retest/dev mode flags

---

## Support

If you encounter problems:

1. Check `logs/` for error logs
2. Make sure LM Studio is running
3. Open an issue with logs and system info

---

**Note**: This tool is intended for development/testing. For production deployments, see the LM Studio documentation.
