# 🚀 Quick Start Guide - LM Studio Benchmark Tool

## Installation

```bash
cd ~/LM-Studio-Bench

# 1) Preview setup (no changes)
./setup.sh --dry-run

# 2) Prepare system + Python environment (recommended)
./setup.sh

# 3) Activate virtual environment
source .venv/bin/activate
```

If you skip `setup.sh`, use this manual fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🌐 Web Dashboard (Recommended)

### Start Web UI

```bash
./run.py --webapp
```

✅ Opens browser automatically at `http://localhost:8080`
✅ Live streaming of benchmark output via WebSocket
✅ Browse all cached results with interactive tables
✅ System info (GPU model detection, LM Studio health, hardware details)
✅ Dark mode by default with 27 theme options
✅ All CLI parameters available as web form with tooltips
✅ Advanced filtering (quantization, architecture, size, context-length)
✅ Separate logs:
`~/.local/share/lm-studio-bench/logs/webapp_*.log` and
`~/.local/share/lm-studio-bench/logs/benchmark_*.log`
✅ Linux tray control with dynamic status icon and quick actions

**Dashboard Features:**

- **Start Benchmark**: Configure and run benchmarks from web interface
  - Filter by quantization, architecture, parameter size
  - Rank results by speed, efficiency, TTFT, or VRAM
  - Set hardware limits (max GPU temp, max power draw)
  - Tooltip help for all options
- **System Info**: OS, Kernel, CPU, GPU (with detailed model names)
- **LM Studio Health**: Live healthcheck status (HTTP API + CLI fallback)
- **Live Output**: Real-time streaming with colored logs and progress
- **Results Browser**: Filter and sort all cached benchmark results
- **Export**: Download JSON/CSV/PDF/HTML reports
- **Network Access**: Access from other devices on same network

### Linux Tray Control

When GTK/AppIndicator dependencies are installed, a tray controller starts
with the web app.

- **Dynamic status icon**:
  - Gray: idle
  - Green: running
  - Yellow: paused
  - Red: API unreachable/error
- **Smart controls**:
  - Start enabled in idle/error states
  - Pause/Stop enabled only in running/paused states
- **Auto refresh**: status and controls refresh every 3 seconds
- **Quit behavior**: tray `Quit` triggers graceful full shutdown

### Network Access

```bash
# Access dashboard from other devices
http://your-ip:8080

# Example:
http://192.168.1.100:8080
```

## 💻 Command Line (CLI)

### Simple Benchmark (All Models)

```bash
./run.py
```

✅ Tests all installed models with 3 runs each (~1-2 hours)
✅ Automatically saves results to `~/.local/share/lm-studio-bench/results/`
✅ Clean output with emoji icons and formatted model lists
✅ Detailed logs saved to
`~/.local/share/lm-studio-bench/logs/benchmark_YYYYMMDD_HHMMSS.log`

#### Monitor Logs in Real-Time

```bash
# Watch benchmark execution
tail -f ~/.local/share/lm-studio-bench/logs/benchmark_*.log

# Watch web dashboard
tail -f ~/.local/share/lm-studio-bench/logs/webapp_*.log

# Search for errors
grep ERROR ~/.local/share/lm-studio-bench/logs/benchmark_*.log
```

### Quick Test (3 NEW Models)

```bash
./run.py --limit 3 --runs 1
```

✅ Fast test with 3 NEW untested models (~5-10 minutes)
✅ Already tested models automatically loaded from cache
✅ Limit applies ONLY to new models, all cached models included

### Development Mode (Fastest)

```bash
./run.py --dev-mode
```

✅ Automatically selects smallest model
✅ Single run for quick validation (~30 seconds)
✅ Perfect for testing changes

### Test Single Model

```bash
./run.py --limit 1 --runs 1
```

✅ Single model benchmark (~1-2 minutes)

## Advanced Features

### 1️⃣ Hardware Profiling (6 Live Charts)

**Enable Complete Hardware Monitoring:**

```bash
./run.py --enable-profiling --runs 1 --limit 3
```

**Monitored Metrics:**

- 🌡️ GPU Temperature (°C)
- ⚡ GPU Power (W)
- 💾 GPU VRAM (GB)
- 🧠 GPU GTT (GB) - AMD only
- 🖥️ System CPU usage (%)
- 💾 System RAM usage (GB)

✅ All metrics are displayed live in the WebApp
✅ 6 interactive Plotly.js charts with Min/Max/Avg stats
✅ Moving average for stable RAM curves
✅ Each metric is measured every second

**With Safety Limits:**

```bash
./run.py --enable-profiling --max-temp 85 --max-power 350
```

✅ Interrupts benchmark when limits are exceeded

### 2️⃣ AMD GTT Support (Shared System RAM)

**Enable GTT (Default):**

```bash
./run.py --limit 3
```

✅ Automatically uses VRAM + GTT (e.g. 2GB VRAM + 46GB GTT = 48GB)
✅ Enables larger models on AMD APUs/iGPUs
✅ Shown in logs: "💾 Memory: 0.4GB VRAM + 44.7GB GTT = 45.1GB total"

**Disable GTT (VRAM-only):**

```bash
./run.py --disable-gtt --limit 3
```

✅ Only uses dedicated VRAM
✅ More conservative offload levels
✅ Useful for benchmarking VRAM-only performance

### 3️⃣ Filtering Models

**By Quantization:**

```bash
./run.py --quants q4,q5 --limit 5
```

**By Architecture:**

```bash
./run.py --arch llama,mistral --limit 5
```

**By Parameter Size:**

```bash
./run.py --params 7B,8B --limit 5
```

**By Context Length:**

```bash
./run.py --min-context 32000 --limit 3
```

**By Model Size:**

```bash
./run.py --max-size 10 --limit 5
```

**Vision Models Only:**

```bash
./run.py --only-vision --runs 1
```

**Regex-based Filtering (Include):**

```bash
# Only Qwen or Phi models
./run.py --include-models "qwen|phi" --runs 1

# Only Llama 7B models
./run.py --include-models "llama.*7b" --runs 1

# Only Q4 quantizations
./run.py --include-models ".*q4.*" --runs 1
```

**Regex-based Filtering (Exclude):**

```bash
# Exclude uncensored models
./run.py --exclude-models "uncensored" --runs 1

# Exclude Q2 and Q3 quantizations
./run.py --exclude-models "q2|q3" --runs 1

# Exclude all vision models
./run.py --exclude-models ".*vision.*" --runs 1
```

**Combined Filters (AND logic):**

```bash
# Include llama, exclude q2, only tools
./run.py --include-models "llama" --exclude-models "q2" --only-tools --runs 1

# Vision models, 7B params, max 12GB
./run.py --only-vision --params 7B --max-size 12 --runs 1
```

### 3️⃣ Ranking & Sorting

**Sort by Efficiency (Default: Speed):**

```bash
./run.py --limit 5 --rank-by efficiency
```

**Sort by TTFT (Lower = Better):**

```bash
./run.py --limit 5 --rank-by ttft
```

**Sort by VRAM Usage (Lower = Better):**

```bash
./run.py --limit 5 --rank-by vram
```

### 4️⃣ Cache Management

**View Cached Results:**

```bash
./run.py --list-cache
```

✅ Shows all cached models with performance metrics

**Force Retest (Ignore Cache):**

```bash
./run.py --retest --limit 3
```

✅ Re-runs benchmarks even if cached

**Regenerate Reports from Database:**

```bash
./run.py --export-only
```

✅ Generates JSON/CSV/PDF/HTML from cached results in <1s
✅ No benchmarking - instant report generation
✅ Supports all filters (--params, --quants, --arch, etc.)

**Examples:**

```bash
# All cached models
./run.py --export-only

# Only 7B models from cache
./run.py --export-only --params 7B

# Q4 quantizations with historical comparison
./run.py --export-only --quants q4 --compare-with latest
```

✅ Retests models even if cached

**Export Cache as JSON:**

```bash
./run.py --export-cache my_backup.json
```

✅ Exports entire cache database

**Cache Behavior:**

- First run: Tests all models (~2 hours for 20 models)
- Second run: Loads from cache (~1 second!)
- Automatic invalidation on parameter changes (prompt, context, temperature)
- Shows "X of Y models cached" before starting

### 5️⃣ Historical Comparison & Trends

**Compare with Latest Benchmark:**

```bash
./run.py --limit 3 --runs 1 --compare-with latest
```

📊 Shows performance delta (%) vs previous run

**Compare with Specific Benchmark:**

```bash
./run.py --limit 3 --runs 1 --compare-with benchmark_results_20260104_170000.json
```

### 6️⃣ Custom Configuration

**Adjust Number of Runs:**

```bash
./run.py --runs 5 --limit 2
```

**Custom Context Length:**

```bash
./run.py --context 4096 --limit 2 --runs 1
```

**Custom Prompt:**

```bash
./run.py -P "Your custom prompt here" --limit 2 --runs 1
```

### 7️⃣ Presets (Fast Setup)

**Show available presets:**

```bash
./run.py --list-presets
```

**Load a built-in preset:**

```bash
# Default presets (readonly)
./run.py --preset default_classic              # Classic benchmark (default)
./run.py --preset default_compatability_test   # Capability-driven test

# Other presets
./run.py --preset quick_test
./run.py --preset high_quality
./run.py --preset resource_limited
```

**Load preset and override values:**

```bash
./run.py --preset quick_test --runs 2 --context 2048
./run.py --preset default_classic --runs 5 --context 4096
```

**Backwards Compatibility:**

```bash
./run.py --preset default      # Automatically loads default_classic
```

Notes:

- Default presets include explicit values for all benchmark form fields, so
  preset comparisons do not show `null` values for missing keys.
- `default_classic` is optimized for full model benchmarking (3 runs)
- `default_compatability_test` is optimized for focused capability testing (1 run)
- Legacy keys in imported/user presets are normalized automatically
  (`context_length`/`top_k`/`top_p`/`min_p` -> current key names).

## 📊 Output Formats

Each benchmark generates 4 files:

### JSON Format

```json
{
  "model_name": "qwen/qwen3-8b",
  "quantization": "q4_k_m",
  "avg_tokens_per_sec": 8.15,
  "tokens_per_sec_per_gb": 1.74,
  "speed_delta_pct": -0.2,
  ...
}
```

✅ Structured data for analysis

### CSV Format

```csv
model_name,quantization,avg_tokens_per_sec,tokens_per_sec_per_gb,speed_delta_pct
qwen/qwen3-8b,q4_k_m,8.15,1.74,-0.2
```

✅ Excel/Sheets compatible

### PDF Report

- Model rankings (sortable)
- Best-of-Quantization analysis
- Quantization comparison tables (Q4 vs Q5 vs Q6)
- Performance statistics & percentiles
- Delta display (Δ% column)

### HTML Report (Interactive Plotly)

- Bar chart: Top 10 models
- Scatter plot: Size vs Performance
- Scatter plot: Efficiency analysis
- **NEW**: Trend chart showing performance over time
- Summary statistics with gradient backgrounds

## 📈 Feature Showcase

### Example: Complete Analysis

```bash
./run.py \
  --quants q4,q5,q6 \
  --limit 5 \
  --runs 1 \
  --rank-by efficiency \
  --compare-with latest
```

Output:

- ✅ Filters to 5 models with 3 quantizations each
- ✅ Ranks by efficiency (Tokens/s per GB)
- ✅ Shows delta vs previous benchmark
- ✅ Generates all 4 export formats
- ✅ Includes percentile statistics (P50, P95, P99)
- ✅ Shows quantization comparison
- ✅ Displays performance trends if history available

## 🎯 Key Metrics

| Metric | Description | Unit |
| ------ | ----------- | ---- |
| Speed | Throughput | tokens/s |
| Efficiency | Speed per GB model size | tokens/s/GB |
| TTFT | Time to First Token | ms |
| Delta | Change vs previous | % |
| VRAM | Memory used | MB |

## 📁 File Structure

```text
results/
├── benchmark_results_20260104_170000.json
├── benchmark_results_20260104_170000.csv
├── benchmark_results_20260104_170000.pdf
└── benchmark_results_20260104_170000.html
```

## 🐛 Troubleshooting

### No models found

- Ensure LM Studio is installed and running
- Check `lms ls --json` output

### Server not responding

- Start LM Studio server manually
- Check `~/.lmstudio/server-logs/`

### Permission denied on results/

```bash
mkdir -p results/
chmod 755 results/
```

## 🔗 Related Files

- `FEATURES.md` - Complete feature list
- `PLAN.md` - Implementation roadmap
- `requirements.txt` - Python dependencies
- `errors.log` - Debug information

---

**Version:** 1.0 (Phases 1-4 Complete) | **Updated:** 2026-01-04
