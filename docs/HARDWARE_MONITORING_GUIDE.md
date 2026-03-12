# Hardware Monitoring Live Charts - Guide

## ✅ Status: Fully Implemented with GPU Detection

Hardware monitoring is now fully functional with stable live charts for all
metrics and improved GPU model detection.

## 📊 Implemented Metrics

### GPU Detection and Model Info

The system automatically detects all installed GPUs:

1. **NVIDIA GPUs**
   - Detection: `nvidia-smi --query-gpu=name`
   - VRAM: `nvidia-smi --query-gpu=memory.total`
   - Temperature: `nvidia-smi --query-gpu=temperature.gpu`
   - Power: `nvidia-smi --query-gpu=power.draw`

2. **AMD GPUs**
   - rocm-smi detection: `rocm-smi --showproductname`
   - Device ID mapping: `lspci -d 1002:{device_id}`
   - Example: `1002:150e` → "Radeon Graphics (Ryzen 9 7950X3D)"
   - rocm-smi search path: `/usr/bin`, `/usr/local/bin`, `/opt/rocm-*/bin/`
   - VRAM: `rocm-smi --showmeminfo vram`
   - GTT: `rocm-smi --showmeminfo gtt`
   - Temperature: `rocm-smi --showtemp`

3. **iGPU detection**
   - Extract from CPU string: regex `r'Radeon\s+(\d+[A-Za-z]*)'`
   - Shows integrated Radeon graphics separately
   - Prevents redundancy with dedicated GPUs

### GPU Metrics

1. **🌡️ GPU Temperature** (°C) - Red
   - NVIDIA: `nvidia-smi --query-gpu=temperature.gpu`
   - AMD: `rocm-smi --showtemp`
   - Intel: `intel-gpu-top` (if available)

2. **⚡ GPU Power** (W) - Blue
   - NVIDIA: `nvidia-smi --query-gpu=power.draw`
   - AMD: `rocm-smi` (Current Socket Graphics Package Power)
   - Intel: alternative measurement methods

3. **💾 GPU VRAM Usage** (GB) - Green
   - NVIDIA: `nvidia-smi --query-gpu=memory.used`
   - AMD: `rocm-smi --showmeminfo vram` (in bytes)

4. **🧠 GPU GTT Usage** (GB) - Purple
   - AMD only: `rocm-smi --showmeminfo gtt`
   - System RAM that is used as VRAM
   - Example: 2GB VRAM + 46GB GTT = 48GB effective

### System Metrics (with --enable-profiling)

1. **🖥️ CPU Usage** (%) - Orange
   - `psutil.cpu_percent(interval=0.1)`
   - 0-100% range
   - System-wide, not per process

2. **💾 System RAM Usage** (GB) - Cyan
   - `psutil.virtual_memory().used`
   - **Smoothing**: moving average over 3 samples
   - Prevents spikes from cache/buffer fluctuations
   - Very stable curves

## 🔧 Activation

Hardware monitoring is automatically enabled with:

```bash
# WebApp with hardware monitoring
./run.py --webapp

# CLI with hardware monitoring
./run.py --enable-profiling

# Only with specific models
./run.py --limit 2 --enable-profiling
```

## 📝 Logger Output

When `--enable-profiling` is active, the benchmark prints metrics every second:

```text
🌡️ GPU Temp: 45.3°C
⚡ GPU Power: 125.5W
💾 GPU VRAM: 8.2GB
🧠 GPU GTT: 0.0GB
🖥️ CPU: 35.2%
💾 RAM: 18.5GB
```

These outputs are:

- ✅ Saved in
   `~/.local/share/lm-studio-bench/logs/benchmark_YYYYMMDD_HHMMSS.log`
- ✅ Shown in the WebApp terminal
- ✅ Visualized as charts

## 🎯 Data Flow

```text
Backend (benchmark.py)
  ↓
HardwareMonitor._monitor_loop()
  ├─ _get_temperature()
  ├─ _get_power_draw()
  ├─ _get_vram_usage()
  ├─ _get_gtt_usage()
  ├─ _get_cpu_usage()
  └─ _get_ram_usage()
       ↓
logger.info() → stdout + log file
       ↓
WebApp Backend (app.py)
  ├─ _consume_output() Task (blocking readline)
  ├─ parse_hardware_metrics() (Regex patterns)
  └─ hardware_history dict
       ↓
WebSocket
  └─ Sends every 2 seconds (last 60 entries)
       ↓
Frontend (dashboard.html.jinja)
  └─ 6 Plotly.js charts with live updates
```

## 🐛 Fixes and Optimizations

### Fix 1: rocm-smi 7.0.1 Format Change

**Problem**: rocm-smi changed its output format
**Solution**: regex parser extracts the last number from the line

```python
match = re.search(r'[\d.]+\s*$', line.strip())
```

### Fix 2: Logger Routing

**Problem**: hardware data did not appear in log files
**Solution**: `print()` → `logger.info()` for stdout + file

All hardware metrics are logged using Python's standard `logging` module:

```python
logger.info(f"🌡️ GPU Temp: {temp:.1f}°C")
logger.info(f"💾 Memory: {vram_mb:.1f}MB VRAM + {gtt_mb:.1f}MB GTT")
```

This ensures metrics appear in both:

- **stdout** - Real-time display in terminal
- **log files** -
   `~/.local/share/lm-studio-bench/logs/benchmark_YYYYMMDD_HHMMSS.log`
   for permanent record
- **WebApp** - Streamed via WebSocket to dashboard

### Fix 3: WebApp Output Streaming

**Problem**: WebApp showed only 10% of the hardware data
**Solution**: `asyncio.wait_for()` → blocking `readline()` in executor

### Fix 4: RAM Monitoring Spikes

**Problem**: RAM chart jumped between 1.8GB and 28.3GB
**Solution**: moving average over 3 samples → very stable curve

### Fix 5: Runtime Counter Does Not Stop

**Problem**: runtime counter continued after benchmark end
**Solution**: `clearInterval(uptimeInterval)` on completion

### Fix 6: WebApp Initialization Race Conditions

**Problem**: links were not interactive, light mode on startup
**Solution**: 3x DOMContentLoaded events → 1x consolidated event

## 📊 Chart Properties

All charts update every 2 seconds with:

- **Min/Max/Avg statistics** - real-time calculation
- **Last 60 data points** - about 2 minutes of history
- **Responsive design** - adapts to window size
- **Dark mode** - default for all charts
- **Hover tooltips** - show exact values on hover
