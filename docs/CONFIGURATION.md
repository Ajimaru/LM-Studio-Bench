# Configuration Reference

Complete documentation of all CLI arguments and configuration options for the LM Studio Benchmark Tool.

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration Files](#configuration-files)
3. [CLI Arguments](#cli-arguments)
   - [Basic Options](#basic-options)
   - [Filter Options](#filter-options)
   - [Cache Management](#cache-management)
   - [Hardware Profiling](#hardware-profiling)
   - [Inference Parameters](#inference-parameters)
   - [Load Config (Performance Tuning)](#load-config-performance-tuning)
   - [REST API Mode](#rest-api-mode)
4. [Examples](#examples)

---

## Overview

The benchmark tool can be configured in three ways:

1. **Project Defaults**: `config/defaults.json` (in Git)
2. **User Configuration**: `~/.config/lm-studio-bench/defaults.json` (optional overrides)
3. **CLI Arguments**: Override all config values

**Priority**: CLI Arguments > User Config > Project Defaults > Hard-coded Defaults

## Configuration Files

### Project Configuration (`config/defaults.json`)

The project configuration file contains all default settings for the benchmark. This file is shipped with the
project and tracked in Git.

**Location**: `<project_root>/config/defaults.json`

### User Configuration (`~/.config/lm-studio-bench/defaults.json`)

Optional user-specific configuration overrides. Only specify fields you want to customize.

**Location**: `~/.config/lm-studio-bench/defaults.json`

**Example** (minimal user config):

```json
{
  "num_runs": 5,
  "lmstudio": {
    "use_rest_api": true
  }
}
```

This overrides only `num_runs` and `use_rest_api`, all other values come from project defaults.

### Complete Structure

```json
{
  "prompt": "Is the sky blue?",
  "context_length": 2048,
  "num_runs": 3,
  "retest": false,
  "enable_profiling": false,
  "lmstudio": {
    "host": "localhost",
    "ports": [1234, 1235],
    "api_token": null,
    "use_rest_api": true
  },
  "inference": {
    "temperature": 0.1,
    "top_k_sampling": 40,
    "top_p_sampling": 0.9,
    "min_p_sampling": 0.05,
    "repeat_penalty": 1.2,
    "max_tokens": 256
  },
  "load": {
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_threads": -1,
    "flash_attention": true,
    "rope_freq_base": 10000,
    "rope_freq_scale": 1.0,
    "use_mmap": true,
    "use_mlock": false,
    "kv_cache_quant": "f16"
  }
}
```

### Field Descriptions

#### Basic Settings

| Field | Type | Default | Description |
| ------ | ----- | ---------- | -------------- |
| `prompt` | string | `"Is the sky blue?"` | Default test prompt for all benchmarks |
| `context_length` | integer | `2048` | Context length in tokens |
| `num_runs` | integer | `3` | Number of measurements per model/quantization |
| `retest` | boolean | `false` | Ignore cache and benchmark all selected models again |
| `enable_profiling` | boolean | `false` | Enable temperature/power monitoring |

#### LM Studio Server (`lmstudio`)

| Field | Type | Default | Description |
| ------ | ----- | ---------- | -------------- |
| `host` | string | `"localhost"` | LM Studio server hostname |
| `ports` | array | `[1234, 1235]` | Ports for server discovery (tries both) |
| `api_token` | string/null | `null` | API permission token (REST API authentication) |
| `use_rest_api` | boolean | `true` | Use REST API v1 instead of SDK/CLI |

#### Inference Parameters (`inference`)

| Field | Type | Default | Description |
| ------ | ----- | ---------- | -------------- |
| `temperature` | float | `0.1` | Sampling temperature (0.0-2.0, low=deterministic) |
| `top_k_sampling` | integer | `40` | Top-K sampling (limits choice to K most likely tokens) |
| `top_p_sampling` | float | `0.9` | Top-P / Nucleus sampling (cumulative probability) |
| `min_p_sampling` | float | `0.05` | Min-P sampling (minimum probability threshold) |
| `repeat_penalty` | float | `1.2` | Repeat penalty (prevents repetitions, 1.0=off) |
| `max_tokens` | integer | `256` | Maximum output tokens |

#### Load Config (`load`)

| Field | Type | Default | Description |
| ------ | ----- | ---------- | -------------- |
| `n_gpu_layers` | integer | `-1` | GPU layers (-1=auto/all, 0=CPU only, >0=specific) |
| `n_batch` | integer | `512` | Batch size for prompt processing |
| `n_threads` | integer | `-1` | CPU threads (-1=auto/all) |
| `flash_attention` | boolean | `true` | Flash attention (faster computation) |
| `rope_freq_base` | float | `10000` | RoPE frequency base |
| `rope_freq_scale` | float | `1.0` | RoPE frequency scaling |
| `use_mmap` | boolean | `true` | Memory mapping (faster model load) |
| `use_mlock` | boolean | `false` | Memory locking (prevents swapping) |
| `kv_cache_quant` | string | `"f16"` | KV cache quantization (f32/f16/q8_0/q4_0/etc.) |

### Preset Defaults and Compatibility

The tool includes two readonly default presets:

#### `default_classic` - Classic Benchmark Mode

Default preset for standard model benchmarking. Contains explicit values for all benchmark
fields to avoid `null` values in preset comparisons.

- **benchmark_mode**: `classic`
- **preset_mode**: `classic`
- **runs**: 3
- **context**: 2048
- Capability fields (agent_model, agent_capabilities, agent_max_tests): `null`

**Backwards Compatibility**: Loading `--preset default` automatically loads `default_classic`.

#### `default_compatibility_test` - Capability-Driven Test Mode

Default preset for focused capability testing of a single model.

**Alias**: The legacy name `default_compatability_test` is accepted as an alias
for this preset for backward compatibility.
- **benchmark_mode**: `capability`
- **preset_mode**: `capability`
- **runs**: 1
- **context**: 2048
- **agent_model**: `qwen2.5-7b-instruct`
- **agent_capabilities**: `general_text,reasoning`
- **agent_max_tests**: `10`
- No `null` values - all fields have explicit defaults

Compatibility mapping is applied automatically when loading and comparing
presets with legacy keys:

- `context_length` -> `context`
- `num_runs` -> `runs`
- `top_k` -> `top_k_sampling`
- `top_p` -> `top_p_sampling`
- `min_p` -> `min_p_sampling`

---

## CLI Arguments

All CLI arguments override the corresponding values from both config files.

### Basic Options

#### `--runs`, `-r` (integer)

Number of measurements per model/quantization.

```bash
./run.py --runs 1              # Fast: only 1 measurement
./run.py --runs 5              # Accurate: 5 measurements (average)
```

**Default**: `3`

---

#### `--context`, `-c` (integer)

Context length in tokens.

```bash
./run.py --context 4096        # 4K context
./run.py --context 32768       # 32K context
```

**Default**: `2048`

---

#### `--list-presets`

List all available presets (readonly + user presets) and exit.

```bash
./run.py --list-presets
```

---

#### `--preset`, `-p` (string)

Load a preset before parsing all remaining CLI arguments.
If omitted, `default_classic` is used. The legacy alias `default` still
loads `default_classic` automatically.

```bash
./run.py --preset quick_test
./run.py --preset high_quality --runs 3
./run.py --preset default_classic
./run.py --preset default_compatability_test
```

Built-in readonly presets:

- `default_classic`
- `default_compatability_test`
- `default` (alias for `default_classic`)
- `quick_test`
- `high_quality`
- `resource_limited`

Readonly preset names cannot be saved, deleted, or imported as user presets.
This restriction also applies to the legacy alias `default`.

For capability-driven runs across many models, individual model load failures
are logged and skipped so the benchmark can continue with the remaining
models.

---

#### `--prompt`, `-P` (string)

Default test prompt.

```bash
./run.py --prompt "Explain machine learning"
./run.py -P "Explain machine learning"
```

**Default**: `"Is the sky blue?"`

---

#### `--limit`, `-l` (integer)

Maximum number of models to test.

```bash
./run.py --limit 1             # Only 1 model (usually smallest)
./run.py --limit 5             # First 5 models
```

**Default**: `None` (all models)

---

#### `--dev-mode`

Development mode: Automatically tests the smallest model with 1 run.

```bash
./run.py --dev-mode            # Equivalent to --limit 1 --runs 1
```

**Default**: `false`

---

### Filter Options

#### `--only-vision`

Test only models with vision capability (multimodal).

```bash
./run.py --only-vision --runs 2
```

**Default**: `false`

---

#### `--only-tools`

Test only models with tool-calling support.

```bash
./run.py --only-tools --runs 2
```

**Default**: `false`

---

#### `--quants` (string)

Test only specific quantizations (comma-separated).

```bash
./run.py --quants "q4,q5,q6"     # Only Q4/Q5/Q6
./run.py --quants "q8"           # Only Q8
```

**Default**: `None` (all quants)

---

#### `--arch` (string)

Test only specific architectures (comma-separated).

```bash
./run.py --arch "llama,mistral"  # Only Llama and Mistral
./run.py --arch "qwen"           # Only Qwen
```

**Default**: `None` (all architectures)

---

#### `--params` (string)

Test only specific parameter sizes (comma-separated).

```bash
./run.py --params "3B,7B,8B"     # 3B, 7B and 8B models
./run.py --params "1B"           # Only 1B models
```

**Default**: `None` (all sizes)

---

#### `--min-context` (integer)

Minimum context length in tokens.

```bash
./run.py --min-context 32000     # Only models with ≥32K context
```

**Default**: `None` (no minimum)

---

#### `--max-size` (float)

Maximum model size in GB.

```bash
./run.py --max-size 10.0         # Only models ≤10GB
./run.py --max-size 5.0          # Only models ≤5GB
```

**Default**: `None` (no limit)

---

#### `--include-models` (string)

Only test models matching the regex pattern.

```bash
./run.py --include-models "llama.*7b"      # All 7B Llama models
./run.py --include-models "qwen|phi"       # Qwen OR Phi
```

**Default**: `None` (all models)

---

#### `--exclude-models` (string)

Exclude models matching the regex pattern.

```bash
./run.py --exclude-models ".*uncensored.*" # No uncensored models
./run.py --exclude-models "test|exp"       # No test/experimental
```

**Default**: `None` (no exclusions)

---

#### `--compare-with` (string)

Compare with previous results.

```bash
./run.py --compare-with "20260104_172200.json"
./run.py --compare-with "latest"           # Latest result
```

**Default**: `None` (no comparison)

---

#### `--rank-by` (choice)

Sort results by metric.

**Options**: `speed`, `efficiency`, `ttft`, `vram`

```bash
./run.py --rank-by speed         # By tokens/s
./run.py --rank-by efficiency    # By tokens/s per GB VRAM
./run.py --rank-by ttft          # By Time to First Token
./run.py --rank-by vram          # By VRAM usage (low→high)
```

**Default**: `speed`

---

---

### Cache Management

#### `--retest`

Ignore cache and retest all models.

```bash
./run.py --retest                # Overwrites old results
```

**Default**: `false` (uses cache if available)

---

#### `--list-cache`

Display all cached models and exit.

```bash
./run.py --list-cache
```

**Output**: Table with all cache entries

---

#### `--export-cache` (string)

Export cache contents as JSON.

```bash
./run.py --export-cache "cache_export.json"
```

**Exits** the program after export.

---

#### `--export-only`

Generate reports from cache without new tests.

```bash
./run.py --export-only           # Creates JSON/CSV/PDF/HTML
```

**Default**: `false`

---

### Hardware Profiling

#### `--enable-profiling`

Enable hardware profiling (GPU temp & power).

```bash
./run.py --enable-profiling
```

**Default**: `false`

---

#### `--max-temp` (float)

Maximum GPU temperature in °C (warning).

```bash
./run.py --enable-profiling --max-temp 80.0
```

**Default**: `None` (no warning)

---

#### `--max-power` (float)

Maximum GPU power draw in Watts (warning).

```bash
./run.py --enable-profiling --max-power 400.0
```

**Default**: `None` (no warning)

---

#### `--disable-gtt`

Disable GTT (Shared System RAM) for AMD GPUs.

```bash
./run.py --disable-gtt           # Only dedicated VRAM
```

**Default**: `false` (GTT enabled)

**Note**: Only relevant for AMD iGPUs (e.g., Radeon 890M).

---

### Inference Parameters

All override values from config files:

#### `--temperature` (float)

```bash
./run.py --temperature 0.7       # More creative responses
./run.py --temperature 0.0       # Deterministic
```

#### `--top-k`, `--top-k-sampling` (integer)

```bash
./run.py --top-k 50
```

#### `--top-p`, `--top-p-sampling` (float)

```bash
./run.py --top-p 0.95
```

#### `--min-p`, `--min-p-sampling` (float)

```bash
./run.py --min-p 0.05
```

#### `--repeat-penalty` (float)

```bash
./run.py --repeat-penalty 1.3
```

#### `--max-tokens` (integer)

```bash
./run.py --max-tokens 512
```

---

### Load Config (Performance Tuning)

All override values from config files:

#### `--n-gpu-layers` (integer)

```bash
./run.py --n-gpu-layers -1       # All layers on GPU (default)
./run.py --n-gpu-layers 0        # CPU only
./run.py --n-gpu-layers 20       # First 20 layers on GPU
```

#### `--n-batch` (integer)

```bash
./run.py --n-batch 1024          # Larger batches (faster)
./run.py --n-batch 128           # Smaller batches (less VRAM)
```

#### `--n-threads` (integer)

```bash
./run.py --n-threads -1          # Auto (default)
./run.py --n-threads 8           # 8 CPU threads
```

#### `--flash-attention` / `--no-flash-attention`

```bash
./run.py --flash-attention       # Enabled (default)
./run.py --no-flash-attention    # Disabled
```

#### `--rope-freq-base` (float)

```bash
./run.py --rope-freq-base 10000.0
```

#### `--rope-freq-scale` (float)

```bash
./run.py --rope-freq-scale 1.0
```

#### `--use-mmap` / `--no-mmap`

```bash
./run.py --use-mmap              # Enabled (default)
./run.py --no-mmap               # Disabled
```

#### `--use-mlock`

```bash
./run.py --use-mlock             # Enabled (prevents swapping)
```

#### `--kv-cache-quant` (choice)

**Options**: `f32`, `f16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`

```bash
./run.py --kv-cache-quant q8_0   # 8-bit quantization (saves VRAM)
./run.py --kv-cache-quant f16    # Half-precision (balanced)
```

**Default**: `null` (model default)

---

### REST API Mode

Uses LM Studio REST API v1 instead of Python SDK/CLI.

#### `--use-rest-api`

```bash
./run.py --use-rest-api --limit 1
```

**Benefits**:

- More detailed stats (TTFT, tok/s)
- Stateful chats (response_id tracking)
- Parallel requests (continuous batching)
- MCP integration
- Response caching

**Default**: `false` (uses SDK/CLI)

---

#### `--api-token` (string)

API permission token for REST API authentication.

```bash
./run.py --use-rest-api --api-token "lms_your_token_here"
```

**Default**: `null` (no token, server must be open)

**Create**: LM Studio → Settings → Server → Generate Token

---

#### `--n-parallel` (integer)

Max parallel predictions per model (REST API only).

```bash
./run.py --use-rest-api --n-parallel 8
```

**Default**: `4`

**Requirement**: LM Studio 0.4.0+, continuous batching support

---

#### `--unified-kv-cache`

Enable unified KV cache (REST API only).

```bash
./run.py --use-rest-api --unified-kv-cache --n-parallel 8
```

**Benefit**: Optimizes VRAM for parallel requests

**Default**: `false`

---

## Examples

### Quick Test of One Model

```bash
./run.py --limit 1 --runs 1
# Or shorter:
./run.py --dev-mode
```

### All 7B Llama Models with Q4/Q5/Q6 Quants

```bash
./run.py --include-models "llama.*7b" --quants "q4,q5,q6" --runs 2
```

### Vision Models Only with Hardware Profiling

```bash
./run.py --only-vision --enable-profiling --max-temp 80.0 --max-power 400.0
```

### REST API with Parallel Requests

```bash
./run.py --use-rest-api --n-parallel 8 --unified-kv-cache --limit 5
```

### Export Without New Tests

```bash
./run.py --export-only
```

### Custom Inference Parameters

```bash
./run.py --temperature 0.7 --top-p 0.95 --max-tokens 512 --limit 3
```

### Preset Workflow

```bash
./run.py --list-presets
./run.py --preset quick_test
./run.py --preset resource_limited --max-size 10 --runs 2
```

### Performance Tuning (VRAM-optimized)

```bash
./run.py --n-batch 128 --kv-cache-quant q8_0 --limit 5
```

### Manage Cache

```bash
./run.py --list-cache                     # Display cache contents
./run.py --export-cache "backup.json"     # Export cache
./run.py --retest --limit 1               # Ignore cache
```

---

## Configuration Priority

1. **CLI Arguments** (highest priority)
2. **User Config** (`~/.config/lm-studio-bench/defaults.json`)
3. **Project Config** (`config/defaults.json`)
4. **Hard-coded Defaults** (in code)

**Example**:

```bash
# User config has "num_runs": 5
# Project config has "num_runs": 3
./run.py --runs 1     # → uses 1 (CLI overrides)
./run.py              # → uses 5 (from user config)
```

---

## Tips & Best Practices

### 1. Persistent REST API Config

If you mainly use REST API:

`config/defaults.json`:

```json
{
  "lmstudio": {
    "use_rest_api": true,
    "api_token": "lms_your_token"
  }
}
```

Then simply:

```bash
./run.py --limit 1   # automatically uses REST API
```

### 2. VRAM Optimization

When VRAM is limited:

```bash
./run.py --kv-cache-quant q8_0 --n-batch 128 --max-size 10.0
```

### 3. Fast Development

```bash
./run.py --dev-mode   # Tests only smallest model with 1 run
```

### 4. Reproducible Benchmarks

```bash
./run.py --temperature 0.0 --runs 5 --retest
```

### 5. Hardware Monitoring

```bash
./run.py --enable-profiling --max-temp 80.0 --max-power 400.0
```

---

## Logging Configuration

The benchmark tool generates timestamped log files for debugging and monitoring.

### Log File Locations

```text
logs/
├── benchmark_YYYYMMDD_HHMMSS.log    # Benchmark execution logs
└── webapp_YYYYMMDD_HHMMSS.log       # Web dashboard logs
```

### Log Format

Each log entry follows this format:

```bash
YYYY-MM-DD HH:MM:SS,mmm - LEVEL - LEVEL_ICON message
2026-03-22 13:35:32,445 - INFO - ℹ️ Starting benchmark...
```

### Log Levels

The tool uses standard Python logging levels:

| Level | Usage | Examples |
| ----- | ----- | --------- |
| `INFO` | General information and progress | Model loading, benchmark completion, hardware metrics |
| `WARNING` | Non-fatal issues and fallbacks | GPU tool missing, using CLI fallback, skipped models |
| `ERROR` | Runtime errors requiring attention | Model load failure, API unavailable, VRAM exceeded |

### Level Icons

Each log level also gets an automatic icon prefix:

| Level | Icon |
| ----- | ---- |
| `DEBUG` | `🐛` |
| `INFO` | `ℹ️` |
| `WARNING` | `⚠️` |
| `ERROR` | `❌` |
| `CRITICAL` | `🔥` |

### Hardware Metrics in Logs

When hardware profiling is enabled (`--enable-profiling`), metrics appear with emoji indicators:

```text
🌡️ GPU Temp: 42°C
⚡ GPU Power: 125W
💾 GPU VRAM: 8.2GB
🧠 GPU GTT: 0.0GB
🖥️ CPU: 35.2%
💾 RAM: 18.5GB
```

### Third-Party Library Logging

The following libraries have suppressed debug output for cleaner logs:

| Library | Level | Reason |
| ------- | ------- | ------------------- |
| `httpx` | WARNING | HTTP client noise |
| `lmstudio` | WARNING | SDK debug output |
| `urllib3` | WARNING | HTTP library noise |
| `websockets` | WARNING | WebSocket protocol noise |

### Viewing Logs

**Real-time monitoring:**

```bash
# Watch benchmark execution
tail -f ~/.local/share/lm-studio-bench/logs/benchmark_*.log

# Watch web dashboard
tail -f ~/.local/share/lm-studio-bench/logs/webapp_*.log
```

**Search and filter:**

```bash
# Find errors
grep ERROR ~/.local/share/lm-studio-bench/logs/benchmark_*.log

# Find warnings
grep WARNING ~/.local/share/lm-studio-bench/logs/benchmark_*.log

# Find specific model errors
grep "model_name_pattern" \
  ~/.local/share/lm-studio-bench/logs/benchmark_*.log

# Count log entries by level
grep -c INFO ~/.local/share/lm-studio-bench/logs/benchmark_*.log
grep -c ERROR ~/.local/share/lm-studio-bench/logs/benchmark_*.log
```

---

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [REST_API_FEATURES.md](REST_API_FEATURES.md) - REST API details
- [HARDWARE_MONITORING_GUIDE.md](HARDWARE_MONITORING_GUIDE.md) - Hardware profiling
- [LLM_METADATA_GUIDE.md](LLM_METADATA_GUIDE.md) - Metadata & capabilities
