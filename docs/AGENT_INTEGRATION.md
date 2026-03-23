# Capability-Driven Benchmark Agent Integration

The new Capability-Driven Benchmark Agent functionality is fully integrated into the project and is now available via `run.py`.

## 3 Operating Modes

The system now supports 3 different operating modes:

### 1. Classic Benchmark (Default)

Measures token/s speed across all installed models:

```bash
./run.py --limit 5              # Test 5 models
./run.py --export-only          # Generate reports from cache
./run.py --runs 1               # Fast-mode with 1 measurement
```

**Metrics:** Tokens/s, latency, VRAM usage

### 2. Capability-Driven Agent ⭐ NEW

Tests model capabilities with quality metrics:

```bash
./run.py --agent "model-id"     # Automatically test all capabilities

# With specific capabilities
./run.py --agent "llama-13b" --capabilities general_text,reasoning

# With output format options
./run.py --agent "llama-13b" --output-dir ./results/ --formats json,html

# Verbose mode
./run.py --agent "llama-13b" --verbose
```

**Detectable Capabilities:**

- `general_text` - Basic language understanding (QA, summarization, classification)
- `reasoning` - Logical and mathematical reasoning
- `vision` - Multimodal understanding (image captioning, VQA, OCR)
- `tooling` - Tool calling and function execution

**Metrics per Capability:**

- Quality: ROUGE, F1, Exact Match, Accuracy, Function Call Accuracy
- Performance: Tokens/s, latency
- Reports: JSON + HTML with visualizations
- **Storage:** SQLite database for historical tracking and comparison

**Runtime Resilience:**

- Multi-model capability runs continue when a single model fails to load or
  execute; failed models are logged and skipped.
- Embedding models are retried automatically without
  `offload_kv_cache_to_gpu` if LM Studio rejects that load option.

**Data Storage:**

Results are automatically saved to:

- **JSON Reports:** `./output/benchmark_results_*.json`
- **HTML Reports:** `./output/benchmark_results_*.html`
- **SQLite Cache:** `~/.local/share/lm-studio-bench/results/benchmark_cache.db`

The SQLite database stores individual test results and capability summaries, allowing you to:

- Track performance over time
- Compare results across models
- Query specific capability metrics
- Build custom dashboards from cached data

### SQLite Metrics Matrix (Classic vs Capability)

The table below lists what is currently persisted in SQLite for both test
types, so missing metrics are easy to spot.

| Metric Group | Classic Benchmark (`benchmark_results`) | Capability Benchmark (`agent_results`, `agent_summaries`) |
| --- | --- | --- |
| Run identity | `id`, `model_key`, `model_name`, `quantization`, `timestamp` | `id`, `model_name`, `model_path`, `capability`, `test_id`, `test_name`, `timestamp` |
| Throughput/latency | `avg_tokens_per_sec`, `avg_ttft`, `avg_gen_time`, `tokens_per_sec_p50`, `tokens_per_sec_p95`, `tokens_per_sec_std`, `ttft_p50`, `ttft_p95`, `ttft_std` | `latency_ms`, `throughput_tokens_per_sec` (per test), `avg_latency_ms`, `avg_throughput` (summary) |
| Token volume | `prompt_tokens`, `completion_tokens` | `prompt_tokens`, `tokens_generated` |
| Quality metrics | Stored for parity columns but normally `NULL` for classic runs | `quality_score`, `rouge_score`, `f1_score`, `exact_match_score`, `accuracy_score`, `function_call_accuracy`, `avg_quality_score`, `avg_rouge`, `avg_f1`, `avg_exact_match`, `avg_accuracy` |
| Success/failure | `success`, `error_message`, `error_count` | `success`, `error_message` (per test), `total_tests`, `successful_tests`, `failed_tests`, `success_rate`, `error_count` |
| Hardware profiling | `gpu_type`, `gpu_offload`, `vram_mb`, `temp_celsius_min/max/avg`, `power_watts_min/max/avg`, `vram_gb_min/max/avg`, `gtt_gb_min/max/avg`, `cpu_percent_min/max/avg`, `ram_gb_min/max/avg` | Same run-level hardware fields are persisted on each capability test row |
| Inference/load params | `context_length`, `temperature`, `top_k_sampling`, `top_p_sampling`, `min_p_sampling`, `repeat_penalty`, `max_tokens`, `n_gpu_layers`, `n_batch`, `n_threads`, `flash_attention`, `rope_freq_base`, `rope_freq_scale`, `use_mmap`, `use_mlock`, `kv_cache_quant` | Same run-level inference/load fields are persisted on each capability test row |
| Environment/version | `lmstudio_version`, `app_version`, `nvidia_driver_version`, `rocm_driver_version`, `intel_driver_version`, `os_name`, `os_version`, `cpu_model`, `python_version` | Same environment/version fields are persisted on each capability test row |
| Derived/comparison | `tokens_per_sec_per_gb`, `tokens_per_sec_per_billion_params`, `speed_delta_pct`, `prev_timestamp` | Same derived/comparison fields are persisted on each capability test row |
| Raw text/reference | `prompt` (full input prompt), `raw_output`, `reference_output` | `prompt`, `raw_output`, `reference_output` |

#### Quick gap summary

- Missing in capability mode: TTFT distribution stats and classic-only
  aggregate throughput percentiles.
- Missing in classic mode: meaningful per-test quality metrics
  (ROUGE/F1/Exact/Accuracy) because classic benchmarks do not execute
  capability test cases.

#### Variant selection in REST mode

- Capability mode now forwards the exact requested model identifier,
  including any `@quantization` suffix, to the LM Studio REST API.
- This keeps `load`, `chat`, and `unload` aligned with the selected
  variant and avoids silently falling back to a server-side default
  quantization.

### 3. Web Dashboard

Modern web UI with live streaming and configuration:

```bash
./run.py --webapp               # Starts on http://localhost:8080
./run.py -w                     # Short form
```

## Agent Options

```bash
./run.py --agent MODEL_PATH [OPTIONS]

OPTIONS:
  --capabilities CAPS        Comma-separated capabilities
                            (general_text, reasoning, vision, tooling)
  --output-dir DIR          Output directory (default: output)
  --config FILE             YAML configuration file
  --formats FORMATS         Output formats: json,html (default: json,html)
  --max-tests N             Max tests per capability
  --context-length N        Model context length (default: 2048)
  --gpu-offload RATIO       GPU offload ratio 0.0-1.0 (default: 1.0)
  --temperature TEMP        Generation temperature (default: 0.1)
  -v, --verbose             Enable verbose logging
```

## Test Data and Prompts

The following test files are available:

```files
tests/
├── data/
│   ├── text/
│   │   ├── qa_samples.json              # QA examples
│   │   ├── reasoning_samples.json       # Reasoning examples
│   │   └── tooling_samples.json         # Tool-calling examples
│   └── images/
│       └── README.md                    # Vision datasets
└── prompts/
    ├── general_text_qa.md
    ├── general_text_summarization.md
    ├── reasoning_logical.md
    ├── reasoning_math.md
    ├── tooling_function_call.md
    ├── vision_caption.md
    └── vision_vqa.md
```

## Example Executions

```bash
# All capabilities (auto-detected)
./run.py --agent "my-model" --output-dir results/

# Only General Text and Reasoning
./run.py --agent "my-model" --capabilities general_text,reasoning

# With custom config
./run.py --agent "my-model" --config config/bench.yaml

# Verbose with all details
./run.py --agent "my-model" --verbose --max-tests 20

# Classic benchmark still available
./run.py --limit 10 --runs 3
```

## Code Structure

```files
cli/
├── main.py                  # CLI entrypoint for agent
├── __main__.py              # Makes cli package executable
├── benchmark.py             # Classic benchmark runner
├── metrics.py               # Metric implementations
├── reporting.py             # JSON & HTML report generation
└── report_template.html.template

config/
└── bench.yaml               # Default configuration

agents/
├── benchmark.py           # Benchmark executor
├── runner.py                # Test orchestration
└── capabilities.py          # Capability detection

core/
├── config.py                # Configuration loading
├── paths.py                 # XDG/user path handling
├── client.py                # LM Studio REST API client
└── tray.py                  # Linux tray controller
```

## Documentation

- [README-bench.md](README-bench.md) - Detailed agent documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration guide

## Logging

Capability benchmark logs use automatic level icons in addition to
benchmark-specific emoji markers:

- `🐛` Debug
- `ℹ️` Info
- `⚠️` Warning
- `❌` Error
- `🔥` Critical
