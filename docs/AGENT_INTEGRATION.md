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

**Data Storage:**

Results are automatically saved to:

- **JSON Reports:** `./output/benchmark_results_*.json`
- **HTML Reports:** `./output/benchmark_results_*.html`
- **SQLite Cache:** `~/.local/share/lm-studio-bench/results/agent_results.db`

The SQLite database stores individual test results and capability summaries, allowing you to:

- Track performance over time
- Compare results across models
- Query specific capability metrics
- Build custom dashboards from cached data

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
./run.py --agent "my-model" --config bench/config.yaml

# Verbose with all details
./run.py --agent "my-model" --verbose --max-tests 20

# Classic benchmark still available
./run.py --limit 10 --runs 3
```

## Code Structure

```files
bench/
├── cli.py                    # CLI entrypoint for agent
├── __main__.py              # Makes bench package executable
├── config.yaml              # Default configuration
├── metrics.py               # Metric implementations
├── reporting.py             # JSON & HTML report generation
└── Dockerfile               # Container definition

agents/
├── bench_agent.py           # Benchmark executor
├── runner.py                # Test orchestration
└── capabilities.py          # Capability detection
```

## Documentation

- [README-bench.md](../README-bench.md) - Detailed agent documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration guide
