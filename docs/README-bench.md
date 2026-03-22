# Capability-Driven Benchmark Agent for LM Studio Bench

This benchmark agent implements capability-driven evaluation for language
models and multimodal models. It detects model capabilities, runs targeted
tests, computes quality metrics, and generates comprehensive reports.

## Features

- Automatic capability detection (general text, reasoning, vision, tooling)
- Per-capability test suites with standardized prompts
- Quality metrics: ROUGE, F1, Exact Match, Accuracy, Function Call Accuracy
- Performance metrics: tokens/sec, latency
- Machine-readable JSON and human-friendly HTML reports
- CLI interface with extensive configuration options
- Docker support for containerized execution
- GitHub Actions integration for CI/CD benchmarking

## Quick Start

### Local Execution

Run a benchmark on a model:

```bash
python -m cli.main "path/to/model" --output-dir output
```

With specific capabilities:

```bash
python -m cli.main "model-id" \
  --capabilities general_text,reasoning \
  --output-dir results
```

### Using Docker

Build the Docker image:

```bash
docker build -f bench/Dockerfile -t lm-bench-agent .
```

Run benchmark in container:

```bash
docker run -v $(pwd)/output:/app/output \
  lm-bench-agent "model-path" \
  --output-dir /app/output
```

## Capabilities

The agent supports four primary capabilities:

### 1. General Text

Tests basic language understanding and generation:

- Question answering
- Summarization
- Classification

Metrics: ROUGE-1, ROUGE-L, F1

### 2. Reasoning

Tests logical and mathematical reasoning:

- Logical reasoning (syllogisms)
- Math problem solving
- Chain-of-thought reasoning

Metrics: Exact Match, F1, Accuracy

### 3. Vision

Tests multimodal understanding (requires vision models):

- Image captioning
- Visual Question Answering (VQA)
- OCR and visual reasoning

Metrics: Accuracy, ROUGE-L

### 4. Tooling

Tests function calling and tool use:

- Function selection
- Parameter extraction
- API interaction patterns

Metrics: Function Call Accuracy, Parameter Accuracy

## CLI Reference

### Basic Usage

```bash
python -m cli.main MODEL_PATH [OPTIONS]
```

### Arguments

- `MODEL_PATH`: Path to model or model identifier (required)

### Options

#### Model Configuration

- `--model-name NAME`: Override model name (default: derived from path)
- `--capabilities CAPS`: Comma-separated capabilities to test
  - Options: `general_text,reasoning,vision,tooling`
  - Default: Auto-detect from model metadata

#### Output Configuration

- `--output-dir DIR`: Output directory (default: `output`)
- `--formats FMTS`: Output formats: `json,html` (default: both)

#### Test Configuration

- `--max-tests N`: Maximum tests per capability (default: 10)
- `--config FILE`: Path to YAML configuration file

#### Model Parameters

- `--context-length N`: Model context length (default: 2048)
- `--gpu-offload RATIO`: GPU offload ratio 0.0-1.0 (default: 1.0)
- `--temperature T`: Generation temperature (default: 0.1)

#### Other

- `--verbose`, `-v`: Enable verbose logging

### Examples

Benchmark with custom configuration:

```bash
python -m cli.main "mymodel" \
  --config custom_config.yaml \
  --max-tests 20 \
  --verbose
```

Test only reasoning capability:

```bash
python -m cli.main "reasoning-model" \
  --capabilities reasoning \
  --temperature 0.0 \
  --max-tests 50
```

Generate only JSON output:

```bash
python -m cli.main "model" \
  --formats json \
  --output-dir json_results
```

## Configuration File

The agent reads configuration from `config/bench.yaml` by default. Override with `--config` flag.

### Configuration Schema

```yaml
context_length: 2048
gpu_offload: 1.0
temperature: 0.1
max_tokens: 256
max_tests_per_capability: 10
use_rest_api: true

data_dir: tests/data
prompts_dir: tests/prompts

timeout_seconds: 300

metric_weights:
  general_text:
    rouge-1: 0.3
    rouge-l: 0.4
    f1: 0.3
  reasoning:
    exact_match: 0.5
    f1: 0.3
    accuracy: 0.2
  vision:
    accuracy: 0.6
    rouge-l: 0.4
  tooling:
    function_call_accuracy: 0.7
    accuracy: 0.3

composite_score_weights:
  quality: 0.6
  performance: 0.2
  efficiency: 0.2

lmstudio:
  host: localhost
  ports:
    - 1234
    - 1235
  api_token: null
```

### Key Configuration Options

- `context_length`: Maximum context length for model
- `gpu_offload`: GPU memory allocation (0.0 = CPU only, 1.0 = full GPU)
- `max_tests_per_capability`: Limit tests to prevent long runs
- `metric_weights`: Per-capability metric importance
- `composite_score_weights`: Overall score composition

## Output Format

### JSON Report

The JSON report follows this schema:

```json
{
  "schema_version": "1.0",
  "generated_at": "2025-01-15T10:30:00",
  "report": {
    "model_name": "model-name",
    "model_path": "path/to/model",
    "capabilities": ["general_text", "reasoning"],
    "timestamp": "2025-01-15T10:30:00",
    "summary": {
      "total_tests": 20,
      "successful_tests": 19,
      "success_rate": 0.95,
      "avg_latency_ms": 245.6,
      "avg_quality_score": 0.823,
      "avg_throughput_tokens_per_sec": 42.3,
      "by_capability": {
        "general_text": {
          "test_count": 10,
          "avg_quality_score": 0.856,
          "success_rate": 1.0
        }
      }
    },
    "results": [
      {
        "test_id": "qa_001",
        "capability": "general_text",
        "latency_ms": 230.5,
        "tokens_generated": 12,
        "throughput": 52.1,
        "quality_score": 0.89,
        "metrics": [
          {
            "name": "rouge-1",
            "value": 0.85,
            "normalized": 0.85
          }
        ],
        "error": null
      }
    ],
    "config": {},
    "raw_outputs_dir": "output/raw"
  }
}
```

### HTML Report

The HTML report provides:

- Summary statistics with visual indicators
- Per-test results table with status, latency, and quality scores
- Capability breakdown with aggregated metrics
- Color-coded quality scores (green/yellow/red)

### Raw Outputs

Individual test outputs are saved in `output/raw/`:

```json
{
  "test_id": "qa_001",
  "capability": "general_text",
  "prompt": "What is the capital of France?",
  "response": "Paris",
  "latency_ms": 230.5,
  "tokens_generated": 12,
  "throughput": 52.1,
  "timestamp": 1642244400.123,
  "error": null
}
```

## GitHub Actions Integration

The workflow `.github/workflows/bench.yml` enables CI benchmarking.

### Triggering the Workflow

#### Manual Trigger

1. Go to Actions tab in GitHub
2. Select "Capability-Driven Benchmark"
3. Click "Run workflow"
4. Enter model path and capabilities
5. Click "Run workflow"

#### Scheduled Trigger

Runs automatically every Sunday at midnight (UTC).

#### Push Trigger

Runs on push to `main` or `dev` branches.

Note: the benchmark step currently reads the model path only from
manual `workflow_dispatch` inputs. Push- and schedule-triggered
runs therefore skip the actual benchmark unless you adapt the
workflow to read the model path from another configuration source
(for example, a repository variable or secret).

### Workflow Outputs

The workflow uploads three artifacts:

1. **benchmark-results-json**: JSON reports (30-day retention)
2. **benchmark-results-html**: HTML reports (30-day retention)
3. **benchmark-raw-outputs**: Raw test outputs (7-day retention)

For pull requests, a summary comment is posted with key metrics.

## Adding Test Data

### General Text Tests

Add test cases to `tests/data/text/qa_samples.json`:

```json
{
  "id": "qa_004",
  "prompt": "Your question here",
  "reference": "Expected answer",
  "category": "domain"
}
```

### Reasoning Tests

Add to `tests/data/text/reasoning_samples.json`:

```json
{
  "id": "reasoning_004",
  "prompt": "Problem statement",
  "reference": "Answer",
  "reasoning": "Explanation of solution",
  "category": "math"
}
```

### Vision Tests

Place images in `tests/data/images/` and reference them in test cases.

### Tooling Tests

Add to `tests/data/text/tooling_samples.json`:

```json
{
  "id": "tool_004",
  "task": "Task description",
  "expected_function": "function_name",
  "expected_parameters": {"param": "value"},
  "category": "function_calling"
}
```

## Customizing Prompts

Prompt templates are in `tests/prompts/`:

- `general_text_qa.md`: Question answering
- `general_text_summarization.md`: Summarization
- `reasoning_logical.md`: Logical reasoning
- `reasoning_math.md`: Math problems
- `vision_caption.md`: Image captioning
- `vision_vqa.md`: Visual QA
- `tooling_function_call.md`: Function calling

Edit templates to adjust instruction format or add few-shot examples.

## Troubleshooting

### Model Loading Fails

Ensure LM Studio is running and the model is available:

```bash
lms status
lms models list
```

### No Tests Execute

Check that test data files exist:

```bash
ls tests/data/text/
```

Verify capabilities are correctly specified:

```bash
python -m cli.main "model" --capabilities general_text --verbose
```

### Metrics Are Zero

This usually means:

- Model output format doesn't match expected format
- Reference answers need normalization
- Wrong capability assigned to test

Check raw outputs in `output/raw/` to inspect actual responses.

### Timeout Errors

Increase timeout in config:

```yaml
timeout_seconds: 600
```

Or reduce test count:

```bash
python -m cli.main "model" --max-tests 5
```

## API Integration

### Using as a Library

```python
from pathlib import Path
from agents.runner import BenchmarkRunner
from cli.reporting import generate_reports

config = {
    "context_length": 2048,
    "max_tests_per_capability": 5,
    "use_rest_api": True
}

runner = BenchmarkRunner(
    config=config,
    output_dir=Path("output")
)

report = runner.run(
    model_path="mymodel",
    model_name="MyModel",
    capabilities=["general_text"]
)

outputs = generate_reports(
    report_data=report,
    output_dir=Path("output"),
    formats=["json", "html"]
)

print(f"JSON: {outputs['json']}")
print(f"HTML: {outputs['html']}")
```

### Custom Model Adapter

Implement `ModelAdapter` interface:

```python
from agents.benchmark import ModelAdapter, InferenceResult

class CustomAdapter(ModelAdapter):
    def load(self, model_path, **kwargs):
        pass

    def unload(self):
        pass

    def infer(self, prompt, image_path=None, **kwargs):
        return InferenceResult(...)

    def is_loaded(self):
        return True
```

Use with runner:

```python
adapter = CustomAdapter()
report = runner.run(
    model_path="model",
    adapter=adapter
)
```

## Architecture

### Components

- `agents/capabilities.py`: Capability detection logic
- `agents/benchmark.py`: Core benchmark agent and model adapters
- `agents/runner.py`: Test orchestration and loading
- `cli/metrics.py`: Metric implementations
- `cli/reporting.py`: Report generation (JSON, HTML)
- `cli/main.py`: Command-line interface
- `config/bench.yaml`: Default configuration
- `tests/data/`: Test datasets
- `tests/prompts/`: Prompt templates

### Data Flow

1. CLI parses arguments and loads configuration
2. Runner detects capabilities from model metadata or flags
3. Test loader creates test cases for detected capabilities
4. Model adapter loads the model
5. Agent runs each test case:
   - Executes inference
   - Saves raw output
   - Computes metrics
6. Reporter generates JSON and HTML from results
7. Outputs are saved to disk

## License

This benchmark agent is part of LM-Studio-Bench and follows the same license.

## Contributing

Contributions are welcome:

- Add new capabilities
- Implement new metrics
- Expand test datasets
- Improve prompt templates
- Enhance reporting formats

Follow the coding standards in `.github/instructions/code-standards.instructions.md`.
