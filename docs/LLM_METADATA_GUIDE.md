# LM Studio CLI - Available LLM Metadata with GPU Analysis

## 📋 Quick Reference

### Main metadata query commands

```bash
lms ls --json           # All downloaded models with metadata
lms ps --json           # Currently loaded models
lms status              # Server status + model size
lms version             # LM Studio version
```

## 🎯 GPU Support and Hardware Requirements

### Automatic GPU detection in the benchmark

The benchmark system automatically detects all your GPUs and specs:

**NVIDIA GPUs:**

- Automatic detection via `nvidia-smi`
- VRAM size recorded for offload optimization
- Temperature and power are monitored

**AMD GPUs (rocm-smi):**

- Detailed device ID mapping for GPU model names
- VRAM and GTT memory are tracked separately
- rocm-smi search paths: `/usr/bin`, `/usr/local/bin`, `/opt/rocm-*/bin/`

**iGPU detection:**

- Radeon iGPUs are extracted from the CPU string
- Regex pattern: `Radeon\s+(\d+[A-Za-z]*)`
- Shows, for example, "Radeon 890M (Ryzen 9 7950X3D)" separately

## 📊 Full Metadata Fields (15 fields per model)

### Category 1: Model identification (5 fields)

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| `type` | string | "llm" | Model type (llm, embedding) |
| `modelKey` | string | "mistralai/ministral-3-3b" | Unique model ID |
| `displayName` | string | "Ministral 3 3B" | Display name |
| `publisher` | string | "mistralai" | Model publisher/developer |
| `path` | string | "mistralai/ministral-3-3b" | Local storage path |

### Category 2: Technical specifications (4 fields)

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| `architecture` | string | "mistral3", "gemma3", "llama" | Model architecture |
| `format` | string | "gguf" | File format (GGUF, etc.) |
| `paramsString` | string | "3B", "7B", "13B" | Parameter size |
| `sizeBytes` | number | 2986817071 | Size in bytes |

### Category 3: Model capabilities (3 fields)

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| `vision` | boolean | true / false | Can process images? |
| `trainedForToolUse` | boolean | true / false | Supports tool calling? |
| `maxContextLength` | number | 131072, 262144 | Maximum context length in tokens |

### Category 4: Quantization and variants (4 fields)

| Field | Type | Example | Description |
| --- | --- | --- | --- |
| `quantization.name` | string | "Q4_K_M", "Q8_0", "F16" | Quantization method |
| `quantization.bits` | number | 4, 8, 16 | Bits per weight |
| `variants` | array | `[@q4_k_m, @q8_0]` | All available quantizations |
| `selectedVariant` | string | "mistralai/ministral-3-3b@q4_k_m" | Current selection |

## 🔍 Practical Examples with Your Models

### Example 1: List vision models

```bash
lms ls --json | jq '.[] | select(.vision == true) | {displayName, paramsString, maxContextLength}'
```

**Output:**

```text
  • Gemma 3 4B (4B) - 131072 tokens
  • Ministral 3 3B (3B) - 262144 tokens
  • Qwen3 Vl 8B (8B) - 262144 tokens
```

The command uses the `jq` filter shown above.

### Example 2: Tool-calling models only

```bash
lms ls --json | jq '.[] | select(.trainedForToolUse == true) | .displayName'
```

### Example 3: Sort models by size

```bash
lms ls --json | jq 'sort_by(.sizeBytes) | .[] | {displayName, sizeGB: (.sizeBytes/1024/1024/1024|round*100/100)}'
```

### Example 4: Models with large context length (≥128k tokens)

```bash
lms ls --json | jq '.[] | select(.maxContextLength >= 131072) | {modelKey, maxContextLength}'
```

### Example 5: Model architecture distribution

```bash
lms ls --json | jq -r '.[] | .architecture' | sort | uniq -c
```

## 🐍 Python SDK Access

### SDK methods for metadata queries

```python
import lmstudio

# 1. Fetch all downloaded models
models = lmstudio.list_downloaded_models()
for model in models:
    print(f"Model: {model.model_key}")
    print(f"  Size: {model.info.sizeBytes / 1024**3:.2f} GB")
    print(f"  Vision: {model.info.vision}")
    print(f"  Maximum context length: {model.info.maxContextLength} tokens")
    print(f"  Architecture: {model.info.architecture}")
    print()

# 2. Currently loaded models
loaded_models = lmstudio.list_loaded_models()
for llm in loaded_models:
    print(f"Loaded: {llm.identifier}")

# 3. Filter models
vision_models = [m for m in models if m.info.vision]
print(f"Vision models: {len(vision_models)}")

# 4. Sort by size
large_models = sorted(models, key=lambda m: m.info.sizeBytes, reverse=True)[:3]
for model in large_models:
    print(f"{model.info.displayName}: {model.info.sizeBytes / 1024**3:.2f} GB")
```

## 💡 Common Use Cases

### Use case 1: Quick performance tests

Filter only small models < 1GB for fast benchmarks:

```bash
lms ls --json | jq '.[] | select(.sizeBytes < 1000000000) | .modelKey'
```

### Use case 2: Long-form processing

Models with large context for document analysis:

```bash
lms ls --json | jq '.[] | select(.maxContextLength >= 100000) | .displayName'
```

### Use case 3: Image processing

Multi-modal models for vision tasks:

```bash
lms ls --json | jq '.[] | select(.vision == true) | .modelKey'
```

### Use case 4: Tool integration

Models with function calling for agent systems:

```bash
lms ls --json | jq '.[] | select(.trainedForToolUse == true) | .displayName'
```

### Use case 5: Quantization comparison

All available quantizations for a model:

```bash
lms ls "google/gemma-3-1b" --json | jq '.variants[]'
```

## 🎯 Benchmarking with Metadata

Integration into benchmark scripts:

```python
import subprocess
import json

# Load model metadata
result = subprocess.run(
    ['lms', 'ls', '--json'],
    capture_output=True,
    text=True,
    check=False
)
models = json.loads(result.stdout)

# Filter for benchmarking
benchmark_candidates = [
    m for m in models
    if m['sizeBytes'] < 5e9  # < 5GB
    and m['vision'] is False  # Text only
]

print(f"Benchmark candidates: {len(benchmark_candidates)}")
for model in benchmark_candidates:
    print(f"  - {model['displayName']} ({model['paramsString']})")
```

## 📝 Tips and Tricks

### Convert size

```bash
# Bytes to GB
python3 -c "print(f'{2986817071/1024**3:.2f} GB')"  # Output: 2.78 GB
```

### JSON pretty print

```bash
lms ls --json | jq '.' | less
```

### Quick statistics

```bash
# Average model size
lms ls --json | jq '[.[].sizeBytes] | add / length / 1024 / 1024 / 1024'

# Largest model
lms ls --json | jq 'max_by(.sizeBytes) | .displayName'

# Models per architecture
lms ls --json | jq 'group_by(.architecture) | map({architecture: .[0].architecture, count: length})'
```

## 🔗 Related Commands

```bash
lms status              # Server status (shows loaded models too)
lms version             # LM Studio version
lms load <model>        # Load a model
lms unload --all        # Unload all models
```

## Troubleshooting

### No output for `lms ls --json`

- Ensure the LM Studio server is running: `lms server start`
- Check for port conflicts

### jq not installed

- Install: `sudo apt install jq` (Linux) or `brew install jq` (macOS)
- Alternative: use Python parsing

### Unlimited output

- Use `| head -n 5` to limit
- Or pipe to `less` for paging: `| less`
