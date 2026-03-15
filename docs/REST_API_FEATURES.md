# LM Studio REST API v1 Integration

## Overview

The benchmark tool now supports LM Studio's native REST API v1 (`/api/v1/*`)
in addition to the existing Python SDK/CLI mode. This enables advanced
features such as stateful chats, parallel requests, and more precise metrics.

## New Features

### 1. **REST API Mode** (`--use-rest-api`)

- Uses `/api/v1/chat` for inference instead of the Python SDK
- Stateful chat management (response_id tracking)
- Detailed stats in the response (TTF, tokens/s, tokens in/out)
- Streaming events for more accurate measurement

### 2. **Model Management via API**

- `GET /api/v1/models` — list with capabilities (vision, tool-use)
- `POST /api/v1/models/load` — explicit load with configuration
- `POST /api/v1/models/unload` — explicit unload
- `POST /api/v1/models/download` — download model via API

### 3. **Improved Capabilities Detection**

- **Vision support**: `capabilities.vision` flag from the API
- **Tool calling**: `capabilities.trained_for_tool_use` flag
- Use the `--only-vision` or `--only-tools` filters

### 4. **Parallel Inference** (LM Studio 0.4.0+)

- `--n-parallel N` — max concurrent predictions (default: 4)
- `--unified-kv-cache` — optimizes VRAM usage for parallel requests
- Continuous batching support (llama.cpp 2.0+)

### 5. **API Authentication**

- `--api-token TOKEN` — permission key for protected servers
- Config: `lmstudio.api_token` in `config/defaults.json`

## Usage

### Basic usage (REST API mode)

```bash
# REST API with default settings
./run.py --use-rest-api --limit 1

# With API token
./run.py --use-rest-api --api-token "your-token-here" --limit 1

# With parallel requests (LM Studio 0.4.0+)
./run.py --use-rest-api --n-parallel 8 --unified-kv-cache --limit 1
```

### Filter by capabilities

```bash
# Test only vision-capable models
./run.py --use-rest-api --only-vision --runs 2

# Test only tool-calling models
./run.py --use-rest-api --only-tools --runs 2
```

### Config file (persistent)

`config/defaults.json`:

```json
{
  "lmstudio": {
    "host": "localhost",
    "ports": [1234, 1235],
    "api_token": "your-token-here",
    "use_rest_api": true
  }
}
```

Then simply:

```bash
./run.py --limit 1  # will automatically use REST API from config
```

## Comparison: SDK vs. REST API

| Feature | SDK/CLI Mode | REST API Mode |
| ------- | ------------- | -------------- |
| Model Loading | `lms load` CLI | `POST /api/v1/models/load` |
| Inference | `lmstudio.llm()` | `POST /api/v1/chat` |
| Stats | SDK stats object | Detailed response stats |
| Streaming | SDK stream | SSE stream (Server-Sent Events) |
| Parallel Requests | ❌ | ✅ (with `--n-parallel`) |
| Stateful Chats | ❌ | ✅ (response_id tracking) |
| Capabilities | Metadata parsing | Native API fields |
| Authentication | ❌ | ✅ (permission keys) |

## API Response Format

### Dashboard summary API (`/api/dashboard/stats`)

The web dashboard now exposes additional summary fields for quick visual
analysis of benchmark history. The endpoint is consumed by the Home and
Results views to render KPI cards and charts.

New response fields:

- `speed_summary`: `min`, `p50`, `avg`, `p95`, `max` tokens/s
- `top_models_extended`: Top 10 models by speed (model, quantization,
  speed, VRAM, architecture)
- `quantization_distribution`: count per quantization
- `architecture_distribution`: count per architecture
- `efficiency_top`: top models ranked by `tokens_per_sec_per_gb`

Example (excerpt):

```json
{
  "speed_summary": {
    "min": 22.44,
    "p50": 48.17,
    "avg": 51.26,
    "p95": 86.11,
    "max": 93.88
  },
  "top_models_extended": [
    {
      "model_name": "qwen/qwen3-4b@q4_k_m",
      "quantization": "q4_k_m",
      "speed": 93.88,
      "vram_mb": "6144",
      "architecture": "qwen3"
    }
  ],
  "quantization_distribution": {
    "q4_k_m": 22,
    "q5_k_m": 13
  }
}
```

### `/api/v1/chat` stats

```json
{
  "text": "... generated text ...",
  "stats": {
    "tokens_in": 42,
    "tokens_out": 128,
    "time_to_first_token_ms": 234.5,
    "total_time_ms": 1523.8,
    "tokens_per_second": 84.02
  }
}
```

### `/api/v1/models` capabilities

```json
{
  "models": [
    {
      "key": "llava-1.6-vicuna-7b-q4_k_m",
      "capabilities": {
        "vision": true,
        "trained_for_tool_use": false
      }
    },
    {
      "key": "qwen-2.5-coder-14b-instruct-q5_k_m",
      "capabilities": {
        "vision": false,
        "trained_for_tool_use": true
      }
    }
  ]
}
```

## Implementation details

### New files

- **`src/rest_client.py`**: REST API client with wrapper functions
  - `LMStudioRESTClient`: main class
  - `ModelInfo`, `ModelCapabilities`, `ChatStats`: data classes
  - `is_vision_model()`, `is_tool_model()`: helpers

### Modified files

- **`src/benchmark.py`**:
  - `_run_inference()`: dispatcher (SDK vs REST)
  - `_run_inference_rest()`: REST-based inference
  - `_run_inference_sdk()`: SDK-based inference (renamed)
  - `_load_model_rest()`, `_unload_model_rest()`: REST model management

- **`config/defaults.json`**: added `api_token`, `use_rest_api` fields

- **`src/config_loader.py`**: new config fields in `BASE_DEFAULT_CONFIG`

### CLI flags

```bash
--use-rest-api              Enable REST API mode
--api-token TOKEN           API permission token
--n-parallel N              Max parallel predictions (REST only)
--unified-kv-cache          Unified KV cache (REST only)
```

## Troubleshooting

### Server unreachable

```bash
# Check whether LM Studio is running
curl http://localhost:1234/

# Healthcheck via CLI
lms server status
```

### API token errors

```bash
# Generate token in Settings > Server
# Save it in config or pass via CLI
./run.py --use-rest-api --api-token "lms_..."
```

### REST vs SDK performance

- **REST**: more precise stats, more features
- **SDK**: slightly faster (direct Python access)
- For benchmarking, REST is recommended (better metrics)

## Additional REST Client Features

### 1. Download Progress Tracking

The REST client now supports real-time download progress monitoring:

```python
from rest_client import LMStudioRESTClient

client = LMStudioRESTClient()

def on_progress(status):
    if status["state"] == "downloading":
        print(f"Progress: {status['progress'] * 100:.1f}%")

# Wait for download to complete with progress updates
success = client.download_model(
    model_key="qwen/qwen3-1.7b",
    wait_for_completion=True,
    progress_callback=on_progress
)
```

**API**: Polls `/api/v1/models/download/status` every 2 seconds until completion.

### 2. MCP Integration

Model Context Protocol (MCP) servers can now be attached to chat requests:

```python
# LM Studio v1 API format
mcp_integrations = [
    {
        "type": "ephemeral_mcp",
        "server_label": "filesystem",
        "server_url": "http://localhost:3001/mcp"
    }
]

result = client.chat_stream(
    messages=[{"role": "user", "content": "List files in /tmp"}],
    model="qwen/qwen3-4b",
    mcp_integrations=mcp_integrations
)
```

**Note**: Requires MCP server running. Integrations are passed in the `integrations` array field.

### 3. Stateful Chat History

Enable multi-turn conversations with automatic `response_id` tracking:

```python
client = LMStudioRESTClient()

# First message
result1 = client.chat_stream(
    messages=[{"role": "user", "content": "What is 2+2?"}],
    model="qwen/qwen3-4b",
    use_stateful=True
)
# response_id stored automatically

# Second message - automatically includes previous_response_id
result2 = client.chat_stream(
    messages=[{"role": "user", "content": "Add 3 to that."}],
    model="qwen/qwen3-4b",
    use_stateful=True
)
# Server can maintain conversation context

# Reset state when starting new conversation
client.reset_stateful_chat()
```

**API**: Extracts `response_id` from `chat.end` event, sends `previous_response_id` in subsequent requests.

### 4. Response Caching

Identical requests are cached in memory for instant responses:

```python
client = LMStudioRESTClient(enable_cache=True)

# First request - hits API (slow)
result1 = client.chat_stream(
    messages=[{"role": "user", "content": "Count to 5"}],
    model="qwen/qwen3-4b",
    temperature=0.5
)
# Time: ~0.5s

# Second identical request - hits cache (instant)
result2 = client.chat_stream(
    messages=[{"role": "user", "content": "Count to 5"}],
    model="qwen/qwen3-4b",
    temperature=0.5
)
# Time: ~0.0s (10,000x faster!)

# Cache management
cache_size = len(client._RESPONSE_CACHE)  # Check cache size
cleared = client.clear_cache()             # Clear all cached responses
```

**Cache Key**: MD5 hash of `(messages, model, temperature)`  
**Bypassed**: When using `use_stateful=True` or `mcp_integrations` (non-deterministic)

## Documentation links

- [LM Studio REST API Docs](https://lmstudio.ai/docs/developer/rest)
- [/api/v1/models endpoint](https://lmstudio.ai/docs/developer/rest/list)
- [/api/v1/chat endpoint](https://lmstudio.ai/docs/developer/rest/chat)
- [Headless mode](https://lmstudio.ai/docs/developer/core/headless)
- [LM Studio 0.4.0 blog](https://lmstudio.ai/blog/0.4.0)
