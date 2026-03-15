# Architecture Documentation

<!-- markdownlint-disable MD033 MD046 -->

> Comprehensive architecture documentation with Mermaid diagrams showing how the Python modules interact and
> how CLI arguments and configuration files are processed.

---

## Table of Contents

- [Architecture Documentation](#architecture-documentation)
  - [Table of Contents](#table-of-contents)
  - [System Architecture Overview](#system-architecture-overview)
  - [Startup Flow](#startup-flow)
    - [AppImage Entry Point](#appimage-entry-point)
    - [run.py Flow](#runpy-flow)
  - [Setup Flow (Installation \& Configuration)](#setup-flow-installation--configuration)
  - [Tray Control Flow (Linux)](#tray-control-flow-linux)
  - [Tray Quit Sequence (Linux)](#tray-quit-sequence-linux)
  - [Configuration Loading](#configuration-loading)
  - [Configuration Priority](#configuration-priority)
  - [Benchmark Execution Flow](#benchmark-execution-flow)
  - [REST API vs SDK Mode](#rest-api-vs-sdk-mode)
  - [Component Details](#component-details)
    - [1. run.py (Entry Point)](#1-runpy-entry-point)
    - [2. config\_loader.py (Configuration Manager)](#2-config_loaderpy-configuration-manager)
    - [3. benchmark.py (Main Engine)](#3-benchmarkpy-main-engine)
    - [4. rest\_client.py (REST API Client)](#4-rest_clientpy-rest-api-client)
    - [5. tray.py (Linux Tray Controller)](#5-traypy-linux-tray-controller)
    - [6. web/app.py + dashboard.html.jinja (Dashboard Analytics)](#6-webapppy--dashboardhtmljinja-dashboard-analytics)
  - [Data Flow Summary](#data-flow-summary)
  - [Testing Architecture](#testing-architecture)
    - [Test Organization](#test-organization)
    - [Test Coverage by Component](#test-coverage-by-component)
    - [Testing Approach](#testing-approach)
  - [See Also](#see-also)

---

## System Architecture Overview

```mermaid
graph TB
    User([User]) --> RunPy[run.py<br/>Entry Point]

    RunPy -->|--webapp/-w flag| WebApp[web/app.py<br/>FastAPI Server]
    RunPy -->|benchmark mode| Benchmark[src/benchmark.py<br/>Benchmark Engine]
    
    Benchmark --> ConfigLoader[src/config_loader.py<br/>Configuration Manager]
    Benchmark --> PresetManager[src/preset_manager.py<br/>Preset Manager]
    Benchmark --> RestClient[src/rest_client.py<br/>REST API Client]
    
    ConfigLoader -->|reads| ProjectConfig[config/defaults.json<br/>Project Defaults]
    ConfigLoader -->|reads| UserConfig[~/.config/lm-studio-bench/defaults.json<br/>User Overrides]
    ConfigLoader -->|provides| DefaultConfig[(DEFAULT_CONFIG<br/>Merged)]
    
    Benchmark -->|uses| LMStudio[LM Studio Server<br/>localhost:1234/1235]
    RestClient -->|HTTP API v1| LMStudio
    
    Benchmark -->|writes| ResultsDB[(~/.local/share/lm-studio-bench/results/<br/>benchmark_cache.db)]
    Benchmark -->|exports| Reports[JSON/CSV/PDF/HTML<br/>Reports]
    
    WebApp -->|launches| Benchmark
    WebApp -->|reads| ResultsDB
    WebApp -->|serves| Dashboard[Web Dashboard<br/>http://localhost:PORT]
    RunPy -->|starts background process| Tray[src/tray.py<br/>Linux Tray Controller]
    Tray -->|polls /api/status| WebApp
    Tray -->|calls /api/benchmark/*| WebApp
    Tray -->|Quit calls /api/system/shutdown| WebApp
    
    style RunPy fill:#e1f5ff
    style Benchmark fill:#ffe1e1
    style ConfigLoader fill:#e1ffe1
    style RestClient fill:#fff4e1
    style DefaultsJSON fill:#f0f0f0
    style LMStudio fill:#e8deff
```

**Key Components:**

- **run.py**: Wrapper script that decides between web dashboard and CLI benchmark mode
- **benchmark.py**: Main benchmark engine (~4,683 lines) with argparse, model discovery, and execution
- **config_loader.py**: Loads and merges configuration from JSON file with built-in defaults
- **preset_manager.py**: Manages readonly/user presets and maps presets to CLI args
- **rest_client.py**: REST API client for LM Studio v1 endpoints (optional mode)
- **web/app.py**: FastAPI web dashboard with live streaming and results browser
- **tray.py**: Linux AppIndicator tray controller for benchmark controls

---

## Startup Flow

### AppImage Entry Point

When the AppImage is executed, the bundled `lmstudio-bench` shell script runs
**before** `run.py` and splits on whether real arguments are present:

```mermaid
flowchart TD
    AppImg([LM-Studio-Bench.AppImage args]) --> CheckArgs{Real args<br/>besides --debug/-d?}
    CheckArgs -->|No args| TrayOnly[exec tray.py --url localhost:1234<br/>stays in system tray]
    CheckArgs -->|Any other arg| RunPy[delegate to run.py + args]

    style AppImg fill:#d0e8ff
    style TrayOnly fill:#e1ffe1
    style RunPy fill:#ffe1ff
```

> `--debug` / `-d` is exempt: `./AppImage --debug` still enters tray-only mode
> with verbose logging.

### run.py Flow

```mermaid
flowchart TD
    Start([./run.py args]) --> CheckHelp{--help or -h?}
    CheckHelp -->|Yes| ShowHelp[Show Extended Help<br/>+ benchmark.py --help]
    CheckHelp -->|No| CheckWebFlag{--webapp or -w<br/>in args?}

    CheckWebFlag -->|Yes| RemoveFlag[Remove --webapp/-w<br/>from args]
    RemoveFlag --> ResolvePort[Extract or assign<br/>web port]
    ResolvePort --> StartTrayWeb[start tray.py<br/>with --url dashboard]
    StartTrayWeb --> FindWebApp{web/app.py<br/>exists?}
    FindWebApp -->|Yes| StartWeb[subprocess.call<br/>python web/app.py + args]
    FindWebApp -->|No| ErrorWeb[Error: app.py not found]

    CheckWebFlag -->|No| StartTrayCLI[start tray.py<br/>with localhost:1234]
    StartTrayCLI --> FindBenchmark{src/benchmark.py<br/>exists?}
    FindBenchmark -->|Yes| StartBenchmark[subprocess.call<br/>python src/benchmark.py + args]
    FindBenchmark -->|No| ErrorBench[Error: benchmark.py not found]

    ShowHelp --> Exit1([exit 0])
    StartWeb --> Exit2([exit with app.py status])
    StartBenchmark --> Exit3([exit with benchmark.py status])
    ErrorWeb --> Exit4([exit 1])
    ErrorBench --> Exit5([exit 1])

    style Start fill:#e1f5ff
    style StartWeb fill:#ffe1ff
    style StartBenchmark fill:#ffe1e1
```

**Decision Logic (run.py):**

1. **Help Mode** (`--help`/`-h`): Displays extended help combining run.py explanation + benchmark.py CLI options
2. **Web Mode** (`--webapp`/`-w`): Launches tray + FastAPI dashboard on a free
   local port
3. **Benchmark Mode** (default): Launches tray + benchmark.py with all CLI
   arguments

**AppImage vs. run.py — default behaviour difference:**

| Invocation | No-argument default |
| --- | --- |
| `./LM-Studio-Bench.AppImage` | Tray-only (stays in panel, no benchmark) |
| `./run.py` | Tray + benchmark.py (runs full benchmark) |

---

## Setup Flow (Installation & Configuration)

```mermaid
flowchart TD
    Start([./setup.sh args]) --> ParseArgs{Parse Arguments}
    
    ParseArgs -->|--help| ShowHelp["Show Usage Info<br/>+ Exit 0"]
    ParseArgs -->|--dry-run| DryMode["Set DRY_RUN=1<br/>Set INTERACTIVE=0"]
    ParseArgs -->|--yes| AutoMode["Set INTERACTIVE=0<br/>Auto-answer 'no'"]
    ParseArgs -->|--interactive| InterMode["Set INTERACTIVE=1<br/>Force Interactive"]
    
    DryMode --> LogSetup["Setup Logging<br/>logs/setup_YYYYMMDD_HHMMSS.log"]
    AutoMode --> LogSetup
    InterMode --> LogSetup
    
    LogSetup --> CheckLinux{OS = Linux?}
    CheckLinux -->|No| ErrorOS["❌ Error:<br/>Not Linux"]
    CheckLinux -->|Yes| DetectPKG["✅ Detect Package Manager<br/>apt/dnf/pacman/zypper/apk"]
    
    ErrorOS --> Exit1([Exit 1])
    
    DetectPKG --> CoreDeps["🔧 Check Core Dependencies<br/>Python3, Git, curl, pkg-config"]
    CoreDeps --> SysLibs["📦 Check System Libraries<br/>gobject-introspection, cairo, PyGObject"]
    
    SysLibs --> CheckLMS["🔍 Check LM Studio Stack<br/>lms CLI / llmster-headless"]
    CheckLMS -->|Found| LMSFound["✅ LM Studio/llmster<br/>detected"]
    CheckLMS -->|Not Found| LMSMissing["⚠️ LM Studio missing<br/>Offer download link"]
    
    LMSFound --> GPUDetect["🎮 Detect GPU<br/>lspci → NVIDIA/AMD/Intel"]
    LMSMissing --> GPUDetect
    
    GPUDetect --> GPUTools{GPU Found?}
    GPUTools -->|NVIDIA| NVIDIACheck["Check nvidia-smi<br/>+ Install if needed"]
    GPUTools -->|AMD| AMDCheck["Check rocm-smi<br/>+ AMD Driver Check"]
    GPUTools -->|Intel| IntelCheck["Check intel_gpu_top<br/>+ Install if needed"]
    GPUTools -->|None| NoGPU["⚠️ No GPU detected"]
    
    NVIDIACheck --> CreateVenv["🐍 Create Python venv<br/>python3 -m venv .venv"]
    AMDCheck --> AMDDriver["🔍 Check AMD Drivers<br/>amdgpu, libdrm, ROCm"]
    IntelCheck --> CreateVenv
    NoGPU --> CreateVenv
    AMDDriver --> CreateVenv
    
    CreateVenv -->|venv already exists| RecreatChoice{"Recreate .venv?"}
    CreateVenv -->|New venv| VenvOK["✅ venv created<br/>.venv/"]
    
    RecreatChoice -->|Yes| VenvOK
    RecreatChoice -->|No| UseExisting["Use existing .venv"]
    
    VenvOK --> InstallReqs["📥 Install Requirements<br/>pip install -r requirements.txt"]
    UseExisting --> InstallReqs
    
    InstallReqs --> CheckConflict["Check pip conflicts<br/>pip check"]
    CheckConflict --> Summary["📋 Print Summary<br/>Next steps (activation, run, etc)"]
    
    Summary --> LogExit["📄 Save log file<br/>logs/setup_latest.log → symlink"]
    LogExit --> Exit0([Exit 0])
    
    ShowHelp --> Exit0
    
    style Start fill:#e1f5ff
    style LogSetup fill:#fff4e1
    style DetectPKG fill:#e1ffe1
    style CoreDeps fill:#e1ffe1
    style CreateVenv fill:#ffe1e1
    style InstallReqs fill:#ffe1e1
    style Summary fill:#f0e1ff
    style ErrorOS fill:#ffcccc
    style LMSMissing fill:#fff9e1
```

**Setup Flow Summary:**

1. **Parse Arguments**: Handle `--help`, `--dry-run`, `--yes`, `--interactive` flags
2. **Logging Setup**: Create timestamped log file in `logs/setup_YYYYMMDD_HHMMSS.log`
3. **Environment Checks**:
   - Verify Linux OS
   - Detect package manager (apt/dnf/pacman/zypper/apk)
   - Check core dependencies (Python 3, Git, curl, pkg-config)
   - Verify system libraries (gobject-introspection, cairo, PyGObject for tray support)

4. **LM Studio Stack**:
   - Check for `lms` CLI or `llmster` headless binary
   - Offer download link if missing

5. **GPU & Monitoring Tools**:
   - Detect GPU type via `lspci` (NVIDIA, AMD, Intel)
   - Install/check GPU-specific tools (`nvidia-smi`, `rocm-smi`, `intel_gpu_top`)
   - For AMD: Check drivers, ROCm, libdrm, X.Org AMDGPU driver

6. **Python Environment**:
   - Create virtual environment (`.venv/`)
   - Install Python dependencies from `requirements.txt`
   - Check for pip conflicts

7. **Summary**:
   - Print next steps for user:
     - Activate venv: `source .venv/bin/activate`
     - Run webapp: `python run.py --webapp`
     - Run CLI: `python run.py`
   - Log file symlink: `logs/setup_latest.log`

**Modes:**

| Mode | Behavior |
| ---- | -------- |
| `--help` | Show usage and exit |
| `--dry-run` | Preview all actions (no changes) |
| `--yes` | Non-interactive (auto-answer 'no' to optional prompts) |
| `--interactive` | Force interactive mode (default if TTY detected) |

---

## Tray Control Flow (Linux)

```mermaid
flowchart TD
    TrayStart([tray.py start]) --> Poll[Poll /api/status<br/>every 3 seconds]
    Poll --> Reachable{API reachable?}

    Reachable -->|No| IconRed[Set icon: red<br/>error/unreachable]
    Reachable -->|Yes| ReadStatus[Read status field]

    ReadStatus -->|idle| IconGray[Set icon: gray]
    ReadStatus -->|running| IconGreen[Set icon: green]
    ReadStatus -->|paused| IconYellow[Set icon: yellow]

    ReadStatus --> BtnLogic[Update Start/Pause/Stop states]
    BtnLogic --> UserAction{User action}

    UserAction -->|Start| StartCall[POST /api/benchmark/start]
    UserAction -->|Pause/Resume| PauseCall[POST /api/benchmark/pause or resume]
    UserAction -->|Stop| StopCall[POST /api/benchmark/stop]
    UserAction -->|Quit| QuitCall[POST /api/system/shutdown]

    QuitCall --> ExitTray[GTK main loop exit]
```

**Tray behavior summary:**

- Dynamic status icons: gray (idle), green (running), yellow (paused), red
  (API error/unreachable)
- Smart controls: Start enabled in idle/error, Pause and Stop enabled only in
  running or paused state
- Quit path: Tray triggers graceful shutdown endpoint, then exits

## Tray Quit Sequence (Linux)

```mermaid
sequenceDiagram
    participant U as User
    participant T as Tray (GTK/AppIndicator)
    participant A as web/app.py (FastAPI)
    participant B as Benchmark Manager
    participant P as Process Signal Handler

    U->>T: Click Quit
    T->>A: POST /api/system/shutdown
    A->>B: stop_benchmark()
    B-->>A: benchmark stopped or no-op
    A-->>T: 200 OK (shutdown accepted)
    A->>P: Start delayed SIGTERM thread
    T->>T: Stop polling + GTK main_quit()
    P->>A: Send SIGTERM to process
    A-->>A: Uvicorn graceful shutdown
```

---

## Configuration Loading

```mermaid
flowchart TD
    Start([config_loader.py<br/>import]) --> BaseConfig[BASE_DEFAULT_CONFIG<br/>Hard-coded Defaults]

    BaseConfig --> LoadFunc[load_default_config]
    LoadFunc --> ReadProject[Read config/defaults.json<br/>Project Defaults]
    
    ReadProject --> CheckUser{~/.config/lm-studio-bench/<br/>defaults.json exists?}
    
    CheckUser -->|Yes| ReadUser[Read User Config<br/>Deep Merge]
    CheckUser -->|No| UseProject[Use Project Only]
    CheckFile -->|No| UseBase[Use BASE_DEFAULT_CONFIG]
    
    ReadJSON --> ParseJSON[Parse JSON]
    ParseJSON --> DeepMerge[_deep_merge<br/>Base + User Config]
    
    DeepMerge --> NormalizePorts[_normalize_ports<br/>Ensure valid LM Studio ports]
    UseBase --> NormalizePorts
    
    NormalizePorts --> FinalConfig[(DEFAULT_CONFIG<br/>Global Singleton)]
    
    FinalConfig --> BenchmarkImport[benchmark.py imports<br/>DEFAULT_CONFIG]
    FinalConfig --> WebAppImport[web/app.py imports<br/>DEFAULT_CONFIG]
    
    style BaseConfig fill:#f0f0f0
    style FinalConfig fill:#e1ffe1
    style DeepMerge fill:#fff4e1
```

**Configuration Layers:**

| Layer | Source | Priority |
| ----- | ------ | -------- |
| **1. Hard-coded** | `BASE_DEFAULT_CONFIG` in config_loader.py | Lowest |
| **2. User Config** | `~/.config/lm-studio-bench/defaults.json` | Medium |
| **3. Project Config** | `config/defaults.json` | Low |
| **3. CLI Arguments** | argparse in benchmark.py | Highest |

**Merge Strategy:**

- `_deep_merge()` recursively merges nested dictionaries
- User config values override base config
- `None` values in user config are skipped (base value retained)

---

## Configuration Priority

```mermaid
flowchart LR
    CLI[CLI Arguments<br/>--runs 5<br/>--context 4096] -->|Highest Priority| Merge[Configuration<br/>Merge]

    UserCfg[~/.config/.../defaults.json<br/>context_length: 4096] -->|High Priority| Merge
    ProjCfg[config/defaults.json<br/>num_runs: 3<br/>context_length: 2048] -->|Medium Priority| Merge
    
    Base[BASE_DEFAULT_CONFIG<br/>prompt: default<br/>temperature: 0.1] -->|Lowest Priority| Merge
    
    Merge --> Final[Final Configuration<br/>runs=5<br/>context=4096<br/>temperature=0.1]
    
    style CLI fill:#ffe1e1
    style JSON fill:#fff4e1
    style Base fill:#f0f0f0
    style Final fill:#e1ffe1
```

**Example Priority Resolution:**

```python
# BASE_DEFAULT_CONFIG
{
  "num_runs": 3,
  "context_length": 2048,
  "prompt": "Is the sky blue?"
}

# config/defaults.json
{
  "num_runs": 5,
  "prompt": "Explain machine learning"
}

# CLI: ./run.py --runs 1 --context 4096

# FINAL RESULT:
{
  "num_runs": 1,           # ← CLI override
  "context_length": 4096,  # ← CLI override
  "prompt": "Explain..."   # ← JSON override (no CLI arg)
}
```

---

## Benchmark Execution Flow

```mermaid
flowchart TD
    Start([benchmark.py main]) --> ParseArgs[Parse CLI Arguments<br/>argparse.ArgumentParser]

    ParseArgs --> LoadConfig[Load DEFAULT_CONFIG<br/>from config_loader]
    
    LoadConfig --> CheckFlags{Special Flags?}
    
    CheckFlags -->|--list-cache| ListCache[Display Cache Entries<br/>exit]
    CheckFlags -->|--export-cache| ExportCache[Export Cache to JSON<br/>exit]
    CheckFlags -->|--export-only| ExportOnly[Generate Reports Only<br/>skip benchmark]
    CheckFlags -->|Normal Mode| CreateBenchmark[Create LMStudioBenchmark<br/>instance]
    
    CreateBenchmark --> MergeConfig[Merge Config Layers:<br/>CLI > JSON > Base]
    
    MergeConfig --> InitComponents[Initialize Components:<br/>• GPUMonitor<br/>• BenchmarkCache<br/>• HardwareMonitor<br/>• REST Client optional]
    
    InitComponents --> CheckServer{LM Studio<br/>Server Running?}
    
    CheckServer -->|No| StartServer[Auto-start Server<br/>lms server start]
    CheckServer -->|Yes| DiscoverModels[Discover Models<br/>lms ls --json]
    StartServer --> DiscoverModels
    
    DiscoverModels --> FilterModels[Apply Filters:<br/>--quants, --arch<br/>--only-vision, etc.]
    
    FilterModels --> CheckCache{use_cache<br/>enabled?}
    
    CheckCache -->|Yes| LoadCache[Load Cached Results<br/>SQLite lookup]
    CheckCache -->|No| SkipCache[Skip Cache]
    
    LoadCache --> RunBenchmarks[Run Benchmarks<br/>for Each Model]
    SkipCache --> RunBenchmarks
    
    RunBenchmarks --> TestModel[Test Model:<br/>1. Load Model<br/>2. Warmup Run<br/>3. N Measurement Runs<br/>4. Collect Stats]
    
    TestModel --> Profiling{Profiling<br/>enabled?}
    
    Profiling -->|Yes| MonitorHW[Monitor GPU/CPU/RAM<br/>Background Thread]
    Profiling -->|No| SkipMonitor[Skip Monitoring]
    
    MonitorHW --> SaveCache[Save Results to Cache<br/>SQLite INSERT]
    SkipMonitor --> SaveCache
    
    SaveCache --> NextModel{More Models?}
    
    NextModel -->|Yes| RunBenchmarks
    NextModel -->|No| Export[Export Reports:<br/>JSON, CSV, PDF, HTML]
    
    Export --> End([Done])
    
    ListCache --> End
    ExportCache --> End
    ExportOnly --> Export
    
    style Start fill:#e1f5ff
    style CreateBenchmark fill:#ffe1e1
    style RunBenchmarks fill:#ffe1ff
    style Export fill:#e1ffe1
```

**Key Execution Steps:**

1. **Argument Parsing**: 49 CLI arguments processed by argparse
2. **Configuration Merge**: CLI args override JSON file, JSON overrides base
3. **Component Initialization**: GPU monitor, cache, profiler, REST client
4. **Model Discovery**: `lms ls --json` fetches all installed models
5. **Filtering**: Regex, quantization, architecture, capabilities filters
6. **Cache Lookup**: Skip already-tested models (unless `--retest`)
7. **Benchmark Loop**: For each model: load → warmup → measure (N runs) → unload
8. **Hardware Monitoring**: Optional background thread for GPU/CPU/RAM stats
9. **Cache Storage**: Save results to SQLite for future runs
10. **Report Generation**: Export to JSON/CSV/PDF/HTML

---

## REST API vs SDK Mode

```mermaid
flowchart TD
    Start([Benchmark Init]) --> CheckMode{use_rest_api?<br/>CLI or config}

    CheckMode -->|True| InitREST[Initialize REST Client<br/>LMStudioRESTClient]
    CheckMode -->|False| InitSDK[Use Python SDK<br/>lmstudio package]
    
    InitREST --> RESTURL[base_url from config:<br/>http://localhost:1234]
    RESTURL --> RESTToken{api_token<br/>set?}
    
    RESTToken -->|Yes| RESTAuth[Add Bearer Token<br/>to headers]
    RESTToken -->|No| RESTNoAuth[No Authentication]
    
    RESTAuth --> RESTReady[REST Client Ready]
    RESTNoAuth --> RESTReady
    
    RESTReady --> RESTFeatures[REST API Features:<br/>• Download Progress<br/>• MCP Integration<br/>• Stateful Chat<br/>• Response Caching<br/>• Parallel Inference<br/>• Unified KV Cache]
    
    InitSDK --> SDKReady[SDK Ready]
    SDKReady --> SDKFeatures[SDK Features:<br/>• Simple Python API<br/>• Model Loading<br/>• Inference<br/>• Basic Stats]
    
    RESTFeatures --> Benchmark[Run Benchmarks]
    SDKFeatures --> Benchmark
    
    Benchmark --> RESTCall{Mode?}
    
    RESTCall -->|REST| CallREST[HTTP POST /v1/chat/completions<br/>+ parse response stats]
    RESTCall -->|SDK| CallSDK[client.llm.predict<br/>+ parse Model response]
    
    CallREST --> Results[Collect Results:<br/>TTFT, tokens/s, VRAM]
    CallSDK --> Results
    
    style InitREST fill:#e1f5ff
    style InitSDK fill:#ffe1e1
    style RESTFeatures fill:#e1ffe1
    style SDKFeatures fill:#fff4e1
```

**Mode Comparison:**

| Feature | REST API Mode | SDK/CLI Mode |
| --- | --- | --- |
| **Configuration** | `use_rest_api: true` in config or `--use-rest-api` | Default mode |
| **Endpoint** | HTTP `/v1/chat/completions` | Python SDK `client.llm.predict()` |
| **Stats** | Detailed (TTFT, prompt/completion tokens, tok/s) | Basic (tokens/s only) |
| **Authentication** | Optional Bearer token | Not needed |
| **Parallel Inference** | ✅ `--n-parallel` (continuous batching) | ❌ Sequential only |
| **Stateful Chats** | ✅ response_id tracking | ❌ Stateless |
| **MCP Integration** | ✅ `mcp_integrations` parameter | ❌ Not available |
| **Response Caching** | ✅ MD5 hash caching (10,000x speedup) | ❌ No caching |
| **Download Progress** | ✅ Real-time model loading status | ❌ No progress |

**Configuration Example:**

```json
{
  "lmstudio": {
    "host": "localhost",
    "ports": [1234, 1235],
    "use_rest_api": true,
    "api_token": "lms_your_token_here"
  }
}
```

---

## Component Details

### 1. run.py (Entry Point)

**Responsibilities:**

- Parse `--webapp`/`-w` flag
- Route to web dashboard or benchmark
- Show extended help (`--help`)

**Key Functions:**

- Flag detection: `"--webapp" in sys.argv or "-w" in sys.argv`
- Subprocess launching: `subprocess.call([sys.executable, script] + args)`

---

### 2. config_loader.py (Configuration Manager)

**Responsibilities:**

- Load `config/defaults.json` (project) + `~/.config/lm-studio-bench/defaults.json` (user overrides)
- Merge with `BASE_DEFAULT_CONFIG`
- Provide `DEFAULT_CONFIG` singleton

**Key Functions:**

- `load_default_config()`: Loads and merges config
- `_deep_merge()`: Recursive dict merge
- `_normalize_ports()`: Validates LM Studio ports

**Configuration Fields:**

| Section | Fields |
| --- | --- |
| **Root** | `prompt`, `context_length`, `num_runs` |
| **lmstudio** | `host`, `ports`, `api_token`, `use_rest_api` |
| **inference** | `temperature`, `top_k_sampling`, `top_p_sampling`, `min_p_sampling`, `repeat_penalty`, `max_tokens` |
| **load** | `n_gpu_layers`, `n_batch`, `n_threads`, `flash_attention`, `rope_freq_base`, `rope_freq_scale`, `use_mmap`, `use_mlock`, `kv_cache_quant` |

---

### 3. benchmark.py (Main Engine)

**Responsibilities:**

- Parse 49 CLI arguments
- Manage benchmark lifecycle
- Model discovery and filtering
- Cache management (SQLite)
- Runtime-safe cache schema migration for optional columns
- Hardware monitoring
- Report generation

**Key Classes:**

- `LMStudioBenchmark`: Main orchestrator
- `BenchmarkCache`: SQLite caching
- `GPUMonitor`: GPU detection (NVIDIA/AMD/Intel)
- `HardwareMonitor`: Live profiling (GPU temp, power, VRAM, GTT, CPU, RAM)
- `ModelDiscovery`: Model listing and metadata

**Reliability Behaviors (2026-03):**

- **Runtime cache migration**:
  Missing optional SQLite columns are added automatically at startup and,
  if needed, once again during insert error recovery.
- **Inference retry guard**:
  If LM Studio returns a server error containing `Model unloaded`, the
  benchmark reloads the model and retries inference once.

**CLI Arguments (49 total):**

| Category | Arguments |
| --- | --- |
| **Basic** | `--runs`, `--context`, `--prompt`, `--limit`, `--dev-mode` |
| **Presets** | `--list-presets`, `--preset` |
| **Filter** | `--only-vision`, `--only-tools`, `--quants`, `--arch`, `--params`, `--min-context`, `--max-size`, `--include-models`, `--exclude-models` |
| **Cache** | `--retest`, `--list-cache`, `--export-cache`, `--export-only` |
| **Profiling** | `--enable-profiling`, `--max-temp`, `--max-power`, `--disable-gtt` |
| **Inference** | `--temperature`, `--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--max-tokens` |
| **Load Config** | `--n-gpu-layers`, `--n-batch`, `--n-threads`, `--flash-attention`, `--rope-freq-base`, `--rope-freq-scale`, `--use-mmap`, `--use-mlock`, `--kv-cache-quant` |
| **REST API** | `--use-rest-api`, `--api-token`, `--n-parallel`, `--unified-kv-cache` |
| **Comparison** | `--compare-with`, `--rank-by` |

---

### 4. rest_client.py (REST API Client)

**Responsibilities:**

- HTTP communication with LM Studio v1 API
- Model loading and unloading
- Chat completions with stats
- Download progress tracking
- MCP integration
- Stateful chat history
- Response caching

**Key Classes:**

- `LMStudioRESTClient`: Main REST client
- `ModelInfo`: Model metadata
- `ChatStats`: Response statistics (TTFT, tokens/s, etc.)
- `ModelCapabilities`: Vision, tools detection

**New Features (✨ 2026-02-23):**

1. **Download Progress Tracking**
   - `wait_for_completion()` with progress callbacks
   - Real-time model loading status

2. **MCP Integration**
   - `mcp_integrations` parameter in chat requests
   - Model Context Protocol support

3. **Stateful Chat History**
   - `use_stateful=True` for conversation continuity
   - `last_response_id` tracking

4. **Response Caching**
   - MD5 hash-based caching
   - 10,000x+ speedup for repeated prompts
   - `enable_cache` parameter

**Example Usage:**

```python
client = LMStudioRESTClient(
    base_url="http://localhost:1234",
    api_token="lms_token"
)

# Load model with progress tracking
def on_progress(percent, status):
    print(f"Loading: {percent:.1f}% - {status}")

client.load_model("model@q4", wait_for_completion=True, progress_callback=on_progress)

# Chat with caching
response = client.chat(
    model="model@q4",
    messages=[{"role": "user", "content": "Hello"}],
    enable_cache=True,  # 10,000x speedup for repeated prompts
    use_stateful=True   # Conversation continuity
)
```

---

### 5. tray.py (Linux Tray Controller)

**Responsibilities:**

- Provide Linux AppIndicator tray UI with benchmark controls
- Poll benchmark status and update icon/button state
- Trigger benchmark actions via web API
- Trigger graceful full shutdown via `/api/system/shutdown`

**Key Behaviors:**

- 3-second polling loop via GLib timeout
- Icon states: gray (idle), green (running), yellow (paused), red (error)
- Control state logic:
  - Start enabled in idle and recovery/error state
  - Pause/Stop enabled only while benchmark is active

---

### 6. web/app.py + dashboard.html.jinja (Dashboard Analytics)

**Responsibilities:**

- Aggregate benchmark history for fast visual summaries
- Serve chart-ready payloads via `/api/dashboard/stats`
- Render Home/Results overview charts in the browser with Plotly
- Support quick navigation from ranking tables to model comparison

**Home View (Executive Summary):**

- KPI cards: cached models, avg speed, median (P50), P95, architectures,
  quantizations
- Top 10 bar chart (speed ranking)
- Quantization donut chart (distribution)

**Results View (Exploration):**

- Scatter: `Speed vs VRAM`
- Heatmap: `Model x Quantization -> avg tokens/s`
- Shared data source with table (`/api/results`), so table and charts stay
  consistent

**Quick Compare Flow:**

- Compare actions in Home and Results tables call
  `openComparisonForModel(modelName)`
- Function opens Comparison view, selects the model, then loads full
  historical trends via `/api/comparison/{model_name}`

**Dashboard Summary Fields (`/api/dashboard/stats`):**

- `speed_summary` (`min`, `p50`, `avg`, `p95`, `max`)
- `top_models_extended` (Top 10 models)
- `quantization_distribution`
- `architecture_distribution`
- `efficiency_top`

---

## Data Flow Summary

```mermaid
graph LR
    User([User]) -->|./run.py --runs 5| CLI[CLI Arguments]

    ProjJSON[config/defaults.json] --> Config[Configuration<br/>Merge]
    UserJSON[~/.config/.../defaults.json] --> Config
    CLI --> Config
    Base[BASE_DEFAULT_CONFIG] --> Config
    
    Config --> Benchmark[Benchmark<br/>Execution]
    
    Benchmark -->|lms ls| Models[Model<br/>Discovery]
    Models --> Filter[Model<br/>Filtering]
    
    Filter --> Cache{Cache<br/>Hit?}
    Cache -->|Yes| Skip[Skip Test]
    Cache -->|No| Test[Run Test]
    
    Test --> LMStudio[LM Studio<br/>Server]
    LMStudio --> Results[Collect<br/>Results]
    
    Results --> DB[(SQLite<br/>Cache)]
    Results --> Reports[JSON/CSV<br/>PDF/HTML]
    
    Skip --> Reports
    
    style CLI fill:#ffe1e1
    style Config fill:#e1ffe1
    style Cache fill:#fff4e1
    style Reports fill:#e1f5ff
```

---

## Testing Architecture

LM-Studio-Bench includes a comprehensive test suite with 520+ tests and 51% code coverage to ensure reliability and maintainability.

### Test Organization

```mermaid
graph TB
    Tests[tests/] --> Fixtures[conftest.py<br/>Test Fixtures & Utilities]

    Tests --> BenchmarkTests[test_benchmark.py<br/>55+ tests]
    Tests --> AppTests[test_app.py<br/>23+ tests]
    Tests --> APITests[test_api_endpoints.py<br/>32+ tests]
    Tests --> RestTests[test_rest_client.py<br/>22+ tests]
    Tests --> TrayTests[test_tray.py<br/>26+ tests]
    Tests --> PresetTests[test_preset_manager.py<br/>19+ tests]
    Tests --> ConfigTests[test_config_loader.py<br/>9+ tests]
    Tests --> PathTests[test_user_paths.py<br/>4+ tests]
    Tests --> VersionTests[test_version_checker.py<br/>7+ tests]
    Tests --> MetadataTests[test_scrape_metadata.py<br/>24+ tests]
    Tests --> RunTests[test_run.py<br/>10+ tests]

    BenchmarkTests --> Benchmark[src/benchmark.py]
    AppTests --> WebApp[web/app.py]
    APITests --> WebApp
    RestTests --> RestClient[src/rest_client.py]
    TrayTests --> Tray[src/tray.py]
    PresetTests --> PresetMgr[src/preset_manager.py]
    ConfigTests --> ConfigLoader[src/config_loader.py]
    PathTests --> UserPaths[src/user_paths.py]
    VersionTests --> VersionChecker[src/version_checker.py]
    MetadataTests --> Metadata[tools/scrape_metadata.py]
    RunTests --> RunPy[run.py]

    style Tests fill:#e1f5ff
    style Fixtures fill:#fff4e1
    style BenchmarkTests fill:#ffe1e1
    style AppTests fill:#e1ffe1
```

### Test Coverage by Component

| Component | Test Module | Test Count | Coverage |
| --------- | ----------- | ---------- | -------- |
| Benchmark Engine | `test_benchmark.py` | 55+ | High |
| Web Dashboard | `test_app.py` | 23+ | Medium |
| API Endpoints | `test_api_endpoints.py` | 32+ | High |
| REST Client | `test_rest_client.py` | 22+ | High |
| Linux Tray | `test_tray.py` | 26+ | Medium |
| Preset Manager | `test_preset_manager.py` | 19+ | High |
| Config Loader | `test_config_loader.py` | 9+ | High |
| User Paths | `test_user_paths.py` | 4+ | High |
| Version Checker | `test_version_checker.py` | 7+ | High |
| Metadata Scraping | `test_scrape_metadata.py` | 24+ | Medium |
| Entry Point | `test_run.py` | 10+ | Medium |

### Testing Approach

**Unit Testing:**

- Mock external dependencies (LM Studio API, system commands, file I/O)
- Isolated test cases that can run in any order
- Fast execution (no real API calls or file system operations)
- Use pytest fixtures for common setup and teardown

**Test Fixtures (`conftest.py`):**

- Mock LM Studio client and server responses
- Temporary directories for file operations
- Mock system commands (nvidia-smi, rocm-smi, etc.)
- Sample configuration and model data

**Continuous Integration:**

- GitHub Actions runs full test suite on every PR
- Code quality checks (flake8, pylint)
- Security scans (Bandit, CodeQL, Snyk)
- Test results reported in PR status checks

**Running Tests:**

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific module
pytest tests/test_benchmark.py

# Run with coverage report
pytest --cov=src --cov=web --cov-report=html

# Run tests matching a pattern
pytest -k "test_gpu_detection"
```

---

## See Also

- [Configuration Reference](CONFIGURATION.md) - All CLI arguments and config file options
- [REST API Features](REST_API_FEATURES.md) - REST API integration details
- [Quickstart Guide](QUICKSTART.md) - Get started in 5 minutes
