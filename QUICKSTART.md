# 🚀 Quick Start Guide - LM Studio Benchmark Tool

## Installation

```bash
cd /home/robby/Temp/local-llm-bench
pip install -r requirements.txt
```

## Basic Usage

### Simple Benchmark (All Models)
```bash
python benchmark.py
```
✅ Tests all installed models with 3 runs each (~1-2 hours)

### Quick Test (3 Models)
```bash
python benchmark.py --limit 3 --runs 1
```
✅ Fast test with 3 random models (~5-10 minutes)

### Test Single Model
```bash
python benchmark.py --limit 1 --runs 1
```
✅ Single model benchmark (~1-2 minutes)

## Advanced Features

### 1️⃣ Filtering Models

**By Quantization:**
```bash
python benchmark.py --quants q4,q5 --limit 5
```

**By Architecture:**
```bash
python benchmark.py --arch llama,mistral --limit 5
```

**By Parameter Size:**
```bash
python benchmark.py --params 7B,8B --limit 5
```

**By Context Length:**
```bash
python benchmark.py --min-context 32000 --limit 3
```

**By Model Size:**
```bash
python benchmark.py --max-size 10 --limit 5
```

**Vision Models Only:**
```bash
python benchmark.py --only-vision --runs 1
```

### 2️⃣ Ranking & Sorting

**Sort by Efficiency (Default: Speed):**
```bash
python benchmark.py --limit 5 --rank-by efficiency
```

**Sort by TTFT (Lower = Better):**
```bash
python benchmark.py --limit 5 --rank-by ttft
```

**Sort by VRAM Usage (Lower = Better):**
```bash
python benchmark.py --limit 5 --rank-by vram
```

### 3️⃣ Historical Comparison & Trends

**Compare with Latest Benchmark:**
```bash
python benchmark.py --limit 3 --runs 1 --compare-with latest
```
📊 Shows performance delta (%) vs previous run

**Compare with Specific Benchmark:**
```bash
python benchmark.py --limit 3 --runs 1 --compare-with benchmark_results_20260104_170000.json
```

### 4️⃣ Custom Configuration

**Adjust Number of Runs:**
```bash
python benchmark.py --runs 5 --limit 2
```

**Custom Context Length:**
```bash
python benchmark.py --context 4096 --limit 2 --runs 1
```

**Custom Prompt:**
```bash
python benchmark.py --prompt "Your custom prompt here" --limit 2 --runs 1
```

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
python benchmark.py \
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
|--------|-------------|------|
| Speed | Throughput | tokens/s |
| Efficiency | Speed per GB model size | tokens/s/GB |
| TTFT | Time to First Token | ms |
| Delta | Change vs previous | % |
| VRAM | Memory used | MB |

## 📁 File Structure

```
results/
├── benchmark_results_20260104_170000.json
├── benchmark_results_20260104_170000.csv
├── benchmark_results_20260104_170000.pdf
└── benchmark_results_20260104_170000.html
```

## 🐛 Troubleshooting

**"No models found"**
- Ensure LM Studio is installed and running
- Check `lms ls --json` output

**"Server not responding"**
- Start LM Studio server manually
- Check `~/.lmstudio/server-logs/`

**"Permission denied on results/"**
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
