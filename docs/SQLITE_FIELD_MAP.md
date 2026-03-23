# SQLite Metric Parity Map

This table is intentionally compact: one metric per row.

Legend:

- `[x]` = metric is stored in both test modes
- `[ ]` = metric is missing in at least one mode

Notes:

- Capability rows normalize quantization to an uppercase label such as
    `Q4_K_M`; classic rows keep the classic benchmark format such as
    `q4_k_m`.

- Capability `lmstudio_version` stores a parsed version or
    `pkg_version (commit:<sha>)`, not the raw `lms version` banner output.

- Capability REST runs forward the exact model variant key, including the
    `@quantization` suffix, to LM Studio load/chat/unload requests.

- Classic rows intentionally leave capability-only fields such as
    `quality_score`, `raw_output`, `reference_output`, `capability`, and
    `test_id` empty.

- Historical rows created before recent schema/runtime fixes may still
  contain `NULL` values in parity columns. New rows should populate them.

| Metric | benchmark_results column | agent_results column | Stored in both tests |
| --- | --- | --- | --- |
| Row id | `id` | `id` | `[x]` |
| Model name | `model_name` | `model_name` | `[x]` |
| Timestamp | `timestamp` | `timestamp` | `[x]` |
| Model path/source | `model_key` | `model_path` | `[x]` |
| Capability label | `capability` | `capability` | `[x]` |
| Test case id | `test_id` | `test_id` | `[x]` |
| Test case name | `test_name` | `test_name` | `[x]` |
| Quantization | `quantization` | `quantization` | `[x]` |
| Inference params hash | `inference_params_hash` | `inference_params_hash` | `[x]` |
| Tokens per second | `avg_tokens_per_sec` | `throughput_tokens_per_sec` | `[x]` |
| Latency | `avg_gen_time` | `latency_ms`, `agent_summaries.avg_latency_ms` | `[x]` |
| TTFT | `avg_ttft` | `avg_ttft` | `[x]` |
| Prompt token count | `prompt_tokens` | `prompt_tokens` | `[x]` |
| Completion/generated tokens | `completion_tokens` | `tokens_generated` | `[x]` |
| Primary quality score | `quality_score` | `quality_score`, `agent_summaries.avg_quality_score` | `[x]` |
| ROUGE | `rouge_score` | `rouge_score`, `agent_summaries.avg_rouge` | `[x]` |
| F1 | `f1_score` | `f1_score`, `agent_summaries.avg_f1` | `[x]` |
| Exact match | `exact_match_score` | `exact_match_score`, `agent_summaries.avg_exact_match` | `[x]` |
| Accuracy | `accuracy_score` | `accuracy_score`, `agent_summaries.avg_accuracy` | `[x]` |
| Function-call accuracy | `function_call_accuracy` | `function_call_accuracy` | `[x]` |
| Success flag | `success` | `success` | `[x]` |
| Error message | `error_message` | `error_message` | `[x]` |
| Error counter | `error_count` | `error_count` | `[x]` |
| Total tests per capability | `-` | `agent_summaries.total_tests` | `[ ]` |
| Successful tests per capability | `-` | `agent_summaries.successful_tests` | `[ ]` |
| Failed tests per capability | `-` | `agent_summaries.failed_tests` | `[ ]` |
| Success rate per capability | `-` | `agent_summaries.success_rate` | `[ ]` |
| GPU type | `gpu_type` | `gpu_type` | `[x]` |
| GPU offload ratio | `gpu_offload` | `gpu_offload` | `[x]` |
| VRAM (MB) | `vram_mb` | `vram_mb` | `[x]` |
| Temperature stats | `temp_celsius_min/max/avg` | `temp_celsius_min/max/avg` | `[x]` |
| Power stats | `power_watts_min/max/avg` | `power_watts_min/max/avg` | `[x]` |
| VRAM GB stats | `vram_gb_min/max/avg` | `vram_gb_min/max/avg` | `[x]` |
| GTT GB stats | `gtt_gb_min/max/avg` | `gtt_gb_min/max/avg` | `[x]` |
| CPU usage stats | `cpu_percent_min/max/avg` | `cpu_percent_min/max/avg` | `[x]` |
| RAM GB stats | `ram_gb_min/max/avg` | `ram_gb_min/max/avg` | `[x]` |
| Context length | `context_length` | `context_length` | `[x]` |
| Temperature sampling param | `temperature` | `temperature` | `[x]` |
| Top-K sampling param | `top_k_sampling` | `top_k_sampling` | `[x]` |
| Top-P sampling param | `top_p_sampling` | `top_p_sampling` | `[x]` |
| Min-P sampling param | `min_p_sampling` | `min_p_sampling` | `[x]` |
| Repeat penalty | `repeat_penalty` | `repeat_penalty` | `[x]` |
| Max tokens param | `max_tokens` | `max_tokens` | `[x]` |
| GPU layer setting | `n_gpu_layers` | `n_gpu_layers` | `[x]` |
| Batch setting | `n_batch` | `n_batch` | `[x]` |
| Thread setting | `n_threads` | `n_threads` | `[x]` |
| Flash attention setting | `flash_attention` | `flash_attention` | `[x]` |
| RoPE base setting | `rope_freq_base` | `rope_freq_base` | `[x]` |
| RoPE scale setting | `rope_freq_scale` | `rope_freq_scale` | `[x]` |
| mmap setting | `use_mmap` | `use_mmap` | `[x]` |
| mlock setting | `use_mlock` | `use_mlock` | `[x]` |
| KV cache quant setting | `kv_cache_quant` | `kv_cache_quant` | `[x]` |
| LM Studio version | `lmstudio_version` | `lmstudio_version` | `[x]` |
| App version | `app_version` | `app_version` | `[x]` |
| Driver versions | `nvidia/rocm/intel_driver_version` | `nvidia/rocm/intel_driver_version` | `[x]` |
| OS info | `os_name`, `os_version` | `os_name`, `os_version` | `[x]` |
| CPU model | `cpu_model` | `cpu_model` | `[x]` |
| Python version | `python_version` | `python_version` | `[x]` |
| Benchmark duration | `benchmark_duration_seconds` | `benchmark_duration_seconds` | `[x]` |
| Raw model output | `raw_output` | `raw_output` | `[x]` |
| Reference output | `reference_output` | `reference_output` | `[x]` |
| Efficiency per GB | `tokens_per_sec_per_gb` | `tokens_per_sec_per_gb` | `[x]` |
| Efficiency per B params | `tokens_per_sec_per_billion_params` | `tokens_per_sec_per_billion_params` | `[x]` |
| Speed delta vs previous | `speed_delta_pct` | `speed_delta_pct` | `[x]` |
| Previous timestamp link | `prev_timestamp` | `prev_timestamp` | `[x]` |
| Prompt hash | `prompt_hash` | `prompt_hash` | `[x]` |
| Full params hash | `params_hash` | `params_hash` | `[x]` |
| Prompt text | `prompt` | `prompt` | `[x]` |

## Historical Validation Queries

Use these queries to find older rows that predate parity fixes.

```sql
-- Classic rows that still miss parity fields introduced later.
SELECT id, model_name, timestamp,
         quantization, lmstudio_version, app_version, success
FROM benchmark_results
WHERE quantization IS NULL
    OR lmstudio_version IS NULL
    OR app_version IS NULL
    OR success IS NULL
ORDER BY id DESC;

-- Capability rows that still miss core parity fields.
SELECT id, model_name, capability, test_id,
         quantization, lmstudio_version, app_version,
         prompt_hash, params_hash
FROM agent_results
WHERE quantization IS NULL
    OR lmstudio_version IS NULL
    OR app_version IS NULL
    OR prompt_hash IS NULL
    OR params_hash IS NULL
ORDER BY id DESC;

-- Summary/detail consistency for capability runs.
WITH detail AS (
     SELECT model_name,
              capability,
              COUNT(*) AS detail_tests,
              AVG(latency_ms) AS detail_avg_latency,
              AVG(quality_score) AS detail_avg_quality
     FROM agent_results
     GROUP BY model_name, capability
)
SELECT s.model_name,
         s.capability,
         s.total_tests,
         d.detail_tests,
         ROUND(s.avg_latency_ms, 3) AS summary_latency,
         ROUND(d.detail_avg_latency, 3) AS detail_latency,
         ROUND(s.avg_quality_score, 6) AS summary_quality,
         ROUND(d.detail_avg_quality, 6) AS detail_quality
FROM agent_summaries AS s
JOIN detail AS d
  ON s.model_name = d.model_name
 AND s.capability = d.capability
WHERE s.total_tests != d.detail_tests
    OR ABS(COALESCE(s.avg_latency_ms, 0) - COALESCE(d.detail_avg_latency, 0)) > 0.001
    OR ABS(COALESCE(s.avg_quality_score, 0) - COALESCE(d.detail_avg_quality, 0)) > 0.000001
ORDER BY s.id DESC;
```
