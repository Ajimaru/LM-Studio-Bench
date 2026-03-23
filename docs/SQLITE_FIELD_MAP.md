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

| Metric | benchmark_results (classic) | benchmark_results (compatibility) | Stored in both tests |
| --- | --- | --- | --- |
| Row id | `id` | `id` | `[x]` |
| Model name | `model_name` | `model_name` | `[x]` |
| Timestamp | `timestamp` | `timestamp` | `[x]` |
| Model path/source | `model_key` | `model_key` | `[x]` |
| Capability label | `capability` | `capability` | `[x]` |
| Test case id | `test_id` | `test_id` | `[x]` |
| Test case name | `test_name` | `test_name` | `[x]` |
| Quantization | `quantization` | `quantization` | `[x]` |
| Inference params hash | `inference_params_hash` | `inference_params_hash` | `[x]` |
| Tokens per second | `avg_tokens_per_sec` | `avg_tokens_per_sec` | `[x]` |
| Latency | `avg_gen_time` | `avg_gen_time` | `[x]` |
| TTFT | `avg_ttft` | `avg_ttft` | `[x]` |
| Prompt token count | `prompt_tokens` | `prompt_tokens` | `[x]` |
| Completion/generated tokens | `completion_tokens` | `tokens_generated` | `[x]` |
| Primary quality score | `quality_score` | `quality_score` | `[x]` |
| ROUGE | `rouge_score` | `rouge_score` | `[x]` |
| F1 | `f1_score` | `f1_score` | `[x]` |
| Exact match | `exact_match_score` | `exact_match_score` | `[x]` |
| Accuracy | `accuracy_score` | `accuracy_score` | `[x]` |
| Function-call accuracy | `function_call_accuracy` | `function_call_accuracy` | `[x]` |
| Success flag | `success` | `success` | `[x]` |
| Error message | `error_message` | `error_message` | `[x]` |
| Error counter | `error_count` | `error_count` | `[x]` |
| Total tests per capability | `-` | aggregate `COUNT(*)` by capability | `[ ]` |
| Successful tests per capability | `-` | aggregate `SUM(success = 1)` | `[ ]` |
| Failed tests per capability | `-` | aggregate `SUM(success != 1)` | `[ ]` |
| Success rate per capability | `-` | derived aggregate (`successful / total`) | `[ ]` |
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

-- Compatibility rows that still miss core parity fields.
SELECT id, model_name, capability, test_id,
         quantization, lmstudio_version, app_version,
         prompt_hash, params_hash
FROM benchmark_results
WHERE source = 'compatibility'
        AND (
            quantization IS NULL
            OR lmstudio_version IS NULL
            OR app_version IS NULL
            OR prompt_hash IS NULL
            OR params_hash IS NULL
        )
ORDER BY id DESC;

-- Compatibility summary directly from benchmark_results.
SELECT model_name,
             capability,
             COUNT(*) AS total_tests,
             SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) AS successful_tests,
             SUM(CASE WHEN success = 1 THEN 0 ELSE 1 END) AS failed_tests,
             AVG(avg_gen_time) AS avg_latency_ms,
             AVG(avg_tokens_per_sec) AS avg_throughput,
             AVG(quality_score) AS avg_quality_score,
             AVG(rouge_score) AS avg_rouge,
             AVG(f1_score) AS avg_f1,
             AVG(exact_match_score) AS avg_exact_match,
             AVG(accuracy_score) AS avg_accuracy
FROM benchmark_results
WHERE source = 'compatibility'
GROUP BY model_name, capability
ORDER BY MAX(id) DESC;
```
