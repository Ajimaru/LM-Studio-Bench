"""Tests for embedding-model skip behavior in BenchmarkRunner."""

from pathlib import Path
from unittest.mock import MagicMock

from agents.runner import BenchmarkRunner


def test_run_skips_embedding_model(tmp_path: Path) -> None:
    """Embedding models are skipped before chat benchmark execution."""
    output_dir = tmp_path / "out"
    config = {
        "data_dir": str(tmp_path / "data"),
        "prompts_dir": str(tmp_path / "prompts"),
    }

    runner = BenchmarkRunner(config=config, output_dir=output_dir)

    report = runner.run(
        model_path="text-embedding-nomic-embed-text-v1.5@q4_k_m",
        model_name="text-embedding-nomic-embed-text-v1.5@q4_k_m",
    )

    assert report["skipped_reason"] == "embedding_model_not_chat_capable"
    assert report["results"] == []
    assert report["summary"]["total_tests"] == 0
    assert report["capabilities"] == []


def test_cache_results_saves_detail_metrics_and_real_summary() -> None:
    """Cache writes per-test metrics and computed capability averages."""
    runner = object.__new__(BenchmarkRunner)
    runner.cache = MagicMock()

    report_dict = {
        "results": [
            {
                "test_id": "qa_001",
                "test_name": "qa_001",
                "capability": "general_text",
                "latency_ms": 1000.0,
                "throughput": 20.0,
                "tokens_generated": 20,
                "quality_score": 0.8,
                "rouge_score": 0.6,
                "f1_score": 0.7,
                "exact_match_score": None,
                "accuracy_score": 0.9,
                "function_call_accuracy": None,
                "raw_output": "answer-a",
                "reference_output": "ref-a",
                "error": None,
            },
            {
                "test_id": "qa_002",
                "test_name": "qa_002",
                "capability": "general_text",
                "latency_ms": 3000.0,
                "throughput": 10.0,
                "tokens_generated": 30,
                "quality_score": 0.6,
                "rouge_score": 0.4,
                "f1_score": 0.5,
                "exact_match_score": None,
                "accuracy_score": 0.7,
                "function_call_accuracy": None,
                "raw_output": "answer-b",
                "reference_output": "ref-b",
                "error": None,
            },
        ],
        "summary": {
            "by_capability": {
                "general_text": {
                    "test_count": 2,
                    "success_rate": 1.0,
                    "avg_quality_score": 0.7,
                }
            }
        },
    }

    runner._cache_results(
        report_dict=report_dict,
        model_name="test-model",
        model_path="test-model-path",
        classic_metrics={
            "gpu_type": "NVIDIA",
            "gpu_offload": 1.0,
            "vram_mb": "1024",
            "context_length": 2048,
            "temperature": 0.1,
            "top_k_sampling": 40,
            "top_p_sampling": 0.9,
            "min_p_sampling": 0.05,
            "repeat_penalty": 1.2,
            "max_tokens": 256,
            "benchmark_duration_seconds": 12.5,
            "lmstudio_version": "v1.0.0",
            "app_version": "0.0.1",
            "python_version": "3.11.0",
        },
    )

    assert runner.cache.save_test_result.call_count == 2
    first_call = runner.cache.save_test_result.call_args_list[0].kwargs
    assert first_call["rouge_score"] == 0.6
    assert first_call["f1_score"] == 0.7
    assert first_call["raw_output"] == "answer-a"
    assert first_call["reference_output"] == "ref-a"
    assert first_call["gpu_type"] == "NVIDIA"
    assert first_call["context_length"] == 2048
    assert first_call["top_k_sampling"] == 40
    assert first_call["benchmark_duration_seconds"] == 12.5

    summary_call = runner.cache.save_summary.call_args.kwargs
    assert summary_call["avg_latency_ms"] == 2000.0
    assert summary_call["avg_throughput"] == 15.0
    assert summary_call["avg_rouge"] == 0.5
    assert summary_call["avg_f1"] == 0.6
    assert summary_call["avg_accuracy"] == 0.8
