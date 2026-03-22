"""Tests for embedding-model skip behavior in BenchmarkRunner."""

from pathlib import Path

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
