"""Tests for agents/cache.py - agent result caching."""

import sqlite3
from unittest.mock import patch

from agents.cache import AgentCache


class TestAgentCache:
    """Tests for AgentCache class."""

    def test_initializer_creates_database(self, tmp_path):
        """AgentCache creates SQLite database."""
        db_file = tmp_path / "cache.db"
        cache = AgentCache(db_path=db_file)

        assert cache.db_path == db_file
        assert db_file.exists()

    def test_initializer_creates_classic_metric_columns(self, tmp_path):
        """agent_results contains classic benchmark metric columns."""
        db_file = tmp_path / "cache.db"
        AgentCache(db_path=db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(agent_results)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "error_count",
            "gpu_type",
            "gpu_offload",
            "vram_mb",
            "temp_celsius_min",
            "power_watts_avg",
            "context_length",
            "top_k_sampling",
            "max_tokens",
            "n_gpu_layers",
            "kv_cache_quant",
            "lmstudio_version",
            "app_version",
            "nvidia_driver_version",
            "rocm_driver_version",
            "intel_driver_version",
            "os_name",
            "os_version",
            "cpu_model",
            "python_version",
            "benchmark_duration_seconds",
        }
        assert expected.issubset(columns)

    def test_initializer_default_path(self, tmp_path):
        """AgentCache uses default path when none provided."""
        with patch("agents.cache.USER_RESULTS_DIR", str(tmp_path)):
            cache = AgentCache()
            assert cache.db_path is not None

    def test_save_test_result_returns_true(self, tmp_path):
        """save_test_result returns True on success."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        success = cache.save_test_result(
            model_name="test-model",
            model_path="/path/to/model",
            capability="general_text",
            test_id="test_1",
            test_name="Basic QA",
            latency_ms=100.0,
            tokens_generated=50,
            throughput_tokens_per_sec=25.0,
            quality_score=0.85,
        )

        assert success is True

    def test_save_summary_returns_true(self, tmp_path):
        """save_summary returns True on success."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        success = cache.save_summary(
            model_name="test-model",
            model_path="/path/to/model",
            capability="general_text",
            total_tests=10,
            successful_tests=9,
            failed_tests=1,
            success_rate=0.9,
            avg_latency_ms=100.0,
            avg_throughput=25.0,
            avg_quality_score=0.85,
        )

        assert success is True

    def test_save_test_result_with_all_fields(self, tmp_path):
        """save_test_result accepts all optional fields."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        success = cache.save_test_result(
            model_name="model",
            model_path="/path",
            capability="vision",
            test_id="v1",
            test_name="Image captioning",
            latency_ms=200.0,
            tokens_generated=100,
            throughput_tokens_per_sec=50.0,
            quality_score=0.9,
            rouge_score=0.85,
            f1_score=0.88,
            exact_match_score=0.95,
            accuracy_score=0.92,
            function_call_accuracy=1.0,
            raw_output="Generated caption",
            reference_output="Expected caption",
            error_message=None,
            success=True,
        )

        assert success is True

    def test_get_model_results_returns_empty_for_new_model(self, tmp_path):
        """get_model_results returns empty list for new model."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        results = cache.get_model_results("nonexistent-model")

        assert results == []

    def test_get_model_results_returns_saved_result(self, tmp_path):
        """get_model_results returns previously saved results."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        cache.save_test_result(
            model_name="test-model",
            model_path="/path",
            capability="general_text",
            test_id="test_1",
            test_name="QA",
            latency_ms=100.0,
            tokens_generated=50,
            throughput_tokens_per_sec=25.0,
            quality_score=0.85,
        )

        results = cache.get_model_results("test-model")

        assert len(results) > 0

    def test_get_model_results_with_capability_filter(self, tmp_path):
        """Capability filter returns only matching rows."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        cache.save_test_result(
            model_name="test-model",
            model_path="/path",
            capability="general_text",
            test_id="test_1",
            test_name="QA",
            latency_ms=100.0,
            tokens_generated=50,
            throughput_tokens_per_sec=25.0,
            quality_score=0.85,
        )
        cache.save_test_result(
            model_name="test-model",
            model_path="/path",
            capability="vision",
            test_id="test_2",
            test_name="VQA",
            latency_ms=110.0,
            tokens_generated=30,
            throughput_tokens_per_sec=20.0,
            quality_score=0.82,
        )

        results = cache.get_model_results("test-model", capability="vision")

        assert len(results) >= 1
        assert all(row["capability"] == "vision" for row in results)

    def test_get_latest_summary_returns_none_when_missing(self, tmp_path):
        """get_latest_summary returns None for missing summary."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        result = cache.get_latest_summary("missing-model", "general_text")

        assert result is None

    def test_get_latest_summary_returns_saved_summary(self, tmp_path):
        """get_latest_summary returns dictionary for saved summary."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        saved = cache.save_summary(
            model_name="test-model",
            model_path="/path/to/model",
            capability="general_text",
            total_tests=10,
            successful_tests=8,
            failed_tests=2,
            success_rate=0.8,
            avg_latency_ms=120.0,
            avg_throughput=30.0,
            avg_quality_score=0.81,
        )
        result = cache.get_latest_summary("test-model", "general_text")

        assert saved is True
        assert result is not None
        assert result["model_name"] == "test-model"
        assert result["capability"] == "general_text"

    def test_save_test_result_returns_false_on_db_error(self, tmp_path):
        """save_test_result returns False when sqlite connect fails."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        with patch(
            "agents.cache.sqlite3.connect",
            side_effect=sqlite3.Error("db down"),
        ):
            success = cache.save_test_result(
                model_name="test-model",
                model_path="/path/to/model",
                capability="general_text",
                test_id="test_1",
                test_name="Basic QA",
                latency_ms=100.0,
                tokens_generated=50,
                throughput_tokens_per_sec=25.0,
                quality_score=0.85,
            )

        assert success is False

    def test_save_summary_returns_false_on_db_error(self, tmp_path):
        """save_summary returns False when sqlite connect fails."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        with patch(
            "agents.cache.sqlite3.connect",
            side_effect=sqlite3.Error("db down"),
        ):
            success = cache.save_summary(
                model_name="test-model",
                model_path="/path/to/model",
                capability="general_text",
                total_tests=10,
                successful_tests=9,
                failed_tests=1,
                success_rate=0.9,
                avg_latency_ms=100.0,
                avg_throughput=25.0,
                avg_quality_score=0.85,
            )

        assert success is False

    def test_get_model_results_returns_empty_on_db_error(self, tmp_path):
        """get_model_results returns empty list on sqlite error."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        with patch(
            "agents.cache.sqlite3.connect",
            side_effect=sqlite3.Error("db down"),
        ):
            result = cache.get_model_results("test-model", capability="vision")

        assert result == []

    def test_get_latest_summary_returns_none_on_db_error(self, tmp_path):
        """get_latest_summary returns None on sqlite error."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        with patch(
            "agents.cache.sqlite3.connect",
            side_effect=sqlite3.Error("db down"),
        ):
            result = cache.get_latest_summary("test-model", "general_text")

        assert result is None
