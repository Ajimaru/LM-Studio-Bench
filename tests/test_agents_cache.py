"""Tests for agents/cache.py unified benchmark_results storage."""

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

    def test_initializer_creates_benchmark_results_source_column(self, tmp_path):
        """benchmark_results contains source and compatibility fields."""
        db_file = tmp_path / "cache.db"
        AgentCache(db_path=db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(benchmark_results)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "model_key",
            "model_name",
            "capability",
            "test_id",
            "quality_score",
            "avg_tokens_per_sec",
            "avg_gen_time",
            "source",
        }
        assert expected.issubset(columns)

    def test_initializer_migrates_and_drops_legacy_tables(self, tmp_path):
        """Legacy agent tables are migrated into benchmark_results and dropped."""
        db_file = tmp_path / "cache.db"

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE agent_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT,
                timestamp TEXT NOT NULL,
                capability TEXT NOT NULL,
                test_id TEXT,
                test_name TEXT,
                latency_ms REAL,
                throughput_tokens_per_sec REAL,
                quality_score REAL,
                success INTEGER
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO agent_results (
                model_name, model_path, timestamp, capability,
                test_id, test_name, latency_ms,
                throughput_tokens_per_sec, quality_score, success
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-model",
                "/path/model",
                "2026-01-01T12:00:00",
                "general_text",
                "t1",
                "QA",
                120.0,
                23.5,
                0.8,
                1,
            ),
        )
        cursor.execute(
            """
            CREATE TABLE agent_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT,
                timestamp TEXT NOT NULL,
                capability TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        AgentCache(db_path=db_file)

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        table_names = {row[0] for row in cursor.fetchall()}

        cursor.execute(
            """
            SELECT source, capability, test_id, avg_tokens_per_sec, avg_gen_time
            FROM benchmark_results
            WHERE model_name = ?
            """,
            ("test-model",),
        )
        row = cursor.fetchone()
        conn.close()

        assert "agent_results" not in table_names
        assert "agent_summaries" not in table_names
        assert row is not None
        assert row[0] == "compatibility"
        assert row[1] == "general_text"
        assert row[2] == "t1"
        assert row[3] == 23.5
        assert row[4] == 120.0

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
        """save_summary keeps API compatibility and returns True."""
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

    def test_get_model_results_returns_saved_result(self, tmp_path):
        """get_model_results returns previously saved compatibility rows."""
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
        assert all(row.get("source") == "compatibility" for row in results)

    def test_get_model_results_with_capability_filter(self, tmp_path):
        """Capability filter returns only matching compatibility rows."""
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

    def test_get_latest_summary_aggregates_from_results(self, tmp_path):
        """get_latest_summary aggregates metrics from benchmark_results."""
        cache = AgentCache(db_path=tmp_path / "cache.db")

        cache.save_test_result(
            model_name="test-model",
            model_path="/path",
            capability="general_text",
            test_id="test_1",
            test_name="QA1",
            latency_ms=100.0,
            tokens_generated=40,
            throughput_tokens_per_sec=20.0,
            quality_score=0.8,
            success=True,
        )
        cache.save_test_result(
            model_name="test-model",
            model_path="/path",
            capability="general_text",
            test_id="test_2",
            test_name="QA2",
            latency_ms=120.0,
            tokens_generated=50,
            throughput_tokens_per_sec=30.0,
            quality_score=0.9,
            success=False,
        )

        result = cache.get_latest_summary("test-model", "general_text")

        assert result is not None
        assert result["model_name"] == "test-model"
        assert result["capability"] == "general_text"
        assert result["total_tests"] == 2
        assert result["successful_tests"] == 1
        assert result["failed_tests"] == 1
        assert result["success_rate"] == 0.5

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
