"""SQLite cache for capability-driven benchmark results."""

from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

from core.paths import USER_RESULTS_DIR

logger = logging.getLogger(__name__)


class AgentCache:
    """SQLite cache for capability-driven benchmark metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize agent cache and ensure shared schema is available."""
        default_db_path = Path(USER_RESULTS_DIR) / "benchmark_cache.db"
        legacy_db_path = Path(USER_RESULTS_DIR) / "agent_results.db"

        if db_path is None:
            db_path = default_db_path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._create_schema()
        if self.db_path == default_db_path:
            self._migrate_legacy_cache(legacy_db_path)

    @staticmethod
    def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
        """Return True if a table exists in the current SQLite database."""
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None

    def _create_schema(self) -> None:
        """Create or migrate the shared benchmark_results schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                quantization TEXT NOT NULL,
                inference_params_hash TEXT NOT NULL,
                gpu_type TEXT NOT NULL,
                gpu_offload REAL NOT NULL,
                vram_mb TEXT NOT NULL,
                avg_tokens_per_sec REAL NOT NULL,
                avg_ttft REAL NOT NULL,
                avg_gen_time REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                params_size TEXT NOT NULL,
                architecture TEXT NOT NULL,
                max_context_length INTEGER NOT NULL,
                model_size_gb REAL NOT NULL,
                has_vision INTEGER NOT NULL,
                has_tools INTEGER NOT NULL,
                tokens_per_sec_per_gb REAL,
                tokens_per_sec_per_billion_params REAL,
                speed_delta_pct REAL,
                prev_timestamp TEXT,
                prompt TEXT NOT NULL,
                context_length INTEGER NOT NULL,
                temperature REAL,
                top_k_sampling INTEGER,
                top_p_sampling REAL,
                min_p_sampling REAL,
                repeat_penalty REAL,
                max_tokens INTEGER,
                num_runs INTEGER,
                runs_averaged_from INTEGER,
                warmup_runs INTEGER,
                run_index INTEGER,
                lmstudio_version TEXT,
                app_version TEXT,
                nvidia_driver_version TEXT,
                rocm_driver_version TEXT,
                intel_driver_version TEXT,
                prompt_hash TEXT,
                params_hash TEXT,
                os_name TEXT,
                os_version TEXT,
                cpu_model TEXT,
                python_version TEXT,
                benchmark_duration_seconds REAL,
                error_count INTEGER,
                n_gpu_layers INTEGER,
                n_batch INTEGER,
                n_threads INTEGER,
                flash_attention INTEGER,
                rope_freq_base REAL,
                rope_freq_scale REAL,
                use_mmap INTEGER,
                use_mlock INTEGER,
                kv_cache_quant TEXT,
                temp_celsius_min REAL,
                temp_celsius_max REAL,
                temp_celsius_avg REAL,
                power_watts_min REAL,
                power_watts_max REAL,
                power_watts_avg REAL,
                vram_gb_min REAL,
                vram_gb_max REAL,
                vram_gb_avg REAL,
                gtt_gb_min REAL,
                gtt_gb_max REAL,
                gtt_gb_avg REAL,
                cpu_percent_min REAL,
                cpu_percent_max REAL,
                cpu_percent_avg REAL,
                ram_gb_min REAL,
                ram_gb_max REAL,
                ram_gb_avg REAL,
                tokens_per_sec_p50 REAL,
                tokens_per_sec_p95 REAL,
                tokens_per_sec_std REAL,
                ttft_p50 REAL,
                ttft_p95 REAL,
                ttft_std REAL,
                capability TEXT,
                test_id TEXT,
                test_name TEXT,
                quality_score REAL,
                rouge_score REAL,
                f1_score REAL,
                exact_match_score REAL,
                accuracy_score REAL,
                function_call_accuracy REAL,
                success INTEGER,
                error_message TEXT,
                raw_output TEXT,
                reference_output TEXT,
                source TEXT
            )
        """)

        cursor.execute("PRAGMA table_info(benchmark_results)")
        benchmark_columns = {row[1] for row in cursor.fetchall()}
        if "source" not in benchmark_columns:
            cursor.execute("ALTER TABLE benchmark_results ADD COLUMN source TEXT")

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_benchmark_source
            ON benchmark_results(source)
        """)

        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_benchmark_compat_unique
            ON benchmark_results(model_name, timestamp, capability, test_id)
            WHERE source = 'compatibility'
            AND capability IS NOT NULL
            AND test_id IS NOT NULL
        """)

        self._migrate_embedded_legacy_tables(conn, cursor)
        cursor.execute(
            "UPDATE benchmark_results SET source='classic' "
            "WHERE source IS NULL"
        )

        conn.commit()
        conn.close()

        logger.info("Agent cache initialized: %s", self.db_path)

    def _migrate_embedded_legacy_tables(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
    ) -> None:
        """Migrate agent_results/agent_summaries tables from this database."""
        if not self._table_exists(cursor, "agent_results"):
            if self._table_exists(cursor, "agent_summaries"):
                cursor.execute("DROP TABLE IF EXISTS agent_summaries")
            return

        cursor.execute("PRAGMA table_info(agent_results)")
        agent_columns = {row[1] for row in cursor.fetchall()}

        if "model_key" in agent_columns:
            cursor.execute("""
                INSERT OR IGNORE INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, prompt,
                    context_length, capability, test_id, test_name,
                    quality_score, success, error_message, raw_output,
                    reference_output, source
                )
                SELECT
                    COALESCE(model_key, ''),
                    COALESCE(model_name, ''),
                    '',
                    '',
                    '',
                    0.0,
                    '',
                    COALESCE(throughput_tokens_per_sec, 0.0),
                    0.0,
                    COALESCE(latency_ms, 0.0),
                    0,
                    0,
                    COALESCE(timestamp, ''),
                    '',
                    '',
                    0,
                    0.0,
                    0,
                    0,
                    '',
                    0,
                    COALESCE(capability, ''),
                    test_id,
                    test_name,
                    quality_score,
                    CASE WHEN success IS NULL THEN 0 ELSE success END,
                    NULL,
                    NULL,
                    NULL,
                    'compatibility'
                FROM agent_results
            """)
        elif "model_path" in agent_columns:
            cursor.execute("""
                INSERT OR IGNORE INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, prompt,
                    context_length, capability, test_id, test_name,
                    quality_score, success, error_message, raw_output,
                    reference_output, source
                )
                SELECT
                    COALESCE(model_path, ''),
                    COALESCE(model_name, ''),
                    '',
                    '',
                    '',
                    0.0,
                    '',
                    COALESCE(throughput_tokens_per_sec, 0.0),
                    0.0,
                    COALESCE(latency_ms, 0.0),
                    0,
                    0,
                    COALESCE(timestamp, ''),
                    '',
                    '',
                    0,
                    0.0,
                    0,
                    0,
                    '',
                    0,
                    COALESCE(capability, ''),
                    test_id,
                    test_name,
                    quality_score,
                    CASE WHEN success IS NULL THEN 0 ELSE success END,
                    NULL,
                    NULL,
                    NULL,
                    'compatibility'
                FROM agent_results
            """)
        else:
            cursor.execute("""
                INSERT OR IGNORE INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, prompt,
                    context_length, capability, test_id, test_name,
                    quality_score, success, error_message, raw_output,
                    reference_output, source
                )
                SELECT
                    '',
                    COALESCE(model_name, ''),
                    '',
                    '',
                    '',
                    0.0,
                    '',
                    COALESCE(throughput_tokens_per_sec, 0.0),
                    0.0,
                    COALESCE(latency_ms, 0.0),
                    0,
                    0,
                    COALESCE(timestamp, ''),
                    '',
                    '',
                    0,
                    0.0,
                    0,
                    0,
                    '',
                    0,
                    COALESCE(capability, ''),
                    test_id,
                    test_name,
                    quality_score,
                    CASE WHEN success IS NULL THEN 0 ELSE success END,
                    NULL,
                    NULL,
                    NULL,
                    'compatibility'
                FROM agent_results
            """)

        if self._table_exists(cursor, "agent_summaries"):
            cursor.execute("DROP TABLE IF EXISTS agent_summaries")
        cursor.execute("DROP TABLE IF EXISTS agent_results")
        conn.commit()

    def _migrate_legacy_cache(self, legacy_db_path: Path) -> None:
        """Migrate legacy external agent_results.db into benchmark_results."""
        if not legacy_db_path.exists() or legacy_db_path == self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("ATTACH DATABASE ? AS legacy", (str(legacy_db_path),))
            cursor.execute(
                "SELECT name FROM legacy.sqlite_master "
                "WHERE type='table' AND name='agent_results'"
            )
            if cursor.fetchone() is None:
                return

            cursor.execute("PRAGMA legacy.table_info(agent_results)")
            agent_columns = {row[1] for row in cursor.fetchall()}

            if "model_key" in agent_columns:
                cursor.execute("""
                    INSERT OR IGNORE INTO benchmark_results (
                        model_key, model_name, quantization,
                        inference_params_hash, gpu_type, gpu_offload,
                        vram_mb, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                        prompt_tokens, completion_tokens, timestamp,
                        params_size, architecture, max_context_length,
                        model_size_gb, has_vision, has_tools, prompt,
                        context_length, capability, test_id, test_name,
                        quality_score, success, error_message, raw_output,
                        reference_output, source
                    )
                    SELECT
                        COALESCE(model_key, ''),
                        COALESCE(model_name, ''),
                        '',
                        '',
                        '',
                        0.0,
                        '',
                        COALESCE(throughput_tokens_per_sec, 0.0),
                        0.0,
                        COALESCE(latency_ms, 0.0),
                        0,
                        0,
                        COALESCE(timestamp, ''),
                        '',
                        '',
                        0,
                        0.0,
                        0,
                        0,
                        '',
                        0,
                        COALESCE(capability, ''),
                        test_id,
                        test_name,
                        quality_score,
                        CASE WHEN success IS NULL THEN 0 ELSE success END,
                        NULL,
                        NULL,
                        NULL,
                        'compatibility'
                    FROM legacy.agent_results
                """)
            elif "model_path" in agent_columns:
                cursor.execute("""
                    INSERT OR IGNORE INTO benchmark_results (
                        model_key, model_name, quantization,
                        inference_params_hash, gpu_type, gpu_offload,
                        vram_mb, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                        prompt_tokens, completion_tokens, timestamp,
                        params_size, architecture, max_context_length,
                        model_size_gb, has_vision, has_tools, prompt,
                        context_length, capability, test_id, test_name,
                        quality_score, success, error_message, raw_output,
                        reference_output, source
                    )
                    SELECT
                        COALESCE(model_path, ''),
                        COALESCE(model_name, ''),
                        '',
                        '',
                        '',
                        0.0,
                        '',
                        COALESCE(throughput_tokens_per_sec, 0.0),
                        0.0,
                        COALESCE(latency_ms, 0.0),
                        0,
                        0,
                        COALESCE(timestamp, ''),
                        '',
                        '',
                        0,
                        0.0,
                        0,
                        0,
                        '',
                        0,
                        COALESCE(capability, ''),
                        test_id,
                        test_name,
                        quality_score,
                        CASE WHEN success IS NULL THEN 0 ELSE success END,
                        NULL,
                        NULL,
                        NULL,
                        'compatibility'
                    FROM legacy.agent_results
                """)
            else:
                cursor.execute("""
                    INSERT OR IGNORE INTO benchmark_results (
                        model_key, model_name, quantization,
                        inference_params_hash, gpu_type, gpu_offload,
                        vram_mb, avg_tokens_per_sec, avg_ttft, avg_gen_time,
                        prompt_tokens, completion_tokens, timestamp,
                        params_size, architecture, max_context_length,
                        model_size_gb, has_vision, has_tools, prompt,
                        context_length, capability, test_id, test_name,
                        quality_score, success, error_message, raw_output,
                        reference_output, source
                    )
                    SELECT
                        '',
                        COALESCE(model_name, ''),
                        '',
                        '',
                        '',
                        0.0,
                        '',
                        COALESCE(throughput_tokens_per_sec, 0.0),
                        0.0,
                        COALESCE(latency_ms, 0.0),
                        0,
                        0,
                        COALESCE(timestamp, ''),
                        '',
                        '',
                        0,
                        0.0,
                        0,
                        0,
                        '',
                        0,
                        COALESCE(capability, ''),
                        test_id,
                        test_name,
                        quality_score,
                        CASE WHEN success IS NULL THEN 0 ELSE success END,
                        NULL,
                        NULL,
                        NULL,
                        'compatibility'
                    FROM legacy.agent_results
                """)
            conn.commit()
        except sqlite3.Error:
            # Keep non-fatal to avoid blocking benchmark execution.
            pass
        finally:
            try:
                cursor.execute("DETACH DATABASE legacy")
            except sqlite3.Error:
                pass
            conn.close()

    def save_test_result(
        self,
        model_name: str,
        model_path: str,
        capability: str,
        test_id: str,
        test_name: str,
        latency_ms: float,
        tokens_generated: Optional[int],
        throughput_tokens_per_sec: Optional[float],
        quality_score: float,
        rouge_score: Optional[float] = None,
        f1_score: Optional[float] = None,
        exact_match_score: Optional[float] = None,
        accuracy_score: Optional[float] = None,
        function_call_accuracy: Optional[float] = None,
        raw_output: str = "",
        reference_output: str = "",
        error_message: Optional[str] = None,
        success: bool = True,
        error_count: Optional[int] = None,
        gpu_type: Optional[str] = None,
        gpu_offload: Optional[float] = None,
        vram_mb: Optional[str] = None,
        temp_celsius_min: Optional[float] = None,
        temp_celsius_max: Optional[float] = None,
        temp_celsius_avg: Optional[float] = None,
        power_watts_min: Optional[float] = None,
        power_watts_max: Optional[float] = None,
        power_watts_avg: Optional[float] = None,
        vram_gb_min: Optional[float] = None,
        vram_gb_max: Optional[float] = None,
        vram_gb_avg: Optional[float] = None,
        gtt_gb_min: Optional[float] = None,
        gtt_gb_max: Optional[float] = None,
        gtt_gb_avg: Optional[float] = None,
        cpu_percent_min: Optional[float] = None,
        cpu_percent_max: Optional[float] = None,
        cpu_percent_avg: Optional[float] = None,
        ram_gb_min: Optional[float] = None,
        ram_gb_max: Optional[float] = None,
        ram_gb_avg: Optional[float] = None,
        context_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k_sampling: Optional[int] = None,
        top_p_sampling: Optional[float] = None,
        min_p_sampling: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        n_batch: Optional[int] = None,
        n_threads: Optional[int] = None,
        flash_attention: Optional[bool] = None,
        rope_freq_base: Optional[float] = None,
        rope_freq_scale: Optional[float] = None,
        use_mmap: Optional[bool] = None,
        use_mlock: Optional[bool] = None,
        kv_cache_quant: Optional[str] = None,
        lmstudio_version: Optional[str] = None,
        app_version: Optional[str] = None,
        nvidia_driver_version: Optional[str] = None,
        rocm_driver_version: Optional[str] = None,
        intel_driver_version: Optional[str] = None,
        os_name: Optional[str] = None,
        os_version: Optional[str] = None,
        cpu_model: Optional[str] = None,
        python_version: Optional[str] = None,
        benchmark_duration_seconds: Optional[float] = None,
        quantization: Optional[str] = None,
        inference_params_hash: Optional[str] = None,
        avg_ttft: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        tokens_per_sec_per_gb: Optional[float] = None,
        tokens_per_sec_per_billion_params: Optional[float] = None,
        speed_delta_pct: Optional[float] = None,
        prev_timestamp: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        params_hash: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> bool:
        """Save individual compatibility test result into benchmark_results."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()

            values = (
                model_path,
                model_name,
                quantization or "",
                inference_params_hash or "",
                gpu_type or "",
                gpu_offload if gpu_offload is not None else 0.0,
                vram_mb or "",
                throughput_tokens_per_sec if throughput_tokens_per_sec is not None
                else 0.0,
                avg_ttft if avg_ttft is not None else 0.0,
                latency_ms,
                prompt_tokens if prompt_tokens is not None else 0,
                tokens_generated if tokens_generated is not None else 0,
                timestamp,
                "",
                "",
                0,
                0.0,
                0,
                0,
                tokens_per_sec_per_gb,
                tokens_per_sec_per_billion_params,
                speed_delta_pct,
                prev_timestamp,
                prompt or "",
                context_length if context_length is not None else 0,
                temperature,
                top_k_sampling,
                top_p_sampling,
                min_p_sampling,
                repeat_penalty,
                max_tokens,
                1,
                1,
                0,
                None,
                lmstudio_version,
                app_version,
                nvidia_driver_version,
                rocm_driver_version,
                intel_driver_version,
                prompt_hash,
                params_hash,
                os_name,
                os_version,
                cpu_model,
                python_version,
                benchmark_duration_seconds,
                error_count,
                n_gpu_layers,
                n_batch,
                n_threads,
                int(flash_attention) if flash_attention is not None else None,
                rope_freq_base,
                rope_freq_scale,
                int(use_mmap) if use_mmap is not None else None,
                int(use_mlock) if use_mlock is not None else None,
                kv_cache_quant,
                temp_celsius_min,
                temp_celsius_max,
                temp_celsius_avg,
                power_watts_min,
                power_watts_max,
                power_watts_avg,
                vram_gb_min,
                vram_gb_max,
                vram_gb_avg,
                gtt_gb_min,
                gtt_gb_max,
                gtt_gb_avg,
                cpu_percent_min,
                cpu_percent_max,
                cpu_percent_avg,
                ram_gb_min,
                ram_gb_max,
                ram_gb_avg,
                None,
                None,
                None,
                None,
                None,
                None,
                capability,
                test_id,
                test_name,
                quality_score,
                rouge_score,
                f1_score,
                exact_match_score,
                accuracy_score,
                function_call_accuracy,
                int(success),
                error_message,
                raw_output,
                reference_output,
                "compatibility",
            )
            placeholders = ", ".join("?" for _ in values)

            cursor.execute(
                f"""
                INSERT OR IGNORE INTO benchmark_results (
                    model_key, model_name, quantization, inference_params_hash,
                    gpu_type, gpu_offload, vram_mb, avg_tokens_per_sec,
                    avg_ttft, avg_gen_time, prompt_tokens, completion_tokens,
                    timestamp, params_size, architecture, max_context_length,
                    model_size_gb, has_vision, has_tools, tokens_per_sec_per_gb,
                    tokens_per_sec_per_billion_params, speed_delta_pct,
                    prev_timestamp, prompt, context_length, temperature,
                    top_k_sampling, top_p_sampling, min_p_sampling,
                    repeat_penalty, max_tokens, num_runs, runs_averaged_from,
                    warmup_runs, run_index, lmstudio_version, app_version,
                    nvidia_driver_version, rocm_driver_version,
                    intel_driver_version, prompt_hash, params_hash, os_name,
                    os_version, cpu_model, python_version,
                    benchmark_duration_seconds, error_count, n_gpu_layers,
                    n_batch, n_threads, flash_attention, rope_freq_base,
                    rope_freq_scale, use_mmap, use_mlock, kv_cache_quant,
                    temp_celsius_min, temp_celsius_max, temp_celsius_avg,
                    power_watts_min, power_watts_max, power_watts_avg,
                    vram_gb_min, vram_gb_max, vram_gb_avg,
                    gtt_gb_min, gtt_gb_max, gtt_gb_avg,
                    cpu_percent_min, cpu_percent_max, cpu_percent_avg,
                    ram_gb_min, ram_gb_max, ram_gb_avg,
                    tokens_per_sec_p50, tokens_per_sec_p95, tokens_per_sec_std,
                    ttft_p50, ttft_p95, ttft_std, capability, test_id,
                    test_name, quality_score, rouge_score, f1_score,
                    exact_match_score, accuracy_score,
                    function_call_accuracy, success, error_message,
                    raw_output, reference_output, source
                ) VALUES ({placeholders})
                """,
                values,
            )

            conn.commit()
            conn.close()

            logger.debug("Saved result: %s/%s/%s", model_name, capability, test_id)
            return True
        except sqlite3.Error as error:
            logger.error("Error saving test result: %s", error)
            return False

    def save_summary(
        self,
        model_name: str,
        model_path: str,
        capability: str,
        total_tests: int,
        successful_tests: int,
        failed_tests: int,
        success_rate: float,
        avg_latency_ms: float,
        avg_throughput: Optional[float],
        avg_quality_score: float,
        avg_rouge: Optional[float] = None,
        avg_f1: Optional[float] = None,
        avg_exact_match: Optional[float] = None,
        avg_accuracy: Optional[float] = None,
    ) -> bool:
        """Keep API compatibility; summaries are now derived on read."""
        _ = (
            model_name,
            model_path,
            capability,
            total_tests,
            successful_tests,
            failed_tests,
            success_rate,
            avg_latency_ms,
            avg_throughput,
            avg_quality_score,
            avg_rouge,
            avg_f1,
            avg_exact_match,
            avg_accuracy,
        )
        return True

    def get_model_results(
        self,
        model_name: str,
        capability: Optional[str] = None,
    ) -> List[Dict]:
        """Retrieve compatibility benchmark rows from benchmark_results."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if capability:
                cursor.execute(
                    """
                    SELECT * FROM benchmark_results
                    WHERE model_name = ?
                    AND source = 'compatibility'
                    AND capability = ?
                    ORDER BY timestamp DESC
                    """,
                    (model_name, capability),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM benchmark_results
                    WHERE model_name = ?
                    AND source = 'compatibility'
                    ORDER BY timestamp DESC
                    """,
                    (model_name,),
                )

            rows = cursor.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except sqlite3.Error as error:
            logger.error("Error retrieving results: %s", error)
            return []

    def get_latest_summary(
        self,
        model_name: str,
        capability: str,
    ) -> Optional[Dict]:
        """Return aggregated compatibility summary from benchmark_results."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    model_name,
                    model_key,
                    capability,
                    MAX(timestamp) AS timestamp,
                    COUNT(*) AS total_tests,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
                        AS successful_tests,
                    SUM(CASE WHEN success = 1 THEN 0 ELSE 1 END)
                        AS failed_tests,
                    AVG(avg_gen_time) AS avg_latency_ms,
                    AVG(avg_tokens_per_sec) AS avg_throughput,
                    AVG(quality_score) AS avg_quality_score,
                    AVG(rouge_score) AS avg_rouge,
                    AVG(f1_score) AS avg_f1,
                    AVG(exact_match_score) AS avg_exact_match,
                    AVG(accuracy_score) AS avg_accuracy
                FROM benchmark_results
                WHERE model_name = ?
                AND capability = ?
                AND source = 'compatibility'
                GROUP BY model_name, model_key, capability
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (model_name, capability),
            )

            row = cursor.fetchone()
            conn.close()
            if row is None:
                return None

            summary = dict(row)
            total_tests = int(summary.get("total_tests") or 0)
            successful_tests = int(summary.get("successful_tests") or 0)
            summary["success_rate"] = (
                successful_tests / total_tests if total_tests > 0 else 0.0
            )
            return summary
        except sqlite3.Error as error:
            logger.error("Error retrieving summary: %s", error)
            return None
