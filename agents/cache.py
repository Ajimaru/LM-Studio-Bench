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

    CLASSIC_METRIC_COLUMNS = [
        ("error_count", "INTEGER"),
        ("gpu_type", "TEXT"),
        ("gpu_offload", "REAL"),
        ("vram_mb", "TEXT"),
        ("temp_celsius_min", "REAL"),
        ("temp_celsius_max", "REAL"),
        ("temp_celsius_avg", "REAL"),
        ("power_watts_min", "REAL"),
        ("power_watts_max", "REAL"),
        ("power_watts_avg", "REAL"),
        ("vram_gb_min", "REAL"),
        ("vram_gb_max", "REAL"),
        ("vram_gb_avg", "REAL"),
        ("gtt_gb_min", "REAL"),
        ("gtt_gb_max", "REAL"),
        ("gtt_gb_avg", "REAL"),
        ("cpu_percent_min", "REAL"),
        ("cpu_percent_max", "REAL"),
        ("cpu_percent_avg", "REAL"),
        ("ram_gb_min", "REAL"),
        ("ram_gb_max", "REAL"),
        ("ram_gb_avg", "REAL"),
        ("context_length", "INTEGER"),
        ("temperature", "REAL"),
        ("top_k_sampling", "INTEGER"),
        ("top_p_sampling", "REAL"),
        ("min_p_sampling", "REAL"),
        ("repeat_penalty", "REAL"),
        ("max_tokens", "INTEGER"),
        ("n_gpu_layers", "INTEGER"),
        ("n_batch", "INTEGER"),
        ("n_threads", "INTEGER"),
        ("flash_attention", "INTEGER"),
        ("rope_freq_base", "REAL"),
        ("rope_freq_scale", "REAL"),
        ("use_mmap", "INTEGER"),
        ("use_mlock", "INTEGER"),
        ("kv_cache_quant", "TEXT"),
        ("lmstudio_version", "TEXT"),
        ("app_version", "TEXT"),
        ("nvidia_driver_version", "TEXT"),
        ("rocm_driver_version", "TEXT"),
        ("intel_driver_version", "TEXT"),
        ("os_name", "TEXT"),
        ("os_version", "TEXT"),
        ("cpu_model", "TEXT"),
        ("python_version", "TEXT"),
        ("benchmark_duration_seconds", "REAL"),
        ("quantization", "TEXT"),
        ("inference_params_hash", "TEXT"),
        ("avg_ttft", "REAL"),
        ("prompt_tokens", "INTEGER"),
        ("tokens_per_sec_per_gb", "REAL"),
        ("tokens_per_sec_per_billion_params", "REAL"),
        ("speed_delta_pct", "REAL"),
        ("prev_timestamp", "TEXT"),
        ("prompt_hash", "TEXT"),
        ("params_hash", "TEXT"),
        ("prompt", "TEXT"),
    ]

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize agent cache.

        Args:
            db_path: Path to SQLite database (defaults to results dir)
        """
        default_db_path = Path(USER_RESULTS_DIR) / "benchmark_cache.db"
        legacy_db_path = Path(USER_RESULTS_DIR) / "agent_results.db"

        if db_path is None:
            db_path = default_db_path

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._create_schema()
        if self.db_path == default_db_path:
            self._migrate_legacy_cache(legacy_db_path)

    def _migrate_legacy_cache(self, legacy_db_path: Path) -> None:
        """Migrate old capability cache DB into the shared benchmark DB."""
        if not legacy_db_path.exists() or legacy_db_path == self.db_path:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("ATTACH DATABASE ? AS legacy", (str(legacy_db_path),))

            cursor.execute(
                "SELECT name FROM legacy.sqlite_master "
                "WHERE type = 'table' AND name = ?",
                ("agent_results",),
            )
            if cursor.fetchone() is not None:
                cursor.execute("""
                    INSERT OR IGNORE INTO agent_results (
                        model_name, model_path, timestamp, capability,
                        test_id, test_name, latency_ms, tokens_generated,
                        throughput_tokens_per_sec, quality_score, rouge_score,
                        f1_score, exact_match_score, accuracy_score,
                        function_call_accuracy, raw_output, reference_output,
                        error_message, success
                    )
                    SELECT
                        model_name, model_path, timestamp, capability,
                        test_id, test_name, latency_ms, tokens_generated,
                        throughput_tokens_per_sec, quality_score, rouge_score,
                        f1_score, exact_match_score, accuracy_score,
                        function_call_accuracy, raw_output, reference_output,
                        error_message, success
                    FROM legacy.agent_results
                """)

            cursor.execute(
                "SELECT name FROM legacy.sqlite_master "
                "WHERE type = 'table' AND name = ?",
                ("agent_summaries",),
            )
            if cursor.fetchone() is not None:
                cursor.execute("""
                    INSERT OR IGNORE INTO agent_summaries (
                        model_name, model_path, timestamp, capability,
                        total_tests, successful_tests, failed_tests,
                        success_rate, avg_latency_ms, avg_throughput,
                        avg_quality_score, avg_rouge, avg_f1,
                        avg_exact_match, avg_accuracy
                    )
                    SELECT
                        model_name, model_path, timestamp, capability,
                        total_tests, successful_tests, failed_tests,
                        success_rate, avg_latency_ms, avg_throughput,
                        avg_quality_score, avg_rouge, avg_f1,
                        avg_exact_match, avg_accuracy
                    FROM legacy.agent_summaries
                """)

            conn.commit()
        except sqlite3.Error as error:
            logger.warning("Could not migrate legacy agent cache: %s", error)
        finally:
            try:
                cursor.execute("DETACH DATABASE legacy")
            except sqlite3.Error:
                pass
            conn.close()

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT,
                timestamp TEXT NOT NULL,
                capability TEXT NOT NULL,
                test_id TEXT,
                test_name TEXT,
                latency_ms REAL,
                tokens_generated INTEGER,
                throughput_tokens_per_sec REAL,
                quality_score REAL,
                rouge_score REAL,
                f1_score REAL,
                exact_match_score REAL,
                accuracy_score REAL,
                function_call_accuracy REAL,
                raw_output TEXT,
                reference_output TEXT,
                error_message TEXT,
                success BOOLEAN,
                error_count INTEGER,
                gpu_type TEXT,
                gpu_offload REAL,
                vram_mb TEXT,
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
                context_length INTEGER,
                temperature REAL,
                top_k_sampling INTEGER,
                top_p_sampling REAL,
                min_p_sampling REAL,
                repeat_penalty REAL,
                max_tokens INTEGER,
                n_gpu_layers INTEGER,
                n_batch INTEGER,
                n_threads INTEGER,
                flash_attention INTEGER,
                rope_freq_base REAL,
                rope_freq_scale REAL,
                use_mmap INTEGER,
                use_mlock INTEGER,
                kv_cache_quant TEXT,
                lmstudio_version TEXT,
                app_version TEXT,
                nvidia_driver_version TEXT,
                rocm_driver_version TEXT,
                intel_driver_version TEXT,
                os_name TEXT,
                os_version TEXT,
                cpu_model TEXT,
                python_version TEXT,
                benchmark_duration_seconds REAL,
                quantization TEXT,
                inference_params_hash TEXT,
                avg_ttft REAL,
                prompt_tokens INTEGER,
                tokens_per_sec_per_gb REAL,
                tokens_per_sec_per_billion_params REAL,
                speed_delta_pct REAL,
                prev_timestamp TEXT,
                prompt_hash TEXT,
                params_hash TEXT,
                prompt TEXT,
                UNIQUE (model_name, timestamp, capability, test_id)
            )
        """)

        cursor.execute("PRAGMA table_info(agent_results)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        for col_name, col_type in self.CLASSIC_METRIC_COLUMNS:
            if col_name in existing_columns:
                continue
            cursor.execute(
                f"ALTER TABLE agent_results ADD COLUMN {col_name} {col_type}"
            )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_path TEXT,
                timestamp TEXT NOT NULL,
                capability TEXT NOT NULL,
                total_tests INTEGER,
                successful_tests INTEGER,
                failed_tests INTEGER,
                success_rate REAL,
                avg_latency_ms REAL,
                avg_throughput REAL,
                avg_quality_score REAL,
                avg_rouge REAL,
                avg_f1 REAL,
                avg_exact_match REAL,
                avg_accuracy REAL,
                UNIQUE (model_name, timestamp, capability)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_timestamp
            ON agent_results(model_name, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_capability
            ON agent_results(capability)
        """)

        conn.commit()
        conn.close()

        logger.info("Agent cache initialized: %s", self.db_path)

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
        """
        Save individual test result to cache.

        Args:
            model_name: Name of the model
            model_path: Path to model
            capability: Capability being tested
            test_id: Unique test identifier
            test_name: Human-readable test name
            latency_ms: Test latency in milliseconds
            tokens_generated: Number of tokens generated
            throughput_tokens_per_sec: Tokens per second
            quality_score: Overall quality metric
            rouge_score: ROUGE score if applicable
            f1_score: F1 score if applicable
            exact_match_score: Exact match score if applicable
            accuracy_score: Accuracy if applicable
            function_call_accuracy: Function call accuracy if applicable
            raw_output: Model output
            reference_output: Reference/expected output
            error_message: Error message if failed
            success: Whether test succeeded
            error_count: Count of failed capability test cases in run
            gpu_type: GPU vendor/type
            gpu_offload: Configured GPU offload ratio
            vram_mb: GPU VRAM usage in MB
            temp_celsius_min: Minimum GPU temperature
            temp_celsius_max: Maximum GPU temperature
            temp_celsius_avg: Average GPU temperature
            power_watts_min: Minimum GPU power draw
            power_watts_max: Maximum GPU power draw
            power_watts_avg: Average GPU power draw
            vram_gb_min: Minimum VRAM usage in GB
            vram_gb_max: Maximum VRAM usage in GB
            vram_gb_avg: Average VRAM usage in GB
            gtt_gb_min: Minimum GTT usage in GB
            gtt_gb_max: Maximum GTT usage in GB
            gtt_gb_avg: Average GTT usage in GB
            cpu_percent_min: Minimum CPU usage percent
            cpu_percent_max: Maximum CPU usage percent
            cpu_percent_avg: Average CPU usage percent
            ram_gb_min: Minimum RAM usage in GB
            ram_gb_max: Maximum RAM usage in GB
            ram_gb_avg: Average RAM usage in GB
            context_length: Inference context length
            temperature: Sampling temperature
            top_k_sampling: Top-k sampling setting
            top_p_sampling: Top-p sampling setting
            min_p_sampling: Min-p sampling setting
            repeat_penalty: Repeat penalty setting
            max_tokens: Maximum token generation setting
            n_gpu_layers: Number of GPU layers setting
            n_batch: Batch-size setting
            n_threads: Thread-count setting
            flash_attention: Flash attention setting
            rope_freq_base: RoPE frequency base setting
            rope_freq_scale: RoPE frequency scale setting
            use_mmap: Memory mapping setting
            use_mlock: mlock setting
            kv_cache_quant: KV cache quantization setting
            lmstudio_version: LM Studio version
            app_version: LM-Studio-Bench application version
            nvidia_driver_version: NVIDIA driver version
            rocm_driver_version: ROCm/AMD driver version
            intel_driver_version: Intel driver version
            os_name: Operating system name
            os_version: Operating system version
            cpu_model: CPU model name
            python_version: Python runtime version
            benchmark_duration_seconds: Full model benchmark duration
            quantization: Model quantization identifier
            inference_params_hash: SHA-256 hash of inference parameters
            avg_ttft: Average time-to-first-token in milliseconds
            prompt_tokens: Number of tokens in the input prompt
            tokens_per_sec_per_gb: Throughput per GB of model size
            tokens_per_sec_per_billion_params: Throughput per billion params
            speed_delta_pct: Speed change vs. previous run in percent
            prev_timestamp: Timestamp of the previous run
            prompt_hash: Short SHA-256 hash of the prompt text
            params_hash: Short SHA-256 hash of all inference parameters
            prompt: Full prompt text used for the test

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            values = (
                model_name, model_path, timestamp, capability, test_id,
                test_name, latency_ms, tokens_generated,
                throughput_tokens_per_sec, quality_score, rouge_score,
                f1_score, exact_match_score, accuracy_score,
                function_call_accuracy, raw_output, reference_output,
                error_message, success, error_count, gpu_type,
                gpu_offload, vram_mb, temp_celsius_min,
                temp_celsius_max, temp_celsius_avg, power_watts_min,
                power_watts_max, power_watts_avg, vram_gb_min,
                vram_gb_max, vram_gb_avg, gtt_gb_min, gtt_gb_max,
                gtt_gb_avg, cpu_percent_min, cpu_percent_max,
                cpu_percent_avg, ram_gb_min, ram_gb_max, ram_gb_avg,
                context_length, temperature, top_k_sampling,
                top_p_sampling, min_p_sampling, repeat_penalty,
                max_tokens, n_gpu_layers, n_batch, n_threads,
                int(flash_attention) if flash_attention is not None else None,
                rope_freq_base, rope_freq_scale,
                int(use_mmap) if use_mmap is not None else None,
                int(use_mlock) if use_mlock is not None else None,
                kv_cache_quant, lmstudio_version, app_version,
                nvidia_driver_version, rocm_driver_version,
                intel_driver_version, os_name, os_version,
                cpu_model, python_version, benchmark_duration_seconds,
                quantization, inference_params_hash, avg_ttft,
                prompt_tokens, tokens_per_sec_per_gb,
                tokens_per_sec_per_billion_params, speed_delta_pct,
                prev_timestamp, prompt_hash, params_hash, prompt
            )
            placeholders = ", ".join("?" for _ in values)
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO agent_results (
                    model_name, model_path, timestamp, capability, test_id,
                    test_name, latency_ms, tokens_generated,
                    throughput_tokens_per_sec, quality_score, rouge_score,
                    f1_score, exact_match_score, accuracy_score,
                    function_call_accuracy, raw_output, reference_output,
                    error_message, success, error_count, gpu_type,
                    gpu_offload, vram_mb, temp_celsius_min,
                    temp_celsius_max, temp_celsius_avg, power_watts_min,
                    power_watts_max, power_watts_avg, vram_gb_min,
                    vram_gb_max, vram_gb_avg, gtt_gb_min, gtt_gb_max,
                    gtt_gb_avg, cpu_percent_min, cpu_percent_max,
                    cpu_percent_avg, ram_gb_min, ram_gb_max, ram_gb_avg,
                    context_length, temperature, top_k_sampling,
                    top_p_sampling, min_p_sampling, repeat_penalty,
                    max_tokens, n_gpu_layers, n_batch, n_threads,
                    flash_attention, rope_freq_base, rope_freq_scale,
                    use_mmap, use_mlock, kv_cache_quant,
                    lmstudio_version, app_version, nvidia_driver_version,
                    rocm_driver_version, intel_driver_version,
                    os_name, os_version, cpu_model, python_version,
                    benchmark_duration_seconds, quantization,
                    inference_params_hash, avg_ttft, prompt_tokens,
                    tokens_per_sec_per_gb, tokens_per_sec_per_billion_params,
                    speed_delta_pct, prev_timestamp, prompt_hash,
                    params_hash, prompt
                ) VALUES ({placeholders})
            """,
                values,
            )

            conn.commit()
            conn.close()

            logger.debug(
                "Saved result: %s/%s/%s",
                model_name,
                capability,
                test_id,
            )
            return True

        except sqlite3.Error as e:
            logger.error("Error saving test result: %s", e)
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
        """
        Save capability summary to cache.

        Args:
            model_name: Name of the model
            model_path: Path to model
            capability: Capability being tested
            total_tests: Total number of tests
            successful_tests: Number of successful tests
            failed_tests: Number of failed tests
            success_rate: Success rate (0-1)
            avg_latency_ms: Average latency in ms
            avg_throughput: Average throughput
            avg_quality_score: Average quality score
            avg_rouge: Average ROUGE score
            avg_f1: Average F1 score
            avg_exact_match: Average exact match
            avg_accuracy: Average accuracy

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO agent_summaries (
                    model_name, model_path, timestamp, capability,
                    total_tests, successful_tests, failed_tests,
                    success_rate, avg_latency_ms, avg_throughput,
                    avg_quality_score, avg_rouge, avg_f1,
                    avg_exact_match, avg_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name, model_path, timestamp, capability,
                total_tests, successful_tests, failed_tests,
                success_rate, avg_latency_ms, avg_throughput,
                avg_quality_score, avg_rouge, avg_f1,
                avg_exact_match, avg_accuracy
            ))

            conn.commit()
            conn.close()

            logger.info("Saved summary: %s/%s", model_name, capability)
            return True

        except sqlite3.Error as e:
            logger.error("Error saving summary: %s", e)
            return False

    def get_model_results(
        self,
        model_name: str,
        capability: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve results for a model.

        Args:
            model_name: Name of the model
            capability: Optional capability filter

        Returns:
            List of result dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if capability:
                cursor.execute("""
                    SELECT * FROM agent_results
                    WHERE model_name = ? AND capability = ?
                    ORDER BY timestamp DESC
                """, (model_name, capability))
            else:
                cursor.execute("""
                    SELECT * FROM agent_results
                    WHERE model_name = ?
                    ORDER BY timestamp DESC
                """, (model_name,))

            rows = cursor.fetchall()
            conn.close()

            return [dict(row) for row in rows]

        except sqlite3.Error as e:
            logger.error("Error retrieving results: %s", e)
            return []

    def get_latest_summary(
        self,
        model_name: str,
        capability: str,
    ) -> Optional[Dict]:
        """
        Get latest summary for model capability.

        Args:
            model_name: Name of the model
            capability: Capability to query

        Returns:
            Summary dictionary or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM agent_summaries
                WHERE model_name = ? AND capability = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (model_name, capability))

            row = cursor.fetchone()
            conn.close()

            return dict(row) if row else None

        except sqlite3.Error as e:
            logger.error("Error retrieving summary: %s", e)
            return None
