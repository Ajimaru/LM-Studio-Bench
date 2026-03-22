"""SQLite cache for capability-driven benchmark results."""

from datetime import datetime
import logging
from pathlib import Path
import sqlite3
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentCache:
    """SQLite cache for capability-driven benchmark metrics."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize agent cache.

        Args:
            db_path: Path to SQLite database (defaults to results dir)
        """
        if db_path is None:
            from src.user_paths import USER_RESULTS_DIR
            db_path = Path(USER_RESULTS_DIR) / "agent_results.db"

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._create_schema()

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
                UNIQUE (model_name, timestamp, capability, test_id)
            )
        """)

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

        logger.info(f"Agent cache initialized: {self.db_path}")

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

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()

            cursor.execute("""
                INSERT OR REPLACE INTO agent_results (
                    model_name, model_path, timestamp, capability, test_id,
                    test_name, latency_ms, tokens_generated,
                    throughput_tokens_per_sec, quality_score, rouge_score,
                    f1_score, exact_match_score, accuracy_score,
                    function_call_accuracy, raw_output, reference_output,
                    error_message, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name, model_path, timestamp, capability, test_id,
                test_name, latency_ms, tokens_generated,
                throughput_tokens_per_sec, quality_score, rouge_score,
                f1_score, exact_match_score, accuracy_score,
                function_call_accuracy, raw_output, reference_output,
                error_message, success
            ))

            conn.commit()
            conn.close()

            logger.debug(f"Saved result: {model_name}/{capability}/{test_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error saving test result: {e}")
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

            logger.info(f"Saved summary: {model_name}/{capability}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Error saving summary: {e}")
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
            logger.error(f"Error retrieving results: {e}")
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
            logger.error(f"Error retrieving summary: {e}")
            return None
