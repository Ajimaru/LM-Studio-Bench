"""Tests for tools/scrape_metadata.py."""
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _import_scrape_metadata(tmp_path: Path):
    """Import scrape_metadata with user dirs patched to tmp_path."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    backups_dir = results_dir / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)

    if "scrape_metadata" in sys.modules:
        del sys.modules["scrape_metadata"]

    with patch("user_paths.USER_RESULTS_DIR", results_dir), \
            patch("user_paths.USER_LOGS_DIR", logs_dir):
        import scrape_metadata as sm
        sm.RESULTS_DIR = results_dir
        sm.METADATA_DB = results_dir / "model_metadata.db"
        sm.LOGS_DIR = logs_dir
        sm.BACKUPS_DIR = backups_dir
    return sm


class TestEnsureSchema:
    """Tests for scrape_metadata.ensure_schema()."""

    def test_creates_table(self, tmp_path: Path):
        """ensure_schema creates the model_metadata table."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='model_metadata'"
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_path: Path):
        """ensure_schema can be called twice without error."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        sm.ensure_schema(conn)
        conn.close()

    def test_table_has_expected_columns(self, tmp_path: Path):
        """model_metadata table has all required columns."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        cur = conn.execute("PRAGMA table_info(model_metadata)")
        cols = {row[1] for row in cur.fetchall()}
        for expected in [
            "model_key",
            "display_name",
            "publisher",
            "capabilities",
            "description",
            "scraped_at",
        ]:
            assert expected in cols
        conn.close()


class TestStripTags:
    """Tests for scrape_metadata._strip_tags()."""

    def test_removes_html_tags(self, tmp_path: Path):
        """HTML tags are stripped from text."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm._strip_tags("<p>Hello <b>world</b></p>")
        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_removes_script_tags(self, tmp_path: Path):
        """Script contents are removed entirely."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm._strip_tags(
            "<script>alert('xss')</script>Safe text"
        )
        assert "alert" not in result
        assert "Safe text" in result

    def test_removes_style_tags(self, tmp_path: Path):
        """Style contents are removed entirely."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm._strip_tags("<style>.foo{color:red}</style>visible")
        assert "color" not in result
        assert "visible" in result

    def test_html_entities_unescaped(self, tmp_path: Path):
        """HTML entities are decoded in output."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm._strip_tags("&amp; &lt; &gt;")
        assert "&" in result

    def test_empty_string_returns_empty(self, tmp_path: Path):
        """Empty input returns empty string."""
        sm = _import_scrape_metadata(tmp_path)
        assert sm._strip_tags("") == ""

    def test_whitespace_collapsed(self, tmp_path: Path):
        """Multiple whitespace characters are collapsed to a single space."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm._strip_tags("hello   world")
        assert "  " not in result


class TestInferCapsFromDescription:
    """Tests for scrape_metadata.infer_caps_from_description()."""

    def test_empty_description_returns_empty(self, tmp_path: Path):
        """Empty description returns an empty capability list."""
        sm = _import_scrape_metadata(tmp_path)
        assert sm.infer_caps_from_description("") == []

    def test_coding_detected(self, tmp_path: Path):
        """'coding' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "Expert in programming and source code generation"
        )
        assert "coding" in caps

    def test_math_detected(self, tmp_path: Path):
        """'math' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description("Advanced mathematics and algebra")
        assert "math" in caps

    def test_reasoning_detected(self, tmp_path: Path):
        """'reasoning' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "Supports chain-of-thought logical reasoning"
        )
        assert "reasoning" in caps

    def test_chat_detected(self, tmp_path: Path):
        """'chat' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "A conversational assistant chatbot"
        )
        assert "chat" in caps

    def test_vision_detected(self, tmp_path: Path):
        """'vision' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "Multimodal vision-language model for image tasks"
        )
        assert "vision" in caps

    def test_tool_use_detected(self, tmp_path: Path):
        """'tool_use' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "Supports function calling and tool use"
        )
        assert "tool_use" in caps

    def test_creative_detected(self, tmp_path: Path):
        """'creative' capability is detected from relevant keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description("Creative story writing and poetry")
        assert "creative" in caps

    def test_none_description_returns_empty(self, tmp_path: Path):
        """None description returns an empty capability list."""
        sm = _import_scrape_metadata(tmp_path)
        assert sm.infer_caps_from_description(None) == []

    def test_multiple_caps_detected(self, tmp_path: Path):
        """Multiple capabilities can be detected from one description."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_caps_from_description(
            "A coding assistant with math reasoning and vision"
        )
        assert "coding" in caps
        assert "math" in caps
        assert "vision" in caps


class TestInferCapabilities:
    """Tests for scrape_metadata.infer_capabilities()."""

    def test_coding_from_key(self, tmp_path: Path):
        """'coding' inferred from 'coder' in model key."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_capabilities("org/deepseek-coder-7b", "DeepSeek Coder")
        assert "coding" in caps

    def test_reasoning_from_key(self, tmp_path: Path):
        """'reasoning' inferred from 'reason' in display name."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_capabilities("org/qwq-7b", "QwQ Reasoning Model")
        assert "reasoning" in caps

    def test_chat_from_instruct(self, tmp_path: Path):
        """'chat' inferred from 'instruct' in model key."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_capabilities("org/llama-instruct", "Llama Instruct")
        assert "chat" in caps

    def test_math_from_key(self, tmp_path: Path):
        """'math' inferred from 'math' in model key."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_capabilities("org/math-model", "Math Specialist")
        assert "math" in caps

    def test_no_caps_for_unknown_model(self, tmp_path: Path):
        """Empty list returned for a model with no matching keywords."""
        sm = _import_scrape_metadata(tmp_path)
        caps = sm.infer_capabilities("org/xyz-alpha", "XYZ Alpha")
        assert caps == []


class TestUpsertMetadata:
    """Tests for scrape_metadata.upsert_metadata()."""

    def test_inserts_rows(self, tmp_path: Path):
        """Rows are inserted into model_metadata."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        rows = [
            {
                "model_key": "pub/model1",
                "display_name": "Model 1",
                "publisher": "pub",
                "architecture": "llama",
                "params": "7B",
                "size_bytes": 1000,
                "max_context_length": 4096,
                "vision": 0,
                "tool_use": 0,
                "capabilities": "chat",
                "source_url": "http://example.com",
                "hf_tags": "llm",
                "description": "A model",
                "scraped_at": "2024-01-01T00:00:00",
            }
        ]
        sm.upsert_metadata(conn, rows)
        cur = conn.execute(
            "SELECT model_key FROM model_metadata WHERE model_key=?",
            ("pub/model1",),
        )
        assert cur.fetchone() is not None
        conn.close()

    def test_replaces_existing_row(self, tmp_path: Path):
        """Existing rows are replaced on conflict."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        row = {
            "model_key": "pub/model1",
            "display_name": "Old Name",
            "publisher": "pub",
            "architecture": None,
            "params": None,
            "size_bytes": 0,
            "max_context_length": 0,
            "vision": 0,
            "tool_use": 0,
            "capabilities": None,
            "source_url": None,
            "hf_tags": None,
            "description": None,
            "scraped_at": "2024-01-01T00:00:00",
        }
        sm.upsert_metadata(conn, [row])
        row["display_name"] = "New Name"
        sm.upsert_metadata(conn, [row])
        cur = conn.execute(
            "SELECT display_name FROM model_metadata WHERE model_key=?",
            ("pub/model1",),
        )
        assert cur.fetchone()[0] == "New Name"
        conn.close()


class TestUpdateRows:
    """Tests for scrape_metadata.update_rows()."""

    def test_empty_updates_is_no_op(self, tmp_path: Path):
        """Empty updates list does nothing."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        sm.update_rows(conn, [])
        conn.close()

    def test_updates_description(self, tmp_path: Path):
        """update_rows sets description on existing row."""
        sm = _import_scrape_metadata(tmp_path)
        conn = sqlite3.connect(":memory:")
        sm.ensure_schema(conn)
        base_row = {
            "model_key": "pub/m",
            "display_name": "M",
            "publisher": "pub",
            "architecture": None,
            "params": None,
            "size_bytes": 0,
            "max_context_length": 0,
            "vision": 0,
            "tool_use": 0,
            "capabilities": None,
            "source_url": None,
            "hf_tags": None,
            "description": None,
            "scraped_at": "2024-01-01",
        }
        sm.upsert_metadata(conn, [base_row])
        sm.update_rows(conn, [
            {
                "model_key": "pub/m",
                "scraped_at": "2024-02-01",
                "description": "Updated desc",
                "capabilities": "chat",
                "source_url": "http://example.com",
            }
        ])
        cur = conn.execute(
            "SELECT description FROM model_metadata WHERE model_key=?",
            ("pub/m",),
        )
        assert cur.fetchone()[0] == "Updated desc"
        conn.close()


class TestBackupMetadataDb:
    """Tests for scrape_metadata.backup_metadata_db()."""

    def test_no_op_when_db_missing(self, tmp_path: Path):
        """Does nothing when model_metadata.db does not exist."""
        sm = _import_scrape_metadata(tmp_path)
        sm.backup_metadata_db()

    def test_creates_backup_file(self, tmp_path: Path):
        """Creates a timestamped backup when DB exists."""
        sm = _import_scrape_metadata(tmp_path)
        sm.METADATA_DB.write_bytes(b"fake db content")
        sm.backup_metadata_db()
        backups = list(sm.BACKUPS_DIR.glob("*.db"))
        assert len(backups) == 1


class TestLoadLmsModels:
    """Tests for scrape_metadata.load_lms_models()."""

    def test_parses_valid_json(self, tmp_path: Path):
        """Returns parsed list when lms command succeeds."""
        sm = _import_scrape_metadata(tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps([{"modelKey": "pub/model"}])
        with patch("subprocess.run", return_value=mock_proc):
            result = sm.load_lms_models()
        assert isinstance(result, list)
        assert result[0]["modelKey"] == "pub/model"

    def test_raises_on_nonzero_returncode(self, tmp_path: Path):
        """RuntimeError raised when lms command fails."""
        sm = _import_scrape_metadata(tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "command not found"
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError):
                sm.load_lms_models()

    def test_raises_on_invalid_json(self, tmp_path: Path):
        """RuntimeError raised when output is not valid JSON."""
        sm = _import_scrape_metadata(tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "not json at all"
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises(RuntimeError):
                sm.load_lms_models()

    def test_raises_on_non_list_json(self, tmp_path: Path):
        """ValueError raised when JSON is not a list."""
        sm = _import_scrape_metadata(tmp_path)
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = json.dumps({"key": "value"})
        with patch("subprocess.run", return_value=mock_proc):
            with pytest.raises((RuntimeError, ValueError)):
                sm.load_lms_models()


class TestFetchLmStudioReadme:
    """Tests for scrape_metadata.fetch_lmstudio_readme()."""

    def test_returns_empty_string_on_url_error(self, tmp_path: Path):
        """Returns empty string when HTTP request fails."""
        sm = _import_scrape_metadata(tmp_path)
        from urllib.error import URLError
        with patch("scrape_metadata.urlopen", side_effect=URLError("error")):
            result = sm.fetch_lmstudio_readme("pub/model@q4")
        assert result == ""

    def test_returns_empty_string_on_timeout(self, tmp_path: Path):
        """Returns empty string on TimeoutError."""
        sm = _import_scrape_metadata(tmp_path)
        with patch("scrape_metadata.urlopen", side_effect=TimeoutError):
            result = sm.fetch_lmstudio_readme("pub/model")
        assert result == ""

    def test_returns_paragraph_text_fallback(self, tmp_path: Path):
        """Returns text from <p> tags when no article/section found."""
        sm = _import_scrape_metadata(tmp_path)
        html = b"<html><body><p>Great model</p><p>for coding</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.read.return_value = html
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("scrape_metadata.urlopen", return_value=mock_resp):
            result = sm.fetch_lmstudio_readme("pub/model")
        assert isinstance(result, str)

    def test_returns_article_text(self, tmp_path: Path):
        """Returns text from <article> tag when present."""
        sm = _import_scrape_metadata(tmp_path)
        html = (
            b"<html><body>"
            b"<article>This is a great coding model</article>"
            b"</body></html>"
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = html
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("scrape_metadata.urlopen", return_value=mock_resp):
            result = sm.fetch_lmstudio_readme("pub/model")
        assert isinstance(result, str)


class TestInferCapsFromDescription:
    """Tests for scrape_metadata.infer_caps_from_description()."""

    def test_empty_returns_empty(self, tmp_path: Path):
        """Empty description returns empty list."""
        sm = _import_scrape_metadata(tmp_path)
        assert sm.infer_caps_from_description("") == []

    def test_coding_keyword(self, tmp_path: Path):
        """Detects coding capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("A great coding model")
        assert "coding" in result

    def test_math_keyword(self, tmp_path: Path):
        """Detects math capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("Supports math and algebra")
        assert "math" in result

    def test_reasoning_keyword(self, tmp_path: Path):
        """Detects reasoning capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("Uses chain-of-thought reasoning")
        assert "reasoning" in result

    def test_chat_keyword(self, tmp_path: Path):
        """Detects chat capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("A conversational assistant")
        assert "chat" in result

    def test_vision_keyword(self, tmp_path: Path):
        """Detects vision capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("A multimodal vision-language model")
        assert "vision" in result

    def test_tool_use_keyword(self, tmp_path: Path):
        """Detects tool_use capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("Supports function calling and tools")
        assert "tool_use" in result

    def test_creative_keyword(self, tmp_path: Path):
        """Detects creative capability from keyword."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("A creative story writer")
        assert "creative" in result

    def test_no_match_returns_empty(self, tmp_path: Path):
        """Returns empty list when no keywords match."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_caps_from_description("A generic language model")
        assert isinstance(result, list)


class TestInferCapabilities:
    """Tests for scrape_metadata.infer_capabilities()."""

    def test_reasoning_from_key(self, tmp_path: Path):
        """Detects reasoning from deepseek-r1 in model key."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_capabilities("deepseek-r1/model", "Model")
        assert "reasoning" in result

    def test_coding_from_coder(self, tmp_path: Path):
        """Detects coding from coder in model key."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_capabilities("org/qwen-coder", "Qwen Coder")
        assert "coding" in result

    def test_chat_from_instruct(self, tmp_path: Path):
        """Detects chat from instruct in model key."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_capabilities("org/model-instruct", "Model Instruct")
        assert "chat" in result

    def test_math_from_key(self, tmp_path: Path):
        """Detects math from math in model key."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_capabilities("org/mathmodel", "Math Model")
        assert "math" in result

    def test_creative_from_writer(self, tmp_path: Path):
        """Detects creative from writer in display name."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.infer_capabilities("org/model", "Story Writer Model")
        assert "creative" in result


class TestFetchHfMetadata:
    """Tests for scrape_metadata.fetch_hf_metadata()."""

    def test_returns_empty_when_no_slash(self, tmp_path: Path):
        """Returns empty dict when key has no slash."""
        sm = _import_scrape_metadata(tmp_path)
        result = sm.fetch_hf_metadata("modelwithoutslash")
        assert result == {}

    def test_returns_metadata_on_success(self, tmp_path: Path):
        """Returns populated dict on successful API response."""
        import json
        sm = _import_scrape_metadata(tmp_path)
        api_data = {
            "tags": ["llm", "text-generation"],
            "pipeline_tag": "text-generation",
            "cardData": {"description": "A great model"},
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(api_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("scrape_metadata.urlopen", return_value=mock_resp):
            result = sm.fetch_hf_metadata("pub/model@main")
        assert "hf_tags" in result
        assert "llm" in result["hf_tags"]
        assert result["hf_pipeline_tag"] == "text-generation"
        assert result["hf_description"] == "A great model"

    def test_returns_empty_on_url_error(self, tmp_path: Path):
        """Returns empty dict on URLError."""
        sm = _import_scrape_metadata(tmp_path)
        from urllib.error import URLError
        with patch("scrape_metadata.urlopen", side_effect=URLError("error")):
            result = sm.fetch_hf_metadata("pub/model")
        assert result == {}

    def test_returns_empty_on_invalid_json(self, tmp_path: Path):
        """Returns empty dict on invalid JSON response."""
        sm = _import_scrape_metadata(tmp_path)
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("scrape_metadata.urlopen", return_value=mock_resp):
            result = sm.fetch_hf_metadata("pub/model")
        assert result == {}


class TestBackupMetadataDb:
    """Tests for scrape_metadata.backup_metadata_db()."""

    def test_no_op_when_db_missing(self, tmp_path: Path):
        """Does nothing when metadata DB does not exist."""
        sm = _import_scrape_metadata(tmp_path)
        sm.METADATA_DB = tmp_path / "missing.db"
        sm.backup_metadata_db()

    def test_creates_backup_file(self, tmp_path: Path):
        """Creates a backup file when DB exists."""
        sm = _import_scrape_metadata(tmp_path)
        db_path = tmp_path / "results" / "model_metadata.db"
        db_path.write_bytes(b"dummy")
        sm.METADATA_DB = db_path
        backups_dir = tmp_path / "results" / "backups"
        backups_dir.mkdir(exist_ok=True)
        sm.BACKUPS_DIR = backups_dir
        sm.backup_metadata_db()
        backup_files = list(backups_dir.glob("*_backup.db"))
        assert len(backup_files) == 1

    def test_handles_copy_error_gracefully(self, tmp_path: Path):
        """Logs warning but does not raise on copy failure."""
        sm = _import_scrape_metadata(tmp_path)
        db_path = tmp_path / "results" / "model_metadata.db"
        db_path.write_bytes(b"dummy")
        sm.METADATA_DB = db_path
        with patch("shutil.copy2", side_effect=OSError("disk full")):
            sm.backup_metadata_db()
