"""Tests for core/client.py."""
from typing import cast
from unittest.mock import MagicMock, patch

import httpx

from core.client import (
    LMStudioRESTClient,
    ModelCapabilities,
    ModelInfo,
    filter_llm_models,
    filter_tool_models,
    filter_vision_models,
    is_tool_model,
    is_vision_model,
)


def _make_model(
    model_type: str = "llm",
    vision: bool = False,
    tool_use: bool = False,
) -> ModelInfo:
    """Return a minimal ModelInfo instance for testing."""
    caps = ModelCapabilities(vision=vision, trained_for_tool_use=tool_use)
    return ModelInfo(
        model_type=model_type,
        publisher="test-pub",
        key="test/model",
        display_name="Test Model",
        capabilities=caps,
    )


class TestLMStudioRESTClientInit:
    """Tests for LMStudioRESTClient.__init__()."""

    def test_default_base_url(self):
        """Default base URL is localhost:1234."""
        client = LMStudioRESTClient()
        assert client.base_url == "http://localhost:1234"

    def test_trailing_slash_stripped(self):
        """Trailing slash is removed from base URL."""
        client = LMStudioRESTClient(base_url="http://localhost:1234/")
        assert client.base_url == "http://localhost:1234"

    def test_api_token_stored(self):
        """API token is stored on the instance."""
        client = LMStudioRESTClient(api_token="mytoken")
        assert client.api_token == "mytoken"

    def test_custom_timeout(self):
        """Custom timeout is respected."""
        client = LMStudioRESTClient(timeout=60.0)
        assert client.timeout == 60.0

    def test_enable_cache_default_false(self):
        """Response caching is disabled by default."""
        client = LMStudioRESTClient()
        assert client.enable_cache is False

    def test_last_response_id_none(self):
        """last_response_id starts as None."""
        client = LMStudioRESTClient()
        assert client.last_response_id is None


class TestHeaders:
    """Tests for LMStudioRESTClient._headers()."""

    def test_content_type_always_present(self):
        """Content-Type header is always present."""
        client = LMStudioRESTClient()
        headers = client._headers()
        assert headers["Content-Type"] == "application/json"

    def test_no_auth_when_no_token(self):
        """Authorization header is absent when no token is set."""
        client = LMStudioRESTClient()
        headers = client._headers()
        assert "Authorization" not in headers

    def test_bearer_token_added(self):
        """Bearer token is added when api_token is set."""
        client = LMStudioRESTClient(api_token="secret")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer secret"


class TestHealthCheck:
    """Tests for LMStudioRESTClient.health_check()."""

    def test_returns_true_on_200(self):
        """Returns True when server responds with 200."""
        client = LMStudioRESTClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client.client = MagicMock()
        client.client.get.return_value = mock_resp
        assert client.health_check() is True

    def test_returns_false_on_error_status(self):
        """Returns False when server responds with 500."""
        client = LMStudioRESTClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        client.client = MagicMock()
        client.client.get.return_value = mock_resp
        assert client.health_check() is False

    def test_returns_false_on_exception(self):
        """Returns False when a network exception is raised."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.side_effect = httpx.ConnectError("no connection")
        assert client.health_check() is False


class TestListModelsUnit:
    """Tests for LMStudioRESTClient.list_models()."""

    def _make_response(self, models_data: list) -> MagicMock:
        """Build a mock HTTP response returning models JSON."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": models_data}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_list_of_model_info(self):
        """Returns a list of ModelInfo objects."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.return_value = self._make_response([
            {
                "type": "llm",
                "publisher": "pub",
                "key": "pub/model",
                "display_name": "My Model",
            }
        ])
        result = client.list_models()
        assert len(result) == 1
        assert isinstance(result[0], ModelInfo)
        assert result[0].key == "pub/model"

    def test_empty_models_list(self):
        """Returns empty list when no models exist."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.return_value = self._make_response([])
        result = client.list_models()
        assert result == []

    def test_parses_quantization(self):
        """Quantization data is parsed into a Quantization object."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.return_value = self._make_response([
            {
                "type": "llm",
                "publisher": "pub",
                "key": "pub/model",
                "display_name": "Model",
                "quantization": {"name": "Q4_K_M", "bits_per_weight": 4.5},
            }
        ])
        result = client.list_models()
        assert result[0].quantization is not None
        assert result[0].quantization.name == "Q4_K_M"

    def test_parses_capabilities(self):
        """Capabilities data is parsed into a ModelCapabilities object."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.return_value = self._make_response([
            {
                "type": "llm",
                "publisher": "pub",
                "key": "pub/model",
                "display_name": "Model",
                "capabilities": {"vision": True, "trained_for_tool_use": False},
            }
        ])
        result = client.list_models()
        assert result[0].capabilities is not None
        assert result[0].capabilities.vision is True

    def test_parses_loaded_instances(self):
        """Loaded instances are parsed correctly."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.client.get.return_value = self._make_response([
            {
                "type": "llm",
                "publisher": "pub",
                "key": "pub/model",
                "display_name": "Model",
                "loaded_instances": [{"id": "inst-1", "config": {}}],
            }
        ])
        result = client.list_models()
        assert result[0].loaded_instances is not None
        assert len(result[0].loaded_instances) == 1
        assert result[0].loaded_instances[0].instance_id == "inst-1"


class TestLoadModel:
    """Tests for LMStudioRESTClient.load_model()."""

    def test_returns_instance_id(self):
        """Returns the instance_id from the load response."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"instance_id": "inst-abc"}
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        result = client.load_model("pub/model")
        assert result == "inst-abc"

    def test_preserves_quantization_suffix(self):
        """Exact model variant is forwarded to the load API."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"instance_id": "inst-1"}
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        client.load_model("pub/model@q4")
        call_args = client.client.post.call_args
        assert call_args[1]["json"]["model"] == "pub/model@q4"

    def test_context_length_added_when_provided(self):
        """context_length is included in payload when specified."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"instance_id": "inst-1"}
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        client.load_model("pub/model", context_length=4096)
        payload = client.client.post.call_args[1]["json"]
        assert payload["context_length"] == 4096

    def test_retries_without_kv_offload_for_embedding_models(self):
        """400 embedding offload errors are retried without kv-offload."""
        client = LMStudioRESTClient()
        client.client = MagicMock()

        request = httpx.Request("POST", "http://localhost:1234/api/v1/models/load")
        error_response = httpx.Response(
            400,
            request=request,
            json={
                "error": {
                    "type": "invalid_request",
                    "message": (
                        "The following configuration values are not "
                        "supported for embedding models: "
                        "offload_kv_cache_to_gpu"
                    ),
                }
            },
        )
        error = httpx.HTTPStatusError(
            "Client error '400 Bad Request'",
            request=request,
            response=error_response,
        )

        first_response = MagicMock()
        first_response.raise_for_status.side_effect = error

        second_response = MagicMock()
        second_response.raise_for_status = MagicMock()
        second_response.json.return_value = {"instance_id": "inst-embedding"}

        client.client.post.side_effect = [first_response, second_response]

        result = client.load_model(
            "text-embedding-model@q4",
            context_length=2048,
            gpu_offload=1.0,
        )

        assert result == "inst-embedding"
        assert client.client.post.call_count == 2

        first_payload = client.client.post.call_args_list[0][1]["json"]
        second_payload = client.client.post.call_args_list[1][1]["json"]

        assert first_payload["offload_kv_cache_to_gpu"] is True
        assert "offload_kv_cache_to_gpu" not in second_payload


class TestUnloadModel:
    """Tests for LMStudioRESTClient.unload_model()."""

    def test_returns_true_on_success(self):
        """Returns True after successful unload."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        result = client.unload_model("inst-abc")
        assert result is True


class TestDownloadModel:
    """Tests for LMStudioRESTClient.download_model()."""

    def test_already_downloaded_returns_true(self):
        """Returns True immediately when model is already downloaded."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "already_downloaded"}
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        result = client.download_model("pub/model")
        assert result is True

    def test_starts_download_returns_true(self):
        """Returns True when download is started (no wait)."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "downloading"}
        mock_resp.raise_for_status = MagicMock()
        client.client.post.return_value = mock_resp
        result = client.download_model("pub/model", wait_for_completion=False)
        assert result is True


class TestDownloadStatus:
    """Tests for LMStudioRESTClient.download_status()."""

    def test_returns_status_dict(self):
        """Returns dict with download status info."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "downloading", "progress": 50}
        mock_resp.raise_for_status = MagicMock()
        client.client.get.return_value = mock_resp
        result = client.download_status("pub/model")
        assert result["status"] == "downloading"


class TestDownloadProgressPolling:
    """Tests for _poll_download_progress()."""

    def test_poll_download_progress_completed(self):
        """Polling returns True when status becomes completed."""
        client = LMStudioRESTClient()
        callback_calls = []

        def callback(payload):
            callback_calls.append(payload)

        with patch.object(
            client,
            "download_status",
            return_value={"status": "completed", "progress": 100},
        ):
            result = client._poll_download_progress(
                "pub/model", progress_callback=callback
            )

        assert result is True
        assert callback_calls

    def test_poll_download_progress_failed(self):
        """Polling returns False when status becomes failed."""
        client = LMStudioRESTClient()

        with patch.object(
            client,
            "download_status",
            return_value={"status": "failed", "progress": 55},
        ):
            result = client._poll_download_progress("pub/model")

        assert result is False

    def test_poll_download_progress_handles_http_error_then_completes(self):
        """HTTP polling errors are retried and then completion returns True."""
        client = LMStudioRESTClient()
        err = httpx.HTTPError("temporary")

        with patch.object(
            client,
            "download_status",
            side_effect=[err, {"status": "already_downloaded"}],
        ), patch("core.client.time.sleep", return_value=None):
            result = client._poll_download_progress("pub/model")

        assert result is True


class TestCacheOperations:
    """Tests for cache-related methods."""

    def test_clear_cache_returns_zero_when_empty(self):
        """clear_cache returns 0 when nothing is cached."""
        from core.client import _RESPONSE_CACHE
        _RESPONSE_CACHE.clear()
        client = LMStudioRESTClient()
        count = client.clear_cache()
        assert count == 0

    def test_make_cache_key_is_deterministic(self):
        """Same inputs produce the same cache key."""
        client = LMStudioRESTClient()
        messages = [{"role": "user", "content": "hello"}]
        key1 = client._make_cache_key(messages, "model", 0.7)
        key2 = client._make_cache_key(messages, "model", 0.7)
        assert key1 == key2

    def test_make_cache_key_differs_for_different_inputs(self):
        """Different inputs produce different cache keys."""
        client = LMStudioRESTClient()
        messages_a = [{"role": "user", "content": "hello"}]
        messages_b = [{"role": "user", "content": "world"}]
        key_a = client._make_cache_key(messages_a, "model", 0.7)
        key_b = client._make_cache_key(messages_b, "model", 0.7)
        assert key_a != key_b


class TestResetStatefulChat:
    """Tests for LMStudioRESTClient.reset_stateful_chat()."""

    def test_clears_last_response_id(self):
        """last_response_id is set to None after reset."""
        client = LMStudioRESTClient()
        client.last_response_id = "some-id"
        client.reset_stateful_chat()
        assert client.last_response_id is None


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self):
        """__enter__ returns the client itself."""
        client = LMStudioRESTClient()
        result = client.__enter__()
        assert result is client

    def test_exit_calls_close(self):
        """__exit__ closes the HTTP client."""
        client = LMStudioRESTClient()
        client.client = MagicMock()
        client.__exit__(None, None, None)
        client.client.close.assert_called_once()


class TestModelInfoDefaults:
    """Tests for ModelInfo dataclass default values."""

    def test_loaded_instances_defaults_to_empty_list(self):
        """loaded_instances defaults to an empty list."""
        info = ModelInfo(
            model_type="llm",
            publisher="p",
            key="k",
            display_name="d",
        )
        assert info.loaded_instances == []


class TestIsVisionModel:
    """Tests for is_vision_model()."""

    def test_true_when_vision_capability(self):
        """Returns True for model with vision capability."""
        model = _make_model(vision=True)
        assert is_vision_model(model) is True

    def test_false_when_no_vision(self):
        """Returns False for model without vision capability."""
        model = _make_model(vision=False)
        assert is_vision_model(model) is False

    def test_false_when_capabilities_none(self):
        """Returns False when capabilities is None."""
        model = _make_model()
        model.capabilities = None
        assert is_vision_model(model) is False


class TestIsToolModel:
    """Tests for is_tool_model()."""

    def test_true_when_tool_capability(self):
        """Returns True for model trained for tool use."""
        model = _make_model(tool_use=True)
        assert is_tool_model(model) is True

    def test_false_when_no_tool(self):
        """Returns False for model not trained for tool use."""
        model = _make_model(tool_use=False)
        assert is_tool_model(model) is False

    def test_false_when_capabilities_none(self):
        """Returns False when capabilities is None."""
        model = _make_model()
        model.capabilities = None
        assert is_tool_model(model) is False


class TestFilterModels:
    """Tests for filter_*_models() functions."""

    def test_filter_llm_models_excludes_embeddings(self):
        """filter_llm_models excludes non-llm model types."""
        models = [
            _make_model(model_type="llm"),
            _make_model(model_type="embedding"),
        ]
        result = filter_llm_models(models)
        assert len(result) == 1
        assert result[0].model_type == "llm"

    def test_filter_vision_models_returns_only_vision(self):
        """filter_vision_models returns only vision-capable models."""
        models = [
            _make_model(vision=True),
            _make_model(vision=False),
        ]
        result = filter_vision_models(models)
        assert len(result) == 1
        caps = result[0].capabilities
        if caps is None:
            raise AssertionError("capabilities must not be None")
        assert caps.vision is True

    def test_filter_tool_models_returns_only_tool(self):
        """filter_tool_models returns only tool-capable models."""
        models = [
            _make_model(tool_use=True),
            _make_model(tool_use=False),
        ]
        result = filter_tool_models(models)
        assert len(result) == 1
        caps = result[0].capabilities
        if caps is None:
            raise AssertionError("capabilities must not be None")
        assert caps.trained_for_tool_use is True

    def test_filter_returns_empty_on_empty_input(self):
        """All filter functions return empty list for empty input."""
        assert filter_llm_models([]) == []
        assert filter_vision_models([]) == []
        assert filter_tool_models([]) == []


class TestChatStreamMethod:
    """Tests for LMStudioRESTClient.chat_stream() method."""

    def _make_sse_response(self, events: list) -> MagicMock:
        """Build a mock streaming HTTP response."""
        import json as _json
        lines = []
        for event in events:
            lines.append(f"data: {_json.dumps(event)}")
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_chat_stream_returns_text_and_stats(self):
        """chat_stream() returns text and stats on successful response."""
        client = LMStudioRESTClient(enable_cache=False)
        event = {
            "type": "chat.end",
            "result": {
                "output": [{"type": "message", "content": "Hello world"}],
                "stats": {
                    "input_tokens": 10,
                    "total_output_tokens": 5,
                    "time_to_first_token_seconds": 0.1,
                    "tokens_per_second": 25.0,
                },
            },
        }
        mock_resp = self._make_sse_response([event])
        with patch.object(client.client, "stream", return_value=mock_resp):
            result = client.chat_stream(
                messages=[{"role": "user", "content": "Hi"}],
                model="test/model@q4",
            )
        assert result["text"] == "Hello world"
        assert "stats" in result
        assert result["stats"].tokens_per_second == 25.0

    def test_chat_stream_with_caching(self):
        """chat_stream() returns cached result on second call."""
        import core.client as rc
        rc._RESPONSE_CACHE.clear()
        client = LMStudioRESTClient(enable_cache=True)
        event = {
            "type": "chat.end",
            "result": {
                "output": [{"type": "message", "content": "cached"}],
                "stats": {
                    "input_tokens": 1, "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 10.0,
                },
            },
        }
        mock_resp = self._make_sse_response([event])
        with patch.object(client.client, "stream", return_value=mock_resp):
            client.chat_stream(
                messages=[{"role": "user", "content": "q"}]
            )
        with patch.object(client.client, "stream") as mock_stream:
            result2 = client.chat_stream(
                messages=[{"role": "user", "content": "q"}]
            )
        mock_stream.assert_not_called()
        assert result2["text"] == "cached"
        rc._RESPONSE_CACHE.clear()

    def test_chat_stream_with_on_chunk_callback(self):
        """chat_stream() calls on_chunk for each content chunk."""
        client = LMStudioRESTClient(enable_cache=False)
        chunks_received = []

        def on_chunk(text, event):
            chunks_received.append(text)

        chunk_event = {
            "type": "chat.chunk",
            "output": [{"type": "message", "content": "chunk1"}],
        }
        end_event = {
            "type": "chat.end",
            "result": {
                "output": [],
                "stats": {
                    "input_tokens": 1, "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 10.0,
                },
            },
        }
        mock_resp = self._make_sse_response([chunk_event, end_event])
        with patch.object(client.client, "stream", return_value=mock_resp):
            client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                on_chunk=on_chunk,
            )
        assert "chunk1" in chunks_received

    def test_chat_stream_stateful_saves_response_id(self):
        """chat_stream() with use_stateful saves response_id."""
        client = LMStudioRESTClient(enable_cache=False)
        event = {
            "type": "chat.end",
            "result": {
                "response_id": "resp-123",
                "output": [],
                "stats": {
                    "input_tokens": 1, "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 10.0,
                },
            },
        }
        mock_resp = self._make_sse_response([event])
        with patch.object(client.client, "stream", return_value=mock_resp):
            client.chat_stream(
                messages=[{"role": "user", "content": "hi"}],
                use_stateful=True,
            )
        assert client.last_response_id == "resp-123"

    def test_chat_stream_sets_optional_sampling_payload_fields(self):
        """Optional sampling and integration fields are forwarded."""
        client = LMStudioRESTClient(enable_cache=False)
        captured_payload = {}

        end_event = {
            "type": "chat.end",
            "result": {
                "output": [{"type": "message", "content": "ok"}],
                "stats": {
                    "input_tokens": 1,
                    "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 5.0,
                },
            },
        }
        mock_resp = self._make_sse_response([end_event])

        def capture_stream(*_args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_resp

        client.last_response_id = "prev-1"
        with patch.object(client.client, "stream", side_effect=capture_stream):
            result = client.chat_stream(
                messages=[{"role": "user", "content": "hello"}],
                model="test/model@q4",
                top_k=20,
                top_p=0.9,
                min_p=0.1,
                repeat_penalty=1.1,
                use_stateful=True,
                mcp_integrations=[{"name": "tool-a"}],
            )

        assert result["text"] == "ok"
        assert captured_payload["top_k"] == 20
        assert captured_payload["top_p"] == 0.9
        assert captured_payload["min_p"] == 0.1
        assert captured_payload["repeat_penalty"] == 1.1
        assert captured_payload["previous_response_id"] == "prev-1"
        assert captured_payload["integrations"] == [{"name": "tool-a"}]

    def test_chat_stream_preserves_quantization_suffix(self):
        """Exact model variant is forwarded to the chat API."""
        client = LMStudioRESTClient(enable_cache=False)
        captured_payload = {}

        end_event = {
            "type": "chat.end",
            "result": {
                "output": [{"type": "message", "content": "ok"}],
                "stats": {
                    "input_tokens": 1,
                    "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 5.0,
                },
            },
        }
        mock_resp = self._make_sse_response([end_event])

        def capture_stream(*_args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_resp

        with patch.object(client.client, "stream", side_effect=capture_stream):
            client.chat_stream(
                messages=[{"role": "user", "content": "hello"}],
                model="test/model@q4_k_m",
            )

        assert captured_payload["model"] == "test/model@q4_k_m"

    def test_chat_stream_handles_invalid_message_payload(self):
        """Invalid message shapes fall back to empty input text."""
        client = LMStudioRESTClient(enable_cache=False)
        captured_payload = {}

        end_event = {
            "type": "chat.end",
            "result": {
                "output": [{"type": "message", "content": "ok"}],
                "stats": {
                    "input_tokens": 1,
                    "total_output_tokens": 1,
                    "time_to_first_token_seconds": 0.0,
                    "tokens_per_second": 5.0,
                },
            },
        }
        mock_resp = self._make_sse_response([end_event])

        def capture_stream(*_args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return mock_resp

        with patch.object(client.client, "stream", side_effect=capture_stream):
            client.chat_stream(messages=cast(list[dict[str, str]], [None]))

        assert captured_payload["input"] == ""

    def test_chat_stream_ignores_non_data_lines_and_bad_json(self):
        """Blank lines and malformed data events are skipped safely."""
        client = LMStudioRESTClient(enable_cache=False)
        lines = [
            "",
            "data: {bad_json}",
            "event: ping",
            (
                'data: {"type": "chat.end", "result": '
                '{"output": [], "stats": '
                '{"input_tokens": 1, "total_output_tokens": 1, '
                '"time_to_first_token_seconds": 0.0, '
                '"tokens_per_second": 1.0}}}'
            ),
        ]

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch.object(client.client, "stream", return_value=mock_resp):
            result = client.chat_stream(messages=[{"role": "user", "content": "x"}])

        assert result["stats"].tokens_per_second == 1.0


class TestListModels:
    """Tests for LMStudioRESTClient.list_models() method."""

    def test_list_models_returns_models(self):
        """list_models returns list of ModelInfo objects from API."""
        client = LMStudioRESTClient()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {
                    "type": "llm",
                    "publisher": "test",
                    "key": "test/model",
                    "display_name": "Test Model",
                }
            ]
        }
        with patch.object(client.client, "get", return_value=mock_resp):
            result = client.list_models()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].key == "test/model"

    def test_load_model_returns_instance_id(self):
        """load_model returns instance_id string on success."""
        client = LMStudioRESTClient()
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"instance_id": "inst-abc"}
        with patch.object(client.client, "post", return_value=mock_resp):
            result = client.load_model(model_key="test/model@q4")
        assert result == "inst-abc"
