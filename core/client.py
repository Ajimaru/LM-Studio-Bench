"""
LM Studio REST API Client (v1).

This module provides a client for interacting with LM Studio's native
REST API (/api/v1/*) endpoints, including model management, chat,
and capability detection.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class ModelCapabilities:
    """Model capabilities from /api/v1/models."""

    vision: bool = False
    trained_for_tool_use: bool = False


@dataclass
class Quantization:
    """Quantization info."""

    name: Optional[str] = None
    bits_per_weight: Optional[float] = None


@dataclass
class LoadedInstance:
    """Loaded model instance info."""

    instance_id: str
    config: Dict[str, Any]


@dataclass
class ModelInfo:
    """Model information from /api/v1/models."""

    model_type: str
    publisher: str
    key: str
    display_name: str
    architecture: Optional[str] = None
    quantization: Optional[Quantization] = None
    size_bytes: int = 0
    params_string: Optional[str] = None
    loaded_instances: Optional[List[LoadedInstance]] = None
    max_context_length: int = 0
    format: Optional[str] = None
    capabilities: Optional[ModelCapabilities] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Initialize default list."""
        if self.loaded_instances is None:
            self.loaded_instances = []


@dataclass
class ChatStats:
    """Statistics from /api/v1/chat response."""

    tokens_in: int = 0
    tokens_out: int = 0
    time_to_first_token_ms: float = 0.0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


class LMStudioRESTClient:
    """Client for LM Studio REST API v1."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        api_token: Optional[str] = None,
        timeout: float = 300.0,
        enable_cache: bool = False,
    ):
        """
        Initialize REST client.

        Args:
            base_url: LM Studio server base URL
            api_token: Optional permission key for authentication
            timeout: Request timeout in seconds
            enable_cache: Enable response caching (for stateful chats)
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.client = httpx.Client(timeout=timeout)
        self.last_response_id: Optional[str] = None

    def _headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        return headers

    def health_check(self) -> bool:
        """
        Check if LM Studio server is running.

        Returns:
            True if server is accessible
        """
        try:
            response = self.client.get(f"{self.base_url}/", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError as exc:
            logger.debug("Health check failed: %s", exc)
            return False

    def list_models(self) -> List[ModelInfo]:
        """
        Get list of available models.

        Returns:
            List of ModelInfo objects

        Raises:
            httpx.HTTPError: on API errors
        """
        response = self.client.get(
            f"{self.base_url}/api/v1/models",
            headers=self._headers(),
        )
        response.raise_for_status()

        data = response.json()
        models = []

        for item in data.get("models", []):
            quant_data = item.get("quantization")
            quantization = None
            if quant_data:
                quantization = Quantization(
                    name=quant_data.get("name"),
                    bits_per_weight=quant_data.get("bits_per_weight"),
                )

            caps_data = item.get("capabilities")
            capabilities = None
            if caps_data:
                capabilities = ModelCapabilities(
                    vision=caps_data.get("vision", False),
                    trained_for_tool_use=caps_data.get("trained_for_tool_use", False),
                )

            loaded_instances = []
            for inst in item.get("loaded_instances", []):
                loaded_instances.append(
                    LoadedInstance(
                        instance_id=inst["id"],
                        config=inst.get("config", {}),
                    )
                )

            model = ModelInfo(
                model_type=item["type"],
                publisher=item["publisher"],
                key=item["key"],
                display_name=item["display_name"],
                architecture=item.get("architecture"),
                quantization=quantization,
                size_bytes=item.get("size_bytes", 0),
                params_string=item.get("params_string"),
                loaded_instances=loaded_instances,
                max_context_length=item.get("max_context_length", 0),
                format=item.get("format"),
                capabilities=capabilities,
                description=item.get("description"),
            )
            models.append(model)

        logger.info("🔍 Found %s models", len(models))
        return models

    def load_model(
        self,
        model_key: str,
        context_length: Optional[int] = None,
        n_parallel: Optional[int] = None,
        unified_kv_cache: Optional[bool] = None,
        gpu_offload: Optional[float] = None,
    ) -> str:
        """
        Load a model into memory.

        Args:
            model_key: Model key (with or without @quantization suffix)
            context_length: Max context length (tokens)
            n_parallel: Max concurrent predictions
            unified_kv_cache: Enable unified KV cache
            gpu_offload: GPU offload ratio (0.0-1.0)

        Returns:
            instance_id of loaded model

        Raises:
            httpx.HTTPError: on API errors
            RuntimeError: If model loading fails without an HTTP response
        """
        payload: Dict[str, Any] = {"model": model_key}

        if context_length is not None:
            payload["context_length"] = context_length

        if n_parallel is not None:
            payload["n_parallel"] = n_parallel

        if unified_kv_cache is not None:
            payload["unified_kv_cache"] = unified_kv_cache

        if gpu_offload is not None:
            payload["offload_kv_cache_to_gpu"] = bool(gpu_offload)

        logger.info("📥 Loading model: %s", model_key)
        load_endpoints = [
            "/api/v1/models/load",
            "/api/v0/model/load",
        ]
        response: Optional[httpx.Response] = None
        last_error: Optional[httpx.HTTPStatusError] = None

        for index, endpoint in enumerate(load_endpoints):
            url = f"{self.base_url}{endpoint}"
            try:
                response = self.client.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as error:
                if self._should_retry_without_kv_offload(error, payload):
                    retry_payload = dict(payload)
                    retry_payload.pop("offload_kv_cache_to_gpu", None)
                    logger.warning(
                        "⚠️ Retrying model load without "
                        "offload_kv_cache_to_gpu: %s",
                        model_key,
                    )
                    try:
                        response = self.client.post(
                            url,
                            headers=self._headers(),
                            json=retry_payload,
                        )
                        response.raise_for_status()
                        break
                    except httpx.HTTPStatusError as retry_error:
                        response_obj = retry_error.response
                        is_not_found = (
                            response_obj is not None
                            and response_obj.status_code == 404
                        )
                        has_fallback = (
                            index < len(load_endpoints) - 1
                        )
                        if is_not_found and has_fallback:
                            fallback_endpoint = load_endpoints[
                                index + 1
                            ]
                            logger.warning(
                                "⚠️ Model load retry on %s returned "
                                "404. Trying fallback endpoint %s",
                                endpoint,
                                fallback_endpoint,
                            )
                            continue

                        last_error = retry_error
                        break
                response_obj = error.response
                is_not_found = (
                    response_obj is not None
                    and response_obj.status_code == 404
                )
                has_fallback = index < len(load_endpoints) - 1
                if is_not_found and has_fallback:
                    fallback_endpoint = load_endpoints[index + 1]
                    logger.warning(
                        "⚠️ Model load endpoint %s returned 404. "
                        "Trying fallback endpoint %s",
                        endpoint,
                        fallback_endpoint,
                    )
                    continue

                last_error = error
                break

        if response is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Model load failed without HTTP response")

        result = response.json()
        instance_id = result.get("instance_id", model_key)
        logger.info("✅ Model loaded: %s", instance_id)
        return instance_id

    @staticmethod
    def _should_retry_without_kv_offload(
        error: httpx.HTTPStatusError,
        payload: Dict[str, Any],
    ) -> bool:
        """Return True for embedding-specific kv-offload incompatibility."""
        response = error.response
        if response is None or response.status_code != 400:
            return False
        if "offload_kv_cache_to_gpu" not in payload:
            return False

        try:
            response_json = response.json()
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            return False

        error_block = response_json.get("error") or {}
        message = str(error_block.get("message", "")).lower()
        return (
            "embedding model" in message
            and "offload_kv_cache_to_gpu" in message
            and "not supported" in message
        )

    def unload_model(self, instance_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            instance_id: Instance ID from load_model()

        Returns:
            True if unload successful

        Raises:
            httpx.HTTPError: on API errors
            RuntimeError: If model unload fails without an HTTP response
        """
        payload = {"instance_id": instance_id}

        logger.info("📤 Unloading model: %s", instance_id)
        unload_endpoints = [
            "/api/v1/models/unload",
            "/api/v0/model/unload",
        ]
        response: Optional[httpx.Response] = None
        last_error: Optional[httpx.HTTPStatusError] = None

        for index, endpoint in enumerate(unload_endpoints):
            try:
                response = self.client.post(
                    f"{self.base_url}{endpoint}",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as error:
                response_obj = error.response
                is_not_found = (
                    response_obj is not None
                    and response_obj.status_code == 404
                )
                has_fallback = index < len(unload_endpoints) - 1
                if is_not_found and has_fallback:
                    fallback_endpoint = unload_endpoints[index + 1]
                    logger.warning(
                        "⚠️ Model unload endpoint %s returned 404. "
                        "Trying fallback endpoint %s",
                        endpoint,
                        fallback_endpoint,
                    )
                    continue
                last_error = error
                break

        if response is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("Model unload failed without HTTP response")

        logger.info("✅ Model unloaded: %s", instance_id)
        return True

    def download_model(
        self,
        model_key: str,
        wait_for_completion: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Download a model with optional progress tracking.

        Args:
            model_key: Model identifier
            wait_for_completion: If True, polls until download completes
            progress_callback: Called with status dict during polling

        Returns:
            True if download completed (or started if not waiting)

        Raises:
            httpx.HTTPError: on API errors
        """
        payload = {"model": model_key}

        logger.info("Downloading model: %s", model_key)
        response = self.client.post(
            f"{self.base_url}/api/v1/models/download",
            headers=self._headers(),
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "already_downloaded":
            logger.info("Model already downloaded: %s", model_key)
            return True

        logger.info("Download started: %s", model_key)

        if wait_for_completion:
            return self._poll_download_progress(model_key, progress_callback)

        return True

    def _poll_download_progress(
        self,
        model_key: str,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """
        Poll download status until completion.

        Args:
            model_key: Model identifier
            progress_callback: Called with status updates

        Returns:
            True if download completed successfully
        """
        while True:
            try:
                status = self.download_status(model_key)
                current_status = status.get("status")

                if progress_callback:
                    progress_callback(status)

                if current_status == "completed":
                    logger.info("Download completed: %s", model_key)
                    return True
                if current_status == "failed":
                    logger.error("Download failed: %s", model_key)
                    return False
                if current_status == "already_downloaded":
                    logger.info("Model already downloaded: %s", model_key)
                    return True

                time.sleep(2)

            except httpx.HTTPError as exc:
                logger.warning("Error polling download status: %s", exc)
                time.sleep(2)

    def download_status(self, model_key: str) -> Dict[str, Any]:
        """
        Get download status for a model.

        Args:
            model_key: Model identifier

        Returns:
            Status dict with progress info

        Raises:
            httpx.HTTPError: on API errors
        """
        response = self.client.get(
            f"{self.base_url}/api/v1/models/download/status",
            headers=self._headers(),
            params={"model": model_key},
        )
        response.raise_for_status()

        return response.json()

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        context_length: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        on_chunk: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        use_stateful: bool = False,
        mcp_integrations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request with streaming.

        Args:
            messages: Chat messages (role, content)
            model: Model key (with or without @quantization suffix)
            context_length: Context length override
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            top_k: Top-k sampling override
            top_p: Top-p sampling override
            min_p: Min-p sampling override
            repeat_penalty: Repeat penalty override
            on_chunk: Callback for streaming chunks (text, event_data)
            use_stateful: Enable stateful chat (uses previous_response_id)
            mcp_integrations: List of MCP server integrations

        Returns:
            Final response with stats and response_id

        Raises:
            httpx.HTTPError: on API errors
        """
        cache_key = None
        if self.enable_cache and not use_stateful and not mcp_integrations:
            cache_key = self._make_cache_key(messages, model, temperature)
            if cache_key in _RESPONSE_CACHE:
                logger.debug("Returning cached response")
                return _RESPONSE_CACHE[cache_key]
        payload: Dict[str, Any] = {"temperature": temperature, "stream": True}

        if model:
            payload["model"] = model
        if context_length:
            payload["context_length"] = context_length
        if max_tokens:
            payload["max_output_tokens"] = max_tokens
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
        if min_p is not None:
            payload["min_p"] = min_p
        if repeat_penalty is not None:
            payload["repeat_penalty"] = repeat_penalty

        if messages:
            try:
                input_text = "\n\n".join([m.get("content", "") for m in messages])
                payload["input"] = input_text
            except (TypeError, AttributeError):
                payload["input"] = ""

        if use_stateful and self.last_response_id:
            payload["previous_response_id"] = self.last_response_id
            logger.debug("Using stateful chat: %s", self.last_response_id)

        if mcp_integrations:
            payload["integrations"] = mcp_integrations
            logger.debug("Using %s MCP integrations", len(mcp_integrations))

        logger.debug("Sending chat request: %s messages", len(messages))

        full_text = ""
        final_stats = None
        final_event = None
        start_time = time.time()

        with self.client.stream(
            "POST",
            f"{self.base_url}/api/v1/chat",
            headers=self._headers(),
            json=payload,
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line.strip():
                    continue

                if line.startswith("data: "):
                    json_str = line[6:]
                    try:
                        event = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse event: %s", json_str)
                        continue

                    if event.get("type") == "chat.end":
                        final_event = event
                        result_data = event.get("result", {})
                        if "output" in result_data:
                            msg_content = ""
                            reasoning_fallback = ""
                            for item in result_data["output"]:
                                item_type = item.get("type", "")
                                content = item.get("content", "")
                                if item_type == "message" and content:
                                    msg_content += content
                                elif item_type == "reasoning" and content:
                                    reasoning_fallback += content
                            final_content = msg_content or reasoning_fallback
                            if final_content and final_content not in full_text:
                                full_text += final_content
                        if "stats" in result_data:
                            final_stats = result_data["stats"]
                    else:
                        event_type = event.get("type", "")
                        if event_type == "message.delta":
                            chunk_text = event.get("content", "")
                        else:
                            chunk_text = ""
                            if "output" in event:
                                for item in event["output"]:
                                    if item.get("type") == "message":
                                        chunk_text += item.get("content", "")

                        if chunk_text:
                            full_text += chunk_text
                            if on_chunk:
                                on_chunk(chunk_text, event)

        total_time = time.time() - start_time

        result = {
            "text": full_text,
            "total_time_s": total_time,
        }

        if final_stats:
            result["stats"] = ChatStats(
                tokens_in=final_stats.get("input_tokens", 0),
                tokens_out=final_stats.get("total_output_tokens", 0),
                time_to_first_token_ms=final_stats.get(
                    "time_to_first_token_seconds", 0.0
                )
                * 1000.0,
                total_time_ms=total_time * 1000.0,
                tokens_per_second=final_stats.get("tokens_per_second", 0.0),
            )

        response_id = None
        if final_event:
            response_id = final_event.get("result", {}).get("response_id")
        if response_id:
            result["response_id"] = response_id
            if use_stateful:
                self.last_response_id = response_id
                logger.debug("Saved response_id for stateful chat: %s", response_id)

        if self.enable_cache and cache_key:
            _RESPONSE_CACHE[cache_key] = result
            logger.debug("Cached response (cache size: %s)", len(_RESPONSE_CACHE))

        logger.info(
            "Chat complete: %s tokens in %.2fs",
            result.get("stats", ChatStats()).tokens_out,
            total_time,
        )

        return result

    def _make_cache_key(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
    ) -> str:
        """
        Generate cache key for response caching.

        Args:
            messages: Chat messages
            model: Model identifier
            temperature: Sampling temperature

        Returns:
            Cache key string
        """

        content = json.dumps(
            {"messages": messages, "model": model, "temp": temperature}, sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def clear_cache(self) -> int:
        """
        Clear response cache.

        Returns:
            Number of cached items removed
        """
        count = len(_RESPONSE_CACHE)
        _RESPONSE_CACHE.clear()
        logger.info("Cleared %s cached responses", count)
        return count

    def reset_stateful_chat(self):
        """
        Reset stateful chat history (clears last_response_id).
        """
        self.last_response_id = None
        logger.debug("Reset stateful chat history")

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *_args):
        """Context manager exit."""
        self.close()


def is_vision_model(model: ModelInfo) -> bool:
    """Check if model supports vision/image inputs."""
    return model.capabilities is not None and model.capabilities.vision


def is_tool_model(model: ModelInfo) -> bool:
    """Check if model was trained for tool/function calling."""
    return model.capabilities is not None and model.capabilities.trained_for_tool_use


def filter_llm_models(models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter list for LLM models only (exclude embeddings)."""
    return [m for m in models if m.model_type == "llm"]


def filter_vision_models(models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter list for vision-capable models."""
    return [m for m in models if is_vision_model(m)]


def filter_tool_models(models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter list for tool-capable models."""
    return [m for m in models if is_tool_model(m)]
