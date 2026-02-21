"""Tests for unified LLM client routing, validation, and retry behavior."""

import asyncio

import pytest

from algoforge.llm.exceptions import (
    LLMConfigurationError,
    LLMConnectionError,
    LLMResponseError,
)
from algoforge.llm.providers.base import CompletionRequest, CompletionResponse, LLMProvider
from algoforge.llm.unified_client import (
    UnifiedLLMClient,
    UnifiedLLMClientConfig,
    create_unified_client,
)


class ScriptedProvider(LLMProvider):
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def acompletion(self, request: CompletionRequest) -> CompletionResponse:
        self.calls += 1
        if not self.outcomes:
            raise RuntimeError("No scripted outcome left")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def completion(self, request: CompletionRequest) -> CompletionResponse:
        return asyncio.run(self.acompletion(request))

    def supports_model(self, model: str) -> bool:
        return True


def _ok_response(model: str) -> CompletionResponse:
    return CompletionResponse(
        content="ok",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        model=model,
        cost=0.123,
    )


def test_invalid_local_endpoint_url_fails_fast():
    with pytest.raises(LLMConfigurationError, match="Invalid endpoint URL"):
        UnifiedLLMClient(
            UnifiedLLMClientConfig(
                local_endpoints={"Qwen/Qwen3-30B": "localhost:8001/v1"},
            )
        )


def test_non_retryable_error_does_not_retry():
    client = UnifiedLLMClient(UnifiedLLMClientConfig(max_retries=3, retry_delay=0.001))
    provider = ScriptedProvider([LLMResponseError("bad schema"), _ok_response("openrouter/x")])
    client._cloud_provider = provider  # type: ignore[attr-defined]

    with pytest.raises(LLMResponseError):
        client.completion(
            model="openrouter/x",
            messages=[{"role": "user", "content": "hello"}],
        )

    assert provider.calls == 1


def test_retryable_error_retries_then_succeeds():
    client = UnifiedLLMClient(
        UnifiedLLMClientConfig(max_retries=3, retry_delay=0.001, retry_backoff=1.0)
    )
    provider = ScriptedProvider(
        [
            LLMConnectionError("temporary network"),
            LLMConnectionError("temporary network"),
            _ok_response("openrouter/x"),
        ]
    )
    client._cloud_provider = provider  # type: ignore[attr-defined]

    response = client.completion(
        model="openrouter/x",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.content == "ok"
    assert provider.calls == 3


def test_local_route_uses_local_provider():
    client = UnifiedLLMClient(
        UnifiedLLMClientConfig(
            local_endpoints={"Qwen/Qwen3-30B-A3B-Instruct-2507": "http://localhost:8001/v1"},
        )
    )
    local_provider = ScriptedProvider([_ok_response("Qwen/Qwen3-30B-A3B-Instruct-2507")])
    cloud_provider = ScriptedProvider([_ok_response("openrouter/x")])
    client._local_provider = local_provider  # type: ignore[attr-defined]
    client._cloud_provider = cloud_provider  # type: ignore[attr-defined]

    response = client.completion(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert response.model == "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert local_provider.calls == 1
    assert cloud_provider.calls == 0


def test_factory_accepts_cloud_registry_alias():
    client = create_unified_client(
        cloud_registry={"google/gemini-3-flash-preview": {"input_cost_per_token": 0.1}},
    )
    assert client._config.model_info == {  # type: ignore[attr-defined]
        "google/gemini-3-flash-preview": {"input_cost_per_token": 0.1}
    }
