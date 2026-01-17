"""OpenAI-compatible provider for local models (vLLM, SGLang, Ollama)."""

import asyncio
from typing import Optional

import httpx

from .base import LLMProvider, CompletionRequest, CompletionResponse


class OpenAICompatibleProvider(LLMProvider):
    """Provider for local OpenAI-compatible servers (vLLM, SGLang, Ollama)."""

    def __init__(self, endpoints: dict[str, str]):
        """
        Args:
            endpoints: Map of model name -> base URL
                e.g., {"Qwen/Qwen3-30B": "http://10.142.0.3:8000/v1"}
        """
        self._endpoints = endpoints
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy initialization of async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        return self._client

    def supports_model(self, model: str) -> bool:
        return model in self._endpoints

    async def acompletion(self, request: CompletionRequest) -> CompletionResponse:
        base_url = self._endpoints[request.model]
        url = f"{base_url}/chat/completions"

        payload = {
            "model": request.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = request.stop

        # Add any extra parameters
        payload.update(request.extras)

        client = self._get_client()
        response = await client.post(
            url,
            json=payload,
            timeout=request.timeout,
        )
        response.raise_for_status()
        data = response.json()

        usage = data.get("usage", {})
        return CompletionResponse(
            content=data["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=request.model,
            cost=0.0,  # Local models are free
            raw_response=data,
        )

    def completion(self, request: CompletionRequest) -> CompletionResponse:
        """Sync wrapper for acompletion."""
        return asyncio.run(self.acompletion(request))

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
