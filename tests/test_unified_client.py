"""Tests for unified LLM client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from levi.llm.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from levi.llm.unified_client import (
    CompletionResponse,
    UnifiedLLMClient,
    UnifiedLLMClientConfig,
    _wrap_litellm_error,
    create_unified_client,
)


class TestCompletionResponse:
    def test_construction(self):
        resp = CompletionResponse(
            content="hello",
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            model="test/model",
            cost=0.001,
        )
        assert resp.content == "hello"
        assert resp.cost == 0.001


class TestWrapLitellmError:
    def test_timeout_error(self):
        err = _wrap_litellm_error("m", Exception("Request timeout"))
        assert isinstance(err, LLMTimeoutError)

    def test_rate_limit_error(self):
        err = _wrap_litellm_error("m", Exception("Rate limit exceeded"))
        assert isinstance(err, LLMRateLimitError)

    def test_auth_error(self):
        err = _wrap_litellm_error("m", Exception("Authentication failed"))
        assert isinstance(err, LLMAuthenticationError)

    def test_connection_error(self):
        err = _wrap_litellm_error("m", Exception("Connection refused"))
        assert isinstance(err, LLMConnectionError)

    def test_generic_error(self):
        err = _wrap_litellm_error("m", Exception("Something unknown"))
        assert isinstance(err, LLMResponseError)


class TestUnifiedLLMClient:
    def test_default_config(self):
        client = UnifiedLLMClient()
        assert client._config.temperature is None
        assert client._config.max_tokens == 16384
        assert client._config.timeout == 300.0

    def test_custom_config(self):
        config = UnifiedLLMClientConfig(temperature=0.5, max_tokens=4096, timeout=60.0)
        client = UnifiedLLMClient(config)
        assert client._config.temperature == 0.5
        assert client._config.max_tokens == 4096

    def test_cost_tracking(self):
        client = UnifiedLLMClient()
        assert client.total_cost == 0.0
        client._total_cost = 1.23
        assert client.total_cost == 1.23

    def test_reset_cost(self):
        client = UnifiedLLMClient()
        client._total_cost = 1.23
        returned = client.reset_cost()
        assert returned == 1.23
        assert client.total_cost == 0.0

    def test_acompletion_calls_litellm(self):
        """Verify acompletion routes through litellm.acompletion."""
        client = UnifiedLLMClient()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.llm.unified_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.005

            resp = asyncio.run(
                client.acompletion(
                    model="test/model",
                    messages=[{"role": "user", "content": "hello"}],
                    temperature=0.7,
                )
            )

        assert resp.content == "test response"
        assert resp.cost == 0.005
        assert resp.model == "test/model"
        assert client.total_cost == 0.005

        # Verify litellm was called with correct args
        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["model"] == "test/model"
        assert call_kwargs["temperature"] == 0.7

    def test_acompletion_wraps_litellm_errors(self):
        """Verify exceptions from litellm are wrapped."""
        client = UnifiedLLMClient()

        with patch("levi.llm.unified_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("Connection refused"))

            with pytest.raises(LLMConnectionError):
                asyncio.run(
                    client.acompletion(
                        model="test/model",
                        messages=[{"role": "user", "content": "hello"}],
                    )
                )

    def test_acompletion_omits_temperature_when_unset(self):
        """Provider defaults should apply unless temperature is explicitly configured."""
        client = UnifiedLLMClient()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.llm.unified_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.0

            asyncio.run(
                client.acompletion(
                    model="test/model",
                    messages=[{"role": "user", "content": "hello"}],
                )
            )

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert "temperature" not in call_kwargs

    def test_acompletion_extras_passed_through(self):
        """Verify extra kwargs are passed to litellm."""
        client = UnifiedLLMClient()

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.llm.unified_client.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.0

            asyncio.run(
                client.acompletion(
                    model="test/model",
                    messages=[{"role": "user", "content": "hello"}],
                    extra_body={"reasoning": {"enabled": False}},
                )
            )

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["extra_body"] == {"reasoning": {"enabled": False}}

    def test_close_is_noop(self):
        client = UnifiedLLMClient()
        asyncio.run(client.close())  # Should not raise


class TestFactory:
    def test_create_unified_client_defaults(self):
        client = create_unified_client()
        assert client._config.temperature is None
        assert client._config.max_tokens == 16384
        assert client._config.timeout == 300.0

    def test_create_unified_client_custom(self):
        client = create_unified_client(temperature=0.5, max_tokens=4096, timeout=60.0)
        assert client._config.temperature == 0.5
        assert client._config.max_tokens == 4096
        assert client._config.timeout == 60.0
