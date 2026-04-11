"""Tests for the LM abstraction."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from levi.clients import ClientResult, LM


class TestLM:
    def test_default_config(self):
        lm = LM("test/model")
        assert lm.timeout == 300.0
        assert lm.defaults == {}

    def test_custom_defaults(self):
        lm = LM("test/model", timeout=60.0, temperature=0.5, max_tokens=4096)
        assert lm.timeout == 60.0
        assert lm.defaults["temperature"] == 0.5
        assert lm.defaults["max_tokens"] == 4096

    def test_client_result(self):
        result = ClientResult(text="hello", cost=0.001)
        assert result.text == "hello"
        assert result.cost == 0.001

    def test_acompletion_calls_litellm(self):
        lm = LM("test/model")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cost = None

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.005

            resp = asyncio.run(
                lm.acompletion(
                    [{"role": "user", "content": "hello"}],
                    temperature=0.7,
                )
            )

        assert resp.text == "test response"
        assert resp.cost == 0.005

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["model"] == "test/model"
        assert call_kwargs["temperature"] == 0.7

    def test_acompletion_uses_provider_cost_when_present(self):
        lm = LM("test/model")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cost = 0.123

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            resp = asyncio.run(lm.acompletion("hello"))

        assert resp.cost == 0.123

    def test_explicit_pricing_overrides_provider_calculation(self):
        lm = LM(
            "test/model",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            cache_read_input_token_cost=0.0005,
        )

        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            cost=99.0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4),
        )
        response = SimpleNamespace(
            usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        )

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=response)
            resp = asyncio.run(lm.acompletion("hello"))

        assert resp.cost == pytest.approx((6 * 0.001) + (4 * 0.0005) + (5 * 0.002))

    def test_acompletion_surfaces_litellm_errors(self):
        lm = LM("test/model")

        class ProviderError(Exception):
            pass

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=ProviderError("Connection refused"))

            with pytest.raises(ProviderError, match="Connection refused"):
                asyncio.run(lm.acompletion("hello"))

    def test_acompletion_invalid_schema_raises_value_error(self):
        lm = LM("test/model")

        bad_response = SimpleNamespace(choices=[])

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=bad_response)

            with pytest.raises(ValueError, match=r"\[test/model\] Invalid completion schema"):
                asyncio.run(lm.acompletion("hello"))

    def test_acompletion_omits_temperature_when_unset(self):
        lm = LM("test/model")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cost = None

        mock_choice = MagicMock()
        mock_choice.message.content = "test response"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.0

            asyncio.run(lm.acompletion("hello"))

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert "temperature" not in call_kwargs

    def test_acompletion_extras_passed_through(self):
        lm = LM("test/model")

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_usage.cost = None

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        with patch("levi.clients.lm.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            mock_litellm.completion_cost.return_value = 0.0

            asyncio.run(
                lm.acompletion(
                    "hello",
                    extra_body={"reasoning": {"enabled": False}},
                )
            )

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["extra_body"] == {"reasoning": {"enabled": False}}

    def test_close_is_noop(self):
        lm = LM("test/model")
        asyncio.run(lm.close())
