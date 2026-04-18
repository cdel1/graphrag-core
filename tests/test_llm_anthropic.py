"""Unit tests for AnthropicLLMClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel


class ExtractModel(BaseModel):
    name: str
    count: int


class TestAnthropicLLMClient:
    @pytest.mark.asyncio
    async def test_complete_json_uses_tool_use(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"name": "test", "count": 42}

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)

            client = AnthropicLLMClient(api_key="test-key")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "Extract"}],
                schema=ExtractModel,
                system="Extract entities",
            )

        assert isinstance(result, ExtractModel)
        assert result.name == "test"
        assert result.count == 42
        call_kwargs = instance.messages.create.call_args[1]
        assert call_kwargs["tools"][0]["name"] == "extract"
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "extract"}
        assert call_kwargs["system"] == "Extract entities"

    @pytest.mark.asyncio
    async def test_complete_json_without_system_prompt(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"name": "alice", "count": 1}

        mock_response = MagicMock()
        mock_response.content = [tool_block]

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            instance = MockClient.return_value
            instance.messages.create = AsyncMock(return_value=mock_response)

            client = AnthropicLLMClient(api_key="test-key")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "Extract"}],
                schema=ExtractModel,
            )

        assert result.name == "alice"
        call_kwargs = instance.messages.create.call_args[1]
        assert "system" not in call_kwargs

    @pytest.mark.asyncio
    async def test_implements_llm_client_protocol(self):
        from graphrag_core.interfaces import LLMClient
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic"):
            client = AnthropicLLMClient(api_key="test-key")
            assert isinstance(client, LLMClient)
