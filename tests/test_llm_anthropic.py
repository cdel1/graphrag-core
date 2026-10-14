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


class TestAnthropicLLMClientDefaults:
    def test_default_model_is_sonnet_4_6(self):
        """Substrate default model: claude-sonnet-4-6 (workhorse tier)."""
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        from unittest.mock import patch
        with patch("graphrag_core.llm.anthropic.AsyncAnthropic"):
            client = AnthropicLLMClient(api_key="test")
        assert client._model == "claude-sonnet-4-6"


def _temperature_rejected():
    import httpx
    from anthropic import BadRequestError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    # body as returned by the live API for claude-sonnet-5-5 (2026-10-14)
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": "`temperature` is deprecated for this model."}}
    return BadRequestError(
        f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body,
    )


class TestAnthropicTemperatureRejected:
    """A model that refuses `temperature` gets the call retried without it (tessera#664)."""

    def _text_response(self, text: str):
        block = MagicMock()
        block.type = "text"
        block.text = text
        response = MagicMock()
        response.content = [block]
        return response

    @pytest.mark.asyncio
    async def test_retries_without_temperature_and_remembers(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            create = AsyncMock(side_effect=[
                _temperature_rejected(),
                self._text_response("ok"),
                self._text_response("again"),
            ])
            MockClient.return_value.messages.create = create

            client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="k")
            first = await client.complete(messages=[{"role": "user", "content": "hi"}])
            second = await client.complete(messages=[{"role": "user", "content": "hi"}])

        assert (first, second) == ("ok", "again")
        sent = [c.kwargs for c in create.call_args_list]
        assert len(sent) == 3
        assert sent[0]["temperature"] == 0.0
        assert "temperature" not in sent[1]
        assert "temperature" not in sent[2]

    @pytest.mark.asyncio
    async def test_complete_json_retries_without_temperature(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"name": "x", "count": 1}
        response = MagicMock()
        response.content = [tool_block]

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            create = AsyncMock(side_effect=[_temperature_rejected(), response])
            MockClient.return_value.messages.create = create

            client = AnthropicLLMClient(api_key="k")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "x"}], schema=ExtractModel,
            )

        assert result == ExtractModel(name="x", count=1)
        assert "temperature" not in create.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_other_bad_request_propagates_without_retry(self):
        import httpx
        from anthropic import BadRequestError

        from graphrag_core.llm.anthropic import AnthropicLLMClient

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        other = BadRequestError(
            "max_tokens: too large", response=httpx.Response(400, request=request), body=None,
        )
        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            create = AsyncMock(side_effect=other)
            MockClient.return_value.messages.create = create

            client = AnthropicLLMClient(api_key="k")
            with pytest.raises(BadRequestError):
                await client.complete(messages=[{"role": "user", "content": "hi"}])

        assert create.call_count == 1


def _tool_choice_rejected():
    import httpx
    from anthropic import BadRequestError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    # body as returned by the live API for claude-sonnet-5-5 (2026-10-14)
    body = {"type": "error", "error": {"type": "invalid_request_error",
            "message": 'tool_choice: type "tool" and "any" are not supported for this model.'}}
    return BadRequestError(
        f"Error code: 400 - {body}", response=httpx.Response(400, request=request), body=body,
    )


def _tool_response(payload: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.input = payload
    response = MagicMock()
    response.content = [block]
    return response


class TestAnthropicForcedToolChoiceRejected:
    """A model that refuses a forced tool_choice is asked to call the tool instead (tessera#664)."""

    @pytest.mark.asyncio
    async def test_retries_with_auto_and_instruction_and_remembers(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            create = AsyncMock(side_effect=[
                _tool_choice_rejected(),
                _tool_response({"name": "a", "count": 1}),
                _tool_response({"name": "b", "count": 2}),
            ])
            MockClient.return_value.messages.create = create

            client = AnthropicLLMClient(api_key="k")
            first = await client.complete_json(
                messages=[{"role": "user", "content": "x"}], schema=ExtractModel, system="Extract.",
            )
            second = await client.complete_json(
                messages=[{"role": "user", "content": "y"}], schema=ExtractModel,
            )

        assert (first.name, second.name) == ("a", "b")
        sent = [c.kwargs for c in create.call_args_list]
        assert sent[0]["tool_choice"] == {"type": "tool", "name": "extract"}
        for retried in sent[1:]:
            assert retried["tool_choice"] == {"type": "auto"}
            assert "calling the `extract` tool" in retried["system"]
        assert sent[1]["system"].startswith("Extract.")

    @pytest.mark.asyncio
    async def test_temperature_and_tool_choice_both_refused(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            create = AsyncMock(side_effect=[
                _temperature_rejected(),
                _tool_choice_rejected(),
                _tool_response({"name": "a", "count": 1}),
            ])
            MockClient.return_value.messages.create = create

            client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="k")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "x"}], schema=ExtractModel,
            )

        assert result.name == "a"
        last = create.call_args_list[-1].kwargs
        assert "temperature" not in last
        assert last["tool_choice"] == {"type": "auto"}

    @pytest.mark.asyncio
    async def test_answer_without_tool_call_raises_validation_error(self):
        from pydantic import ValidationError

        from graphrag_core.llm.anthropic import AnthropicLLMClient

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "I'd rather not."
        prose = MagicMock()
        prose.content = [text_block]

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(
                side_effect=[_tool_choice_rejected(), prose],
            )
            client = AnthropicLLMClient(api_key="k")
            with pytest.raises(ValidationError):
                await client.complete_json(
                    messages=[{"role": "user", "content": "x"}], schema=ExtractModel,
                )


class TestAnthropicConcurrentRefusals:
    @pytest.mark.asyncio
    async def test_calls_in_flight_when_the_flag_flips_also_retry(self):
        """Two calls send `temperature` before either 400 returns; both must recover."""
        import asyncio

        from graphrag_core.llm.anthropic import AnthropicLLMClient

        block = MagicMock()
        block.type = "text"
        block.text = "ok"
        response = MagicMock()
        response.content = [block]

        async def create(**kwargs):
            await asyncio.sleep(0)
            if "temperature" in kwargs:
                raise _temperature_rejected()
            return response

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(side_effect=create)
            client = AnthropicLLMClient(model="claude-sonnet-5-5", api_key="k")
            results = await asyncio.gather(
                client.complete(messages=[{"role": "user", "content": "a"}]),
                client.complete(messages=[{"role": "user", "content": "b"}]),
            )

        assert results == ["ok", "ok"]


class TestAnthropicCompleteSkipsNonTextBlocks:
    """A model may put a `thinking` block before the answer (claude-sonnet-5-5; tessera#664)."""

    @pytest.mark.asyncio
    async def test_returns_the_text_after_a_thinking_block(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        thinking = MagicMock(spec=["type", "thinking", "signature"])
        thinking.type = "thinking"
        text = MagicMock()
        text.type = "text"
        text.text = "the answer"
        response = MagicMock()
        response.content = [thinking, text]

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(return_value=response)
            client = AnthropicLLMClient(api_key="k")
            result = await client.complete(messages=[{"role": "user", "content": "q"}])

        assert result == "the answer"

    @pytest.mark.asyncio
    async def test_joins_several_text_blocks(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        blocks = []
        for part in ("first ", "second"):
            block = MagicMock()
            block.type = "text"
            block.text = part
            blocks.append(block)
        response = MagicMock()
        response.content = blocks

        with patch("graphrag_core.llm.anthropic.AsyncAnthropic") as MockClient:
            MockClient.return_value.messages.create = AsyncMock(return_value=response)
            client = AnthropicLLMClient(api_key="k")
            result = await client.complete(messages=[{"role": "user", "content": "q"}])

        assert result == "first second"
