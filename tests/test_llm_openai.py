"""Unit tests for OpenAILLMClient."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel


class TestOpenAILLMClient:
    @pytest.mark.asyncio
    async def test_complete_sends_messages_and_returns_text(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from OpenAI"

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAILLMClient(model="gpt-4o", api_key="test-key")
            result = await client.complete(
                messages=[{"role": "user", "content": "Hi"}],
                system="You are helpful.",
                temperature=0.0,
                max_tokens=100,
            )

        assert result == "Hello from OpenAI"
        instance.chat.completions.create.assert_called_once()
        call_kwargs = instance.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "Hi"}
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_without_system_prompt(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "No system"

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAILLMClient(api_key="test-key")
            result = await client.complete(
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result == "No system"
        call_kwargs = instance.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_complete_json_uses_response_format(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        class ExtractModel(BaseModel):
            name: str
            count: int

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "test", "count": 42}'

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAILLMClient(model="gpt-4o", api_key="test-key")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "Extract"}],
                schema=ExtractModel,
                system="Extract entities",
            )

        assert isinstance(result, ExtractModel)
        assert result.name == "test"
        assert result.count == 42
        call_kwargs = instance.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["name"] == "ExtractModel"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is False

    @pytest.mark.asyncio
    async def test_complete_json_without_system_prompt(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        class ExtractModel(BaseModel):
            name: str

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "alice"}'

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create = AsyncMock(return_value=mock_response)

            client = OpenAILLMClient(api_key="test-key")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "Extract"}],
                schema=ExtractModel,
            )

        assert result.name == "alice"
        call_kwargs = instance.chat.completions.create.call_args[1]
        # No system message prepended
        assert call_kwargs["messages"] == [{"role": "user", "content": "Extract"}]

    @pytest.mark.asyncio
    async def test_implements_llm_client_protocol(self):
        from graphrag_core.interfaces import LLMClient
        from graphrag_core.llm.openai import OpenAILLMClient

        assert isinstance(OpenAILLMClient, type)
        with patch("graphrag_core.llm.openai.AsyncOpenAI"):
            client = OpenAILLMClient(api_key="test-key")
            assert isinstance(client, LLMClient)


class ExtractModel(BaseModel):
    name: str
    count: int


def _temperature_rejected():
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://api.anthropic.com/v1/chat/completions")
    return BadRequestError(
        "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
        "'message': '`temperature` is deprecated for this model.'}}",
        response=httpx.Response(400, request=request),
        body=None,
    )


class TestOpenAITemperatureRejected:
    """A model that refuses `temperature` gets the call retried without it (tessera#664)."""

    def _response(self, content: str):
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = content
        return response

    @pytest.mark.asyncio
    async def test_retries_without_temperature_and_remembers(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            create = AsyncMock(side_effect=[
                _temperature_rejected(), self._response("ok"), self._response("again"),
            ])
            MockClient.return_value.chat.completions.create = create

            client = OpenAILLMClient(model="claude-sonnet-5-5", api_key="k")
            first = await client.complete(messages=[{"role": "user", "content": "hi"}])
            second = await client.complete(messages=[{"role": "user", "content": "hi"}])

        assert (first, second) == ("ok", "again")
        sent = [c.kwargs for c in create.call_args_list]
        assert sent[0]["temperature"] == 0.0
        assert "temperature" not in sent[1]
        assert "temperature" not in sent[2]

    @pytest.mark.asyncio
    async def test_complete_json_retries_without_temperature(self):
        from graphrag_core.llm.openai import OpenAILLMClient

        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            create = AsyncMock(side_effect=[
                _temperature_rejected(), self._response('{"name": "x", "count": 1}'),
            ])
            MockClient.return_value.chat.completions.create = create

            client = OpenAILLMClient(api_key="k")
            result = await client.complete_json(
                messages=[{"role": "user", "content": "x"}], schema=ExtractModel,
            )

        assert result == ExtractModel(name="x", count=1)
        assert "temperature" not in create.call_args_list[1].kwargs

    @pytest.mark.asyncio
    async def test_other_bad_request_propagates_without_retry(self):
        import httpx
        from openai import BadRequestError

        from graphrag_core.llm.openai import OpenAILLMClient

        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        other = BadRequestError(
            "max_tokens: too large", response=httpx.Response(400, request=request), body=None,
        )
        with patch("graphrag_core.llm.openai.AsyncOpenAI") as MockClient:
            create = AsyncMock(side_effect=other)
            MockClient.return_value.chat.completions.create = create

            client = OpenAILLMClient(api_key="k")
            with pytest.raises(BadRequestError):
                await client.complete(messages=[{"role": "user", "content": "hi"}])

        assert create.call_count == 1
