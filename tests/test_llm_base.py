"""Unit tests for BaseLLMClient fallback complete_json()."""

import pytest
from pydantic import BaseModel

from graphrag_core.llm.base import BaseLLMClient


class SimpleModel(BaseModel):
    name: str
    value: int


class FakeLLMClient(BaseLLMClient):
    """Test subclass that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)

    async def complete(self, messages, system=None, temperature=0.0, max_tokens=4096):
        return next(self._responses)


class TestBaseLLMClientCompleteJson:
    @pytest.mark.asyncio
    async def test_valid_json_returns_model(self):
        client = FakeLLMClient(['{"name": "alice", "value": 42}'])
        result = await client.complete_json(
            messages=[{"role": "user", "content": "extract"}],
            schema=SimpleModel,
        )
        assert isinstance(result, SimpleModel)
        assert result.name == "alice"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        client = FakeLLMClient(['```json\n{"name": "bob", "value": 7}\n```'])
        result = await client.complete_json(
            messages=[{"role": "user", "content": "extract"}],
            schema=SimpleModel,
        )
        assert result.name == "bob"
        assert result.value == 7

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self):
        client = FakeLLMClient([
            '{"name": "bad"}',                    # missing required field "value"
            '{"name": "fixed", "value": 99}',     # valid on retry
        ])
        result = await client.complete_json(
            messages=[{"role": "user", "content": "extract"}],
            schema=SimpleModel,
        )
        assert result.name == "fixed"
        assert result.value == 99

    @pytest.mark.asyncio
    async def test_raises_after_exhausted_retries(self):
        client = FakeLLMClient([
            '{"name": "bad"}',    # invalid
            '{"name": "worse"}',  # still invalid
        ])
        with pytest.raises(Exception):
            await client.complete_json(
                messages=[{"role": "user", "content": "extract"}],
                schema=SimpleModel,
            )

    @pytest.mark.asyncio
    async def test_complete_raises_not_implemented(self):
        client = BaseLLMClient()
        with pytest.raises(NotImplementedError):
            await client.complete(messages=[])
