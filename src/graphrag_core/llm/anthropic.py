"""Anthropic Claude LLM client."""

from __future__ import annotations

from anthropic import AsyncAnthropic
from pydantic import BaseModel


class AnthropicLLMClient:
    """Thin wrapper around the Anthropic SDK implementing the LLMClient Protocol."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        return response.content[0].text

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> BaseModel:
        json_schema = schema.model_json_schema()
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": [
                {
                    "name": "extract",
                    "description": "Extract structured data",
                    "input_schema": json_schema,
                },
            ],
            "tool_choice": {"type": "tool", "name": "extract"},
        }
        if system is not None:
            kwargs["system"] = system
        response = await self._client.messages.create(**kwargs)
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return schema.model_validate(tool_block.input)
