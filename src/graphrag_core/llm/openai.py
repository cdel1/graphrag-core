"""OpenAI LLM client."""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic import BaseModel


class OpenAILLMClient:
    """Thin wrapper around the OpenAI SDK implementing the LLMClient Protocol."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        full_messages = list(messages)
        if system is not None:
            full_messages.insert(0, {"role": "system", "content": system})
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> BaseModel:
        full_messages = list(messages)
        if system is not None:
            full_messages.insert(0, {"role": "system", "content": system})
        json_schema = schema.model_json_schema()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True,
                },
            },
        )
        return schema.model_validate_json(response.choices[0].message.content)
