"""OpenAI LLM client."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The 400 a model returns when it no longer accepts `temperature` (e.g. claude-sonnet-5-5
# behind Anthropic's OpenAI-compatible endpoint).
TEMPERATURE_REJECTED = "`temperature` is deprecated"


def _error_message(exc: BadRequestError) -> str:
    body = exc.body
    if isinstance(body, dict) and "message" in body:
        return str(body["message"])
    return str(exc)


class OpenAILLMClient:
    """Thin wrapper around the OpenAI SDK implementing the LLMClient Protocol."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._send_temperature = True

    async def _create(self, kwargs: dict):
        """Send the request; if the model refuses `temperature`, drop it for this client."""
        if not self._send_temperature:
            kwargs.pop("temperature")
            return await self._client.chat.completions.create(**kwargs)
        try:
            return await self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if TEMPERATURE_REJECTED not in _error_message(exc):
                raise
        if self._send_temperature:
            self._send_temperature = False
            logger.warning(
                "%s rejects `temperature`; falling back for the rest of this client's life",
                self._model,
            )
        kwargs.pop("temperature")
        return await self._client.chat.completions.create(**kwargs)

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
        response = await self._create({
            "model": self._model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
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
        response = await self._create({
            "model": self._model,
            "messages": full_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": False,
                },
            },
        })
        return schema.model_validate_json(response.choices[0].message.content)
