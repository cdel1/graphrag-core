"""Anthropic Claude LLM client."""

from __future__ import annotations

import logging

from anthropic import AsyncAnthropic, BadRequestError
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# The 400s a model returns when it no longer accepts a request parameter (e.g. claude-sonnet-5-5).
TEMPERATURE_REJECTED = "`temperature` is deprecated"
TOOL_CHOICE_REJECTED = 'tool_choice: type "tool" and "any" are not supported'

MUST_CALL_TOOL = "Respond only by calling the `extract` tool, even when there is nothing to extract."


def _error_message(exc: BadRequestError) -> str:
    body = exc.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return str(body["error"].get("message", ""))
    return str(exc)


class AnthropicLLMClient:
    """Thin wrapper around the Anthropic SDK implementing the LLMClient Protocol."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key)
        self._send_temperature = True
        self._force_tool_choice = True

    async def _create(self, kwargs: dict):
        """Send the request, dropping each parameter the model refuses for this client's life.

        A refusal is only retried while the refused parameter is still in the request,
        so a request is sent at most three times. Concurrent calls that were already
        in flight when the flag flipped retry the same way.
        """
        while True:
            if not self._send_temperature:
                kwargs.pop("temperature", None)
            if not self._force_tool_choice and kwargs.get("tool_choice", {}).get("type") == "tool":
                kwargs["tool_choice"] = {"type": "auto"}
                kwargs["system"] = f"{kwargs.get('system', '')}\n\n{MUST_CALL_TOOL}".lstrip()
            try:
                return await self._client.messages.create(**kwargs)
            except BadRequestError as exc:
                message = _error_message(exc)
                if TEMPERATURE_REJECTED in message and "temperature" in kwargs:
                    if self._send_temperature:
                        self._send_temperature = False
                        self._warn_refused("`temperature`")
                elif (
                    TOOL_CHOICE_REJECTED in message
                    and kwargs.get("tool_choice", {}).get("type") == "tool"
                ):
                    if self._force_tool_choice:
                        self._force_tool_choice = False
                        self._warn_refused("a forced `tool_choice`")
                else:
                    raise

    def _warn_refused(self, refused: str) -> None:
        logger.warning(
            "%s rejects %s; falling back for the rest of this client's life", self._model, refused,
        )

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
        response = await self._create(kwargs)
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
        response = await self._create(kwargs)
        tool_input = next((b.input for b in response.content if b.type == "tool_use"), None)
        return schema.model_validate(tool_input)
