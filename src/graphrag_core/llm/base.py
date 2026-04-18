"""Base LLM client with fallback complete_json() via prompt + parse + retry."""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError


class BaseLLMClient:
    """Default complete_json() via prompt + parse + retry.

    Providers with native structured output (OpenAI, Anthropic) override
    complete_json() directly. This base class provides a working fallback
    for providers without native support (e.g., local model clients).
    """

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        raise NotImplementedError

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type[BaseModel],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> BaseModel:
        schema_text = json.dumps(schema.model_json_schema(), indent=2)
        augmented_system = (system or "") + (
            f"\n\nRespond with ONLY a JSON object matching this schema:\n{schema_text}\n"
            "No markdown fences. No explanation. Just the JSON object."
        )

        for attempt in range(2):
            response = await self.complete(
                messages, system=augmented_system, temperature=temperature, max_tokens=max_tokens,
            )
            text = self._strip_json(response)
            try:
                return schema.model_validate_json(text)
            except (json.JSONDecodeError, ValidationError) as exc:
                if attempt == 0:
                    augmented_system += (
                        f"\n\nYour previous response failed validation: {exc}\n"
                        "Try again. Return ONLY valid JSON."
                    )
                else:
                    raise
        raise RuntimeError("unreachable")

    @staticmethod
    def _strip_json(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            text = text[nl + 1 :] if nl != -1 else ""
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
