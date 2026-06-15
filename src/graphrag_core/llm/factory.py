"""Env-var-selected LLMClient construction.

The factory is the substrate's default path for selecting a
provider-portable LLM client. Callers needing more (per-call override,
mixed providers) instantiate the concrete client directly.
"""

from __future__ import annotations

import os

from graphrag_core.llm.base import BaseLLMClient


def from_env() -> BaseLLMClient:
    """Return a configured LLMClient based on GRAPHRAG_LLM_PROVIDER.

    Reads:
        GRAPHRAG_LLM_PROVIDER: ``"anthropic"`` (default) or ``"openai"``.
            Case-insensitive. Unknown values raise ``ValueError``.

    API-key validation is not the factory's responsibility; missing or
    invalid keys propagate as the provider SDK's exception at the first
    request.

    Returns:
        AnthropicLLMClient if provider is ``"anthropic"`` (default),
        OpenAILLMClient if ``"openai"``.

    Raises:
        ValueError: GRAPHRAG_LLM_PROVIDER is set to an unrecognised value.
        ImportError: the chosen provider's optional dependency is not
            installed. Install via the matching extra:
            ``pip install 'graphrag-core[anthropic]'`` or
            ``pip install 'graphrag-core[openai]'``.
    """
    provider = os.environ.get("GRAPHRAG_LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        from graphrag_core.llm.anthropic import AnthropicLLMClient

        return AnthropicLLMClient()
    if provider == "openai":
        from graphrag_core.llm.openai import OpenAILLMClient

        return OpenAILLMClient()
    raise ValueError(
        f"Unknown GRAPHRAG_LLM_PROVIDER value: {provider!r}. "
        f"Supported: 'anthropic', 'openai'."
    )
