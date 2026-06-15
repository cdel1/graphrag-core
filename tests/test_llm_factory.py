"""Unit tests for graphrag_core.llm.factory.from_env()."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestFromEnvDefault:
    def test_unset_provider_returns_anthropic(self):
        """No GRAPHRAG_LLM_PROVIDER set → AnthropicLLMClient."""
        from graphrag_core.llm.anthropic import AnthropicLLMClient
        from graphrag_core.llm.factory import from_env

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GRAPHRAG_LLM_PROVIDER", None)
            with patch("graphrag_core.llm.anthropic.AsyncAnthropic"):
                client = from_env()
        assert isinstance(client, AnthropicLLMClient)
