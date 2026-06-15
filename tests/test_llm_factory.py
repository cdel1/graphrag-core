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


class TestFromEnvExplicit:
    def test_explicit_anthropic(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient
        from graphrag_core.llm.factory import from_env

        with patch.dict(os.environ, {"GRAPHRAG_LLM_PROVIDER": "anthropic"}):
            with patch("graphrag_core.llm.anthropic.AsyncAnthropic"):
                client = from_env()
        assert isinstance(client, AnthropicLLMClient)

    def test_explicit_openai(self):
        from graphrag_core.llm.factory import from_env
        from graphrag_core.llm.openai import OpenAILLMClient

        with patch.dict(os.environ, {"GRAPHRAG_LLM_PROVIDER": "openai"}):
            with patch("graphrag_core.llm.openai.AsyncOpenAI"):
                client = from_env()
        assert isinstance(client, OpenAILLMClient)


class TestFromEnvCaseInsensitive:
    def test_uppercase_anthropic(self):
        from graphrag_core.llm.anthropic import AnthropicLLMClient
        from graphrag_core.llm.factory import from_env

        with patch.dict(os.environ, {"GRAPHRAG_LLM_PROVIDER": "ANTHROPIC"}):
            with patch("graphrag_core.llm.anthropic.AsyncAnthropic"):
                client = from_env()
        assert isinstance(client, AnthropicLLMClient)

    def test_mixed_case_openai(self):
        from graphrag_core.llm.factory import from_env
        from graphrag_core.llm.openai import OpenAILLMClient

        with patch.dict(os.environ, {"GRAPHRAG_LLM_PROVIDER": "OpenAI"}):
            with patch("graphrag_core.llm.openai.AsyncOpenAI"):
                client = from_env()
        assert isinstance(client, OpenAILLMClient)


class TestFromEnvUnknown:
    def test_unknown_raises_value_error_with_offending_value(self):
        from graphrag_core.llm.factory import from_env

        with patch.dict(os.environ, {"GRAPHRAG_LLM_PROVIDER": "gemini"}):
            with pytest.raises(ValueError) as excinfo:
                from_env()
        assert "'gemini'" in str(excinfo.value)
        assert "anthropic" in str(excinfo.value).lower()
        assert "openai" in str(excinfo.value).lower()


class TestPackageExport:
    def test_from_env_importable_from_package(self):
        """from graphrag_core.llm import from_env should work."""
        from graphrag_core.llm import from_env  # noqa: F401
