"""BB7: Tool registry and execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from graphrag_core.models import ToolParameter, ToolResult


@dataclass
class Tool:
    """A callable tool with metadata."""

    name: str
    description: str
    parameters: dict[str, ToolParameter]
    handler: Callable[..., Awaitable[ToolResult]]


class ToolLibrary:
    """Registry of tested, schema-validated graph query tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    async def execute(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        try:
            return await tool.handler(**kwargs)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
