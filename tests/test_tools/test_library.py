"""Tests for ToolLibrary."""

from __future__ import annotations

import pytest

from graphrag_core.models import ToolParameter, ToolResult


class TestToolLibraryRegister:
    def test_register_and_get(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True, data="ok")

        library = ToolLibrary()
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"input": ToolParameter(name="input", type="string", description="test input")},
            handler=handler,
        )
        library.register(tool)

        retrieved = library.get("test_tool")
        assert retrieved.name == "test_tool"

    def test_duplicate_register_raises(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True)

        library = ToolLibrary()
        tool = Tool(name="dup", description="", parameters={}, handler=handler)
        library.register(tool)

        with pytest.raises(ValueError):
            library.register(tool)

    def test_get_unknown_raises(self):
        from graphrag_core.tools.library import ToolLibrary

        library = ToolLibrary()
        with pytest.raises(KeyError):
            library.get("nonexistent")

    def test_list_tools(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True)

        library = ToolLibrary()
        library.register(Tool(name="a", description="", parameters={}, handler=handler))
        library.register(Tool(name="b", description="", parameters={}, handler=handler))

        tools = library.list_tools()
        names = [t.name for t in tools]
        assert "a" in names
        assert "b" in names
        assert len(tools) == 2


class TestToolLibraryExecute:
    @pytest.mark.asyncio
    async def test_execute_calls_handler(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def handler(**kwargs) -> ToolResult:
            return ToolResult(success=True, data=kwargs.get("value"))

        library = ToolLibrary()
        library.register(Tool(name="echo", description="", parameters={}, handler=handler))

        result = await library.execute("echo", value=42)
        assert result.success is True
        assert result.data == 42

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(self):
        from graphrag_core.tools.library import Tool, ToolLibrary

        async def failing_handler(**kwargs) -> ToolResult:
            raise RuntimeError("boom")

        library = ToolLibrary()
        library.register(Tool(name="fail", description="", parameters={}, handler=failing_handler))

        result = await library.execute("fail")
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        from graphrag_core.tools.library import ToolLibrary

        library = ToolLibrary()
        with pytest.raises(KeyError):
            await library.execute("nonexistent")
