"""BB8: Shared context for agent workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Runtime context passed between agents in a workflow."""

    graph_store: Any
    tool_library: Any
    search_engine: Any
    workflow_state: dict[str, Any] = field(default_factory=dict)
