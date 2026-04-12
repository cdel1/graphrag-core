"""BB8: Sequential agent orchestrator."""

from __future__ import annotations

from graphrag_core.agents.context import AgentContext
from graphrag_core.models import WorkflowResult


class SequentialOrchestrator:
    """Runs agents sequentially, stopping on first failure."""

    async def run_workflow(
        self,
        workflow_id: str,
        agents: list,
        context: AgentContext,
    ) -> WorkflowResult:
        agent_results = []

        for agent in agents:
            result = await agent.execute(context)
            agent_results.append(result)
            if not result.success:
                return WorkflowResult(
                    workflow_id=workflow_id,
                    success=False,
                    agent_results=agent_results,
                )

        return WorkflowResult(
            workflow_id=workflow_id,
            success=True,
            agent_results=agent_results,
        )
