"""BB5: Curation pipeline orchestrator."""

from __future__ import annotations

from graphrag_core.interfaces import ApprovalGateway, DetectionLayer, GraphStore, LLMCurationLayer
from graphrag_core.models import CurationReport, OntologySchema


class CurationPipeline:
    """Orchestrates the curation flow: detect -> (curate) -> (approve)."""

    def __init__(
        self,
        detection: DetectionLayer,
        llm_curation: LLMCurationLayer | None = None,
        approval: ApprovalGateway | None = None,
    ) -> None:
        self._detection = detection
        self._llm_curation = llm_curation
        self._approval = approval

    async def run(
        self,
        graph_store: GraphStore,
        schema: OntologySchema,
    ) -> CurationReport:
        issues = await self._detection.detect(graph_store, schema)

        if self._llm_curation is not None:
            issues = await self._llm_curation.curate(issues)

        nodes = await graph_store.list_nodes()
        rel_count = await graph_store.count_relationships()

        return CurationReport(
            issues=issues,
            nodes_scanned=len(nodes),
            relationships_scanned=rel_count,
        )
