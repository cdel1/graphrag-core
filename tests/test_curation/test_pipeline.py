"""Tests for CurationPipeline."""

from __future__ import annotations

import pytest

from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
)


def _schema() -> OntologySchema:
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[],
    )


class TestCurationPipeline:
    @pytest.mark.asyncio
    async def test_runs_detection_and_returns_report(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme"}), "run-1")

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 1
        assert report.relationships_scanned == 0
        assert isinstance(report.issues, list)

    @pytest.mark.asyncio
    async def test_works_without_llm_or_approval(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        pipeline = CurationPipeline(
            detection=DeterministicDetectionLayer(),
            llm_curation=None,
            approval=None,
        )
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 0
        assert report.issues == []

    @pytest.mark.asyncio
    async def test_report_contains_detected_issues(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.apply_schema(_schema())
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert len(report.issues) >= 1
        assert any(i.issue_type == "schema_violation" for i in report.issues)

    @pytest.mark.asyncio
    async def test_report_counts_nodes_and_relationships(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.curation.pipeline import CurationPipeline
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "A"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "B"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="PARTNER"), "run-1"
        )

        pipeline = CurationPipeline(detection=DeterministicDetectionLayer())
        report = await pipeline.run(store, _schema())

        assert report.nodes_scanned == 2
        assert report.relationships_scanned == 1
