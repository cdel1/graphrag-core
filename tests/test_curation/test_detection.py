"""Tests for DeterministicDetectionLayer."""

from __future__ import annotations

import pytest

from graphrag_core.models import (
    GraphNode,
    GraphRelationship,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
    KnownEntity,
)


def _schema() -> OntologySchema:
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Person",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(type="WORKS_AT", source_types=["Person"], target_types=["Company"]),
        ],
    )


async def _populated_store():
    from graphrag_core.graph.memory import InMemoryGraphStore

    store = InMemoryGraphStore()
    await store.apply_schema(_schema())
    await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1")
    await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "ACME Corporation"}), "run-1")
    await store.merge_node(GraphNode(id="n3", label="Person", properties={"name": "Alice"}), "run-1")
    await store.merge_relationship(
        GraphRelationship(source_id="n3", target_id="n1", type="WORKS_AT"), "run-1"
    )
    return store


class TestDuplicateDetection:
    @pytest.mark.asyncio
    async def test_detects_fuzzy_duplicates_with_registry(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.registry.memory import InMemoryEntityRegistry

        registry = InMemoryEntityRegistry()
        await registry.register(KnownEntity(name="Acme Corp", entity_type="Company"))

        store = await _populated_store()
        detector = DeterministicDetectionLayer(entity_registry=registry)
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) >= 1
        assert any("n1" in i.affected_nodes and "n2" in i.affected_nodes for i in dup_issues)

    @pytest.mark.asyncio
    async def test_detects_duplicates_without_registry(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer

        store = await _populated_store()
        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) >= 1

    @pytest.mark.asyncio
    async def test_no_false_positives_on_distinct_names(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Company", properties={"name": "Acme Corp"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Globex Industries"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="PARTNER"), "run-1"
        )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        assert len(dup_issues) == 0


class TestOrphanDetection:
    @pytest.mark.asyncio
    async def test_detects_orphan_nodes(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer

        store = await _populated_store()
        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        orphan_issues = [i for i in issues if i.issue_type == "orphan"]
        assert len(orphan_issues) >= 1
        orphan_node_ids = [nid for i in orphan_issues for nid in i.affected_nodes]
        assert "n2" in orphan_node_ids

    @pytest.mark.asyncio
    async def test_no_orphans_when_all_connected(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.merge_node(GraphNode(id="n1", label="Person", properties={"name": "Alice"}), "run-1")
        await store.merge_node(GraphNode(id="n2", label="Company", properties={"name": "Acme"}), "run-1")
        await store.merge_relationship(
            GraphRelationship(source_id="n1", target_id="n2", type="WORKS_AT"), "run-1"
        )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        orphan_issues = [i for i in issues if i.issue_type == "orphan"]
        assert len(orphan_issues) == 0


class TestSchemaViolationDetection:
    @pytest.mark.asyncio
    async def test_detects_missing_required_property(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        await store.apply_schema(_schema())
        await store.merge_node(GraphNode(id="n1", label="Company", properties={}), "run-1")

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        schema_issues = [i for i in issues if i.issue_type == "schema_violation"]
        assert len(schema_issues) >= 1
        assert schema_issues[0].severity == "error"


class TestPairwiseCap:
    @pytest.mark.asyncio
    async def test_emits_warning_when_label_group_exceeds_cap(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer, _PAIRWISE_CAP
        from graphrag_core.graph.memory import InMemoryGraphStore

        store = InMemoryGraphStore()
        for i in range(_PAIRWISE_CAP + 1):
            await store.merge_node(
                GraphNode(id=f"n{i}", label="Company", properties={"name": f"Company {i}"}), "run-1"
            )

        detector = DeterministicDetectionLayer()
        issues = await detector.detect(store, _schema())

        skip_issues = [i for i in issues if i.issue_type == "skipped_detection"]
        assert len(skip_issues) >= 1
        assert skip_issues[0].severity == "warning"


class TestDetectionLayerProtocol:
    def test_satisfies_detection_layer_protocol(self):
        from graphrag_core.curation.detection import DeterministicDetectionLayer
        from graphrag_core.interfaces import DetectionLayer

        detector = DeterministicDetectionLayer()
        assert isinstance(detector, DetectionLayer)
