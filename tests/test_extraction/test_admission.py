"""Admission is not destruction — the engine hands back what it did not admit.

The schema gates what enters the typed graph; it never deletes what the
extractor returned. Every test here runs against the real
``LLMExtractionEngine``, asserting on what the engine returned rather than on
how it decided.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import pytest

from graphrag_core.extraction.engine import LLMExtractionEngine
from graphrag_core.models import (
    Chunk,
    ChunkExtractionResult,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RejectionReason,
    RelationshipTypeDefinition,
)


class FakeLLMClient:
    """Returns canned JSON responses for testing."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        response = self._responses[self._call_index]
        self._call_index += 1
        return response

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: type,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ChunkExtractionResult:
        response = self._responses[self._call_index]
        self._call_index += 1
        return ChunkExtractionResult(**json.loads(response))


def _schema() -> OntologySchema:
    """Person and Company, with a one-directional WORKS_AT between them."""
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Person",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Company",
                properties=[PropertyDefinition(name="name", type="string", required=True)],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(
                type="WORKS_AT",
                source_types=["Person"],
                target_types=["Company"],
            ),
        ],
    )


def _import_run() -> ImportRun:
    return ImportRun(
        id="run-1",
        timestamp=datetime(2026, 9, 20, 10, 0),
        source_type="text/plain",
        documents_processed=1,
        entities_extracted=0,
    )


def _chunks() -> list[Chunk]:
    return [Chunk(id="chunk-0", text="Alice is an engineer at Acme Corp.", position=0)]


async def _extract(responses: list[str], chunks: list[Chunk] | None = None):
    engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=responses))
    return await engine.extract(
        chunks=chunks or _chunks(), schema=_schema(), import_run=_import_run()
    )


class TestUnadmittedNodes:
    @pytest.mark.asyncio
    async def test_undeclared_label_is_handed_back_verbatim(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "New York"}},
            ],
            "relationships": [],
        })

        result = await _extract([response])

        assert [n.id for n in result.nodes] == ["person-alice"]
        assert len(result.rejected_nodes) == 1
        rejected = result.rejected_nodes[0]
        assert rejected.reason == RejectionReason.UNDECLARED_NODE_LABEL
        assert rejected.node.id == "loc-nyc"
        assert rejected.node.label == "Location"
        assert rejected.node.properties == {"name": "New York"}

    @pytest.mark.asyncio
    async def test_rejected_node_keeps_its_chunk_link(self):
        """The provenance link the node would have had is recoverable."""
        response = json.dumps({
            "nodes": [
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "New York"}},
            ],
            "relationships": [],
        })

        result = await _extract([response])

        assert result.provenance == []
        assert result.rejected_nodes[0].chunk_id == "chunk-0"


class TestUnadmittedRelationships:
    @pytest.mark.asyncio
    async def test_undeclared_type_is_handed_back(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "FOUNDED",
                 "properties": {"excerpt": "Alice founded Acme in 2011."}},
            ],
        })

        result = await _extract([response])

        assert result.relationships == []
        assert len(result.rejected_relationships) == 1
        rejected = result.rejected_relationships[0]
        assert rejected.reason == RejectionReason.UNDECLARED_RELATIONSHIP_TYPE
        assert rejected.relationship.type == "FOUNDED"
        assert rejected.relationship.properties["excerpt"] == "Alice founded Acme in 2011."
        assert rejected.chunk_id == "chunk-0"

    @pytest.mark.asyncio
    async def test_dangling_endpoint_is_handed_back(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-gone", "type": "WORKS_AT",
                 "properties": {}},
            ],
        })

        result = await _extract([response])

        assert result.relationships == []
        assert result.rejected_relationships[0].reason == RejectionReason.DANGLING_ENDPOINT
        assert result.rejected_relationships[0].relationship.target_id == "company-gone"

    @pytest.mark.asyncio
    async def test_endpoint_of_a_rejected_node_is_a_dangling_endpoint(self):
        """An edge onto an un-admitted node is dangling, not a position violation."""
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "loc-nyc", "type": "WORKS_AT",
                 "properties": {}},
            ],
        })

        result = await _extract([response])

        assert result.rejected_nodes[0].reason == RejectionReason.UNDECLARED_NODE_LABEL
        assert result.rejected_relationships[0].reason == RejectionReason.DANGLING_ENDPOINT

    @pytest.mark.asyncio
    async def test_reversed_endpoints_preserve_the_warrant(self):
        """A correct passage is not deleted because the arrow pointed the wrong way."""
        warrant = "Alice has worked at Acme Corp since 2011."
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "company-acme", "target_id": "person-alice", "type": "WORKS_AT",
                 "properties": {"excerpt": warrant}},
            ],
        })

        result = await _extract([response])

        assert result.relationships == []
        rejected = result.rejected_relationships[0]
        assert rejected.reason == RejectionReason.ENDPOINT_TYPE_VIOLATION
        assert rejected.relationship.source_id == "company-acme"
        assert rejected.relationship.target_id == "person-alice"
        assert rejected.relationship.properties["excerpt"] == warrant

        excerpts = [
            r.relationship.properties.get("excerpt") for r in result.rejected_relationships
        ]
        assert warrant in excerpts


class TestNothingIsDestroyed:
    @pytest.mark.asyncio
    async def test_every_emission_is_recoverable_from_the_result(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT",
                 "properties": {}},
                {"source_id": "company-acme", "target_id": "person-alice", "type": "WORKS_AT",
                 "properties": {}},
                {"source_id": "person-alice", "target_id": "company-acme", "type": "FOUNDED",
                 "properties": {}},
                {"source_id": "person-alice", "target_id": "loc-nyc", "type": "WORKS_AT",
                 "properties": {}},
            ],
        })

        result = await _extract([response])

        emitted_node_ids = {"person-alice", "company-acme", "loc-nyc"}
        returned_node_ids = {n.id for n in result.nodes} | {
            r.node.id for r in result.rejected_nodes
        }
        assert returned_node_ids == emitted_node_ids

        emitted_edges = {
            ("person-alice", "company-acme", "WORKS_AT"),
            ("company-acme", "person-alice", "WORKS_AT"),
            ("person-alice", "company-acme", "FOUNDED"),
            ("person-alice", "loc-nyc", "WORKS_AT"),
        }
        returned_edges = {
            (r.source_id, r.target_id, r.type) for r in result.relationships
        } | {
            (r.relationship.source_id, r.relationship.target_id, r.relationship.type)
            for r in result.rejected_relationships
        }
        assert returned_edges == emitted_edges

    @pytest.mark.asyncio
    async def test_admitted_output_is_unaffected_by_rejections(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT",
                 "properties": {}},
                {"source_id": "person-alice", "target_id": "company-acme", "type": "FOUNDED",
                 "properties": {}},
            ],
        })

        result = await _extract([response])

        assert [n.id for n in result.nodes] == ["person-alice", "company-acme"]
        assert [r.type for r in result.relationships] == ["WORKS_AT"]
        assert {p.node_id for p in result.provenance} == {"person-alice", "company-acme"}

    @pytest.mark.asyncio
    async def test_clean_extraction_rejects_nothing(self):
        response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT",
                 "properties": {}},
            ],
        })

        result = await _extract([response])

        assert result.rejected_nodes == []
        assert result.rejected_relationships == []

    @pytest.mark.asyncio
    async def test_schema_admitting_nothing_still_returns_every_emission(self):
        """Zero declared node types: nothing is admitted, nothing is lost."""
        empty_schema = OntologySchema(node_types=[], relationship_types=[])
        response = json.dumps({
            "nodes": [{"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}}],
            "relationships": [
                {"source_id": "person-alice", "target_id": "person-alice", "type": "WORKS_AT",
                 "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[response]))
        result = await engine.extract(
            chunks=_chunks(), schema=empty_schema, import_run=_import_run()
        )

        assert result.nodes == []
        assert result.relationships == []
        assert result.rejected_nodes[0].reason == RejectionReason.UNDECLARED_NODE_LABEL
        assert result.rejected_relationships[0].reason == (
            RejectionReason.UNDECLARED_RELATIONSHIP_TYPE
        )

    @pytest.mark.asyncio
    async def test_rejections_are_logged_with_names(self, caplog):
        response = json.dumps({
            "nodes": [{"id": "loc-nyc", "label": "Location", "properties": {}}],
            "relationships": [
                {"source_id": "loc-nyc", "target_id": "loc-nyc", "type": "FOUNDED",
                 "properties": {}},
            ],
        })

        with caplog.at_level(logging.INFO, logger="graphrag_core.extraction.engine"):
            await _extract([response])

        assert "Location" in caplog.text
        assert "FOUNDED" in caplog.text
        assert "chunk-0" in caplog.text

    @pytest.mark.asyncio
    async def test_a_clean_chunk_logs_nothing(self, caplog):
        response = json.dumps({
            "nodes": [{"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}}],
            "relationships": [],
        })

        with caplog.at_level(logging.INFO, logger="graphrag_core.extraction.engine"):
            await _extract([response])

        assert caplog.text == ""

    @pytest.mark.asyncio
    async def test_rejections_carry_the_chunk_they_came_from(self):
        chunks = [
            Chunk(id="chunk-0", text="Alice is in New York.", position=0),
            Chunk(id="chunk-1", text="Bob is in Berlin.", position=1),
        ]
        response_0 = json.dumps({
            "nodes": [{"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}}],
            "relationships": [],
        })
        response_1 = json.dumps({
            "nodes": [{"id": "loc-berlin", "label": "Location", "properties": {"name": "Berlin"}}],
            "relationships": [],
        })

        result = await _extract([response_0, response_1], chunks=chunks)

        by_node = {r.node.id: r.chunk_id for r in result.rejected_nodes}
        assert by_node == {"loc-nyc": "chunk-0", "loc-berlin": "chunk-1"}
