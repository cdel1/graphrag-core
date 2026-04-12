"""Tests for LLMExtractionEngine."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
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


def _schema() -> OntologySchema:
    """A simple test schema: Company and Person with WORKS_AT relationship."""
    return OntologySchema(
        node_types=[
            NodeTypeDefinition(
                label="Company",
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                ],
                required_properties=["name"],
            ),
            NodeTypeDefinition(
                label="Person",
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                    PropertyDefinition(name="role", type="string", required=False),
                ],
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
        timestamp=datetime(2026, 4, 12, 10, 0),
        source_type="text/plain",
        documents_processed=1,
        entities_extracted=0,
    )


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(id="chunk-0", text="Alice is a software engineer at Acme Corp.", position=0),
    ]


class TestExtractionEngineHappyPath:
    @pytest.mark.asyncio
    async def test_extracts_nodes_and_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice", "role": "software engineer"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme Corp"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 2
        assert result.nodes[0].label == "Person"
        assert result.nodes[0].properties["name"] == "Alice"
        assert result.nodes[1].label == "Company"
        assert result.nodes[1].properties["name"] == "Acme Corp"

        assert len(result.relationships) == 1
        assert result.relationships[0].type == "WORKS_AT"
        assert result.relationships[0].source_id == "person-alice"
        assert result.relationships[0].target_id == "company-acme"

        assert len(result.provenance) == 2
        chunk_ids = {p.chunk_id for p in result.provenance}
        assert chunk_ids == {"chunk-0"}
        node_ids = {p.node_id for p in result.provenance}
        assert node_ids == {"person-alice", "company-acme"}
