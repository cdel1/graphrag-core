"""Tests for LLMExtractionEngine."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from graphrag_core.models import (
    ChunkConfig,
    ChunkExtractionResult,
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
        data = json.loads(response)
        return ChunkExtractionResult(**data)


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


class TestExtractionEngineValidation:
    @pytest.mark.asyncio
    async def test_drops_off_schema_nodes(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "New York"}},
            ],
            "relationships": [],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 1
        assert result.nodes[0].label == "Person"

    @pytest.mark.asyncio
    async def test_drops_off_schema_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "FOUNDED", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_drops_relationship_with_wrong_source_type(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "company-a", "label": "Company", "properties": {"name": "A"}},
                {"id": "company-b", "label": "Company", "properties": {"name": "B"}},
            ],
            "relationships": [
                {"source_id": "company-a", "target_id": "company-b", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_drops_dangling_relationships(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        llm_response = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "loc-nyc", "label": "Location", "properties": {"name": "NYC"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "loc-nyc", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[llm_response]))
        result = await engine.extract(chunks=_chunks(), schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 1
        assert len(result.relationships) == 0

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty_result(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        result = await engine.extract(chunks=[], schema=_schema(), import_run=_import_run())

        assert result.nodes == []
        assert result.relationships == []
        assert result.provenance == []


class TestExtractionEngineMultiChunk:
    @pytest.mark.asyncio
    async def test_extracts_across_multiple_chunks(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        chunks = [
            DocumentChunk(id="chunk-0", text="Alice works at Acme.", position=0),
            DocumentChunk(id="chunk-1", text="Bob works at Globex.", position=1),
        ]

        response_0 = json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })
        response_1 = json.dumps({
            "nodes": [
                {"id": "person-bob", "label": "Person", "properties": {"name": "Bob"}},
                {"id": "company-globex", "label": "Company", "properties": {"name": "Globex"}},
            ],
            "relationships": [
                {"source_id": "person-bob", "target_id": "company-globex", "type": "WORKS_AT", "properties": {}},
            ],
        })

        engine = LLMExtractionEngine(
            llm_client=FakeLLMClient(responses=[response_0, response_1])
        )
        result = await engine.extract(chunks=chunks, schema=_schema(), import_run=_import_run())

        assert len(result.nodes) == 4
        assert len(result.relationships) == 2
        assert len(result.provenance) == 4

        chunk_0_provenance = [p for p in result.provenance if p.chunk_id == "chunk-0"]
        chunk_1_provenance = [p for p in result.provenance if p.chunk_id == "chunk-1"]
        assert len(chunk_0_provenance) == 2
        assert len(chunk_1_provenance) == 2
        assert {p.node_id for p in chunk_0_provenance} == {"person-alice", "company-acme"}
        assert {p.node_id for p in chunk_1_provenance} == {"person-bob", "company-globex"}


class TestExtractionEngineProtocol:
    def test_satisfies_extraction_engine_protocol(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine
        from graphrag_core.interfaces import ExtractionEngine

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        assert isinstance(engine, ExtractionEngine)


class TestNodeTypeDescription:
    def test_node_type_definition_accepts_description(self):
        ntd = NodeTypeDefinition(
            label="Topic",
            properties=[PropertyDefinition(name="name", type="string", required=True)],
            description="A recurring subject or theme identified across documents",
        )
        assert ntd.description == "A recurring subject or theme identified across documents"

    def test_node_type_definition_description_defaults_to_none(self):
        ntd = NodeTypeDefinition(
            label="Topic",
            properties=[PropertyDefinition(name="name", type="string", required=True)],
        )
        assert ntd.description is None


class TestRelationshipTypeDescription:
    def test_relationship_type_definition_accepts_description(self):
        rtd = RelationshipTypeDefinition(
            type="HAS_FINDING",
            source_types=["Topic"],
            target_types=["Finding"],
            description="Links a topic to an observation drawn from evidence",
        )
        assert rtd.description == "Links a topic to an observation drawn from evidence"

    def test_relationship_type_definition_description_defaults_to_none(self):
        rtd = RelationshipTypeDefinition(
            type="HAS_FINDING",
            source_types=["Topic"],
            target_types=["Finding"],
        )
        assert rtd.description is None


class TestDescriptionsInPrompt:
    def test_node_description_appears_in_prompt(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Topic",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                    description="A recurring subject or theme",
                ),
            ],
            relationship_types=[],
        )

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        prompt = engine._build_system_prompt(schema)

        assert "A recurring subject or theme" in prompt
        assert "Topic" in prompt

    def test_node_without_description_has_no_dash_suffix(self):
        from graphrag_core.extraction.engine import LLMExtractionEngine

        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(
                    label="Person",
                    properties=[PropertyDefinition(name="name", type="string", required=True)],
                ),
            ],
            relationship_types=[],
        )

        engine = LLMExtractionEngine(llm_client=FakeLLMClient(responses=[]))
        prompt = engine._build_system_prompt(schema)

        assert "- Person: properties=" in prompt
        lines = [l for l in prompt.split("\n") if "Person" in l]
        assert len(lines) == 1
        assert "\u2014" not in lines[0]  # em dash should NOT appear
