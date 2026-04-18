"""Unit tests for LLMExtractionEngine with structured output."""

import pytest

from graphrag_core.extraction.engine import LLMExtractionEngine
from graphrag_core.models import (
    ChunkExtractionResult,
    DocumentChunk,
    ExtractedNode,
    ExtractedRelationship,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RelationshipTypeDefinition,
)
from datetime import datetime


class MockStructuredLLM:
    """Mock LLM that returns structured output via complete_json()."""

    def __init__(self, nodes=None, relationships=None):
        self._nodes = nodes or []
        self._rels = relationships or []

    async def complete(self, messages, system=None, temperature=0.0, max_tokens=4096):
        return ""

    async def complete_json(self, messages, schema, system=None, temperature=0.0, max_tokens=4096):
        return ChunkExtractionResult(nodes=self._nodes, relationships=self._rels)


SIMPLE_SCHEMA = OntologySchema(
    node_types=[
        NodeTypeDefinition(
            label="Person",
            properties=[PropertyDefinition(name="name", type="string", required=True)],
        ),
        NodeTypeDefinition(
            label="Company",
            properties=[PropertyDefinition(name="name", type="string", required=True)],
        ),
    ],
    relationship_types=[
        RelationshipTypeDefinition(
            type="WORKS_AT", source_types=["Person"], target_types=["Company"],
        ),
    ],
)

SIMPLE_IMPORT_RUN = ImportRun(
    id="run-1", timestamp=datetime.now(), source_type="text/plain",
    documents_processed=1, entities_extracted=0,
)


class TestExtractionEngineStructuredOutput:
    @pytest.mark.asyncio
    async def test_extract_uses_complete_json(self):
        nodes = [
            ExtractedNode(id="person-alice", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="company-acme", label="Company", properties={"name": "Acme"}),
        ]
        rels = [
            ExtractedRelationship(
                source_id="person-alice", target_id="company-acme",
                type="WORKS_AT", properties={},
            ),
        ]
        llm = MockStructuredLLM(nodes=nodes, relationships=rels)
        engine = LLMExtractionEngine(llm)

        result = await engine.extract(
            chunks=[DocumentChunk(id="chunk-1", text="Alice works at Acme.")],
            schema=SIMPLE_SCHEMA,
            import_run=SIMPLE_IMPORT_RUN,
        )

        assert len(result.nodes) == 2
        assert len(result.relationships) == 1
        assert len(result.provenance) == 2
        assert result.nodes[0].id == "person-alice"
        assert result.relationships[0].type == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_extract_validates_against_schema(self):
        """Nodes with labels not in schema are filtered out."""
        nodes = [
            ExtractedNode(id="person-alice", label="Person", properties={"name": "Alice"}),
            ExtractedNode(id="alien-zorg", label="Alien", properties={"name": "Zorg"}),
        ]
        llm = MockStructuredLLM(nodes=nodes)
        engine = LLMExtractionEngine(llm)

        result = await engine.extract(
            chunks=[DocumentChunk(id="chunk-1", text="Alice met Zorg.")],
            schema=SIMPLE_SCHEMA,
            import_run=SIMPLE_IMPORT_RUN,
        )

        assert len(result.nodes) == 1
        assert result.nodes[0].label == "Person"

    @pytest.mark.asyncio
    async def test_extract_empty_result(self):
        llm = MockStructuredLLM()
        engine = LLMExtractionEngine(llm)

        result = await engine.extract(
            chunks=[DocumentChunk(id="chunk-1", text="Nothing here.")],
            schema=SIMPLE_SCHEMA,
            import_run=SIMPLE_IMPORT_RUN,
        )

        assert result.nodes == []
        assert result.relationships == []
        assert result.provenance == []


class TestExtractionEngineCleanup:
    def test_no_strip_fences_method(self):
        """_strip_fences has been removed — structured output handles format."""
        assert not hasattr(LLMExtractionEngine, "_strip_fences")

    def test_no_parse_response_method(self):
        """_parse_response has been removed — complete_json handles parsing."""
        assert not hasattr(LLMExtractionEngine, "_parse_response")
