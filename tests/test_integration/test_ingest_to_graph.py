"""End-to-end integration test: BB1 (Ingestion) -> BB2 (Extraction) -> BB3 (Graph)."""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest

NEO4J_TEST_DB = os.environ.get("NEO4J_TEST_DATABASE", "neo4j")

from graphrag_core.models import (
    ChunkConfig,
    DocumentChunk,
    DocumentMetadata,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    ParsedDocument,
    PropertyDefinition,
    RelationshipTypeDefinition,
    TextSection,
)

pytestmark = pytest.mark.integration


class FakeLLMClient:
    """Returns extraction JSON based on chunk content."""

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        return json.dumps({
            "nodes": [
                {"id": "person-alice", "label": "Person", "properties": {"name": "Alice", "role": "engineer"}},
                {"id": "company-acme", "label": "Company", "properties": {"name": "Acme Corp"}},
            ],
            "relationships": [
                {"source_id": "person-alice", "target_id": "company-acme", "type": "WORKS_AT", "properties": {}},
            ],
        })


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
                properties=[
                    PropertyDefinition(name="name", type="string", required=True),
                    PropertyDefinition(name="role", type="string", required=False),
                ],
                required_properties=["name"],
            ),
        ],
        relationship_types=[
            RelationshipTypeDefinition(type="WORKS_AT", source_types=["Person"], target_types=["Company"]),
        ],
    )


@pytest.fixture
async def neo4j_store():
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    store = Neo4jGraphStore(database=NEO4J_TEST_DB)
    async with store._driver.session(database=NEO4J_TEST_DB) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


class TestIngestToGraph:
    @pytest.mark.asyncio
    async def test_full_pipeline_document_to_graph(self, neo4j_store):
        from graphrag_core.ingestion import TextParser, TokenChunker, IngestionPipeline
        from graphrag_core.extraction.engine import LLMExtractionEngine

        # BB1: Ingest a plain text document
        source = b"Alice is a software engineer at Acme Corp. She has been working there for five years."
        pipeline = IngestionPipeline(
            parser=TextParser(),
            chunker=TokenChunker(),
        )
        chunks = await pipeline.ingest(source, "text/plain", ChunkConfig(max_tokens=50, overlap=0))
        assert len(chunks) >= 1

        # BB2: Extract entities
        engine = LLMExtractionEngine(llm_client=FakeLLMClient())
        import_run = ImportRun(
            id="run-e2e",
            timestamp=datetime(2026, 4, 12, 12, 0),
            source_type="text/plain",
            documents_processed=1,
            entities_extracted=0,
        )
        extraction = await engine.extract(chunks=chunks, schema=_schema(), import_run=import_run)
        assert len(extraction.nodes) >= 2
        assert len(extraction.relationships) >= 1

        # BB3: Store in Neo4j
        for node_data in extraction.nodes:
            from graphrag_core.models import GraphNode
            graph_node = GraphNode(id=node_data.id, label=node_data.label, properties=node_data.properties)
            await neo4j_store.merge_node(graph_node, import_run_id=import_run.id)

        for rel_data in extraction.relationships:
            from graphrag_core.models import GraphRelationship
            graph_rel = GraphRelationship(
                source_id=rel_data.source_id,
                target_id=rel_data.target_id,
                type=rel_data.type,
                properties=rel_data.properties,
            )
            await neo4j_store.merge_relationship(graph_rel, import_run_id=import_run.id)

        # Record provenance
        for prov in extraction.provenance:
            await neo4j_store.record_provenance(
                node_id=prov.node_id, chunk_id=prov.chunk_id, import_run_id=import_run.id
            )

        # Verify: nodes exist in graph
        alice = await neo4j_store.get_node("person-alice")
        assert alice is not None
        assert alice.label == "Person"
        assert alice.properties["name"] == "Alice"

        acme = await neo4j_store.get_node("company-acme")
        assert acme is not None
        assert acme.label == "Company"

        # Verify: relationship exists
        related = await neo4j_store.get_related("person-alice", rel_type="WORKS_AT")
        assert len(related) == 1
        assert related[0].id == "company-acme"

        # Verify: provenance chain exists
        trail = await neo4j_store.get_audit_trail("person-alice")
        assert trail.node_id == "person-alice"
        levels = [s.level for s in trail.provenance_chain]
        assert "node" in levels
        assert "chunk" in levels
