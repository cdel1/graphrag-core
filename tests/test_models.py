from datetime import date, datetime

from graphrag_core.models import (
    AuditTrail,
    ChunkConfig,
    ChunkExtractionResult,
    Community,
    DocumentChunk,
    DocumentMetadata,
    ExtractionResult,
    ExtractedNode,
    ExtractedRelationship,
    GraphNode,
    GraphRelationship,
    ImportRun,
    NodeTypeDefinition,
    OntologySchema,
    ParsedDocument,
    PropertyDefinition,
    ProvenanceLink,
    ProvenanceStep,
    RelationshipTypeDefinition,
    SchemaViolation,
    SearchResult,
    TextSection,
)


class TestIngestionModels:
    def test_document_metadata(self):
        meta = DocumentMetadata(
            title="Q4 Report",
            source="uploads/q4.pdf",
            doc_type="pdf",
            date=date(2025, 12, 1),
            quarter="Q4/2025",
            sha256="abc123",
        )
        assert meta.title == "Q4 Report"
        assert meta.doc_type == "pdf"

    def test_document_metadata_optional_fields(self):
        meta = DocumentMetadata(
            title="Note",
            source="note.txt",
            doc_type="txt",
            date=None,
            quarter=None,
            sha256="def456",
        )
        assert meta.date is None
        assert meta.quarter is None

    def test_text_section(self):
        section = TextSection(heading="Introduction", text="Some text", page=1)
        assert section.heading == "Introduction"

    def test_text_section_optional_heading(self):
        section = TextSection(heading=None, text="Body text", page=None)
        assert section.heading is None

    def test_parsed_document(self):
        meta = DocumentMetadata(
            title="Doc",
            source="doc.pdf",
            doc_type="pdf",
            date=None,
            quarter=None,
            sha256="aaa",
        )
        doc = ParsedDocument(
            sections=[TextSection(heading="H1", text="Content", page=1)],
            metadata=meta,
        )
        assert len(doc.sections) == 1

    def test_chunk_config_defaults(self):
        config = ChunkConfig()
        assert config.max_tokens == 512
        assert config.overlap == 50

    def test_document_chunk(self):
        chunk = DocumentChunk(id="c1", text="Hello world")
        assert chunk.embedding is None
        assert chunk.chunk_type == "text"

    def test_document_chunk_with_embedding(self):
        chunk = DocumentChunk(
            id="c2",
            text="Hello",
            embedding=[0.1, 0.2, 0.3],
            page=2,
            position=5,
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_import_run(self):
        run = ImportRun(
            id="run1",
            timestamp=datetime(2025, 12, 1, 10, 0),
            source_type="pdf",
            documents_processed=5,
            entities_extracted=42,
        )
        assert run.documents_processed == 5


class TestExtractionModels:
    def test_property_definition(self):
        prop = PropertyDefinition(name="status", type="string")
        assert prop.required is False

    def test_node_type_definition(self):
        node_type = NodeTypeDefinition(
            label="Company",
            properties=[PropertyDefinition(name="name", type="string", required=True)],
            required_properties=["name"],
        )
        assert node_type.label == "Company"
        assert len(node_type.properties) == 1

    def test_relationship_type_definition(self):
        rel_type = RelationshipTypeDefinition(
            type="OWNS",
            source_types=["Company"],
            target_types=["Asset"],
        )
        assert rel_type.type == "OWNS"

    def test_ontology_schema(self):
        schema = OntologySchema(
            node_types=[
                NodeTypeDefinition(label="Person", properties=[], required_properties=[]),
            ],
            relationship_types=[
                RelationshipTypeDefinition(
                    type="KNOWS", source_types=["Person"], target_types=["Person"]
                ),
            ],
        )
        assert len(schema.node_types) == 1
        assert len(schema.relationship_types) == 1

    def test_extracted_node(self):
        node = ExtractedNode(id="n1", label="Company", properties={"name": "Acme"})
        assert node.properties["name"] == "Acme"

    def test_extracted_relationship(self):
        rel = ExtractedRelationship(source_id="n1", target_id="n2", type="OWNS")
        assert rel.properties == {}

    def test_provenance_link(self):
        link = ProvenanceLink(chunk_id="c1", node_id="n1", confidence=0.95)
        assert link.confidence == 0.95

    def test_extraction_result(self):
        result = ExtractionResult(
            nodes=[ExtractedNode(id="n1", label="X", properties={})],
            relationships=[],
            provenance=[ProvenanceLink(chunk_id="c1", node_id="n1", confidence=0.9)],
        )
        assert len(result.nodes) == 1
        assert len(result.provenance) == 1


class TestGraphModels:
    def test_graph_node(self):
        node = GraphNode(id="g1", label="Company", properties={"name": "Acme"})
        assert node.label == "Company"

    def test_graph_relationship(self):
        rel = GraphRelationship(source_id="g1", target_id="g2", type="OWNS")
        assert rel.properties == {}

    def test_graph_relationship_with_properties(self):
        rel = GraphRelationship(
            source_id="g1",
            target_id="g2",
            type="OWNS",
            properties={"since": "2024"},
        )
        assert rel.properties["since"] == "2024"

    def test_provenance_step(self):
        step = ProvenanceStep(
            level="chunk", id="c1", metadata={"page": 3}
        )
        assert step.level == "chunk"

    def test_audit_trail(self):
        trail = AuditTrail(
            node_id="g1",
            provenance_chain=[
                ProvenanceStep(level="node", id="g1", metadata={}),
                ProvenanceStep(level="chunk", id="c1", metadata={"page": 1}),
                ProvenanceStep(level="document", id="d1", metadata={"title": "Report"}),
            ],
        )
        assert len(trail.provenance_chain) == 3

    def test_schema_violation(self):
        v = SchemaViolation(
            node_id="g1",
            violation_type="missing_property",
            message="Required property 'name' is missing",
        )
        assert v.violation_type == "missing_property"


class TestSearchModels:
    def test_search_result(self):
        result = SearchResult(
            node_id="g1",
            label="Acme Corp",
            score=0.92,
            source="vector",
        )
        assert result.score == 0.92
        assert result.properties == {}

    def test_search_result_with_properties(self):
        result = SearchResult(
            node_id="g1",
            label="Acme Corp",
            score=0.85,
            source="fulltext",
            properties={"matched_field": "name"},
        )
        assert result.source == "fulltext"
        assert result.properties["matched_field"] == "name"


class TestChunkExtractionResult:
    def test_chunk_extraction_result(self):
        result = ChunkExtractionResult(
            nodes=[
                ExtractedNode(id="doc-1", label="Document", properties={"title": "Report"}),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="doc-1", target_id="person-1", type="MENTIONS",
                ),
            ],
        )
        assert len(result.nodes) == 1
        assert len(result.relationships) == 1
        assert result.nodes[0].label == "Document"

    def test_chunk_extraction_result_empty(self):
        result = ChunkExtractionResult(nodes=[], relationships=[])
        assert result.nodes == []
        assert result.relationships == []


class TestCommunityModel:
    def test_community_basic(self):
        comm = Community(id="comm-1", node_ids=["n1", "n2", "n3"], size=3)
        assert comm.id == "comm-1"
        assert comm.node_ids == ["n1", "n2", "n3"]
        assert comm.size == 3
        assert comm.modularity_score is None
        assert comm.metadata == {}

    def test_community_with_score(self):
        comm = Community(
            id="comm-2",
            node_ids=["n4", "n5"],
            size=2,
            modularity_score=0.75,
            metadata={"level": 0},
        )
        assert comm.modularity_score == 0.75
        assert comm.metadata == {"level": 0}

    def test_community_serialization(self):
        comm = Community(
            id="comm-3",
            node_ids=["n6", "n7"],
            size=2,
            modularity_score=0.5,
            metadata={"level": 1},
        )
        data = comm.model_dump()
        restored = Community.model_validate(data)
        assert restored.id == comm.id
        assert restored.node_ids == comm.node_ids
        assert restored.modularity_score == comm.modularity_score
        assert restored.metadata == comm.metadata
