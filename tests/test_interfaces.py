"""Test that concrete classes can satisfy each Protocol."""

from graphrag_core.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    ExtractionEngine,
    GraphStore,
    SearchEngine,
)
from graphrag_core.models import (
    AuditTrail,
    ChunkConfig,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    OntologySchema,
    ParsedDocument,
    SchemaViolation,
    SearchResult,
)


class TestDocumentParserProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyParser:
            async def parse(self, source: bytes, content_type: str) -> ParsedDocument:
                raise NotImplementedError

        parser: DocumentParser = MyParser()
        assert isinstance(parser, DocumentParser)


class TestChunkerProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyChunker:
            def chunk(
                self, doc: ParsedDocument, config: ChunkConfig
            ) -> list[DocumentChunk]:
                raise NotImplementedError

        chunker: Chunker = MyChunker()
        assert isinstance(chunker, Chunker)


class TestEmbeddingModelProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyEmbedding:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                raise NotImplementedError

        model: EmbeddingModel = MyEmbedding()
        assert isinstance(model, EmbeddingModel)


class TestExtractionEngineProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyEngine:
            async def extract(
                self,
                chunks: list[DocumentChunk],
                schema: OntologySchema,
                import_run: ImportRun,
            ) -> ExtractionResult:
                raise NotImplementedError

        engine: ExtractionEngine = MyEngine()
        assert isinstance(engine, ExtractionEngine)


class TestGraphStoreProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyStore:
            async def merge_node(self, node: GraphNode, import_run_id: str) -> str:
                raise NotImplementedError

            async def merge_relationship(
                self, rel: GraphRelationship, import_run_id: str
            ) -> str:
                raise NotImplementedError

            async def record_provenance(
                self, node_id: str, chunk_id: str, import_run_id: str
            ) -> None:
                raise NotImplementedError

            async def get_node(self, node_id: str) -> GraphNode | None:
                raise NotImplementedError

            async def get_audit_trail(self, node_id: str) -> AuditTrail:
                raise NotImplementedError

            async def get_related(
                self,
                node_id: str,
                rel_type: str | None = None,
                depth: int = 1,
            ) -> list[GraphNode]:
                raise NotImplementedError

            async def apply_schema(self, schema: OntologySchema) -> None:
                raise NotImplementedError

            async def validate_schema(self) -> list[SchemaViolation]:
                raise NotImplementedError

        store: GraphStore = MyStore()
        assert isinstance(store, GraphStore)


class TestSearchEngineProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MySearch:
            async def vector_search(
                self,
                query_embedding: list[float],
                top_k: int = 10,
                filters: dict | None = None,
            ) -> list[SearchResult]:
                raise NotImplementedError

            async def fulltext_search(
                self,
                query: str,
                node_types: list[str] | None = None,
                top_k: int = 10,
            ) -> list[SearchResult]:
                raise NotImplementedError

            async def graph_search(
                self, start_node_id: str, pattern: str, depth: int = 2
            ) -> list[SearchResult]:
                raise NotImplementedError

            async def hybrid_search(
                self,
                query: str,
                embedding: list[float],
                top_k: int = 10,
            ) -> list[SearchResult]:
                raise NotImplementedError

        search: SearchEngine = MySearch()
        assert isinstance(search, SearchEngine)
