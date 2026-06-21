"""Test that concrete classes can satisfy each Protocol."""

from pydantic import BaseModel

from graphrag_core.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    ExtractionPostProcessor,
    GraphStore,
    LLMClient,
    SearchEngine,
)
from graphrag_core.models import (
    ProvenanceTrail,
    ChunkConfig,
    Chunk,
    ExtractionResult,
    GraphNode,
    GraphRelationship,
    ImportRun,
    KnownEntity,
    OntologySchema,
    ParsedDocument,
    RegistryMatch,
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
            ) -> list[Chunk]:
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
                chunks: list[Chunk],
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

            async def get_provenance(self, node_id: str) -> ProvenanceTrail:
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

            async def list_nodes(self) -> list[GraphNode]:
                raise NotImplementedError

            async def count_relationships(self) -> int:
                raise NotImplementedError

            async def list_relationships(self) -> list[GraphRelationship]:
                return []

            async def flush(self) -> None:
                return None

            async def clear(self) -> None:
                return None

        store: GraphStore = MyStore()
        assert isinstance(store, GraphStore)


class TestLLMClientProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyLLMClient:
            async def complete(
                self,
                messages: list[dict[str, str]],
                system: str | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> str:
                raise NotImplementedError

            async def complete_json(
                self,
                messages: list[dict[str, str]],
                schema: type[BaseModel],
                system: str | None = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
            ) -> BaseModel:
                raise NotImplementedError

        client: LLMClient = MyLLMClient()
        assert isinstance(client, LLMClient)


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


class TestEntityRegistryProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyRegistry:
            async def register(self, entity: KnownEntity) -> str:
                raise NotImplementedError

            async def lookup(
                self, name: str, entity_type: str, match_strategy: str = "fuzzy"
            ) -> list[RegistryMatch]:
                raise NotImplementedError

            async def bulk_register(self, entities: list[KnownEntity]) -> int:
                raise NotImplementedError

        registry: EntityRegistry = MyRegistry()
        assert isinstance(registry, EntityRegistry)


class TestExtractionPostProcessorProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MyPostProcessor:
            async def process(
                self,
                result: ExtractionResult,
                existing_entities: list[GraphNode] | None = None,
            ) -> ExtractionResult:
                raise NotImplementedError

        processor: ExtractionPostProcessor = MyPostProcessor()
        assert isinstance(processor, ExtractionPostProcessor)


class TestCommunityDetectorProtocol:
    def test_concrete_class_satisfies_protocol(self):
        from graphrag_core.interfaces import CommunityDetector, GraphStore
        from graphrag_core.models import Community

        class MyDetector:
            async def detect(self, graph_store: GraphStore) -> list[Community]:
                return []

        detector = MyDetector()
        assert isinstance(detector, CommunityDetector)
