"""graphrag-core: Domain-agnostic Graph RAG framework."""

__version__ = "0.1.0"

from graphrag_core.interfaces import (
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    LLMCurationLayer,
    SearchEngine,
)
from graphrag_core.ingestion import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    TokenChunker,
)
from graphrag_core.extraction import LLMExtractionEngine
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.search import InMemorySearchEngine
from graphrag_core.registry import InMemoryEntityRegistry
from graphrag_core.curation import DeterministicDetectionLayer, CurationPipeline
from graphrag_core.models import (
    CurationIssue,
    CurationReport,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    ImportRun,
    KnownEntity,
    NodeTypeDefinition,
    OntologySchema,
    RegistryMatch,
    SearchResult,
)

__all__ = [
    # Protocols
    "ApprovalGateway",
    "Chunker",
    "DetectionLayer",
    "DocumentParser",
    "EmbeddingModel",
    "EntityRegistry",
    "ExtractionEngine",
    "GraphStore",
    "IngestionPipeline",
    "LLMClient",
    "LLMCurationLayer",
    "SearchEngine",
    # BB1 implementations
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
    # BB2 implementations
    "LLMExtractionEngine",
    # BB3 implementations
    "InMemoryGraphStore",
    # BB4 implementations
    "InMemorySearchEngine",
    # BB5 implementations
    "CurationPipeline",
    "DeterministicDetectionLayer",
    # BB6 implementations
    "InMemoryEntityRegistry",
    # Models
    "CurationIssue",
    "CurationReport",
    "DocumentChunk",
    "ExtractionResult",
    "GraphNode",
    "ImportRun",
    "KnownEntity",
    "NodeTypeDefinition",
    "OntologySchema",
    "RegistryMatch",
    "SearchResult",
]

# Optional: Neo4j and Anthropic (require extras)
try:
    from graphrag_core.graph import Neo4jGraphStore
    __all__.append("Neo4jGraphStore")
except ImportError:
    pass

try:
    from graphrag_core.search import Neo4jHybridSearch
    __all__.append("Neo4jHybridSearch")
except ImportError:
    pass

try:
    from graphrag_core.llm import AnthropicLLMClient
    __all__.append("AnthropicLLMClient")
except ImportError:
    pass
