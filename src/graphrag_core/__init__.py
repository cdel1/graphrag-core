"""graphrag-core: Domain-agnostic Graph RAG framework."""

__version__ = "0.13.0"

from graphrag_core.interfaces import (
    Chunker,
    CommunityDetector,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    ExtractionPostProcessor,
    ExtractionPromptBuilder,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    SearchEngine,
)
from graphrag_core.ingestion import (
    DocxParser,
    MarkdownParser,
    PdfParser,
    TextParser,
    TokenChunker,
)
from graphrag_core.extraction import DefaultPromptBuilder, LLMExtractionEngine
from graphrag_core.graph import InMemoryGraphStore
from graphrag_core.search import InMemorySearchEngine
from graphrag_core.registry import InMemoryEntityRegistry
from graphrag_core.tools import Tool, ToolLibrary, register_core_tools
from graphrag_core.llm import BaseLLMClient
from graphrag_core.models import (
    ChunkExtractionResult,
    Community,
    DocumentChunk,
    ExtractionResult,
    GraphNode,
    ImportRun,
    KnownEntity,
    NodeTypeDefinition,
    OntologySchema,
    PropertyDefinition,
    RegistryMatch,
    RelationshipTypeDefinition,
    SearchResult,
    ToolParameter,
    ToolResult,
)

__all__ = [
    # Protocols
    "Chunker",
    "CommunityDetector",
    "DocumentParser",
    "EmbeddingModel",
    "EntityRegistry",
    "ExtractionEngine",
    "ExtractionPostProcessor",
    "ExtractionPromptBuilder",
    "GraphStore",
    "IngestionPipeline",
    "LLMClient",
    "SearchEngine",
    # BB1 implementations
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
    # BB9 LLM client
    "BaseLLMClient",
    # BB2 implementations
    "DefaultPromptBuilder",
    "LLMExtractionEngine",
    # BB3 implementations
    "InMemoryGraphStore",
    # BB4 implementations
    "InMemorySearchEngine",
    # BB6 implementations
    "InMemoryEntityRegistry",
    # BB7 implementations
    "Tool",
    "ToolLibrary",
    "register_core_tools",
    # Models
    "ChunkExtractionResult",
    "Community",
    "DocumentChunk",
    "ExtractionResult",
    "GraphNode",
    "ImportRun",
    "KnownEntity",
    "NodeTypeDefinition",
    "OntologySchema",
    "PropertyDefinition",
    "RegistryMatch",
    "RelationshipTypeDefinition",
    "SearchResult",
    "ToolParameter",
    "ToolResult",
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

try:
    from graphrag_core.llm import OpenAILLMClient
    __all__.append("OpenAILLMClient")
except ImportError:
    pass
