"""graphrag-core: Domain-agnostic Graph RAG framework."""

__version__ = "0.2.0"

from graphrag_core.interfaces import (
    Agent,
    ApprovalGateway,
    Chunker,
    DetectionLayer,
    DocumentParser,
    EmbeddingModel,
    EntityRegistry,
    ExtractionEngine,
    ExtractionPromptBuilder,
    GraphStore,
    IngestionPipeline,
    LLMClient,
    LLMCurationLayer,
    Orchestrator,
    ReportRenderer,
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
from graphrag_core.curation import DeterministicDetectionLayer, CurationPipeline
from graphrag_core.tools import Tool, ToolLibrary, register_core_tools
from graphrag_core.agents import AgentContext, SequentialOrchestrator
from graphrag_core.llm import BaseLLMClient
from graphrag_core.models import (
    AgentResult,
    ChunkExtractionResult,
    CurationIssue,
    CurationReport,
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
    RenderConfig,
    ReportData,
    SearchResult,
    ToolParameter,
    ToolResult,
    WorkflowResult,
)

__all__ = [
    # Protocols
    "Agent",
    "ApprovalGateway",
    "Chunker",
    "DetectionLayer",
    "DocumentParser",
    "EmbeddingModel",
    "EntityRegistry",
    "ExtractionEngine",
    "ExtractionPromptBuilder",
    "GraphStore",
    "IngestionPipeline",
    "LLMClient",
    "LLMCurationLayer",
    "Orchestrator",
    "ReportRenderer",
    "SearchEngine",
    # BB1 implementations
    "DocxParser",
    "MarkdownParser",
    "PdfParser",
    "TextParser",
    "TokenChunker",
    # LLM base
    "BaseLLMClient",
    # BB2 implementations
    "DefaultPromptBuilder",
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
    # BB7 implementations
    "Tool",
    "ToolLibrary",
    "register_core_tools",
    # BB8 implementations
    "AgentContext",
    "SequentialOrchestrator",
    # Models
    "AgentResult",
    "ChunkExtractionResult",
    "CurationIssue",
    "CurationReport",
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
    "RenderConfig",
    "ReportData",
    "SearchResult",
    "ToolParameter",
    "ToolResult",
    "WorkflowResult",
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
