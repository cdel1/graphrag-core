"""BB10: Retrieval models (encoders + rerankers).

Cross-cutting infrastructure seat. Consumed by BB1 ingestion (write-side embedding)
and BB4 search (read-side similarity). Per ADR-0039.
"""

from graphrag_core.interfaces import EmbeddingModel

__all__ = ["EmbeddingModel"]
