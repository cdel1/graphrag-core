"""BB1: Ingestion pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphrag_core.interfaces import Chunker, DocumentParser, EmbeddingModel
from graphrag_core.models import ChunkConfig, Chunk, GraphNode, GraphRelationship

if TYPE_CHECKING:
    from graphrag_core.interfaces import GraphStore


class IngestionPipeline:
    """Wires together parser, chunker, and optional embedding model."""

    def __init__(
        self,
        parser: DocumentParser,
        chunker: Chunker,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embedding_model = embedding_model

    async def ingest(
        self,
        source: bytes,
        content_type: str,
        config: ChunkConfig | None = None,
        *,
        graph_store: "GraphStore | None" = None,
        import_run_id: str | None = None,
    ) -> list[Chunk]:
        parsed = await self._parser.parse(source, content_type)
        chunks = self._chunker.chunk(parsed, config or ChunkConfig())

        if self._embedding_model is not None:
            embeddings = await self._embedding_model.embed([c.text for c in chunks])
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb

        if graph_store is not None:
            if import_run_id is None:
                raise ValueError(
                    "import_run_id is required when graph_store is provided"
                )
            metadata = parsed.metadata
            doc_props = metadata.model_dump()
            # quarter -> period fallback for v0.6.0 transition (Task 6)
            if doc_props.get("period") is None and doc_props.get("quarter"):
                doc_props["period"] = doc_props["quarter"]
            doc_props.pop("quarter", None)  # do not persist deprecated field

            doc_id = f"doc:{metadata.sha256}"
            await graph_store.merge_node(
                GraphNode(id=doc_id, label="Document", properties=doc_props),
                import_run_id,
            )
            for chunk in chunks:
                # :Chunk node must exist before FROM_DOCUMENT edge — Neo4j MERGE
                # requires both endpoints to be matchable.
                chunk_props: dict[str, object] = {"text": chunk.text}
                if chunk.page is not None:
                    chunk_props["page"] = chunk.page
                if chunk.position is not None:
                    chunk_props["position"] = chunk.position
                if chunk.chunk_type:
                    chunk_props["chunk_type"] = chunk.chunk_type
                await graph_store.merge_node(
                    GraphNode(id=chunk.id, label="Chunk", properties=chunk_props),
                    import_run_id,
                )
                await graph_store.merge_relationship(
                    GraphRelationship(
                        source_id=chunk.id,
                        target_id=doc_id,
                        type="FROM_DOCUMENT",
                        properties={},
                    ),
                    import_run_id,
                )
            for prev, nxt in zip(chunks, chunks[1:]):
                await graph_store.merge_relationship(
                    GraphRelationship(
                        source_id=prev.id,
                        target_id=nxt.id,
                        type="NEXT_CHUNK",
                        properties={},
                    ),
                    import_run_id,
                )
            await graph_store.flush()

        return chunks
