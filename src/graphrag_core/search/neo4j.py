"""BB4: Neo4j-backed hybrid search engine."""

from __future__ import annotations

from neo4j import AsyncGraphDatabase

from graphrag_core._cypher import MAX_DEPTH, validate_identifier
from graphrag_core.models import SearchResult
from graphrag_core.search.fusion import reciprocal_rank_fusion


class Neo4jHybridSearch:
    """Neo4j async implementation of the SearchEngine Protocol."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
        vector_index_name: str = "chunk_embeddings",
        fulltext_index_name: str = "node_fulltext",
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._database = database
        self._vector_index_name = vector_index_name
        self._fulltext_index_name = fulltext_index_name

    async def close(self) -> None:
        await self._driver.close()

    async def vector_search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        query = (
            "CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) "
            "YIELD node, score "
            "RETURN node, score, labels(node) AS labels "
            "ORDER BY score DESC"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                index_name=self._vector_index_name,
                top_k=top_k,
                embedding=query_embedding,
            )
            results = []
            async for record in result:
                props = dict(record["node"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                props.pop("embedding", None)

                if filters:
                    if filters.get("label") and label != filters["label"]:
                        continue

                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=record["score"],
                    source="vector",
                    properties=props,
                ))
            return results[:top_k]

    async def fulltext_search(
        self, query: str, node_types: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        cypher = (
            "CALL db.index.fulltext.queryNodes($index_name, $search_query) "
            "YIELD node, score "
            "RETURN node, score, labels(node) AS labels "
            "ORDER BY score DESC "
            "LIMIT $top_k"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                cypher,
                index_name=self._fulltext_index_name,
                search_query=query,
                top_k=top_k,
            )
            results = []
            async for record in result:
                props = dict(record["node"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                props.pop("embedding", None)

                if node_types and label not in node_types:
                    continue

                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=record["score"],
                    source="fulltext",
                    properties=props,
                ))
            return results

    async def graph_search(
        self, start_node_id: str, pattern: str, depth: int = 2
    ) -> list[SearchResult]:
        depth = min(max(depth, 1), MAX_DEPTH)
        validate_identifier(pattern, "relationship type")
        cypher = (
            f"MATCH (start {{id: $start_id}})-[:{pattern}*1..{depth}]-(m) "
            "WHERE m.id <> $start_id "
            "WITH DISTINCT m, labels(m) AS labels, "
            f"  length(shortestPath((start)-[:{pattern}*]-(m))) AS hops "
            "MATCH (start {id: $start_id}) "
            "RETURN m, labels, hops "
            "ORDER BY hops ASC"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, start_id=start_node_id)
            results = []
            async for record in result:
                props = dict(record["m"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                hops = record["hops"]
                score = 1.0 / hops if hops > 0 else 1.0
                results.append(SearchResult(
                    node_id=node_id,
                    label=label,
                    score=score,
                    source="graph",
                    properties=props,
                ))
            return results

    async def hybrid_search(
        self, query: str, embedding: list[float], top_k: int = 10
    ) -> list[SearchResult]:
        vector_results = await self.vector_search(query_embedding=embedding, top_k=top_k)
        fulltext_results = await self.fulltext_search(query=query, top_k=top_k)

        return reciprocal_rank_fusion([vector_results, fulltext_results], top_k=top_k)

    async def ensure_indexes(
        self,
        vector_dimensions: int = 1536,
        vector_node_label: str = "Chunk",
        vector_property: str = "embedding",
        fulltext_node_labels: list[str] | None = None,
        fulltext_properties: list[str] | None = None,
    ) -> None:
        validate_identifier(vector_node_label, "node label")
        validate_identifier(vector_property, "property name")

        fulltext_labels = fulltext_node_labels or [vector_node_label]
        fulltext_props = fulltext_properties or ["name"]

        for lbl in fulltext_labels:
            validate_identifier(lbl, "node label")
        for prop in fulltext_props:
            validate_identifier(prop, "property name")

        async with self._driver.session(database=self._database) as session:
            vector_cypher = (
                f"CREATE VECTOR INDEX {self._vector_index_name} IF NOT EXISTS "
                f"FOR (n:{vector_node_label}) ON (n.{vector_property}) "
                "OPTIONS {indexConfig: {`vector.dimensions`: $dimensions, "
                "`vector.similarity_function`: 'cosine'}}"
            )
            await session.run(vector_cypher, dimensions=vector_dimensions)

            labels_str = "|".join(fulltext_labels)
            props_str = ", ".join(f"n.{p}" for p in fulltext_props)
            fulltext_cypher = (
                f"CREATE FULLTEXT INDEX {self._fulltext_index_name} IF NOT EXISTS "
                f"FOR (n:{labels_str}) ON EACH [{props_str}]"
            )
            await session.run(fulltext_cypher)
