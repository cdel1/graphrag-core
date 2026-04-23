"""BB3: Neo4j-backed GraphStore implementation."""

from __future__ import annotations

from datetime import datetime, timezone

from neo4j import AsyncGraphDatabase

from graphrag_core._cypher import MAX_DEPTH, validate_identifier
from graphrag_core.models import (
    AuditTrail,
    GraphNode,
    GraphRelationship,
    OntologySchema,
    ProvenanceStep,
    SchemaViolation,
)


class Neo4jGraphStore:
    """Neo4j async implementation of the GraphStore Protocol."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple[str, str] = ("neo4j", "development"),
        database: str = "neo4j",
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=auth)
        self._database = database

    async def close(self) -> None:
        await self._driver.close()

    async def merge_node(self, node: GraphNode, import_run_id: str) -> str:
        validate_identifier(node.label, "node label")
        query = (
            f"MERGE (n:{node.label} {{id: $id}}) "
            "SET n += $props, n._import_run_id = $run_id, n._updated_at = $now "
            "RETURN n.id AS id"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                id=node.id,
                props=node.properties,
                run_id=import_run_id,
                now=datetime.now(timezone.utc).isoformat(),
            )
            record = await result.single()
            return record["id"]

    async def merge_relationship(self, rel: GraphRelationship, import_run_id: str) -> str:
        validate_identifier(rel.type, "relationship type")
        query = (
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:{rel.type}]->(b) "
            "SET r += $props, r._import_run_id = $run_id, r._updated_at = $now "
            "RETURN $source_id + '-' + $rel_type + '-' + $target_id AS id"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                query,
                source_id=rel.source_id,
                target_id=rel.target_id,
                rel_type=rel.type,
                props=rel.properties,
                run_id=import_run_id,
                now=datetime.now(timezone.utc).isoformat(),
            )
            record = await result.single()
            return record["id"]

    async def record_provenance(self, node_id: str, chunk_id: str, import_run_id: str) -> None:
        query = (
            "MERGE (c:Chunk {id: $chunk_id}) "
            "WITH c "
            "MATCH (n {id: $node_id}) "
            "MERGE (c)-[r:SOURCED]->(n) "
            "SET r._import_run_id = $run_id"
        )
        async with self._driver.session(database=self._database) as session:
            await session.run(
                query,
                chunk_id=chunk_id,
                node_id=node_id,
                run_id=import_run_id,
            )

    async def get_node(self, node_id: str) -> GraphNode | None:
        query = "MATCH (n {id: $id}) RETURN n, labels(n) AS labels"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            record = await result.single()
            if record is None:
                return None
            props = dict(record["n"])
            label = [l for l in record["labels"] if l != "Chunk"][0] if record["labels"] else "Unknown"
            node_id_val = props.pop("id", node_id)
            props.pop("_import_run_id", None)
            props.pop("_updated_at", None)
            return GraphNode(id=node_id_val, label=label, properties=props)

    async def get_audit_trail(self, node_id: str) -> AuditTrail:
        query = (
            "MATCH (n {id: $id}) "
            "OPTIONAL MATCH (c:Chunk)-[:SOURCED]->(n) "
            "RETURN n, labels(n) AS labels, collect(c.id) AS chunk_ids"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            record = await result.single()

            chain: list[ProvenanceStep] = []
            if record and record["n"]:
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                chain.append(ProvenanceStep(level="node", id=node_id, metadata={"label": label}))
                for chunk_id in record["chunk_ids"]:
                    if chunk_id:
                        chain.append(ProvenanceStep(level="chunk", id=chunk_id, metadata={}))

            return AuditTrail(node_id=node_id, provenance_chain=chain)

    async def get_related(
        self, node_id: str, rel_type: str | None = None, depth: int = 1
    ) -> list[GraphNode]:
        depth = min(max(depth, 1), MAX_DEPTH)
        if rel_type:
            validate_identifier(rel_type, "relationship type")
            query = (
                f"MATCH (n {{id: $id}})-[:{rel_type}*1..{depth}]-(m) "
                "WHERE m.id <> $id "
                "RETURN DISTINCT m, labels(m) AS labels"
            )
        else:
            query = (
                f"MATCH (n {{id: $id}})-[*1..{depth}]-(m) "
                "WHERE m.id <> $id "
                "RETURN DISTINCT m, labels(m) AS labels"
            )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query, id=node_id)
            nodes = []
            async for record in result:
                props = dict(record["m"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                nid = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                nodes.append(GraphNode(id=nid, label=label, properties=props))
            return nodes

    async def apply_schema(self, schema: OntologySchema) -> None:
        async with self._driver.session(database=self._database) as session:
            for nt in schema.node_types:
                validate_identifier(nt.label, "node label")
                constraint_query = (
                    f"CREATE CONSTRAINT IF NOT EXISTS "
                    f"FOR (n:{nt.label}) REQUIRE n.id IS UNIQUE"
                )
                await session.run(constraint_query)
                for prop in nt.required_properties:
                    validate_identifier(prop, "property name")
                    index_query = (
                        f"CREATE INDEX IF NOT EXISTS "
                        f"FOR (n:{nt.label}) ON (n.{prop})"
                    )
                    await session.run(index_query)

    async def list_nodes(self) -> list[GraphNode]:
        query = "MATCH (n) WHERE NOT n:Chunk RETURN n, labels(n) AS labels"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            nodes = []
            async for record in result:
                props = dict(record["n"])
                labels = [l for l in record["labels"] if l != "Chunk"]
                label = labels[0] if labels else "Unknown"
                node_id = props.pop("id", "")
                props.pop("_import_run_id", None)
                props.pop("_updated_at", None)
                nodes.append(GraphNode(id=node_id, label=label, properties=props))
            return nodes

    async def count_relationships(self) -> int:
        query = "MATCH ()-[r]->() WHERE type(r) <> 'SOURCED' RETURN count(r) AS cnt"
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            record = await result.single()
            return record["cnt"] if record else 0

    async def list_relationships(self) -> list[GraphRelationship]:
        query = (
            "MATCH (a)-[r]->(b) "
            "RETURN a.id AS source_id, type(r) AS rel_type, b.id AS target_id, "
            "properties(r) AS props"
        )
        async with self._driver.session(database=self._database) as session:
            result = await session.run(query)
            records = [record async for record in result]
        return [
            GraphRelationship(
                source_id=rec["source_id"],
                target_id=rec["target_id"],
                type=rec["rel_type"],
                properties={
                    k: v for k, v in (rec["props"] or {}).items()
                    if not k.startswith("_")
                },
            )
            for rec in records
        ]

    async def validate_schema(self) -> list[SchemaViolation]:
        violations: list[SchemaViolation] = []
        return violations
