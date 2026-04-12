"""BB2: LLM-powered schema-guided entity extraction engine."""

from __future__ import annotations

import json

from graphrag_core.interfaces import LLMClient
from graphrag_core.models import (
    DocumentChunk,
    ExtractedNode,
    ExtractedRelationship,
    ExtractionResult,
    ImportRun,
    OntologySchema,
    ProvenanceLink,
)


class LLMExtractionEngine:
    """Extracts entities and relationships from text using an LLM, guided by an ontology schema."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def extract(
        self,
        chunks: list[DocumentChunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult:
        all_nodes: list[ExtractedNode] = []
        all_rels: list[ExtractedRelationship] = []
        all_provenance: list[ProvenanceLink] = []

        system_prompt = self._build_system_prompt(schema)

        for chunk in chunks:
            nodes, rels = await self._extract_chunk(chunk, system_prompt)
            nodes, rels = self._validate(nodes, rels, schema)

            for node in nodes:
                all_provenance.append(
                    ProvenanceLink(chunk_id=chunk.id, node_id=node.id, confidence=1.0)
                )

            all_nodes.extend(nodes)
            all_rels.extend(rels)

        return ExtractionResult(
            nodes=all_nodes,
            relationships=all_rels,
            provenance=all_provenance,
        )

    async def _extract_chunk(
        self, chunk: DocumentChunk, system_prompt: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        response = await self._llm.complete(
            messages=[{"role": "user", "content": chunk.text}],
            system=system_prompt,
            temperature=0.0,
        )
        return self._parse_response(response)

    def _build_system_prompt(self, schema: OntologySchema) -> str:
        node_descriptions = []
        for nt in schema.node_types:
            props = ", ".join(
                f"{p.name} ({p.type}{', required' if p.required else ''})"
                for p in nt.properties
            )
            node_descriptions.append(f"- {nt.label}: properties=[{props}]")

        rel_descriptions = []
        for rt in schema.relationship_types:
            rel_descriptions.append(
                f"- {rt.type}: {rt.source_types} -> {rt.target_types}"
            )

        return (
            "You are an entity extraction engine. Extract entities and relationships "
            "from the provided text according to this schema.\n\n"
            "ALLOWED NODE TYPES:\n"
            + "\n".join(node_descriptions)
            + "\n\nALLOWED RELATIONSHIP TYPES:\n"
            + "\n".join(rel_descriptions)
            + "\n\nDo not extract entities or relationships not listed above.\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"nodes": [{"id": "<unique_id>", "label": "<NodeType>", "properties": {<key>: <value>}}], '
            '"relationships": [{"source_id": "<node_id>", "target_id": "<node_id>", "type": "<RelType>", "properties": {}}]}\n\n'
            "Rules:\n"
            "- Every node id must be unique and descriptive (e.g., 'person-alice', 'company-acme')\n"
            "- Only use node types and relationship types listed above\n"
            "- Include all required properties for each node type\n"
            "- Return empty arrays if no entities are found"
        )

    def _parse_response(
        self, response: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        data = json.loads(response)

        nodes = [
            ExtractedNode(
                id=n["id"],
                label=n["label"],
                properties=n.get("properties", {}),
            )
            for n in data.get("nodes", [])
        ]

        rels = [
            ExtractedRelationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                type=r["type"],
                properties=r.get("properties", {}),
            )
            for r in data.get("relationships", [])
        ]

        return nodes, rels

    def _validate(
        self,
        nodes: list[ExtractedNode],
        rels: list[ExtractedRelationship],
        schema: OntologySchema,
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        allowed_labels = {nt.label for nt in schema.node_types}
        allowed_rel_types = {rt.type for rt in schema.relationship_types}
        rel_constraints = {
            rt.type: (set(rt.source_types), set(rt.target_types))
            for rt in schema.relationship_types
        }

        valid_nodes = [n for n in nodes if n.label in allowed_labels]
        valid_node_ids = {n.id for n in valid_nodes}
        node_labels = {n.id: n.label for n in valid_nodes}

        valid_rels = []
        for rel in rels:
            if rel.type not in allowed_rel_types:
                continue
            if rel.source_id not in valid_node_ids or rel.target_id not in valid_node_ids:
                continue
            source_types, target_types = rel_constraints[rel.type]
            if node_labels[rel.source_id] not in source_types:
                continue
            if node_labels[rel.target_id] not in target_types:
                continue
            valid_rels.append(rel)

        return valid_nodes, valid_rels
