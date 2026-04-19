"""BB2: LLM-powered schema-guided entity extraction engine."""

from __future__ import annotations

from graphrag_core.interfaces import LLMClient
from graphrag_core.models import (
    ChunkExtractionResult,
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
        result = await self._llm.complete_json(
            messages=[{"role": "user", "content": chunk.text}],
            schema=ChunkExtractionResult,
            system=system_prompt,
            temperature=0.0,
        )
        return result.nodes, result.relationships

    def _build_system_prompt(self, schema: OntologySchema) -> str:
        node_descriptions = []
        for nt in schema.node_types:
            props = ", ".join(
                f"{p.name} ({p.type}{', required' if p.required else ''})"
                for p in nt.properties
            )
            line = f"- {nt.label}: properties=[{props}]"
            if nt.description:
                line += f" \u2014 {nt.description}"
            node_descriptions.append(line)

        rel_descriptions = []
        for rt in schema.relationship_types:
            line = f"- {rt.type}: {rt.source_types} -> {rt.target_types}"
            if rt.description:
                line += f" \u2014 {rt.description}"
            rel_descriptions.append(line)

        return (
            "You are an entity extraction engine. Extract entities and relationships "
            "from the provided text according to this schema.\n\n"
            "ALLOWED NODE TYPES:\n"
            + "\n".join(node_descriptions)
            + "\n\nALLOWED RELATIONSHIP TYPES:\n"
            + "\n".join(rel_descriptions)
            + "\n\nDo not extract entities or relationships not listed above.\n\n"
            "Rules:\n"
            "- Every node id must be unique and descriptive (e.g., 'person-alice', 'company-acme')\n"
            "- Only use node types and relationship types listed above\n"
            "- Include all required properties for each node type\n"
            "- Return empty arrays if no entities are found"
        )

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
