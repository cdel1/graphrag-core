"""BB2: LLM-powered schema-guided entity extraction engine."""

from __future__ import annotations

import logging

import pydantic

from graphrag_core.interfaces import ExtractionPromptBuilder, LLMClient
from graphrag_core.models import (
    ChunkExtractionResult,
    Chunk,
    ExtractedNode,
    ExtractedRelationship,
    ExtractionResult,
    ImportRun,
    OntologySchema,
    ProvenanceLink,
    RejectedNode,
    RejectedRelationship,
    RejectionReason,
    SchemaAdmission,
)

logger = logging.getLogger(__name__)


def validate_extraction(
    nodes: list[ExtractedNode],
    rels: list[ExtractedRelationship],
    schema: OntologySchema,
    chunk_id: str | None = None,
) -> SchemaAdmission:
    """Split extracted nodes and relationships into admitted and rejected.

    The schema decides what enters the typed graph; nothing the extractor
    emitted is discarded. A rejected emission comes back verbatim — properties
    included, so the passage that justified it survives — typed by the reason
    it was not admitted.

    Args:
        nodes: Nodes the extractor emitted.
        rels: Relationships the extractor emitted.
        schema: The ontology schema that decides admission.
        chunk_id: Chunk the emissions came from, recorded on each rejection so
            an un-admitted node's chunk link survives with it. ``None`` when
            validating outside a chunk context.

    Returns:
        The admitted nodes and relationships plus every rejection, typed by
        :class:`RejectionReason`.
    """
    allowed_labels = {nt.label for nt in schema.node_types}
    rel_constraints = {
        rt.type: (set(rt.source_types), set(rt.target_types))
        for rt in schema.relationship_types
    }

    admitted_nodes = [n for n in nodes if n.label in allowed_labels]
    rejected_nodes = [
        RejectedNode(
            node=n,
            reason=RejectionReason.UNDECLARED_NODE_LABEL,
            chunk_id=chunk_id,
        )
        for n in nodes
        if n.label not in allowed_labels
    ]
    node_labels = {n.id: n.label for n in admitted_nodes}

    admitted_rels: list[ExtractedRelationship] = []
    rejected_rels: list[RejectedRelationship] = []
    for rel in rels:
        reason = _rejection_reason(rel, node_labels, rel_constraints)
        if reason is None:
            admitted_rels.append(rel)
        else:
            rejected_rels.append(
                RejectedRelationship(relationship=rel, reason=reason, chunk_id=chunk_id)
            )

    return SchemaAdmission(
        nodes=admitted_nodes,
        relationships=admitted_rels,
        rejected_nodes=rejected_nodes,
        rejected_relationships=rejected_rels,
    )


def _rejection_reason(
    rel: ExtractedRelationship,
    node_labels: dict[str, str],
    rel_constraints: dict[str, tuple[set[str], set[str]]],
) -> RejectionReason | None:
    """Return why the schema rejects this relationship, or None if it admits it.

    Checks run in order and the first failure wins: an undeclared type is
    reported as such even when its endpoints are also unresolvable.
    """
    if rel.type not in rel_constraints:
        return RejectionReason.UNDECLARED_RELATIONSHIP_TYPE
    if rel.source_id not in node_labels or rel.target_id not in node_labels:
        return RejectionReason.DANGLING_ENDPOINT
    source_types, target_types = rel_constraints[rel.type]
    if (
        node_labels[rel.source_id] not in source_types
        or node_labels[rel.target_id] not in target_types
    ):
        return RejectionReason.ENDPOINT_TYPE_VIOLATION
    return None


def _log_rejections(chunk_id: str, admission: SchemaAdmission) -> None:
    """Report what a chunk's emissions failed on — names, not just counts."""
    if not admission.rejected_nodes and not admission.rejected_relationships:
        return
    logger.info(
        "Chunk %s: schema did not admit %d node(s) %s and %d relationship(s) %s",
        chunk_id,
        len(admission.rejected_nodes),
        sorted({r.node.label for r in admission.rejected_nodes}),
        len(admission.rejected_relationships),
        sorted({r.relationship.type for r in admission.rejected_relationships}),
    )


class DefaultPromptBuilder:
    """Builds the default system prompt for LLM-based entity extraction."""

    def build_system_prompt(self, schema: OntologySchema) -> str:
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


class LLMExtractionEngine:
    """Extracts entities and relationships from text using an LLM, guided by an ontology schema."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: ExtractionPromptBuilder | None = None,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder or DefaultPromptBuilder()

    async def extract(
        self,
        chunks: list[Chunk],
        schema: OntologySchema,
        import_run: ImportRun,
    ) -> ExtractionResult:
        all_nodes: list[ExtractedNode] = []
        all_rels: list[ExtractedRelationship] = []
        all_provenance: list[ProvenanceLink] = []
        all_rejected_nodes: list[RejectedNode] = []
        all_rejected_rels: list[RejectedRelationship] = []
        malformed_chunk_extractions = 0

        system_prompt = self._prompt_builder.build_system_prompt(schema)

        for chunk in chunks:
            try:
                nodes, rels = await self._extract_chunk(chunk, system_prompt)
            except pydantic.ValidationError:
                logger.warning(
                    "Skipping chunk %s: LLM response failed schema validation",
                    chunk.id,
                )
                malformed_chunk_extractions += 1
                continue

            admission = self._validate(nodes, rels, schema, chunk.id)
            _log_rejections(chunk.id, admission)

            all_provenance.extend(
                ProvenanceLink(chunk_id=chunk.id, node_id=node.id, confidence=1.0)
                for node in admission.nodes
            )
            all_nodes.extend(admission.nodes)
            all_rels.extend(admission.relationships)
            all_rejected_nodes.extend(admission.rejected_nodes)
            all_rejected_rels.extend(admission.rejected_relationships)

        quality_signals = (
            {"malformed_chunk_extractions": malformed_chunk_extractions}
            if malformed_chunk_extractions
            else None
        )

        return ExtractionResult(
            nodes=all_nodes,
            relationships=all_rels,
            provenance=all_provenance,
            rejected_nodes=all_rejected_nodes,
            rejected_relationships=all_rejected_rels,
            quality_signals=quality_signals,
        )

    async def _extract_chunk(
        self, chunk: Chunk, system_prompt: str
    ) -> tuple[list[ExtractedNode], list[ExtractedRelationship]]:
        result = await self._llm.complete_json(
            messages=[{"role": "user", "content": chunk.text}],
            schema=ChunkExtractionResult,
            system=system_prompt,
            temperature=0.0,
        )
        return result.nodes, result.relationships

    def _validate(
        self,
        nodes: list[ExtractedNode],
        rels: list[ExtractedRelationship],
        schema: OntologySchema,
        chunk_id: str | None = None,
    ) -> SchemaAdmission:
        return validate_extraction(nodes, rels, schema, chunk_id)
