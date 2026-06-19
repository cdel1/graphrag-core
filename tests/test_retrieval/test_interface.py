"""Protocol-conformance smoke tests for BB10 retrieval models."""

import pytest

from graphrag_core.retrieval import EmbeddingModel


class FakeEmbedder:
    """Minimal conformance fake — satisfies the EmbeddingModel Protocol."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 3 for _ in texts]


def test_embedding_model_protocol_conformance():
    """A class implementing the embed() signature satisfies EmbeddingModel."""
    fake: EmbeddingModel = FakeEmbedder()
    assert hasattr(fake, "embed")


async def test_fake_embedder_preserves_length():
    """FakeEmbedder honors the length-preservation contract."""
    fake = FakeEmbedder()
    out = await fake.embed(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(vec) == 3 for vec in out)
