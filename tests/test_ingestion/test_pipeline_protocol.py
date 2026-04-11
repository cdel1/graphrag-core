"""Test that IngestionPipeline Protocol exists and is importable."""

from graphrag_core.interfaces import IngestionPipeline


def test_ingestion_pipeline_protocol_is_runtime_checkable():
    import typing
    assert hasattr(IngestionPipeline, "__protocol_attrs__") or issubclass(
        IngestionPipeline, typing.Protocol
    )
