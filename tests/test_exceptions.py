"""ADR-0006b Rule 4 seed: typed exception hierarchy, GraphStore family first."""

from graphrag_core.exceptions import GraphStoreError


def test_graph_store_error_is_exception() -> None:
    assert issubclass(GraphStoreError, Exception)


def test_graph_store_error_carries_message() -> None:
    err = GraphStoreError("flush failed: disk full")
    assert "disk full" in str(err)
