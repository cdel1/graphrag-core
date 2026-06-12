"""ADR-0006b Rule 4 seed: typed exception hierarchy, GraphStore family first."""

from graphrag_core.exceptions import GraphStoreError, MissingEndpointError


def test_graph_store_error_is_exception() -> None:
    assert issubclass(GraphStoreError, Exception)


def test_graph_store_error_carries_message() -> None:
    err = GraphStoreError("flush failed: disk full")
    assert "disk full" in str(err)


def test_missing_endpoint_error_is_graph_store_error() -> None:
    exc = MissingEndpointError(source_id="a", target_id="b")
    assert isinstance(exc, GraphStoreError)
    assert exc.source_id == "a"
    assert exc.target_id == "b"
    assert "a" in str(exc) and "b" in str(exc)
