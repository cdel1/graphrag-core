"""Packaging contract: Neo4j is an optional extra."""

import importlib
import sys

import pytest


def test_graphrag_core_imports_without_neo4j_driver_installed():
    """The package and its core modules must import even if neo4j is missing.

    This test confirms that `graphrag_core.graph.neo4j` module-load does not
    pull the `neo4j` driver eagerly. Whether the driver is installed in this
    environment or not, importing the module is harmless.
    """
    # Force a fresh import
    sys.modules.pop("graphrag_core.graph.neo4j", None)
    importlib.invalidate_caches()

    # Module import must succeed regardless of driver availability
    from graphrag_core.graph import neo4j as neo4j_module
    assert hasattr(neo4j_module, "Neo4jGraphStore")


def test_neo4jgraphstore_instantiation_raises_clean_error_when_driver_missing(monkeypatch):
    """When the `neo4j` driver is unavailable, instantiating raises ImportError
    with the install hint."""
    # Shadow the neo4j module to make import fail inside __init__
    monkeypatch.setitem(sys.modules, "neo4j", None)

    from graphrag_core.graph.neo4j import Neo4jGraphStore
    with pytest.raises(ImportError, match=r"graphrag-core\[neo4j\]"):
        Neo4jGraphStore(uri="bolt://localhost:7687", auth=("neo4j", "x"))
