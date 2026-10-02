"""Shared fixtures for graphrag-core tests."""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest

NEO4J_TEST_URI_VAR = "NEO4J_TEST_URI"

_NOT_NOMINATED = (
    f"{NEO4J_TEST_URI_VAR} is not set. Every Neo4j fixture here wipes the database it "
    "connects to, so it only runs against an instance you nominated as disposable:\n"
    "  docker run --rm -d -p 7690:7687 -e NEO4J_AUTH=neo4j/development neo4j:5.15-community\n"
    f"  export {NEO4J_TEST_URI_VAR}=bolt://localhost:7690\n"
    "NEO4J_URI is deliberately not consulted — it points at real data."
)


class Neo4jTestTarget(NamedTuple):
    """Connection details for a Neo4j the invocation nominated as disposable."""

    uri: str
    auth: tuple[str, str]
    database: str


def nominated_neo4j() -> Neo4jTestTarget:
    """The nominated throwaway Neo4j, or skip the test.

    `NEO4J_TEST_URI` has no default on purpose (tessera#586): the safe outcome
    comes from the absence of configuration, not from a guess about whether the
    database on the other end holds anything worth keeping. Getting it wrong
    costs a visible skip; the previous default cost a developer their dev graph.
    """
    uri = os.environ.get(NEO4J_TEST_URI_VAR)
    if not uri:
        pytest.skip(_NOT_NOMINATED)
    return Neo4jTestTarget(
        uri=uri,
        auth=(
            os.environ.get("NEO4J_TEST_USER", "neo4j"),
            os.environ.get("NEO4J_TEST_PASSWORD", "development"),
        ),
        database=os.environ.get("NEO4J_TEST_DATABASE", "neo4j"),
    )


@pytest.fixture
async def neo4j_test_store():
    """A Neo4jGraphStore on the nominated throwaway, wiped before the test.

    Shared because `test_graph/test_audit_trail_v060.py` and
    `test_ingestion/test_document_node_creation.py` each held a byte-identical
    copy. The `store` fixture in `test_graph/test_neo4j.py` and `neo4j_store` in
    `test_integration/test_ingest_to_graph.py` are the same shape under local
    names; folding them in means renaming 25 call sites, so they stay put.
    """
    from graphrag_core.graph.neo4j import Neo4jGraphStore

    target = nominated_neo4j()
    store = Neo4jGraphStore(uri=target.uri, auth=target.auth, database=target.database)
    async with store._driver.session(database=target.database) as session:
        await session.run("MATCH (n) DETACH DELETE n")
    yield store
    await store.close()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration tests unless --run-integration is passed or RUN_INTEGRATION=1."""
    run_integration = config.getoption("--run-integration", default=False) or os.environ.get(
        "RUN_INTEGRATION", ""
    ) == "1"
    if run_integration:
        return
    skip = pytest.mark.skip(reason="integration tests require --run-integration or RUN_INTEGRATION=1")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that require external services",
    )
