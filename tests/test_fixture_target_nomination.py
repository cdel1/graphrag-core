"""No test fixture may wipe a database the invocation did not nominate.

The hazard these tests close: every Neo4j fixture in this suite starts by
running `MATCH (n) DETACH DELETE n`, and until tessera#586 they all resolved
their target from a default that happened to be the developer's dev container.
Installing the `neo4j` extra for any reason and running the suite once was
enough to empty it.

The guard is the *absence* of configuration, not a guess about what the data is
worth: `NEO4J_TEST_URI` has no default, and an unset one skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import NEO4J_TEST_URI_VAR, nominated_neo4j

TESTS_DIR = Path(__file__).parent


def test_unset_nomination_skips_even_when_neo4j_uri_names_a_live_instance(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.delenv(NEO4J_TEST_URI_VAR, raising=False)

    with pytest.raises(pytest.skip.Exception) as excinfo:
        nominated_neo4j()

    assert NEO4J_TEST_URI_VAR in str(excinfo.value)


def test_nominated_target_is_the_one_returned(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv(NEO4J_TEST_URI_VAR, "bolt://localhost:7690")

    assert nominated_neo4j().uri == "bolt://localhost:7690"


def test_no_test_module_reads_the_cli_s_neo4j_uri():
    """`NEO4J_URI` points at real data. Nothing under `tests/` may read it.

    A grep rather than a behavioural assertion on purpose: this is what catches
    the *next* fixture someone adds, which is the failure mode #586 was.
    """
    offenders = [
        path.relative_to(TESTS_DIR).as_posix()
        for path in TESTS_DIR.rglob("*.py")
        if path.name != Path(__file__).name and '"NEO4J_URI"' in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


# Modules that name a Neo4j client but never open a connection with it, so they
# cannot reach a database and need no nomination. Anything else must go through
# the helper.
NON_CONNECTING = {
    # Asserts the ImportError when the driver is absent — construction raises.
    "test_packaging.py",
    # `Neo4jGraphStore()` for an `isinstance(store, GraphStore)` check. Routing
    # it through the helper would make a pure structural assertion need a
    # running container.
    "test_graph/test_neo4j.py",
}


def test_every_module_that_opens_a_neo4j_goes_through_the_nomination_helper():
    """The guard that would have caught #586 in this repo.

    Anchored on *construction*, not on `DETACH DELETE`: the contract suite in
    `test_contracts/` wipes its store through `GraphStoreContractTests.clear()`
    and contains no Cypher of its own, so a query-text grep would have declared
    it safe. Both clients default to `bolt://localhost:7687`, which is what made
    the dev container reachable from a plain `pytest tests/` in the first place.
    """
    offenders = [
        rel
        for path in TESTS_DIR.rglob("*.py")
        if (rel := path.relative_to(TESTS_DIR).as_posix()) not in NON_CONNECTING
        and rel != Path(__file__).name
        and any(
            client in (text := path.read_text(encoding="utf-8"))
            for client in ("Neo4jGraphStore(", "Neo4jHybridSearch(")
        )
        and "nominated_neo4j" not in text
    ]

    assert offenders == []
