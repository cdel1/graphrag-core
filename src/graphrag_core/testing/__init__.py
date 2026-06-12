"""Conformance-testing utilities for graphrag-core Protocol implementers.

IMPORT-TIME REQUIREMENT: this subpackage requires pytest and
pytest-asyncio. They are NOT runtime dependencies of graphrag-core —
nothing outside graphrag_core.testing imports this subpackage. Consumers
importing it are by definition running a test suite. (ADR-0006b Rule 9,
ADR-0034.)
"""
