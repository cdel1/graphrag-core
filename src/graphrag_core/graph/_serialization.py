"""Neo4j property serialization helpers.

Neo4j only accepts primitive types or arrays of primitives as property values.
These two helpers transparently encode nested structures (dicts, lists-of-dicts,
etc.) as sentinel-marked JSON strings on write and restore them on read, giving
the Neo4j backend round-trip parity with InMemoryGraphStore.
"""

from __future__ import annotations

import json

# Prefix that cannot appear at the start of a genuine property string.
# Leading space makes it unambiguous; trailing space separates it from the payload.
_JSON_MARKER = " graphrag-json "

_NEO4J_PRIMITIVE = (bool, int, float, str, type(None))


def _is_primitive_list(value: object) -> bool:
    """Return True iff *value* is a list/tuple of Neo4j-primitive elements.

    Args:
        value: The value to inspect.

    Returns:
        True when *value* is a list or tuple whose every element is a
        Neo4j-acceptable primitive (None, bool, int, float, str).
    """
    return isinstance(value, (list, tuple)) and all(
        isinstance(elem, _NEO4J_PRIMITIVE) for elem in value
    )


def _encode_props(props: dict) -> dict:
    """Encode non-primitive property values as sentinel-marked JSON strings.

    Values that are already Neo4j primitives (None, bool, int, float, str) or
    lists/tuples of primitives are passed through unchanged.  Everything else
    (dicts, lists-of-dicts, nested lists, …) is serialised to a JSON string
    prefixed with ``_JSON_MARKER``.

    Args:
        props: Raw property mapping as it comes from a ``GraphNode`` or
            ``GraphRelationship``.

    Returns:
        A new dict safe to pass as a Neo4j parameter bag.
    """
    result: dict = {}
    for key, value in props.items():
        if isinstance(value, _NEO4J_PRIMITIVE) or _is_primitive_list(value):
            result[key] = value
        else:
            result[key] = _JSON_MARKER + json.dumps(value)
    return result


def _select_label(labels: list[str]) -> str:
    """Select the most meaningful label from a Neo4j label list.

    Prefers the first non-``Chunk`` label (a node may carry both a domain label
    and the ``Chunk`` substrate label).  Falls back to ``"Chunk"`` when that is
    the only label present, and to ``"Unknown"`` when the list is empty.

    Args:
        labels: The list of Neo4j labels returned by ``labels(n)``.

    Returns:
        A single label string — never raises ``IndexError``.
    """
    non_chunk = [lbl for lbl in labels if lbl != "Chunk"]
    return non_chunk[0] if non_chunk else (labels[0] if labels else "Unknown")


def _decode_props(props: dict) -> dict:
    """Restore property values that were encoded by ``_encode_props``.

    Any string value whose first characters match ``_JSON_MARKER`` is decoded
    via ``json.loads``; all other values are passed through unchanged.

    Args:
        props: Property mapping as returned from a Neo4j record.

    Returns:
        A new dict with sentinel-marked strings replaced by their original
        Python structures.
    """
    result: dict = {}
    for key, value in props.items():
        if isinstance(value, str) and value.startswith(_JSON_MARKER):
            try:
                result[key] = json.loads(value[len(_JSON_MARKER):])
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = value
    return result
