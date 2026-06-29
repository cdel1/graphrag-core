"""Pure unit tests for Neo4j property serialization helpers.

No Neo4j instance is required — these test only the encode/decode functions.
"""

from __future__ import annotations

import json

from graphrag_core.graph._serialization import (
    _JSON_MARKER,
    _decode_props,
    _encode_props,
)


class TestEncodeProps:
    def test_string_primitive_unchanged(self):
        assert _encode_props({"name": "Zone A"}) == {"name": "Zone A"}

    def test_int_primitive_unchanged(self):
        assert _encode_props({"count": 42}) == {"count": 42}

    def test_float_primitive_unchanged(self):
        assert _encode_props({"ratio": 1.5}) == {"ratio": 1.5}

    def test_bool_primitive_unchanged(self):
        assert _encode_props({"active": True}) == {"active": True}

    def test_none_unchanged(self):
        assert _encode_props({"value": None}) == {"value": None}

    def test_list_of_primitives_not_marked(self):
        """A list of primitive values must pass through as-is, NOT as a JSON string."""
        encoded = _encode_props({"tags": ["a", "b", "c"]})
        assert encoded["tags"] == ["a", "b", "c"]
        assert not isinstance(encoded["tags"], str)

    def test_nested_dict_is_marked(self):
        encoded = _encode_props({"meta": {"key": "val"}})
        assert isinstance(encoded["meta"], str)
        assert encoded["meta"].startswith(_JSON_MARKER)

    def test_list_of_dicts_is_marked(self):
        """Attestation shape: list-of-dicts must be sentinel-marked."""
        encoded = _encode_props({"name": [{"value": "Zone A", "run_id": "x"}]})
        assert isinstance(encoded["name"], str)
        assert encoded["name"].startswith(_JSON_MARKER)

    def test_empty_dict(self):
        assert _encode_props({}) == {}


class TestDecodeProps:
    def test_plain_string_unchanged(self):
        assert _decode_props({"name": "Zone A"}) == {"name": "Zone A"}

    def test_marked_string_decoded(self):
        original = {"key": "val", "num": 42}
        marked = {"meta": _JSON_MARKER + json.dumps(original)}
        assert _decode_props(marked) == {"meta": original}

    def test_non_string_values_unchanged(self):
        assert _decode_props({"x": 7, "y": True}) == {"x": 7, "y": True}

    def test_empty_dict(self):
        assert _decode_props({}) == {}

    def test_invalid_json_after_marker_passes_through(self):
        """A string that starts with the marker but is not valid JSON must not raise."""
        bad = _JSON_MARKER + "not valid json"
        assert _decode_props({"x": bad}) == {"x": bad}


class TestRoundTrip:
    def test_all_primitive_props(self):
        p = {"name": "Zone A", "count": 5, "active": True, "value": None}
        assert _decode_props(_encode_props(p)) == p

    def test_list_of_primitives(self):
        """Lists of primitives survive the round-trip unchanged (no marker added)."""
        p = {"tags": ["a", "b", "c"]}
        assert _decode_props(_encode_props(p)) == p

    def test_nested_dict(self):
        p = {"meta": {"key": "val", "num": 42}}
        assert _decode_props(_encode_props(p)) == p

    def test_list_of_dicts_attestation_shape(self):
        """The attestation shape used in real pipelines must round-trip exactly."""
        p = {"name": [{"value": "Zone A", "run_id": "x"}]}
        assert _decode_props(_encode_props(p)) == p

    def test_genuine_string_no_marker_unaltered(self):
        p = {"text": "hello world"}
        assert _decode_props(_encode_props(p)) == {"text": "hello world"}

    def test_empty_dict(self):
        assert _decode_props(_encode_props({})) == {}

    def test_empty_dict_as_value(self):
        """An empty dict value must round-trip without loss."""
        assert _decode_props(_encode_props({"meta": {}})) == {"meta": {}}

    def test_mixed_props(self):
        """Mix of primitive, list-of-primitives, and nested — all round-trip."""
        p = {
            "label": "EntityA",
            "score": 0.95,
            "tags": ["t1", "t2"],
            "attestation": [{"value": "v", "run_id": "r1"}],
            "meta": {"source": "doc-1"},
        }
        assert _decode_props(_encode_props(p)) == p
