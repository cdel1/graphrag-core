"""Tests for fuzzy matching utilities."""

from __future__ import annotations


class TestNormalizeName:
    def test_lowercases(self):
        from graphrag_core.registry.matching import normalize_name
        assert normalize_name("ACME Corp") == "acme corp"

    def test_strips_punctuation(self):
        from graphrag_core.registry.matching import normalize_name
        assert normalize_name("Acme, Inc.") == "acme inc"

    def test_sorts_tokens(self):
        from graphrag_core.registry.matching import normalize_name
        assert normalize_name("Smith Alice") == "alice smith"

    def test_strips_extra_whitespace(self):
        from graphrag_core.registry.matching import normalize_name
        assert normalize_name("  Acme   Corp  ") == "acme corp"


class TestFuzzyScore:
    def test_exact_match_returns_one(self):
        from graphrag_core.registry.matching import fuzzy_score
        assert fuzzy_score("Acme Corp", "Acme Corp") == 1.0

    def test_case_insensitive(self):
        from graphrag_core.registry.matching import fuzzy_score
        assert fuzzy_score("acme corp", "ACME CORP") == 1.0

    def test_reordering_scores_high(self):
        from graphrag_core.registry.matching import fuzzy_score
        score = fuzzy_score("Alice Smith", "Smith, Alice")
        assert score >= 0.9

    def test_completely_different_scores_low(self):
        from graphrag_core.registry.matching import fuzzy_score
        score = fuzzy_score("Acme Corp", "Zebra Industries")
        assert score < 0.5

    def test_minor_typo_scores_high(self):
        from graphrag_core.registry.matching import fuzzy_score
        score = fuzzy_score("Acme Corporation", "Acme Corporaton")
        assert score >= 0.8

    def test_empty_strings(self):
        from graphrag_core.registry.matching import fuzzy_score
        assert fuzzy_score("", "") == 1.0
