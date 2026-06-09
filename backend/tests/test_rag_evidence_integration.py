"""
Integration tests for RAG evidence pipeline.

Covers:
  - source_type mapping completeness
  - unknown source detection
  - duplicate dedup
  - empty sources handling
  - collection selector per intent
  - evidence item_count correctness
  - confidence fallback
"""

from __future__ import annotations

import pytest

from app.rag.collection_selector import CollectionSelector, INTENT_COLLECTION_MAP
from app.rag.evidence import (
    EvidenceItem,
    EvidencePack,
    _map_evidence_type,
    _SOURCE_TO_EVIDENCE,
    build_evidence_pack,
)
from app.schemas.consultation import Source


# ── source_type mapping tests ────────────────────────────────────

class TestSourceTypeMapping:
    """Verify all known source_types map to valid evidence types."""

    def test_collection_source_types_all_mapped(self):
        """All 4 canonical collection source_types should have direct mappings."""
        canonical = [
            "advisory_knowledge",
            "compliance_knowledge",
            "education_knowledge",
            "risk_knowledge",
            "financial_report_knowledge",
        ]
        for st in canonical:
            ev_type = _map_evidence_type(st)
            assert ev_type != "unknown", (
                f"Canonical source_type '{st}' must not map to 'unknown'"
            )

    def test_prefixed_source_types_mapped(self):
        """Prefix-matched source_types should resolve correctly."""
        test_cases = [
            ("risk_indicators", "risk"),
            ("compliance_checklists", "compliance"),
            ("education_tutorials", "education"),
            ("advisory_templates", "advisory"),
            ("financial_report_templates", "advisory"),
            ("investment_knowledge", "advisory"),
            ("asset_allocation_v2", "advisory"),
            ("market_indicators", "risk"),
            ("regulatory_docs", "compliance"),
        ]
        for source_type, expected in test_cases:
            result = _map_evidence_type(source_type)
            assert result == expected, (
                f"source_type '{source_type}' → '{result}', expected '{expected}'"
            )

    def test_unknown_source_type_fallback(self):
        """Truly unknown source_types should map to 'unknown'."""
        unknown_types = [
            "",
            "garbage_data",
            "xyz_abc_123",
            "random_type_no_prefix",
        ]
        for st in unknown_types:
            assert _map_evidence_type(st) == "unknown", (
                f"'{st}' should map to 'unknown'"
            )

    def test_knowledge_base_fallback(self):
        """Generic 'knowledge_base' should map to unknown."""
        assert _map_evidence_type("knowledge_base") == "unknown"


# ── Evidence builder tests ───────────────────────────────────────

class TestEvidenceBuilder:
    """Verify evidence building from sources."""

    def test_empty_sources_returns_empty_pack(self):
        """Empty sources should produce empty EvidencePack, not crash."""
        pack = build_evidence_pack([])
        assert len(pack.items) == 0
        assert pack.low_confidence is True
        assert pack.has_advisory is False
        assert pack.has_risk is False

    def test_mixed_sources_all_typed(self):
        """Sources with various source_types should all have evidence_type."""
        sources = [
            Source(title="Advisory doc", source_type="advisory_knowledge", confidence=0.85),
            Source(title="Risk doc", source_type="risk_knowledge", confidence=0.75),
            Source(title="Compliance doc", source_type="compliance_knowledge", confidence=0.90),
            Source(title="Education doc", source_type="education_knowledge", confidence=0.65),
        ]
        pack = build_evidence_pack(sources)
        assert len(pack.items) == 4
        assert pack.has_advisory is True
        assert pack.has_risk is True
        assert pack.has_compliance is True
        assert pack.has_education is True
        assert pack.low_confidence is False  # top confidence 0.85 > 0.45
        # No unknown items
        unknown = [it for it in pack.items if it.evidence_type == "unknown"]
        assert len(unknown) == 0

    def test_unknown_source_marked_unknown(self):
        """Source with unknown source_type should have evidence_type 'unknown'."""
        sources = [
            Source(title="Something weird", source_type="bizarre", confidence=0.3),
        ]
        pack = build_evidence_pack(sources)
        assert len(pack.items) == 1
        assert pack.items[0].evidence_type == "unknown"

    def test_confidence_missing_treated_low(self):
        """Missing confidence (0.0) should be treated as low confidence."""
        sources = [
            Source(title="No conf doc", source_type="advisory_knowledge", confidence=0.0),
        ]
        pack = build_evidence_pack(sources)
        assert pack.low_confidence is True

    def test_low_confidence_threshold(self):
        """Confidence below 0.45 should trigger low_confidence."""
        sources = [
            Source(title="Low conf", source_type="risk_knowledge", confidence=0.44),
        ]
        pack = build_evidence_pack(sources)
        assert pack.low_confidence is True

    def test_high_confidence_not_low(self):
        """Confidence above 0.45 should not trigger low_confidence."""
        sources = [
            Source(title="Good conf", source_type="advisory_knowledge", confidence=0.85),
        ]
        pack = build_evidence_pack(sources)
        assert pack.low_confidence is False

    def test_duplicate_titles_deduplicated(self):
        """Duplicate titles should be deduplicated."""
        sources = [
            Source(title="Same title", source_type="advisory_knowledge", confidence=0.8),
            Source(title="Same title", source_type="risk_knowledge", confidence=0.7),
            Source(title="Unique", source_type="compliance_knowledge", confidence=0.6),
        ]
        pack = build_evidence_pack(sources)
        assert len(pack.items) == 2
        titles = [it.title for it in pack.items]
        assert titles.count("Same title") == 1

    def test_max_items_limit(self):
        """Evidence pack should respect max_items limit."""
        sources = [
            Source(title=f"Doc {i}", source_type="advisory_knowledge", confidence=0.9)
            for i in range(10)
        ]
        pack = build_evidence_pack(sources, max_items=6)
        assert len(pack.items) == 6

    def test_empty_title_skipped(self):
        """Sources with empty title should be skipped."""
        sources = [
            Source(title="", source_type="advisory_knowledge", confidence=0.8),
            Source(title="Valid", source_type="risk_knowledge", confidence=0.7),
        ]
        pack = build_evidence_pack(sources)
        assert len(pack.items) == 1

    def test_summary_includes_evidence_type(self):
        """Evidence summary should reference the Chinese type label."""
        sources = [
            Source(title="资产配置基础原则", source_type="advisory_knowledge", confidence=0.8),
        ]
        pack = build_evidence_pack(sources)
        assert "投顾知识" in pack.items[0].summary


# ── Collection selector tests ────────────────────────────────────

class TestCollectionSelector:
    """Verify collection selection per intent."""

    def test_all_intents_have_collections(self):
        """All 5 intents should have at least one collection mapped."""
        expected_intents = ["advisory", "financial_report", "risk_control", "compliance", "education"]
        for intent in expected_intents:
            colls = INTENT_COLLECTION_MAP.get(intent, [])
            assert len(colls) > 0, f"Intent '{intent}' has no collections"

    def test_advisory_includes_risk(self):
        """Advisory should include risk_knowledge for risk context."""
        colls = INTENT_COLLECTION_MAP.get("advisory", [])
        assert "risk_knowledge" in colls

    def test_education_includes_risk_and_compliance(self):
        """Education should include risk and compliance for scam detection context."""
        colls = INTENT_COLLECTION_MAP.get("education", [])
        assert "risk_knowledge" in colls
        assert "compliance_knowledge" in colls

    def test_selectors_get_collections_for_intents(self):
        """get_collections_for_intents should merge without duplicates."""
        selector = CollectionSelector()
        merged = selector.get_collections_for_intents("advisory", ["risk_control"])
        # Should have advisory_knowledge first, then risk additions
        assert merged[0] == "advisory_knowledge"
        assert "risk_knowledge" in merged
        # No duplicates
        assert len(merged) == len(set(merged))


# ── source_type inference tests ──────────────────────────────────

class TestSourceTypeInference:
    """Verify source_type inference from collection names."""

    def test_collection_to_source_type_mapping(self):
        """Each collection name maps to a canonical source_type."""
        from app.rag.retriever import _infer_source_type_from_collection
        test_cases = [
            ("advisory_knowledge", "advisory_knowledge"),
            ("compliance_knowledge", "compliance_knowledge"),
            ("education_knowledge", "education_knowledge"),
            ("risk_knowledge", "risk_knowledge"),
            ("unknown_collection", ""),
        ]
        for coll, expected in test_cases:
            result = _infer_source_type_from_collection(coll)
            assert result == expected, f"'{coll}' → '{result}', expected '{expected}'"
