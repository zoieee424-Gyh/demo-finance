"""
Tests for CollectionSelector — intent → collection whitelist mapping.

Covers:
  - All 5 intents map to correct collections
  - Fallback when intent unknown
  - Filtering by ChromaStore existence
  - is_collection_allowed
  - get_collection_meta
  - list_all_collections
"""
from __future__ import annotations

import pytest

from app.rag.collection_selector import (
    CollectionSelector,
    INTENT_COLLECTION_MAP,
    FALLBACK_COLLECTION,
)


@pytest.fixture
def selector():
    return CollectionSelector()


# ── Basic mapping ────────────────────────────────────────────────────

def test_advisory_maps_correctly(selector):
    """Advisory intent should include advisory, education, and compliance."""
    colls = selector.get_collections("advisory")
    assert "advisory_knowledge" in colls
    assert "education_knowledge" in colls
    assert "compliance_knowledge" in colls


def test_compliance_maps_correctly(selector):
    colls = selector.get_collections("compliance")
    assert "compliance_knowledge" in colls
    assert colls[0] == "compliance_knowledge"  # Primary collection first


def test_education_maps_correctly(selector):
    colls = selector.get_collections("education")
    assert "education_knowledge" in colls
    assert colls[0] == "education_knowledge"  # Primary collection first


def test_risk_control_maps_correctly(selector):
    colls = selector.get_collections("risk_control")
    assert "risk_knowledge" in colls
    assert "compliance_knowledge" in colls


def test_financial_report_maps_correctly(selector):
    """Financial report intent uses advisory/risk/compliance (no dedicated collection)."""
    colls = selector.get_collections("financial_report")
    assert len(colls) >= 1  # At minimum has advisory_knowledge
    assert colls[0] == "advisory_knowledge"  # Primary is advisory


# ── Fallback ─────────────────────────────────────────────────────────

def test_unknown_intent_falls_back(selector):
    colls = selector.get_collections("nonexistent_intent")
    assert colls == [FALLBACK_COLLECTION]


def test_fallback_collection_is_education():
    assert FALLBACK_COLLECTION == "education_knowledge"


# ── is_collection_allowed ────────────────────────────────────────────

def test_allowed_collection_passes(selector):
    assert selector.is_collection_allowed("advisory_knowledge", "advisory") is True


def test_disallowed_collection_fails(selector):
    # risk_knowledge is now allowed for advisory
    assert selector.is_collection_allowed("risk_knowledge", "advisory") is True
    # But a truly unrelated collection is not allowed
    assert selector.is_collection_allowed("market_knowledge", "compliance") is False


def test_allowed_for_unknown_intent(selector):
    """For unknown intent, only fallback collection is allowed."""
    assert selector.is_collection_allowed("education_knowledge", "unknown") is True
    assert selector.is_collection_allowed("advisory_knowledge", "unknown") is False


# ── Metadata ─────────────────────────────────────────────────────────

def test_get_collection_meta_returns_dict(selector):
    meta = selector.get_collection_meta("advisory_knowledge")
    assert isinstance(meta, dict)
    assert meta.get("display_name") == "投顾知识库"
    assert meta.get("domain") == "investment_advisory"


def test_get_collection_meta_unknown_returns_empty(selector):
    meta = selector.get_collection_meta("nonexistent")
    assert meta == {}


def test_list_all_collections(selector):
    all_colls = selector.list_all_collections()
    assert len(all_colls) >= 4
    names = [c["name"] for c in all_colls]
    assert "advisory_knowledge" in names
    assert "compliance_knowledge" in names
    assert "education_knowledge" in names
    assert "risk_knowledge" in names


# ── Multi-intent collection merging ───────────────────────────────────


class TestGetCollectionsForIntents:
    """Tests for get_collections_for_intents (primary + secondary merge)."""

    @pytest.fixture
    def selector(self):
        return CollectionSelector()

    def test_primary_only_same_as_get_collections(self, selector):
        """When no secondary intents, result should equal get_collections."""
        primary = selector.get_collections("advisory")
        merged = selector.get_collections_for_intents("advisory")
        assert merged == primary

    def test_merges_primary_and_secondary(self, selector):
        """advisory + risk_control → merge advisory's + risk's collections."""
        merged = selector.get_collections_for_intents(
            "advisory", secondary_intents=["risk_control"]
        )
        # advisory: advisory_knowledge, education_knowledge, compliance_knowledge
        # risk_control: risk_knowledge, compliance_knowledge
        # merged: all four (compliance_knowledge appears once)
        assert "advisory_knowledge" in merged
        assert "education_knowledge" in merged
        assert "compliance_knowledge" in merged
        assert "risk_knowledge" in merged
        # Primary collections come first
        advisory_idx = merged.index("advisory_knowledge")
        risk_idx = merged.index("risk_knowledge")
        assert advisory_idx < risk_idx

    def test_deduplication_preserves_order(self, selector):
        """Duplicate collections removed, first occurrence kept."""
        merged = selector.get_collections_for_intents(
            "advisory", secondary_intents=["risk_control"]
        )
        # compliance_knowledge is in both → should appear once
        assert merged.count("compliance_knowledge") == 1

    def test_empty_secondary_ok(self, selector):
        """Empty secondary list should not crash."""
        merged = selector.get_collections_for_intents("advisory", secondary_intents=[])
        assert merged == selector.get_collections("advisory")

    def test_none_secondary_ok(self, selector):
        """None secondary should not crash."""
        merged = selector.get_collections_for_intents("advisory", secondary_intents=None)
        assert merged == selector.get_collections("advisory")

    def test_multiple_secondary_intents(self, selector):
        """education primary + risk_control + advisory secondary."""
        merged = selector.get_collections_for_intents(
            "education",
            secondary_intents=["risk_control", "advisory"],
        )
        # education: education_knowledge
        # risk_control: risk_knowledge, compliance_knowledge
        # advisory: advisory_knowledge, education_knowledge, compliance_knowledge
        assert "education_knowledge" in merged
        assert "risk_knowledge" in merged
        assert "advisory_knowledge" in merged
        # compliance comes from risk_control first
        assert merged.count("compliance_knowledge") == 1

    def test_unknown_secondary_skipped_gracefully(self, selector):
        """Unknown intent in secondary should be skipped (falls back to education_knowledge)."""
        # The fallback collection for unknown is education_knowledge, which
        # is likely already in the result set — so this is effectively a no-op.
        merged = selector.get_collections_for_intents(
            "advisory", secondary_intents=["nonexistent"]
        )
        # Should still have advisory's collections
        assert "advisory_knowledge" in merged


class TestGetCollectionsForIntentsWithStore:
    """Tests with a fake/mock ChromaStore for collection filtering."""

    class _FakeStore:
        """Minimal fake that only has advisory and education collections."""
        def __init__(self):
            self._existing = {"advisory_knowledge", "education_knowledge"}

        def collection_exists(self, name):
            return name in self._existing

    @pytest.fixture
    def selector(self):
        return CollectionSelector()

    @pytest.fixture
    def fake_store(self):
        return TestGetCollectionsForIntentsWithStore._FakeStore()

    def test_store_filters_missing_collections(self, selector, fake_store):
        """risk_knowledge not in fake store → filtered out."""
        merged = selector.get_collections_for_intents(
            "advisory", secondary_intents=["risk_control"],
            chroma_store=fake_store,
        )
        assert "risk_knowledge" not in merged
        assert "advisory_knowledge" in merged
        assert "education_knowledge" in merged

    def test_financial_report_not_exist(self, selector, fake_store):
        """financial_report maps to advisory/risk/compliance — advisory survives."""
        merged = selector.get_collections_for_intents(
            "financial_report", secondary_intents=[],
            chroma_store=fake_store,
        )
        assert merged == ["advisory_knowledge"]  # only advisory exists in fake store

    def test_all_filtered_returns_remaining(self, selector, fake_store):
        """risk_control maps to risk/compliance/advisory — advisory survives filtering."""
        merged = selector.get_collections_for_intents(
            "risk_control", secondary_intents=[],
            chroma_store=fake_store,
        )
        assert merged == ["advisory_knowledge"]  # risk/compliance filtered, advisory remains


# ── Static method ────────────────────────────────────────────────────

def test_is_collection_allowed_static():
    """Static method should work without instantiation."""
    assert CollectionSelector.is_collection_allowed("advisory_knowledge", "advisory")
    assert CollectionSelector.is_collection_allowed("risk_knowledge", "advisory")  # Now allowed
    assert not CollectionSelector.is_collection_allowed("market_knowledge", "compliance")
