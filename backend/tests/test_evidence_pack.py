"""
Tests for evidence.py — EvidencePack construction and formatting.

No LLM calls, no API calls — pure logic tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.evidence import (
    EvidencePack,
    EvidenceItem,
    build_evidence_pack,
    format_evidence_section,
    format_evidence_inline,
    _map_evidence_type,
    _build_short_summary,
)
from app.schemas.consultation import Source


# ── Helper ──────────────────────────────────────────────────────────────

def _src(title, source_type="investment_knowledge", confidence=0.75):
    return Source(title=title, source_type=source_type, confidence=confidence)


# ── source_type → evidence_type mapping ─────────────────────────────────


class TestEvidenceTypeMapping:
    """source_type should map to correct evidence_type."""

    def test_investment_knowledge_to_advisory(self):
        assert _map_evidence_type("investment_knowledge") == "advisory"

    def test_asset_allocation_to_advisory(self):
        assert _map_evidence_type("asset_allocation") == "advisory"

    def test_advisory_collection_to_advisory(self):
        assert _map_evidence_type("advisory_knowledge") == "advisory"

    def test_risk_models_to_risk(self):
        assert _map_evidence_type("risk_models") == "risk"

    def test_regulations_to_compliance(self):
        assert _map_evidence_type("regulations") == "compliance"

    def test_financial_education_to_education(self):
        assert _map_evidence_type("financial_education") == "education"

    def test_glossary_to_education(self):
        assert _map_evidence_type("glossary") == "education"

    def test_unknown_type(self):
        assert _map_evidence_type("bizarre_type_xyz") == "unknown"

    def test_prefix_match_risk(self):
        assert _map_evidence_type("risk_indicators") == "risk"
        assert _map_evidence_type("risk_management") == "risk"

    def test_prefix_match_compliance(self):
        assert _map_evidence_type("compliance_checklists") == "compliance"

    def test_prefix_match_education(self):
        assert _map_evidence_type("education_knowledge") == "education"


# ── build_evidence_pack ─────────────────────────────────────────────────


class TestBuildEvidencePack:
    """Core EvidencePack construction tests."""

    def test_empty_sources_low_confidence(self):
        pack = build_evidence_pack([])
        assert pack.low_confidence is True
        assert pack.items == []
        assert not pack.has_advisory
        assert not pack.has_risk

    def test_low_top_confidence(self):
        srcs = [
            _src("Test", confidence=0.30),
            _src("Test2", confidence=0.50),
        ]
        pack = build_evidence_pack(srcs)
        assert pack.low_confidence is True  # top < 0.45

    def test_high_confidence_normal(self):
        srcs = [
            _src("Test", confidence=0.80),
        ]
        pack = build_evidence_pack(srcs)
        assert pack.low_confidence is False

    def test_missing_top_confidence_is_low_confidence(self):
        srcs = [
            Source.model_construct(
                title="Test",
                source_type="investment_knowledge",
                confidence=None,
                url=None,
            ),
            _src("Test2", confidence=0.80),
        ]
        pack = build_evidence_pack(srcs)
        assert pack.low_confidence is True

    def test_max_items_truncation(self):
        srcs = [_src(f"Title {i}") for i in range(10)]
        pack = build_evidence_pack(srcs, max_items=5)
        assert len(pack.items) == 5

    def test_by_type_classification(self):
        srcs = [
            _src("Advisory doc", source_type="investment_knowledge"),
            _src("Risk doc", source_type="risk_models"),
            _src("Compliance doc", source_type="regulations"),
        ]
        pack = build_evidence_pack(srcs)
        assert pack.has_advisory is True
        assert pack.has_risk is True
        assert pack.has_compliance is True
        assert pack.has_education is False
        assert "advisory" in pack.by_type
        assert "risk" in pack.by_type
        assert "compliance" in pack.by_type
        assert len(pack.by_type["advisory"]) == 1

    def test_deduplication_by_title(self):
        srcs = [
            _src("Same Title"),
            _src("Same Title"),
            _src("Different Title"),
        ]
        pack = build_evidence_pack(srcs)
        assert len(pack.items) == 2

    def test_top_titles(self):
        srcs = [_src("A"), _src("B"), _src("C")]
        pack = build_evidence_pack(srcs, max_items=2)
        assert pack.top_titles == ["A", "B"]

    def test_usage_hints_correct(self):
        srcs = [
            _src("A", source_type="investment_knowledge"),
            _src("R", source_type="risk_models"),
            _src("C", source_type="regulations"),
            _src("E", source_type="financial_education"),
        ]
        pack = build_evidence_pack(srcs)
        for item in pack.items:
            if item.evidence_type == "advisory":
                assert "资产配置" in item.usage_hint
            elif item.evidence_type == "risk":
                assert "风险管理" in item.usage_hint
            elif item.evidence_type == "compliance":
                assert "合规边界" in item.usage_hint
            elif item.evidence_type == "education":
                assert "概念解释" in item.usage_hint

    def test_summary_contains_title(self):
        srcs = [_src("资产配置基础原则")]
        pack = build_evidence_pack(srcs)
        assert "资产配置基础原则" in pack.items[0].summary

    def test_confidence_stored(self):
        srcs = [_src("Test", confidence=0.66)]
        pack = build_evidence_pack(srcs)
        assert pack.items[0].confidence == 0.66

    def test_url_stored(self):
        src = Source(title="Test", source_type="investment_knowledge",
                     confidence=0.75, url="http://example.com")
        pack = build_evidence_pack([src])
        assert pack.items[0].url == "http://example.com"


# ── format_evidence_section ─────────────────────────────────────────────


class TestFormatEvidenceSection:
    """Evidence section formatting tests."""

    def test_empty_pack(self):
        pack = build_evidence_pack([])
        result = format_evidence_section(pack)
        assert "无可用的知识库参考资料" in result

    def test_low_confidence_warning(self):
        srcs = [_src("T", confidence=0.30)]
        pack = build_evidence_pack(srcs)
        result = format_evidence_section(pack)
        assert "置信度有限" in result

    def test_formatted_items(self):
        srcs = [
            _src("资产配置基础原则", source_type="investment_knowledge", confidence=0.66),
            _src("投资建议合规边界", source_type="regulations", confidence=0.83),
        ]
        pack = build_evidence_pack(srcs)
        result = format_evidence_section(pack)
        assert "资产配置基础原则" in result
        assert "投资建议合规边界" in result
        assert "66%" in result
        assert "83%" in result


# ── format_evidence_inline ──────────────────────────────────────────────


class TestFormatEvidenceInline:
    """Inline evidence formatting."""

    def test_filter_by_type(self):
        srcs = [
            _src("Advisory A", source_type="investment_knowledge"),
            _src("Risk R", source_type="risk_models"),
        ]
        pack = build_evidence_pack(srcs)
        result = format_evidence_inline(pack, evidence_type="advisory")
        assert "Advisory A" in result
        assert "Risk R" not in result

    def test_max_items(self):
        srcs = [
            _src("A", source_type="investment_knowledge"),
            _src("B", source_type="investment_knowledge"),
            _src("C", source_type="investment_knowledge"),
        ]
        pack = build_evidence_pack(srcs)
        result = format_evidence_inline(pack, evidence_type="advisory", max_items=2)
        # Should only contain 2 titles
        assert result.count("《") == 2

    def test_empty_type_returns_empty(self):
        pack = build_evidence_pack([])
        result = format_evidence_inline(pack, evidence_type="risk")
        assert result == ""
