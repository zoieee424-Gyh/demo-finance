"""
Tests for SecondaryIntentDetector — rule-based keyword detection.

No LLM, no API calls — pure logic tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.secondary_intent_detector import SecondaryIntentDetector


@pytest.fixture
def detector():
    return SecondaryIntentDetector()


class TestBasicDetection:
    """Core detection logic."""

    def test_advisory_with_risk_and_compliance(self, detector):
        """定投+风险+合规 → secondary = [risk_control, compliance]"""
        result = detector.detect("定投基金有风险吗？合规吗？", primary_intent="advisory")
        assert "risk_control" in result
        assert "compliance" in result
        assert "advisory" not in result  # primary excluded

    def test_conservative_bond_fund_risk(self, detector):
        """保守型+债券基金+风险 → secondary includes education/risk_control"""
        result = detector.detect(
            "保守型投资者能买债券基金吗？风险大吗？", primary_intent="advisory"
        )
        assert "risk_control" in result
        # "风险大" is a strong risk signal
        assert "advisory" not in result

    def test_beginner_risk_learning(self, detector):
        """学理财+风险+入门 → secondary includes risk_control/advisory"""
        result = detector.detect(
            "我想学理财，有什么风险需要注意？入门该看什么？", primary_intent="education"
        )
        assert "risk_control" in result
        assert "advisory" in result
        assert "education" not in result

    def test_market_drop_stop_loss(self, detector):
        """市场下跌+止损+追涨杀跌 → secondary includes advisory"""
        result = detector.detect(
            "市场下跌时应该卖出止损吗？这算不算追涨杀跌？", primary_intent="risk_control"
        )
        assert "advisory" in result
        assert "risk_control" not in result

    def test_primary_excluded(self, detector):
        """Primary intent is never in secondary results."""
        for primary in ["advisory", "compliance", "education", "risk_control"]:
            result = detector.detect("test query", primary_intent=primary)
            assert primary not in result

    def test_deduplication(self, detector):
        """Results should be deduplicated (each intent appears at most once)."""
        result = detector.detect(
            "风险很大，市场风险，系统性风险，信用风险，流动性风险",
            primary_intent="education",
        )
        assert result.count("risk_control") <= 1
        # Only risk keywords → only risk_control
        assert len(result) == 1
        assert result == ["risk_control"]


class TestOutputFormat:
    """Output contract tests."""

    def test_returns_list_of_strings(self, detector):
        result = detector.detect("定投有风险", primary_intent="advisory")
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_all_intents_valid(self, detector):
        """Every returned intent must be a valid intent label."""
        valid = {"advisory", "financial_report", "risk_control", "compliance", "education"}
        result = detector.detect(
            "我想买财报稳健、ROE高的公司股票，但担心市场风险，这合规吗？",
            primary_intent="advisory",
        )
        for item in result:
            assert item in valid

    def test_stable_ordering(self, detector):
        """Same input → same output order (deterministic)."""
        query = "风险很大，合规吗？是什么？怎么配置？"
        r1 = detector.detect(query, primary_intent="education")
        r2 = detector.detect(query, primary_intent="education")
        assert r1 == r2


class TestEdgeCases:
    """Boundary and edge case handling."""

    def test_empty_query(self, detector):
        result = detector.detect("", primary_intent="advisory")
        assert result == []

    def test_no_match(self, detector):
        """Query with no secondary keywords returns empty list."""
        result = detector.detect("你好", primary_intent="advisory")
        assert result == []

    def test_none_primary(self, detector):
        """primary_intent=None should work (no exclusion)."""
        result = detector.detect("什么是基金？风险大吗？", primary_intent=None)
        assert len(result) >= 2  # education + risk_control
        # Without primary exclusion, education can appear
        assert "education" in result

    def test_primary_not_in_valid_set(self, detector):
        """Arbitrary primary string should work (no crash)."""
        result = detector.detect("风险很大", primary_intent="unknown_intent")
        assert "risk_control" in result
        assert "unknown_intent" not in result

    def test_compliance_signals(self, detector):
        """Strong compliance signals detected."""
        result = detector.detect(
            "这个产品保证收益年化8%，能买吗？", primary_intent="advisory"
        )
        assert "compliance" in result

    def test_education_signals(self, detector):
        """Education signals detected with primary advisory."""
        result = detector.detect(
            "什么是基金定投？完全不懂，怎么看？", primary_intent="advisory"
        )
        assert "education" in result

    def test_financial_report_signal(self, detector):
        """Financial report keywords detected."""
        result = detector.detect(
            "这家公司的ROE和现金流怎么样？财报好看吗？", primary_intent="advisory"
        )
        assert "financial_report" in result

    def test_multiple_risk_signals_stable(self, detector):
        """Many risk keywords should still return risk_control once."""
        result = detector.detect(
            "风险大，有市场风险、信用风险、系统性风险、流动性风险，会大跌吗？",
            primary_intent="education",
        )
        assert "risk_control" in result
        # Order stable: risk_control should be first (highest weight)
        assert result[0] == "risk_control"


class TestCrossDomainMatches:
    """Test the specific XDM cases from the evaluation set."""

    def test_xdm001(self, detector):
        """XDM-001: 保守型+债券基金+风险"""
        result = detector.detect(
            "保守型投资者能买债券基金吗？风险大吗？", primary_intent="advisory"
        )
        assert "risk_control" in result

    def test_xdm003(self, detector):
        """XDM-003: 学理财+风险+入门"""
        result = detector.detect(
            "我想学理财，有什么风险需要注意？入门该看什么？", primary_intent="education"
        )
        assert "risk_control" in result
        assert "advisory" in result

    def test_xdm006(self, detector):
        """XDM-006: 市场下跌+止损+追涨杀跌"""
        result = detector.detect(
            "市场下跌时应该卖出止损吗？这算不算追涨杀跌？", primary_intent="risk_control"
        )
        assert "advisory" in result
