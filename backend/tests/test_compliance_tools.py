"""Unit tests for compliance tools."""

import pytest
from app.agents.compliance_tools.compliance_input_parser import parse
from app.agents.compliance_tools.prohibited_expression_detector import detect
from app.agents.compliance_tools.suitability_risk_checker import check as suitability_check
from app.agents.compliance_tools.disclosure_completeness_checker import check as disclosure_check
from app.agents.compliance_tools.regulatory_basis_matcher import match
from app.agents.compliance_tools.compliance_rewrite_planner import plan
from app.agents.compliance_tools.compliance_output_policy import review

BAD_CONTENT = "建议买入某某股票，目标价30元，稳赚不赔。"


class TestComplianceInputParser:
    def test_parse_content(self):
        r = parse(question="审查", user_profile={"content_to_review": BAD_CONTENT, "scenario": "investment_advisory"})
        assert "建议买入" in r["review_content"]
        assert r["scenario"] == "investment_advisory"

    def test_missing_content(self):
        r = parse(question="审查", user_profile={})
        assert "content_to_review" in r["missing_fields"]


class TestProhibitedExpressionDetector:
    def test_detect_buy(self):
        r = detect(review_content=BAD_CONTENT)
        assert r["overall_severity"] == "high"
        assert any("建议买入" in f["expression"] for f in r["prohibited_flags"])

    def test_detect_target_price(self):
        r = detect(review_content=BAD_CONTENT)
        assert any("目标价" in f["expression"] for f in r["prohibited_flags"])

    def test_detect_guarantee(self):
        r = detect(review_content=BAD_CONTENT)
        assert any("稳赚不赔" in f.get("expression","") for f in r["prohibited_flags"])


class TestSuitabilityRiskChecker:
    def test_high_risk_to_retail(self):
        r = suitability_check(review_content="推荐股票", audience="retail_investor")
        assert r["suitability_level"] in ("high_risk", "elevated")

    def test_ok(self):
        r = suitability_check(review_content="基本介绍", audience="professional", scenario="education")
        assert r["suitability_level"] == "adequate"


class TestDisclosureChecker:
    def test_missing_disclosures(self):
        r = disclosure_check(review_content=BAD_CONTENT)
        assert len(r["missing_disclosures"]) >= 3

    def test_adequate(self):
        r = disclosure_check(review_content="投资有风险，不构成投资建议。数据来源：公开财报。仅供参考。过往业绩不代表未来。")
        assert r["disclosure_level"] == "adequate"


class TestRegulatoryBasisMatcher:
    def test_match_stock_rec(self):
        flags = [{"expression": "建议买入", "category": "个股推荐"}]
        r = match(prohibited_flags=flags)
        assert r["principle_count"] >= 1


class TestComplianceRewritePlanner:
    def test_generates_alternatives(self):
        r = plan(flags_result={"prohibited_flags": [{"expression": "建议买入", "category": "个股推荐"}]})
        assert len(r["remediation_actions"]) >= 1
        assert len(r["alternative_phrasings"]) >= 1

    def test_no_trading_terms(self):
        r = plan(flags_result={"prohibited_flags": [{"expression": "建议买入", "category": "个股推荐"}]})
        forbidden = {"买入", "卖出", "加仓", "减仓"}
        for a in r["remediation_actions"]:
            for t in forbidden:
                assert t not in a, f"Found '{t}' in remediation"


class TestComplianceOutputPolicy:
    def test_clean(self):
        r = review(answer="合规审查报告。仅供参考，不构成法律意见。")
        assert r["is_compliant"]

    def test_detect_absolute_compliance(self):
        r = review(answer="该内容完全合规。")
        assert len(r["warnings"]) >= 1

    def test_detect_buy(self):
        r = review(answer="建议买入。")
        assert len(r["warnings"]) >= 1
