"""Unit tests for risk control tools."""

import pytest
from app.agents.risk_control_tools.risk_input_parser import parse
from app.agents.risk_control_tools.concentration_risk_checker import check
from app.agents.risk_control_tools.risk_preference_matcher import match
from app.agents.risk_control_tools.liquidity_risk_assessor import assess
from app.agents.risk_control_tools.financial_quality_risk_detector import detect
from app.agents.risk_control_tools.risk_mitigation_planner import plan
from app.agents.risk_control_tools.risk_control_compliance_policy import review

SAMPLE_HOLDINGS = [
    {"asset_class": "权益类", "ratio": 70},
    {"asset_class": "债券类", "ratio": 20},
    {"asset_class": "现金及货币类", "ratio": 10},
]
SAMPLE_FIN_TEXT = "某公司2024年净利润增长20%，经营现金流为-3亿元，资产负债率68%。"


class TestRiskInputParser:
    def test_parse_holdings(self):
        r = parse(question="帮我看看组合风险", user_profile={"risk_preference": "conservative", "holdings": SAMPLE_HOLDINGS})
        assert len(r["parsed_holdings"]) == 3
        assert r["risk_preference"] == "conservative"

    def test_missing_fields(self):
        r = parse(question="评估风险", user_profile={})
        assert "holdings" in r["missing_fields"]

    def test_risk_pref_mapping(self):
        r = parse(question="", user_profile={"risk_preference": "low"})
        assert r["risk_preference"] == "conservative"


class TestConcentrationRiskChecker:
    def test_high_equity_flagged(self):
        r = check(holdings=[{"asset_class": "权益类", "ratio": 80}, {"asset_class": "债券类", "ratio": 20}])
        assert r["concentration_level"] in ("high", "moderate")

    def test_single_class_over_50(self):
        r = check(holdings=[{"asset_class": "权益类", "ratio": 60}, {"asset_class": "债券类", "ratio": 40}])
        has_conc = any("60%" in i for i in r["issues"])
        assert has_conc

    def test_low_cash_flagged(self):
        r = check(holdings=SAMPLE_HOLDINGS)
        assert "cash_elevated" in r["concentration_flags"] or "cash_insufficient" in r["concentration_flags"] or len(r["concentration_flags"]) > 0


class TestRiskPreferenceMatcher:
    def test_conservative_mismatch_high_equity(self):
        r = match(holdings=SAMPLE_HOLDINGS, risk_preference="conservative")
        assert r["match_status"] == "mismatch"
        assert len(r["mismatch_reasons"]) >= 1

    def test_match_when_within_band(self):
        r = match(holdings=[{"asset_class": "债券类", "ratio": 80}, {"asset_class": "现金及货币类", "ratio": 20}], risk_preference="conservative")
        assert r["match_status"] in ("match", "conservative_vs_preference")


class TestLiquidityRiskAssessor:
    def test_assess_cash_flow_pressure(self):
        r = assess(holdings=SAMPLE_HOLDINGS, liquidity_need="high", financial_text=SAMPLE_FIN_TEXT)
        assert r["liquidity_level"] in ("high_risk", "moderate_risk", "elevated")

    def test_adequate_liquidity(self):
        r = assess(holdings=[{"asset_class": "现金及货币类", "ratio": 30}, {"asset_class": "债券类", "ratio": 70}], liquidity_need="low")
        assert r["liquidity_level"] == "adequate"


class TestFinancialQualityRiskDetector:
    def test_detect_profit_ocf_divergence(self):
        r = detect(financial_text=SAMPLE_FIN_TEXT)
        assert len(r["risk_flags"]) >= 1
        assert any("经营现金流" in f for f in r["risk_flags"])

    def test_no_signal_clean(self):
        r = detect(financial_text="公司经营稳定，各项指标正常。")
        assert r["financial_quality_level"] == "no_signal"


class TestRiskMitigationPlanner:
    def test_generates_actions(self):
        conc = check(holdings=SAMPLE_HOLDINGS)
        r = plan(concentration_result=conc)
        assert len(r["mitigation_actions"]) >= 1

    def test_no_trading_terms(self):
        r = plan(concentration_result={"concentration_level": "high", "issues": ["权益类过高"]})
        forbidden = {"买入", "卖出", "加仓", "减仓", "满仓", "空仓"}
        for action in r["mitigation_actions"]:
            for term in forbidden:
                assert term not in action, f"Found '{term}' in '{action}'"


class TestRiskControlCompliancePolicy:
    def test_clean(self):
        r = review(answer="风险审查报告。投资有风险，不构成投资建议。")
        assert r["is_compliant"]

    def test_detect_buy(self):
        r = review(answer="建议买入该股票。")
        assert len(r["warnings"]) >= 1

    def test_detect_position_cmd(self):
        r = review(answer="建议立刻加仓。")
        assert len(r["warnings"]) >= 1

    def test_detect_stock_code(self):
        r = review(answer="推荐600519。")
        assert len(r["warnings"]) >= 1
