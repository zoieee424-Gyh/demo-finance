"""Unit tests for financial report analysis tools."""

import pytest
from app.agents.financial_report_tools.financial_text_parser import parse
from app.agents.financial_report_tools.financial_metric_extractor import extract_metrics
from app.agents.financial_report_tools.profitability_analyzer import analyze_profitability
from app.agents.financial_report_tools.solvency_liquidity_analyzer import analyze_solvency
from app.agents.financial_report_tools.growth_efficiency_analyzer import analyze_growth
from app.agents.financial_report_tools.anomaly_risk_detector import detect_risks
from app.agents.financial_report_tools.financial_report_compliance_policy import review


SAMPLE_TEXT = (
    "某科技公司2024年实现营业收入120亿元，同比增长15%；"
    "净利润18亿元，同比增长20%；毛利率45%；净利率15%；"
    "资产负债率62%；经营现金流为-3亿元。"
    "应收账款35亿元，存货22亿元，商誉12亿元。"
)


class TestFinancialTextParser:
    def test_parse_company_name(self):
        r = parse(question="分析某科技公司财报", financial_text=SAMPLE_TEXT)
        assert r["company_name"] == "某科技公司"

    def test_parse_period(self):
        r = parse(question="分析2024年财报", financial_text=SAMPLE_TEXT)
        assert r["report_period"] == "2024"

    def test_parse_sections(self):
        r = parse(question="", financial_text=SAMPLE_TEXT)
        assert "income_statement" in r["extracted_sections"]

    def test_missing_fields_no_text(self):
        r = parse(question="分析一下")
        assert len(r["missing_fields"]) >= 2


class TestFinancialMetricExtractor:
    def test_extract_revenue(self):
        m = extract_metrics(financial_text=SAMPLE_TEXT)
        assert m["revenue"] == pytest.approx(12_000_000_000, rel=0.01)  # 120亿

    def test_extract_net_profit(self):
        m = extract_metrics(financial_text=SAMPLE_TEXT)
        assert m["net_profit"] == pytest.approx(1_800_000_000, rel=0.01)  # 18亿

    def test_extract_debt_ratio(self):
        m = extract_metrics(financial_text=SAMPLE_TEXT)
        assert m["debt_ratio"] == pytest.approx(0.62, abs=0.01)

    def test_extract_operating_cash_flow(self):
        m = extract_metrics(financial_text=SAMPLE_TEXT)
        assert m["operating_cash_flow"] == pytest.approx(-300_000_000, rel=0.01)

    def test_extract_growth(self):
        m = extract_metrics(financial_text=SAMPLE_TEXT)
        assert m["revenue_growth"] == "+15%"
        assert m["profit_growth"] == "+20%"

    def test_unknown_when_empty(self):
        m = extract_metrics(financial_text="")
        assert m["revenue"] is None
        assert m["net_profit"] is None


class TestProfitabilityAnalyzer:
    def test_strong_profitability(self):
        metrics = {
            "revenue": 12_000_000_000, "net_profit": 1_800_000_000,
            "net_margin": 0.15, "gross_margin": 0.45,
            "revenue_growth": "+15%", "profit_growth": "+20%",
        }
        r = analyze_profitability(metrics=metrics)
        assert r["profitability_level"] in ("strong", "moderate")

    def test_unknown_without_data(self):
        r = analyze_profitability(metrics={})
        assert r["profitability_level"] == "unknown"


class TestSolvencyAnalyzer:
    def test_moderate_solvency(self):
        metrics = {"debt_ratio": 0.62, "operating_cash_flow": -300_000_000, "net_profit": 1_800_000_000}
        r = analyze_solvency(metrics=metrics)
        assert "debt_risks" in r
        # Net profit positive + negative OCF → should flag
        assert len(r["debt_risks"]) >= 1

    def test_unknown_no_data(self):
        r = analyze_solvency(metrics={})
        assert r["solvency_level"] == "unknown"


class TestGrowthAnalyzer:
    def test_growth_analysis(self):
        metrics = {"revenue_growth": "+15%", "profit_growth": "+20%", "revenue": 12_000_000_000}
        r = analyze_growth(metrics=metrics)
        assert r["growth_level"] in ("moderate", "strong", "unknown")

    def test_ar_concern(self):
        metrics = {"revenue": 10_000_000_000, "accounts_receivable": 4_000_000_000}
        r = analyze_growth(metrics=metrics)
        assert len(r["efficiency_concerns"]) >= 1


class TestAnomalyRiskDetector:
    def test_profit_negative_ocf_flag(self):
        metrics = {"net_profit": 1_800_000_000, "operating_cash_flow": -300_000_000}
        r = detect_risks(metrics=metrics)
        assert r["risk_count"] >= 1
        assert any("经营现金流" in flag for flag in r["risk_flags"])

    def test_no_risks_clean(self):
        metrics = {"net_profit": 1_000_000_000, "operating_cash_flow": 500_000_000}
        r = detect_risks(metrics=metrics)
        assert r["risk_count"] == 0


class TestFinancialReportCompliancePolicy:
    def test_clean_report(self):
        r = review(answer="公司盈利能力较强，资产负债率合理。投资有风险，入市需谨慎。")
        assert r["is_compliant"] is True
        assert len(r["warnings"]) == 0

    def test_detect_buy_recommendation(self):
        r = review(answer="建议买入该股票，目标价50元。")
        assert len(r["warnings"]) >= 1

    def test_detect_stock_code(self):
        r = review(answer="推荐600519贵州茅台。")
        assert len(r["warnings"]) >= 1

    def test_detect_target_price(self):
        r = review(answer="目标价100元，强烈推荐。")
        assert len(r["warnings"]) >= 1

    def test_detect_return_promise(self):
        r = review(answer="保证年收益20%。")
        assert len(r["warnings"]) >= 1
