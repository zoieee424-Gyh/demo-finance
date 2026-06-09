"""Tests for IntentRouter — one test per intent type."""
import pytest
from app.services.intent_router import IntentRouter


class TestIntentRouter:

    @pytest.fixture
    def router(self) -> IntentRouter:
        return IntentRouter()

    # ── Advisory ────────────────────────────────────────────────
    def test_advisory_routing(self, router):
        """Investment advisory question should route to advisory."""
        result = router.route("我月薪1万，风险厌恶，3年后买房，该如何配置资产？")
        assert result == "advisory"

    # ── Financial Report ────────────────────────────────────────
    def test_financial_report_routing(self, router):
        """Financial report question should route to financial_report."""
        result = router.route("帮我分析一下这家公司的财报，ROE和现金流怎么样？")
        assert result == "financial_report"

    # ── Risk Control ────────────────────────────────────────────
    def test_risk_control_routing(self, router):
        """Risk assessment question should route to risk_control."""
        result = router.route("我想做一次风险评估，看看这个标的有没有财务造假风险")
        assert result == "risk_control"

    def test_holding_concentration_risk_routes_to_risk_control(self, router):
        """Holding concentration risk should not be swallowed by advisory."""
        result = router.route("如何管理持仓集中度风险？")
        assert result == "risk_control"

    # ── Compliance ──────────────────────────────────────────────
    def test_compliance_routing(self, router):
        """Regulatory compliance question should route to compliance."""
        result = router.route("请问最新的资管新规对私募基金备案有什么要求？")
        assert result == "compliance"

    # ── Education ───────────────────────────────────────────────
    def test_education_routing(self, router):
        """Financial literacy question should route to education."""
        result = router.route("什么是市盈率？基金入门需要知道哪些基础知识？")
        assert result == "education"

    # ── Edge case: empty question defaults to education ─────────
    def test_no_match_defaults_to_education(self, router):
        """No-match question should default to education."""
        result = router.route("今天天气真好")
        assert result == "education"
