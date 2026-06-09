"""Tests for ComplianceGuard — covering key rule categories."""
import pytest
from app.services.compliance_guard import ComplianceGuard, _needs_risk_notice
from app.schemas.consultation import ConsultationResponse


class TestComplianceGuard:

    @pytest.fixture
    def guard(self) -> ComplianceGuard:
        return ComplianceGuard()

    def _make_response(self, answer: str, intent: str = "advisory") -> ConsultationResponse:
        return ConsultationResponse(
            intent=intent,
            agent="test_agent",
            answer=answer,
            risk_notice="",
            sources=[],
            warnings=[],
        )

    # ── Stock tipping detection ─────────────────────────────────
    def test_detect_stock_recommendation(self, guard):
        """Answers recommending buying stocks should be flagged."""
        resp = self._make_response("根据分析，我推荐买入贵州茅台的股票，现在正是建仓的好时机。")
        result = guard.review(resp)
        assert len(result.warnings) > 0
        assert any("荐股" in w or "交易操作" in w for w in result.warnings)

    def test_detect_stock_code(self, guard):
        """Mentioning stock codes should trigger a warning."""
        resp = self._make_response("这只股票的代码是600519，可以考虑关注。")
        result = guard.review(resp)
        assert any("股票代码" in w for w in result.warnings)

    # ── Price prediction detection ──────────────────────────────
    def test_detect_price_prediction(self, guard):
        """Predicting price movements should be flagged."""
        resp = self._make_response("这只基金肯定会大涨，预计涨幅达到20%以上。")
        result = guard.review(resp)
        assert len(result.warnings) >= 2  # 肯定大涨 + 涨幅20%

    # ── Return promise detection ────────────────────────────────
    def test_detect_return_promise(self, guard):
        """Promising guaranteed returns should be flagged."""
        resp = self._make_response("这个产品保本保收益，年化收益8%，稳赚不赔。")
        result = guard.review(resp)
        assert len(result.warnings) >= 2  # 保本 + 稳赚

    # ── Investment decision substitution ────────────────────────
    def test_detect_decision_substitution(self, guard):
        """Deciding for the user should be flagged."""
        resp = self._make_response("根据你的情况，你应该买入沪深300ETF，赶紧上车别错过机会。")
        result = guard.review(resp)
        assert any("替用户" in w or "投资决策" in w for w in result.warnings)

    # ── Clean answer passes ─────────────────────────────────────
    def test_clean_answer_has_no_violations(self, guard):
        """A compliant answer should have no violation-level warnings."""
        resp = self._make_response(
            "根据公开财报数据，该公司过去三年营收复合增长率为15%，"
            "但2025年因行业调整增速放缓至8%。投资前请充分了解相关风险，"
            "过往业绩不代表未来表现。以上信息仅供参考，不构成投资建议。"
        )
        result = guard.review(resp)
        # May have risk_notice, but should have no violation warnings
        violations = [w for w in result.warnings if w.startswith("[违规]")]
        assert len(violations) == 0

    # ── Risk notice attachment ──────────────────────────────────
    def test_risk_notice_for_investment_content(self, guard):
        """Investment-related answers should receive a risk notice."""
        resp = self._make_response("建议采用股债60/40的配置比例，定期再平衡。")
        result = guard.review(resp)
        assert result.risk_notice != ""
        assert "投资有风险" in result.risk_notice

    def test_risk_notice_not_duplicated(self, guard):
        """Existing risk_notice should not be overwritten."""
        existing = "本回答仅供参考。"
        resp = ConsultationResponse(
            intent="advisory",
            agent="test",
            answer="建议配置一些债券基金",
            risk_notice=existing,
            sources=[],
            warnings=[],
        )
        result = guard.review(resp)
        assert result.risk_notice == existing  # Should keep original

    # ── Warnings are accumulated ────────────────────────────────
    def test_warnings_are_appended_not_replaced(self, guard):
        """Existing warnings should be preserved, new ones appended."""
        resp = ConsultationResponse(
            intent="advisory",
            agent="test",
            answer="推荐买入这只股票，肯定会涨！",
            risk_notice="",
            sources=[],
            warnings=["pre-existing-warning"],
        )
        result = guard.review(resp)
        assert "pre-existing-warning" in result.warnings
        assert len(result.warnings) > 1  # original + new ones


class TestRiskNoticeHeuristic:

    def test_investment_content_triggers(self):
        assert _needs_risk_notice("建议配置一些债券基金")
        assert _needs_risk_notice("股票投资的风险需要注意")

    def test_non_investment_does_not_trigger(self):
        assert not _needs_risk_notice("今天天气很好")
        assert not _needs_risk_notice("请问怎么开户")
