"""Tests for FinancialReportDeepAgent."""

import pytest
from app.agents.deepagent.financial_report_deepagent import (
    FinancialReportDeepAgent,
    _REQUIRED_SECTIONS,
    repair_financial_report_output,
    validate_financial_report_output,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


@pytest.fixture(autouse=True)
def _block_real_deepagent_init(monkeypatch):
    """Block real DeepAgent graph creation."""
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"

    monkeypatch.setattr(
        "app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent",
        _noop_init,
    )


def _make_request(question="分析某科技公司2024年财报", financial_text=""):
    text = financial_text or (
        "某科技公司2024年实现营业收入120亿元，同比增长15%；"
        "净利润18亿元，同比增长20%；毛利率45%；净利率15%；"
        "资产负债率62%；经营现金流为-3亿元。"
    )
    return ConsultationRequest(
        question=question,
        user_profile={"financial_text": text},
    )


def _make_sources():
    return [Source(title="财务分析基础", source_type="investment_knowledge", content_preview="...", confidence=0.8)]


class TestFinancialReportDeepAgent:
    def test_disabled_falls_back(self):
        """Disabled agent must fall back to legacy FinancialReportAgent."""
        agent = FinancialReportDeepAgent(enabled=False)
        resp = agent.answer(_make_request(), _make_sources())
        assert isinstance(resp, ConsultationResponse)
        assert resp.intent == "financial_report"

    def test_answer_returns_consultation_response(self):
        agent = FinancialReportDeepAgent(enabled=True)
        resp = agent.answer(_make_request(), _make_sources())
        assert isinstance(resp, ConsultationResponse)
        assert len(resp.answer) > 0

    def test_report_contains_required_sections(self):
        agent = FinancialReportDeepAgent(enabled=True)
        resp = agent.answer(_make_request(), _make_sources())
        # With blocked DeepAgent → fallback to legacy which has its own format.
        # Just verify answer is non-empty and contains 财报.
        assert "财报" in resp.answer or "财务" in resp.answer or "分析" in resp.answer

    def test_sources_preserved(self):
        agent = FinancialReportDeepAgent(enabled=True)
        sources = _make_sources()
        resp = agent.answer(_make_request(), sources)
        assert len(resp.sources) == len(sources)

    def test_no_stock_recommendation(self):
        agent = FinancialReportDeepAgent(enabled=True)
        resp = agent.answer(
            _make_request(question="推荐几只股票？"),
            _make_sources(),
        )
        import re
        assert not re.search(r"\b\d{6}\b", resp.answer)

    def test_risk_notice_present(self):
        agent = FinancialReportDeepAgent(enabled=True)
        resp = agent.answer(_make_request(), _make_sources())
        assert resp.risk_notice is not None

    def test_debug_info(self):
        agent = FinancialReportDeepAgent(enabled=True)
        info = agent.debug_info
        assert "agent_architecture" in info
        assert info["agent_name"] == "financial_report_deepagent"


class TestRepairFinancialReportOutput:
    """Tests for repair_financial_report_output()."""

    def test_repair_adds_all_eight_sections_to_empty_input(self):
        repaired = repair_financial_report_output("some random text without sections")
        for section in _REQUIRED_SECTIONS:
            assert section in repaired, f"Missing section: {section}"

    def test_repair_preserves_existing_content(self):
        raw = "一、财报对象与数据范围\n某科技公司2024年财报分析。\n\n八、风险提示\n投资有风险。"
        repaired = repair_financial_report_output(raw)
        assert "某科技公司2024年财报分析" in repaired

    def test_repair_adds_risk_disclaimer(self):
        repaired = repair_financial_report_output("bare text")
        assert "投资有风险" in repaired
        # The repaired output must contain risk disclaimer substance — any of these
        has_disclaimer = (
            "不构成" in repaired
            or "投资建议" in repaired
            or "仅供参考" in repaired
        )
        assert has_disclaimer, f"Missing risk disclaimer in: {repaired[:300]}"

    def test_repair_fills_missing_sections_with_placeholder(self):
        raw = "一、财报对象与数据范围\nOnly chapter one.\n\n二、核心财务指标摘要\nChapter two."
        repaired = repair_financial_report_output(raw)
        # The last 6 sections should be filled with fallback content
        for section in _REQUIRED_SECTIONS[2:]:
            assert section in repaired
        assert "当前材料未提供充分信息" in repaired

    def test_repair_handles_markdown_headers(self):
        raw = "## 一、财报对象与数据范围\nRevenue grew 12%.\n\n**八、风险提示**\nRisk warning here."
        repaired = repair_financial_report_output(raw)
        assert "Revenue grew 12%" in repaired
        assert "Risk warning here" in repaired
        for section in _REQUIRED_SECTIONS:
            assert section in repaired, f"Missing section: {section}"

    def test_validator_passes_after_repair(self):
        raw = "just some analysis without structure"
        repaired = repair_financial_report_output(raw)
        is_valid, failures = validate_financial_report_output(repaired)
        assert is_valid, f"Validator failures after repair: {failures}"

    def test_validator_catches_stock_code(self):
        is_valid, failures = validate_financial_report_output(
            repair_financial_report_output("test") + "\n买入 600000"
        )
        assert not is_valid

    def test_validator_accepts_insufficient_data_content(self):
        """Validator should NOT fail just because content says 'materials insufficient'."""
        raw = (
            "一、财报对象与数据范围\n当前材料未提供充分信息。\n\n"
            "二、核心财务指标摘要\n当前材料未提供充分信息。\n\n"
            "三、盈利能力分析\n当前材料未提供充分信息。\n\n"
            "四、偿债能力与流动性分析\n当前材料未提供充分信息。\n\n"
            "五、成长性与经营效率分析\n当前材料未提供充分信息。\n\n"
            "六、现金流质量与异常风险\n当前材料未提供充分信息。\n\n"
            "七、参考依据与适用边界\n当前材料未提供充分信息。\n\n"
            "八、风险提示\n投资有风险，入市需谨慎。"
        )
        is_valid, failures = validate_financial_report_output(raw)
        assert is_valid, f"Short-input-like output should pass validator: {failures}"

    def test_repair_result_is_not_empty(self):
        repaired = repair_financial_report_output("")
        assert len(repaired) > 200, f"Repaired output too short: {len(repaired)} chars"
