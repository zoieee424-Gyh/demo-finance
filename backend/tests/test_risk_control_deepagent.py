"""Tests for RiskControlDeepAgent."""

import pytest
from app.agents.deepagent.risk_control_deepagent import (
    RISK_CONTROL_SYSTEM_PROMPT,
    RiskControlDeepAgent,
    repair_risk_control_output,
    validate_risk_control_output,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source

_VALID_8_SECTION = """\
一、审查对象与输入范围
测试范围
二、总体风险等级
高风险
三、资产配置与集中度风险
集中度问题
四、流动性与现金流风险
流动性不足
五、财务质量与主体风险
财务风险
六、市场与外部事件风险
外部风险
七、风险缓释建议与监测指标
缓释建议
八、参考依据与风险提示
投资有风险，不构成投资建议。本报告仅作风险管理参考，请结合自身情况审慎判断。
"""


@pytest.fixture(autouse=True)
def _block_real_deepagent_init(monkeypatch):
    def _noop_init(self):
        self._deep_agent_graph = None
        self._deepagent_available = False
        self._fallback_reason = "blocked by test fixture"
    monkeypatch.setattr("app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent", _noop_init)


def _make_req(**kw):
    return ConsultationRequest(
        question=kw.get("question", "评估风险"),
        user_profile=kw.get("user_profile", {}),
    )


def _make_sources():
    return [Source(title="风险管理基础", source_type="risk_indicators", content_preview="...", confidence=0.8)]


class TestValidateRiskControlOutput:
    def test_valid_8_section(self):
        is_valid, failures = validate_risk_control_output(_VALID_8_SECTION)
        assert is_valid is True
        assert len(failures) == 0

    def test_missing_section(self):
        is_valid, failures = validate_risk_control_output("一、审查对象与输入范围\n二、总体风险等级")
        assert is_valid is False

    def test_forbidden_buy(self):
        is_valid, failures = validate_risk_control_output(_VALID_8_SECTION + "\n建议买入该股票。")
        assert is_valid is False
        assert any("买入" in f for f in failures)

    def test_forbidden_stock_code(self):
        is_valid, failures = validate_risk_control_output(_VALID_8_SECTION + "\n推荐600519。")
        assert is_valid is False


class TestRiskControlDeepAgent:
    def test_answer_returns_correct_intent(self):
        agent = RiskControlDeepAgent(enabled=True)
        resp = agent.answer(
            _make_req(question="评估风险", user_profile={
                "risk_preference": "conservative",
                "holdings": [{"asset_class": "权益类", "ratio": 70}, {"asset_class": "债券类", "ratio": 30}],
            }),
            _make_sources(),
        )
        assert isinstance(resp, ConsultationResponse)
        assert resp.intent == "risk_control"

    def test_sources_preserved(self):
        agent = RiskControlDeepAgent(enabled=True)
        sources = _make_sources()
        resp = agent.answer(_make_req(), sources)
        assert len(resp.sources) == len(sources)

    def test_no_stock_code(self):
        agent = RiskControlDeepAgent(enabled=True)
        resp = agent.answer(_make_req(question="评估风险，推荐600519？"), _make_sources())
        import re
        assert not re.search(r"600519", resp.answer)

    def test_debug_info(self):
        agent = RiskControlDeepAgent(enabled=True)
        info = agent.debug_info
        assert info["agent_name"] == "risk_control_deepagent"
        assert "agent_architecture" in info


class TestSystemPrompt:
    """Verify the system prompt contains all required sections and disclaimers."""

    _TITLES = [
        "一、审查对象与输入范围",
        "二、总体风险等级",
        "三、资产配置与集中度风险",
        "四、流动性与现金流风险",
        "五、财务质量与主体风险",
        "六、市场与外部事件风险",
        "七、风险缓释建议与监测指标",
        "八、参考依据与风险提示",
    ]

    def test_prompt_contains_all_8_titles(self):
        for title in self._TITLES:
            assert title in RISK_CONTROL_SYSTEM_PROMPT, f"Missing title: {title}"

    def test_prompt_contains_risk_disclaimer(self):
        assert "本报告仅供风险管理参考" in RISK_CONTROL_SYSTEM_PROMPT
        assert "不构成投资建议" in RISK_CONTROL_SYSTEM_PROMPT

    def test_prompt_contains_output_template(self):
        assert "输出格式模板" in RISK_CONTROL_SYSTEM_PROMPT

    def test_prompt_forbids_trading(self):
        assert "加仓" in RISK_CONTROL_SYSTEM_PROMPT  # mentioned in forbidden context
        assert "买入" in RISK_CONTROL_SYSTEM_PROMPT


class TestRepair:
    """Test deterministic repair of risk control outputs."""

    def test_repair_adds_missing_sections(self):
        raw = "风险评估结果：高风险。建议分散配置。"
        repaired = repair_risk_control_output(raw)
        for title in TestSystemPrompt._TITLES:
            assert title in repaired, f"Repair missing: {title}"

    def test_repair_includes_disclaimer(self):
        repaired = repair_risk_control_output("测试")
        assert "不构成投资建议" in repaired

    def test_repair_preserves_existing_content(self):
        raw = "一、审查对象与输入范围\n测试用户组合审查"
        repaired = repair_risk_control_output(raw)
        assert "测试用户组合审查" in repaired

    def test_repair_passes_validator(self):
        raw = "粗略的风险分析，内容不够完整。"
        repaired = repair_risk_control_output(raw)
        is_valid, failures = validate_risk_control_output(repaired)
        assert is_valid, f"Repair should pass validator: {failures}"
