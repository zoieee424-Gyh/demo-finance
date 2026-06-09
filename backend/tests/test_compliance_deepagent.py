"""Tests for ComplianceDeepAgent."""

import pytest
from app.agents.deepagent.compliance_deepagent import (
    ComplianceDeepAgent, validate_compliance_output, repair_compliance_output,
)

_VALID_8 = """\
一、审查对象与场景说明
测试
二、合规风险等级
高
三、违规或高风险表述识别
违规内容
四、适用监管原则与依据
监管原则
五、整改建议
整改
六、可替代表述示例
替代表述
七、审查边界与不确定性
边界
八、合规提示
本报告仅供合规风险识别参考，不构成正式法律意见，不替代律师、持牌机构或合规部门判断，请结合具体业务规则和适用法律法规审慎判断。
"""


@pytest.fixture(autouse=True)
def _block(monkeypatch):
    def _noop(self):
        self._deep_agent_graph = None; self._deepagent_available = False; self._fallback_reason = "blocked"
    monkeypatch.setattr("app.agents.deepagent.base.DeepAgentWrapper._init_deep_agent", _noop)


def _make_req(**kw):
    from app.schemas.consultation import ConsultationRequest
    return ConsultationRequest(question=kw.get("question", "审查合规"), user_profile=kw.get("user_profile", {}))
def _sources():
    from app.schemas.consultation import Source
    return [Source(title="合规基础", source_type="regulations", content_preview="...", confidence=0.8)]


class TestValidator:
    def test_valid(self):
        ok, f = validate_compliance_output(_VALID_8)
        assert ok and f == []
    def test_missing(self):
        ok, f = validate_compliance_output("只有一章")
        assert not ok
    def test_forbidden(self):
        ok, f = validate_compliance_output(_VALID_8 + "\n该内容完全合规，保证通过监管。")
        assert not ok


class TestRepair:
    def test_repair_missing(self):
        repaired = repair_compliance_output("检测到违规表述。")
        for s in ["一、审查对象与场景说明", "八、合规提示"]:
            assert s in repaired
    def test_repair_passes(self):
        ok, _ = validate_compliance_output(repair_compliance_output("粗略分析"))
        assert ok


class TestAgent:
    def test_intent(self):
        a = ComplianceDeepAgent(enabled=True)
        resp = a.answer(_make_req(), _sources())
        assert resp.intent == "compliance"
    def test_debug_info(self):
        a = ComplianceDeepAgent(enabled=True)
        assert a.debug_info["agent_name"] == "compliance_deepagent"
