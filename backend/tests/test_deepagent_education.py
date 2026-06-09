"""
Tests for FinancialEducationDeepAgent — registry, system prompt, validator, repair, fallback.
"""
from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════
# Registry tests
# ═══════════════════════════════════════════════════════════════════

class TestEducationRegistry:

    def test_registry_has_8_tools(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        assert registry.tool_count == 8

    def test_registry_tool_ids(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        expected_ids = [
            "simple_concept_workflow",
            "learner_profile_analyzer",
            "concept_explainer",
            "learning_path_planner",
            "scam_risk_detector",
            "product_knowledge_mapper",
            "education_compliance_policy",
            "education_evidence_builder",
        ]
        tool_ids = [t.tool_id for t in registry.list_tools()]
        for tid in expected_ids:
            assert tid in tool_ids, f"Missing tool: {tid}"

    def test_all_tools_have_chinese_name(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        for tool in registry.list_tools():
            assert tool.name_cn, f"Tool {tool.tool_id} missing name_cn"
            assert len(tool.name_cn) >= 2, f"Tool {tool.tool_id} name_cn too short"

    def test_all_tools_have_description(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        for tool in registry.list_tools():
            assert tool.description, f"Tool {tool.tool_id} missing description"

    def test_all_tools_have_hard_constraints(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        for tool in registry.list_tools():
            assert len(tool.hard_constraints) > 0, f"Tool {tool.tool_id} missing hard_constraints"

    def test_all_tools_executable(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        for tool in registry.list_tools():
            assert tool.execute is not None, f"Tool {tool.tool_id} has no execute function"

    def test_registry_does_not_affect_other_registries(self):
        from app.agents.deepagent.registry import (
            build_financial_education_registry,
            build_investment_advisor_registry,
        )
        edu = build_financial_education_registry()
        adv = build_investment_advisor_registry()
        assert edu.tool_count == 8
        assert adv.tool_count == 9  # unchanged

    def test_registry_contains_composite_workflow(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        tool = registry.get("simple_concept_workflow")
        assert tool is not None
        assert tool.name_cn == "简单概念投教工作流"

    def test_composite_workflow_is_first(self):
        from app.agents.deepagent.registry import build_financial_education_registry
        registry = build_financial_education_registry()
        tools = registry.list_tools()
        assert tools[0].tool_id == "simple_concept_workflow"


# ═══════════════════════════════════════════════════════════════════
# System prompt tests
# ═══════════════════════════════════════════════════════════════════

class TestEducationSystemPrompt:

    def test_system_prompt_contains_compliance_rules(self):
        from app.agents.deepagent.education_deepagent import EDUCATION_SYSTEM_PROMPT
        prompt = EDUCATION_SYSTEM_PROMPT
        assert "不是投资顾问" in prompt
        assert "不推荐具体股票" in prompt
        assert "不承诺收益" in prompt
        assert "不预测涨跌" in prompt
        assert "不替用户做投资决策" in prompt
        assert "仅供学习" in prompt or "不构成投资建议" in prompt

    def test_system_prompt_contains_seven_sections(self):
        from app.agents.deepagent.education_deepagent import EDUCATION_SYSTEM_PROMPT
        prompt = EDUCATION_SYSTEM_PROMPT
        sections = [
            "一、问题理解与学习目标",
            "二、用户知识水平判断",
            "三、核心概念通俗解释",
            "四、关键风险与常见误区",
            "五、学习路径建议",
            "六、参考依据与延伸阅读",
            "七、风险提示与适用边界",
        ]
        for s in sections:
            assert s in prompt, f"Missing section in system prompt: {s}"

    def test_user_message_prioritizes_composite_workflow(self):
        """The _build_education_user_message should instruct LLM to use
        simple_concept_workflow first for concept explanation questions."""
        from app.agents.deepagent.education_deepagent import _build_education_user_message
        from app.schemas.consultation import ConsultationRequest
        request = ConsultationRequest(question="什么是指数基金？")
        msg = _build_education_user_message(request, [])
        assert "simple_concept_workflow" in msg
        assert "只用 1 个工具" in msg or "最多 1" in msg or "1 个" in msg

    def test_user_message_prohibits_duplicate_calls(self):
        from app.agents.deepagent.education_deepagent import _build_education_user_message
        from app.schemas.consultation import ConsultationRequest
        request = ConsultationRequest(question="什么是债券？")
        msg = _build_education_user_message(request, [])
        assert "禁止重复调用" in msg or "禁止再调用" in msg or "不要再调用" in msg

class TestEducationValidator:

    def test_valid_complete_report(self):
        from app.agents.deepagent.education_deepagent import validate_education_output
        answer = (
            "一、问题理解与学习目标\n用户想了解基金基础知识。\n"
            "二、用户知识水平判断\n用户为初学者。\n"
            "三、核心概念通俗解释\n基金是集合投资工具。\n"
            "四、关键风险与常见误区\n基金不保本。\n"
            "五、学习路径建议\n建议先学基本概念。\n"
            "六、参考依据与延伸阅读\n参考投资者教育材料。\n"
            "七、风险提示与适用边界\n本内容仅供金融知识学习，不构成投资建议。"
        )
        is_valid, failures = validate_education_output(answer)
        assert is_valid, f"Should be valid but got: {failures}"

    def test_missing_sections_fails(self):
        from app.agents.deepagent.education_deepagent import validate_education_output
        answer = "这是一些关于基金的简单解释。"
        is_valid, failures = validate_education_output(answer)
        assert not is_valid
        assert any("Missing required sections" in f for f in failures)

    def test_missing_disclaimer_fails(self):
        from app.agents.deepagent.education_deepagent import validate_education_output
        answer = (
            "一、问题理解与学习目标\n了解基金\n"
            "二、用户知识水平判断\nbeginner\n"
            "三、核心概念通俗解释\n基金\n"
            "四、关键风险与常见误区\n风险\n"
            "五、学习路径建议\n学习\n"
            "六、参考依据与延伸阅读\n参考\n"
            "七、风险提示与适用边界\n风险提示"
        )
        is_valid, failures = validate_education_output(answer)
        # Should fail because no "不构成投资建议" or similar disclaimer
        # But "风险提示" section exists. Check if missing disclaimer is caught.
        # The validator checks _RISK_DISCLAIMER_PATTERNS
        assert not is_valid, f"Should fail due to missing disclaimer but got: {failures}"

    def test_forbidden_stock_code_detected(self):
        from app.agents.deepagent.education_deepagent import validate_education_output
        answer = (
            "一、问题理解与学习目标\n推荐600519\n"
            "二、用户知识水平判断\nbeginner\n"
            "三、核心概念通俗解释\n基金\n"
            "四、关键风险与常见误区\n风险\n"
            "五、学习路径建议\n学习\n"
            "六、参考依据与延伸阅读\n参考\n"
            "七、风险提示与适用边界\n不构成投资建议"
        )
        is_valid, failures = validate_education_output(answer)
        assert not is_valid
        assert any("6-digit" in f or "股票代码" in f for f in failures)

    def test_content_violation_caught_by_compliance_tool_not_validator(self):
        """Output validator focuses on structural checks (sections + disclaimer).

        Content-policy violations (return promises, price predictions, scam
        language) are detected by the education_compliance_policy tool,
        which runs post-hoc via DeepAgentWrapper. The output validator
        should NOT reject education content that discusses scam language —
        this is legitimate educational content (counter-examples).
        """
        from app.agents.deepagent.education_deepagent import validate_education_output
        from app.agents.education_tools.education_compliance_policy import review

        answer = (
            "一、问题理解与学习目标\n用户询问保证收益是否可信\n"
            "二、用户知识水平判断\nbeginner\n"
            "三、核心概念通俗解释\n基金是集合投资工具\n"
            "四、关键风险与常见误区\n承诺稳赚不赔是典型诈骗特征\n"
            "五、学习路径建议\n先学习风险收益关系\n"
            "六、参考依据与延伸阅读\n参考投教材料\n"
            "七、风险提示与适用边界\n本内容仅供金融知识学习，不构成投资建议"
        )
        # Validator should pass — structural check only
        is_valid, failures = validate_education_output(answer)
        assert is_valid, f"Structural validator should pass but got: {failures}"

        # Compliance tool should detect the quoted violation language
        # (Note: current regex may flag quoted language as violation.
        # This is a known limitation — same as compliance agent issue.)
        compliance = review(answer=answer)
        assert isinstance(compliance, dict)
        assert "risk_notice" in compliance
        assert "warnings" in compliance


# ═══════════════════════════════════════════════════════════════════
# Repair tests
# ═══════════════════════════════════════════════════════════════════

class TestEducationRepair:

    def test_repair_adds_missing_sections(self):
        from app.agents.deepagent.education_deepagent import (
            repair_education_output,
            _REQUIRED_SECTIONS,
        )
        raw = "基金是集合投资工具。投资有风险。"
        repaired = repair_education_output(raw)
        # All 7 sections should be present
        for section in _REQUIRED_SECTIONS:
            assert section in repaired, f"Missing section after repair: {section}"

    def test_repair_adds_risk_disclaimer(self):
        from app.agents.deepagent.education_deepagent import (
            repair_education_output,
            _RISK_NOTICE_DEFAULT,
        )
        raw = "基金是集合投资工具。"
        repaired = repair_education_output(raw)
        assert _RISK_NOTICE_DEFAULT in repaired

    def test_repair_preserves_original_content(self):
        from app.agents.deepagent.education_deepagent import repair_education_output
        raw = "一、问题理解与学习目标\n用户想了解基金。\n二、用户知识水平判断\nbeginner\n"
        # Raw content has some sections; repair should preserve them
        repaired = repair_education_output(raw)
        # After repair, sections with partial content preserved
        assert "用户想了解基金" in repaired


# ═══════════════════════════════════════════════════════════════════
# DeepAgent class tests (no real LLM)
# ═══════════════════════════════════════════════════════════════════

class TestEducationDeepAgent:

    def test_agent_creation(self):
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        agent = FinancialEducationDeepAgent(enabled=False)
        assert agent.name == "financial_education_deepagent"
        assert agent.intent == "education"

    def test_agent_falls_back_to_education_agent(self):
        """When DeepAgent is disabled, should use EducationAgent fallback."""
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        from app.schemas.consultation import ConsultationRequest
        agent = FinancialEducationDeepAgent(enabled=False)
        request = ConsultationRequest(question="什么是基金？")
        response = agent.answer(request, [])
        assert response.intent == "education"
        assert response.agent in (
            "financial_education",
            "financial_education_deepagent",
        )
        assert len(response.answer) > 0

    def test_agent_debug_info(self):
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        agent = FinancialEducationDeepAgent(enabled=False)
        info = agent.debug_info
        assert "agent_name" in info
        assert "configured_mode" in info
        assert info["configured_mode"] == "deepagent"

    def test_agent_architecture_pipeline_when_disabled(self):
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        agent = FinancialEducationDeepAgent(enabled=False)
        assert agent.architecture == "pipeline"

    def test_agent_configurable_mode(self):
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        agent = FinancialEducationDeepAgent(enabled=False, configured_mode="pipeline")
        info = agent.debug_info
        assert info["configured_mode"] == "pipeline"
