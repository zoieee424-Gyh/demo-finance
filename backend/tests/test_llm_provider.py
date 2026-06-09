
"""Tests for LLMProvider adapter layer and integration points."""
import os
import pytest


# ================================================================
# LLMProvider
# ================================================================

class TestLLMProvider:

    def test_default_model_is_deepseek_v4_flash(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider()
        assert provider.model == "deepseek-v4-flash"

    def test_not_configured_without_api_key(self):
        from app.llm.provider import LLMProvider
        # Clear env var to ensure clean state
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        try:
            provider = LLMProvider(enabled=False)
            assert not provider.is_configured()
        finally:
            if old_key is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_key

    def test_not_configured_when_disabled(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider(
            api_key="test-key",
            enabled=False,
        )
        assert not provider.is_configured()

    def test_configured_when_enabled_and_api_key_set(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider(
            api_key="test-key",
            enabled=True,
        )
        assert provider.is_configured()

    def test_complete_returns_mock_when_not_configured(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider()
        result = provider.complete([{"role": "user", "content": "hello"}])
        assert result.startswith("[LLM Mock]")

    def test_complete_returns_mock_when_configured_but_no_network(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider(
            api_key="test-key",
            enabled=True,
            mock_mode=True,
        )
        result = provider.complete([{"role": "user", "content": "hello"}])
        assert "Mock" in result

    def test_mock_mode_can_be_disabled_without_base_url(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider(
            api_key="test-key",
            enabled=True,
            mock_mode=False,
        )
        assert provider.is_configured()
        assert provider.mock_mode is False

    def test_complete_does_not_throw_on_empty_messages(self):
        from app.llm.provider import LLMProvider
        provider = LLMProvider()
        result = provider.complete([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_provider_returns_singleton(self):
        from app.llm.provider import get_provider
        p1 = get_provider()
        p2 = get_provider()
        assert p1 is p2

    def test_api_key_from_env(self):
        from app.llm.provider import LLMProvider
        os.environ["DEEPSEEK_API_KEY"] = "env-test-key"
        try:
            provider = LLMProvider()
            assert provider.api_key == "env-test-key"
        finally:
            del os.environ["DEEPSEEK_API_KEY"]


# ================================================================
# Prompts
# ================================================================

class TestPrompts:

    def test_profile_prompt_contains_compliance_rules(self):
        from app.llm.prompts import build_profile_prompt
        prompt = build_profile_prompt("{}", "income_level", "test question")
        assert "不得推荐" in prompt
        assert "市场涨跌" in prompt or "预测" in prompt
        assert "投资决策" in prompt

    def test_report_prompt_contains_compliance_rules(self):
        from app.llm.prompts import build_report_rewrite_prompt
        prompt = build_report_rewrite_prompt("test report")
        assert "不得修改" in prompt
        assert "风险提示" in prompt
        assert "不构成投资决策依据" in prompt or "投资建议" in prompt

    def test_profile_prompt_includes_user_question(self):
        from app.llm.prompts import build_profile_prompt
        prompt = build_profile_prompt("{}", "income_level", "如何配置资产？")
        assert "如何配置资产？" in prompt

    def test_report_prompt_includes_original_report(self):
        from app.llm.prompts import build_report_rewrite_prompt
        original = "这是原始报告内容"
        prompt = build_report_rewrite_prompt(original)
        assert original in prompt


# ================================================================
# Profile Enrichment
# ================================================================

class TestProfileEnrichment:

    def test_enrich_with_none_provider_returns_unchanged(self):
        from app.agents.investment_advisor_tools.profile_analyzer import (
            analyze, enrich_profile_with_llm,
        )
        profile = analyze("月薪1万，风险厌恶，如何配置资产？")
        original_missing = list(profile.get("missing_fields", []))
        result = enrich_profile_with_llm(profile, "如何配置资产？", provider=None)
        assert result is profile  # Same object returned
        assert result.get("missing_fields") == original_missing

    def test_enrich_with_unconfigured_provider_returns_unchanged(self):
        from app.llm.provider import LLMProvider
        from app.agents.investment_advisor_tools.profile_analyzer import (
            analyze, enrich_profile_with_llm,
        )
        provider = LLMProvider()  # Not configured
        profile = analyze("月薪1万，如何理财？")
        result = enrich_profile_with_llm(profile, "如何理财？", provider=provider)
        # Should be unchanged since provider is not configured
        assert "income_level" in result

    def test_enrich_does_not_override_explicit_fields(self):
        from app.llm.provider import LLMProvider
        from app.agents.investment_advisor_tools.profile_analyzer import (
            analyze, enrich_profile_with_llm,
        )
        provider = LLMProvider()  # Not configured = no change
        profile = analyze(
            "我想买股票",
            user_profile={"risk_preference": "low"},
        )
        result = enrich_profile_with_llm(profile, "我想买股票", provider=provider)
        assert result["risk_preference"] == "conservative"


# ================================================================
# Report Rewrite
# ================================================================

class TestReportRewrite:

    def test_rewrite_with_none_provider_returns_original(self):
        from app.agents.investment_advisor import rewrite_report_with_llm
        original = "测试报告内容"
        result = rewrite_report_with_llm(original, provider=None)
        assert result == original

    def test_rewrite_with_unconfigured_provider_returns_original(self):
        from app.llm.provider import LLMProvider
        from app.agents.investment_advisor import rewrite_report_with_llm
        provider = LLMProvider()  # Not configured
        original = "包含用户画像和风险提示的完整报告"
        result = rewrite_report_with_llm(original, provider=provider)
        assert result == original

    def test_rewrite_preserves_original_for_mock_response(self):
        from app.llm.provider import LLMProvider
        from app.agents.investment_advisor import rewrite_report_with_llm
        provider = LLMProvider(
            api_key="test-key",
            enabled=True,
            mock_mode=True,
        )
        original = "包含用户画像和风险提示的完整报告"
        result = rewrite_report_with_llm(original, provider=provider)
        # Even when configured, mock mode returns original (via [LLM Mock] detection)
        assert result == original

    def test_rewrite_rejects_changed_percentage(self):
        from app.agents.investment_advisor import rewrite_report_with_llm

        class FakeProvider:
            def is_configured(self):
                return True

            def complete(self, messages, temperature=0.2, max_tokens=800):
                return (
                    "一、用户画像摘要\n"
                    "二、投资目标分析\n"
                    "三、风险评估结果\n风险等级：保守型\n"
                    "四、资产配置建议\n现金及货币类：50%\n"
                    "五、基金定投规划\n"
                    "六、持仓诊断\n"
                    "七、市场热点解读\n"
                    "八、参考依据与适用边界\n"
                    "九、风险提示\n投资有风险。"
                )

        original = (
            "一、用户画像摘要\n"
            "二、投资目标分析\n"
            "三、风险评估结果\n风险等级：保守型\n"
            "四、资产配置建议\n现金及货币类：40%\n"
            "五、基金定投规划\n"
            "六、持仓诊断\n"
            "七、市场热点解读\n"
            "八、参考依据与适用边界\n"
            "九、风险提示\n投资有风险。"
        )
        assert rewrite_report_with_llm(original, provider=FakeProvider()) == original

    def test_rewrite_accepts_structure_preserving_text(self):
        from app.agents.investment_advisor import rewrite_report_with_llm

        class FakeProvider:
            def is_configured(self):
                return True

            def complete(self, messages, temperature=0.2, max_tokens=800):
                return (
                    "一、用户画像摘要\n表达更流畅。\n"
                    "二、投资目标分析\n"
                    "三、风险评估结果\n风险等级：保守型\n"
                    "四、资产配置建议\n现金及货币类：40%\n"
                    "五、基金定投规划\n"
                    "六、持仓诊断\n"
                    "七、市场热点解读\n"
                    "八、参考依据与适用边界\n"
                    "九、风险提示\n投资有风险。"
                )

        original = (
            "一、用户画像摘要\n"
            "二、投资目标分析\n"
            "三、风险评估结果\n风险等级：保守型\n"
            "四、资产配置建议\n现金及货币类：40%\n"
            "五、基金定投规划\n"
            "六、持仓诊断\n"
            "七、市场热点解读\n"
            "八、参考依据与适用边界\n"
            "九、风险提示\n投资有风险。"
        )
        result = rewrite_report_with_llm(original, provider=FakeProvider())
        assert result != original
        assert "40%" in result
