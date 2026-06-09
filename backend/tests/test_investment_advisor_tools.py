"""Tests for InvestmentAdvisorAgent internal tool modules and end-to-end flow."""
import pytest
from app.schemas.consultation import ConsultationRequest, Source
from app.agents.investment_advisor import InvestmentAdvisorAgent
from app.agents.investment_advisor_tools.profile_analyzer import analyze as analyze_profile
from app.agents.investment_advisor_tools.goal_planner import plan as plan_goals
from app.agents.investment_advisor_tools.risk_assessor import assess as assess_risk
from app.agents.investment_advisor_tools.allocation_engine import allocate as allocate_assets
from app.agents.investment_advisor_tools.advisory_compliance_policy import review as review_compliance


class TestProfileAnalyzer:

    def test_extract_risk_preference_from_question(self):
        result = analyze_profile("risk厌恶，不想亏钱，请问如何理财？")
        assert result["risk_preference"] == "conservative"

    def test_extract_beginner_experience(self):
        result = analyze_profile("我是投资小白，刚入门，想学习理财")
        assert result["investment_experience"] == "beginner"

    def test_extract_income_from_wages(self):
        result = analyze_profile("月薪1万，如何配置资产？")
        assert result["income_level"] == "medium"

    def test_extract_income_high(self):
        result = analyze_profile("月薪3万，想投资")
        assert result["income_level"] == "high"

    def test_extract_income_low(self):
        result = analyze_profile("月薪5千，想存钱")
        assert result["income_level"] == "low"

    def test_extract_income_thousand_unit_as_medium(self):
        result = analyze_profile("月薪9千，想做稳健理财")
        assert result["income_level"] == "medium"

    def test_detect_house_purchase_constraint(self):
        result = analyze_profile("计划3年后买房，该如何理财？")
        assert any("买房" in c for c in result["constraints"])

    def test_liquidity_need_for_house_purchase(self):
        result = analyze_profile("3年后买房，如何配置资产？")
        assert result["liquidity_need"] == "high"

    def test_missing_fields_when_incomplete(self):
        result = analyze_profile("我想投资")
        assert len(result["missing_fields"]) >= 1

    def test_explicit_profile_overrides_question(self):
        result = analyze_profile(
            "我想买股票",
            user_profile={"risk_preference": "low", "investment_experience": "experienced"},
        )
        assert result["risk_preference"] == "conservative"
        assert result["investment_experience"] == "experienced"

class TestGoalPlanner:

    def test_detect_house_purchase_goal(self):
        goals = plan_goals("3年后买房，该如何存首付？")
        assert len(goals) >= 1
        house_goal = [g for g in goals if g["goal_type"] == "house_purchase"]
        assert len(house_goal) == 1

    def test_house_purchase_has_correct_time_horizon(self):
        goals = plan_goals("3年后买房")
        house = [g for g in goals if g["goal_type"] == "house_purchase"][0]
        assert house["time_horizon_months"] == 36

    def test_short_term_house_is_capital_preservation(self):
        goals = plan_goals("1年后买房")
        house = [g for g in goals if g["goal_type"] == "house_purchase"][0]
        assert house["priority"] == "capital_preservation"

    def test_detect_retirement_goal(self):
        goals = plan_goals("想为养老做准备，30年后退休")
        retirement = [g for g in goals if g["goal_type"] == "retirement"]
        assert len(retirement) >= 1

    def test_detect_education_goal(self):
        goals = plan_goals("为孩子10年后的教育做准备")
        education = [g for g in goals if g["goal_type"] == "education_fund"]
        assert len(education) >= 1

    def test_detect_wealth_growth_goal(self):
        goals = plan_goals("想让资产增值，长期投资")
        wealth = [g for g in goals if g["goal_type"] == "wealth_growth"]
        assert len(wealth) >= 1

    def test_long_term_wealth_is_growth_priority(self):
        goals = plan_goals("想长期投资，让资产翻倍")
        wealth = [g for g in goals if g["goal_type"] == "wealth_growth"][0]
        assert wealth["priority"] == "growth"

    def test_mid_long_term_is_not_shortened_to_mid_term(self):
        goals = plan_goals("中长期财富增值")
        wealth = [g for g in goals if g["goal_type"] == "wealth_growth"][0]
        assert wealth["time_horizon_months"] == 60

class TestRiskAssessor:

    def test_conservative_profile_yields_conservative_risk(self):
        profile = {
            "risk_preference": "conservative",
            "investment_experience": "beginner",
            "liquidity_need": "high",
            "constraints": ["3年后买房"],
            "missing_fields": [],
        }
        goals = [{"goal_type": "house_purchase", "time_horizon_months": 36, "priority": "capital_preservation"}]
        result = assess_risk(profile, goals)
        assert result["risk_level"] == "conservative"

    def test_aggressive_profile_yields_aggressive_risk(self):
        profile = {
            "risk_preference": "aggressive",
            "investment_experience": "experienced",
            "liquidity_need": "low",
            "constraints": [],
            "missing_fields": [],
        }
        goals = [{"goal_type": "wealth_growth", "time_horizon_months": 120, "priority": "growth"}]
        result = assess_risk(profile, goals)
        assert result["risk_level"] == "aggressive"

    def test_conservative_excludes_high_risk_assets(self):
        profile = {
            "risk_preference": "conservative",
            "investment_experience": "beginner",
            "liquidity_need": "high",
            "constraints": [],
            "missing_fields": [],
        }
        result = assess_risk(profile, [])
        unsuitable = result["unsuitable_assets"]
        assert any("股票" in a for a in unsuitable)

    def test_no_goals_defaults_reasonable(self):
        profile = {
            "risk_preference": "stable",
            "investment_experience": "beginner",
            "liquidity_need": "medium",
            "constraints": [],
            "missing_fields": [],
        }
        result = assess_risk(profile, [])
        assert result["risk_level"] in ["conservative", "stable", "balanced", "aggressive"]

class TestAllocationEngine:

    def test_allocation_sum_is_100(self):
        for risk_level in ["conservative", "stable", "balanced", "aggressive"]:
            result = allocate_assets(risk_level)
            total = sum(item["ratio"] for item in result["allocation"])
            assert abs(total - 100.0) < 0.1, f"{risk_level}: sum = {total}"

    def test_conservative_allocation_has_liquid_assets(self):
        result = allocate_assets("conservative")
        asset_classes = [item["asset_class"] for item in result["allocation"]]
        assert any("现金" in ac for ac in asset_classes)

    def test_aggressive_has_equity_focus(self):
        result = allocate_assets("aggressive")
        equity_ratio = sum(
            item["ratio"] for item in result["allocation"]
            if "指数" in item["asset_class"] or "混合" in item["asset_class"]
        )
        assert equity_ratio > 40

    def test_allocation_includes_rationale(self):
        result = allocate_assets("balanced")
        assert len(result["rationale"]) > 0

    def test_no_specific_products_in_allocation(self):
        import re
        for risk_level in ["conservative", "stable", "balanced", "aggressive"]:
            result = allocate_assets(risk_level)
            for item in result["allocation"]:
                ac = item["asset_class"]
                assert not re.search(r"\d{6}", ac)

class TestAdvisoryCompliancePolicy:

    def test_detect_stock_recommendation(self):
        result = review_compliance("推荐买入贵州茅台股票，现在正是建仓好时机。")
        assert not result["is_compliant"]
        assert len(result["warnings"]) > 0

    def test_detect_return_promise(self):
        result = review_compliance("这个方案保本保收益，稳赚不赔。")
        assert not result["is_compliant"]

    def test_clean_answer_passes(self):
        result = review_compliance(
            "建议采用股债60/40的配置比例。投资有风险，本回答仅供参考，不构成投资决策依据。"
        )
        violations = [w for w in result["warnings"] if w.startswith("[违规]")]
        assert len(violations) == 0

    def test_missing_risk_notice_is_flagged(self):
        result = review_compliance("建议配置一些债券基金和宽基指数基金。")
        assert any("风险提示" in w for w in result["warnings"])

    def test_decision_substitution_detected(self):
        result = review_compliance("根据你的情况，你应该马上买入沪深300ETF。")
        assert not result["is_compliant"]

    def test_risk_notice_always_included(self):
        result = review_compliance("任何回答")
        assert result["risk_notice"] != ""

class TestInvestmentAdvisorAgentE2E:

    @pytest.fixture
    def agent(self) -> InvestmentAdvisorAgent:
        return InvestmentAdvisorAgent()

    def _make_sources(self, n: int = 2) -> list[Source]:
        return [
            Source(title="asset allocation basics", source_type="investment_knowledge", confidence=0.85),
            Source(title="lifecycle investment theory", source_type="investment_knowledge", confidence=0.78),
        ][:n]

    def test_returns_valid_response(self, agent):
        request = ConsultationRequest(
            question="月薪1万，风险厌恶，3年后买房，该如何配置资产？",
        )
        sources = self._make_sources()
        response = agent.answer(request, sources)
        assert response.intent == "advisory"
        assert response.agent == "investment_advisor"
        assert len(response.answer) > 100
        assert response.sources == sources

    def test_answer_contains_required_sections(self, agent):
        request = ConsultationRequest(question="月薪1万，风险厌恶，3年后买房，该如何配置资产？")
        response = agent.answer(request, self._make_sources())
        answer = response.answer
        assert "用户画像" in answer
        assert "投资目标" in answer
        assert "风险评估" in answer
        assert "资产配置" in answer

    def test_answer_has_no_stock_codes(self, agent):
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, self._make_sources())
        import re
        assert not re.search(r"\d{6}", response.answer)

    def test_conservative_user_gets_conservative_advice(self, agent):
        request = ConsultationRequest(
            question="风险厌恶，不能承受亏损，3年后买房",
            user_profile={"risk_preference": "low"},
        )
        response = agent.answer(request, self._make_sources())
        assert "保守" in response.answer or "conservative" in response.answer.lower()

    def test_no_violations_in_template_answer(self, agent):
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, self._make_sources())
        violations = [w for w in response.warnings if w.startswith("[违规]")]
        assert len(violations) == 0

    def test_evidence_section_with_source_titles(self, agent):
        """E2E: evidence section references source titles."""
        sources = [
            Source(title="资产配置基础原则", source_type="investment_knowledge", confidence=0.66),
        ]
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, sources)
        assert "参考依据与适用边界" in response.answer
        assert "资产配置基础原则" in response.answer
        assert "66%" in response.answer or "0.66" in response.answer

    def test_low_confidence_with_empty_sources(self, agent):
        """Empty sources → low confidence warning in answer."""
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, [])
        assert "置信度有限" in response.answer or "无可用的" in response.answer


class TestFundDcaPlanner:

    def test_dca_no_specific_fund_codes(self):
        from app.agents.investment_advisor_tools.fund_dca_planner import plan_dca
        import re
        result = plan_dca(
            profile={"income_level": "medium"},
            goals=[{"goal_type": "house_purchase", "time_horizon_months": 36, "priority": "capital_preservation"}],
            risk_level="conservative",
        )
        result_str = str(result)
        assert not re.search(r'\d{6}', result_str)

    def test_dca_conservative_is_low_equity(self):
        from app.agents.investment_advisor_tools.fund_dca_planner import plan_dca
        result = plan_dca(
            profile={"income_level": "medium"},
            goals=[{"goal_type": "house_purchase", "time_horizon_months": 36, "priority": "capital_preservation"}],
            risk_level="conservative",
        )
        categories = result["suitable_categories"]
        has_equity_heavy = any("偏股" in c or "行业指数" in c for c in categories)
        assert not has_equity_heavy

    def test_dca_has_all_required_fields(self):
        from app.agents.investment_advisor_tools.fund_dca_planner import plan_dca
        result = plan_dca(
            profile={"income_level": "medium"}, goals=[], risk_level="stable",
        )
        for key in ["frequency", "amount_ratio", "suitable_categories", "review_conditions", "pause_conditions"]:
            assert key in result


class TestHoldingDiagnostic:

    def test_detect_equity_overweight(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        holdings = [
            {"asset_class": "宽基指数基金类", "ratio": 70},
            {"asset_class": "现金及货币类", "ratio": 30},
        ]
        result = diagnose(holdings, risk_level="conservative")
        assert "equity_overweight" in result["risk_flags"]

    def test_detect_concentration_risk(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        holdings = [
            {"asset_class": "行业指数基金类", "ratio": 60},
            {"asset_class": "现金及货币类", "ratio": 40},
        ]
        result = diagnose(holdings, risk_level="balanced")
        assert "concentration_risk" in result["risk_flags"]

    def test_detect_liquidity_insufficient(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        holdings = [
            {"asset_class": "宽基指数基金类", "ratio": 60},
            {"asset_class": "债券类", "ratio": 40},
        ]
        result = diagnose(holdings, risk_level="stable")
        assert "liquidity_insufficient" in result["risk_flags"]

    def test_clean_portfolio_no_issues(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        holdings = [
            {"asset_class": "现金及货币类", "ratio": 10},
            {"asset_class": "短债/固收类", "ratio": 40},
            {"asset_class": "宽基指数基金类", "ratio": 30},
            {"asset_class": "债券类", "ratio": 20},
        ]
        result = diagnose(holdings, risk_level="stable")
        assert len(result["issues"]) == 0

    def test_empty_holdings_returns_empty(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        result = diagnose([], risk_level="stable")
        assert result["issues"] == []

    def test_sum_not_100_is_flagged(self):
        from app.agents.investment_advisor_tools.holding_diagnostic import diagnose
        holdings = [
            {"asset_class": "宽基指数基金类", "ratio": 50},
            {"asset_class": "现金及货币类", "ratio": 30},
        ]
        result = diagnose(holdings, risk_level="stable")
        assert any("总和" in issue for issue in result["issues"])


class TestMarketHotspotInterpreter:

    def test_detect_ai_hotspot(self):
        from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret
        result = interpret("AI板块最近很火，值得关注吗？")
        assert result["is_relevant"] is True

    def test_detect_new_energy_hotspot(self):
        from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret
        result = interpret("新能源板块怎么看？")
        assert result["is_relevant"] is True

    def test_no_hotspot_returns_irrelevant(self):
        from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret
        result = interpret("如何配置资产？")
        assert result["is_relevant"] is False

    def test_hotspot_has_no_price_prediction(self):
        from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret
        result = interpret("半导体板块未来会涨吗？")
        result_str = str(result)
        assert "目标价" not in result_str

    def test_hotspot_output_has_required_fields(self):
        from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret
        result = interpret("医药板块投资机会")
        for key in ["is_relevant", "topic", "drivers", "risk_points", "neutral_view"]:
            assert key in result


class TestInvestmentAdvisorAgentWithBranches:

    @pytest.fixture
    def agent(self) -> InvestmentAdvisorAgent:
        return InvestmentAdvisorAgent()

    def _make_sources(self) -> list[Source]:
        return [Source(title="test source", source_type="investment_knowledge", confidence=0.8)]

    def test_holdings_triggers_diagnostic_section(self, agent):
        request = ConsultationRequest(
            question="如何配置资产？",
            user_profile={
                "risk_preference": "low",
                "holdings": [
                    {"asset_class": "宽基指数基金类", "ratio": 70},
                    {"asset_class": "现金及货币类", "ratio": 30},
                ],
            },
        )
        response = agent.answer(request, self._make_sources())
        assert "权益类资产占比" in response.answer

    def test_no_holdings_shows_placeholder(self, agent):
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, self._make_sources())
        assert "未提供持仓" in response.answer

    def test_hotspot_question_triggers_interpretation(self, agent):
        request = ConsultationRequest(question="新能源板块怎么看？")
        response = agent.answer(request, self._make_sources())
        assert "新能源" in response.answer

    def test_all_new_sections_present(self, agent):
        request = ConsultationRequest(question="月薪1万，风险厌恶，3年后买房，如何配置资产？")
        response = agent.answer(request, self._make_sources())
        assert "基金定投规划" in response.answer
        assert "持仓诊断" in response.answer
        assert "市场热点解读" in response.answer

    def test_evidence_section_present(self, agent):
        """Evidence section (八) should appear with source titles."""
        request = ConsultationRequest(question="如何配置资产？")
        sources = [
            Source(title="资产配置基础原则", source_type="investment_knowledge", confidence=0.66),
            Source(title="投资建议合规边界", source_type="regulations", confidence=0.83),
        ]
        response = agent.answer(request, sources)
        assert "参考依据与适用边界" in response.answer
        assert "资产配置基础原则" in response.answer
        assert "投资建议合规边界" in response.answer

    def test_low_confidence_warning_in_answer(self, agent):
        """Low-confidence sources should trigger a warning in the answer."""
        request = ConsultationRequest(question="如何配置资产？")
        sources = [
            Source(title="弱信号来源", source_type="knowledge_base", confidence=0.30),
        ]
        response = agent.answer(request, sources)
        assert "置信度有限" in response.answer

    def test_hard_constraints_preserved_with_evidence(self, agent):
        """All 9 required sections must still be present with evidence injection."""
        from app.agents.investment_advisor import _REQUIRED_REPORT_SECTIONS
        request = ConsultationRequest(question="月薪1万，如何配置资产？")
        sources = [
            Source(title="资产配置基础原则", source_type="investment_knowledge", confidence=0.80),
            Source(title="投资建议合规边界", source_type="regulations", confidence=0.85),
        ]
        response = agent.answer(request, sources)
        for section in _REQUIRED_REPORT_SECTIONS:
            assert section in response.answer, f"Missing section: {section}"

    def test_no_stock_recommendation_with_compliance_evidence(self, agent):
        """Even with compliance sources, answer must NOT recommend stocks."""
        request = ConsultationRequest(question="能给我推荐几只股票吗？")
        sources = [
            Source(title="投资建议合规边界", source_type="regulations", confidence=0.85),
        ]
        response = agent.answer(request, sources)
        # Must NOT contain stock codes or recommendations
        assert "600" not in response.answer
        assert "000" not in response.answer
        assert "推荐买入" not in response.answer

    def test_evidence_inline_in_allocation(self, agent):
        """Advisory evidence should be referenced in allocation section."""
        request = ConsultationRequest(question="保守型如何配置资产？")
        sources = [
            Source(title="资产配置基础原则", source_type="investment_knowledge", confidence=0.80),
        ]
        response = agent.answer(request, sources)
        # Allocation section should have evidence reference
        import re
        sec4 = response.answer.split("四、资产配置建议")[1].split("五、")[0]
        assert "参考依据" in sec4 or "资产配置基础原则" in sec4

    def test_empty_sources_produces_placeholder(self, agent):
        """Empty sources should trigger low confidence and a placeholder."""
        request = ConsultationRequest(question="如何配置资产？")
        response = agent.answer(request, [])
        assert "置信度有限" in response.answer or "无可用的" in response.answer
