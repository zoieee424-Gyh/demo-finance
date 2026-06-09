"""
Tests for DeepAgent tool registry and tool contracts.

Covers:
  - Tool registration count and Chinese names
  - Hard constraints presence on every tool
  - Tool execution (wrapping existing rule-based modules)
  - Risk assessor and allocation engine output integrity
"""

import pytest

from app.agents.deepagent.tool_contracts import DeepAgentTool
from app.agents.deepagent.registry import (
    ToolRegistry,
    build_investment_advisor_registry,
)


# ── Expected tools ──────────────────────────────────────────────────

EXPECTED_TOOLS = [
    ("profile_analyzer", "用户画像解析器"),
    ("goal_planner", "投资目标规划器"),
    ("risk_assessor", "风险评估器"),
    ("allocation_engine", "资产配置引擎"),
    ("fund_dca_planner", "基金定投规划器"),
    ("holding_diagnostic", "持仓诊断器"),
    ("market_hotspot_interpreter", "市场热点解读器"),
    ("advisory_compliance_policy", "投顾合规审查器"),
    ("evidence_builder", "RAG证据构建器"),
]


# ═══════════════════════════════════════════════════════════════════
# Tool registry tests
# ═══════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Test the ToolRegistry class and investment advisor registry builder."""

    def test_registry_builder_creates_nine_tools(self):
        """build_investment_advisor_registry should register exactly 9 tools."""
        reg = build_investment_advisor_registry()
        assert reg.tool_count == 9

    @pytest.mark.parametrize("tool_id, name_cn", EXPECTED_TOOLS)
    def test_tool_has_chinese_name(self, tool_id, name_cn):
        """Every registered tool must have its expected Chinese business name."""
        reg = build_investment_advisor_registry()
        tool = reg.get(tool_id)
        assert tool is not None, f"Tool '{tool_id}' not found in registry"
        assert tool.name_cn == name_cn, (
            f"Tool '{tool_id}' name_cn mismatch: "
            f"expected '{name_cn}', got '{tool.name_cn}'"
        )

    @pytest.mark.parametrize("tool_id, _", EXPECTED_TOOLS)
    def test_tool_has_hard_constraints(self, tool_id, _):
        """Every tool must declare at least one hard constraint."""
        reg = build_investment_advisor_registry()
        tool = reg.get(tool_id)
        assert tool is not None
        assert len(tool.hard_constraints) >= 1, (
            f"Tool '{tool_id}' has no hard_constraints"
        )

    @pytest.mark.parametrize("tool_id, _", EXPECTED_TOOLS)
    def test_tool_has_execute_function(self, tool_id, _):
        """Every tool must have an execute callable bound."""
        reg = build_investment_advisor_registry()
        tool = reg.get(tool_id)
        assert tool is not None
        assert callable(tool.execute), (
            f"Tool '{tool_id}' has no callable execute function"
        )

    @pytest.mark.parametrize("tool_id, _", EXPECTED_TOOLS)
    def test_tool_does_not_require_llm(self, tool_id, _):
        """No investment advisor tool should require LLM internally."""
        reg = build_investment_advisor_registry()
        tool = reg.get(tool_id)
        assert tool is not None
        assert tool.requires_llm is False, (
            f"Tool '{tool_id}' unexpectedly requires_llm=True"
        )

    def test_registry_prevents_duplicate_tool_ids(self):
        """Registering two tools with the same tool_id must raise."""
        reg = ToolRegistry()
        tool_a = DeepAgentTool(
            tool_id="test_tool",
            name_cn="测试工具",
            description="A test tool.",
        )
        tool_b = DeepAgentTool(
            tool_id="test_tool",
            name_cn="测试工具B",
            description="Duplicate ID.",
        )
        reg.register(tool_a)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(tool_b)

    def test_registry_get_returns_none_for_missing(self):
        """get() should return None for unregistered tool IDs."""
        reg = ToolRegistry()
        assert reg.get("nonexistent") is None


# ═══════════════════════════════════════════════════════════════════
# Tool execution tests — verify tools wrap existing modules correctly
# ═══════════════════════════════════════════════════════════════════

class TestToolExecution:
    """Verify that registered tools execute correctly when called."""

    def test_profile_analyzer_tool_executes(self):
        """profile_analyzer should return valid profile dict."""
        reg = build_investment_advisor_registry()
        tool = reg.get("profile_analyzer")
        result = tool(
            question="我是保守型投资者，如何配置资产？",
            user_profile={"risk_preference": "conservative"},
        )
        assert isinstance(result, dict)
        assert "risk_preference" in result
        assert "missing_fields" in result
        # ProfileAnalyzer is rule-based and parses both question text
        # and user_profile — the result is a synthesis, not a pass-through.
        # We verify it produces a valid risk_preference value.
        assert result["risk_preference"] in (
            "conservative", "stable", "balanced", "aggressive",
        )

    def test_profile_analyzer_preserves_explicit_fields(self):
        """LLM/DeepAgent must not override explicit user profile fields.

        ProfileAnalyzer reads from both question text and user_profile;
        we verify that explicit user_profile values influence the result.
        """
        reg = build_investment_advisor_registry()
        tool = reg.get("profile_analyzer")
        result = tool(
            question="我想做长期理财",
            user_profile={
                "income_level": "low",
            },
        )
        # Explicit income_level input should be reflected
        assert result["income_level"] == "low"

    def test_goal_planner_tool_executes(self):
        """goal_planner should return list of goal dicts."""
        reg = build_investment_advisor_registry()
        tool = reg.get("goal_planner")
        result = tool(
            question="我想为退休做准备，有20年投资期限",
            profile={"risk_preference": "stable", "investment_experience": "experienced"},
        )
        assert isinstance(result, list)
        if result:
            goal = result[0]
            assert "goal_type" in goal
            assert "time_horizon_months" in goal

    def test_risk_assessor_tool_executes(self):
        """risk_assessor must return risk_level as a deterministic result."""
        reg = build_investment_advisor_registry()
        tool = reg.get("risk_assessor")
        result = tool(
            profile={
                "risk_preference": "conservative",
                "investment_experience": "beginner",
                "liquidity_need": "high",
            },
            goals=[{"goal_type": "retirement", "time_horizon_months": 240, "priority": "capital_preservation"}],
        )
        assert isinstance(result, dict)
        assert "risk_level" in result
        assert "suitable_assets" in result
        assert "unsuitable_assets" in result
        # conservative profile should get conservative risk level
        assert result["risk_level"] == "conservative"

    def test_risk_assessor_output_not_empty(self):
        """risk_assessor must return non-empty suitable_assets."""
        reg = build_investment_advisor_registry()
        tool = reg.get("risk_assessor")
        result = tool(
            profile={"risk_preference": "aggressive", "investment_experience": "experienced"},
            goals=[{"goal_type": "growth", "time_horizon_months": 120, "priority": "growth"}],
        )
        assert len(result["suitable_assets"]) > 0
        assert result["risk_level"] == "aggressive"

    def test_allocation_engine_tool_executes(self):
        """allocation_engine must output allocations summing to 100%."""
        reg = build_investment_advisor_registry()
        tool = reg.get("allocation_engine")
        result = tool(
            risk_level="conservative",
            goals=[{"goal_type": "capital_preservation", "time_horizon_months": 60, "priority": "capital_preservation"}],
        )
        assert isinstance(result, dict)
        assert "allocation" in result
        assert "rationale" in result
        allocations = result["allocation"]
        total = sum(item["ratio"] for item in allocations)
        assert abs(total - 100) < 0.5, f"Allocation sum {total} != 100"

    def test_allocation_engine_output_no_products(self):
        """allocation_engine must not output stock/fund codes."""
        reg = build_investment_advisor_registry()
        tool = reg.get("allocation_engine")
        result = tool(
            risk_level="balanced",
            goals=[{"goal_type": "wealth_growth", "time_horizon_months": 120, "priority": "balanced"}],
        )
        output_str = str(result)
        # No numeric stock codes like 600xxx or 000xxx
        import re
        assert not re.search(r"\b\d{6}\b", output_str), "Found suspicious numeric code in allocation output"

    def test_fund_dca_planner_tool_executes(self):
        """fund_dca_planner should output category-level plan."""
        reg = build_investment_advisor_registry()
        tool = reg.get("fund_dca_planner")
        result = tool(
            profile={"risk_preference": "stable", "income_level": "medium"},
            goals=[{"goal_type": "wealth_growth", "time_horizon_months": 60, "priority": "balanced"}],
            risk_level="stable",
        )
        assert isinstance(result, dict)
        assert "frequency" in result
        assert "suitable_categories" in result

    def test_holding_diagnostic_tool_executes(self):
        """holding_diagnostic should detect issues in suboptimal holdings."""
        reg = build_investment_advisor_registry()
        tool = reg.get("holding_diagnostic")
        result = tool(
            holdings=[
                {"asset_class": "宽基指数基金类", "ratio": 80},
                {"asset_class": "现金及货币类", "ratio": 20},
            ],
            risk_level="conservative",
        )
        assert isinstance(result, dict)
        assert "issues" in result
        assert "adjustment_directions" in result
        # 80% equity for conservative should flag issues
        assert len(result["issues"]) > 0

    def test_market_hotspot_interpreter_tool_executes(self):
        """market_hotspot_interpreter should return is_relevant flag."""
        reg = build_investment_advisor_registry()
        tool = reg.get("market_hotspot_interpreter")
        # Query with no hotspot keywords
        result = tool(question="如何做资产配置？")
        assert isinstance(result, dict)
        assert "is_relevant" in result
        assert result["is_relevant"] is False

    def test_market_hotspot_no_prediction(self):
        """market_hotspot_interpreter must not predict market direction."""
        reg = build_investment_advisor_registry()
        tool = reg.get("market_hotspot_interpreter")
        result = tool(question="人工智能板块最近很火，能分析一下吗？")
        output_str = str(result)
        # Must not contain bullish/bearish predictions
        forbidden = ["会涨", "会跌", "必涨", "必跌", "买入时机", "卖出时机"]
        for term in forbidden:
            assert term not in output_str, f"Found prediction term '{term}' in hotspot output"

    def test_advisory_compliance_tool_executes(self):
        """advisory_compliance_policy should flag stock recommendations."""
        reg = build_investment_advisor_registry()
        tool = reg.get("advisory_compliance_policy")
        clean_answer = "建议配置50%债券类、30%宽基指数类、20%现金类。投资有风险，入市需谨慎。"
        result = tool(answer=clean_answer)
        assert isinstance(result, dict)
        assert "warnings" in result
        assert "is_compliant" in result

    def test_advisory_compliance_detects_stock_tip(self):
        """advisory_compliance_policy must detect stock recommendations."""
        reg = build_investment_advisor_registry()
        tool = reg.get("advisory_compliance_policy")
        bad_answer = "建议买入贵州茅台（600519）和宁德时代（300750）。"
        result = tool(answer=bad_answer)
        assert len(result["warnings"]) > 0, "Should have detected stock recommendation"

    def test_evidence_builder_tool_executes(self):
        """evidence_builder should produce EvidencePack from sources."""
        from app.schemas.consultation import Source

        reg = build_investment_advisor_registry()
        tool = reg.get("evidence_builder")
        sources = [
            Source(
                title="风险等级与资产类别匹配",
                source_type="investment_knowledge",  # maps to "advisory"
                content_preview="保守型投资者应...",
                confidence=0.85,
            ),
            Source(
                title="投资者适当性管理规定概要",
                source_type="compliance_knowledge",  # prefix "compliance_" → "compliance"
                content_preview="金融机构应当...",
                confidence=0.72,
            ),
        ]
        result = tool(sources=sources)
        assert result is not None
        assert hasattr(result, "items")
        assert result.has_advisory is True
        assert result.has_compliance is True

    def test_evidence_builder_no_fabrication(self):
        """evidence_builder must not fabricate document content beyond titles."""
        from app.schemas.consultation import Source

        reg = build_investment_advisor_registry()
        tool = reg.get("evidence_builder")
        sources = [
            Source(
                title="风险等级匹配规则",
                source_type="advisory_knowledge",
                content_preview="保守型投资者适合低风险资产。",
                confidence=0.90,
            ),
        ]
        result = tool(sources=sources)
        # Should have items based on source titles, not fabricated content
        assert len(result.items) == 1
        assert result.items[0].title == "风险等级匹配规则"
        assert result.items[0].evidence_type == "advisory"

    def test_evidence_builder_accepts_dict_sources(self):
        """evidence_builder should normalize dict sources from tool calls."""
        reg = build_investment_advisor_registry()
        tool = reg.get("evidence_builder")
        result = tool(sources=[
            {
                "title": "资产配置基础原则",
                "source_type": "advisory_knowledge",
                "confidence": 0.82,
                "url": None,
            }
        ])
        assert len(result.items) == 1
        assert result.items[0].title == "资产配置基础原则"
        assert result.has_advisory is True


# ═══════════════════════════════════════════════════════════════════
# DeepAgentTool contract tests
# ═══════════════════════════════════════════════════════════════════

class TestDeepAgentToolContract:
    """Test the DeepAgentTool dataclass itself."""

    def test_tool_call_without_execute_raises(self):
        """Calling a tool with no execute function must raise NotImplementedError."""
        tool = DeepAgentTool(
            tool_id="empty",
            name_cn="空工具",
            description="No execute.",
        )
        with pytest.raises(NotImplementedError):
            tool()

    def test_tool_with_execute_works(self):
        """Calling a tool with execute should invoke it."""
        tool = DeepAgentTool(
            tool_id="echo",
            name_cn="回声工具",
            description="Echo back input.",
            execute=lambda **kw: kw,
        )
        result = tool(foo="bar")
        assert result == {"foo": "bar"}

    def test_system_prompt_fragment_includes_constraints(self):
        """get_system_prompt_fragment must mention hard constraints."""
        tool = DeepAgentTool(
            tool_id="test",
            name_cn="测试",
            description="Test tool.",
            hard_constraints=["禁止修改输出", "必须保留警告"],
        )
        fragment = tool.get_system_prompt_fragment()
        assert "硬约束" in fragment
        assert "禁止修改输出" in fragment
        assert "必须保留警告" in fragment
