"""
Tool registry for DeepAgent-powered financial agents.

Registers deterministic tools for the investment advisor (and later other agents)
so the DeepAgent orchestrator can discover and invoke them.

Pattern:
  - Each agent type gets its own tool list.
  - Tools wrap existing rule-based modules — no business logic duplication.
  - Every tool has a Chinese name (name_cn) for bilingual reporting.
"""

from __future__ import annotations

from app.agents.deepagent.tool_contracts import DeepAgentTool


# ═══════════════════════════════════════════════════════════════════
# Tool factory functions — wrap existing rule-based modules
# ═══════════════════════════════════════════════════════════════════

def _make_profile_analyzer_tool() -> DeepAgentTool:
    """用户画像解析器 — wraps ProfileAnalyzer.analyze()."""
    from app.agents.investment_advisor_tools.profile_analyzer import analyze

    def _execute(*, question: str, user_profile: dict | None = None, **kwargs):
        return analyze(question=question, user_profile=user_profile)

    return DeepAgentTool(
        tool_id="profile_analyzer",
        name_cn="用户画像解析器",
        description=(
            "分析用户画像，从用户问题和填写的画像信息中提取："
            "风险偏好（risk_preference）、收入水平（income_level）、"
            "流动性需求（liquidity_need）、投资经验（investment_experience）、"
            "特殊约束（constraints）、缺失字段（missing_fields）。"
            "此工具为纯规则引擎，不调用 LLM。"
        ),
        hard_constraints=[
            "禁止 LLM 覆盖用户显式填写的画像字段",
            "禁止 LLM 自行推断 risk_preference 替代工具输出",
            "缺失字段应如实报告 missing_fields，不得编造补全",
        ],
        execute=_execute,
        input_schema={
            "question": "用户咨询问题原文",
            "user_profile": "用户画像 dict，可选字段：risk_preference, income_level, liquidity_need, investment_experience, constraints, holdings",
        },
        output_schema={
            "income_level": "收入水平 (low/medium/high/unknown)",
            "risk_preference": "风险偏好 (conservative/stable/balanced/aggressive/unknown)",
            "liquidity_need": "流动性需求 (low/medium/high/unknown)",
            "investment_experience": "投资经验 (beginner/experienced/unknown)",
            "constraints": "特殊约束列表",
            "missing_fields": "缺失字段列表",
        },
        requires_llm=False,
    )


def _make_goal_planner_tool() -> DeepAgentTool:
    """投资目标规划器 — wraps GoalPlanner.plan()."""
    from app.agents.investment_advisor_tools.goal_planner import plan

    def _execute(*, question: str, profile: dict, **kwargs):
        return plan(question=question, profile=profile)

    return DeepAgentTool(
        tool_id="goal_planner",
        name_cn="投资目标规划器",
        description=(
            "从用户问题和画像中识别、排序投资目标。"
            "每个目标包含 goal_type、description、time_horizon_months、priority。"
            "priority 取 capital_preservation/balanced/growth 之一。"
        ),
        hard_constraints=[
            "目标期限和优先级必须来自工具输出，LLM 不得自行修改",
            "不得编造用户未提出的投资目标",
        ],
        execute=_execute,
        input_schema={
            "question": "用户咨询问题原文",
            "profile": "ProfileAnalyzer 输出的画像 dict",
        },
        output_schema={
            "goals": "投资目标列表，每项含 goal_type, description, time_horizon_months, priority",
        },
        requires_llm=False,
    )


def _make_risk_assessor_tool() -> DeepAgentTool:
    """风险评估器 — wraps RiskAssessor.assess()."""
    from app.agents.investment_advisor_tools.risk_assessor import assess

    def _execute(*, profile: dict, goals: list[dict], **kwargs):
        return assess(profile=profile, goals=goals)

    return DeepAgentTool(
        tool_id="risk_assessor",
        name_cn="风险评估器",
        description=(
            "基于用户画像和投资目标评估风险承受能力，输出："
            "风险等级（conservative/stable/balanced/aggressive）、"
            "最大回撤容忍度、适合/不适合的资产类别。"
        ),
        hard_constraints=[
            "风险等级必须来自此工具的输出，LLM 严禁自行决定风险等级",
            "适合/不适合资产类别必须来自工具输出，LLM 不得自行增删",
            "保守型用户不得被建议高风险资产",
        ],
        execute=_execute,
        input_schema={
            "profile": "ProfileAnalyzer 输出",
            "goals": "GoalPlanner 输出的目标列表",
        },
        output_schema={
            "risk_level": "风险等级",
            "max_drawdown_tolerance": "最大回撤容忍度",
            "suitable_assets": "适合资产类别列表",
            "unsuitable_assets": "不适合资产类别列表",
        },
        requires_llm=False,
    )


def _make_allocation_engine_tool() -> DeepAgentTool:
    """资产配置引擎 — wraps AllocationEngine.allocate()."""
    from app.agents.investment_advisor_tools.allocation_engine import allocate

    def _execute(*, risk_level: str, goals: list[dict], **kwargs):
        return allocate(risk_level=risk_level, goals=goals)

    return DeepAgentTool(
        tool_id="allocation_engine",
        name_cn="资产配置引擎",
        description=(
            "根据风险等级和投资目标生成资产配置建议，输出各类资产的比例（合计 100%）和配置逻辑说明。"
            "只输出资产类别（如宽基指数类、债券类、现金类），不输出具体产品。"
        ),
        hard_constraints=[
            "资产配置比例必须来自此工具的输出，LLM 严禁自行生成或修改比例",
            "输出只含资产类别（asset_class），不得包含具体股票代码、基金代码、基金名称或产品平台",
            "所有比例合计必须为 100%",
        ],
        execute=_execute,
        input_schema={
            "risk_level": "RiskAssessor 输出的风险等级",
            "goals": "GoalPlanner 输出的目标列表",
        },
        output_schema={
            "allocation": "资产配置列表，每项含 asset_class, ratio",
            "rationale": "配置逻辑说明",
        },
        requires_llm=False,
    )


def _make_fund_dca_planner_tool() -> DeepAgentTool:
    """基金定投规划器 — wraps FundDcaPlanner.plan_dca()."""
    from app.agents.investment_advisor_tools.fund_dca_planner import plan_dca

    def _execute(*, profile: dict, goals: list[dict], risk_level: str, **kwargs):
        return plan_dca(profile=profile, goals=goals, risk_level=risk_level)

    return DeepAgentTool(
        tool_id="fund_dca_planner",
        name_cn="基金定投规划器",
        description=(
            "根据用户画像和风险等级生成基金定投方案，包括定投频率、金额比例、"
            "适合类别、复盘条件和暂停条件。只推荐基金类别，不推荐具体基金。"
        ),
        hard_constraints=[
            "只输出基金类别，不输出具体基金名称、代码或购买平台",
            "定投参数（频率、金额比例）必须来自工具输出",
        ],
        execute=_execute,
        input_schema={
            "profile": "ProfileAnalyzer 输出",
            "goals": "GoalPlanner 输出",
            "risk_level": "RiskAssessor 输出",
        },
        output_schema={
            "frequency": "定投频率",
            "amount_ratio": "建议金额比例",
            "suitable_categories": "适合基金类别",
            "review_conditions": "复盘条件",
            "pause_conditions": "暂停条件",
        },
        requires_llm=False,
    )


def _make_holding_diagnostic_tool() -> DeepAgentTool:
    """持仓诊断器 — wraps HoldingDiagnostic.diagnose()."""
    from app.agents.investment_advisor_tools.holding_diagnostic import diagnose

    def _execute(*, holdings: list[dict], risk_level: str, **kwargs):
        return diagnose(holdings=holdings, risk_level=risk_level)

    return DeepAgentTool(
        tool_id="holding_diagnostic",
        name_cn="持仓诊断器",
        description=(
            "对用户当前持仓进行诊断：检查比例总和、权益占比超限、"
            "单一类别集中度（>50%）、流动性不足（<5%）。"
            "输出 issues 列表、adjustment_directions 和 risk_flags。"
            "仅在 user_profile.holdings 存在时调用。"
        ),
        hard_constraints=[
            "调整方向（adjustment_directions）必须使用建议语气（'可考虑''复核'），禁止使用交易指令语气",
            "诊断结论必须来自此工具，LLM 不得自行判断持仓好坏",
        ],
        execute=_execute,
        input_schema={
            "holdings": "持仓列表 [{'asset_class': str, 'ratio': float}]",
            "risk_level": "风险等级",
        },
        output_schema={
            "issues": "发现的问题列表",
            "adjustment_directions": "调整方向建议",
            "risk_flags": "风险标记",
        },
        requires_llm=False,
    )


def _make_market_hotspot_interpreter_tool() -> DeepAgentTool:
    """市场热点解读器 — wraps MarketHotspotInterpreter.interpret()."""
    from app.agents.investment_advisor_tools.market_hotspot_interpreter import (
        interpret,
    )

    def _execute(*, question: str, **kwargs):
        return interpret(question=question)

    return DeepAgentTool(
        tool_id="market_hotspot_interpreter",
        name_cn="市场热点解读器",
        description=(
            "检测用户问题中的市场热点主题关键词（AI、新能源、半导体、医药等 8 类），"
            "如检测到则输出驱动因素、风险点和中性观察建议。"
            "如未检测到热点，is_relevant=false，输出占位说明。"
        ),
        hard_constraints=[
            "不预测涨跌",
            "不推荐个股",
            "不鼓励追热点",
            "解读必须是中性观察，不得使用'布局''关注某类指数基金'等偏操作化表达",
        ],
        execute=_execute,
        input_schema={"question": "用户问题原文"},
        output_schema={
            "is_relevant": "是否涉及热点",
            "topic": "热点主题",
            "drivers": "驱动因素",
            "risk_points": "风险点",
            "neutral_view": "中性观察建议",
        },
        requires_llm=False,
    )


def _make_advisory_compliance_tool() -> DeepAgentTool:
    """投顾合规审查器 — wraps AdvisoryCompliancePolicy.review()."""
    from app.agents.investment_advisor_tools.advisory_compliance_policy import (
        review,
    )

    def _execute(*, answer: str, **kwargs):
        return review(answer)

    return DeepAgentTool(
        tool_id="advisory_compliance_policy",
        name_cn="投顾合规审查器",
        description=(
            "对生成的投顾回答进行合规审查，识别：荐股、预测涨跌、承诺收益、替代决策等违规内容。"
            "返回 warnings 列表、risk_notice 和 is_compliant 标志。"
            "必须在最终输出前调用，违规警告不能被忽略。"
        ),
        hard_constraints=[
            "违规警告（warnings）必须全部保留并返回给用户",
            "不可被 DeepAgent 或 LLM 忽略或过滤",
            "存在严重违规（荐股/预测/承诺/替代决策）时必须降级或拒答",
        ],
        execute=_execute,
        input_schema={"answer": "待审查的投顾回答全文"},
        output_schema={
            "warnings": "违规警告列表",
            "risk_notice": "风险提示文本",
            "is_compliant": "是否合规",
        },
        requires_llm=False,
    )


def _make_evidence_builder_tool() -> DeepAgentTool:
    """RAG证据构建器 — wraps build_evidence_pack()."""
    from app.rag.evidence import build_evidence_pack
    from app.schemas.consultation import Source

    def _execute(*, sources: list, **kwargs):
        normalized_sources = [
            src if isinstance(src, Source) else Source.model_validate(src)
            for src in sources
        ]
        return build_evidence_pack(normalized_sources)

    return DeepAgentTool(
        tool_id="evidence_builder",
        name_cn="RAG证据构建器",
        description=(
            "将 RAG 检索到的 Source 列表转换为结构化 EvidencePack，"
            "用于在报告中注入依据来源。"
            "evidence_type 按 source_type 映射（advisory/risk/compliance/education）。"
            "low_confidence 阈值 <0.45。"
        ),
        hard_constraints=[
            "只基于 Source.title 和 Source.source_type 做摘要，不得编造文档内容",
            "confidence 缺失时按低置信处理，不得假设高置信",
            "不得添加 sources 中不存在的文档引用",
        ],
        execute=_execute,
        input_schema={"sources": "KnowledgeRetriever 返回的 Source 列表"},
        output_schema={
            "items": "EvidenceItem 列表",
            "has_advisory": "是否有投顾证据",
            "has_risk": "是否有风控证据",
            "has_compliance": "是否有合规证据",
            "has_education": "是否有科普证据",
            "low_confidence": "是否低置信",
            "top_titles": "Top-3 证据标题",
        },
        requires_llm=False,
    )


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry of DeepAgentTool instances for a financial agent.

    Each agent type (advisory, financial_report, risk_control, compliance,
    education) will have its own tool list built from this registry pattern.
    """

    def __init__(self) -> None:
        self._tools: dict[str, DeepAgentTool] = {}

    def register(self, tool: DeepAgentTool) -> None:
        """Register a tool. Raises if tool_id already exists."""
        if tool.tool_id in self._tools:
            raise ValueError(
                f"Tool '{tool.tool_id}' is already registered."
            )
        self._tools[tool.tool_id] = tool

    def get(self, tool_id: str) -> DeepAgentTool | None:
        """Get a tool by ID."""
        return self._tools.get(tool_id)

    def list_tools(self) -> list[DeepAgentTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def build_system_prompt_fragment(self) -> str:
        """Generate the tools section of a DeepAgent system prompt."""
        if not self._tools:
            return ""
        lines = ["## 可用工具", ""]
        for tool in self._tools.values():
            lines.append(tool.get_system_prompt_fragment())
            lines.append("")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Pre-built investment advisor tool registry
# ═══════════════════════════════════════════════════════════════════

def build_investment_advisor_registry() -> ToolRegistry:
    """Build the standard tool registry for InvestmentAdvisor DeepAgent.

    Registered tools (all have Chinese business names):
      1. profile_analyzer       — 用户画像解析器
      2. goal_planner           — 投资目标规划器
      3. risk_assessor          — 风险评估器
      4. allocation_engine      — 资产配置引擎
      5. fund_dca_planner       — 基金定投规划器
      6. holding_diagnostic     — 持仓诊断器
      7. market_hotspot_interpreter — 市场热点解读器
      8. advisory_compliance_policy  — 投顾合规审查器
      9. evidence_builder       — RAG证据构建器
    """
    registry = ToolRegistry()
    registry.register(_make_profile_analyzer_tool())
    registry.register(_make_goal_planner_tool())
    registry.register(_make_risk_assessor_tool())
    registry.register(_make_allocation_engine_tool())
    registry.register(_make_fund_dca_planner_tool())
    registry.register(_make_holding_diagnostic_tool())
    registry.register(_make_market_hotspot_interpreter_tool())
    registry.register(_make_advisory_compliance_tool())
    registry.register(_make_evidence_builder_tool())
    return registry


# ═══════════════════════════════════════════════════════════════════
# 7 Financial Report tools
# ═══════════════════════════════════════════════════════════════════

def _make_financial_text_parser_tool() -> DeepAgentTool:
    """财报文本解析器 — wraps financial_text_parser.parse()."""
    from app.agents.financial_report_tools.financial_text_parser import parse

    def _execute(*, question: str, financial_text: str | None = None, **kwargs):
        return parse(question=question, financial_text=financial_text)

    return DeepAgentTool(
        tool_id="financial_text_parser",
        name_cn="财报文本解析器",
        description="解析用户提供的财报文本/摘要，提取公司名称、报告期间、报告类型。",
        hard_constraints=["文本不足时明确 missing_fields，不得编造"],
        execute=_execute,
        input_schema={"question": "用户问题原文", "financial_text": "财报文本/摘要"},
        output_schema={"company_name": "公司名称", "report_period": "报告期间", "report_type": "报告类型", "missing_fields": "缺失字段列表"},
        requires_llm=False,
    )


def _make_financial_metric_extractor_tool() -> DeepAgentTool:
    """财务指标提取器 — wraps financial_metric_extractor.extract_metrics()."""
    from app.agents.financial_report_tools.financial_metric_extractor import (
        extract_metrics,
    )

    def _execute(*, financial_text: str = "", **kwargs):
        return extract_metrics(financial_text=financial_text)

    return DeepAgentTool(
        tool_id="financial_metric_extractor",
        name_cn="财务指标提取器",
        description="从财报文本中用正则提取营收、净利润、毛利率、净利率、资产负债率、经营现金流、应收账款、存货、商誉等关键指标。",
        hard_constraints=["提取不到则报告 unknown/null，不得编造数据"],
        execute=_execute,
        input_schema={"financial_text": "财报文本/摘要"},
        output_schema={"revenue": "营业收入", "net_profit": "净利润", "debt_ratio": "资产负债率", "operating_cash_flow": "经营现金流", "...": "其他指标"},
        requires_llm=False,
    )


def _make_profitability_analyzer_tool() -> DeepAgentTool:
    """盈利能力分析器 — wraps profitability_analyzer.analyze_profitability()."""
    from app.agents.financial_report_tools.profitability_analyzer import (
        analyze_profitability,
    )

    def _execute(*, metrics: dict, **kwargs):
        return analyze_profitability(metrics=metrics)

    return DeepAgentTool(
        tool_id="profitability_analyzer",
        name_cn="盈利能力分析器",
        description="基于财务指标分析盈利能力：净利率、毛利率、利润增速等，输出强弱级别和关注要点。",
        hard_constraints=["只根据 metrics 判断，不做投资建议"],
        execute=_execute,
        input_schema={"metrics": "MetricExtractor 输出"},
        output_schema={"profitability_level": "盈利能力级别", "key_findings": "关键发现", "concerns": "关注要点"},
        requires_llm=False,
    )


def _make_solvency_liquidity_analyzer_tool() -> DeepAgentTool:
    """偿债与流动性分析器 — wraps solvency_liquidity_analyzer.analyze_solvency()."""
    from app.agents.financial_report_tools.solvency_liquidity_analyzer import (
        analyze_solvency,
    )

    def _execute(*, metrics: dict, **kwargs):
        return analyze_solvency(metrics=metrics)

    return DeepAgentTool(
        tool_id="solvency_liquidity_analyzer",
        name_cn="偿债与流动性分析器",
        description="基于资产负债率、经营现金流等分析偿债能力与流动性。",
        hard_constraints=["只根据 metrics 判断，不做投资建议"],
        execute=_execute,
        input_schema={"metrics": "MetricExtractor 输出"},
        output_schema={"solvency_level": "偿债能力级别", "liquidity_findings": "流动性发现", "debt_risks": "债务风险"},
        requires_llm=False,
    )


def _make_growth_efficiency_analyzer_tool() -> DeepAgentTool:
    """成长性与经营效率分析器 — wraps growth_efficiency_analyzer.analyze_growth()."""
    from app.agents.financial_report_tools.growth_efficiency_analyzer import (
        analyze_growth,
    )

    def _execute(*, metrics: dict, **kwargs):
        return analyze_growth(metrics=metrics)

    return DeepAgentTool(
        tool_id="growth_efficiency_analyzer",
        name_cn="成长性与经营效率分析器",
        description="基于营收/利润增速、应收/存货周转等分析成长性与经营效率。",
        hard_constraints=["只根据 metrics 判断，不做投资建议"],
        execute=_execute,
        input_schema={"metrics": "MetricExtractor 输出"},
        output_schema={"growth_level": "成长性级别", "growth_findings": "成长性发现", "efficiency_concerns": "效率关注"},
        requires_llm=False,
    )


def _make_anomaly_risk_detector_tool() -> DeepAgentTool:
    """异常风险识别器 — wraps anomaly_risk_detector.detect_risks()."""
    from app.agents.financial_report_tools.anomaly_risk_detector import (
        detect_risks,
    )

    def _execute(*, metrics: dict, financial_text: str = "", **kwargs):
        return detect_risks(metrics=metrics, financial_text=financial_text)

    return DeepAgentTool(
        tool_id="anomaly_risk_detector",
        name_cn="异常风险识别器",
        description="识别财务异常与风险信号：利润与现金流背离、应收高企、存货积压、商誉减值、高负债等。",
        hard_constraints=["输出必须是风险提示语气，不得做投资建议或评级"],
        execute=_execute,
        input_schema={"metrics": "MetricExtractor 输出", "financial_text": "财报原文"},
        output_schema={"risk_flags": "风险标记列表", "risk_count": "风险数量"},
        requires_llm=False,
    )


def _make_financial_report_compliance_tool() -> DeepAgentTool:
    """财报分析合规审查器 — wraps financial_report_compliance_policy.review()."""
    from app.agents.financial_report_tools.financial_report_compliance_policy import (
        review,
    )

    def _execute(*, answer: str, **kwargs):
        return review(answer=answer)

    return DeepAgentTool(
        tool_id="financial_report_compliance",
        name_cn="财报分析合规审查器",
        description="审查财报分析报告是否包含个股推荐、买入/卖出/持有评级、目标价、涨跌预测、收益承诺等违规内容。",
        hard_constraints=["违规警告必须全部保留并返回给用户", "不可被 LLM 忽略或过滤"],
        execute=_execute,
        input_schema={"answer": "待审查的财报分析报告全文"},
        output_schema={"warnings": "违规警告列表", "risk_notice": "风险提示文本", "is_compliant": "是否合规"},
        requires_llm=False,
    )


def build_financial_report_registry() -> ToolRegistry:
    """Build the tool registry for FinancialReportDeepAgent.

    Registered tools (7 total):
      1. financial_text_parser         — 财报文本解析器
      2. financial_metric_extractor    — 财务指标提取器
      3. profitability_analyzer        — 盈利能力分析器
      4. solvency_liquidity_analyzer   — 偿债与流动性分析器
      5. growth_efficiency_analyzer    — 成长性与经营效率分析器
      6. anomaly_risk_detector         — 异常风险识别器
      7. financial_report_compliance   — 财报分析合规审查器
    """
    registry = ToolRegistry()
    registry.register(_make_financial_text_parser_tool())
    registry.register(_make_financial_metric_extractor_tool())
    registry.register(_make_profitability_analyzer_tool())
    registry.register(_make_solvency_liquidity_analyzer_tool())
    registry.register(_make_growth_efficiency_analyzer_tool())
    registry.register(_make_anomaly_risk_detector_tool())
    registry.register(_make_financial_report_compliance_tool())
    return registry


# ═══════════════════════════════════════════════════════════════════
# 7 Risk Control tools
# ═══════════════════════════════════════════════════════════════════

def _make_risk_input_parser_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.risk_input_parser import parse

    def _execute(*, question: str, user_profile: dict | None = None, **kwargs):
        return parse(question=question, user_profile=user_profile)

    return DeepAgentTool(
        tool_id="risk_input_parser",
        name_cn="风险输入解析器",
        description="解析用户持仓、风险偏好、财报摘要、流动性需求等风险审查输入。",
        hard_constraints=["数据不足时明确 missing_fields，不得编造"],
        execute=_execute,
        input_schema={"question": "用户问题", "user_profile": "用户画像"},
        output_schema={"review_scope": "审查范围", "parsed_holdings": "持仓", "missing_fields": "缺失字段"},
        requires_llm=False,
    )


def _make_concentration_risk_checker_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.concentration_risk_checker import check

    def _execute(*, holdings: list | None = None, **kwargs):
        return check(holdings=holdings)

    return DeepAgentTool(
        tool_id="concentration_risk_checker",
        name_cn="组合集中度检测器",
        description="检测持仓资产集中度：单一类别>50%、权益类过高、现金不足等。",
        hard_constraints=["只根据 holdings 数据判断，不得假设未提供的信息"],
        execute=_execute,
        input_schema={"holdings": "持仓列表"},
        output_schema={"concentration_level": "集中度风险级别", "issues": "问题列表"},
        requires_llm=False,
    )


def _make_risk_preference_matcher_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.risk_preference_matcher import match

    def _execute(*, holdings: list | None = None, risk_preference: str = "", **kwargs):
        return match(holdings=holdings, risk_preference=risk_preference)

    return DeepAgentTool(
        tool_id="risk_preference_matcher",
        name_cn="风险偏好匹配器",
        description="判断组合风险是否超过用户风险偏好上限。",
        hard_constraints=["匹配结果必须基于风险偏好乐队和持仓数据"],
        execute=_execute,
        input_schema={"holdings": "持仓", "risk_preference": "风险偏好"},
        output_schema={"match_status": "匹配状态", "mismatch_reasons": "不匹配原因"},
        requires_llm=False,
    )


def _make_liquidity_risk_assessor_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.liquidity_risk_assessor import assess

    def _execute(*, holdings: list | None = None, liquidity_need: str = "", financial_text: str = "", **kwargs):
        return assess(holdings=holdings, liquidity_need=liquidity_need, financial_text=financial_text)

    return DeepAgentTool(
        tool_id="liquidity_risk_assessor",
        name_cn="流动性风险评估器",
        description="评估组合流动性：现金占比、流动性需求匹配、经营现金流压力。",
        hard_constraints=["仅根据持仓和财务文本判断，不做预测"],
        execute=_execute,
        input_schema={"holdings": "持仓", "liquidity_need": "流动性需求", "financial_text": "财报文本"},
        output_schema={"liquidity_level": "流动性风险级别", "cashflow_pressure_flags": "现金流压力标记"},
        requires_llm=False,
    )


def _make_financial_quality_risk_detector_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.financial_quality_risk_detector import detect

    def _execute(*, financial_text: str = "", **kwargs):
        return detect(financial_text=financial_text)

    return DeepAgentTool(
        tool_id="financial_quality_risk_detector",
        name_cn="财务质量风险识别器",
        description="识别利润现金流背离、高负债、应收存货高企、商誉减值等财务质量风险。",
        hard_constraints=["仅根据 financial_text 识别，不得编造未提及的风险"],
        execute=_execute,
        input_schema={"financial_text": "财报文本"},
        output_schema={"financial_quality_level": "财务质量风险级别", "risk_flags": "风险标记"},
        requires_llm=False,
    )


def _make_risk_mitigation_planner_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.risk_mitigation_planner import plan

    def _execute(*, concentration_result=None, preference_result=None,
                 liquidity_result=None, financial_quality_result=None, **kwargs):
        return plan(concentration_result=concentration_result, preference_result=preference_result,
                    liquidity_result=liquidity_result, financial_quality_result=financial_quality_result)

    return DeepAgentTool(
        tool_id="risk_mitigation_planner",
        name_cn="风险缓释建议生成器",
        description="生成风险管理建议和监测指标。只能使用风险管理语气，禁止交易指令。",
        hard_constraints=["禁止输出买入/卖出/加仓/减仓/满仓/空仓等交易指令", "只能使用风险管理语气"],
        execute=_execute,
        input_schema={"concentration_result": "集中度结果", "preference_result": "偏好匹配结果", "liquidity_result": "流动性结果"},
        output_schema={"mitigation_actions": "缓释建议", "monitoring_indicators": "监测指标"},
        requires_llm=False,
    )


def _make_risk_control_compliance_tool() -> DeepAgentTool:
    from app.agents.risk_control_tools.risk_control_compliance_policy import review

    def _execute(*, answer: str, **kwargs):
        return review(answer=answer)

    return DeepAgentTool(
        tool_id="risk_control_compliance",
        name_cn="风控合规审查器",
        description="审查风控报告是否包含个股推荐、交易指令、目标价、涨跌预测、收益承诺。",
        hard_constraints=["违规警告必须全部保留", "不可被 LLM 忽略或过滤"],
        execute=_execute,
        input_schema={"answer": "待审查的风控报告全文"},
        output_schema={"warnings": "违规警告", "risk_notice": "风险提示", "is_compliant": "是否合规"},
        requires_llm=False,
    )


def build_risk_control_registry() -> ToolRegistry:
    """Build the tool registry for RiskControlDeepAgent (7 tools)."""
    registry = ToolRegistry()
    registry.register(_make_risk_input_parser_tool())
    registry.register(_make_concentration_risk_checker_tool())
    registry.register(_make_risk_preference_matcher_tool())
    registry.register(_make_liquidity_risk_assessor_tool())
    registry.register(_make_financial_quality_risk_detector_tool())
    registry.register(_make_risk_mitigation_planner_tool())
    registry.register(_make_risk_control_compliance_tool())
    return registry


# ═══════════════════════════════════════════════════════════════════
# 7 Compliance tools
# ═══════════════════════════════════════════════════════════════════

def _make_compliance_input_parser_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.compliance_input_parser import parse
    def _execute(*, question: str, user_profile: dict | None = None, **kwargs):
        return parse(question=question, user_profile=user_profile)
    return DeepAgentTool(tool_id="compliance_input_parser", name_cn="合规输入解析器",
        description="解析待审查内容、业务场景、目标受众、业务类型。",
        hard_constraints=["数据不足时明确 missing_fields，不编造场景"],
        execute=_execute, input_schema={"question": "用户问题", "user_profile": "用户画像"},
        output_schema={"review_content": "待审内容", "scenario": "场景", "audience": "受众", "missing_fields": "缺失字段"}, requires_llm=False)


def _make_prohibited_expression_detector_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.prohibited_expression_detector import detect
    def _execute(*, review_content: str = "", **kwargs):
        return detect(review_content=review_content)
    return DeepAgentTool(tool_id="prohibited_expression_detector", name_cn="违规话术检测器",
        description="检测荐股、目标价、收益承诺、交易指令、夸大宣传等违规话术。",
        hard_constraints=["只检测文本中存在的模式，不做主观判断"],
        execute=_execute, input_schema={"review_content": "待审文本"},
        output_schema={"prohibited_flags": "违规标记", "overall_severity": "整体严重程度"}, requires_llm=False)


def _make_suitability_risk_checker_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.suitability_risk_checker import check
    def _execute(*, review_content: str = "", audience: str = "", scenario: str = "", **kwargs):
        return check(review_content=review_content, audience=audience, scenario=scenario)
    return DeepAgentTool(tool_id="suitability_risk_checker", name_cn="适当性风险检测器",
        description="检查内容是否与目标受众的风险承受能力匹配。",
        hard_constraints=["只根据 audience 和 content 判断，不假设适用法规"],
        execute=_execute, input_schema={"review_content": "待审文本", "audience": "目标受众", "scenario": "场景"},
        output_schema={"suitability_level": "适当性级别", "suitability_findings": "发现"}, requires_llm=False)


def _make_disclosure_completeness_checker_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.disclosure_completeness_checker import check
    def _execute(*, review_content: str = "", **kwargs):
        return check(review_content=review_content)
    return DeepAgentTool(tool_id="disclosure_completeness_checker", name_cn="信息披露完整性检查器",
        description="检查是否缺少风险提示、数据来源、适用边界、不构成投资建议声明等。",
        hard_constraints=["只检查缺失项，不做合规结论"],
        execute=_execute, input_schema={"review_content": "待审文本"},
        output_schema={"disclosure_level": "披露完整性级别", "missing_disclosures": "缺失披露项"}, requires_llm=False)


def _make_regulatory_basis_matcher_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.regulatory_basis_matcher import match
    def _execute(*, prohibited_flags=None, suitability_findings=None, missing_disclosures=None, **kwargs):
        return match(prohibited_flags=prohibited_flags, suitability_findings=suitability_findings, missing_disclosures=missing_disclosures)
    return DeepAgentTool(tool_id="regulatory_basis_matcher", name_cn="监管依据匹配器",
        description="根据违规标记匹配适用的监管原则和依据说明。不输出正式法律结论。",
        hard_constraints=["不输出正式法律条文结论，只用'监管原则/合规要求'表达"],
        execute=_execute, input_schema={"prohibited_flags": "违规标记", "suitability_findings": "适当性发现", "missing_disclosures": "缺失披露"},
        output_schema={"regulatory_principles": "监管原则", "basis_notes": "依据说明"}, requires_llm=False)


def _make_compliance_rewrite_planner_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.compliance_rewrite_planner import plan
    def _execute(*, flags_result=None, suitability_result=None, disclosure_result=None, basis_result=None, **kwargs):
        return plan(flags_result=flags_result, suitability_result=suitability_result, disclosure_result=disclosure_result, basis_result=basis_result)
    return DeepAgentTool(tool_id="compliance_rewrite_planner", name_cn="合规整改建议生成器",
        description="生成整改建议和替代表述示例。不输出正式法律意见。",
        hard_constraints=["不输出交易指令（买入/卖出/加仓/减仓等）", "不直接给正式法律意见"],
        execute=_execute, input_schema={"flags_result": "违规检测结果", "suitability_result": "适当性结果", "disclosure_result": "披露结果"},
        output_schema={"remediation_actions": "整改建议", "alternative_phrasings": "替代表述"}, requires_llm=False)


def _make_compliance_output_policy_tool() -> DeepAgentTool:
    from app.agents.compliance_tools.compliance_output_policy import review
    def _execute(*, answer: str, **kwargs):
        return review(answer=answer)
    return DeepAgentTool(tool_id="compliance_output_policy", name_cn="合规输出审查器",
        description="审查最终合规报告是否包含绝对化法律结论、投资建议、个股推荐、收益承诺等。",
        hard_constraints=["违规警告必须全部保留", "不可被 LLM 忽略或过滤"],
        execute=_execute, input_schema={"answer": "待审查的合规报告全文"},
        output_schema={"warnings": "违规警告", "risk_notice": "合规提示", "is_compliant": "是否合规"}, requires_llm=False)


def build_compliance_registry() -> ToolRegistry:
    """Build the tool registry for ComplianceDeepAgent (7 tools)."""
    registry = ToolRegistry()
    registry.register(_make_compliance_input_parser_tool())
    registry.register(_make_prohibited_expression_detector_tool())
    registry.register(_make_suitability_risk_checker_tool())
    registry.register(_make_disclosure_completeness_checker_tool())
    registry.register(_make_regulatory_basis_matcher_tool())
    registry.register(_make_compliance_rewrite_planner_tool())
    registry.register(_make_compliance_output_policy_tool())
    return registry


# ═══════════════════════════════════════════════════════════════════
# 7 Financial Education tools
# ═══════════════════════════════════════════════════════════════════

def _make_simple_concept_workflow_tool() -> DeepAgentTool:
    """简单概念投教工作流 — wraps simple_concept_workflow.run_simple_concept_workflow()."""
    from app.agents.education_tools.simple_concept_workflow import (
        run_simple_concept_workflow,
    )

    def _execute(*, question: str, user_profile: dict | None = None,
                 sources: list | None = None, **kwargs):
        return run_simple_concept_workflow(
            question=question, user_profile=user_profile, sources=sources,
        )

    return DeepAgentTool(
        tool_id="simple_concept_workflow",
        name_cn="简单概念投教工作流",
        description=(
            "【优先使用】一站式简单概念投教工作流。内部串联：学习者画像分析 → "
            "概念解释 → 证据构建 → 合规审查 → 生成七章报告草稿。"
            "适用于：什么是XX？XX是什么意思？等基础概念解释问题。"
            "返回 draft_sections 包含完整七章内容，DeepAgent 可直接基于此生成最终报告。"
        ),
        hard_constraints=[
            "仅用于简单金融概念解释场景，不用于复杂问题",
            "不得输出具体产品推荐、股票/基金代码、收益承诺",
            "如果该工具已返回 draft_sections，禁止重复调用其他投教工具补同类信息",
            "DeepAgent 应优先使用此工具处理概念解释类问题",
        ],
        execute=_execute,
        input_schema={
            "question": "用户问题原文",
            "user_profile": "用户画像 dict（可选）",
            "sources": "RAG Source 列表（可选，空则跳过）",
        },
        output_schema={
            "workflow_type": "工作流类型",
            "learner_profile": "学习者画像",
            "concepts": "概念解释结果",
            "evidence": "证据构建结果",
            "draft_sections": "七章报告草稿",
            "compliance": "合规审查结果",
            "risk_notice": "风险提示",
            "tool_chain": "内部调用链",
        },
        requires_llm=False,
    )


def _make_learner_profile_analyzer_tool() -> DeepAgentTool:
    """学习者画像解析器 — wraps learner_profile_analyzer.analyze()."""
    from app.agents.education_tools.learner_profile_analyzer import analyze

    def _execute(*, question: str, user_profile: dict | None = None, **kwargs):
        return analyze(question=question, user_profile=user_profile)

    return DeepAgentTool(
        tool_id="learner_profile_analyzer",
        name_cn="学习者画像解析器",
        description=(
            "分析用户的知识水平（beginner/basic/intermediate/advanced）、"
            "学习目标（入门学习/概念解释/风险识别/产品理解/理财规划基础/防诈骗）、"
            "偏好风格（通俗解释/结构化清单/案例化解释/术语解释）。"
            "此工具为纯规则引擎，不调用 LLM。"
        ),
        hard_constraints=[
            "不得把用户投资意图自动转换成投资建议",
            "不得根据画像推荐具体产品",
            "知识水平默认 beginner，不得高估用户水平",
        ],
        execute=_execute,
        input_schema={
            "question": "用户咨询问题原文",
            "user_profile": "用户画像 dict，可选字段：knowledge_level, learning_goal, preferred_style",
        },
        output_schema={
            "knowledge_level": "知识水平 (beginner/basic/intermediate/advanced)",
            "learning_goal": "学习目标",
            "preferred_style": "偏好风格",
            "missing_context": "缺失信息列表",
        },
        requires_llm=False,
    )


def _make_concept_explainer_tool() -> DeepAgentTool:
    """金融概念解释器 — wraps concept_explainer.explain()."""
    from app.agents.education_tools.concept_explainer import explain

    def _execute(*, question: str, learner_profile: dict | None = None, **kwargs):
        return explain(question=question, learner_profile=learner_profile)

    return DeepAgentTool(
        tool_id="concept_explainer",
        name_cn="金融概念解释器",
        description=(
            "解释基金、债券、股票、保险、指数、净值、估值、风险收益等基础金融概念。"
            "输出通俗定义、核心要点、常见误区。"
            "自动匹配用户知识水平调整解释深度。"
        ),
        hard_constraints=[
            "必须用用户知识水平匹配解释深度",
            "不推荐具体证券或产品",
            "不编造不存在的概念定义",
        ],
        execute=_execute,
        input_schema={
            "question": "用户问题原文",
            "learner_profile": "LearnerProfileAnalyzer 输出的学习者画像",
        },
        output_schema={
            "concepts": "概念解释列表，每项含 concept, explanation, key_points, common_misunderstandings",
            "concept_count": "匹配到的概念数量",
            "knowledge_level_used": "使用的知识水平",
        },
        requires_llm=False,
    )


def _make_learning_path_planner_tool() -> DeepAgentTool:
    """学习路径规划器 — wraps learning_path_planner.plan()."""
    from app.agents.education_tools.learning_path_planner import plan

    def _execute(*, question: str, learner_profile: dict | None = None,
                 concepts: list | None = None, **kwargs):
        return plan(question=question, learner_profile=learner_profile, concepts=concepts)

    return DeepAgentTool(
        tool_id="learning_path_planner",
        name_cn="学习路径规划器",
        description=(
            "根据用户知识水平和学习目标生成渐进式学习路径。"
            "例如：先理解风险收益 → 再理解资产类别 → 再学习基金/债券/股票 → 再学习配置原则。"
            "输出学习步骤、预估难度、后续建议主题。"
        ),
        hard_constraints=[
            "学习路径是教育内容，不得变成投资操作指令",
            "不得在路径中插入具体产品推荐",
            "难度预估需匹配用户当前水平",
        ],
        execute=_execute,
        input_schema={
            "question": "用户问题原文",
            "learner_profile": "LearnerProfileAnalyzer 输出",
            "concepts": "ConceptExplainer 输出的概念列表",
        },
        output_schema={
            "learning_steps": "学习步骤列表",
            "step_count": "步骤数量",
            "estimated_difficulty": "预估整体难度",
            "next_topics": "后续建议主题",
        },
        requires_llm=False,
    )


def _make_scam_risk_detector_tool() -> DeepAgentTool:
    """金融诈骗风险识别器 — wraps scam_risk_detector.detect()."""
    from app.agents.education_tools.scam_risk_detector import detect

    def _execute(*, question: str, **kwargs):
        return detect(question=question)

    return DeepAgentTool(
        tool_id="scam_risk_detector",
        name_cn="金融诈骗风险识别器",
        description=(
            "识别用户问题或内容中的金融诈骗风险信号："
            "保本高收益、内幕消息、老师带单、群里荐股、充值返利、虚假平台等。"
            "输出风险类型、警告和防范建议。"
            "仅做风险教育和防骗提示，不判断具体案件法律结论。"
        ),
        hard_constraints=[
            "只做风险教育和防骗提示",
            "不替用户判断具体案件法律结论",
            "不输出正式法律意见",
        ],
        execute=_execute,
        input_schema={"question": "用户问题原文"},
        output_schema={
            "scam_signals": "检测到的诈骗信号列表",
            "signal_count": "信号数量",
            "risk_level": "风险等级 (high/medium/low)",
            "warnings": "警告信息列表",
            "safe_actions": "安全行动建议",
        },
        requires_llm=False,
    )


def _make_product_knowledge_mapper_tool() -> DeepAgentTool:
    """金融产品知识映射器 — wraps product_knowledge_mapper.map_products()."""
    from app.agents.education_tools.product_knowledge_mapper import map_products

    def _execute(*, question: str, learner_profile: dict | None = None, **kwargs):
        return map_products(question=question, learner_profile=learner_profile)

    return DeepAgentTool(
        tool_id="product_knowledge_mapper",
        name_cn="金融产品知识映射器",
        description=(
            "将用户提到的产品类别映射为知识主题："
            "货币基金、债券基金、指数基金、股票基金、混合基金、保险、理财产品等。"
            "解释基本特征、主要风险、前置学习知识。"
        ),
        hard_constraints=[
            "只讲产品类别，不推荐具体产品名称、代码或买卖时点",
            "不给出投资评级或买卖建议",
            "风险说明必须客观完整",
        ],
        execute=_execute,
        input_schema={
            "question": "用户问题原文",
            "learner_profile": "LearnerProfileAnalyzer 输出",
        },
        output_schema={
            "product_categories": "产品类别知识列表",
            "category_count": "类别数量",
            "education_note": "教育提示",
        },
        requires_llm=False,
    )


def _make_education_compliance_tool() -> DeepAgentTool:
    """投教合规审查器 — wraps education_compliance_policy.review()."""
    from app.agents.education_tools.education_compliance_policy import review

    def _execute(*, answer: str, **kwargs):
        return review(answer=answer)

    return DeepAgentTool(
        tool_id="education_compliance_policy",
        name_cn="投教合规审查器",
        description=(
            "审查投教回答是否越界为投资建议："
            "检查是否出现具体股票/基金代码推荐、收益承诺、涨跌预测、替用户决策。"
            "给出 warnings 和 risk_notice。"
            "注意：教育场景中引用骗局话术作为反面教材不视为违规。"
        ),
        hard_constraints=[
            "发现违规必须保留 warnings",
            "投教内容必须包含'仅供学习，不构成投资建议'类风险提示",
            "不可被 DeepAgent 或 LLM 忽略或过滤",
        ],
        execute=_execute,
        input_schema={"answer": "待审查的投教回答全文"},
        output_schema={
            "warnings": "违规警告列表",
            "risk_notice": "风险提示文本",
            "is_compliant": "是否合规",
        },
        requires_llm=False,
    )


def _make_education_evidence_builder_tool() -> DeepAgentTool:
    """投教证据构建器 — wraps education_evidence_builder.build()."""
    from app.agents.education_tools.education_evidence_builder import build
    from app.schemas.consultation import Source

    def _execute(*, sources: list, **kwargs):
        normalized_sources = [
            src if isinstance(src, Source) else Source.model_validate(src)
            for src in sources
        ]
        return build(sources=normalized_sources)

    return DeepAgentTool(
        tool_id="education_evidence_builder",
        name_cn="投教证据构建器",
        description=(
            "基于 RAG 检索到的 Source 列表构建教育类 evidence pack。"
            "优先使用 education_knowledge，必要时引用 risk_knowledge / compliance_knowledge。"
            "不编造文档内容。证据不足时会提示'当前资料有限'。"
        ),
        hard_constraints=[
            "不得编造不存在的知识库内容",
            "证据不足时必须提示'当前资料有限'",
            "只基于 Source.title 和 Source.source_type 做摘要",
        ],
        execute=_execute,
        input_schema={"sources": "KnowledgeRetriever 返回的 Source 列表"},
        output_schema={
            "evidence_summary": "证据摘要",
            "items": "EvidenceItem 列表",
            "has_education": "是否有教育证据",
            "has_risk": "是否有风控证据",
            "has_compliance": "是否有合规证据",
            "low_confidence": "是否低置信",
            "top_titles": "Top-3 证据标题",
        },
        requires_llm=False,
    )


def build_financial_education_registry() -> ToolRegistry:
    """Build the tool registry for FinancialEducationDeepAgent (8 tools).

    Registered tools (all have Chinese business names):
      1. simple_concept_workflow      — 简单概念投教工作流（复合工具，优先）
      2. learner_profile_analyzer     — 学习者画像解析器
      3. concept_explainer            — 金融概念解释器
      4. learning_path_planner        — 学习路径规划器
      5. scam_risk_detector           — 金融诈骗风险识别器
      6. product_knowledge_mapper     — 金融产品知识映射器
      7. education_compliance_policy  — 投教合规审查器
      8. education_evidence_builder   — 投教证据构建器
    """
    registry = ToolRegistry()
    registry.register(_make_simple_concept_workflow_tool())  # Composite — first
    registry.register(_make_learner_profile_analyzer_tool())
    registry.register(_make_concept_explainer_tool())
    registry.register(_make_learning_path_planner_tool())
    registry.register(_make_scam_risk_detector_tool())
    registry.register(_make_product_knowledge_mapper_tool())
    registry.register(_make_education_compliance_tool())
    registry.register(_make_education_evidence_builder_tool())
    return registry
