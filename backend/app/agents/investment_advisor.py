"""
InvestmentAdvisorAgent（智能投顾智能体）

Single master agent + tool modules architecture.
Orchestrates eight internal tool modules to produce structured advisory answers.

Pipeline:
  ProfileAnalyzer → GoalPlanner → RiskAssessor → AllocationEngine
  → FundDcaPlanner → HoldingDiagnostic (conditional) → MarketHotspotInterpreter (conditional)
  → build_answer() → AdvisoryCompliancePolicy → ConsultationResponse

Compliance:
  - Does NOT recommend specific stocks, funds, or products.
  - Does NOT predict price movements.
  - Does NOT guarantee returns.
  - Does NOT make investment decisions for users.
  - Always includes risk notice and data sources.
"""
from __future__ import annotations

from app.agents.base import FinancialAgent
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source
from app.agents.investment_advisor_tools.profile_analyzer import analyze as analyze_profile
from app.agents.investment_advisor_tools.goal_planner import plan as plan_goals
from app.agents.investment_advisor_tools.risk_assessor import assess as assess_risk
from app.agents.investment_advisor_tools.allocation_engine import allocate as allocate_assets
from app.agents.investment_advisor_tools.advisory_compliance_policy import review as review_compliance
from app.agents.investment_advisor_tools.fund_dca_planner import plan_dca
from app.agents.investment_advisor_tools.holding_diagnostic import diagnose as diagnose_holdings
from app.agents.investment_advisor_tools.market_hotspot_interpreter import interpret as interpret_hotspot
from app.agents.investment_advisor_tools.profile_analyzer import enrich_profile_with_llm
from app.llm.provider import get_provider
from app.rag.evidence import (
    EvidencePack,
    build_evidence_pack,
    format_evidence_section,
    format_evidence_inline,
)


# ── Risk level display mapping ───────────────────────────────────

_RISK_LABELS: dict[str, str] = {
    "conservative": "保守型",
    "stable": "稳健型",
    "balanced": "平衡型",
    "aggressive": "进取型",
}

_PRIORITY_LABELS: dict[str, str] = {
    "capital_preservation": "本金保护优先",
    "balanced": "稳健增值",
    "growth": "长期增长优先",
}

_REQUIRED_REPORT_SECTIONS = [
    "一、用户画像摘要",
    "二、投资目标分析",
    "三、风险评估结果",
    "四、资产配置建议",
    "五、基金定投规划",
    "六、持仓诊断",
    "七、市场热点解读",
    "八、参考依据与适用边界",
    "九、风险提示",
]


def _extract_protected_terms(text: str) -> set[str]:
    """Extract facts that LLM rewriting must not alter."""
    import re

    protected: set[str] = set()
    protected.update(re.findall(r"\d+(?:\.\d+)?%", text))
    protected.update(re.findall(r"\d+\s*个月", text))
    protected.update(re.findall(r"约\d+年", text))
    for label in ("保守型", "稳健型", "平衡型", "进取型", "低", "中", "高"):
        if label in text:
            protected.add(label)
    return protected


def _rewrite_preserves_hard_constraints(original: str, rewritten: str) -> bool:
    """Validate that LLM rewrite kept report structure and protected facts."""
    if not rewritten or rewritten.startswith("[LLM Mock]"):
        return False
    if any(section not in rewritten for section in _REQUIRED_REPORT_SECTIONS):
        return False
    for term in _extract_protected_terms(original):
        if term not in rewritten:
            return False
    return True


def _build_answer(
    profile: dict,
    goals: list[dict],
    risk: dict,
    allocation: dict,
    sources: list[Source],
    evidence: EvidencePack | None = None,
    dca_plan: dict | None = None,
    holding_diag: dict | None = None,
    hotspot: dict | None = None,
) -> str:
    """Build a structured Chinese advisory answer from tool outputs.

    Evidence from RAG sources is injected into multiple sections:
      - Section 四 (allocation): inline advisory evidence
      - Section 五 (DCA): inline advisory evidence
      - Section 六 (holdings): inline risk evidence
      - Section 八 (sources): detailed evidence list with usage hints
      - Section 九 (risk notice): compliance evidence reinforcement
    """
    if evidence is None:
        evidence = build_evidence_pack(sources)

    # ── Header ────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("【智能投顾分析报告】")
    lines.append("")

    if evidence.low_confidence:
        lines.append("⚠ 当前参考资料置信度有限，以下建议仅作通用框架，请结合自身情况谨慎参考。")
        lines.append("")

    # ── 1. User profile summary ───────────────────────────────
    lines.append("一、用户画像摘要")
    lines.append(f"  风险偏好：{_RISK_LABELS.get(profile.get('risk_preference', ''), profile.get('risk_preference', '未明确'))}")
    lines.append(f"  投资经验：{'新手' if profile.get('investment_experience') == 'beginner' else '有经验'}")
    lines.append(f"  流动性需求：{'高' if profile.get('liquidity_need') == 'high' else '中' if profile.get('liquidity_need') == 'medium' else '低'}")

    constraints = profile.get("constraints", [])
    if constraints:
        lines.append(f"  特殊约束：{'；'.join(constraints)}")

    missing = profile.get("missing_fields", [])
    if missing:
        field_labels = {
            "income_level": "收入水平",
            "risk_preference": "风险偏好",
            "investment_horizon": "投资期限",
        }
        missing_cn = [field_labels.get(f, f) for f in missing]
        lines.append(f"  ⚠ 缺少信息：{'、'.join(missing_cn)}，建议补充以获得更精准建议。")
    lines.append("")

    # ── 2. Investment goals ───────────────────────────────────
    lines.append("二、投资目标分析")
    if goals:
        for i, g in enumerate(goals, 1):
            goal_type = g.get("description", g.get("goal_type", ""))
            horizon = g.get("time_horizon_months")
            horizon_str = f"{horizon}个月（约{horizon//12}年）" if horizon else "未明确期限"
            priority = _PRIORITY_LABELS.get(g.get("priority", ""), g.get("priority", ""))
            lines.append(f"  {i}. {goal_type}：期限{horizon_str}，优先级：{priority}")
    else:
        lines.append("  未检测到明确的投资目标，以下建议基于通用长期增值假设。")
    lines.append("")

    # ── 3. Risk assessment ────────────────────────────────────
    lines.append("三、风险评估结果")
    risk_inline = format_evidence_inline(evidence, "risk")
    lines.append(f"  风险等级：{_RISK_LABELS.get(risk.get('risk_level', ''), risk.get('risk_level', ''))}")
    lines.append(f"  最大回撤容忍度：{'低' if risk.get('max_drawdown_tolerance') == 'low' else '中' if risk.get('max_drawdown_tolerance') == 'medium' else '高'}")
    lines.append(f"  适合资产类别：{'、'.join(risk.get('suitable_assets', []))}")
    if risk_inline:
        lines.append(f"  {risk_inline}")

    unsuitable = risk.get("unsuitable_assets", [])
    if unsuitable:
        lines.append(f"  不适合资产类别：{'、'.join(unsuitable)}")
    lines.append("")

    # ── 4. Asset allocation ───────────────────────────────────
    lines.append("四、资产配置建议")
    alloc_inline = format_evidence_inline(evidence, "advisory")
    alloc_items = allocation.get("allocation", [])
    for item in alloc_items:
        pct = item.get("ratio", 0)
        bar = "█" * int(pct / 5)  # Visual bar
        lines.append(f"  {item.get('asset_class', '')}：{pct}% {bar}")

    rationale = allocation.get("rationale", "")
    if rationale:
        lines.append(f"")
        lines.append(f"  配置逻辑：{rationale}")
    if alloc_inline:
        lines.append(f"  {alloc_inline}")
    lines.append("")

    # ── 5. Fund DCA plan ──────────────────────────────────────
    lines.append("五、基金定投规划")
    dca_inline = format_evidence_inline(evidence, "advisory")
    if dca_plan:
        lines.append(f"  定投频率：{dca_plan.get('frequency', 'monthly')}")
        lines.append(f"  建议金额：{dca_plan.get('amount_ratio', '月结余的30%-50%')}")
        categories = dca_plan.get("suitable_categories", [])
        if categories:
            lines.append(f"  适合类别：{'；'.join(categories)}")
        review_items = dca_plan.get("review_conditions", [])
        if review_items:
            lines.append("  复盘条件：")
            for rc in review_items:
                lines.append(f"    - {rc}")
        pause_items = dca_plan.get("pause_conditions", [])
        if pause_items:
            lines.append("  暂停条件：")
            for pc in pause_items:
                lines.append(f"    - {pc}")
        if dca_inline:
            lines.append(f"  {dca_inline}")
    else:
        lines.append("  本次未提供定投规划信息。")
    lines.append("")

    # ── 6. Holding diagnostic ─────────────────────────────────
    lines.append("六、持仓诊断")
    hold_inline = format_evidence_inline(evidence, "risk")
    if holding_diag:
        issues = holding_diag.get("issues", [])
        directions = holding_diag.get("adjustment_directions", [])
        if issues:
            lines.append("  发现以下问题：")
            for issue in issues:
                lines.append(f"    ⚠ {issue}")
        else:
            lines.append("  未发现明显持仓问题。")
        if directions:
            lines.append("  调整方向建议：")
            for d in directions:
                lines.append(f"    → {d}")
    else:
        lines.append("  本次未提供持仓信息，无法进行持仓诊断。如需诊断，请在用户画像中提供持仓数据。")
    if hold_inline:
        lines.append(f"  {hold_inline}")
    lines.append("")

    # ── 7. Market hotspot interpretation ──────────────────────
    lines.append("七、市场热点解读")
    if hotspot and hotspot.get("is_relevant"):
        lines.append(f"  热点主题：{hotspot.get('topic', '')}")
        drivers = hotspot.get("drivers", [])
        if drivers:
            lines.append(f"  驱动因素：{'；'.join(drivers)}")
        risk_points = hotspot.get("risk_points", [])
        if risk_points:
            lines.append(f"  风险点：{'；'.join(risk_points)}")
        neutral = hotspot.get("neutral_view", "")
        if neutral:
            lines.append(f"  中性观察建议：{neutral}")
    else:
        lines.append("  本次咨询未涉及当前市场热点主题，建议关注长期资产配置而非短期热点。")
    lines.append("")

    # ── 8. Evidence & sources ─────────────────────────────────
    lines.append("八、参考依据与适用边界")
    evidence_text = format_evidence_section(evidence)
    lines.append(evidence_text)
    if not evidence.items:
        lines.append(f"  （原始 sources 数量：{len(sources)}）")
        if sources:
            for i, src in enumerate(sources, 1):
                conf_str = f"{src.confidence:.0%}" if src.confidence is not None else "N/A"
                lines.append(f"  {i}. {src.title}（置信度：{conf_str}）")
    lines.append("")

    # ── 9. Risk notice ────────────────────────────────────────
    lines.append("九、风险提示")
    if evidence.has_compliance:
        compliance_inline = format_evidence_inline(evidence, "compliance", max_items=2)
        lines.append(f"  {compliance_inline}")
    lines.append("  投资有风险，入市需谨慎。本报告仅作信息参考，不构成投资决策依据。")
    lines.append("  资产配置建议基于通用模型，具体投资行为请结合个人实际情况独立判断。")
    lines.append("  过往业绩不代表未来表现，市场有风险，投资需谨慎。")
    lines.append("")

    # ── Footer ────────────────────────────────────────────────
    lines.append("— 以上分析由智能投顾智能体（InvestmentAdvisorAgent）生成 —")

    return "\n".join(lines)


def rewrite_report_with_llm(
    answer_text: str,
    provider: object | None = None,
) -> str:
    """Optionally polish the advisory report with LLM rewriting.

    This is an OPTIONAL post-processing step. If the LLM provider is not
    configured, the original answer is returned unchanged.

    The LLM is ONLY allowed to:
      - Improve fluency, wording, and readability.
      - Fix grammar and typos.

    The LLM MUST NOT:
      - Change any numbers, ratios, percentages, or risk levels.
      - Add new investment conclusions or advice.
      - Recommend stocks, funds, or products.
      - Remove or alter the risk notice section.
      - Modify the 9-section report structure.

    After LLM rewriting, the text MUST still pass AdvisoryCompliancePolicy
    and global ComplianceGuard.

    Args:
        answer_text: The original structured advisory report.
        provider: LLMProvider instance (or None).

    Returns:
        Polished report text, or original text if LLM is unavailable.
    """
    if provider is None:
        return answer_text

    if not hasattr(provider, "is_configured") or not provider.is_configured():
        return answer_text

    from app.llm.prompts import build_report_rewrite_prompt

    prompt = build_report_rewrite_prompt(original_report=answer_text)

    raw = provider.complete(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max(min(len(answer_text) * 2, 4000), 2000),
    )

    # If mock response, return original
    if raw.startswith("[LLM Mock]"):
        return answer_text

    if not _rewrite_preserves_hard_constraints(answer_text, raw):
        return answer_text

    return raw


class InvestmentAdvisorAgent(FinancialAgent):
    """Intelligent investment advisory agent.

    Orchestrates: ProfileAnalyzer → GoalPlanner → RiskAssessor →
    AllocationEngine → FundDcaPlanner → HoldingDiagnostic (conditional) →
    MarketHotspotInterpreter (conditional) → AdvisoryCompliancePolicy.
    """

    name = "investment_advisor"
    intent = "advisory"

    def answer(self, request: ConsultationRequest, sources: list[Source]) -> ConsultationResponse:
        """Generate a compliance-checked advisory answer.

        Args:
            request: User consultation with question and optional profile.
            sources: Evidence retrieved by KnowledgeRetriever.

        Returns:
            ConsultationResponse with structured answer, warnings, and risk notice.
        """
        # ── Step 1: Analyze user profile ──────────────────────
        profile = analyze_profile(
            question=request.question,
            user_profile=request.user_profile,
        )
        profile = enrich_profile_with_llm(
            profile=profile,
            question=request.question,
            provider=get_provider(),
        )

        # ── Step 2: Plan investment goals ─────────────────────
        goals = plan_goals(
            question=request.question,
            profile=profile,
        )

        # ── Step 3: Assess risk tolerance ─────────────────────
        risk = assess_risk(profile=profile, goals=goals)

        # ── Step 4: Generate asset allocation ─────────────────
        allocation = allocate_assets(
            risk_level=risk["risk_level"],
            goals=goals,
        )

        # ── Step 5: Branch tools ──────────────────────────────
        # 5a: Fund DCA plan (always generated)
        dca_plan = plan_dca(
            profile=profile,
            goals=goals,
            risk_level=risk["risk_level"],
        )

        # 5b: Holding diagnostic (only when holdings provided)
        holdings = request.user_profile.get("holdings", []) if request.user_profile else []
        holding_diag = None
        if holdings:
            holding_diag = diagnose_holdings(
                holdings=holdings,
                risk_level=risk["risk_level"],
            )

        # 5c: Market hotspot interpretation (only when relevant keywords detected)
        hotspot = interpret_hotspot(question=request.question)

        # ── Step 6: Build evidence pack from sources ──────────
        evidence = build_evidence_pack(sources)

        # ── Step 7: Build structured answer ───────────────────
        answer_text = _build_answer(
            profile=profile,
            goals=goals,
            risk=risk,
            allocation=allocation,
            sources=sources,
            evidence=evidence,
            dca_plan=dca_plan,
            holding_diag=holding_diag,
            hotspot=hotspot,
        )

        # ── Step 8: Optional LLM report polish ────────────────
        answer_text = rewrite_report_with_llm(answer_text, provider=get_provider())

        # ── Step 9: Internal compliance review ────────────────
        # MUST run AFTER any LLM modifications to catch violations
        compliance = review_compliance(answer_text)

        # ── Step 10: Assemble response ────────────────────────
        return ConsultationResponse(
            intent=self.intent,
            agent=self.name,
            answer=answer_text,
            risk_notice=compliance.get("risk_notice", ""),
            sources=sources,
            warnings=compliance.get("warnings", []),
        )
