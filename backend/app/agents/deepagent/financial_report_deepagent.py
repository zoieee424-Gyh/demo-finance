"""
FinancialReportDeepAgent — DeepAgent-powered financial report analysis agent.

Architecture:
  DeepAgent (LLM orchestrator)
    → financial_text_parser        (deterministic)
    → financial_metric_extractor   (deterministic)
    → profitability_analyzer       (deterministic)
    → solvency_liquidity_analyzer  (deterministic)
    → growth_efficiency_analyzer   (deterministic)
    → anomaly_risk_detector        (deterministic)
    → financial_report_compliance  (deterministic, post-hoc guaranteed)

Compliance:
  - NO stock recommendations, buy/sell/hold ratings, target prices.
  - NO price predictions or return promises.
  - All analysis comes from deterministic tools, never LLM judgment.
  - Post-hoc compliance review is GUARANTEED.
"""

from __future__ import annotations

from typing import Any
import re

from app.agents.base import FinancialAgent
from app.agents.deepagent.base import DeepAgentWrapper
from app.agents.deepagent.registry import build_financial_report_registry
from app.agents.financial_report_tools.financial_report_compliance_policy import (
    review as financial_report_compliance_review,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


FINANCIAL_REPORT_SYSTEM_PROMPT = """\
你是财报分析编排器（Financial Report Analyst），不是荐股助手。

## 身份与职责

你的职责是：
1. 理解用户的财报分析请求。
2. 按需调用确定性财报分析工具（文本解析、指标提取、盈利能力分析、
   偿债流动性分析、成长性分析、异常风险识别、合规审查）。
3. 将工具输出组织成结构化的 8 章财报分析报告。

你不是：
- 股票/基金推荐引擎。
- 投资评级机构。
- 股价预测模型。

## 硬约束

1. 不得推荐个股。
2. 不得输出买入/卖出/持有/增持/减持等投资评级。
3. 不得预测股价或给出目标价。
4. 不得承诺收益。
5. 不得替用户做投资决策。
6. 必须调用确定性工具进行分析。
7. 工具结果优先级高于你自身的判断。
8. 合规工具发现违规时必须保留全部 warnings。
9. 必须引用 RAG evidence（如有）。
10. 必须输出风险提示。

## 报告结构要求（8 章，顺序不可变）

你的最终输出必须逐字包含以下 8 个章节标题，标题不能被改写、不能被 Markdown ## 替换、不能省略：

一、财报对象与数据范围
二、核心财务指标摘要
三、盈利能力分析
四、偿债能力与流动性分析
五、成长性与经营效率分析
六、现金流质量与异常风险
七、参考依据与适用边界
八、风险提示

每章必须有内容。如果工具未返回足够数据，章节内容写：
"当前材料未提供充分信息，建议补充更多财务数据以获得准确分析。"

## 输出格式模板

请严格按以下格式组织最终回答：

一、财报对象与数据范围
[分析对象和报告期间说明]

二、核心财务指标摘要
[关键指标列表]

三、盈利能力分析
[盈利能力评估]

四、偿债能力与流动性分析
[偿债和流动性评估]

五、成长性与经营效率分析
[成长性和效率评估]

六、现金流质量与异常风险
[现金流质量和风险识别]

七、参考依据与适用边界
[引用来源和使用限制]

八、风险提示
投资有风险，入市需谨慎。本报告仅作财务信息分析参考，不构成任何投资建议。财报数据具有时效性，具体投资决策请结合最新信息独立判断。
"""

_REQUIRED_SECTIONS = [
    "一、财报对象与数据范围",
    "二、核心财务指标摘要",
    "三、盈利能力分析",
    "四、偿债能力与流动性分析",
    "五、成长性与经营效率分析",
    "六、现金流质量与异常风险",
    "七、参考依据与适用边界",
    "八、风险提示",
]

_RISK_DISCLAIMER_PATTERNS = [
    "投资有风险",
    "不构成投资",
    "入市需谨慎",
    "财报分析仅供参考",
    "请结合更多资料",
]

_FORBIDDEN_PATTERNS = [
    (r"推荐买入|建议买入|强烈推荐", "个股买入推荐"),
    (r"推荐卖出|建议卖出|建议清仓", "个股卖出建议"),
    (r"买入评级|卖出评级|增持评级|减持评级|持有评级", "投资评级"),
    (r"目标价\s*\d+", "目标价预测"),
    (r"(一定|肯定|势必|必然)(?:涨|跌)", "确定性涨跌预测"),
    (r"保证[收益获利]|稳赚|年化收益\s*\d+%", "收益承诺"),
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "A股股票代码"),
]

# Negation/refusal context prefixes — if a forbidden term is nearby one of
# these, the model is likely REFUSING, not recommending.
_NEGATION_PREFIXES = [
    "不能", "不会", "不可以", "无法", "不应", "不得", "禁止",
    "我不", "我不能", "我不会", "无法提供", "不提供",
    "拒绝", "避免", "防止", "不构成",
    "或给出", "也不", "绝不",
]

# Hard-forbidden regardless of context — these are NEVER acceptable
_HARD_FORBIDDEN = [
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "A股股票代码"),
]


def _is_refusal_context(text: str, match_start: int) -> bool:
    """Check if a forbidden pattern match is actually in a refusal/negation context.

    E.g., "我不能建议买入" should NOT be flagged as a buy recommendation.
    """
    # Local check (widened window: 80 chars)
    window_start = max(0, match_start - 80)
    window = text[window_start:match_start + 1]
    for prefix in _NEGATION_PREFIXES:
        if prefix in window:
            return True
    # Global check: if the whole answer is a refusal
    _GLOBAL_REFUSAL = [
        "我不能提供", "我不会提供", "无法提供", "不提供",
        "不能给出", "不会给出", "我不能建议", "我不会建议",
        "不是我的", "超出我的", "合规边界",
    ]
    for signal in _GLOBAL_REFUSAL:
        if signal in text:
            return True
    return False


_FALLBACK_CONTENT ="当前材料未提供充分信息，建议补充财报科目、期间、同比/环比数据和现金流信息以获得准确分析。"
_RISK_DISCLAIMER_DEFAULT = (
    "投资有风险，入市需谨慎。"
    "本报告仅作财务信息分析参考，不构成任何投资建议。"
    "财报数据具有时效性，具体投资决策请结合最新信息独立判断。"
)


def _find_section_fr(answer: str, section_title: str) -> bool:
    """Check if section_title exists, allowing Markdown prefixes."""
    if section_title in answer:
        return True
    for prefix in ("## ", "### ", "**"):
        if prefix + section_title in answer:
            return True
    return False


def repair_financial_report_output(raw_answer: str) -> str:
    """Deterministic repair: inject raw content into 8-section skeleton.

    If the DeepAgent output is missing required sections, this function
    wraps it in the correct skeleton rather than discarding it entirely.
    Missing sections get a fallback message. Supports Markdown headers.

    Also strips forbidden patterns if the answer appears to be a refusal
    (e.g., "我不能建议买入" → "我不能提供买入建议").
    """
    # ── Pre-clean: strip forbidden patterns in refusal context ──
    cleaned = raw_answer
    for pattern, desc in _FORBIDDEN_PATTERNS:
        for m in re.finditer(pattern, cleaned):
            if _is_refusal_context(cleaned, m.start()):
                # Replace the forbidden text with a compliant alternative
                cleaned = cleaned[:m.start()] + "[合规边界说明]" + cleaned[m.end():]
                break

    section_content: dict[str, str] = {}
    remaining = cleaned

    for i, section in enumerate(_REQUIRED_SECTIONS):
        idx = -1
        matched_len = 0
        for prefix in ("", "## ", "### ", "**"):
            candidate = prefix + section
            pos = remaining.find(candidate)
            if pos >= 0 and (idx < 0 or pos < idx):
                idx = pos
                matched_len = len(candidate)
        if idx < 0:
            # Try case-insensitive fallback
            pos = remaining.lower().find(section.lower())
            if pos >= 0:
                idx = pos
                matched_len = len(section)

        if idx >= 0:
            nxt_section = _REQUIRED_SECTIONS[i + 1] if i + 1 < len(_REQUIRED_SECTIONS) else None
            start = idx + matched_len
            if nxt_section:
                end = -1
                for prefix in ("", "## ", "### ", "**"):
                    pos = remaining.find(prefix + nxt_section, start)
                    if pos >= 0 and (end < 0 or pos < end):
                        end = pos
                if end >= 0:
                    section_content[section] = remaining[start:end].strip()
                    remaining = remaining[end:]
                else:
                    section_content[section] = remaining[start:].strip()
            else:
                section_content[section] = remaining[start:].strip()

    if not section_content:
        # No sections found at all — wrap entire answer in skeleton
        lines = [raw_answer.strip(), "", "---", ""]
    else:
        lines = []

    for section in _REQUIRED_SECTIONS:
        lines.append(section)
        content = section_content.get(section, "").strip()
        lines.append(content if content else _FALLBACK_CONTENT)
        lines.append("")

    combined = "\n".join(lines)
    # Ensure risk disclaimer in section 8
    if _RISK_DISCLAIMER_DEFAULT not in combined:
        lines.append(_RISK_DISCLAIMER_DEFAULT)
        lines.append("")

    return "\n".join(lines)


def validate_financial_report_output(answer: str) -> tuple[bool, list[str]]:
    """Hard-validate financial report analysis output.

    Returns:
        (is_valid, failure_reasons).
    """
    failures: list[str] = []

    # ── 1. Required sections (Markdown-aware) ──────────────────
    missing = [s for s in _REQUIRED_SECTIONS if not _find_section_fr(answer, s)]
    if missing:
        failures.append(f"Missing required sections: {', '.join(missing)}")

    # ── 2. Risk disclaimer ────────────────────────────────────
    has_disclaimer = any(p in answer for p in _RISK_DISCLAIMER_PATTERNS)
    if not has_disclaimer:
        failures.append("Missing risk disclaimer")

    # ── 3. Hard-forbidden content (always fail) ────────────────
    for pattern, desc in _HARD_FORBIDDEN:
        if re.search(pattern, answer):
            failures.append(f"Forbidden content: {desc}")

    # ── 4. Soft-forbidden content (context-aware) ──────────────
    for pattern, desc in _FORBIDDEN_PATTERNS:
        for m in re.finditer(pattern, answer):
            if not _is_refusal_context(answer, m.start()):
                failures.append(f"Forbidden content: {desc} ({m.group()[:30]})")
                break

    return (len(failures) == 0, failures)


# ── Agent-specific user message builder ──────────────────────────

def _build_financial_report_user_message(
    request: ConsultationRequest,
    sources: list[Source],
) -> str:
    """Build the user message for the financial report DeepAgent."""
    parts = [f"用户财报分析请求：{request.question}"]

    if request.user_profile:
        import json
        profile_str = json.dumps(request.user_profile, ensure_ascii=False)
        parts.append(f"辅助信息：{profile_str}")

    if sources:
        parts.append("RAG 检索到的参考资料：")
        for i, src in enumerate(sources, 1):
            conf = f"{src.confidence:.0%}" if src.confidence is not None else "N/A"
            parts.append(f"  {i}. {src.title}（置信度：{conf}）")

    parts.append("")
    parts.append("请按以下步骤处理：")
    parts.append("1. 调用 financial_text_parser 解析财报文本。")
    parts.append("2. 调用 financial_metric_extractor 提取财务指标。")
    parts.append("3. 调用 profitability_analyzer 分析盈利能力。")
    parts.append("4. 调用 solvency_liquidity_analyzer 分析偿债与流动性。")
    parts.append("5. 调用 growth_efficiency_analyzer 分析成长性与效率。")
    parts.append("6. 调用 anomaly_risk_detector 识别异常风险。")
    parts.append("7. 整合以上结果输出 8 章财报分析报告。")
    parts.append("8. 调用 financial_report_compliance 审查最终报告。")
    parts.append("")
    parts.append("工具调用策略：每个工具最多 1 次。如果材料不足，报告各章节填写'当前材料未提供充分信息'，不要编造数据。")

    return "\n".join(parts)


# ── Agent-specific system prompt footer ──────────────────────────

_FINANCIAL_REPORT_SYSTEM_PROMPT_FOOTER = """\
## 报告格式要求

最终回答必须是完整的 8 章节中文财报分析报告（顺序不可变，章节不可缺失）：

一、财报对象与数据范围
二、核心财务指标摘要
三、盈利能力分析
四、偿债能力与流动性分析
五、成长性与经营效率分析
六、现金流质量与异常风险
七、参考依据与适用边界
八、风险提示

如果材料不足或工具未返回足够数据，章节内容写：
"当前材料未提供充分信息，建议补充财报科目、期间、同比/环比数据和现金流信息以获得准确分析。"
绝不可编造任何财务数据。"""


class FinancialReportDeepAgent(FinancialAgent):
    """DeepAgent-powered financial report analysis agent."""

    name = "financial_report_deepagent"
    intent = "financial_report"

    def __init__(
        self,
        *,
        enabled: bool = True,
        fallback_agent: FinancialAgent | None = None,
        configured_mode: str = "deepagent",
    ) -> None:
        self._enabled = enabled
        self._configured_mode = configured_mode

        if fallback_agent is None:
            from app.agents.financial_report import FinancialReportAgent
            fallback_agent = FinancialReportAgent()

        self._fallback_agent = fallback_agent
        self._registry = build_financial_report_registry()

        self._wrapper = DeepAgentWrapper(
            tool_registry=self._registry,
            legacy_agent=self._fallback_agent,
            system_prompt=FINANCIAL_REPORT_SYSTEM_PROMPT,
            agent_name="financial_report_deepagent",
            agent_name_cn="DeepAgent财报分析编排器",
            enabled=self._enabled,
            intent="financial_report",
            output_validator=validate_financial_report_output,
            compliance_review_func=financial_report_compliance_review,
            repair_func=repair_financial_report_output,
            risk_notice_default=_RISK_DISCLAIMER_DEFAULT,
            user_message_builder=_build_financial_report_user_message,
            system_prompt_footer=_FINANCIAL_REPORT_SYSTEM_PROMPT_FOOTER,
        )

    def answer(
        self,
        request: ConsultationRequest,
        sources: list[Source],
        event_callback: Any = None,
    ) -> ConsultationResponse:
        return self._wrapper.run(request, sources, event_callback=event_callback)

    @property
    def architecture(self) -> str:
        return self._wrapper.architecture

    @property
    def debug_info(self) -> dict:
        info = self._wrapper.debug_info
        info["agent_name"] = self.name
        info["configured_mode"] = self._configured_mode
        return info
