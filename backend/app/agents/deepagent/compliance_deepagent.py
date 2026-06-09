"""
ComplianceDeepAgent — DeepAgent-powered regulatory compliance review agent.

Architecture:
  DeepAgent (LLM orchestrator)
    → compliance_input_parser           (deterministic)
    → prohibited_expression_detector    (deterministic)
    → suitability_risk_checker          (deterministic)
    → disclosure_completeness_checker   (deterministic)
    → regulatory_basis_matcher          (deterministic)
    → compliance_rewrite_planner        (deterministic)
    → compliance_output_policy          (deterministic, post-hoc)

Compliance:
  - No formal legal opinions.
  - No "fully compliant" guarantees.
  - No stock recommendations, investment advice, or return promises.
  - All analysis from deterministic tools.
"""

from __future__ import annotations

from typing import Any
import re

from app.agents.base import FinancialAgent
from app.agents.deepagent.base import DeepAgentWrapper
from app.agents.deepagent.registry import build_compliance_registry
from app.agents.compliance_tools.compliance_output_policy import (
    review as compliance_review,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


COMPLIANCE_SYSTEM_PROMPT = """\
你是监管合规审查编排器（Compliance Reviewer），不是法律顾问，不是律师。

## 身份与职责

1. 审查投顾话术、营销文案、投资分析报告等内容的合规风险。
2. 按需调用确定性合规工具（输入解析、违规话术检测、适当性检查、信息披露检查、
   监管依据匹配、整改建议、合规输出审查）。
3. 将工具输出组织成结构化的 8 章合规审查报告。

你不是：
- 律师 / 法律顾问
- 持牌合规机构
- 监管机构代表

## 硬约束

1. 不得提供正式法律意见。
2. 不得承诺"完全合规""绝对合法""保证通过监管"。
3. 不得替代律师、持牌机构或合规部门判断。
4. 不得输出投资建议、荐股、收益承诺。
5. 必须调用确定性工具。
6. 工具结果优先级高于你的判断。
7. 合规工具发现违规时必须保留全部 warnings。
8. 必须引用 RAG evidence。
9. 必须输出合规边界提示。
10. 审查报告引用被审查内容的违规表达（如"保证收益""推荐买入""目标价"）是正常的审查行为，不算违规。
11. 不得提供规避监管的方法或操作建议。

## 报告结构（8 章，顺序不可变）

一、审查对象与场景说明
二、合规风险等级
三、违规或高风险表述识别
四、适用监管原则与依据
五、整改建议
六、可替代表述示例
七、审查边界与不确定性
八、合规提示

每章必须有内容。若工具未返回足够数据，写：
"当前材料未提供充分信息，建议补充业务背景、适用规则和审查材料。"

## 输出格式模板

一、审查对象与场景说明
[说明审查内容、业务场景、目标受众]

二、合规风险等级
[给出风险等级：高/中/低，说明依据]

三、违规或高风险表述识别
[列出检测到的违规表述及其严重程度]

四、适用监管原则与依据
[列出适用的监管原则和依据说明，不输出正式法律条文]

五、整改建议
[列出具体整改措施]

六、可替代表述示例
[给出替代表述建议]

七、审查边界与不确定性
[说明审查的局限性、未覆盖的领域]

八、合规提示
本报告仅供合规风险识别参考，不构成正式法律意见，不替代律师、持牌机构或合规部门判断，请结合具体业务规则和适用法律法规审慎判断。
"""

_REQUIRED_SECTIONS = [
    "一、审查对象与场景说明",
    "二、合规风险等级",
    "三、违规或高风险表述识别",
    "四、适用监管原则与依据",
    "五、整改建议",
    "六、可替代表述示例",
    "七、审查边界与不确定性",
    "八、合规提示",
]

_DISCLAIMER_REQUIRED = [
    "不构成正式法律意见",
    "审慎判断",
]

# Only truly forbidden: model making absolute legal claims,
# pretending to be a lawyer, or outputting stock codes
_FORBIDDEN_STRICT = [
    (r"完全合规|绝对合法|保证通过监管", "绝对化合规结论"),
    (r"本律师|本法律顾问|法律意见书", "替代律师判断"),
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "股票代码"),
]

# Patterns that are flagged ONLY if NOT in review context
# These become warnings, not failures
_SOFT_WARN_PATTERNS = [
    (r"保证[收益获利]|保证.*年化|稳赚|包赚|必赚", "收益承诺表述"),
    (r"推荐买入|建议买入|强烈推荐买入", "买入推荐表述"),
    (r"推荐卖出|建议卖出|建议清仓", "卖出建议表述"),
    (r"目标价\s*\d+", "目标价表述"),
    (r"(?:一定|肯定|势必|必然).{0,2}(?:涨|跌|上涨|下跌)", "涨跌预测表述"),
]

_FALLBACK_CONTENT = "当前材料未提供充分信息，建议补充业务背景、适用规则和审查材料。"
_COMPLIANCE_NOTICE = (
    "本报告仅供合规风险识别参考，不构成正式法律意见，"
    "不替代律师、持牌机构或合规部门判断，"
    "请结合具体业务规则和适用法律法规审慎判断。"
)

# ── Review context detection ─────────────────────────────────────

# Review context keywords — MUST appear nearby for a match to be considered
# a "quoted/identified" violation rather than the model's own output.
_REVIEW_CONTEXT_KW = [
    # Strong: section 三 style — explicitly identifying violations
    "违规或高风险表述识别", "违规表述", "原文", "被审查",
    "审查对象", "存在以下违规", "识别到", "检测到",
    "该表述", "该内容", "该文案", "该话术", "该宣传",
    "审查发现", "违反", "不符合",
    "涉嫌违规", "涉嫌违法", "存在合规风险",
    # Medium: remediation style — removing/fixing violations
    "需要删除", "应删除", "必须删除", "需整改",
    "禁止使用", "不得使用", "不应出现", "避免使用",
    "建议修改为", "应改为", "调整为",
    "违规示例", "违规话术", "以下表述存在合规风险",
]

# Negative signals — if these appear near match, it's likely MODEL suggestion
_NEGATIVE_SIGNALS = [
    "你可以这样", "建议你这样", "你可以参考", "可以写",
    "可以这样写", "如下所示", "例如这样",
]

_ABSOLUTE_CLAIM_REVIEW_KW = [
    "不得", "不能", "不应", "不可", "禁止", "严禁", "避免", "删除", "去除",
    "绝对化", "检测到", "警告",
    "不得承诺", "不应承诺", "不得宣称", "不应宣称", "不能宣称",
    "合规输出审查", "输出审查", "审查通过",
]


def _is_review_context(text: str, match_start: int, match_end: int | None = None) -> bool:
    """Check if a matched pattern at match_start is in a review/quoting context.

    Only returns True if:
    1. Global: the document contains compliance review section headers
       (indicating the entire text is a review report), OR
    2. Local: a review-context keyword appears in the surrounding window, AND
       no negative signal (model suggestion language) appears nearby.

    If neither condition is met, the match is treated as the model's own
    output — and will be flagged by the validator.
    """
    window_start = max(0, match_start - 100)
    window_end_val = min(len(text), match_start + 100)
    window = text[window_start:window_end_val]

    # Negative check FIRST: if model is making suggestions, NEVER review context
    for ns in _NEGATIVE_SIGNALS:
        if ns in window:
            return False

    # Global check: if text contains compliance review section headers
    # AND no negative signals nearby, treat as review context
    _GLOBAL_REVIEW_HEADERS = [
        "三、违规或高风险表述识别",
        "一、审查对象与场景说明",
        "六、可替代表述示例",
    ]
    if any(h in text for h in _GLOBAL_REVIEW_HEADERS):
        return True

    # Local check: review keyword must appear in window
    for kw in _REVIEW_CONTEXT_KW:
        if kw in window:
            return True

    return False


def _is_absolute_claim_review_context(
    text: str, match_start: int, match_end: int | None = None
) -> bool:
    """Allow absolute-compliance phrases only when they are being reviewed.

    "该内容完全合规" is forbidden, but "不得宣称完全合规" or
    "合规输出审查通过" should not trigger fallback.
    """
    window_start = max(0, match_start - 60)
    window_end_val = min(len(text), (match_end or match_start) + 24)
    window = text[window_start:window_end_val]

    for ns in _NEGATIVE_SIGNALS:
        if ns in window:
            return False

    return any(kw in window for kw in _ABSOLUTE_CLAIM_REVIEW_KW)


# ── Validator ─────────────────────────────────────────────────────

def _find_section_compliance(answer: str, section_title: str) -> bool:
    """Check if section_title exists, allowing Markdown prefixes."""
    if section_title in answer:
        return True
    for prefix in ("## ", "### ", "**"):
        if prefix + section_title in answer:
            return True
    return False


def validate_compliance_output(answer: str) -> tuple[bool, list[str]]:
    """Validate compliance agent output.

    Structure checks (HARD — trigger fallback):
      1. 8-section completeness
      2. Disclaimer presence

    Content checks:
      - Strict forbidden (absolute legal claims, lawyer impersonation, stock codes): HARD fail
      - Soft warn patterns (收益承诺/买入推荐/目标价/涨跌预测):
        WARN only if NOT in review context. If in review context, pass.

    Returns:
        (is_valid, failure_reasons).
    """
    failures: list[str] = []

    # ── 1. Required sections (Markdown-aware) ──────────────────
    missing = [s for s in _REQUIRED_SECTIONS if not _find_section_compliance(answer, s)]
    if missing:
        failures.append(f"Missing required sections: {', '.join(missing)}")

    # ── 2. Disclaimer check ────────────────────────────────────
    has_disclaimer = any(p in answer for p in _DISCLAIMER_REQUIRED)
    if not has_disclaimer:
        failures.append("Missing compliance disclaimer ('不构成正式法律意见')")

    # ── 3. Strict forbidden (always fail) ──────────────────────
    for pattern, desc in _FORBIDDEN_STRICT:
        for m in re.finditer(pattern, answer):
            if desc == "绝对化合规结论" and _is_absolute_claim_review_context(
                answer, m.start(), m.end()
            ):
                continue
            failures.append(f"Strict forbidden content: {desc}")
            break

    # ── 4. Soft warn patterns (context-aware) ──────────────────
    for pattern, desc in _SOFT_WARN_PATTERNS:
        for m in re.finditer(pattern, answer):
            if not _is_review_context(answer, m.start(), m.end()):
                # Only fail if NOT in review context
                failures.append(f"Forbidden content (not in review context): {desc} ({m.group()[:30]})")
                break

    return (len(failures) == 0, failures)


# ── Repair ────────────────────────────────────────────────────────

def repair_compliance_output(raw_answer: str) -> str:
    """Deterministic repair: inject raw content into 8-section skeleton.

    Supports Markdown section headers (## 一、..., **一、..., etc.).
    """
    section_content: dict[str, str] = {}
    remaining = raw_answer

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
        lines = [raw_answer.strip(), "", "---", ""]
    else:
        lines = []

    for section in _REQUIRED_SECTIONS:
        lines.append(section)
        lines.append(section_content.get(section, _FALLBACK_CONTENT))
        lines.append("")

    combined = "\n".join(lines)
    if _COMPLIANCE_NOTICE not in combined:
        lines.append(_COMPLIANCE_NOTICE)
        lines.append("")

    return "\n".join(lines)


# ── Agent-specific user message ──────────────────────────────────

def _build_compliance_user_message(
    request: ConsultationRequest,
    sources: list[Source],
) -> str:
    """Build the user message for the compliance DeepAgent."""
    parts = [f"待审查的内容：{request.question}"]

    if request.user_profile:
        import json
        profile_str = json.dumps(request.user_profile, ensure_ascii=False)
        parts.append(f"审查背景信息：{profile_str}")

    if sources:
        parts.append("RAG 检索到的参考资料：")
        for i, src in enumerate(sources, 1):
            conf = f"{src.confidence:.0%}" if src.confidence is not None else "N/A"
            parts.append(f"  {i}. {src.title}（置信度：{conf}）")

    parts.append("")
    parts.append("请按以下步骤审查：")
    parts.append("1. 调用 compliance_input_parser 解析待审查内容。")
    parts.append("2. 调用 prohibited_expression_detector 检测违规话术。")
    parts.append("3. 调用 suitability_risk_checker 检查适当性。")
    parts.append("4. 调用 disclosure_completeness_checker 检查信息披露。")
    parts.append("5. 调用 regulatory_basis_matcher 匹配监管依据。")
    parts.append("6. 调用 compliance_rewrite_planner 生成整改建议。")
    parts.append("7. 整合以上结果输出 8 章合规审查报告。")
    parts.append("8. 调用 compliance_output_policy 审查最终报告。")
    parts.append("")
    parts.append("工具调用策略：每个工具最多 1 次，调完立即输出 8 章报告。")

    return "\n".join(parts)


# ── Agent-specific system prompt footer ──────────────────────────

_COMPLIANCE_SYSTEM_PROMPT_FOOTER = """\
## 报告格式要求

最终回答必须是完整的 8 章节中文合规审查报告（顺序不可变，章节不可缺失）：

一、审查对象与场景说明
二、合规风险等级
三、违规或高风险表述识别
四、适用监管原则与依据
五、整改建议
六、可替代表述示例
七、审查边界与不确定性
八、合规提示

在报告末尾附上完整的合规提示：本报告仅供合规风险识别参考，不构成正式法律意见。
"""


# ── DeepAgent class ───────────────────────────────────────────────

class ComplianceDeepAgent(FinancialAgent):
    """DeepAgent-powered compliance review agent."""

    name = "compliance_deepagent"
    intent = "compliance"

    def __init__(self, *, enabled: bool = True, fallback_agent: FinancialAgent | None = None,
                 configured_mode: str = "deepagent") -> None:
        self._enabled = enabled
        self._configured_mode = configured_mode
        if fallback_agent is None:
            from app.agents.compliance import ComplianceAgent
            fallback_agent = ComplianceAgent()
        self._fallback_agent = fallback_agent
        self._registry = build_compliance_registry()
        self._wrapper = DeepAgentWrapper(
            tool_registry=self._registry, legacy_agent=self._fallback_agent,
            system_prompt=COMPLIANCE_SYSTEM_PROMPT, agent_name="compliance_deepagent",
            agent_name_cn="DeepAgent合规审查编排器", enabled=self._enabled,
            intent="compliance", output_validator=validate_compliance_output,
            compliance_review_func=compliance_review, repair_func=repair_compliance_output,
            risk_notice_default=_COMPLIANCE_NOTICE,
            user_message_builder=_build_compliance_user_message,
            system_prompt_footer=_COMPLIANCE_SYSTEM_PROMPT_FOOTER,
        )

    def answer(self, request: ConsultationRequest, sources: list[Source], event_callback: Any = None) -> ConsultationResponse:
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
