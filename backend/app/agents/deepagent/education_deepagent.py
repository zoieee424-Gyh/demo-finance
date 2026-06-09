"""
FinancialEducationDeepAgent — DeepAgent-powered financial education agent.

Architecture:
  DeepAgent (LLM orchestrator)
    → learner_profile_analyzer       (deterministic)
    → concept_explainer              (deterministic)
    → learning_path_planner          (deterministic)
    → scam_risk_detector             (deterministic)
    → product_knowledge_mapper       (deterministic)
    → education_evidence_builder     (deterministic)
    → education_compliance_policy    (deterministic, post-hoc guaranteed)

Compliance:
  - Education only — NOT investment advice.
  - No stock/fund code recommendations.
  - No return promises or price predictions.
  - No replacing user investment decisions.
  - All analysis from deterministic tools.
"""

from __future__ import annotations

from typing import Any
import re

from app.agents.base import FinancialAgent
from app.agents.deepagent.base import DeepAgentWrapper
from app.agents.deepagent.registry import build_financial_education_registry
from app.agents.education_tools.education_compliance_policy import (
    review as education_compliance_review,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


EDUCATION_SYSTEM_PROMPT = """\
你是金融科普与投资者教育智能体（Financial Education Agent），不是投资顾问，不是荐股助手。

## 身份与职责

你的职责是：
1. 理解用户的金融学习需求。
2. 按需调用确定性投教工具（学习者画像分析、概念解释、学习路径规划、
   诈骗风险识别、产品知识映射、证据构建、合规审查）。
3. 将工具输出组织成结构化的 7 章投教报告。

你不是：
- 投资顾问。
- 股票/基金推荐引擎。
- 投资决策系统。
- 市场预测模型。

## 硬约束（违反任一条即为失败）

1. 你是金融科普与投资者教育智能体，不是投资顾问。
2. 只做知识解释、风险教育、学习路径规划。
3. 不推荐具体股票、基金代码、理财产品、保险产品。
4. 不承诺收益。
5. 不预测涨跌。
6. 不替用户做投资决策。
7. 涉及投资相关内容必须提示"仅供学习，不构成投资建议"。
8. 必须尽量引用 RAG evidence（如果有）。
9. 用户表达投资冲动、被骗风险、高收益诱导时，优先进行风险教育和防诈骗提示。
10. 输出必须通俗、分层、适合用户知识水平。
11. 工具结果优先级高于你自身的判断——工具给出的知识水平、学习目标必须如实反映。
12. 合规工具（education_compliance_policy）发现违规时必须保留全部 warnings。

## 报告结构要求（7 章，顺序不可变）

你的最终输出必须逐字包含以下 7 个章节标题，标题不能被改写、不能省略：

一、问题理解与学习目标
二、用户知识水平判断
三、核心概念通俗解释
四、关键风险与常见误区
五、学习路径建议
六、参考依据与延伸阅读
七、风险提示与适用边界

每章必须有内容。如果工具未返回足够数据，章节内容写：
"当前材料未提供充分信息，建议补充更多学习需求以获取针对性帮助。"

## 输出格式模板

请严格按以下格式组织最终回答：

一、问题理解与学习目标
[理解用户学习需求和学习目标]

二、用户知识水平判断
[判断用户当前知识水平和学习偏好]

三、核心概念通俗解释
[针对用户问题的核心金融概念进行通俗化解释]

四、关键风险与常见误区
[指出相关风险和常见误解]

五、学习路径建议
[提供渐进式学习路径和后续学习建议]

六、参考依据与延伸阅读
[引用知识库参考资料和延伸阅读建议]

七、风险提示与适用边界
本内容仅供金融知识学习和投资者教育参考，不构成任何投资建议、产品推荐或投资决策依据。投资有风险，入市需谨慎。请根据自身风险承受能力独立判断，必要时咨询专业投资顾问。
"""

# ── Required 7 sections ───────────────────────────────────────────

_REQUIRED_SECTIONS = [
    "一、问题理解与学习目标",
    "二、用户知识水平判断",
    "三、核心概念通俗解释",
    "四、关键风险与常见误区",
    "五、学习路径建议",
    "六、参考依据与延伸阅读",
    "七、风险提示与适用边界",
]

# ── Risk disclaimer patterns ──────────────────────────────────────

_RISK_DISCLAIMER_PATTERNS = [
    "不构成投资建议",
    "投资有风险",
    "入市需谨慎",
    "仅供学习",
    "仅供金融知识学习",
    "投资者教育参考",
]

# ── Forbidden content patterns ────────────────────────────────────

_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    # (regex, description)
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "A股股票代码"),
    (r"基金代码\s*\d{6}", "具体基金代码"),
    (r"推荐买入|建议买入|强烈推荐买入", "个股买入推荐"),
    (r"推荐卖出|建议卖出|建议清仓", "个股卖出建议"),
    (r"保证[收益获利]|保证.*年化", "收益承诺"),
    (r"稳赚不赔|包赚|必赚|只赚不赔|绝不亏损", "绝对盈利承诺"),
    (r"(?:一定|肯定|势必|必然|绝对).{0,2}(?:涨|跌|上涨|下跌)", "确定性涨跌预测"),
    (r"目标价\s*\d+", "目标价预测"),
]

# ── Fallback content placeholder ──────────────────────────────────

_FALLBACK_CONTENT = "当前材料未提供充分信息，建议补充更多学习需求以获取针对性帮助。"

_RISK_NOTICE_DEFAULT = (
    "本内容仅供金融知识学习和投资者教育参考，不构成任何投资建议、"
    "产品推荐或投资决策依据。投资有风险，入市需谨慎。"
    "请根据自身风险承受能力独立判断，必要时咨询专业投资顾问。"
)

# ── Agent-specific system prompt footer ──────────────────────────

_EDUCATION_SYSTEM_PROMPT_FOOTER = """\
## 报告格式要求

最终回答必须是完整的 7 章节中文投教报告（顺序不可变，章节不可缺失）：

一、问题理解与学习目标
二、用户知识水平判断
三、核心概念通俗解释
四、关键风险与常见误区
五、学习路径建议
六、参考依据与延伸阅读
七、风险提示与适用边界

每章必须有内容。如果工具未返回足够数据，章节内容写：
"当前材料未提供充分信息，建议补充更多学习需求以获取针对性帮助。"

在报告末尾附上完整的风险提示（包含"不构成投资建议"）。
"""


# ── Agent-specific user message builder ───────────────────────────

def _build_education_user_message(
    request: ConsultationRequest,
    sources: list[Source],
) -> str:
    """Build the user message for the education DeepAgent.

    This replaces the default advisor-centric instructions with
    education-specific tool calling instructions.
    """
    parts = [f"用户咨询问题：{request.question}"]

    if request.user_profile:
        import json
        profile_str = json.dumps(request.user_profile, ensure_ascii=False)
        parts.append(f"用户画像信息：{profile_str}")

    if sources:
        parts.append("RAG 检索到的参考资料：")
        for i, src in enumerate(sources, 1):
            conf = f"{src.confidence:.0%}" if src.confidence is not None else "N/A"
            parts.append(f"  {i}. {src.title}（置信度：{conf}，类型：{src.source_type}）")

    parts.append("")
    parts.append("## 工具调用策略（按问题类型选择最少工具）")
    parts.append("")
    parts.append("【最高优先】简单概念解释（什么是XX？XX是什么意思？解释XX）")
    parts.append("  → 必须用 simple_concept_workflow，只用 1 个工具！")
    parts.append("  → 该工具返回 draft_sections，直接基于它生成最终 7 章报告。")
    parts.append("  → 调用 simple_concept_workflow 后，禁止再调用任何其他工具。")
    parts.append("")
    parts.append("涉及诈骗/骗局/保证收益/稳赚 → 只用 3 个: learner_profile_analyzer, scam_risk_detector, education_compliance_policy")
    parts.append("涉及多个产品类别比较 → 只用 4 个: learner_profile_analyzer, product_knowledge_mapper, concept_explainer, education_compliance_policy")
    parts.append("问学习路径/入门方法 → 只用 4 个: learner_profile_analyzer, concept_explainer, learning_path_planner, education_compliance_policy")
    parts.append("以上不匹配的综合问题 → 最多 5 个工具")
    parts.append("")
    parts.append("硬性规则：每个工具最多 1 次。education_compliance_policy 必须最后调用（除非用了 simple_concept_workflow）。调完工具立即输出 7 章报告。禁止重复调用。")

    return "\n".join(parts)


# ── Validator ─────────────────────────────────────────────────────

def _find_section(answer: str, section_title: str) -> bool:
    """Check if section_title exists in answer, allowing Markdown prefixes.

    Education LLM output often uses `## 一、标题` or `**一、标题**` format.
    This function matches both raw section titles and Markdown-decorated ones.
    """
    if section_title in answer:
        return True
    # Allow Markdown heading or bold prefixes
    for prefix in ("## ", "### ", "**"):
        if prefix + section_title in answer:
            return True
    return False


def validate_education_output(answer: str) -> tuple[bool, list[str]]:
    """Hard-validate education agent output.

    IMPORTANT: Education content legitimately discusses scam language,
    return promises, and price predictions as EDUCATIONAL COUNTER-EXAMPLES.
    This validator focuses on STRUCTURAL integrity (7 sections + disclaimer)
    and delegates content-policy compliance to the education_compliance_policy
    tool (called post-hoc by DeepAgentWrapper).

    Returns:
        (is_valid, failure_reasons).
    """
    failures: list[str] = []

    # ── 1. Required sections (allow Markdown prefixes) ────────
    missing = [s for s in _REQUIRED_SECTIONS if not _find_section(answer, s)]
    if missing:
        failures.append(f"Missing required sections: {', '.join(missing)}")

    # ── 2. Risk disclaimer ────────────────────────────────────
    has_disclaimer = any(p in answer for p in _RISK_DISCLAIMER_PATTERNS)
    if not has_disclaimer:
        failures.append("Missing risk disclaimer（缺少'不构成投资建议'类风险提示）")

    # ── 3. Forbidden content: only check for actual stock/fund CODES ──
    # Education content may discuss scam language, return promises, and
    # price predictions as counter-examples. Regex is too blunt to
    # distinguish "quoting a scam" from "making a promise".
    # Delegate content-policy compliance to education_compliance_policy tool.
    _STRUCTURAL_FORBIDDEN = [
        (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "A股股票代码"),
        (r"基金代码\s*\d{6}", "具体基金代码"),
    ]
    for pattern, desc in _STRUCTURAL_FORBIDDEN:
        if re.search(pattern, answer):
            failures.append(f"Forbidden content: {desc}")

    return (len(failures) == 0, failures)


# ── Repair ────────────────────────────────────────────────────────

def repair_education_output(raw_answer: str) -> str:
    """Deterministic repair: inject raw content into 7-section skeleton.

    Extracts content between recognized section headers (allowing Markdown
    prefixes like `## ` or `**`) and fills missing sections with fallback
    text. Ensures risk disclaimer is present.
    """
    section_content: dict[str, str] = {}
    remaining = raw_answer

    for i, section in enumerate(_REQUIRED_SECTIONS):
        # Find section with possible Markdown prefix
        idx = -1
        matched_len = 0
        for prefix in ("", "## ", "### ", "**"):
            candidate = prefix + section
            pos = remaining.find(candidate)
            if pos >= 0 and (idx < 0 or pos < idx):
                idx = pos
                matched_len = len(candidate)
        if idx < 0:
            # Try case-insensitive
            remaining_lower = remaining.lower()
            section_lower = section.lower()
            pos = remaining_lower.find(section_lower)
            if pos >= 0:
                idx = pos
                matched_len = len(section)

        if idx >= 0:
            nxt_section = _REQUIRED_SECTIONS[i + 1] if i + 1 < len(_REQUIRED_SECTIONS) else None
            start = idx + matched_len
            if nxt_section:
                # Find next section with any Markdown prefix
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
        # No recognized sections — wrap the whole answer
        lines = [raw_answer.strip(), "", "---", ""]
    else:
        lines = []

    for section in _REQUIRED_SECTIONS:
        lines.append(section)
        lines.append(section_content.get(section, _FALLBACK_CONTENT))
        lines.append("")

    # Ensure risk disclaimer is present
    combined = "\n".join(lines)
    if _RISK_NOTICE_DEFAULT not in combined:
        lines.append(_RISK_NOTICE_DEFAULT)
        lines.append("")

    return "\n".join(lines)


# ── DeepAgent class ───────────────────────────────────────────────

class FinancialEducationDeepAgent(FinancialAgent):
    """DeepAgent-powered financial education agent.

    Provides:
      - 7 deterministic education tools
      - DeepAgent-first with pipeline fallback
      - 7-chapter structured education report
      - Hard output validation + deterministic repair
      - Post-hoc compliance review
    """

    name = "financial_education_deepagent"
    intent = "education"

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
            from app.agents.education import EducationAgent
            fallback_agent = EducationAgent()

        self._fallback_agent = fallback_agent
        self._registry = build_financial_education_registry()

        self._wrapper = DeepAgentWrapper(
            tool_registry=self._registry,
            legacy_agent=self._fallback_agent,
            system_prompt=EDUCATION_SYSTEM_PROMPT,
            agent_name="financial_education_deepagent",
            agent_name_cn="DeepAgent金融科普编排器",
            enabled=self._enabled,
            intent="education",
            output_validator=validate_education_output,
            compliance_review_func=education_compliance_review,
            repair_func=repair_education_output,
            risk_notice_default=_RISK_NOTICE_DEFAULT,
            user_message_builder=_build_education_user_message,
            system_prompt_footer=_EDUCATION_SYSTEM_PROMPT_FOOTER,
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
        info["agent_name_cn"] = "DeepAgent金融科普编排器"
        info["configured_mode"] = self._configured_mode
        return info
