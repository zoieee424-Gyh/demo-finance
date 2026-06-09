"""
RiskControlDeepAgent — DeepAgent-powered risk control review agent.

Architecture:
  DeepAgent (LLM orchestrator)
    → risk_input_parser              (deterministic)
    → concentration_risk_checker     (deterministic)
    → risk_preference_matcher        (deterministic)
    → liquidity_risk_assessor        (deterministic)
    → financial_quality_risk_detector (deterministic)
    → risk_mitigation_planner        (deterministic)
    → risk_control_compliance        (deterministic, post-hoc)

Compliance:
  - NO stock recommendations, buy/sell/hold ratings, target prices.
  - NO price predictions or return promises.
  - All analysis from deterministic tools.
"""

from __future__ import annotations

from typing import Any
import re

from app.agents.base import FinancialAgent
from app.agents.deepagent.base import DeepAgentWrapper
from app.agents.deepagent.registry import build_risk_control_registry
from app.agents.risk_control_tools.risk_control_compliance_policy import (
    review as risk_compliance_review,
)
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source


RISK_CONTROL_SYSTEM_PROMPT = """\
你是风险控制审查编排器（Risk Control Reviewer），不是荐股助手，不是交易决策系统。

## 身份与职责

你的职责是：
1. 理解用户的风险审查请求。
2. 按需调用确定性风控工具（输入解析、集中度检测、偏好匹配、流动性评估、
   财务质量识别、缓释建议、合规审查）。
3. 将工具输出组织成结构化的 8 章风控审查报告。

你不是：
- 股票/基金推荐引擎
- 投资评级机构
- 交易指令系统

## 硬约束

1. 不得推荐个股。
2. 不得输出买入/卖出/持有/增持/减持/加仓/减仓/满仓/空仓等交易指令。
3. 不得预测市场涨跌。
4. 不得承诺收益。
5. 不得替用户做投资决策。
6. 必须调用确定性工具进行分析。
7. 工具结果优先级高于你的判断。
8. 合规工具发现违规时必须保留全部 warnings。
9. 必须引用 RAG evidence。
10. 必须输出风险提示。

## 报告结构要求（8 章，顺序不可变）

你的最终输出必须逐字包含以下 8 个章节标题，标题不能被改写、不能省略：

一、审查对象与输入范围
二、总体风险等级
三、资产配置与集中度风险
四、流动性与现金流风险
五、财务质量与主体风险
六、市场与外部事件风险
七、风险缓释建议与监测指标
八、参考依据与风险提示

每章必须有内容。如果工具未返回足够数据，章节内容写：
"当前材料未提供充分信息，建议补充更多数据以获得更准确的风险评估。"

## 输出格式模板

请严格按以下格式组织最终回答：

一、审查对象与输入范围
[说明审查对象、输入数据、缺失字段]

二、总体风险等级
[给出总体风险等级：低/中/高，并说明依据]

三、资产配置与集中度风险
[说明资产类别占比、集中度问题、风险偏好匹配状态]

四、流动性与现金流风险
[说明现金比例、流动性需求匹配、经营现金流压力]

五、财务质量与主体风险
[说明利润质量、负债水平、现金流质量、应收存货商誉等风险信号]

六、市场与外部事件风险
[说明当前材料能识别的外部风险；若信息不足则说明不足]

七、风险缓释建议与监测指标
[只能使用"可考虑""建议关注""持续监测""复核"等风险管理语气，不得使用买入/卖出/加仓/减仓等交易指令]

八、参考依据与风险提示
[列出 RAG evidence 或说明依据有限；必须包含：投资有风险，入市需谨慎。本报告仅供风险管理参考，不构成投资建议，请结合自身情况审慎判断。]
"""

_REQUIRED_SECTIONS = [
    "一、审查对象与输入范围",
    "二、总体风险等级",
    "三、资产配置与集中度风险",
    "四、流动性与现金流风险",
    "五、财务质量与主体风险",
    "六、市场与外部事件风险",
    "七、风险缓释建议与监测指标",
    "八、参考依据与风险提示",
]

_RISK_DISCLAIMERS = [
    "投资有风险",
    "不构成投资",
    "风险管理参考",
    "请结合自身情况",
    "审慎判断",
]

_FORBIDDEN = [
    (r"推荐买入|建议买入|强烈推荐", "买入推荐"),
    (r"推荐卖出|建议卖出|建议清仓", "卖出建议"),
    (r"加仓|减仓|满仓|空仓|建仓|平仓", "仓位操作指令"),
    (r"买入评级|卖出评级|增持评级|减持评级|持有评级", "投资评级"),
    (r"目标价\s*\d+", "目标价"),
    (r"(一定|肯定|势必|必然)(?:涨|跌)", "确定性预测"),
    (r"保证[收益获利]|稳赚", "收益承诺"),
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "股票代码"),
]


def validate_risk_control_output(answer: str) -> tuple[bool, list[str]]:
    """Hard-validate risk control analysis output."""
    failures: list[str] = []

    missing = [s for s in _REQUIRED_SECTIONS if s not in answer]
    if missing:
        failures.append(f"Missing required sections: {', '.join(missing)}")

    has_disclaimer = any(p in answer for p in _RISK_DISCLAIMERS)
    if not has_disclaimer:
        failures.append("Missing risk disclaimer")

    for pattern, desc in _FORBIDDEN:
        if re.search(pattern, answer):
            failures.append(f"Forbidden content: {desc}")

    return (len(failures) == 0, failures)


_FALLBACK_CONTENT = "当前材料未提供充分信息，建议补充更多数据以获得更准确的风险评估。"
_RISK_DISCLAIMER_TEXT = (
    "投资有风险，入市需谨慎。"
    "本报告仅供风险管理参考，不构成投资建议，请结合自身情况审慎判断。"
)


def repair_risk_control_output(raw_answer: str) -> str:
    """Deterministic repair: inject raw content into 8-section skeleton.

    If the DeepAgent output is missing required sections, this function
    wraps it in the correct skeleton rather than discarding it entirely.
    Missing sections get a fallback message.
    """
    # Extract any content between known section markers
    section_content: dict[str, str] = {}
    remaining = raw_answer

    for i, section in enumerate(_REQUIRED_SECTIONS):
        start_idx = remaining.find(section)
        if start_idx >= 0:
            next_section = _REQUIRED_SECTIONS[i + 1] if i + 1 < len(_REQUIRED_SECTIONS) else None
            content_start = start_idx + len(section)
            if next_section:
                end_idx = remaining.find(next_section, content_start)
                if end_idx >= 0:
                    section_content[section] = remaining[content_start:end_idx].strip()
                    remaining = remaining[end_idx:]
                else:
                    section_content[section] = remaining[content_start:].strip()
            else:
                section_content[section] = remaining[content_start:].strip()

    # If no sections found at all, use raw answer as-is and wrap in skeleton
    if not section_content:
        lines = [raw_answer.strip(), "", "---", ""]
    else:
        lines = []

    # Build skeleton
    for section in _REQUIRED_SECTIONS:
        lines.append(section)
        if section in section_content:
            content = section_content[section].strip()
            lines.append(content if content else _FALLBACK_CONTENT)
        else:
            lines.append(_FALLBACK_CONTENT)
        lines.append("")

    # Ensure risk disclaimer in section 8
    if _RISK_DISCLAIMER_TEXT not in "\n".join(lines):
        lines.append(_RISK_DISCLAIMER_TEXT)
        lines.append("")

    return "\n".join(lines)


class RiskControlDeepAgent(FinancialAgent):
    """DeepAgent-powered risk control review agent."""

    name = "risk_control_deepagent"
    intent = "risk_control"

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
            from app.agents.risk_control import RiskControlAgent
            fallback_agent = RiskControlAgent()

        self._fallback_agent = fallback_agent
        self._registry = build_risk_control_registry()

        self._wrapper = DeepAgentWrapper(
            tool_registry=self._registry,
            legacy_agent=self._fallback_agent,
            system_prompt=RISK_CONTROL_SYSTEM_PROMPT,
            agent_name="risk_control_deepagent",
            agent_name_cn="DeepAgent风控审查编排器",
            enabled=self._enabled,
            intent="risk_control",
            output_validator=validate_risk_control_output,
            compliance_review_func=risk_compliance_review,
            repair_func=repair_risk_control_output,
            risk_notice_default=(
                "投资有风险，入市需谨慎。"
                "本报告仅作风险管理参考，不构成任何投资建议。"
                "风险分析基于有限输入信息，可能未覆盖全部风险因子，"
                "请结合自身情况独立判断。"
            ),
        )

    def answer(self, request: ConsultationRequest, sources: list[Source], event_callback: Any = None) -> ConsultationResponse:
        """Answer. Repair is handled inside DeepAgentWrapper if validation fails."""
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
