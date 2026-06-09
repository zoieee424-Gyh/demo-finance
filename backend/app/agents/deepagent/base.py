"""
DeepAgent wrapper — unified interface for DeepAgent-powered financial agents.

Provides:
  - DeepAgentWrapper: orchestrates a DeepAgent with deterministic tools.
  - Automatic fallback to legacy pipeline agent when deepagents is unavailable.
  - Tool invocation tracing for auditability.
  - Hard output validation with automatic fallback on compliance failure.

The wrapper is MODEL-AGNOSTIC: it can use any LangChain-compatible chat model
(currently deepseek-v4-flash via langchain-deepseek).

IMPORTANT: The DeepAgent orchestrates tool calls but MUST NOT make financial
decisions. Risk levels, allocation ratios, compliance verdicts — all come from
deterministic tools, never from the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import traceback
from typing import Any

from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Vendor path injection — ensures deepagents is importable
# regardless of user site-packages configuration.
# ═══════════════════════════════════════════════════════════════════

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor")
_VENDOR_DIR = os.path.normpath(_VENDOR_DIR)
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)


# ═══════════════════════════════════════════════════════════════════
# Fallback error sentinel
# ═══════════════════════════════════════════════════════════════════

class DeepAgentNotAvailableError(Exception):
    """Raised when deepagents cannot be initialised and no fallback is set."""


# ═══════════════════════════════════════════════════════════════════
# System prompt for the Investment Advisor DeepAgent
# ═══════════════════════════════════════════════════════════════════

INVESTMENT_ADVISOR_SYSTEM_PROMPT = """\
你是智能投顾编排器（Investment Advisor Orchestrator），不是荐股助手。

## 身份与职责

你的职责是：
1. 理解用户的金融咨询问题。
2. 按需调用确定性金融工具（用户画像分析、目标规划、风险评估、资产配置、
   基金定投、持仓诊断、市场热点解读、合规审查、RAG 证据构建）。
3. 将工具输出组织成结构化的投顾分析报告（9 个章节，含风险提示和依据来源）。

你不是：
- 股票/基金推荐引擎。
- 市场预测模型。
- 投资决策系统。
- 收益承诺机器。

## 硬约束（违反任一条即为失败）

1. 你必须调用确定性工具获得风险等级和资产配置比例。
2. 你不能自行生成风险等级。
3. 你不能自行生成资产配置比例。
4. 你不能推荐个股。
5. 你不能推荐具体基金代码或具体基金产品名称。
6. 你不能承诺收益。
7. 你不能预测市场涨跌。
8. 你不能替用户做最终投资决策。
9. 你必须引用 RAG evidence（如果有）并在报告中注明依据来源。
10. 你必须输出风险提示（"投资有风险，入市需谨慎"）。
11. 工具结果优先级高于你自身的判断——工具说保守型就是保守型，不得质疑或覆盖。
12. 合规工具（advisory_compliance_policy）发现违规时，你必须保留全部 warnings，
    并在报告末尾展示。不得忽略、过滤或弱化合规警告。
13. 资产配置只输出资产类别（如"宽基指数类""债券类""现金类"），
    不得输出具体股票代码、基金代码、基金名称或购买平台。

## 报告结构要求

最终回答必须包含以下 9 个章节（顺序不可变，章节不可缺失）：

一、用户画像摘要
二、投资目标分析
三、风险评估结果
四、资产配置建议
五、基金定投规划
六、持仓诊断
七、市场热点解读
八、参考依据与适用边界
九、风险提示

## 工作流程

1. 先调用 profile_analyzer 解析用户画像。
2. 调用 goal_planner 识别投资目标。
3. 调用 risk_assessor 评估风险承受力。
4. 调用 allocation_engine 生成资产配置。
5. 调用 fund_dca_planner 生成定投方案。
6. 如用户提供了持仓数据，调用 holding_diagnostic 诊断。
7. 如问题涉及市场热点，调用 market_hotspot_interpreter 解读。
8. 调用 evidence_builder 构建 RAG 证据。
9. 将上述结果整合为 9 章报告。
10. 调用 advisory_compliance_policy 审查整份报告。
11. 如合规审查通过，输出报告；如有违规，保留 warnings 并输出。
"""


# ═══════════════════════════════════════════════════════════════════
# Hard output validation constants
# ═══════════════════════════════════════════════════════════════════

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

_RISK_DISCLAIMER_PATTERNS = [
    "投资有风险",
    "入市需谨慎",
    "不构成投资",
    "投资需谨慎",
]

_FORBIDDEN_CONTENT_PATTERNS: list[tuple[str, str]] = [
    # (regex, description)
    (r"\b\d{6}\b", "6-digit stock/fund code"),
    (r"推荐买入", "stock buy recommendation"),
    (r"建议买入", "stock buy recommendation"),
    (r"保证收益", "return promise"),
    (r"年化收益.*%", "specific return percentage promise"),
    (r"肯定会涨", "definite price rise prediction"),
    (r"势必下跌", "definite price drop prediction"),
    (r"稳赚不赔", "guaranteed profit claim"),
    (r"本人建议(您|你)买入", "agent making buy decision for user"),
    (r"建议(您|你)立即卖出", "agent making sell decision for user"),
    (r"贵州茅台|宁德时代|比亚迪|隆基绿能", "named stock recommendation"),
]


def validate_deepagent_output(answer: str) -> tuple[bool, list[str]]:
    """Hard-validate DeepAgent output for structural and compliance integrity.

    Returns:
        (is_valid, failure_reasons)
    """
    failures: list[str] = []

    # 1. Check all 9 required sections present
    missing_sections = [
        s for s in _REQUIRED_REPORT_SECTIONS if s not in answer
    ]
    if missing_sections:
        failures.append(
            f"Missing required sections: {', '.join(missing_sections)}"
        )

    # 2. Check risk disclaimer present
    has_disclaimer = any(p in answer for p in _RISK_DISCLAIMER_PATTERNS)
    if not has_disclaimer:
        failures.append("Missing risk disclaimer")

    # 3. Check for forbidden content
    for pattern, desc in _FORBIDDEN_CONTENT_PATTERNS:
        if re.search(pattern, answer):
            failures.append(f"Forbidden content detected: {desc}")

    is_valid = len(failures) == 0
    return is_valid, failures


# ═══════════════════════════════════════════════════════════════════
# Tool call trace entry
# ═══════════════════════════════════════════════════════════════════

class ToolCallTrace:
    """Record of a single tool invocation by the DeepAgent."""

    __slots__ = (
        "tool_id", "name_cn", "input_keys", "success",
        "error_message", "output_preview", "elapsed_ms", "_start_ts",
    )

    def __init__(
        self,
        tool_id: str = "",
        name_cn: str = "",
        input_keys: list[str] | None = None,
        success: bool = True,
        error_message: str | None = None,
        output_preview: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.tool_id = tool_id
        self.name_cn = name_cn
        self.input_keys = input_keys or []
        self.success = success
        self.error_message = error_message
        self.output_preview = output_preview
        self.elapsed_ms = elapsed_ms

    def to_dict(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "name_cn": self.name_cn,
            "input_keys": self.input_keys,
            "success": self.success,
            "error_message": self.error_message,
            "output_preview": self.output_preview,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ═══════════════════════════════════════════════════════════════════
# DeepAgentWrapper
# ═══════════════════════════════════════════════════════════════════

class DeepAgentWrapper:
    """Unified wrapper around a DeepAgent-powered financial agent.

    This wrapper:
      - Accepts a ToolRegistry and a legacy agent as fallback.
      - Optionally accepts a LangChain chat model for the DeepAgent.
      - Attempts to use deepagents.create_deep_agent when enabled.
      - Falls back to the legacy pipeline agent when deepagents is unavailable
        or when explicitly disabled.
      - Records tool call traces for auditability.
      - Hard-validates DeepAgent output and auto-falls-back on failure.

    Lifecycle:
      1. __init__ → register tools, set up model, optionally create deep agent.
      2. run(request, sources) → ConsultationResponse.
    """

    def __init__(
        self,
        *,
        tool_registry: Any = None,  # ToolRegistry
        legacy_agent: Any = None,   # FinancialAgent (legacy fallback)
        system_prompt: str = "",
        agent_name: str = "deepagent",
        agent_name_cn: str = "DeepAgent智能体",
        enabled: bool = False,
        model: Any = None,  # BaseChatModel | None
        intent: str = "advisory",
        output_validator: Any = None,        # Callable[[str], tuple[bool, list[str]]] | None
        compliance_review_func: Any = None,  # Callable[[str], dict] | None
        risk_notice_default: str = "",
        repair_func: Any = None,             # Callable[[str], str] | None
        user_message_builder: Any = None,    # Callable[[ConsultationRequest, list[Source]], str] | None
        system_prompt_footer: str = "",      # Agent-specific footer appended after tool list
    ) -> None:
        """
        Args:
            tool_registry: ToolRegistry with registered DeepAgentTools.
            legacy_agent: Fallback agent (must implement answer(request, sources)).
            system_prompt: System prompt for the DeepAgent orchestrator.
            agent_name: Internal identifier for this agent.
            agent_name_cn: Chinese name for logging and debug display.
            enabled: Whether to attempt real DeepAgent creation.
            model: LangChain BaseChatModel (if None, will try to create one).
            intent: One of the five financial intents.
            output_validator: Callable(answer_text) → (is_valid, failures).
            compliance_review_func: Callable(answer_text) → dict with
                warnings, risk_notice, is_compliant.
            risk_notice_default: Default risk notice.
            repair_func: Optional Callable(raw_answer) → repaired_answer.
                If validator fails, repair is attempted before fallback.
            user_message_builder: Optional callable(request, sources) → str.
                If provided, used instead of the default advisor-centric
                _build_user_message. Agent-specific agents SHOULD pass this.
            system_prompt_footer: Optional agent-specific footer appended
                after tool descriptions (e.g., report structure requirements).
                If empty, the default advisor footer is used.
        """
        self.tool_registry = tool_registry
        self.legacy_agent = legacy_agent
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.agent_name_cn = agent_name_cn
        self.enabled = enabled
        self.model = model
        self.intent = intent
        self.output_validator = output_validator
        self.compliance_review_func = compliance_review_func
        self.risk_notice_default = risk_notice_default
        self.repair_func = repair_func
        self._user_message_builder = user_message_builder
        self._system_prompt_footer = system_prompt_footer

        # Internal state (init-time)
        self._deep_agent_graph: Any = None
        self._deepagent_available: bool = False
        self._fallback_reason: str = ""    # init-time fallback reason
        self._tool_traces: list[ToolCallTrace] = []

        # Internal state (runtime — reset each run())
        self._runtime_fallback_used: bool = False
        self._runtime_fallback_reason: str | None = None
        self._last_actual_architecture: str = "pipeline"

        # Initialise
        if self.enabled:
            self._init_deep_agent()
        else:
            self._fallback_reason = "DeepAgent mode disabled by configuration"

    # ── Initialisation ──────────────────────────────────────────────

    def _init_deep_agent(self) -> None:
        """Attempt to create a real deepagents-based agent.

        If any step fails, the wrapper stays in fallback mode (no crash).
        """
        # Step 1: check deepagents import
        try:
            from deepagents import create_deep_agent  # noqa: F401
        except ImportError as exc:
            self._fallback_reason = f"deepagents import failed: {exc}"
            logger.info("DeepAgentWrapper: %s — using legacy fallback", self._fallback_reason)
            return

        # Step 2: check tool registry
        if self.tool_registry is None or self.tool_registry.tool_count == 0:
            self._fallback_reason = "no tools registered"
            logger.info("DeepAgentWrapper: %s — using legacy fallback", self._fallback_reason)
            return

        # Step 3: build model if not provided
        if self.model is None:
            try:
                self.model = self._build_model()
            except Exception as exc:
                self._fallback_reason = f"failed to create model: {exc}"
                logger.info("DeepAgentWrapper: %s — using legacy fallback", self._fallback_reason)
                return

        # Step 4: build LangChain tools from our registry
        try:
            lc_tools = self._build_langchain_tools()
        except Exception as exc:
            self._fallback_reason = f"failed to build LangChain tools: {exc}"
            logger.info("DeepAgentWrapper: %s — using legacy fallback", self._fallback_reason)
            return

        # Step 5: build system prompt with tool descriptions
        full_system_prompt = self._build_full_system_prompt()

        # Step 6: create the deep agent
        try:
            from deepagents import create_deep_agent

            self._deep_agent_graph = create_deep_agent(
                model=self.model,
                tools=lc_tools,
                system_prompt=full_system_prompt,
            )
            self._deepagent_available = True
            self._fallback_reason = ""
            logger.info(
                "DeepAgentWrapper: agent '%s' created successfully with %d tools",
                self.agent_name,
                self.tool_registry.tool_count,
            )
        except Exception as exc:
            self._fallback_reason = f"create_deep_agent failed: {exc}"
            logger.info("DeepAgentWrapper: %s — using legacy fallback", self._fallback_reason)

    def _build_model(self):
        """Build a LangChain chat model for the DeepAgent.

        Uses the same deepseek-v4-flash model as the rest of the system.
        Reads DEEPSEEK_API_KEY from environment as fallback.
        """
        import os
        from langchain_deepseek import ChatDeepSeek
        from app.core.config import settings

        api_key = settings.llm_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        return ChatDeepSeek(
            model=settings.llm_model,
            api_key=api_key,
            temperature=0.2,
        )

    def _build_langchain_tools(self) -> list:
        """Convert our DeepAgentTool registry into LangChain-compatible tools.

        Uses StructuredTool.from_function with explicit args_schema to ensure
        the LLM sees proper parameter names (not just 'kwargs').
        """
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field, create_model

        lc_tools = []

        for dt in self.tool_registry.list_tools():
            # Build description INSIDE the factory to avoid closure bug
            def _make_tool_fn(tool: Any) -> Any:
                desc_parts = [tool.description]
                if tool.hard_constraints:
                    desc_parts.append("硬约束：")
                    for c in tool.hard_constraints:
                        desc_parts.append(f"  - {c}")
                description = "\n".join(desc_parts)

                # Build a dynamic Pydantic model for the tool's input schema
                # so the LLM sees proper parameter names.
                if tool.input_schema:
                    fields: dict[str, Any] = {}
                    for key, key_desc in tool.input_schema.items():
                        # Use str type with description for all fields
                        fields[key] = (str | dict | list | None, Field(default=None, description=key_desc))
                    # Always add a generic input field as catch-all
                    ArgsModel = create_model(
                        f"{tool.tool_id}_args",
                        **fields,
                    )
                else:
                    # Fallback: generic dict input
                    class GenericArgs(BaseModel):
                        input_json: str = Field(default="{}", description="JSON string with all tool arguments")

                    ArgsModel = GenericArgs

                def _execute_wrapper(**kwargs: Any) -> str:
                    """Execute the tool and return JSON result."""
                    self._record_tool_call_start(tool, kwargs)
                    # Filter out None values
                    filtered = {k: v for k, v in kwargs.items() if v is not None}
                    # Record start timestamp for elapsed_ms tracking
                    _start = time.perf_counter()
                    if self._tool_traces:
                        self._tool_traces[-1]._start_ts = _start
                    try:
                        result = tool(**filtered)
                        self._record_tool_call_success(tool, result)
                        return json.dumps(result, ensure_ascii=False, default=str)
                    except Exception as exc:
                        self._record_tool_call_failure(tool, exc)
                        return json.dumps(
                            {"error": str(exc), "tool_id": tool.tool_id},
                            ensure_ascii=False,
                        )

                # Use StructuredTool with dynamic name
                # The function name must match the tool name for the LLM to call it
                _execute_wrapper.__name__ = tool.tool_id

                return StructuredTool.from_function(
                    func=_execute_wrapper,
                    name=tool.tool_id,
                    description=description,
                    args_schema=ArgsModel,
                )

            lc_tools.append(_make_tool_fn(dt))

        return lc_tools

    # ── Tool call trace ─────────────────────────────────────────────

    def _emit_tool_event(self, event: dict) -> None:
        """通过本次 run 注入的回调推送工具事件。

        这里故意把事件回调做成“可选能力”：普通 HTTP/debug 接口不传
        callback 时行为完全不变；SSE 路径传入 callback 后，工具开始、
        成功、失败会实时进入事件队列。回调异常只会丢失时间线事件，
        不能影响主报告生成。
        """
        cb = getattr(self, "_event_callback", None)
        if cb is not None:
            try:
                cb(event)
            except Exception:
                pass  # Never let a callback exception break the agent

    def _record_tool_call_start(self, tool: Any, kwargs: dict) -> None:
        """Record the start of a tool call."""
        self._tool_traces.append(ToolCallTrace(
            tool_id=tool.tool_id,
            name_cn=tool.name_cn,
            input_keys=list(kwargs.keys()),
            success=True,
        ))
        # SSE 时间线：工具刚被 DeepAgent 调用时立即推送。
        self._emit_tool_event({
            "type": "tool_started",
            "stage": "tools",
            "label": "调用工具",
            "message": f"正在调用 {tool.name_cn}",
            "tool_id": tool.tool_id,
            "name_cn": tool.name_cn,
            "input_keys": list(kwargs.keys()),
        })

    def _record_tool_call_success(self, tool: Any, result: Any) -> None:
        """Mark the last trace entry as successful with output preview."""
        output_preview = ""
        if self._tool_traces:
            trace = self._tool_traces[-1]
            trace.success = True
            # 只有 start 阶段记录了 _start_ts 时，才计算工具执行耗时。
            if hasattr(trace, '_start_ts'):
                trace.elapsed_ms = (time.perf_counter() - trace._start_ts) * 1000
            try:
                output_preview = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                output_preview = str(result)
            trace.output_preview = output_preview[:200]
        # SSE 时间线：工具成功完成时推送简短输出预览。
        self._emit_tool_event({
            "type": "tool_done",
            "stage": "tools",
            "label": "工具完成",
            "message": f"{tool.name_cn} — ✓",
            "tool_id": tool.tool_id,
            "name_cn": tool.name_cn,
            "elapsed_ms": round(self._tool_traces[-1].elapsed_ms, 1) if self._tool_traces else 0.0,
            "output_preview": output_preview[:120],
            "success": True,
        })

    def _record_tool_call_failure(self, tool: Any, exc: Exception) -> None:
        """Mark the last trace entry as failed."""
        if self._tool_traces:
            trace = self._tool_traces[-1]
            trace.success = False
            if hasattr(trace, '_start_ts'):
                trace.elapsed_ms = (time.perf_counter() - trace._start_ts) * 1000
            trace.error_message = f"{type(exc).__name__}: {exc}"
        # SSE 时间线：工具异常不会中断主流程，但会让前端明确展示失败。
        self._emit_tool_event({
            "type": "tool_failed",
            "stage": "tools",
            "label": "工具失败",
            "message": f"{tool.name_cn} — ✗",
            "tool_id": tool.tool_id,
            "name_cn": tool.name_cn,
            "elapsed_ms": round(self._tool_traces[-1].elapsed_ms, 1) if self._tool_traces else 0.0,
            "error_message": f"{type(exc).__name__}: {exc}",
            "success": False,
        })

    @property
    def tool_trace_summary(self) -> list[dict]:
        """Return a summary of tool call traces for debug/audit."""
        return [t.to_dict() for t in self._tool_traces]

    # ── System prompt building ──────────────────────────────────────

    def _build_full_system_prompt(self) -> str:
        """Combine the base system prompt with tool registry info."""
        parts = [self.system_prompt]
        if self.tool_registry is not None:
            tool_section = self.tool_registry.build_system_prompt_fragment()
            if tool_section:
                parts.append("")
                parts.append(tool_section)
        parts.append("")
        if self._system_prompt_footer:
            parts.append(self._system_prompt_footer)
        else:
            # Legacy default — investment advisor footer
            parts.append("## 报告格式要求")
            parts.append("最终回答必须是完整的 9 章节中文投顾分析报告。")
            parts.append("在报告末尾附上完整的风险提示。")
        return "\n".join(parts)

    # ── Public API ──────────────────────────────────────────────────

    @property
    def is_deepagent_available(self) -> bool:
        """Whether a real DeepAgent graph is ready to use."""
        return self._deepagent_available and self._deep_agent_graph is not None

    @property
    def fallback_reason(self) -> str:
        """Why the wrapper is in fallback mode (empty string if DeepAgent is active)."""
        return self._fallback_reason

    @property
    def architecture(self) -> str:
        """Architecture label for debug/API metadata."""
        return "deepagent" if self.is_deepagent_available else "pipeline"

    @property
    def debug_info(self) -> dict:
        """Return debug metadata for API responses.

        Includes runtime state (from the last run() call) in addition
        to init-time state.
        """
        # Effective fallback: init-time OR runtime
        effective_fallback = (
            not self.is_deepagent_available or self._runtime_fallback_used
        )
        effective_reason = (
            self._runtime_fallback_reason
            or self._fallback_reason
        )
        effective_architecture = self._last_actual_architecture or (
            "deepagent" if self.is_deepagent_available else "pipeline"
        )
        return {
            "agent_architecture": effective_architecture,
            "deepagent_enabled": self.enabled,
            "deepagent_available": self.is_deepagent_available,
            "fallback_used": effective_fallback,
            "fallback_reason": effective_reason if effective_reason else None,
            "tool_count": self.tool_registry.tool_count if self.tool_registry else 0,
            "tool_traces": self.tool_trace_summary,
        }

    def run(
        self,
        request: ConsultationRequest,
        sources: list[Source],
        event_callback: Any = None,
    ) -> ConsultationResponse:
        """Execute the agent and return a ConsultationResponse.

        If the real DeepAgent is available, it orchestrates tool calls via the LLM.
        Otherwise, it delegates to the legacy pipeline agent.

        DeepAgent output is hard-validated; if validation fails, it auto-falls-back
        to the legacy pipeline.

        Args:
            request: User consultation with question and optional profile.
            sources: Evidence retrieved by KnowledgeRetriever.
            event_callback: Optional callable(dict) for real-time tool events.
                Called on tool_started/tool_done/tool_failed with event payload.
                Set per-run to avoid cached-agent event pollution.

        Returns:
            ConsultationResponse with structured answer, warnings, and risk notice.
        """
        # event_callback 只在本次 run 生命周期内有效；finally 中必须清理，
        # 避免进程级缓存 agent 在下一次请求里沿用旧的 SSE 回调。
        self._event_callback = event_callback

        # ── Reset per-run state ────────────────────────────────
        self._tool_traces.clear()
        self._runtime_fallback_used = False
        self._runtime_fallback_reason = None
        self._last_actual_architecture = (
            "deepagent" if self.is_deepagent_available else "pipeline"
        )
        if self.is_deepagent_available:
            self._fallback_reason = ""

        try:
            if self.is_deepagent_available:
                try:
                    return self._run_deepagent_with_validation(request, sources)
                except Exception as exc:
                    logger.warning(
                        "DeepAgent run failed: %s — falling back to legacy",
                        exc,
                    )
                    self._runtime_fallback_used = True
                    self._runtime_fallback_reason = f"DeepAgent runtime error: {exc}"
                    self._last_actual_architecture = "pipeline"
                    # Fall through to legacy fallback

            # ── Fallback (either init-time or runtime) ─────────────
            if not self.is_deepagent_available and not self._runtime_fallback_used:
                # Init-time fallback — never had a chance to run DeepAgent
                self._runtime_fallback_used = True
                self._runtime_fallback_reason = self._runtime_fallback_reason or self._fallback_reason

            if self.legacy_agent is not None:
                logger.info(
                    "DeepAgentWrapper: delegating to legacy agent '%s' (reason: %s)",
                    getattr(self.legacy_agent, "name", "unknown"),
                    self._runtime_fallback_reason or self._fallback_reason or "deepagent not available",
                )
                return self.legacy_agent.answer(request, sources)

            raise DeepAgentNotAvailableError(
                "No DeepAgent available and no legacy fallback configured. "
                f"Reason: {self._fallback_reason or 'unknown'}"
            )
        finally:
            # 无论 DeepAgent 成功、fallback 还是异常，都清理本次事件回调。
            self._event_callback = None

    def _run_deepagent_with_validation(
        self,
        request: ConsultationRequest,
        sources: list[Source],
    ) -> ConsultationResponse:
        """Run the DeepAgent and validate output. Fallback on failure."""
        # ── Run DeepAgent ────────────────────────────────────────
        if self._user_message_builder is not None:
            user_message = self._user_message_builder(request, sources)
        else:
            user_message = self._build_user_message(request, sources)

        _invoke_start = time.perf_counter()
        logger.info(
            "DeepAgentWrapper: invoking _deep_agent_graph for '%s' (msg_len=%d)",
            self.agent_name, len(user_message),
        )
        try:
            result = self._deep_agent_graph.invoke({
                "messages": [{"role": "user", "content": user_message}],
            })
            _invoke_elapsed = time.perf_counter() - _invoke_start
            logger.info(
                "DeepAgentWrapper: _deep_agent_graph returned in %.1fs (%d tool traces)",
                _invoke_elapsed, len(self._tool_traces),
            )
        except Exception:
            _invoke_elapsed = time.perf_counter() - _invoke_start
            logger.error(
                "DeepAgentWrapper: _deep_agent_graph FAILED after %.1fs",
                _invoke_elapsed, exc_info=True,
            )
            raise

        # Extract answer from messages
        answer_text = self._extract_answer(result)

        # ── Hard validation (agent-specific) ─────────────────────
        if self.output_validator is not None:
            is_valid, failures = self.output_validator(answer_text)
        else:
            # No custom validator → default pass
            is_valid, failures = True, []

        if not is_valid:
            reason = (
                f"DeepAgent output validation failed: {'; '.join(failures)}"
            )
            logger.warning("DeepAgentWrapper: %s", reason)

            # ── Attempt deterministic repair before fallback ─────
            if self.repair_func is not None:
                try:
                    repaired = self.repair_func(answer_text)
                    re_is_valid, re_failures = (
                        self.output_validator(repaired)
                        if self.output_validator else (True, [])
                    )
                    if re_is_valid:
                        logger.info("DeepAgentWrapper: repair succeeded")
                        answer_text = repaired
                        is_valid = True  # proceed to compliance review
                    else:
                        logger.warning(
                            "DeepAgentWrapper: repair also failed: %s",
                            re_failures,
                        )
                except Exception as exc:
                    logger.warning("DeepAgentWrapper: repair exception: %s", exc)

        if not is_valid:
            self._runtime_fallback_used = True
            self._runtime_fallback_reason = reason
            self._last_actual_architecture = "pipeline"
            self._fallback_reason = reason
            # Auto-fallback to legacy
            if self.legacy_agent is not None:
                logger.info(
                    "DeepAgentWrapper: auto-fallback to legacy agent '%s'",
                    getattr(self.legacy_agent, "name", "unknown"),
                )
                return self.legacy_agent.answer(request, sources)
            raise DeepAgentNotAvailableError(
                f"DeepAgent output invalid and no fallback: {reason}"
            )

        # ── Post-hoc compliance review (agent-specific) ──────────
        if self.compliance_review_func is not None:
            compliance = self.compliance_review_func(answer=answer_text)
        else:
            compliance = {"warnings": [], "risk_notice": "", "is_compliant": True}

        risk_notice = compliance.get("risk_notice", "") or self.risk_notice_default

        return ConsultationResponse(
            intent=self.legacy_agent.intent if self.legacy_agent else self.intent,
            agent=self.agent_name,
            answer=answer_text,
            risk_notice=risk_notice,
            sources=sources,
            warnings=compliance.get("warnings", []),
        )

    def _build_user_message(
        self,
        request: ConsultationRequest,
        sources: list[Source],
    ) -> str:
        """Build the user message for the DeepAgent, including sources context."""
        parts = [f"用户咨询问题：{request.question}"]

        if request.user_profile:
            profile_str = json.dumps(request.user_profile, ensure_ascii=False)
            parts.append(f"用户画像信息：{profile_str}")

        if sources:
            parts.append("RAG 检索到的参考资料：")
            for i, src in enumerate(sources, 1):
                conf = f"{src.confidence:.0%}" if src.confidence is not None else "N/A"
                parts.append(f"  {i}. {src.title}（置信度：{conf}，类型：{src.source_type}）")

        parts.append("")
        parts.append("请按以下步骤处理：")
        parts.append("1. 调用 profile_analyzer 解析用户画像。")
        parts.append("2. 调用 goal_planner 识别投资目标。")
        parts.append("3. 调用 risk_assessor 评估风险。")
        parts.append("4. 调用 allocation_engine 生成资产配置。")
        parts.append("5. 调用 fund_dca_planner 生成定投方案。")
        parts.append("6. 如用户提供了持仓，调用 holding_diagnostic。")
        parts.append("7. 如问题涉及市场热点，调用 market_hotspot_interpreter。")
        parts.append("8. 调用 evidence_builder 构建 RAG 证据。")
        parts.append("9. 整合以上结果输出 9 章报告。")
        parts.append("10. 调用 advisory_compliance_policy 审查报告。")

        return "\n".join(parts)

    def _extract_answer(self, result: dict) -> str:
        """Extract the final answer text from a DeepAgent graph result.

        The result is a LangGraph StateGraph output. We look for the last
        AI message in the messages list.
        """
        messages = result.get("messages", [])
        # Find the last AI message
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai":
                content = getattr(msg, "content", "")
                if content:
                    return str(content)
            elif isinstance(msg, dict):
                if msg.get("type") == "ai" and msg.get("content"):
                    return str(msg["content"])
        # Fallback: stringify the whole result
        return str(result)
