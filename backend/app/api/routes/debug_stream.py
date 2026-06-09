"""
SSE streaming debug endpoint for real-time agent execution visibility.

API:
  POST /api/debug/stream/{agent_id}/prepare   → prepare a run
  GET  /api/debug/stream/{run_id}              → SSE event stream

The prepare step avoids GET query-length limits and aligns with
EventSource's GET-only constraint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.rag.evidence import build_evidence_pack
from app.rag.planner import RagPlanner
from app.schemas.consultation import ConsultationRequest, ConsultationResponse, Source
from app.schemas.streaming import (
    PrepareRunRequest,
    PrepareRunResponse,
    VALID_AGENT_IDS,
)
from app.services.agent_router import route_intent
from app.services.compliance_guard import ComplianceGuard
from app.services.consultation_service import _create_retriever
from app.services.streaming_run_store import (
    create_run,
    delete_run,
    get_run,
    mark_run,
)

# ── Extended valid agent IDs (including "auto") ──────────────────
_ALL_VALID_AGENT_IDS = VALID_AGENT_IDS | {"auto"}

logger = logging.getLogger(__name__)
router = APIRouter(tags=["debug-stream"])

_HEARTBEAT_INTERVAL = 5  # seconds


# ── Agent dispatch ───────────────────────────────────────────────

def _get_agent_for_intent(intent: str, mode: str = "deepagent"):
    """Return a cached agent instance for the given intent.

    Uses the same factory pattern as /api/agent/query.
    """
    if intent == "advisory":
        from app.services.advisory_agent_factory import get_advisory_agent
        return get_advisory_agent(mode)
    elif intent == "financial_report":
        from app.services.financial_report_agent_factory import get_financial_report_agent
        return get_financial_report_agent(mode)
    elif intent == "risk_control":
        from app.services.risk_control_agent_factory import get_risk_control_agent
        return get_risk_control_agent(mode)
    elif intent == "compliance":
        from app.services.compliance_agent_factory import get_compliance_agent
        return get_compliance_agent(mode)
    elif intent == "education":
        from app.services.education_agent_factory import get_education_agent
        return get_education_agent(mode)
    else:
        raise ValueError(f"Unknown intent: {intent}")


def _get_agent_for_stream(agent_id: str):
    """Return a cached agent instance for the given agent_id.

    Uses factories where available, lazy-instantiation elsewhere.
    Note: "auto" is NOT supported here — auto-routing is handled
    in the stream generator itself.
    """
    return _get_agent_for_intent(agent_id, settings.advisor_mode if agent_id == "advisory" else "deepagent")


def _agent_intent(agent_id: str) -> str:
    """Map agent_id to intent string for RagPlanner."""
    mapping = {
        "advisory": "advisory",
        "financial_report": "financial_report",
        "risk_control": "risk_control",
        "compliance": "compliance",
        "education": "education",
    }
    return mapping.get(agent_id, "advisory")


# ── SSE helpers ──────────────────────────────────────────────────

def _should_replay_tool_traces(agent) -> bool:
    """Return True if tool traces should be replayed as legacy tool_call events.

    Real-time DeepAgent tool events are emitted via callback. Pipeline fallback
    agents don't call the callback, so we replay traces for those cases.
    """
    arch = getattr(agent, "architecture", "pipeline")
    if arch == "deepagent":
        # Check if the wrapper was actually available
        da = getattr(agent, "_wrapper", None)
        if da is not None and da.is_deepagent_available:
            return False  # Real-time events already emitted
    return True  # Pipeline — replay traces


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sse_event(event_type: str, payload: dict) -> str:
    """Format a single SSE event."""
    payload["timestamp"] = _now_iso()
    payload["type"] = event_type
    data = json.dumps(payload, ensure_ascii=False, default=str)
    return f"event: {event_type}\ndata: {data}\n\n"


# ── Prepare endpoint ─────────────────────────────────────────────

@router.post("/debug/stream/{agent_id}/prepare", response_model=PrepareRunResponse)
def prepare_stream(agent_id: str, body: PrepareRunRequest):
    """Prepare a stream run. Returns run_id and stream_url for SSE connection."""
    if agent_id not in _ALL_VALID_AGENT_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent_id: {agent_id}. Must be one of {sorted(_ALL_VALID_AGENT_IDS)}",
        )

    if not body.question or len(body.question.strip()) < 2:
        raise HTTPException(
            status_code=422,
            detail="question must have at least 2 characters",
        )
    request = ConsultationRequest(
        question=body.question,
        user_profile=body.user_profile or {},
    )
    entry = create_run(agent_id, request)

    return PrepareRunResponse(
        run_id=entry.run_id,
        stream_url=f"/api/debug/stream/{entry.run_id}",
        agent_id=agent_id,
    )


# ── Stream endpoint ──────────────────────────────────────────────

@router.get("/debug/stream/{run_id}")
async def stream_run(run_id: str):
    """SSE event stream for a prepared run.

    Event sequence:
      run_started → intent_selected → retrieval_started → retrieval_done
      → agent_started → heartbeat(s) → tool_call(s) → [fallback_used]
      → report_done → run_done (or run_error)
    """
    entry = get_run(run_id)
    if entry is None:
        return StreamingResponse(
            _error_stream(run_id, f"Run '{run_id}' not found or expired"),
            media_type="text/event-stream",
        )

    mark_run(run_id, "streaming")
    return StreamingResponse(
        _run_stream(entry),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _error_stream(run_id: str, message: str):
    """A single-event error stream."""
    yield sse_event("run_error", {
        "run_id": run_id,
        "stage": "error",
        "label": "运行失败",
        "message": message,
    })
    yield sse_event("run_done", {"run_id": run_id, "label": "结束"})


async def _run_stream(entry):
    """Main SSE generator — orchestrates the full agent run with heartbeat.

    When agent_id="auto", the system uses route_intent() to determine
    the best agent based on the question content, then emits
    intent_selected with full router metadata.
    """
    run_id = entry.run_id
    agent_id = entry.agent_id
    request = entry.request

    try:
        # ── Phase 1: Init & Routing ─────────────────────────────
        yield sse_event("run_started", {
            "run_id": run_id, "stage": "init", "label": "已接收任务",
            "message": f"智能体: {agent_id}",
        })

        # ── Auto-routing when agent_id is "auto" ────────────────
        router_decision = None
        if agent_id == "auto":
            router_decision = route_intent(request.question, request.user_profile)
            intent = router_decision.selected_intent
            yield sse_event("intent_selected", {
                "run_id": run_id, "stage": "routing", "label": "自动识别智能体",
                "message": f"路由至: {intent}（置信度: {router_decision.confidence:.0%}）",
                "router": router_decision.model_dump(mode="json"),
            })
        else:
            intent = _agent_intent(agent_id)
            yield sse_event("intent_selected", {
                "run_id": run_id, "stage": "routing", "label": "确认智能体",
                "message": f"意图: {intent}",
            })

        # ── Phase 2: Retrieval ──────────────────────────────────
        yield sse_event("retrieval_started", {
            "run_id": run_id, "stage": "retrieval", "label": "检索知识库",
            "message": "正在检索相关知识",
        })

        planner = RagPlanner()
        retriever = _create_retriever()
        planned_queries = planner.build_queries(request, intent)
        sources = retriever.retrieve(planned_queries)

        yield sse_event("retrieval_done", {
            "run_id": run_id, "stage": "retrieval", "label": "检索完成",
            "message": f"找到 {len(sources)} 条参考依据",
        })

        # ── Phase 3: Agent ──────────────────────────────────────
        # When auto-routing, use the resolved intent; otherwise use agent_id directly
        resolved_id = intent if agent_id == "auto" else agent_id
        agent = _get_agent_for_stream(resolved_id)
        guard = ComplianceGuard()
        evidence_pack = build_evidence_pack(sources)

        yield sse_event("agent_started", {
            "run_id": run_id, "stage": "agent", "label": "启动智能体",
            "message": "DeepAgent 开始多轮工具调用…",
        })

        # ── 在线程中执行同步 agent，主协程负责持续吐 SSE ──────
        # DeepAgent / LangGraph 调用是同步阻塞的；如果直接在 async generator
        # 中运行，前端只能等到全部结束。这里用线程跑 agent，用 Queue 把
        # 工具事件从工作线程送回 SSE 主协程，从而做到工具级实时展示。
        event_queue: queue.Queue[dict] = queue.Queue()

        def tool_callback(event: dict) -> None:
            """线程安全回调：DeepAgentWrapper 每次工具开始/结束都会调用它。"""
            event["run_id"] = run_id
            event_queue.put(event)

        agent_error: Exception | None = None
        agent_result: ConsultationResponse | None = None

        def _run_agent():
            nonlocal agent_result, agent_error
            try:
                raw = agent.answer(request, sources, event_callback=tool_callback)
                agent_result = guard.review(raw)
            except Exception as exc:
                agent_error = exc

        task = asyncio.create_task(asyncio.to_thread(_run_agent))

        heartbeat_count = 0
        last_heartbeat = time.time()
        while not task.done():
            await asyncio.sleep(0.2)  # 高频 drain，降低工具事件到前端的感知延迟。

            # 先吐工具事件，再吐 heartbeat；避免“心跳刷屏”压过真实进展。
            while True:
                try:
                    ev = event_queue.get_nowait()
                    yield sse_event(ev.pop("type", "tool_call"), ev)
                except queue.Empty:
                    break

            # heartbeat 只表达“后端仍在运行”，不是工具进度本身。
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                heartbeat_count += 1
                last_heartbeat = now
                yield sse_event("heartbeat", {
                    "run_id": run_id, "stage": "agent", "label": "运行中…",
                    "message": f"DeepAgent 仍在处理（已等待 {heartbeat_count * _HEARTBEAT_INTERVAL}s）",
                })

        # Wait for task to propagate any exception
        await task

        # agent 线程刚结束时，队列里可能还有最后一个 tool_done/tool_failed。
        while True:
            try:
                ev = event_queue.get_nowait()
                yield sse_event(ev.pop("type", "tool_call"), ev)
            except queue.Empty:
                break

        if agent_error is not None:
            raise agent_error

        if agent_result is None:
            raise RuntimeError("Agent returned None — this should never happen")

        # ── Phase 4: Result processing ──────────────────────────
        response = agent_result
        debug_info = agent.debug_info if hasattr(agent, "debug_info") else {}
        traces = debug_info.get("tool_traces", [])
        arch_info = debug_info
        fallback = debug_info.get("fallback_used", False)
        fallback_reason = debug_info.get("fallback_reason")

        # pipeline fallback 不经过 DeepAgentWrapper 的实时回调，只能继续补发
        # legacy traces；真正 DeepAgent 路径已经在运行中推过 tool_started/done。
        if _should_replay_tool_traces(agent):
            for i, trace in enumerate(traces):
                yield sse_event("tool_call", {
                    "run_id": run_id, "stage": "tools",
                    "label": f"工具调用 #{i + 1}",
                    "message": f"{trace.get('name_cn', trace.get('tool_id', ''))} — {'✓' if trace.get('success', False) else '✗'}",
                    "tool_id": trace.get("tool_id", ""),
                    "tool_name": trace.get("name_cn", ""),
                    "tool_success": trace.get("success", False),
                    "elapsed_ms": trace.get("elapsed_ms", 0),
                })

        # Fallback event
        if fallback:
            yield sse_event("fallback_used", {
                "run_id": run_id, "stage": "result", "label": "回退",
                "message": fallback_reason or "DeepAgent 回退到 pipeline",
            })

        # Report done
        yield sse_event("report_done", {
            "run_id": run_id, "stage": "result", "label": "报告生成完成",
            "message": f"answer 长度: {len(response.answer)} 字符",
        })

        # ── Phase 5: Complete ───────────────────────────────────
        run_done_payload: dict = {
            "run_id": run_id, "stage": "done", "label": "运行结束",
            "message": "分析完成",
            "response": response.model_dump(mode="json") if hasattr(response, "model_dump") else response,
            "agent_architecture": arch_info if isinstance(arch_info, dict) else {},
            "evidence": evidence_pack.model_dump(mode="json") if hasattr(evidence_pack, "model_dump") else {},
            "planned_queries": planned_queries,
        }
        if router_decision is not None:
            run_done_payload["router"] = router_decision.model_dump(mode="json")
            run_done_payload["selected_intent"] = router_decision.selected_intent
        yield sse_event("run_done", run_done_payload)

    except Exception as exc:
        logger.exception("Stream run failed for %s", run_id)
        yield sse_event("run_error", {
            "run_id": run_id, "stage": "error", "label": "运行失败",
            "message": f"{type(exc).__name__}: {exc}",
        })
        yield sse_event("run_done", {
            "run_id": run_id, "label": "结束", "message": "运行异常终止",
        })

    finally:
        delete_run(run_id)
