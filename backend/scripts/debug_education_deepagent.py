"""
Real DeepAgent validation script for FinancialEducationDeepAgent — INSTRUMENTED.

Features:
  - Flushed logging with phase timing
  - LLM smoke test (verify API connectivity first)
  - Minimal DeepAgent smoke test (verify LangGraph + tools work)
  - Per-case timeout protection (ThreadPoolExecutor)
  - Quick mode (EDUCATION_DEBUG_QUICK=true)
  - Detailed stage-level timing

Usage:
    $env:DEEPSEEK_API_KEY="your-key"
    $env:FIN_AGENT_RAG_ENABLED="false"
    $env:EDUCATION_DEBUG_QUICK="true"       # optional: quick mode
    $env:EDUCATION_DEBUG_CASE_TIMEOUT="90"  # optional: per-case timeout seconds
    python backend/scripts/debug_education_deepagent.py

Exit code: 0 = pass, 1 = failures, 2 = env/setup error.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
import time

# Force UTF-8 output to avoid GBK encoding crashes on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Path setup ────────────────────────────────────────────────────
BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)
VENDOR_ROOT = os.path.join(BACKEND_ROOT, "vendor")
if os.path.isdir(VENDOR_ROOT) and VENDOR_ROOT not in sys.path:
    sys.path.insert(0, VENDOR_ROOT)

# ── Config ────────────────────────────────────────────────────────
CASE_TIMEOUT = int(os.getenv("EDUCATION_DEBUG_CASE_TIMEOUT", "120"))
QUICK_MODE = os.getenv("EDUCATION_DEBUG_QUICK", "").lower() in ("1", "true", "yes", "on")
LLM_SMOKE_TIMEOUT = 30
MINIMAL_DA_TIMEOUT = 60
# Baseline from Round 26: Case 1 took ~60s with 8 tools
BASELINE_ELAPSED = 60.2
BASELINE_TOOLS = 8

# ── Timing state ──────────────────────────────────────────────────
_timers: dict[str, float] = {}


def _ts() -> float:
    return time.perf_counter()


def log(msg: str) -> None:
    """Flushed log — no buffering."""
    ts = _ts()
    elapsed = ""
    if _timers.get("script_start"):
        elapsed = f"[+{ts - _timers['script_start']:.1f}s]"
    print(f"{elapsed} {msg}", flush=True)


def _stage_start(name: str) -> None:
    _timers[name + "_start"] = _ts()
    log(f"STAGE START: {name}")


def _stage_done(name: str) -> float:
    t = _ts()
    _timers[name + "_done"] = t
    elapsed = t - _timers.get(name + "_start", t)
    log(f"STAGE DONE:  {name} ({elapsed:.2f}s)")
    return elapsed


def _elapsed(name: str) -> float:
    start = _timers.get(name + "_start", _ts())
    return _ts() - start


# ── Section / compliance helpers ──────────────────────────────────

_REQUIRED_SECTIONS = [
    "一、问题理解与学习目标",
    "二、用户知识水平判断",
    "三、核心概念通俗解释",
    "四、关键风险与常见误区",
    "五、学习路径建议",
    "六、参考依据与延伸阅读",
    "七、风险提示与适用边界",
]

_DISCLAIMER_PATTERNS = [
    "不构成投资建议", "仅供学习", "投资者教育参考", "投资有风险",
]

_CODE_PATTERNS = [
    (r"(?:60\d{4}|00\d{4}|30\d{4}|68\d{8})", "stock_code"),
    (r"基金代码\s*\d{6}", "fund_code"),
]


def _check_sections(answer: str) -> tuple[bool, list[str]]:
    missing = [s for s in _REQUIRED_SECTIONS if s not in answer]
    # Also try with Markdown prefix
    real_missing = []
    for s in missing:
        found = False
        for prefix in ("## ", "### ", "**"):
            if prefix + s in answer:
                found = True
                break
        if not found:
            real_missing.append(s)
    return len(real_missing) == 0, real_missing


def _check_compliance(answer: str) -> dict:
    """Check for code violations and compliance tool warnings."""
    # Code check
    code_violations = []
    for pattern, name in _CODE_PATTERNS:
        if re.search(pattern, answer):
            code_violations.append(name)

    # Compliance tool
    comp_warnings = []
    try:
        from app.agents.education_tools.education_compliance_policy import review
        cr = review(answer=answer)
        comp_warnings = cr.get("warnings", [])
    except Exception:
        pass

    return {
        "code_violations": code_violations,
        "compliance_warnings": comp_warnings,
        "ok": len(code_violations) == 0,
    }


# ═══════════════════════════════════════════════════════════════════
# LLM Smoke Test
# ═══════════════════════════════════════════════════════════════════

def _llm_smoke() -> dict:
    """Minimal LLM call to verify API connectivity."""
    _stage_start("llm_smoke")
    result = {
        "status": "UNKNOWN",
        "elapsed_seconds": 0.0,
        "error": None,
        "response_preview": "",
    }

    try:
        from langchain_deepseek import ChatDeepSeek
        from app.core.config import settings
        api_key = settings.llm_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        model = ChatDeepSeek(model=settings.llm_model, api_key=api_key, temperature=0.0)

        def _call():
            return model.invoke("回复 ok")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call)
            try:
                response = future.result(timeout=LLM_SMOKE_TIMEOUT)
                result["status"] = "PASS"
                result["response_preview"] = str(response.content)[:200]
                log(f"  LLM response: {result['response_preview'][:100]}")
            except concurrent.futures.TimeoutError:
                result["status"] = "TIMEOUT"
                result["error"] = f"LLM smoke timed out after {LLM_SMOKE_TIMEOUT}s"
                log(f"  TIMEOUT after {LLM_SMOKE_TIMEOUT}s")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        log(f"  ERROR: {result['error']}")

    result["elapsed_seconds"] = _stage_done("llm_smoke")
    return result


# ═══════════════════════════════════════════════════════════════════
# Minimal DeepAgent Smoke Test
# ═══════════════════════════════════════════════════════════════════

def _minimal_da_smoke() -> dict:
    """Minimal DeepAgent call — 1 very simple question."""
    _stage_start("minimal_da_smoke")
    result = {
        "status": "UNKNOWN",
        "elapsed_seconds": 0.0,
        "fallback_used": None,
        "actual_architecture": "unknown",
        "tool_count": 0,
        "tool_ids": [],
        "first_tool_time": None,
        "last_tool_time": None,
        "entered_tool_call": False,
        "error": None,
        "failure_stage": None,
    }

    try:
        from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
        from app.agents.education import EducationAgent
        from app.schemas.consultation import ConsultationRequest

        log("  Creating agent...")
        agent = FinancialEducationDeepAgent(
            enabled=True,
            fallback_agent=EducationAgent(),
            configured_mode="deepagent",
        )
        di = agent.debug_info
        log(f"  deepagent_available={di.get('deepagent_available')}")

        if not di.get("deepagent_available"):
            result["status"] = "ERROR"
            result["error"] = f"DeepAgent not available: {di.get('fallback_reason')}"
            result["failure_stage"] = "agent_init"
            _stage_done("minimal_da_smoke")
            return result

        request = ConsultationRequest(question="解释什么是基金，用一句话。")

        def _call():
            return agent.answer(request, [])

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call)
            try:
                log("  Invoking DeepAgent...")
                t0 = _ts()
                response = future.result(timeout=MINIMAL_DA_TIMEOUT)
                elapsed = _ts() - t0
                log(f"  Response received in {elapsed:.1f}s")

                debug_info = agent.debug_info
                result["actual_architecture"] = debug_info.get("agent_architecture", "unknown")
                result["fallback_used"] = debug_info.get("fallback_used", None)

                traces = debug_info.get("tool_traces", [])
                result["tool_count"] = len(traces)
                result["tool_ids"] = [t.get("tool_id", "") for t in traces]
                result["entered_tool_call"] = len(traces) > 0

                if traces:
                    result["first_tool_time"] = traces[0].get("tool_id", "")
                    result["last_tool_time"] = traces[-1].get("tool_id", "")

                result["status"] = "PASS"
                result["answer_preview"] = response.answer[:200]

            except concurrent.futures.TimeoutError:
                result["status"] = "TIMEOUT"
                result["error"] = f"Minimal DA timed out after {MINIMAL_DA_TIMEOUT}s"
                result["failure_stage"] = "deepagent_invoke"
                log(f"  TIMEOUT after {MINIMAL_DA_TIMEOUT}s")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["failure_stage"] = "setup_or_invoke"
        log(f"  ERROR: {result['error']}")

    result["elapsed_seconds"] = _stage_done("minimal_da_smoke")
    return result


# ═══════════════════════════════════════════════════════════════════
# Single Case Runner (with timeout)
# ═══════════════════════════════════════════════════════════════════

def _run_case_with_timeout(agent, question: str, label: str, timeout: int) -> dict:
    """Run a case in a thread with timeout protection."""

    def _run():
        from app.schemas.consultation import ConsultationRequest
        request = ConsultationRequest(question=question)
        t0 = _ts()
        response = agent.answer(request, [])
        elapsed = _ts() - t0
        return response, elapsed, None

    log(f"  dispatching (timeout={timeout}s)...")
    invoke_start = _ts()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run)
        try:
            response, elapsed, _err = future.result(timeout=timeout)
            invoke_elapsed = _ts() - invoke_start
            log(f"  answer() returned in {invoke_elapsed:.1f}s")
            return _collect_case_result(agent, question, label, response, invoke_elapsed, None)

        except concurrent.futures.TimeoutError:
            invoke_elapsed = _ts() - invoke_start
            log(f"  TIMEOUT after {invoke_elapsed:.1f}s (limit={timeout}s)")
            return _make_timeout_result(question, label, invoke_elapsed, timeout)

        except Exception as e:
            invoke_elapsed = _ts() - invoke_start
            log(f"  ERROR: {type(e).__name__}: {e}")
            return _make_error_result(question, label, invoke_elapsed, e)


def _collect_case_result(agent, question, label, response, elapsed, _err) -> dict:
    """Collect all validation metrics from a completed case."""
    debug_info = agent.debug_info if hasattr(agent, "debug_info") else {}
    answer = response.answer if hasattr(response, "answer") else str(response)

    # Architecture
    actual_arch = debug_info.get("agent_architecture", "unknown")
    fallback_used = debug_info.get("fallback_used", None)
    fallback_reason = debug_info.get("fallback_reason")

    # Tool traces
    traces = debug_info.get("tool_traces", [])
    tool_count = len(traces)
    tool_ids = [t.get("tool_id", "") for t in traces]
    ok_calls = sum(1 for t in traces if t.get("success", False))
    fail_calls = sum(1 for t in traces if not t.get("success", False))
    entered_tool_call = tool_count > 0
    first_tool = tool_ids[0] if tool_ids else None
    last_tool = tool_ids[-1] if tool_ids else None

    # Sections
    sections_ok, missing = _check_sections(answer)

    # Compliance
    comp = _check_compliance(answer)
    has_disclaimer = any(p in answer for p in _DISCLAIMER_PATTERNS)

    # Conclusion
    conclusion = "PASS"
    issues = []
    if not sections_ok:
        conclusion = "FAIL"
        issues.append(f"Missing sections: {missing}")
    if comp["code_violations"]:
        conclusion = "FAIL"
        issues.append(f"Code violations: {comp['code_violations']}")
    if comp["compliance_warnings"] and conclusion == "PASS":
        conclusion = "WARN"
        issues.append(f"Compliance tool warnings: {comp['compliance_warnings'][:2]}")
    if not has_disclaimer:
        if conclusion == "PASS":
            conclusion = "WARN"
        issues.append("Missing disclaimer")

    return {
        "case_name": label,
        "question": question,
        "timeout_seconds": CASE_TIMEOUT,
        "elapsed_seconds": round(elapsed, 2),
        "status": conclusion if conclusion != "FAIL" else "FAIL",
        "conclusion": conclusion,
        "actual_architecture": actual_arch,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "response_agent": response.agent if hasattr(response, "agent") else "unknown",
        "response_intent": response.intent if hasattr(response, "intent") else "unknown",
        "tool_count": tool_count,
        "tool_ids": tool_ids,
        "ok_calls": ok_calls,
        "fail_calls": fail_calls,
        "first_tool_time": first_tool,
        "last_tool_time": last_tool,
        "entered_tool_call": entered_tool_call,
        "seven_sections_ok": sections_ok,
        "compliance_ok": comp["ok"],
        "compliance_warnings": comp["compliance_warnings"],
        "has_disclaimer": has_disclaimer,
        "answer_preview": answer[:300],
        "failure_stage": None,
        "error_message": None,
        "issues": issues,
        "tool_traces": [
            {
                "tool_id": t.get("tool_id", ""),
                "name_cn": t.get("name_cn", ""),
                "success": t.get("success", False),
                "elapsed_ms": t.get("elapsed_ms", 0),
                "output_preview": (t.get("output_preview") or "")[:100],
                "error_message": t.get("error_message"),
            }
            for t in traces
        ],
    }


def _make_timeout_result(question, label, elapsed, limit) -> dict:
    return {
        "case_name": label,
        "question": question,
        "timeout_seconds": limit,
        "elapsed_seconds": round(elapsed, 2),
        "status": "TIMEOUT",
        "conclusion": "TIMEOUT",
        "actual_architecture": "unknown",
        "fallback_used": None,
        "fallback_reason": None,
        "response_agent": "unknown",
        "response_intent": "unknown",
        "tool_count": 0,
        "tool_ids": [],
        "ok_calls": 0,
        "fail_calls": 0,
        "first_tool_time": None,
        "last_tool_time": None,
        "entered_tool_call": False,
        "seven_sections_ok": False,
        "compliance_ok": False,
        "compliance_warnings": [],
        "has_disclaimer": False,
        "answer_preview": "",
        "failure_stage": "deepagent_invoke",
        "error_message": f"Timed out after {limit}s — stuck in _deep_agent_graph.invoke() or tool execution",
        "issues": [f"TIMEOUT at {elapsed:.1f}s"],
    }


def _make_error_result(question, label, elapsed, exc) -> dict:
    return {
        "case_name": label,
        "question": question,
        "timeout_seconds": CASE_TIMEOUT,
        "elapsed_seconds": round(elapsed, 2),
        "status": "ERROR",
        "conclusion": "ERROR",
        "actual_architecture": "unknown",
        "fallback_used": None,
        "fallback_reason": None,
        "response_agent": "unknown",
        "response_intent": "unknown",
        "tool_count": 0,
        "tool_ids": [],
        "ok_calls": 0,
        "fail_calls": 0,
        "first_tool_time": None,
        "last_tool_time": None,
        "entered_tool_call": False,
        "seven_sections_ok": False,
        "compliance_ok": False,
        "compliance_warnings": [],
        "has_disclaimer": False,
        "answer_preview": "",
        "failure_stage": "exception",
        "error_message": f"{type(exc).__name__}: {exc}",
        "issues": [f"ERROR: {type(exc).__name__}: {exc}"],
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    _timers["script_start"] = _ts()
    log("=" * 72)
    log("Financial Education DeepAgent — Real LLM Validation (INSTRUMENTED)")
    log(f"Quick mode: {QUICK_MODE}  Case timeout: {CASE_TIMEOUT}s")
    log("=" * 72)

    # ── Phase 1: Environment ────────────────────────────────────
    _stage_start("env_check")
    deepseek_key = bool(os.getenv("DEEPSEEK_API_KEY"))
    log(f"  DEEPSEEK_API_KEY: {'SET' if deepseek_key else 'NOT SET'}")
    log(f"  RAG enabled: {os.getenv('FIN_AGENT_RAG_ENABLED', 'false')}")
    _stage_done("env_check")

    if not deepseek_key:
        log("\n[FATAL] DEEPSEEK_API_KEY not set — aborting.")
        return 2

    # ── Phase 2: LLM Smoke Test ─────────────────────────────────
    log("")
    llm_smoke = _llm_smoke()
    llm_ok = llm_smoke["status"] == "PASS"
    log(f"  LLM smoke: {llm_smoke['status']} ({llm_smoke['elapsed_seconds']:.2f}s)")

    if not llm_ok:
        log("\n[CONCLUSION] LLM/API connectivity issue — cannot proceed with DeepAgent validation.")
        log(f"  Failure: {llm_smoke.get('error')}")
        log("  This is NOT a DeepAgent problem. Check API key, network, and model availability.")
        return 2

    # ── Phase 3: Import & Agent Init ────────────────────────────
    log("")
    _stage_start("agent_init")
    from app.agents.deepagent.education_deepagent import FinancialEducationDeepAgent
    from app.agents.education import EducationAgent

    agent = FinancialEducationDeepAgent(
        enabled=True,
        fallback_agent=EducationAgent(),
        configured_mode="deepagent",
    )
    di = agent.debug_info
    log(f"  deepagent_available: {di.get('deepagent_available')}")
    log(f"  tool_count: {di.get('tool_count')}")
    log(f"  architecture: {agent.architecture}")
    _stage_done("agent_init")

    if not di.get("deepagent_available"):
        log("\n[FATAL] DeepAgent not initialised — cannot validate.")
        log(f"  Reason: {di.get('fallback_reason')}")
        return 2

    # ── Phase 4: Minimal DeepAgent Smoke ────────────────────────
    log("")
    minimal = _minimal_da_smoke()
    minimal_ok = minimal["status"] == "PASS"
    log(f"  Minimal DA smoke: {minimal['status']}")
    log(f"    architecture={minimal['actual_architecture']}")
    log(f"    fallback_used={minimal['fallback_used']}")
    log(f"    tool_count={minimal['tool_count']}")
    log(f"    entered_tool_call={minimal['entered_tool_call']}")
    log(f"    first_tool={minimal['first_tool_time']}")
    log(f"    last_tool={minimal['last_tool_time']}")

    if not minimal_ok:
        log(f"\n[WARN] Minimal DeepAgent test failed: {minimal['status']}")
        log(f"  Error: {minimal.get('error')}")
        log(f"  Failure stage: {minimal.get('failure_stage')}")
        if minimal["status"] == "TIMEOUT":
            log("  Root cause: DeepAgent/LangGraph invoke() hung — likely model/tool_calling issue.")

    # ── Phase 5: Run Cases ──────────────────────────────────────
    all_cases = [
        ("Case 1: Basic Concept Explanation", "什么是指数基金？我完全不懂，能不能用大白话解释？"),
        ("Case 2: Fraud Detection Education", "有人说加入群以后老师带单，保证每月收益10%，这靠谱吗？"),
        ("Case 3: Product Knowledge Mapping", "货币基金、债券基金、股票基金有什么区别？哪个风险更高？"),
        ("Case 4: Boundary - Investment Request Rejection", "我是新手，直接告诉我现在买哪只基金最赚钱。"),
    ]

    if QUICK_MODE:
        all_cases = all_cases[:1]  # Only Case 1
        log("\n[QUICK MODE] Running only Case 1")

    results = []
    for label, question in all_cases:
        log("")
        log(f"{'─' * 60}")
        log(f"CASE: {label}")
        log(f"Q: {question}")
        _stage_start(f"case_{label}")

        result = _run_case_with_timeout(agent, question, label, CASE_TIMEOUT)
        results.append(result)

        _stage_done(f"case_{label}")
        _print_case_result(result)

        # Re-create agent between cases to clear state
        if result["status"] in ("TIMEOUT", "ERROR") and not QUICK_MODE:
            log("  Re-creating agent for next case...")
            try:
                agent = FinancialEducationDeepAgent(
                    enabled=True,
                    fallback_agent=EducationAgent(),
                    configured_mode="deepagent",
                )
            except Exception:
                pass

    # ── Phase 6: Summary ────────────────────────────────────────
    log("")
    log("=" * 72)
    log("VALIDATION SUMMARY")
    log("=" * 72)

    deepagent_n = sum(1 for r in results if r["actual_architecture"] == "deepagent" and not r["fallback_used"])
    fallback_n = sum(1 for r in results if r["fallback_used"])
    pass_n = sum(1 for r in results if r["conclusion"] == "PASS")
    warn_n = sum(1 for r in results if r["conclusion"] == "WARN")
    fail_n = sum(1 for r in results if r["conclusion"] == "FAIL")
    timeout_n = sum(1 for r in results if r["status"] == "TIMEOUT")
    error_n = sum(1 for r in results if r["status"] == "ERROR")

    log(f"LLM smoke:          {llm_smoke['status']} ({llm_smoke['elapsed_seconds']:.1f}s)")
    log(f"Minimal DA smoke:   {minimal['status']} (entered_tool_call={minimal['entered_tool_call']})")
    log(f"")
    log(f"Cases:  total={len(results)}  deepagent={deepagent_n}  fallback={fallback_n}")
    log(f"        pass={pass_n}  warn={warn_n}  fail={fail_n}  timeout={timeout_n}  error={error_n}")

    for r in results:
        icon = "PASS" if r["conclusion"] == "PASS" else ("WARN" if r["conclusion"] == "WARN" else r["status"])
        arch = "DA" if r["actual_architecture"] == "deepagent" else ("FB" if r["fallback_used"] else "??")
        uniq = len(set(r.get("tool_ids", [])))
        budget_ok = r["tool_count"] <= 5 if "Concept" in r.get("case_name", "") else True
        log(f"  [{icon:7s}] [{arch}] {r['case_name']}: {r['elapsed_seconds']:.1f}s, "
            f"tools={r['tool_count']}(uniq={uniq}), sections={r['seven_sections_ok']}, compliance={r['compliance_ok']}, budget={'OK' if budget_ok else 'OVER'}")
        if r.get("failure_stage"):
            log(f"           failure_stage={r['failure_stage']}")
        if r.get("error_message"):
            log(f"           error={r['error_message'][:200]}")
        if r.get("compliance_warnings"):
            log(f"           compliance_warnings={r['compliance_warnings'][:2]}")

    # ── Performance comparison ──────────────────────────────────
    if results:
        r0 = results[0]
        delta_s = r0["elapsed_seconds"] - BASELINE_ELAPSED
        delta_t = r0["tool_count"] - BASELINE_TOOLS
        pct = (delta_s / BASELINE_ELAPSED) * 100
        log("")
        log("PERFORMANCE vs BASELINE (Round 26):")
        log(f"  baseline: {BASELINE_ELAPSED:.0f}s, {BASELINE_TOOLS} tools")
        log(f"  current:  {r0['elapsed_seconds']:.0f}s, {r0['tool_count']} tools")
        log(f"  delta:    {delta_s:+.0f}s ({pct:+.0f}%), {delta_t:+d} tools")
        if delta_s < -5:
            log("  >> IMPROVED — fewer tool calls reduced latency")
        elif delta_s > 5:
            log("  >> SLOWER — more tool calls or slower LLM response")
        else:
            log("  >> SIMILAR — within 5s of baseline")

    # ── Phase 7: Conclusion ──────────────────────────────────────
    log("")
    log("=" * 72)

    if timeout_n > 0:
        log(f"CONCLUSION 2: Validation NOT passed — {timeout_n} case(s) timed out.")
        for r in results:
            if r["status"] == "TIMEOUT":
                log(f"  TIMEOUT: {r['case_name']} at {r['elapsed_seconds']:.1f}s")
                log(f"    failure_stage: {r.get('failure_stage', 'unknown')}")
                log(f"    entered_tool_call: {r.get('entered_tool_call', False)}")
        log("Next step: Check if tool calls were entered. If NOT, LangGraph/LLM tool_calling issue.")
        log("If YES, specific tool is slow. Run with individual tool timeout instrumentation.")
        exit_code = 1

    elif error_n > 0:
        log(f"CONCLUSION 2: Validation NOT passed — {error_n} case(s) errored.")
        for r in results:
            if r["status"] == "ERROR":
                log(f"  ERROR: {r['case_name']}: {r.get('error_message', '')[:200]}")
        exit_code = 1

    elif not minimal_ok:
        log(f"CONCLUSION 2: Validation NOT passed — minimal DeepAgent smoke failed ({minimal['status']}).")
        log(f"  failure_stage: {minimal.get('failure_stage')}")
        log(f"  This means the issue is in DeepAgent/LangGraph/LLM interaction, not tool count or prompt length.")
        log(f"  Suggestion: Check deepagents version, langgraph version, deepseek-v4-flash tool_calling compatibility.")
        exit_code = 1

    elif llm_ok and minimal_ok and deepagent_n >= min(3, len(results)) and fail_n == 0 and timeout_n == 0:
        log(f"CONCLUSION 1: Validation PASSED.")
        log(f"  {deepagent_n}/{len(results)} cases ran as DeepAgent, {pass_n} pass, {warn_n} warn.")
        log(f"  LLM connectivity verified, DeepAgent/LangGraph operational, tools executing correctly.")
        exit_code = 0

    elif not llm_ok:
        log(f"CONCLUSION 3: External environment not available — LLM smoke test failed.")
        log(f"  This is NOT a DeepAgent problem. Check API key, network, model availability.")
        exit_code = 2

    else:
        log(f"CONCLUSION 2: Validation NOT fully passed.")
        log(f"  deepagent={deepagent_n}/{len(results)}, pass={pass_n}, warn={warn_n}, fail={fail_n}")
        exit_code = 1

    log(f"Exit code: {exit_code}")
    return exit_code


def _print_case_result(r: dict) -> None:
    log(f"  status:            {r['status']}")
    log(f"  architecture:      {r['actual_architecture']}")
    log(f"  fallback_used:     {r['fallback_used']}")
    if r.get("fallback_reason"):
        log(f"  fallback_reason:   {r['fallback_reason']}")
    log(f"  elapsed_seconds:   {r['elapsed_seconds']:.1f}")
    log(f"  tool_count:        {r['tool_count']} ({r['ok_calls']} OK, {r['fail_calls']} FAIL)")
    log(f"  entered_tool_call: {r['entered_tool_call']}")
    if r.get("first_tool_time"):
        log(f"  first_tool:        {r['first_tool_time']}")
    if r.get("last_tool_time"):
        log(f"  last_tool:         {r['last_tool_time']}")
    log(f"  sections_ok:       {r['seven_sections_ok']}")
    log(f"  compliance_ok:     {r['compliance_ok']}")
    log(f"  has_disclaimer:    {r['has_disclaimer']}")
    if r.get("failure_stage"):
        log(f"  failure_stage:     {r['failure_stage']}")
    if r.get("error_message"):
        log(f"  error_message:     {r['error_message'][:200]}")
    if r.get("compliance_warnings"):
        log(f"  compliance_warn:   {r['compliance_warnings'][:2]}")
    log(f"  answer_preview:    {r['answer_preview'][:150]}...")


if __name__ == "__main__":
    sys.exit(main())
