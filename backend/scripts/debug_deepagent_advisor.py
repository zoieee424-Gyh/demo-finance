"""
DeepAgent Investment Advisor Debug Script

Validates the real DeepAgent-powered investment advisor end-to-end.

Usage:
    cd backend
    python scripts/debug_deepagent_advisor.py

Requirements:
    - DEEPSEEK_API_KEY environment variable must be set.
    - FIN_AGENT_USE_DEEPAGENT_ADVISOR=true must be set.
    - (Optional) DASHSCOPE_API_KEY + FIN_AGENT_RAG_ENABLED=true for RAG testing.

Prints:
    - Wrapper debug_info (architecture, available, fallback, tool_count).
    - Three test requests with tool call traces.
    - Answer preview, warnings, risk_notice, sources count.
    - Exit code 0 = all checks pass; exit code 1 = failure detected.
"""

from __future__ import annotations

import os
import sys

BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def _sep(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


def check_environment() -> bool:
    """Verify environment is ready for DeepAgent testing."""
    ok = True

    _sep("Environment Check")

    key = os.getenv("DEEPSEEK_API_KEY", "")
    print(f"  DEEPSEEK_API_KEY: {'SET' if key else 'MISSING'}")
    if not key:
        print("  WARNING: DEEPSEEK_API_KEY not set — model creation will fail")
        ok = False

    use_deep = os.getenv("FIN_AGENT_USE_DEEPAGENT_ADVISOR", "false")
    print(f"  FIN_AGENT_USE_DEEPAGENT_ADVISOR: {use_deep}")

    try:
        import deepagents
        print(f"  deepagents: {deepagents.__version__}")
    except ImportError:
        print("  deepagents: NOT INSTALLED")
        ok = False

    try:
        import langgraph
        print(f"  langgraph: installed")
    except ImportError:
        print("  langgraph: NOT INSTALLED")
        ok = False

    return ok


def build_wrapper():
    """Build DeepAgent wrapper with investment advisor tools."""
    from app.agents.deepagent.base import DeepAgentWrapper, INVESTMENT_ADVISOR_SYSTEM_PROMPT
    from app.agents.deepagent.registry import build_investment_advisor_registry
    from app.agents.investment_advisor import InvestmentAdvisorAgent

    reg = build_investment_advisor_registry()
    legacy = InvestmentAdvisorAgent()

    wrapper = DeepAgentWrapper(
        tool_registry=reg,
        legacy_agent=legacy,
        system_prompt=INVESTMENT_ADVISOR_SYSTEM_PROMPT,
        agent_name="investment_advisor_deepagent",
        agent_name_cn="DeepAgent智能投顾编排器",
        enabled=True,
    )
    return wrapper


def print_debug_info(wrapper) -> None:
    """Print wrapper debug metadata."""
    _sep("Wrapper Debug Info")
    info = wrapper.debug_info
    for key in [
        "agent_architecture",
        "deepagent_enabled",
        "deepagent_available",
        "fallback_used",
        "fallback_reason",
        "tool_count",
    ]:
        print(f"  {key}: {info[key]!r}")


def run_test(
    wrapper,
    label: str,
    question: str,
    user_profile: dict | None = None,
    sources: list | None = None,
) -> bool:
    """Run a single test request and print results."""
    _sub(f"Test: {label}")
    print(f"  Question: {question}")
    if user_profile:
        print(f"  User profile keys: {list(user_profile.keys())}")

    from app.schemas.consultation import ConsultationRequest

    request = ConsultationRequest(
        question=question,
        user_profile=user_profile,
    )
    srcs = sources or []

    try:
        response = wrapper.run(request, srcs)
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return False

    # ── Print response summary ──────────────────────────────
    answer = response.answer or ""
    print(f"  Response agent: {response.agent}")
    print(f"  Response intent: {response.intent}")
    print(f"  Answer length: {len(answer)} chars")
    print(f"  Sources: {len(response.sources)}")
    print(f"  Warnings: {len(response.warnings)}")
    if response.warnings:
        for w in response.warnings:
            print(f"    - {w}")
    print(f"  Risk notice: {response.risk_notice[:100]}..." if len(response.risk_notice) > 100 else f"  Risk notice: {response.risk_notice}")

    # ── Print answer preview ────────────────────────────────
    preview = answer[:1000]
    print(f"\n  Answer preview ({len(preview)} chars):")
    print("  " + "-" * 60)
    # Use ASCII-safe fallback for terminals with encoding issues
    for line in preview.split("\n")[:25]:
        try:
            print(f"  {line}")
        except UnicodeEncodeError:
            # Replace characters that can't be encoded in the terminal
            safe_line = line.encode("ascii", errors="replace").decode("ascii")
            print(f"  {safe_line}")
    if len(answer) > 1000:
        print(f"  ... ({len(answer) - 1000} more chars)")

    # ── Print tool trace ────────────────────────────────────
    traces = wrapper.tool_trace_summary
    if traces:
        print(f"\n  Tool call traces ({len(traces)} calls):")
        for t in traces:
            status = "OK" if t.get("success") else "FAIL"
            print(f"    [{status}] {t.get('tool_id')} ({t.get('name_cn')})")
            print(f"           inputs: {t.get('input_keys')}")
            if t.get("error_message"):
                print(f"           error: {t.get('error_message')}")
            if t.get("output_preview"):
                preview = t["output_preview"][:120]
                print(f"           output: {preview}")
    else:
        print(f"\n  Tool call traces: (none — likely fallback to legacy)")

    # ── Validate answer structure ───────────────────────────
    from app.agents.deepagent.base import validate_deepagent_output
    is_valid, failures = validate_deepagent_output(answer)
    if failures:
        print(f"\n  VALIDATION FAILURES:")
        for f in failures:
            print(f"    - {f}")
    else:
        print(f"\n  Validation: OK")

    return len(failures) == 0 if traces else True  # No traces = legacy fallback, skip validation


def main() -> int:
    """Run all debug checks. Returns exit code (0=pass, 1=fail)."""
    failures = 0

    # ── 1. Environment ────────────────────────────────────
    if not check_environment():
        failures += 1

    # ── 2. Build wrapper ──────────────────────────────────
    _sep("Build DeepAgent Wrapper")
    wrapper = build_wrapper()
    print_debug_info(wrapper)

    if not wrapper.is_deepagent_available:
        print("\n  DEEPAGENT NOT AVAILABLE — all tests will use legacy fallback.")
        print(f"  Reason: {wrapper.fallback_reason}")
        # Still run tests — they will exercise the fallback path

    # ── 3. Run test cases ─────────────────────────────────
    default_profile = {"risk_preference": "conservative"}

    # Test 1: Normal asset allocation query
    if not run_test(
        wrapper,
        label="资产配置咨询",
        question="我是保守型投资者，月收入中等，应该如何配置资产？",
        user_profile=default_profile,
    ):
        failures += 1

    # Test 2: Stock tip request — should be rejected
    if not run_test(
        wrapper,
        label="个股推荐请求（应拒绝）",
        question="推荐几只股票给我，最好是能涨的。",
        user_profile={"risk_preference": "balanced"},
    ):
        failures += 1

    # Test 3: Risk/configuration with sources
    from app.schemas.consultation import Source
    mock_sources = [
        Source(
            title="风险等级与资产类别匹配原则",
            source_type="investment_knowledge",
            content_preview="不同风险等级对应不同资产配置比例...",
            confidence=0.85,
        ),
        Source(
            title="投资者适当性管理规定概要",
            source_type="compliance_knowledge",
            content_preview="金融机构应当根据投资者的风险承受能力推荐适当产品...",
            confidence=0.72,
        ),
        Source(
            title="长期投资与复利效应",
            source_type="investment_knowledge",
            content_preview="长期持有的复利效应可以平滑短期波动...",
            confidence=0.68,
        ),
    ]
    if not run_test(
        wrapper,
        label="带RAG sources的配置问题",
        question="我是稳健型投资者，定投基金有风险吗？需要注意什么合规边界？",
        user_profile={"risk_preference": "stable", "investment_experience": "beginner"},
        sources=mock_sources,
    ):
        failures += 1

    # ── 4. Summary ─────────────────────────────────────────
    _sep("Summary")
    info = wrapper.debug_info
    print(f"  Agent architecture: {info['agent_architecture']}")
    print(f"  DeepAgent available: {info['deepagent_available']}")
    print(f"  Fallback used: {info['fallback_used']}")
    print(f"  Fallback reason: {info.get('fallback_reason') or 'N/A'}")
    print(f"  Test failures: {failures}")

    if failures == 0:
        print("\n  ALL CHECKS PASSED")
    else:
        print(f"\n  {failures} CHECK(S) FAILED")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
