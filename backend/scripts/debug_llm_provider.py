#!/usr/bin/env python
"""LLM Provider Debug Script — checks config and runs minimal real-LLM tests."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import json

def header(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}")
    if detail and not passed:
        print(f"         {detail}")

def main():
    failures = 0

    header("1. Environment Variables")
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        masked = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
        check("DEEPSEEK_API_KEY is set", True, f"masked={masked}")
    else:
        check("DEEPSEEK_API_KEY is set", False, "Set: export DEEPSEEK_API_KEY=sk-...")
        print("  -> LLM tests SKIP (no key).")
        return 1

    enabled = os.getenv("FIN_AGENT_LLM_ENABLED", "")
    mock = os.getenv("FIN_AGENT_LLM_MOCK_MODE", "")
    print(f"  [INFO] FIN_AGENT_LLM_ENABLED={enabled or '(unset)'}")
    print(f"  [INFO] FIN_AGENT_LLM_MOCK_MODE={mock or '(unset)'}")

    header("2. Dependencies")
    try:
        import langchain_deepseek
        check("langchain-deepseek importable", True)
    except ImportError:
        check("langchain-deepseek importable", False, "Install: pip install langchain-deepseek")
        return 1

    try:
        from app.llm.provider import LLMProvider, get_provider
        check("LLMProvider importable", True)
    except ImportError as e:
        check("LLMProvider importable", False, str(e))
        return 1

    header("3. Provider Default Configuration")
    provider = get_provider()
    check(f"Default model = {provider.model}", provider.model == "deepseek-v4-flash")
    if provider.model != "deepseek-v4-flash":
        failures += 1
    check("is_configured() is False by default", not provider.is_configured())

    header("4. Minimal Real LLM Call")
    real = LLMProvider(enabled=True, mock_mode=False)
    if not real.is_configured():
        check("Real provider configured", False)
        failures += 1
    else:
        check("Real provider configured", True)
        try:
            resp = real.complete(
                messages=[{"role": "user", "content": "只回复两个小写字母 ok，不要其他任何内容。"}],
                temperature=0.0, max_tokens=200,
            )
            if "ok" in resp.strip().lower():
                check("Minimal call: reply ok", True, f"Response: [{resp.strip()}]")
            else:
                check("Minimal call: reply ok", False, f"Unexpected: [{resp[:100]}]")
                failures += 1
        except Exception as e:
            check("Minimal call: no exception", False, f"{type(e).__name__}: {e}")
            failures += 1

    header("5. Profile Enrichment (real LLM)")
    try:
        from app.agents.investment_advisor_tools.profile_analyzer import analyze, enrich_profile_with_llm
        from app.llm.prompts import build_profile_prompt
        import json as _json
        # Use clearer signals so LLM is more likely to fill fields
        profile = analyze("我每月税后到手4200元，完全不懂投资，想开始存钱但不知道要存多久")
        missing_before = list(profile.get("missing_fields", []))
        print(f"  [INFO] Before: missing={missing_before}, income={profile.get('income_level')}")
        # Show raw LLM response for diagnostics
        prompt = build_profile_prompt(
            current_profile=str({
                k: v for k, v in profile.items()
                if k in ("income_level", "risk_preference", "investment_experience", "liquidity_need", "constraints")
            }),
            missing_fields=", ".join(missing_before),
            user_question="我每月税后到手4200元，完全不懂投资，想开始存钱但不知道要存多久",
        )
        raw = real.complete(messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=2000)
        print(f"  [INFO] Raw LLM: {raw[:200]}")
        try:
            js = raw[raw.find("{"):raw.rfind("}")+1] if "{" in raw else "{}"
            print(f"  [INFO] Parsed: {_json.dumps(_json.loads(js), ensure_ascii=False)}")
        except Exception:
            print(f"  [INFO] Could not parse JSON from response")
        enriched = enrich_profile_with_llm(
            profile, "我每月税后到手4200元，完全不懂投资，想开始存钱但不知道要存多久", real,
        )
        check("risk_preference not set by LLM",
              enriched.get("risk_preference") == profile.get("risk_preference"),
              f"risk_preference={enriched.get('risk_preference')}")
        missing_after = enriched.get("missing_fields", [])
        print(f"  [INFO] After: missing={missing_after}, income={enriched.get('income_level')}")
        if len(missing_after) < len(missing_before):
            check("LLM filled at least one missing field", True,
                  f"Filled {len(missing_before)-len(missing_after)} field(s)")
        else:
            check("LLM filled at least one missing field", False,
                  "LLM returned 'unknown' for all fields — may be non-deterministic, re-run")
            failures += 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        check("Profile enrichment: no exception", False, f"{type(e).__name__}: {e}")
        failures += 1

    header("6. Report Rewrite (real LLM)")
    try:
        from app.agents.investment_advisor import (
            InvestmentAdvisorAgent, rewrite_report_with_llm,
            _rewrite_preserves_hard_constraints,
        )
        from app.schemas.consultation import ConsultationRequest, Source
        agent = InvestmentAdvisorAgent()
        sources = [Source(title="asset allocation basics", source_type="investment_knowledge", confidence=0.85)]
        resp = agent.answer(
            ConsultationRequest(question="月薪1万，风险厌恶，3年后买房，如何配置资产？",
                              user_profile={"risk_preference": "low"}), sources=sources)
        original = resp.answer
        print(f"  [INFO] Original: {len(original)} chars")
        rewritten = rewrite_report_with_llm(original, provider=real)
        if rewritten == original:
            check("Report rewritten (not original)", False)
            failures += 1
        else:
            print(f"  [INFO] Rewritten: {len(rewritten)} chars")
            if _rewrite_preserves_hard_constraints(original, rewritten):
                check("Hard constraints satisfied", True)
            else:
                check("Hard constraints satisfied", False)
                failures += 1
        from app.agents.investment_advisor_tools.advisory_compliance_policy import review as review_compliance
        final = rewritten if rewritten != original else original
        comp = review_compliance(final)
        violations = [w for w in comp.get("warnings", []) if w.startswith("[违规]")]
        check("No compliance violations", len(violations) == 0, f"Violations: {violations}")
        if violations:
            failures += 1
    except Exception as e:
        check("Report rewrite: no exception", False, f"{type(e).__name__}: {e}")
        failures += 1

    header("Summary")
    if failures == 0:
        print("  ALL CHECKS PASSED")
    else:
        print(f"  {failures} CHECK(S) FAILED")
    print()
    return failures

if __name__ == "__main__":
    sys.exit(min(main(), 1))
