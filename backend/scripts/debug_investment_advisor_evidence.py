#!/usr/bin/env python
"""
Debug script: verify evidence injection in InvestmentAdvisorAgent answers.

Usage:
    $env:FIN_AGENT_RAG_ENABLED = "true"
    $env:DASHSCOPE_API_KEY = "sk-..."     # must be set already
    D:/AI/soft/conda/envs/python3.11/python.exe backend/scripts/debug_investment_advisor_evidence.py

Does NOT call real LLM. Uses real Chroma / DashScope embedding if configured.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from app.core.config import settings
from app.schemas.consultation import ConsultationRequest
from app.services.consultation_service import ConsultationService
from app.rag.evidence import build_evidence_pack


def main() -> int:
    print("=" * 60)
    print("  Investment Advisor Evidence Injection Debug")
    print("=" * 60)

    # ── 1. Environment ──────────────────────────────────────────────
    print("\n[1] Environment:")
    print(f"    FIN_AGENT_RAG_ENABLED  = {os.getenv('FIN_AGENT_RAG_ENABLED', 'NOT SET')}")
    print(f"    settings.rag_enabled   = {settings.rag_enabled}")
    print(f"    DASHSCOPE_API_KEY      = {'SET' if os.getenv('DASHSCOPE_API_KEY') else 'NOT SET'}")
    print(f"    FIN_AGENT_LLM_ENABLED  = {os.getenv('FIN_AGENT_LLM_ENABLED', 'NOT SET')}")
    print(f"    settings.llm_enabled   = {settings.llm_enabled}")

    # ── 2. Create service ──────────────────────────────────────────
    print("\n[2] Creating ConsultationService...")
    service = ConsultationService()
    print(f"    retriever.has_chroma = {service.retriever.has_chroma}")

    # ── 3. Test queries ────────────────────────────────────────────
    test_cases = [
        ("保守型投资者能买债券基金吗？风险大吗？", "Cross-domain advisory+risk"),
        ("定投基金有风险吗？合规吗？", "Cross-domain advisory+risk+compliance"),
        ("能给我推荐几只股票吗？", "Compliance boundary — stock tip rejection"),
    ]

    all_passed = True

    for question, description in test_cases:
        print(f"\n[3] Query: \"{question}\"")
        print(f"    Description: {description}")

        try:
            response = service.handle(ConsultationRequest(question=question))
            print(f"    Intent:          {response.intent}")
            print(f"    Agent:           {response.agent}")
            print(f"    Sources count:   {len(response.sources)}")

            # Show sources
            for i, s in enumerate(response.sources, 1):
                score_str = f"{s.confidence:.4f}" if s.confidence else "N/A"
                print(f"      [{i}] {s.title} (type={s.source_type}, score={score_str})")

            # Build evidence pack
            evidence = build_evidence_pack(response.sources)
            print(f"    Evidence types:  advisory={evidence.has_advisory}, "
                  f"risk={evidence.has_risk}, compliance={evidence.has_compliance}, "
                  f"education={evidence.has_education}")
            print(f"    Low confidence:  {evidence.low_confidence}")

            # Check answer for evidence
            has_evidence_section = "八、参考依据与适用边界" in response.answer
            has_source_titles = any(s.title in response.answer for s in response.sources)
            has_low_conf_warning = "置信度有限" in response.answer
            has_risk_notice = bool(response.risk_notice)

            if response.agent == "investment_advisor":
                print(f"    Evidence section: {'✅' if has_evidence_section else '❌'}")
                print(f"    Source titles:    {'✅' if has_source_titles else '❌'}")
            else:
                print("    Evidence section: N/A (non-advisory agent)")
                print("    Source titles:    N/A (non-advisory agent)")
            print(f"    Low conf warn:    {'✅' if has_low_conf_warning else '—'}")
            print(f"    Risk notice:      {'✅' if has_risk_notice else '❌'}")

            # For stock tip query: verify rejection
            if "推荐" in question and "股票" in question:
                has_rejection = (
                    "不荐股" in response.answer
                    or "不推荐" in response.answer
                    or "不能推荐" in response.answer
                    or "无法推荐" in response.answer
                    or "合规" in response.answer
                )
                print(f"    Stock tip reject: {'✅' if has_rejection else '❌'}")
                if not has_rejection:
                    all_passed = False

            if response.intent == "advisory":
                if not has_evidence_section:
                    print("    [FAIL] Advisory answer missing evidence section!")
                    all_passed = False
                if response.sources and not has_source_titles:
                    print("    [FAIL] Advisory answer missing retrieved source titles!")
                    all_passed = False
                if not has_risk_notice:
                    print("    [FAIL] Advisory answer missing risk notice!")
                    all_passed = False

            # Print evidence excerpt from answer
            if has_evidence_section:
                # Extract the evidence section
                lines = response.answer.split("\n")
                in_evidence = False
                evidence_lines = []
                for line in lines:
                    if "八、参考依据与适用边界" in line:
                        in_evidence = True
                        continue
                    if in_evidence and ("九、" in line):
                        break
                    if in_evidence and line.strip():
                        evidence_lines.append(line)
                if evidence_lines:
                    print(f"    Evidence excerpt:")
                    for el in evidence_lines[:5]:
                        print(f"      {el.strip()}")
        except Exception as e:
            print(f"    [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # ── 4. Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL CHECKS PASSED — Evidence is injected into advisory answers!")
    else:
        print("  Some checks failed — see details above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
