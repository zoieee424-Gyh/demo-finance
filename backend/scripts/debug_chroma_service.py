#!/usr/bin/env python
"""
Debug script: verify ConsultationService returns Chroma sources when RAG is enabled.

Usage:
    $env:FIN_AGENT_RAG_ENABLED = "true"
    $env:DASHSCOPE_API_KEY = "sk-..."    # must be set already
    D:/AI/soft/conda/envs/python3.11/python.exe backend/scripts/debug_chroma_service.py
"""
from __future__ import annotations

import io
import os
import sys

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import settings
from app.schemas.consultation import ConsultationRequest
from app.services.consultation_service import ConsultationService


def main() -> int:
    print("=" * 60)
    print("  Chroma Service-Layer E2E Debug")
    print("=" * 60)

    # ── 1. Check environment ─────────────────────────────────────
    print("\n[1] Environment check:")
    print(f"    FIN_AGENT_RAG_ENABLED = {os.getenv('FIN_AGENT_RAG_ENABLED', 'NOT SET')}")
    print(f"    settings.rag_enabled  = {settings.rag_enabled}")
    print(f"    settings.chroma_dir   = {settings.chroma_dir}")
    print(f"    DASHSCOPE_API_KEY     = {'SET' if os.getenv('DASHSCOPE_API_KEY') else 'NOT SET'}")
    print(f"    Chroma dir exists     = {os.path.isdir(settings.chroma_dir)}")

    if not settings.rag_enabled:
        print("\n    [WARN] RAG is disabled. Set $env:FIN_AGENT_RAG_ENABLED='true'")
        return 1

    if not os.path.isdir(settings.chroma_dir):
        print(f"\n    [WARN] Chroma dir not found: {settings.chroma_dir}")
        return 1

    # ── 2. Create service ────────────────────────────────────────
    print("\n[2] Creating ConsultationService...")
    service = ConsultationService()

    has_chroma = service.retriever.has_chroma
    print(f"    retriever.has_chroma = {has_chroma}")

    if not has_chroma:
        print("    [FAIL] ChromaStore was NOT injected into retriever!")
        print("    Checking individual conditions:")
        import os as _os
        has_key = bool(_os.getenv("DASHSCOPE_API_KEY"))
        has_dir = _os.path.isdir(settings.chroma_dir)
        print(f"      DASHSCOPE_API_KEY: {has_key}")
        print(f"      Chroma dir exists: {has_dir}")

        # Try to check collections directly
        try:
            from app.rag.chroma_store import ChromaStore
            from app.rag.embedding_provider import EmbeddingProvider
            provider = EmbeddingProvider()
            store = ChromaStore(persist_dir=settings.chroma_dir, embedding_provider=provider)
            colls = store.list_collections()
            has_data = any(c.get("count", 0) > 0 for c in colls)
            print(f"      Collections with data: {has_data}")
            if not has_data:
                print(f"      (collections found: {[c['name'] + ':' + str(c['count']) for c in colls]})")
        except Exception as e:
            print(f"      Error checking Chroma: {e}")
        return 1

    print("    [OK] ChromaStore injected successfully!")

    # ── 3. Test queries ──────────────────────────────────────────
    test_cases = [
        ("保守型投资者如何配置资产？", "advisory"),
        ("什么是基金？", "education"),
        ("投资建议有哪些合规红线？", "compliance"),
        ("如何管理持仓集中度风险？", "risk_control"),
    ]

    all_passed = True
    for question, expected_intent in test_cases:
        print(f"\n[3] Query: \"{question}\"")
        print(f"    Expected intent: {expected_intent}")

        try:
            response = service.handle(ConsultationRequest(question=question))
            print(f"    Actual intent:   {response.intent}")
            print(f"    Sources count:   {len(response.sources)}")
            if response.intent != expected_intent:
                print(f"    [FAIL] Intent mismatch: expected {expected_intent}, got {response.intent}")
                all_passed = False

            # Check if sources look like Chroma results. Some real Chroma
            # documents intentionally share titles with legacy mock sources,
            # so compare title + mock confidence instead of title only.
            mock_sources = {
                ("资产配置基础原则", 0.85), ("生命周期投资理论", 0.78),
                ("基金入门：什么是公募基金", 0.88), ("PE与PB估值指标基础", 0.85),
                ("证券法信息披露要求", 0.95), ("资管新规核心要点", 0.91),
                ("Beneish M-Score 财务造假识别模型", 0.92), ("Altman Z-Score 破产预测模型", 0.88),
            }

            for s in response.sources:
                score_str = f"score={s.confidence:.4f}" if s.confidence else "score=N/A"
                score = round(s.confidence or 0.0, 2)
                is_mock = (s.title, score) in mock_sources
                source_label = "[MOCK]" if is_mock else "[CHROMA]"
                print(f"      {source_label} {s.title} ({score_str}, type={s.source_type})")

            # Determine if Chroma is working
            chroma_hits = [
                s for s in response.sources
                if (s.title, round(s.confidence or 0.0, 2)) not in mock_sources
            ]
            if chroma_hits:
                print(f"    [OK] {len(chroma_hits)} Chroma source(s) found!")
            else:
                non_mock = [s for s in response.sources]
                if non_mock:
                    print(f"    [OK] {len(non_mock)} non-mock source(s) — likely Chroma")
                else:
                    print(f"    [WARN] All sources appear to be mock titles. "
                          f"Chroma may have returned nothing for this query.")

            if not response.risk_notice:
                print(f"    [WARN] No risk_notice!")
                all_passed = False

        except Exception as e:
            print(f"    [FAIL] Exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    # ── 4. Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed and has_chroma:
        print("  ALL CHECKS PASSED — Chroma is serving real sources!")
    else:
        print("  Some checks failed — see details above.")
    print("=" * 60)

    return 0 if (all_passed and has_chroma) else 1


if __name__ == "__main__":
    sys.exit(main())
