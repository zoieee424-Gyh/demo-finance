#!/usr/bin/env python
"""
Real RAG Evidence validation script — 10 cases coverage.

Runs without DeepAgent by default (--no-llm mode) to validate:
  - Intent routing
  - Collection selection
  - Chroma retrieval
  - Evidence builder
  - source_type mapping
  - duplicate detection

With --with-agent, also runs through /api/agent/query for full E2E.

Usage:
    # Quick: validate retrieval + evidence only (no LLM)
    D:/AI/soft/conda/envs/python3.11/python.exe backend/scripts/debug_rag_evidence.py

    # With real DeepAgent
    D:/AI/soft/conda/envs/python3.11/python.exe backend/scripts/debug_rag_evidence.py --with-agent

    # Custom top-k
    D:/AI/soft/conda/envs/python3.11/python.exe backend/scripts/debug_rag_evidence.py --top-k 3
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

# ── Ensure backend is on path ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from app.rag.chroma_store import ChromaStore
from app.rag.collection_selector import CollectionSelector
from app.rag.evidence import build_evidence_pack
from app.rag.planner import RagPlanner
from app.schemas.consultation import Source
from app.services.agent_router import route_intent
from app.services.consultation_service import _create_retriever


# ── Test cases ───────────────────────────────────────────────────

TEST_CASES = [
    # Advisory
    {
        "id": "ADV-001",
        "question": "我是稳健型投资者，想做三年资产配置，应该怎么分配资产类别？",
        "expected_intent": "advisory",
        "expected_collections": ["advisory_knowledge", "risk_knowledge"],
        "min_sources": 2,
    },
    {
        "id": "ADV-002",
        "question": "基金定投适合什么样的人？有哪些风险需要注意？",
        "expected_intent": "advisory",
        "expected_collections": ["advisory_knowledge"],
        "min_sources": 1,
    },
    # Financial Report
    {
        "id": "FR-001",
        "question": "营业收入增长但净利润下降，经营现金流为正，这种财报怎么看？",
        "expected_intent": "financial_report",
        "expected_collections": ["advisory_knowledge", "risk_knowledge"],
        "min_sources": 1,
    },
    {
        "id": "FR-002",
        "question": "毛利率下降、应收账款周转变慢说明企业可能存在什么问题？",
        "expected_intent": "financial_report",
        "expected_collections": ["advisory_knowledge", "risk_knowledge"],
        "min_sources": 1,
    },
    # Risk Control
    {
        "id": "RISK-001",
        "question": "组合权益类占比70%，回撤很大，怎么审查风险？",
        "expected_intent": "risk_control",
        "expected_collections": ["risk_knowledge"],
        "min_sources": 1,
    },
    {
        "id": "RISK-002",
        "question": "仓位集中度太高会带来哪些风险？如何管理集中度风险？",
        "expected_intent": "risk_control",
        "expected_collections": ["risk_knowledge"],
        "min_sources": 1,
    },
    # Compliance
    {
        "id": "CMP-001",
        "question": "这句话术'保证年化收益8%'合规吗？",
        "expected_intent": "compliance",
        "expected_collections": ["compliance_knowledge"],
        "min_sources": 1,
    },
    {
        "id": "CMP-002",
        "question": "在营销材料里直接写目标价30元可以吗？合规风险有哪些？",
        "expected_intent": "compliance",
        "expected_collections": ["compliance_knowledge"],
        "min_sources": 1,
    },
    # Education
    {
        "id": "EDU-001",
        "question": "什么是基金定投？新手怎么理解定投的原理和风险？",
        "expected_intent": "education",
        "expected_collections": ["education_knowledge"],
        "min_sources": 1,
    },
    {
        "id": "EDU-002",
        "question": "如何识别'保本高收益'骗局？常见的金融诈骗套路有哪些？",
        "expected_intent": "education",
        "expected_collections": ["education_knowledge", "compliance_knowledge", "risk_knowledge"],
        "min_sources": 1,
    },
]


# ── Helpers ──────────────────────────────────────────────────────

def check_rag_environment():
    """Verify RAG is enabled and Chroma/MYSQL are configured."""
    from app.core.config import settings

    print("=" * 70)
    print("  RAG Environment Check")
    print("=" * 70)

    checks = []
    # RAG enabled
    rag_ok = settings.rag_enabled
    checks.append(("FIN_AGENT_RAG_ENABLED=true", rag_ok))

    # Chroma dir
    chroma_ok = os.path.isdir(settings.chroma_dir)
    checks.append((f"Chroma dir exists: {settings.chroma_dir}", chroma_ok))

    # API key
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    api_ok = len(api_key) > 0
    checks.append(("DASHSCOPE_API_KEY set", api_ok))

    # MySQL
    mysql_host = os.getenv("MYSQL_HOST", "")
    mysql_ok = len(mysql_host) > 0
    checks.append(("MYSQL_HOST set", mysql_ok))

    all_ok = True
    for label, ok in checks:
        status = "OK" if ok else "MISSING"
        print(f"  [{status:>7}] {label}")
        if not ok:
            all_ok = False

    print()
    return all_ok


def check_chroma_collections(store):
    """List all Chroma collections with counts."""
    print("=" * 70)
    print("  Chroma Collections")
    print("=" * 70)

    collections = store.list_collections()
    if not collections:
        print("  [EMPTY] No collections found!")
        return {}, 0

    total = 0
    coll_info = {}
    for col in collections:
        name = col.get("name", "")
        count = col.get("count", 0)
        total += count
        coll_info[name] = count
        print(f"  [{name}] → {count} chunks")

    print(f"\n  Total: {total} chunks across {len(collections)} collections")
    print()
    return coll_info, total


def check_mysql_metadata():
    """Compare MySQL metadata counts with Chroma."""
    from app.core.config import settings

    print("=" * 70)
    print("  MySQL Metadata Check")
    print("=" * 70)

    if not settings.mysql_enabled:
        print("  [SKIP] MySQL not enabled (FIN_AGENT_MYSQL_ENABLED not set)")
        print()
        return None

    try:
        from app.rag.metadata_store import KnowledgeMetadataStore

        with KnowledgeMetadataStore() as store:
            doc_count = store.get_document_count()
            chunk_count = store.get_chunk_count()
            coll_doc_counts = store.get_document_counts_by_collection()
            coll_chunk_counts = store.get_chunk_counts_by_collection()

            print(f"  Documents: {doc_count}")
            print(f"  Chunks:    {chunk_count}")
            for coll in sorted(coll_chunk_counts.keys()):
                dc = coll_doc_counts.get(coll, 0)
                cc = coll_chunk_counts.get(coll, 0)
                print(f"  [{coll}] → {dc} docs, {cc} chunks")
            print()
            return {"doc_count": doc_count, "chunk_count": chunk_count,
                    "by_collection": coll_chunk_counts}
    except Exception as exc:
        print(f"  [ERROR] MySQL check failed: {exc}")
        print()
        return None


def compare_chroma_mysql(chroma_info, mysql_info):
    """Compare Chroma and MySQL chunk counts."""
    if not mysql_info:
        print("  [SKIP] Cannot compare — MySQL not available")
        return

    print("=" * 70)
    print("  Chroma ↔ MySQL Comparison")
    print("=" * 70)

    total_match = True
    for coll_name, chroma_count in sorted(chroma_info.items()):
        mysql_count = mysql_info.get("by_collection", {}).get(coll_name, 0)
        match = "OK" if chroma_count == mysql_count else "MISMATCH"
        if chroma_count != mysql_count:
            total_match = False
        print(f"  [{match:>8}] {coll_name}: Chroma={chroma_count}, MySQL={mysql_count}")

    print(f"\n  Overall: {'MATCH' if total_match else 'MISMATCH — consider re-running ingest_knowledge.py --sync-mysql'}")
    print()


def run_single_case(case, top_k, retriever, planner, selector):
    """Run a single RAG evidence case (no LLM)."""
    question = case["question"]
    expected = case["expected_intent"]

    # Step 1: Route intent
    decision = route_intent(question)
    routed = decision.selected_intent
    intent_ok = (routed == expected)

    # Step 2: Get collections
    collections = selector.get_collections(routed, chroma_store=retriever._chroma_store)
    # Also try with retriever (which uses secondary intent detection)
    colls_from_retriever = _get_actual_collections(question, routed)

    # Step 3: Build queries & retrieve
    from app.schemas.consultation import ConsultationRequest
    req = ConsultationRequest(question=question, user_profile={})
    queries = planner.build_queries(req, routed)
    sources = retriever.retrieve(queries)

    # Step 4: Evidence
    evidence = build_evidence_pack(sources)

    # Step 5: Analysis
    source_count = len(sources)
    evidence_count = len(evidence.items)
    unknown_count = sum(
        1 for it in evidence.items if it.evidence_type == "unknown"
    )
    # Detect duplicates: check if any titles repeat
    titles = [s.title for s in sources]
    duplicate_count = len(titles) - len(set(titles))
    conf_values = [s.confidence for s in sources if s.confidence is not None]
    conf_range = f"{min(conf_values):.2f}–{max(conf_values):.2f}" if conf_values else "N/A"
    top3_titles = [s.title[:40] for s in sources[:3]]

    # Determine pass/fail
    expected_collections = case.get("expected_collections", [])
    coll_ok = any(ec in str(collections) for ec in expected_collections)
    sources_ok = source_count >= case.get("min_sources", 1)
    unknown_ok = (unknown_count / max(evidence_count, 1)) < 0.5 if evidence_count > 0 else True
    passed = intent_ok and sources_ok and unknown_ok

    return {
        "id": case["id"],
        "question": question[:60],
        "expected_intent": expected,
        "routed_intent": routed,
        "intent_ok": intent_ok,
        "collections": collections,
        "source_count": source_count,
        "evidence_count": evidence_count,
        "unknown_count": unknown_count,
        "duplicate_count": duplicate_count,
        "conf_range": conf_range,
        "top_titles": top3_titles,
        "evidence_types": list(set(it.evidence_type for it in evidence.items)),
        "passed": passed,
        "coll_ok": coll_ok,
        "sources_ok": sources_ok,
        "unknown_ok": unknown_ok,
    }


def _get_actual_collections(question, intent):
    """Get collections the retriever would actually use (with secondary intent)."""
    try:
        from app.rag.secondary_intent_detector import SecondaryIntentDetector
        detector = SecondaryIntentDetector()
        secondary = detector.detect(question, primary_intent=intent)
        selector = CollectionSelector()
        return selector.get_collections_for_intents(intent, secondary)
    except Exception:
        return [intent]


def run_with_agent(case):
    """Run a case through the full /api/agent/query pipeline."""
    from app.api.routes.agent_query import _get_agent_for_intent, _get_architecture_info, _build_evidence_debug
    from app.rag.planner import RagPlanner
    from app.schemas.consultation import ConsultationRequest
    from app.services.agent_router import route_intent
    from app.services.compliance_guard import ComplianceGuard
    from app.services.consultation_service import _create_retriever

    question = case["question"]
    decision = route_intent(question)
    intent = decision.selected_intent

    try:
        agent = _get_agent_for_intent(intent, "pipeline")  # Use pipeline for speed
    except ValueError:
        return None

    planner = RagPlanner()
    retriever = _create_retriever()
    req = ConsultationRequest(question=question, user_profile={})
    queries = planner.build_queries(req, intent)
    sources = retriever.retrieve(queries)

    try:
        raw = agent.answer(req, sources)
        guard = ComplianceGuard()
        response = guard.review(raw)
        evidence = _build_evidence_debug(build_evidence_pack(response.sources))
        arch = _get_architecture_info(agent)

        source_titles = [s.title if hasattr(s, 'title') else str(s) for s in response.sources[:3]]
        return {
            "intent": intent,
            "agent": response.agent,
            "answer_len": len(response.answer),
            "sources": len(response.sources),
            "evidence_items": evidence.item_count,
            "fallback": arch.fallback_used,
            "top_sources": source_titles,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Real RAG Evidence Validation")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K results per search")
    parser.add_argument("--no-llm", action="store_true", default=True,
                        help="Only test retrieval + evidence (default)")
    parser.add_argument("--with-agent", action="store_true",
                        help="Also run through /api/agent/query pipeline")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs to run (default: all)")
    args = parser.parse_args()

    t_start = time.perf_counter()

    # ── Env check ──────────────────────────────────────────────────
    env_ok = check_rag_environment()
    if not env_ok:
        print("[WARN] RAG environment not fully configured. ")
        print("Set FIN_AGENT_RAG_ENABLED=true and DASHSCOPE_API_KEY.\n")
        print("Proceeding with available configuration...\n")

    # ── Chroma check ───────────────────────────────────────────────
    from app.core.config import settings

    store = ChromaStore(persist_dir=settings.chroma_dir)
    chroma_info, total_chunks = check_chroma_collections(store)

    if total_chunks == 0:
        print("[FATAL] No Chroma data. Run ingest_knowledge.py first.")
        return 1

    # ── MySQL check ────────────────────────────────────────────────
    mysql_info = check_mysql_metadata()
    compare_chroma_mysql(chroma_info, mysql_info)

    # ── Prepare components ─────────────────────────────────────────
    retriever = _create_retriever()
    planner = RagPlanner()
    selector = CollectionSelector()

    if not retriever.has_chroma and not settings.rag_enabled:
        print("[WARN] Retriever has no Chroma. Sources will be mock!")
        print("Set FIN_AGENT_RAG_ENABLED=true and DASHSCOPE_API_KEY.\n")
        # Continue anyway — we want to see what happens

    # ── Run cases ──────────────────────────────────────────────────
    target_cases = TEST_CASES
    if args.cases:
        ids = set(args.cases.split(","))
        target_cases = [c for c in TEST_CASES if c["id"] in ids]

    print("=" * 70)
    print(f"  RAG Evidence Validation — {len(target_cases)} cases (top-k={args.top_k})")
    print(f"  Chroma: {'available' if retriever.has_chroma else 'mock-only'}")
    print("=" * 70)

    results = []
    for idx, case in enumerate(target_cases):
        r = run_single_case(case, args.top_k, retriever, planner, selector)
        results.append(r)

        status = "PASS" if r["passed"] else "FAIL"
        print(f"\n  ── [{status}] {r['id']}: {r['question']}…")
        print(f"    Intent: {r['routed_intent']} (expected {r['expected_intent']}) {'✓' if r['intent_ok'] else '✗'}")
        print(f"    Collections: {r['collections']}")
        print(f"    Sources: {r['source_count']} (evidence: {r['evidence_count']}, unknown: {r['unknown_count']}, dup: {r['duplicate_count']})")
        print(f"    Evidence types: {r['evidence_types']}")
        print(f"    Confidence: {r['conf_range']}")
        if r["top_titles"]:
            print(f"    Top titles: {', '.join(r['top_titles'][:3])}")

    # ── Summary ────────────────────────────────────────────────────
    passed = sum(1 for r in results if r["passed"])
    total_sources_count = sum(r["source_count"] for r in results)
    total_unknown = sum(r["unknown_count"] for r in results)
    total_dup = sum(r["duplicate_count"] for r in results)
    unknown_rate = total_unknown / max(sum(r["evidence_count"] for r in results), 1)

    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  Cases passed:     {passed}/{len(results)}")
    print(f"  Intent accuracy:  {sum(1 for r in results if r['intent_ok'])}/{len(results)}")
    print(f"  Total sources:    {total_sources_count}")
    print(f"  Unknown evidence: {total_unknown} ({unknown_rate:.0%})")
    print(f"  Duplicates found: {total_dup}")
    print(f"  Elapsed:          {time.perf_counter() - t_start:.1f}s")

    # Per-case table
    print(f"\n  ── Case Detail Table ──")
    print(f"  {'ID':<9} {'Intent':<18} {'Sources':>7} {'Evidence':>8} {'Unknown':>7} {'Dup':>4} {'Result':>6}")
    print(f"  {'-'*9} {'-'*18} {'-'*7} {'-'*8} {'-'*7} {'-'*4} {'-'*6}")
    for r in results:
        print(f"  {r['id']:<9} {r['routed_intent']:<18} {r['source_count']:>7} {r['evidence_count']:>8} {r['unknown_count']:>7} {r['duplicate_count']:>4} {'PASS' if r['passed'] else 'FAIL':>6}")

    # ── Agent test (optional) ──────────────────────────────────────
    if args.with_agent:
        print(f"\n{'=' * 70}")
        print(f"  /api/agent/query Pipeline Test (pipeline mode)")
        print(f"{'=' * 70}")
        agent_results = []
        for case in target_cases[:3]:  # First 3 only to save time
            ar = run_with_agent(case)
            agent_results.append(ar)
            if ar:
                if "error" in ar:
                    print(f"  [{case['id']}] ERROR: {ar['error']}")
                else:
                    print(f"  [{case['id']}] intent={ar['intent']}, sources={ar['sources']}, evidence={ar['evidence_items']}, answer_len={ar['answer_len']}")
                    if ar.get("top_sources"):
                        for ts in ar["top_sources"]:
                            print(f"    → {ts}")

    return 0 if passed >= 8 else 1


if __name__ == "__main__":
    sys.exit(main())
