#!/usr/bin/env python
"""
RAG Retrieval Quality Evaluation Script.

Evaluates the Chroma knowledge base retrieval quality against a curated set
of test cases. Measures intent routing accuracy, collection coverage, top-1
accuracy, recall@k, and MRR.

Usage:
    # Dry-run: validate case format only, no API calls
    python backend/scripts/evaluate_rag.py --dry-run

    # Real evaluation
    python backend/scripts/evaluate_rag.py --top-k 5

    # Custom thresholds
    python backend/scripts/evaluate_rag.py --fail-under-top1 0.60 --fail-under-recall 0.80

Requirements:
    - DASHSCOPE_API_KEY must be set (for real evaluation)
    - Chroma data must exist at data/chroma
    - FIN_AGENT_RAG_ENABLED not required (script bypasses ConsultationService)
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chroma_store import ChromaStore
from app.rag.collection_selector import CollectionSelector
from app.rag.embedding_provider import EmbeddingProvider
from app.rag.secondary_intent_detector import SecondaryIntentDetector
from app.services.intent_router import IntentRouter

# ── Default paths ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "backend" / "eval" / "rag_eval_cases.json"
DEFAULT_JSON_OUT = PROJECT_ROOT / "backend" / "eval" / "rag_eval_results.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "backend" / "eval" / "rag_eval_report.md"
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma")

# ── Required case fields ─────────────────────────────────────────────
REQUIRED_CASE_FIELDS = [
    "id", "query", "expected_intent", "expected_collections",
    "expected_titles", "difficulty", "category", "notes",
]
VALID_INTENTS = {"advisory", "financial_report", "risk_control", "compliance", "education"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality against curated test cases."
    )
    parser.add_argument(
        "--cases", type=str, default=str(DEFAULT_CASES_PATH),
        help=f"Path to eval cases JSON (default: {DEFAULT_CASES_PATH})",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top results to consider for recall@k (default: 5).",
    )
    parser.add_argument(
        "--json-output", type=str, default=str(DEFAULT_JSON_OUT),
        help=f"Path for JSON results output (default: {DEFAULT_JSON_OUT})",
    )
    parser.add_argument(
        "--markdown-output", type=str, default=str(DEFAULT_MD_OUT),
        help=f"Path for Markdown report output (default: {DEFAULT_MD_OUT})",
    )
    parser.add_argument(
        "--fail-under-top1", type=float, default=0.70,
        help="Fail if top-1 accuracy is below this threshold (default: 0.70).",
    )
    parser.add_argument(
        "--fail-under-recall", type=float, default=0.85,
        help="Fail if recall@k is below this threshold (default: 0.85).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate case format only; do not call embedding API or Chroma.",
    )
    parser.add_argument(
        "--no-api-check", action="store_true",
        help="Same as --dry-run. Validate case format only.",
    )
    args = parser.parse_args()

    dry_run = args.dry_run or args.no_api_check

    # ── Load cases ─────────────────────────────────────────────────
    print("=" * 60)
    print("  RAG Retrieval Quality Evaluation")
    print("=" * 60)

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"\n[ERROR] Cases file not found: {cases_path}")
        return 1

    try:
        with open(cases_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Invalid JSON in cases file: {e}")
        return 1

    cases = data.get("cases", [])
    if not isinstance(cases, list):
        print("[ERROR] 'cases' must be a list.")
        return 1

    # ── Validate case format ───────────────────────────────────────
    print(f"\n[1] Validating {len(cases)} cases...")
    validation_errors = validate_cases(cases)

    if validation_errors:
        print(f"\n  [FAIL] {len(validation_errors)} validation error(s):")
        for err in validation_errors:
            print(f"    - {err}")
        return 1
    print(f"  [OK] All {len(cases)} cases valid.")

    # Summary counts
    intent_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    for c in cases:
        intent_counts[c["expected_intent"]] = intent_counts.get(c["expected_intent"], 0) + 1
        diff_counts[c["difficulty"]] = diff_counts.get(c["difficulty"], 0) + 1
    hard_count = diff_counts.get("hard", 0)
    cross_count = sum(1 for c in cases if c["id"].startswith("XDM-"))

    print(f"  Intents: {intent_counts}")
    print(f"  Difficulty: {diff_counts}")
    print(f"  Hard cases: {hard_count}, Cross-domain: {cross_count}")

    if dry_run:
        print(f"\n  [DRY-RUN] Case format validation passed. No API calls made.")
        return 0

    # ── Check prerequisites ────────────────────────────────────────
    print("\n[2] Checking prerequisites...")

    if not os.path.isdir(CHROMA_DIR):
        print(f"  [ERROR] Chroma directory not found: {CHROMA_DIR}")
        print("  Run ingest_knowledge.py first.")
        return 1

    provider = EmbeddingProvider()
    if not provider.is_configured:
        print("  [ERROR] DASHSCOPE_API_KEY is not set.")
        print("  Set it via: $env:DASHSCOPE_API_KEY = 'sk-...'")
        print("  Or use --dry-run to validate cases without API calls.")
        return 1
    print(f"  [OK] Embedding provider configured (model={provider.model})")

    store = ChromaStore(persist_dir=CHROMA_DIR, embedding_provider=provider)
    collections = store.list_collections()
    total_vectors = sum(c.get("count", 0) for c in collections)
    print(f"  [OK] Chroma: {len(collections)} collections, {total_vectors} vectors")

    # ── Run evaluation ─────────────────────────────────────────────
    print(f"\n[3] Running evaluation (top-k={args.top_k})...")

    router = IntentRouter()
    selector = CollectionSelector()
    secondary_detector = SecondaryIntentDetector()
    start_time = time.time()

    results = run_evaluation(cases, router, selector, secondary_detector, store, args.top_k)

    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.1f}s")

    # ── Calculate metrics ──────────────────────────────────────────
    metrics = calculate_metrics(results)

    # ── Terminal summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Cases:              {metrics['total_cases']}")
    print(f"  Intent Accuracy:    {metrics['intent_accuracy']:.2%}")
    print(f"  Collection Coverage:{metrics['collection_coverage']:.2%}")
    print(f"  Top-1 Accuracy:     {metrics['top1_accuracy']:.2%}")
    print(f"  Recall@{args.top_k}:        {metrics['recall_at_k']:.2%}")
    print(f"  MRR:                {metrics['mrr']:.4f}")

    # ── Threshold check ────────────────────────────────────────────
    top1_pass = metrics["top1_accuracy"] >= args.fail_under_top1
    recall_pass = metrics["recall_at_k"] >= args.fail_under_recall
    intent_pass = metrics["intent_accuracy"] >= 0.90

    print(f"\n  Thresholds:")
    print(f"    Top-1 >= {args.fail_under_top1:.0%}:  {'PASS' if top1_pass else 'FAIL'}")
    print(f"    Recall >= {args.fail_under_recall:.0%}: {'PASS' if recall_pass else 'FAIL'}")
    print(f"    Intent >= 0.90: {'PASS' if intent_pass else 'FAIL'}")

    if top1_pass and recall_pass and intent_pass:
        print("\n  >>> OVERALL: PASS <<<")
    elif top1_pass and recall_pass:
        print("\n  >>> OVERALL: WARN (intent below threshold) <<<")
    else:
        print("\n  >>> OVERALL: FAIL <<<")

    # ── Write outputs ──────────────────────────────────────────────
    print(f"\n[4] Writing outputs...")

    # JSON results
    json_output = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": sys.executable,
            "top_k": args.top_k,
            "chroma_collections": len(collections),
            "chroma_vectors": total_vectors,
            "thresholds": {
                "top1_accuracy": args.fail_under_top1,
                "recall_at_k": args.fail_under_recall,
                "intent_accuracy": 0.90,
            },
        },
        "metrics": metrics,
        "results": results,
    }

    json_out = Path(args.json_output)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)
    print(f"  [OK] JSON: {json_out}")

    # Markdown report
    md_report = build_markdown_report(
        metrics, results, args, total_vectors, len(collections), elapsed,
        top1_pass, recall_pass, intent_pass,
    )
    md_out = Path(args.markdown_output)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    with open(md_out, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"  [OK] Markdown: {md_out}")

    print("\nDone.")
    return 0 if (top1_pass and recall_pass and intent_pass) else 1


# ── Validation ────────────────────────────────────────────────────────


def validate_cases(cases: list[dict]) -> list[str]:
    """Validate eval case format. Returns list of error strings."""
    errors: list[str] = []
    seen_ids: set[str] = set()

    if len(cases) < 32:
        errors.append(f"Expected >= 32 cases, got {len(cases)}")

    intent_counts: dict[str, int] = {}
    hard_count = 0
    xdm_count = 0

    for i, case in enumerate(cases):
        case_id = case.get("id", f"<case #{i}>")

        # Check required fields
        for field in REQUIRED_CASE_FIELDS:
            if field not in case:
                errors.append(f"{case_id}: missing field '{field}'")

        # Check id uniqueness
        if case.get("id") in seen_ids:
            errors.append(f"{case_id}: duplicate id")
        seen_ids.add(case.get("id", ""))

        # Validate intent
        intent = case.get("expected_intent", "")
        if intent and intent not in VALID_INTENTS:
            errors.append(f"{case_id}: invalid expected_intent '{intent}'")
        if intent:
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

        # Validate difficulty
        diff = case.get("difficulty", "")
        if diff and diff not in VALID_DIFFICULTIES:
            errors.append(f"{case_id}: invalid difficulty '{diff}'")
        if diff == "hard":
            hard_count += 1

        # Validate expected_titles
        titles = case.get("expected_titles", [])
        if not isinstance(titles, list) or len(titles) == 0:
            errors.append(f"{case_id}: expected_titles must be non-empty list")
        elif len(titles) > 3:
            errors.append(f"{case_id}: expected_titles has >3 entries")

        # Validate expected_collections
        colls = case.get("expected_collections", [])
        if not isinstance(colls, list) or len(colls) == 0:
            errors.append(f"{case_id}: expected_collections must be non-empty list")

        # Count cross-domain
        cid = case.get("id", "")
        if isinstance(cid, str) and cid.startswith("XDM-"):
            xdm_count += 1

    # Check per-intent minimums (at least 8 per collection)
    intent_to_label = {
        "advisory": "advisory_knowledge",
        "compliance": "compliance_knowledge",
        "education": "education_knowledge",
        "risk_control": "risk_knowledge",
    }
    for intent, label in intent_to_label.items():
        count = intent_counts.get(intent, 0)
        # Cross-domain cases may also count; count cases where this intent is expected
        direct_count = sum(
            1 for c in cases
            if c.get("expected_intent") == intent
        )
        # Also include XDM cases that list this collection in expected_collections
        coll_count = sum(
            1 for c in cases
            if label in c.get("expected_collections", [])
        )
        if coll_count < 8:
            errors.append(
                f"Collection '{label}' has only {coll_count} cases with expected_collections "
                f"containing it (need >=8)"
            )

    if hard_count < 8:
        errors.append(f"Only {hard_count} hard cases (need >=8)")

    if xdm_count < 6:
        errors.append(f"Only {xdm_count} cross-domain cases (need >=6)")

    return errors


# ── Evaluation core ───────────────────────────────────────────────────


def run_evaluation(
    cases: list[dict],
    router: IntentRouter,
    selector: CollectionSelector,
    secondary_detector: SecondaryIntentDetector,
    store: ChromaStore,
    top_k: int,
) -> list[dict]:
    """Run evaluation for each case. Returns list of per-case result dicts."""
    results: list[dict] = []

    for case in cases:
        case_id = case["id"]
        query = case["query"]
        expected_intent = case["expected_intent"]
        expected_collections = set(case["expected_collections"])
        expected_titles = case["expected_titles"]
        difficulty = case.get("difficulty", "medium")

        item: dict[str, Any] = {
            "id": case_id,
            "query": query,
            "expected_intent": expected_intent,
            "expected_collections": list(expected_collections),
            "expected_titles": expected_titles,
            "difficulty": difficulty,
            "category": case.get("category", ""),
            # Results (filled below)
            "actual_intent": None,
            "secondary_intents": [],
            "intent_correct": False,
            "selected_collections": [],
            "collection_correct": False,
            "top_results": [],
            "top1_title": None,
            "top1_score": None,
            "top1_correct": False,
            "recall_correct": False,
            "first_hit_rank": None,
            "errors": [],
        }

        try:
            # a. Intent routing
            actual_intent = router.route(query)
            item["actual_intent"] = actual_intent
            item["intent_correct"] = (actual_intent == expected_intent)

            # b. Secondary intent detection
            secondary = secondary_detector.detect(query, primary_intent=actual_intent)
            item["secondary_intents"] = secondary

            # c. Collection selection (multi-intent)
            selected = selector.get_collections_for_intents(
                actual_intent, secondary, store
            )
            item["selected_collections"] = selected
            item["collection_correct"] = expected_collections.issubset(set(selected))

            # c & d. Chroma search
            if selected:
                search_results = store.search(query, selected, top_k=top_k)
            else:
                search_results = []

            # Format top results
            top_results_formatted: list[dict] = []
            found_expected: set[str] = set()
            first_hit_rank = None

            for rank, sr in enumerate(search_results, 1):
                title = sr.get("title", "")
                score = sr.get("score", 0.0)
                top_results_formatted.append({
                    "rank": rank,
                    "title": title,
                    "score": round(score, 4),
                    "collection": sr.get("collection_name", ""),
                    "source_type": sr.get("source_type", ""),
                })

                # Check if this title is in expected_titles
                for et in expected_titles:
                    if et in title or title in et:
                        found_expected.add(et)
                        if first_hit_rank is None:
                            first_hit_rank = rank

            item["top_results"] = top_results_formatted
            item["top1_title"] = top_results_formatted[0]["title"] if top_results_formatted else None
            item["top1_score"] = top_results_formatted[0]["score"] if top_results_formatted else None
            item["top1_correct"] = (
                item["top1_title"] is not None and
                any(et in item["top1_title"] or item["top1_title"] in et for et in expected_titles)
            )
            item["recall_correct"] = len(found_expected) > 0
            item["first_hit_rank"] = first_hit_rank

        except Exception as e:
            item["errors"].append(str(e))

        results.append(item)

    return results


# ── Metrics calculation ───────────────────────────────────────────────


def calculate_metrics(results: list[dict]) -> dict[str, Any]:
    """Calculate aggregate metrics from per-case results."""
    total = len(results)

    # Overall metrics
    intent_correct = sum(1 for r in results if r["intent_correct"])
    collection_correct = sum(1 for r in results if r["collection_correct"])
    top1_correct = sum(1 for r in results if r["top1_correct"])
    recall_correct = sum(1 for r in results if r["recall_correct"])

    # MRR
    reciprocal_ranks: list[float] = []
    for r in results:
        rank = r.get("first_hit_rank")
        if rank is not None and rank > 0:
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / total if total > 0 else 0.0

    metrics: dict[str, Any] = {
        "total_cases": total,
        "intent_accuracy": intent_correct / total if total > 0 else 0.0,
        "collection_coverage": collection_correct / total if total > 0 else 0.0,
        "top1_accuracy": top1_correct / total if total > 0 else 0.0,
        "recall_at_k": recall_correct / total if total > 0 else 0.0,
        "mrr": round(mrr, 4),
        "intent_correct_count": intent_correct,
        "collection_correct_count": collection_correct,
        "top1_correct_count": top1_correct,
        "recall_correct_count": recall_correct,
    }

    # By intent
    intents = sorted(set(
        r["expected_intent"] for r in results if r["expected_intent"]
    ))
    by_intent: dict[str, dict] = {}
    for intent in intents:
        subset = [r for r in results if r["expected_intent"] == intent]
        n = len(subset)
        if n == 0:
            continue
        by_intent[intent] = _subset_metrics(subset, n)

    metrics["by_intent"] = by_intent

    # By difficulty
    difficulties = sorted(set(
        r.get("difficulty", "medium") for r in results
    ))
    by_difficulty: dict[str, dict] = {}
    for diff in difficulties:
        subset = [r for r in results if r.get("difficulty") == diff]
        n = len(subset)
        if n == 0:
            continue
        by_difficulty[diff] = _subset_metrics(subset, n)

    metrics["by_difficulty"] = by_difficulty

    # Low-score warnings (top1_score < 0.45 and top1 was a hit, or any result with top1 < 0.35)
    low_score_items: list[dict] = []
    for r in results:
        score = r.get("top1_score")
        if score is not None and score < 0.45:
            low_score_items.append({
                "id": r["id"],
                "query": r["query"],
                "top1_title": r.get("top1_title"),
                "top1_score": score,
                "top1_correct": r["top1_correct"],
            })
    metrics["low_score_count"] = len(low_score_items)
    metrics["low_score_items"] = low_score_items

    return metrics


def _subset_metrics(subset: list[dict], n: int) -> dict:
    return {
        "count": n,
        "intent_accuracy": sum(1 for r in subset if r["intent_correct"]) / n,
        "collection_coverage": sum(1 for r in subset if r["collection_correct"]) / n,
        "top1_accuracy": sum(1 for r in subset if r["top1_correct"]) / n,
        "recall_at_k": sum(1 for r in subset if r["recall_correct"]) / n,
        "mrr": round(
            sum(
                (1.0 / r["first_hit_rank"]) if r.get("first_hit_rank") else 0.0
                for r in subset
            ) / n, 4
        ),
    }


# ── Markdown report builder ────────────────────────────────────────────


def build_markdown_report(
    metrics: dict,
    results: list[dict],
    args: argparse.Namespace,
    total_vectors: int,
    num_collections: int,
    elapsed: float,
    top1_pass: bool,
    recall_pass: bool,
    intent_pass: bool,
) -> str:
    """Build a detailed Markdown evaluation report."""

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append("# RAG Retrieval Quality Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {now_str}")
    lines.append(f"**Python:** `{sys.executable}`")
    lines.append(f"**Top-K:** {args.top_k}")
    lines.append("")

    # ── 1. Environment ─────────────────────────────────────────────
    lines.append("## 1. Execution Environment")
    lines.append("")
    lines.append(f"| Item | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Timestamp | {now_str} |")
    lines.append(f"| Python | `{sys.executable}` |")
    lines.append(f"| Chroma collections | {num_collections} |")
    lines.append(f"| Chroma vectors | {total_vectors} |")
    lines.append(f"| Eval cases | {metrics['total_cases']} |")
    lines.append(f"| Top-K | {args.top_k} |")
    lines.append(f"| Elapsed | {elapsed:.1f}s |")
    lines.append("")

    # ── 2. Overall Metrics ─────────────────────────────────────────
    lines.append("## 2. Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")
    lines.append(
        f"| Cases | {metrics['total_cases']} | — | — |"
    )
    lines.append(
        f"| Intent Accuracy | {metrics['intent_accuracy']:.2%} | ≥ 0.90 | "
        f"{'✅ PASS' if intent_pass else '❌ FAIL'} |"
    )
    lines.append(
        f"| Collection Coverage | {metrics['collection_coverage']:.2%} | — | — |"
    )
    lines.append(
        f"| Top-1 Accuracy | {metrics['top1_accuracy']:.2%} | ≥ {args.fail_under_top1:.0%} | "
        f"{'✅ PASS' if top1_pass else '❌ FAIL'} |"
    )
    lines.append(
        f"| Recall@{args.top_k} | {metrics['recall_at_k']:.2%} | ≥ {args.fail_under_recall:.0%} | "
        f"{'✅ PASS' if recall_pass else '❌ FAIL'} |"
    )
    lines.append(
        f"| MRR | {metrics['mrr']:.4f} | — | — |"
    )
    lines.append("")

    # Overall verdict
    if top1_pass and recall_pass and intent_pass:
        verdict = "✅ PASS"
    elif top1_pass and recall_pass:
        verdict = "⚠️ WARN — intent accuracy below threshold"
    else:
        verdict = "❌ FAIL — retrieval metrics below threshold"
    lines.append(f"**Overall Verdict: {verdict}**")
    lines.append("")

    # ── 3. By Intent ───────────────────────────────────────────────
    lines.append("## 3. Metrics by Expected Intent")
    lines.append("")
    lines.append("| Intent | Cases | Intent Acc | Coll Cov | Top-1 Acc | Recall@K | MRR |")
    lines.append("|--------|-------|------------|----------|-----------|----------|-----|")

    for intent, m in metrics.get("by_intent", {}).items():
        lines.append(
            f"| {intent} | {m['count']} | {m['intent_accuracy']:.2%} | "
            f"{m['collection_coverage']:.2%} | {m['top1_accuracy']:.2%} | "
            f"{m['recall_at_k']:.2%} | {m['mrr']:.4f} |"
        )
    lines.append("")

    # ── 4. By Difficulty ───────────────────────────────────────────
    lines.append("## 4. Metrics by Difficulty")
    lines.append("")
    lines.append("| Difficulty | Cases | Top-1 Acc | Recall@K | MRR |")
    lines.append("|------------|-------|-----------|----------|-----|")

    for diff, m in metrics.get("by_difficulty", {}).items():
        lines.append(
            f"| {diff} | {m['count']} | {m['top1_accuracy']:.2%} | "
            f"{m['recall_at_k']:.2%} | {m['mrr']:.4f} |"
        )
    lines.append("")

    # ── 5. Top Failures ────────────────────────────────────────────
    lines.append("## 5. Top Failures")
    lines.append("")

    # Collect failures
    failures: list[dict] = []
    for r in results:
        failure_types: list[str] = []
        if not r["intent_correct"]:
            failure_types.append("intent_error")
        if not r["collection_correct"]:
            failure_types.append("collection_miss")
        if not r["top1_correct"]:
            failure_types.append("top1_miss")
        if not r["recall_correct"]:
            failure_types.append("recall_miss")

        if failure_types:
            failures.append({**r, "failure_types": failure_types})

    # Sort: intent errors first, then recall misses, then top1 misses
    def failure_sort_key(f: dict) -> int:
        types = f["failure_types"]
        if "intent_error" in types:
            return 0
        if "recall_miss" in types:
            return 1
        if "top1_miss" in types:
            return 2
        return 3

    failures.sort(key=failure_sort_key)

    # Show top 15 failures
    top_failures = failures[:15]
    for i, f in enumerate(top_failures, 1):
        lines.append(f"### {i}. [{f['id']}] {f['query']}")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Expected Intent | `{f['expected_intent']}` |")
        lines.append(f"| Actual Intent | `{f['actual_intent']}` |")
        sec = f.get("secondary_intents", [])
        lines.append(f"| Secondary Intents | {', '.join(f'`{s}`' for s in sec) if sec else '—'} |")
        lines.append(f"| Selected Collections | {', '.join(f'`{c}`' for c in f.get('selected_collections', []))} |")
        lines.append(f"| Expected Titles | {', '.join(f['expected_titles'])} |")
        lines.append(f"| Difficulty | {f.get('difficulty', '—')} |")
        lines.append(f"| Failure Types | {', '.join(f['failure_types'])} |")
        lines.append("")

        # Top-3 actual results
        lines.append("**Top search results:**")
        lines.append("")
        lines.append("| Rank | Title | Score | Collection |")
        lines.append("|------|-------|-------|------------|")
        for tr in f.get("top_results", [])[:3]:
            lines.append(
                f"| {tr['rank']} | {tr['title']} | {tr['score']:.4f} | "
                f"{tr['collection']} |"
            )
        if not f.get("top_results"):
            lines.append("| — | *(no results)* | — | — |")
        lines.append("")

        # Diagnosis
        lines.append("**Diagnosis:**")
        lines.append("")
        if "intent_error" in f["failure_types"]:
            lines.append(f"- ❌ Intent misrouted: expected `{f['expected_intent']}`, "
                         f"got `{f['actual_intent']}`.")
            # Suggest fix
            lines.append(f"- **Possible fix:** Add/boost keywords for `{f['expected_intent']}` "
                         f"or check keyword overlap with `{f['actual_intent']}`.")
        if "collection_miss" in f["failure_types"]:
            lines.append(
                f"- ⚠️ Selected collections do not fully cover expected collections. "
                f"Expected: {', '.join(f['expected_collections'])}; "
                f"selected: {', '.join(f.get('selected_collections', []))}."
            )
            lines.append(
                "- **Possible fix:** For cross-domain questions, consider query decomposition "
                "or extending CollectionSelector to include secondary collections."
            )
        if "recall_miss" in f["failure_types"]:
            lines.append(f"- ❌ No expected title found in top-{args.top_k}.")
            lines.append(f"- **Possible fix:** Add knowledge document covering this topic, "
                         f"or adjust collection selector to include more collections.")
        if "top1_miss" in f["failure_types"] and "recall_miss" not in f["failure_types"]:
            lines.append(f"- ⚠️ Top-1 miss but recall hit. Ranking issue.")
            lines.append(f"- **Possible fix:** Improve embedding quality or add reranker.")
        lines.append("")

    if not failures:
        lines.append("✅ No failures! All cases passed.")
        lines.append("")

    # ── 6. Low Score Warnings ──────────────────────────────────────
    lines.append("## 6. Low Score Warnings")
    lines.append("")
    lines.append("Cases where top-1 result score < 0.45 (low confidence):")
    lines.append("")
    low_items = metrics.get("low_score_items", [])
    if low_items:
        lines.append("| ID | Query | Top-1 Title | Score | Hit? |")
        lines.append("|----|-------|-------------|-------|------|")
        for item in low_items:
            hit = "✅" if item["top1_correct"] else "❌"
            lines.append(
                f"| {item['id']} | {item['query']} | {item['top1_title']} | "
                f"{item['top1_score']:.4f} | {hit} |"
            )
        lines.append("")
        lines.append(f"**{len(low_items)} cases** have low top-1 scores. "
                     f"These may indicate weak semantic matches or knowledge gaps.")
    else:
        lines.append("No low-score cases detected.")
    lines.append("")

    # ── 7. Overall Assessment ──────────────────────────────────────
    lines.append("## 7. Overall Assessment")
    lines.append("")
    lines.append(f"**Verdict: {verdict}**")
    lines.append("")

    # Intent accuracy detail
    intent_accuracy = metrics["intent_accuracy"]
    if intent_accuracy < 0.90:
        intent_errors = [r for r in results if not r["intent_correct"]]
        lines.append(f"- Intent Accuracy is **below target** ({intent_accuracy:.2%} < 0.90). "
                     f"{len(intent_errors)} case(s) misrouted.")
        lines.append("  - Primary action: **adjust IntentRouter keywords** for affected intents.")
    else:
        lines.append(f"- Intent Accuracy **meets target** ({intent_accuracy:.2%} ≥ 0.90).")

    top1_accuracy = metrics["top1_accuracy"]
    if top1_accuracy < args.fail_under_top1:
        lines.append(f"- Top-1 Accuracy is **below target** ({top1_accuracy:.2%} < {args.fail_under_top1:.0%}).")
        lines.append("  - Primary action: **improve document coverage** and consider **reranking**.")
    else:
        lines.append(f"- Top-1 Accuracy **meets target** ({top1_accuracy:.2%} ≥ {args.fail_under_top1:.0%}).")

    recall = metrics["recall_at_k"]
    if recall < args.fail_under_recall:
        lines.append(f"- Recall@{args.top_k} is **below target** ({recall:.2%} < {args.fail_under_recall:.0%}).")
        lines.append("  - Primary action: **fill knowledge gaps** with targeted documents.")
    else:
        lines.append(f"- Recall@{args.top_k} **meets target** ({recall:.2%} ≥ {args.fail_under_recall:.0%}).")
    lines.append("")

    # ── 8. Recommendations ─────────────────────────────────────────
    lines.append("## 8. Next-Step Recommendations")
    lines.append("")
    lines.append("Priority-ordered recommendations based on evaluation results:")
    lines.append("")

    recs: list[tuple[int, str]] = []

    # Priority 1: Intent routing fixes
    intent_errors = [r for r in results if not r["intent_correct"]]
    if intent_errors:
        intent_pairs = sorted(set(
            f"{r['expected_intent']}→{r['actual_intent']}" for r in intent_errors
        ))
        recs.append((
            1,
            f"**Fix IntentRouter keywords** — {len(intent_errors)} misrouted cases. "
            f"Review keyword overlap for affected intent pairs: "
            f"{', '.join(intent_pairs)}."
        ))

    # Priority 2: Knowledge gaps
    recall_misses = [r for r in results if not r["recall_correct"]]
    if recall_misses:
        topics = [r["query"] for r in recall_misses]
        topic_text = (
            f"{', '.join(topics[:5])}..."
            if len(topics) > 5
            else f"{', '.join(topics)}."
        )
        recs.append((
            2,
            f"**Fill knowledge gaps** — {len(recall_misses)} cases with zero recall. "
            f"Missing topics include: {topic_text}"
        ))

    # Priority 3: Collection selector adjustments
    collection_misses = [r for r in results if not r["collection_correct"]]
    if collection_misses:
        recs.append((
            3,
            f"**Adjust CollectionSelector** — {len(collection_misses)} cases where "
            f"selected collections don't fully cover expected collections."
        ))

    # Priority 4: Reranker / hybrid
    top1_only_misses = [
        r for r in results
        if not r["top1_correct"] and r["recall_correct"]
    ]
    if top1_only_misses:
        recs.append((
            4,
            f"**Consider reranker or BM25 hybrid** — {len(top1_only_misses)} cases "
            f"have correct recall but wrong top-1. Ranking quality could be improved."
        ))

    # Priority 5: More documents
    recs.append((
        5,
        "**Expand knowledge base** — current 31 vectors across 4 collections "
        "is relatively small. Add 2-3 documents per collection to improve coverage."
    ))

    # Priority 6: Hard case analysis
    hard_failures = [
        r for r in results
        if r.get("difficulty") == "hard" and (not r["top1_correct"] or not r["recall_correct"])
    ]
    if hard_failures:
        recs.append((
            6,
            f"**Review hard case failures** — {len(hard_failures)} hard cases failed. "
            f"These may require specialized documents or hybrid retrieval."
        ))

    recs.sort(key=lambda x: x[0])
    for index, (_priority, text) in enumerate(recs, 1):
        lines.append(f"{index}. {text}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `backend/scripts/evaluate_rag.py` at {now_str}*")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
