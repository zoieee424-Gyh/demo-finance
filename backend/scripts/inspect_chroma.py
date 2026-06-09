#!/usr/bin/env python
"""
Inspect Chroma collections — list, count, and search.

Usage:
    python backend/scripts/inspect_chroma.py                       # List all collections
    python backend/scripts/inspect_chroma.py --query "风险厌恶"      # Search (default all collections)
    python backend/scripts/inspect_chroma.py --query "资产配置" --collection advisory_knowledge  # Specific collection
    python backend/scripts/inspect_chroma.py --details              # Full detail per collection
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import io

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from app.rag.chroma_store import ChromaStore
from app.rag.collection_selector import COLLECTION_META, INTENT_COLLECTION_MAP

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Chroma collections.")
    parser.add_argument(
        "--query", type=str, default=None,
        help="Search across collections with this query text.",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        help="Limit search to this collection (otherwise searches all).",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to return per collection (default: 5).",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="Show full metadata for each collection.",
    )
    args = parser.parse_args()

    # ── Check Chroma directory ───────────────────────────────────
    if not os.path.isdir(CHROMA_DIR):
        print(f"[ERROR] Chroma directory not found: {CHROMA_DIR}")
        print("Run ingest_knowledge.py first to populate collections.")
        return 1

    store = ChromaStore(persist_dir=CHROMA_DIR)

    # ── Search mode ──────────────────────────────────────────────
    if args.query:
        return _do_search(store, args)

    # ── List mode ───────────────────────────────────────────────
    return _do_list(store, args)


def _do_list(store: ChromaStore, args: argparse.Namespace) -> int:
    """List collections with counts."""
    print("=" * 60)
    print("  Chroma Collections")
    print(f"  Path: {CHROMA_DIR}")
    print("=" * 60)

    collections = store.list_collections()

    if not collections:
        print("\n  [EMPTY] No collections found. Run ingest_knowledge.py first.")
        return 0

    print(f"\n  Total collections: {len(collections)}")
    total_vectors = 0
    for col in collections:
        total_vectors += col.get("count", 0)
        meta = COLLECTION_META.get(col["name"], {})
        display = meta.get("display_name", "—")
        desc = meta.get("description", "—")
        print(f"\n  [{col['name']}]")
        print(f"    Display name: {display}")
        print(f"    Vectors:      {col['count']}")
        print(f"    Domain:       {meta.get('domain', '—')}")
        print(f"    Description:  {desc}")

        if args.details:
            col_info = store.collection_info(col["name"])
            if col_info:
                print(f"    Metadata:     {col_info.get('metadata', {})}")

    print(f"\n  Total vectors across all collections: {total_vectors}")

    # Also show intent → collection mapping
    print("\n" + "-" * 60)
    print("  Intent → Collection Mapping")
    print("-" * 60)
    for intent, colls in INTENT_COLLECTION_MAP.items():
        existing = [c for c in colls if store.collection_exists(c)]
        print(f"  {intent:25s} → {', '.join(colls)}")
        if existing != colls:
            print(f"  {'':25s}   (available: {', '.join(existing) if existing else 'none yet'})")

    return 0


def _do_search(store: ChromaStore, args: argparse.Namespace) -> int:
    """Search Chroma and display results."""
    if args.collection:
        collections = [args.collection]
    else:
        # Search all existing collections
        collections = [c["name"] for c in store.list_collections()]

    existing = [c for c in collections if store.collection_exists(c) or not store.list_collections()]
    if not existing:
        print("[INFO] No populated collections to search. Run ingest_knowledge.py first.")
        return 0

    print("=" * 60)
    print(f"  Search: \"{args.query}\"")
    print(f"  Collections: {', '.join(existing)}")
    print(f"  Top-K: {args.top_k}")
    print("=" * 60)

    results = store.search(args.query, existing, top_k=args.top_k)

    if not results:
        print("\n  [NO RESULTS] No matching chunks found.")
        return 0

    print(f"\n  Found {len(results)} result(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  ── Result #{i} ──")
        print(f"  Title:          {r['title']}")
        print(f"  Collection:     {r['collection_name']}")
        print(f"  Source type:    {r['source_type']}")
        print(f"  Score:          {r['score']:.4f}")
        print(f"  Chunk index:    {r['chunk_index']}")

        # Preview: first 300 chars of content
        preview = r["content"][:300]
        if len(r["content"]) > 300:
            preview += "…"
        print(f"  Preview:        {preview}")

        # Metadata
        meta = r.get("metadata", {})
        if meta:
            # Show a few key fields
            key_fields = ["file_name", "format", "authority", "tags"]
            for kf in key_fields:
                if kf in meta:
                    print(f"  {kf}:      {meta[kf]}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
