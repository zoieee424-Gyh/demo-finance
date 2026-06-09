#!/usr/bin/env python
"""
Inspect MySQL knowledge metadata and optionally compare with Chroma.

Usage:
    # Basic check
    python backend/scripts/inspect_knowledge_metadata.py

    # Compare MySQL chunk counts with Chroma vector counts
    python backend/scripts/inspect_knowledge_metadata.py --compare-chroma

Requirements:
    - FIN_AGENT_MYSQL_ENABLED=true
    - MYSQL_PASSWORD set
    - For --compare-chroma: Chroma data must exist at data/chroma
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

from app.core.config import settings
from app.rag.metadata_store import KnowledgeMetadataStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect MySQL knowledge metadata."
    )
    parser.add_argument(
        "--compare-chroma", action="store_true",
        help="Compare MySQL chunk counts with Chroma vector counts.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MySQL Knowledge Metadata Inspector")
    print("=" * 60)

    # ── 1. Configuration check ──────────────────────────────────────
    print("\n[1] Configuration:")
    print(f"    FIN_AGENT_MYSQL_ENABLED = {os.getenv('FIN_AGENT_MYSQL_ENABLED', 'NOT SET')}")
    print(f"    settings.mysql_enabled  = {settings.mysql_enabled}")
    print(f"    MYSQL_HOST              = {settings.mysql_host}")
    print(f"    MYSQL_PORT              = {settings.mysql_port}")
    print(f"    MYSQL_USER              = {settings.mysql_user}")
    print(f"    MYSQL_DATABASE          = {settings.mysql_database}")
    print(f"    MYSQL_PASSWORD          = {'SET' if settings.mysql_password else 'NOT SET'}")

    store = KnowledgeMetadataStore()

    if not store.is_configured():
        print("\n[ERROR] MySQL is not configured.")
        print("  Set: $env:FIN_AGENT_MYSQL_ENABLED='true'")
        print("       $env:MYSQL_PASSWORD='...'")
        return 1

    # ── 2. Test connection ──────────────────────────────────────────
    print("\n[2] Testing connection...")
    try:
        conn = store._get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            ver = cur.fetchone()
        print(f"    [OK] Connected. MySQL version: {ver.get('VERSION()', 'unknown')}")
    except Exception as e:
        print(f"    [FAIL] Connection error: {e}")
        return 1

    # ── 3. Collection overview ──────────────────────────────────────
    print("\n[3] Collections:")
    collections = store.get_collection_counts()
    if not collections:
        print("    [EMPTY] No collections found. Run init_mysql_schema.py first.")
    else:
        for c in collections:
            print(f"    - {c['name']}")
            print(f"      Display: {c.get('display_name', '—')}")
            print(f"      Domain:  {c.get('domain', '—')}")
            print(f"      Enabled: {bool(c.get('enabled', 0))}")

    # ── 4. Document counts ──────────────────────────────────────────
    print("\n[4] Document counts:")
    doc_counts = store.get_document_counts()
    total_docs = sum(d["doc_count"] for d in doc_counts) if doc_counts else 0
    if not doc_counts:
        print("    [EMPTY] No documents found.")
    else:
        for d in doc_counts:
            print(f"    - {d['collection_name']}: {d['doc_count']} docs")
        print(f"    Total: {total_docs}")

    # ── 5. Chunk counts ─────────────────────────────────────────────
    print("\n[5] Chunk counts:")
    chunk_counts = store.get_chunk_counts_by_collection()
    total_chunks = sum(chunk_counts.values())
    if not chunk_counts:
        print("    [EMPTY] No chunks found.")
    else:
        for name in sorted(chunk_counts):
            print(f"    - {name}: {chunk_counts[name]} chunks")
        print(f"    Total: {total_chunks}")

    # ── 6. Compare with Chroma (optional) ───────────────────────────
    if args.compare_chroma:
        print("\n[6] Chroma comparison:")
        if not os.path.isdir(CHROMA_DIR):
            print(f"    [SKIP] Chroma directory not found: {CHROMA_DIR}")
        else:
            try:
                from app.rag.chroma_store import ChromaStore
                chroma = ChromaStore(persist_dir=CHROMA_DIR)
                chroma_colls = chroma.list_collections()

                print(f"    {'Collection':<30s} {'MySQL':>6s} {'Chroma':>6s} {'Match?':>7s}")
                print(f"    {'-'*30} {'-'*6} {'-'*6} {'-'*7}")

                all_match = True
                for cc in chroma_colls:
                    name = cc["name"]
                    cv = cc["count"]
                    mv = chunk_counts.get(name, 0)
                    match = "✅" if mv == cv else "❌"
                    if mv != cv:
                        all_match = False
                    print(f"    {name:<30s} {mv:>6d} {cv:>6d} {match:>7s}")

                # Check for MySQL collections not in Chroma
                for name in chunk_counts:
                    if not any(cc["name"] == name for cc in chroma_colls):
                        print(f"    {name:<30s} {chunk_counts[name]:>6d} {'—':>6s} {'⚠️':>7s}")

                print(f"    Total MySQL:  {total_chunks}")
                print(f"    Total Chroma: {sum(c['count'] for c in chroma_colls)}")

                if all_match:
                    print("\n    ✅ All collections match!")
                else:
                    print("\n    ❌ Mismatch detected. Re-run ingest_knowledge.py --sync-mysql.")

            except Exception as e:
                print(f"    [FAIL] Chroma comparison error: {e}")

    store.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
