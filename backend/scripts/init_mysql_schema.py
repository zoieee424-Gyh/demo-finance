#!/usr/bin/env python
"""
Initialise the MySQL knowledge metadata schema.

Executes backend/sql/knowledge_schema.sql against the configured MySQL
database. Safe to run multiple times — CREATE IF NOT EXISTS and
ON DUPLICATE KEY UPDATE are used throughout.

Usage:
    $env:FIN_AGENT_MYSQL_ENABLED = "true"
    $env:MYSQL_PASSWORD = "..."
    python backend/scripts/init_mysql_schema.py

    # Custom SQL path
    python backend/scripts/init_mysql_schema.py --sql backend/sql/knowledge_schema.sql
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialise MySQL knowledge metadata schema."
    )
    parser.add_argument(
        "--sql", type=str, default=None,
        help="Path to knowledge_schema.sql (default: backend/sql/knowledge_schema.sql).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MySQL Schema Initialisation")
    print("=" * 60)

    # ── Check configuration ─────────────────────────────────────────
    print("\n[1] Configuration check:")
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
        print("  Set the following environment variables:")
        print("    $env:FIN_AGENT_MYSQL_ENABLED = 'true'")
        print("    $env:MYSQL_PASSWORD = '...'")
        print("  Also ensure MYSQL_HOST/PORT/USER/DATABASE if non-default.")
        return 1

    # ── Test connection (without database, CREATE DATABASE handled by schema) ─
    print("\n[2] Testing server connection...")
    try:
        import pymysql
        conn = pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            charset="utf8mb4",
        )
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION()")
            ver = cur.fetchone()
        conn.close()
        ver_str = ver[0] if ver else "unknown"
        print(f"    [OK] MySQL version: {ver_str}")
    except Exception as e:
        print(f"    [FAIL] Cannot connect to MySQL server: {e}")
        return 1

    # ── Execute schema ──────────────────────────────────────────────
    print("\n[3] Executing schema (creates database + tables + seed data)...")
    try:
        store.initialize_schema(sql_path=args.sql)
        print("    [OK] Schema executed successfully.")
    except Exception as e:
        print(f"    [FAIL] Schema execution error: {e}")
        return 1

    # ── Verify ──────────────────────────────────────────────────────
    print("\n[4] Verifying...")
    try:
        counts = store.get_collection_counts()
        print(f"    Collections: {len(counts)}")
        for c in counts:
            print(f"      - {c['name']} ({c['display_name']}) domain={c['domain']}")
        doc_counts = store.get_document_counts()
        print(f"    Documents:    {sum(d['doc_count'] for d in doc_counts)}")
        totals = store.get_total_counts()
        print(f"    Chunks:       {totals['chunks']}")
    except Exception as e:
        print(f"    [WARN] Verification query failed: {e}")

    store.close()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
