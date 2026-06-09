#!/usr/bin/env python
"""
Ingest knowledge documents into Chroma.

Scans data/knowledge_base/{domain}/ for .md / .txt / .json files,
loads, cleans, splits, embeds, and stores them in Chroma.

Usage:
    python backend/scripts/ingest_knowledge.py              # Full ingest
    python backend/scripts/ingest_knowledge.py --dry-run    # Scan only, no API calls
    python backend/scripts/ingest_knowledge.py --collection advisory_knowledge  # Single collection
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

# Fix Unicode output on Windows GBK terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Ensure the backend package is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.document_loader import load_documents_from_dir
from app.rag.text_splitter import split_document
from app.rag.embedding_provider import EmbeddingProvider
from app.rag.chroma_store import ChromaStore
from app.rag.collection_selector import COLLECTION_META

# ── Paths ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"
CHROMA_DIR = str(PROJECT_ROOT / "data" / "chroma")

# ── Directory → collection mapping ──────────────────────────────────

DIR_TO_COLLECTION: dict[str, str] = {
    "advisory":   "advisory_knowledge",
    "compliance": "compliance_knowledge",
    "education":  "education_knowledge",
    "risk":       "risk_knowledge",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest knowledge documents into Chroma.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report, but do not call the embedding API or write to Chroma.",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        help="Only process the specified collection (e.g. advisory_knowledge).",
    )
    parser.add_argument(
        "--sync-mysql", action="store_true",
        help="After writing to Chroma, also write document/chunk metadata to MySQL.",
    )
    args = parser.parse_args()

    if not KB_DIR.exists():
        print(f"[ERROR] Knowledge base directory not found: {KB_DIR}")
        return 1

    # ── Discovery ────────────────────────────────────────────────
    print("=" * 60)
    print("  Knowledge Ingest — discovery phase")
    print("=" * 60)

    # Discover documents per collection
    collection_docs: dict[str, list[dict]] = {}
    skipped: dict[str, list[str]] = {}
    total_docs = 0

    for dir_name in sorted(os.listdir(KB_DIR)):
        dir_path = KB_DIR / dir_name
        if not dir_path.is_dir():
            continue

        collection = DIR_TO_COLLECTION.get(dir_name)
        if collection is None:
            print(f"  [SKIP] Unknown directory: {dir_name} (no collection mapping)")
            continue

        if args.collection and collection != args.collection:
            continue

        print(f"\n  Scanning: {dir_path} → {collection}")
        docs = load_documents_from_dir(str(dir_path))
        collection_docs[collection] = docs
        skipped[collection] = []
        total_docs += len(docs)

        for doc in docs:
            print(f"    ✓ {doc['title']} ({len(doc['content'])} chars)")

    if total_docs == 0:
        print("\n[INFO] No documents found. Nothing to ingest.")
        return 0

    # ── Dry-run report ───────────────────────────────────────────
    if args.dry_run:
        print("\n" + "=" * 60)
        print("  DRY-RUN Summary (no API calls made)")
        print("=" * 60)
        for coll, docs in collection_docs.items():
            total_chunks = 0
            for doc in docs:
                chunks = split_document(doc)
                total_chunks += len(chunks)
            print(f"  {coll}: {len(docs)} docs → ~{total_chunks} chunks")
        print(f"\n  Total: {total_docs} documents across {len(collection_docs)} collections")
        return 0

    # ── Embedding provider check ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  Checking embedding provider")
    print("=" * 60)

    provider = EmbeddingProvider()
    if not provider.is_configured:
        print("  [ERROR] DASHSCOPE_API_KEY is not set.")
        print("  Set it via: export DASHSCOPE_API_KEY=sk-...")
        print("  Or use --dry-run to scan without API calls.")
        return 1
    print(f"  [OK] Embedding provider configured (model={provider.model})")

    # ── Ingest loop ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ingesting into Chroma")
    print("=" * 60)

    store = ChromaStore(persist_dir=CHROMA_DIR, embedding_provider=provider)

    # ── MySQL metadata store (only when --sync-mysql) ──────────────
    mysql_store = None
    if args.sync_mysql:
        from app.rag.metadata_store import KnowledgeMetadataStore
        mysql_store = KnowledgeMetadataStore()
        if not mysql_store.is_configured():
            print("  [ERROR] --sync-mysql requires FIN_AGENT_MYSQL_ENABLED=true "
                  "and MYSQL_PASSWORD set.")
            return 1
        print("  [OK] MySQL metadata sync enabled.")
        # Upsert collections from COLLECTION_META
        for coll_name in collection_docs:
            meta = COLLECTION_META.get(coll_name, {})
            try:
                mysql_store.upsert_collection(
                    name=coll_name,
                    display_name=meta.get("display_name", coll_name),
                    domain=meta.get("domain", ""),
                    description=meta.get("description", ""),
                    agent_scope=meta.get("agent_scope", []),
                )
                print(f"    ✓ MySQL collection upserted: {coll_name}")
            except Exception as e:
                print(f"    [FAIL] MySQL collection upsert failed for {coll_name}: {e}")
                return 1

    summary: dict[str, dict] = {}

    for collection, docs in collection_docs.items():
        print(f"\n  Collection: {collection}")
        coll_doc_count = 0
        coll_chunk_count = 0

        for doc in docs:
            chunks = split_document(doc)
            if not chunks:
                print(f"    [SKIP] {doc['title']}: no chunks generated")
                skipped[collection].append(f"{doc['title']} (no chunks)")
                continue

            # Inject collection metadata into each chunk
            for chunk in chunks:
                chunk["metadata"]["collection_name"] = collection
                chunk["metadata"]["source_type"] = _infer_source_type(collection)

            # Prepare chunk records BEFORE upsert (stable chroma_id)
            chunk_records = store.prepare_chunk_records(chunks, collection)

            added = store.add_chunks(chunks, collection)
            if added > 0:
                coll_chunk_count += added
                coll_doc_count += 1
                print(f"    ✓ {doc['title']}: {added} chunks")
            else:
                skipped[collection].append(f"{doc['title']} (add returned 0)")

            # ── Sync to MySQL ────────────────────────────────────
            if mysql_store and added > 0:
                try:
                    metadata = doc.get("metadata", {})
                    mysql_store.upsert_document_with_chunks(
                        collection_name=collection,
                        title=doc.get("title", ""),
                        chunks_with_chroma_ids=[
                            {
                                "chroma_id": cr["chroma_id"],
                                "chunk_index": cr["chunk_index"],
                                "title": cr["title"],
                                "content_hash": cr["content_hash"],
                                "token_count": cr["token_count"],
                                "metadata_json": cr["metadata"],
                            }
                            for cr in chunk_records
                        ],
                        source_type=_infer_source_type(collection),
                        authority=metadata.get("authority", "medium"),
                        url=metadata.get("url"),
                        file_path=metadata.get("file_path"),
                        version="1.0",
                        status="active",
                        tags=metadata.get("tags", []),
                        summary=None,
                        chunk_count=added,
                    )
                except Exception as e:
                    print(f"    [FAIL] MySQL sync error for {doc['title']}: {e}")
                    mysql_store.close()
                    return 1

        summary[collection] = {
            "documents": coll_doc_count,
            "chunks": coll_chunk_count,
            "skipped": skipped.get(collection, []),
        }

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Ingest Summary")
    print("=" * 60)
    total_chunks = 0
    for coll, info in summary.items():
        print(f"  {coll}: {info['documents']} docs, {info['chunks']} chunks")
        total_chunks += info["chunks"]
        for s in info["skipped"]:
            print(f"    ⚠ Skipped: {s}")

    print(f"\n  Total: {total_docs} documents → {total_chunks} chunks")
    print(f"  Chroma data: {CHROMA_DIR}")
    if mysql_store:
        try:
            totals = mysql_store.get_total_counts()
            print(f"  MySQL metadata: {totals['documents']} docs, {totals['chunks']} chunks")
        except Exception:
            pass
        mysql_store.close()
    return 0


def _infer_source_type(collection: str) -> str:
    """Map collection name to standard source_type."""
    mapping = {
        "advisory_knowledge":   "investment_knowledge",
        "compliance_knowledge": "regulations",
        "education_knowledge":  "financial_education",
        "risk_knowledge":       "risk_models",
    }
    return mapping.get(collection, "knowledge_base")


if __name__ == "__main__":
    sys.exit(main())
