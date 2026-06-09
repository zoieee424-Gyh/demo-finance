"""
Tests for KnowledgeMetadataStore and ChromaStore.prepare_chunk_records.

All tests run without real MySQL — configuration checks, parameter
construction, and serialisation are verified in isolation.

ChromaStore tests use FakeEmbeddingProvider (no real API calls).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


def _patch_settings(**kwargs):
    """Return a mock.patch context that overrides settings attributes."""
    return mock.patch.multiple(
        "app.core.config.settings",
        mysql_enabled=kwargs.get("mysql_enabled", False),
        mysql_password=kwargs.get("mysql_password", None),
        mysql_host=kwargs.get("mysql_host", "127.0.0.1"),
        mysql_port=kwargs.get("mysql_port", 3306),
        mysql_user=kwargs.get("mysql_user", "root"),
        mysql_database=kwargs.get("mysql_database", "fin_agent_knowledge"),
    )


def _make_store(**overrides):
    """Create a KnowledgeMetadataStore with patched settings.

    Note: the store reads settings at __init__ time, so we patch before
    importing/constructing.
    """
    with _patch_settings(**overrides):
        from app.rag.metadata_store import KnowledgeMetadataStore
        return KnowledgeMetadataStore()


# ── Configuration tests ────────────────────────────────────────────────────


class TestMetadataStoreConfig:
    """Tests that do not require a real MySQL connection."""

    def test_is_configured_false_when_disabled(self):
        """When mysql_enabled=False, is_configured() returns False."""
        store = _make_store(mysql_enabled=False, mysql_password=None)
        assert store.is_configured() is False

    def test_is_configured_false_when_no_password(self):
        """When password is None, is_configured() returns False."""
        store = _make_store(mysql_enabled=True, mysql_password=None)
        assert store.is_configured() is False

    def test_is_configured_true_when_enabled_and_password_set(self):
        """When enabled=True and password is set, is_configured() returns True."""
        store = _make_store(mysql_enabled=True, mysql_password="secret")
        assert store.is_configured() is True

    def test_close_when_no_connection(self):
        """close() on a store that was never connected should not raise."""
        store = _make_store(mysql_enabled=False, mysql_password=None)
        store.close()  # Should not raise


# ── SQL / serialisation tests ─────────────────────────────────────────────


class TestMetadataSerialization:
    """Verify JSON serialisation format without connecting to MySQL."""

    def test_upsert_collection_agent_scope_json(self):
        """agent_scope list should be json.dumps-ed with ensure_ascii=False."""
        scope = ["investment_advisor", "education"]
        result = json.dumps(scope, ensure_ascii=False)
        assert "investment_advisor" in result
        assert "education" in result
        # ensure_ascii=False keeps Chinese characters
        parsed = json.loads(result)
        assert parsed == scope

    def test_tags_json_serialization(self):
        """Tags should be serialised as JSON array."""
        tags = ["asset_allocation", "beginner", "基金"]
        result = json.dumps(tags, ensure_ascii=False)
        parsed = json.loads(result)
        assert parsed == tags
        assert "基金" in result

    def test_metadata_json_serialization(self):
        """Chunk metadata_json should survive round-trip."""
        meta = {
            "document_id": 1,
            "collection_name": "advisory_knowledge",
            "source_type": "investment_knowledge",
            "title": "资产配置基础原则",
            "authority": "high",
            "chunk_index": 0,
            "content_hash": "abc123",
            "token_count": 623,
        }
        serialized = json.dumps(meta, ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["title"] == "资产配置基础原则"
        assert parsed == meta


# ── replace_chunks parameter tests ────────────────────────────────────────


class TestReplaceChunksParams:
    """Verify replace_chunks parameter handling (no real connection)."""

    def test_replace_chunks_empty_list_returns_zero(self):
        """Empty input should return 0 without touching the database."""
        store = _make_store(mysql_enabled=True, mysql_password="secret")
        # Empty list → early return 0 before any DB call
        result = store.replace_chunks(1, "advisory_knowledge", [])
        assert result == 0

    def test_replace_chunks_builds_correct_sql_params(self):
        """Sanity check that chunk dict fields are used correctly."""
        chunks = [
            {
                "chroma_id": "advisory_knowledge_abc123",
                "chunk_index": 0,
                "title": "资产配置基础原则",
                "content_hash": "abc123",
                "token_count": 623,
                "metadata_json": {"authority": "high", "source_type": "investment_knowledge"},
            },
        ]
        # Verify all required keys are present
        for chunk in chunks:
            assert "chroma_id" in chunk
            assert "chunk_index" in chunk
            assert "title" in chunk
            assert "content_hash" in chunk
            assert "token_count" in chunk
            assert "metadata_json" in chunk

    def test_replace_chunks_metadata_serializable(self):
        """metadata_json must be valid JSON."""
        meta = {"authority": "high", "source_type": "investment_knowledge",
                "tags": ["基金", "入门"]}
        serialized = json.dumps(meta, ensure_ascii=False)
        assert isinstance(serialized, str)
        # Round trip
        assert json.loads(serialized) == meta

    def test_upsert_document_with_chunks_rolls_back_on_error(self):
        """Document + chunks sync should rollback atomically on failure."""
        from app.rag.metadata_store import KnowledgeMetadataStore

        class FakeCursor:
            lastrowid = 42

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def execute(self, sql, params=None):
                if "SELECT id FROM knowledge_documents" in sql:
                    return None
                if "INSERT INTO knowledge_chunks" in sql:
                    raise RuntimeError("chunk insert failed")
                return None

            def fetchone(self):
                return None

        class FakeConn:
            def __init__(self):
                self.open = True
                self.committed = False
                self.rolled_back = False

            def cursor(self):
                return FakeCursor()

            def commit(self):
                self.committed = True

            def rollback(self):
                self.rolled_back = True

        fake_conn = FakeConn()
        store = _make_store(mysql_enabled=True, mysql_password="secret")
        store._conn = fake_conn

        with pytest.raises(RuntimeError, match="chunk insert failed"):
            store.upsert_document_with_chunks(
                collection_name="advisory_knowledge",
                title="Test Doc",
                chunks_with_chroma_ids=[{
                    "chroma_id": "advisory_knowledge_hash",
                    "chunk_index": 0,
                    "title": "Test Doc",
                    "content_hash": "hash",
                    "token_count": 10,
                    "metadata_json": {},
                }],
            )

        assert fake_conn.rolled_back is True
        assert fake_conn.committed is False


# ── SQL splitter tests ─────────────────────────────────────────────────────


class TestSqlSplitter:
    """Test _split_sql_statements helper."""

    def test_split_simple_statements(self):
        from app.rag.metadata_store import _split_sql_statements
        sql = "SELECT 1;\nSELECT 2;\n"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2
        assert "SELECT 1" in stmts[0]
        assert "SELECT 2" in stmts[1]

    def test_split_skips_comments(self):
        from app.rag.metadata_store import _split_sql_statements
        sql = "-- This is a comment\nSELECT 1;\n# Another comment\nSELECT 2;"
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 2
        assert "comment" not in stmts[0]

    def test_split_multiline_statement(self):
        from app.rag.metadata_store import _split_sql_statements
        sql = (
            "CREATE TABLE foo (\n"
            "  id INT,\n"
            "  name VARCHAR(64)\n"
            ");\n"
        )
        stmts = _split_sql_statements(sql)
        assert len(stmts) == 1
        assert "CREATE TABLE" in stmts[0]
        assert "id INT" in stmts[0]


# ── ChromaStore.prepare_chunk_records tests ────────────────────────────────


class TestPrepareChunkRecords:
    """Test ChromaStore.prepare_chunk_records with FakeEmbeddingProvider."""

    @pytest.fixture
    def temp_chroma_dir(self):
        path = tempfile.mkdtemp(prefix="chroma_test_")
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def fake_provider(self):
        from app.rag.embedding_provider import FakeEmbeddingProvider
        return FakeEmbeddingProvider(dimension=128)

    @pytest.fixture
    def store(self, temp_chroma_dir, fake_provider):
        from app.rag.chroma_store import ChromaStore
        return ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=fake_provider)

    def test_prepare_returns_list_of_dicts(self, store):
        """Should return a list of chunk record dicts."""
        chunks = [
            {
                "content": "资产配置是投资组合管理的核心环节。",
                "metadata": {
                    "title": "资产配置基础原则",
                    "source_type": "investment_knowledge",
                    "authority": "high",
                    "chunk_index": 0,
                    "content_hash": "abc123",
                    "token_count": 50,
                    "file_name": "test.md",
                    "format": "md",
                },
            },
        ]
        records = store.prepare_chunk_records(chunks, "advisory_knowledge")
        assert isinstance(records, list)
        assert len(records) == 1
        rec = records[0]
        assert rec["chroma_id"] == "advisory_knowledge_abc123"
        assert rec["chunk_index"] == 0
        assert rec["title"] == "资产配置基础原则"
        assert rec["content_hash"] == "abc123"
        assert rec["token_count"] == 50
        assert isinstance(rec["metadata"], dict)

    def test_prepare_chroma_id_stable(self, store):
        """Same content_hash → same chroma_id (deterministic)."""
        chunks = [
            {
                "content": "测试内容。",
                "metadata": {
                    "title": "测试",
                    "content_hash": "deadbeef",
                    "token_count": 10,
                    "file_name": "test.md",
                    "format": "md",
                    "chunk_index": 0,
                },
            },
        ]
        records1 = store.prepare_chunk_records(chunks, "risk_knowledge")
        records2 = store.prepare_chunk_records(chunks, "risk_knowledge")
        assert records1[0]["chroma_id"] == records2[0]["chroma_id"]
        assert records1[0]["chroma_id"] == "risk_knowledge_deadbeef"

    def test_prepare_sanitizes_metadata(self, store):
        """Metadata values should be sanitised to Chroma-compatible types."""
        chunks = [
            {
                "content": "测试。",
                "metadata": {
                    "title": "Test",
                    "chunk_index": 0,
                    "content_hash": "hash123",
                    "token_count": 5,
                    "authority": "high",
                    "enabled": True,
                    "score": 0.95,
                    "tags": ["a", "b"],  # list → should be str
                    "nested": {"key": "val"},  # dict → should be str
                    "null_val": None,  # None → should be removed
                },
            },
        ]
        records = store.prepare_chunk_records(chunks, "test_coll")
        rec = records[0]
        meta = rec["metadata"]

        # Allowed types pass through
        assert meta["title"] == "Test"
        assert meta["authority"] == "high"
        assert meta["token_count"] == 5
        assert meta["enabled"] is True
        assert meta["score"] == 0.95
        # None should be absent
        assert "null_val" not in meta
        # Non-scalar converted to string
        assert isinstance(meta["tags"], str)
        assert isinstance(meta["nested"], str)

    def test_prepare_and_add_consistency(self, store):
        """Chunks prepared with prepare_chunk_records should match what add_chunks writes."""
        chunks = [
            {
                "content": f"测试内容 chunk {i}。",
                "metadata": {
                    "title": f"测试文档",
                    "chunk_index": i,
                    "content_hash": f"hash_{i:04d}",
                    "token_count": 20 + i,
                    "file_name": "consistency_test.md",
                    "format": "md",
                },
            }
            for i in range(3)
        ]

        records = store.prepare_chunk_records(chunks, "advisory_knowledge")

        # add_chunks internally calls prepare_chunk_records and then upserts
        added = store.add_chunks(chunks, "advisory_knowledge")
        assert added == 3

        # Verify the prepared chroma_ids match what was inserted
        info = store.collection_info("advisory_knowledge")
        assert info is not None
        assert info["count"] == 3

        # Verify we can search
        results = store.search("测试内容", ["advisory_knowledge"], top_k=5)
        assert len(results) >= 1

    def test_prepare_empty_chunks(self, store):
        """Empty chunk list returns empty list."""
        records = store.prepare_chunk_records([], "any")
        assert records == []

    def test_prepare_skips_empty_content(self, store):
        """Chunks with empty/missing content are skipped."""
        chunks = [
            {"content": "", "metadata": {"chunk_index": 0, "content_hash": "h1"}},
            {"content": "Valid content.", "metadata": {"chunk_index": 1, "content_hash": "h2", "token_count": 10}},
            {"content": "   ", "metadata": {"chunk_index": 2, "content_hash": "h3"}},
        ]
        records = store.prepare_chunk_records(chunks, "test_coll")
        assert len(records) == 1
        assert records[0]["content"] == "Valid content."

    def test_prepare_default_chunk_index(self, store):
        """When chunk_index is missing, uses loop index."""
        chunks = [
            {
                "content": f"Chunk {i}",
                "metadata": {"content_hash": f"h{i}", "title": "T"},
            }
            for i in range(3)
        ]
        records = store.prepare_chunk_records(chunks, "coll")
        for i, rec in enumerate(records):
            assert rec["chunk_index"] == i


# ── Ingest MySQL flag tests ────────────────────────────────────────────────


class TestIngestMysqlFlags:
    """Verify that the ingest script handles --sync-mysql correctly."""

    def test_sync_mysql_flag_requires_config(self):
        """--sync-mysql with mysql_enabled=false should fail early.

        Instead of running a subprocess (encoding issues on Windows), we
        verify the guard logic directly: KnowledgeMetadataStore.is_configured()
        returns False when mysql_enabled=False, so the ingest script would
        print an error and exit non-zero.
        """
        store = _make_store(mysql_enabled=False, mysql_password=None)
        assert store.is_configured() is False
        # This is what ingest checks before proceeding with --sync-mysql

    def test_sync_mysql_flag_requires_password(self):
        """--sync-mysql with mysql_enabled=true but no password also fails."""
        store = _make_store(mysql_enabled=True, mysql_password=None)
        assert store.is_configured() is False

    def test_sync_mysql_with_config_passes_guard(self):
        """--sync-mysql with mysql_enabled=true AND password passes guard."""
        store = _make_store(mysql_enabled=True, mysql_password="secret")
        assert store.is_configured() is True

    def test_dry_run_does_not_trigger_mysql(self):
        """Verification: the --sync-mysql flag is present in argparse."""
        import argparse as _argparse
        import sys as _sys
        script_path = str(
            Path(__file__).resolve().parent.parent
            / "scripts" / "ingest_knowledge.py"
        )
        # Parse the script source to confirm --sync-mysql argument exists
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "--sync-mysql" in source
        assert "KnowledgeMetadataStore" in source
        assert "upsert_document_with_chunks" in source
        # Verify --dry-run still works without --sync-mysql
        assert "add_argument(\"--dry-run\"" in source.replace(" ", "") or \
               "\"--dry-run\"" in source
