"""
Tests for ChromaStore — using FakeEmbeddingProvider (no real API calls).

Covers:
  - Collection create/get
  - add_chunks
  - search
  - collection_exists
  - list_collections
  - sanitize_metadata
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from app.rag.embedding_provider import FakeEmbeddingProvider
from app.rag.chroma_store import ChromaStore, _sanitize_metadata
from app.rag.text_splitter import split_text, clean_text


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for Chroma persistence."""
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def fake_provider():
    return FakeEmbeddingProvider(dimension=128)


@pytest.fixture
def store(temp_chroma_dir, fake_provider):
    return ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=fake_provider)


# ── Sample data ──────────────────────────────────────────────────────

def _sample_chunks(n: int = 3) -> list[dict]:
    """Create n sample chunks for testing."""
    texts = [
        "资产配置是投资组合管理的核心环节。通过合理分配不同资产类别的比例，"
        "投资者可以在风险可控的前提下追求长期稳健的收益。",
        "保守型投资者通常将大部分资金配置在固收类资产中。这类投资者无法接受"
        "本金的显著波动，投资期限通常较短。",
        "定期再平衡是维持目标配置比例的重要手段。建议每半年或每年对投资组合"
        "进行一次检视和调整。",
    ]
    chunks = []
    for i in range(min(n, len(texts))):
        chunks.append({
            "content": texts[i],
            "metadata": {
                "title": f"Test Document {i + 1}",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "chunk_index": i,
                "content_hash": f"hash_{i:04d}",
                "token_count": len(texts[i]),
                "authority": "medium",
            },
        })
    return chunks


# ── Collection management ────────────────────────────────────────────

def test_get_or_create_collection(store):
    col = store.get_or_create_collection("test_collection")
    assert col is not None
    assert col.name == "test_collection"
    assert col.count() == 0


def test_collection_exists_empty(store):
    """A collection that exists but has no data should return False."""
    store.get_or_create_collection("empty_col")
    assert store.collection_exists("empty_col") is False


def test_collection_exists_with_data(store):
    store.add_chunks(_sample_chunks(2), "advisory_knowledge")
    assert store.collection_exists("advisory_knowledge") is True


def test_collection_exists_nonexistent(store):
    assert store.collection_exists("nonexistent") is False


# ── add_chunks ───────────────────────────────────────────────────────

def test_add_chunks_basic(store):
    chunks = _sample_chunks(2)
    count = store.add_chunks(chunks, "advisory_knowledge")
    assert count == 2


def test_add_chunks_empty(store):
    assert store.add_chunks([], "advisory_knowledge") == 0


def test_add_chunks_sets_metadata(store):
    chunks = _sample_chunks(1)
    store.add_chunks(chunks, "advisory_knowledge")
    col = store.get_or_create_collection("advisory_knowledge")
    assert col.count() == 1


def test_add_chunks_replaces_same_file_chunks(store):
    """Re-ingesting an edited file should remove stale content-hash chunks."""
    first = [{
        "content": "旧版本内容：包含具体收益率示例。",
        "metadata": {
            "title": "Same File",
            "source_type": "investment_knowledge",
            "collection_name": "advisory_knowledge",
            "chunk_index": 0,
            "content_hash": "old_hash",
            "file_name": "same_file.md",
        },
    }]
    second = [{
        "content": "新版本内容：只保留中性原则描述。",
        "metadata": {
            "title": "Same File",
            "source_type": "investment_knowledge",
            "collection_name": "advisory_knowledge",
            "chunk_index": 0,
            "content_hash": "new_hash",
            "file_name": "same_file.md",
        },
    }]

    assert store.add_chunks(first, "advisory_knowledge") == 1
    assert store.add_chunks(second, "advisory_knowledge") == 1

    col = store.get_or_create_collection("advisory_knowledge")
    assert col.count() == 1
    results = store.search("中性原则", ["advisory_knowledge"], top_k=3)
    assert len(results) == 1
    assert "新版本内容" in results[0]["content"]


# ── search ───────────────────────────────────────────────────────────

def test_search_returns_results(store):
    store.add_chunks(_sample_chunks(3), "advisory_knowledge")
    results = store.search("资产配置", ["advisory_knowledge"], top_k=2)
    assert len(results) > 0
    assert len(results) <= 2


def test_search_result_structure(store):
    store.add_chunks(_sample_chunks(1), "advisory_knowledge")
    results = store.search("投资", ["advisory_knowledge"], top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "content" in r
    assert "score" in r
    assert "metadata" in r
    assert "collection_name" in r
    assert "title" in r
    assert "source_type" in r
    # score should be between 0 and 1
    assert 0.0 <= r["score"] <= 1.0


def test_search_scores_sorted(store):
    store.add_chunks(_sample_chunks(3), "advisory_knowledge")
    results = store.search("保守型投资者的资产配置", ["advisory_knowledge"], top_k=3)
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


def test_search_missing_collection_graceful(store):
    """Searching a collection that doesn't exist should not error."""
    results = store.search("test query", ["nonexistent_collection"], top_k=3)
    assert results == []


def test_search_multiple_collections(store):
    store.add_chunks(_sample_chunks(1), "advisory_knowledge")
    store.add_chunks(_sample_chunks(1), "education_knowledge")
    results = store.search("投资", ["advisory_knowledge", "education_knowledge"], top_k=5)
    assert len(results) >= 1


# ── list_collections ─────────────────────────────────────────────────

def test_list_collections(store):
    store.add_chunks(_sample_chunks(1), "test_coll_a")
    store.add_chunks(_sample_chunks(1), "test_coll_b")
    colls = store.list_collections()
    names = [c["name"] for c in colls]
    assert "test_coll_a" in names
    assert "test_coll_b" in names


# ── collection_info ──────────────────────────────────────────────────

def test_collection_info(store):
    store.add_chunks(_sample_chunks(2), "test_info")
    info = store.collection_info("test_info")
    assert info is not None
    assert info["name"] == "test_info"
    assert info["count"] == 2


def test_collection_info_missing(store):
    assert store.collection_info("nonexistent") is None


# ── delete_collection ────────────────────────────────────────────────

def test_delete_collection(store):
    store.add_chunks(_sample_chunks(1), "to_delete")
    assert store.collection_exists("to_delete")
    assert store.delete_collection("to_delete") is True


def test_delete_nonexistent_collection(store):
    assert store.delete_collection("never_existed") is False


# ── metadata sanitization ────────────────────────────────────────────

def test_sanitize_metadata_passes_allowed_types():
    meta = {"title": "test", "count": 5, "score": 0.85, "active": True}
    result = _sanitize_metadata(meta)
    assert result == meta


def test_sanitize_metadata_converts_complex_types():
    meta = {"title": "test", "nested": {"a": 1}, "arr": [1, 2, 3]}
    result = _sanitize_metadata(meta)
    assert "title" in result
    assert isinstance(result.get("nested"), str)
    assert isinstance(result.get("arr"), str)


def test_sanitize_metadata_drops_none():
    meta = {"title": "test", "null_val": None, "count": 5}
    result = _sanitize_metadata(meta)
    assert "null_val" not in result
    assert result["title"] == "test"
    assert result["count"] == 5
