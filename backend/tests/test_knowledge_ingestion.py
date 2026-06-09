"""
Tests for knowledge ingestion pipeline — document loading, cleaning, splitting.

All tests use local data and FakeEmbeddingProvider — NO real API calls.

Covers:
  - .md document loading
  - .txt document loading
  - .json document loading
  - Text cleaning (control chars, newline normalization)
  - Chunk splitting with overlap
  - Chroma store with fake embedding (write + retrieve)
  - KnowledgeRetriever mock fallback
  - KnowledgeRetriever Chroma path (with fake embedding)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest

from app.rag.document_loader import load_document, load_documents_from_dir
from app.rag.text_splitter import (
    clean_text,
    split_text,
    split_document,
    compute_content_hash,
)
from app.rag.embedding_provider import FakeEmbeddingProvider
from app.rag.chroma_store import ChromaStore
from app.rag.collection_selector import CollectionSelector
from app.rag.retriever import KnowledgeRetriever
from app.schemas.consultation import Source


# ═══════════════════════════════════════════════════════════════════════
# Document Loading
# ═══════════════════════════════════════════════════════════════════════

# ── .md ──────────────────────────────────────────────────────────────

_MD_CONTENT = """# 资产配置基础原则

资产配置是指根据投资者的风险承受能力、投资目标和投资期限，
将资金分配在不同资产类别之间的过程。

## 核心原则

1. 风险与收益匹配
2. 分散投资
3. 长期持有与定期再平衡
"""


def test_load_md(tmp_path):
    path = tmp_path / "test.md"
    path.write_text(_MD_CONTENT, encoding="utf-8")
    doc = load_document(str(path))
    assert doc["title"] == "资产配置基础原则"
    assert "风险与收益匹配" in doc["content"]
    assert doc["metadata"]["format"] == "md"
    assert "file_name" in doc["metadata"]


def test_load_md_no_heading_uses_filename(tmp_path):
    path = tmp_path / "no_heading.md"
    path.write_text("This document has no markdown heading.\nJust paragraphs.", encoding="utf-8")
    doc = load_document(str(path))
    assert doc["title"] == "no_heading"


# ── .txt ─────────────────────────────────────────────────────────────

_TXT_CONTENT = """资产配置基础原则

这是一篇关于资产配置基础原则的说明文档。

资产配置是投资管理的核心环节。"""


def test_load_txt(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text(_TXT_CONTENT, encoding="utf-8")
    doc = load_document(str(path))
    assert "资产配置" in doc["content"]
    assert doc["metadata"]["format"] == "txt"


def test_load_txt_first_line_title(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text(_TXT_CONTENT, encoding="utf-8")
    doc = load_document(str(path))
    # First line < 80 chars, doesn't end with period → used as title
    assert "资产配置" in doc["title"]


# ── .json ────────────────────────────────────────────────────────────

_JSON_DICT = json.dumps({
    "title": "JSON Test Document",
    "content": "This is the body content from a JSON dict.",
    "source": "internal",
})

_JSON_ARRAY = json.dumps([
    "第一段内容：资产配置的核心在于分散风险。",
    "第二段内容：不同资产类别之间相关性越低，分散效果越好。",
])


def test_load_json_dict(tmp_path):
    path = tmp_path / "test.json"
    path.write_text(_JSON_DICT, encoding="utf-8")
    doc = load_document(str(path))
    assert doc["title"] == "JSON Test Document"
    assert "body content" in doc["content"]


def test_load_json_array(tmp_path):
    path = tmp_path / "test_array.json"
    path.write_text(_JSON_ARRAY, encoding="utf-8")
    doc = load_document(str(path))
    assert "第一段内容" in doc["content"]
    assert "第二段内容" in doc["content"]


# ── unsupported format ───────────────────────────────────────────────

def test_load_unsupported_format_raises(tmp_path):
    path = tmp_path / "test.xyz"
    path.write_text("some content", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_document(str(path))


def test_load_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        load_document("/nonexistent/path/file.md")


# ── directory loading ────────────────────────────────────────────────

def test_load_documents_from_dir(tmp_path):
    (tmp_path / "doc1.md").write_text("# Doc One\n\nContent one.", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("Doc Two\nContent two.", encoding="utf-8")
    (tmp_path / "ignore.log").write_text("should be ignored", encoding="utf-8")

    docs = load_documents_from_dir(str(tmp_path))
    assert len(docs) == 2
    titles = {d["title"] for d in docs}
    assert "Doc One" in titles


# ═══════════════════════════════════════════════════════════════════════
# Text Cleaning & Splitting
# ═══════════════════════════════════════════════════════════════════════

def test_clean_text_removes_control_chars():
    dirty = "正常文本\x00\x01\x1f夹杂控制字符�测试"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\x01" not in cleaned
    assert "测试" in cleaned


def test_clean_text_normalizes_newlines():
    text = "line1\r\nline2\rline3\n\nline4"
    cleaned = clean_text(text)
    assert "\r\n" not in cleaned
    assert "\r" not in cleaned
    assert cleaned.count("\n") >= 2  # at least some newlines preserved


def test_clean_text_collapses_blank_lines():
    text = "paragraph one\n\n\n\n\nparagraph two"
    cleaned = clean_text(text)
    assert "\n\n\n\n" not in cleaned


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""


def test_split_with_overlap():
    text = "这是第一段测试文本。" + "内容" * 300 + "这是结尾段落。" + "补充" * 200
    chunks = split_text(text, chunk_size=600, chunk_overlap=100)
    assert len(chunks) >= 1
    # Each chunk should have required fields
    for chunk in chunks:
        assert "content" in chunk
        assert "chunk_index" in chunk
        assert "content_hash" in chunk
        assert len(chunk["content_hash"]) == 12
        assert "token_count" in chunk


def test_split_short_text():
    short = "这是很短的文本。"
    chunks = split_text(short, chunk_size=600, chunk_overlap=100)
    assert len(chunks) >= 1
    assert chunks[0]["chunk_index"] == 0


def test_split_document_enriches_metadata():
    doc = {
        "title": "Test",
        "content": "这是测试内容。" + "词" * 500,
        "metadata": {"file_name": "test.md", "format": "md"},
    }
    chunks = split_document(doc)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk["metadata"]["title"] == "Test"
        assert "chunk_index" in chunk["metadata"]
        assert "content_hash" in chunk["metadata"]


def test_compute_content_hash():
    h1 = compute_content_hash("hello")
    h2 = compute_content_hash("hello")
    h3 = compute_content_hash("world")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 12


# ═══════════════════════════════════════════════════════════════════════
# Fake Embedding Provider
# ═══════════════════════════════════════════════════════════════════════

def test_fake_embedding_provider():
    provider = FakeEmbeddingProvider(dimension=64)
    assert provider.is_configured is True

    vecs = provider.embed(["text one", "text two"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 64
    assert len(vecs[1]) == 64

    # Same text → same vector
    v1 = provider.embed_single("hello")
    v2 = provider.embed_single("hello")
    assert v1 == v2

    # Different text → different vector
    v3 = provider.embed_single("world")
    assert v1 != v3


# ═══════════════════════════════════════════════════════════════════════
# Embedding Provider (real — error when unconfigured)
# ═══════════════════════════════════════════════════════════════════════

def test_real_provider_unconfigured_raises():
    """Real provider with no API key should raise clear error."""
    from app.rag.embedding_provider import EmbeddingProvider
    # Temporarily unset DASHSCOPE_API_KEY
    old_key = os.environ.pop("DASHSCOPE_API_KEY", None)
    try:
        provider = EmbeddingProvider(api_key="")
        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            provider.embed(["test"])
    finally:
        if old_key is not None:
            os.environ["DASHSCOPE_API_KEY"] = old_key


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeRetriever — mock fallback
# ═══════════════════════════════════════════════════════════════════════

def test_retriever_mock_fallback():
    """Without Chroma, retriever should return mock sources."""
    retriever = KnowledgeRetriever()
    queries = [{"query": "如何配置资产", "source_type": "investment_knowledge"}]
    sources = retriever.retrieve(queries)
    assert len(sources) >= 1
    assert all(isinstance(s, Source) for s in sources)


def test_retriever_mock_returns_different_intents():
    """Different intent queries should return different mock sources."""
    retriever = KnowledgeRetriever()
    advisory_sources = retriever.retrieve([
        {"query": "资产配置", "source_type": "investment_knowledge"}
    ])
    edu_sources = retriever.retrieve([
        {"query": "什么是基金", "source_type": "financial_education"}
    ])
    # The titles should differ
    a_titles = {s.title for s in advisory_sources}
    e_titles = {s.title for s in edu_sources}
    assert a_titles != e_titles


def test_retriever_mock_deduplicates():
    """Duplicate titles should be removed."""
    retriever = KnowledgeRetriever()
    sources = retriever.retrieve([
        {"query": "test", "source_type": "investment_knowledge"},
    ])
    titles = [s.title for s in sources]
    assert len(titles) == len(set(titles))


# ═══════════════════════════════════════════════════════════════════════
# KnowledgeRetriever — Chroma path (with fake embedding)
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_chroma_dir():
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def chroma_store_with_data(temp_chroma_dir):
    provider = FakeEmbeddingProvider(dimension=128)
    store = ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=provider)

    # Add sample chunks to advisory_knowledge
    chunks = [
        {
            "content": "资产配置的核心原则是风险与收益的匹配。保守型投资者应配置更大比例的固收类资产。",
            "metadata": {
                "title": "资产配置核心原则",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "content_hash": "abc123def456",
                "chunk_index": 0,
                "token_count": 50,
            },
        },
        {
            "content": "基金定投是通过定期定额投资来分散买入时点风险的策略。适合长期储蓄目标。",
            "metadata": {
                "title": "基金定投原则",
                "source_type": "investment_knowledge",
                "collection_name": "advisory_knowledge",
                "content_hash": "def789abc012",
                "chunk_index": 0,
                "token_count": 45,
            },
        },
    ]
    store.add_chunks(chunks, "advisory_knowledge")
    return store


def test_retriever_chroma_path(chroma_store_with_data):
    """Retriever with Chroma should return Chroma results (not mock)."""
    retriever = KnowledgeRetriever(chroma_store=chroma_store_with_data)
    assert retriever.has_chroma is True

    sources = retriever.retrieve([
        {"query": "资产配置", "source_type": "investment_knowledge"},
    ])
    assert len(sources) >= 1
    # The title should come from our chroma data, not mock
    titles = {s.title for s in sources}
    assert "资产配置核心原则" in titles or "基金定投原则" in titles


def test_retriever_fallback_when_chroma_empty(temp_chroma_dir):
    """Empty Chroma store → should fall back to mock."""
    provider = FakeEmbeddingProvider(dimension=128)
    store = ChromaStore(persist_dir=temp_chroma_dir, embedding_provider=provider)
    # Don't add any data — store is empty

    retriever = KnowledgeRetriever(chroma_store=store)
    assert retriever.has_chroma is False  # No populated collections

    sources = retriever.retrieve([
        {"query": "资产配置", "source_type": "investment_knowledge"},
    ])
    # Should get mock results
    assert len(sources) >= 1
    mock_titles = {s.title for s in sources}
    assert "资产配置基础原则" in mock_titles  # from mock store
