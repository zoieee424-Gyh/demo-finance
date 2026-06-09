"""
Text Splitter — clean and chunk documents for Chroma ingestion.

Strategy:
  - Clean: normalize newlines, remove control chars, collapse blank lines,
           preserve Chinese punctuation.
  - Split: RecursiveCharacterTextSplitter from langchain-text-splitters
           (if available) with chunk_size=600-800, overlap=100-150,
           separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""].
  - Fallback: light rule-based splitter when langchain is unavailable.

Each chunk carries:
  - chunk_index: int (0-based)
  - content_hash: str  (SHA-256, first 12 chars)
  - token_count: int  (approximate character count)
"""
from __future__ import annotations

import hashlib
import re
from typing import Any


# ── Default splitter settings ──────────────────────────────────────

DEFAULT_CHUNK_SIZE = 700       # characters
DEFAULT_CHUNK_OVERLAP = 120    # characters
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "，", " ", ""]
CLEAN_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


# ── Public API ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean text for ingestion.

    Steps:
      1. Normalize Windows-style line endings (\\r\\n → \\n).
      2. Remove control characters (except tabs, newlines).
      3. Collapse 3+ consecutive newlines into 2.
      4. Strip leading/trailing whitespace.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CLEAN_PATTERN.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def compute_content_hash(content: str) -> str:
    """Return a short SHA-256 hex digest of the content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]


def split_text(
    content: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Split cleaned text into overlapping chunks.

    Args:
        content: Cleaned text string.
        chunk_size: Target characters per chunk.
        chunk_overlap: Characters of overlap between chunks.

    Returns:
        [
            {
                "content": str,
                "chunk_index": int,
                "content_hash": str,
                "token_count": int,
            },
            ...
        ]
    """
    # Prefer LangChain splitter
    try:
        chunks = _split_with_langchain(content, chunk_size, chunk_overlap)
    except (ImportError, Exception):
        chunks = _split_fallback(content, chunk_size, chunk_overlap)

    # Enrich each chunk with metadata fields
    enriched: list[dict] = []
    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if not chunk_text:
            continue
        enriched.append({
            "content": chunk_text,
            "chunk_index": i,
            "content_hash": compute_content_hash(chunk_text),
            "token_count": len(chunk_text),
        })

    return enriched


def split_document(
    doc: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Clean a document and split it into chunks.

    Each chunk inherits the document's metadata plus its own chunk-level fields.
    """
    cleaned = clean_text(doc.get("content", ""))
    chunks = split_text(cleaned, chunk_size, chunk_overlap)

    base_meta = dict(doc.get("metadata", {}))
    base_meta["title"] = doc.get("title", "")

    for chunk in chunks:
        chunk_meta = dict(base_meta)
        chunk_meta.update({
            "chunk_index": chunk["chunk_index"],
            "content_hash": chunk["content_hash"],
            "token_count": chunk["token_count"],
        })
        chunk["metadata"] = chunk_meta

    return chunks


# ── LangChain splitter ──────────────────────────────────────────────

def _split_with_langchain(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(DEFAULT_SEPARATORS),
        length_function=len,  # character-level, not token-level
    )
    # split_text returns list[str]
    return splitter.split_text(text)


# ── Fallback splitter ───────────────────────────────────────────────

def _split_fallback(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Light rule-based splitter when langchain-text-splitters is unavailable.

    Splits on paragraph breaks first, then sentence breaks, then character
    boundaries. Maintains chunk_overlap between consecutive chunks.
    """
    if len(text) <= chunk_size:
        return [text]

    # First, split into paragraphs
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph exceeds chunk_size, split it on sentences
            if len(para) > chunk_size:
                sub_chunks = _split_long_paragraph(para, chunk_size, chunk_overlap)
                # The last sub-chunk becomes the new current for overlap
                if sub_chunks:
                    mid = sub_chunks[:-1]
                    tail = sub_chunks[-1]
                    chunks.extend(mid)
                    # Apply overlap: keep tail as the start of next segment
                    current = tail
                else:
                    current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Apply chunk_overlap: prepend a suffix of the previous chunk
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            if len(prev) > chunk_overlap:
                overlap_text = prev[-chunk_overlap:]
                overlapped.append(overlap_text + "\n" + chunks[i])
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return chunks if chunks else [text]


def _split_long_paragraph(
    text: str, chunk_size: int, _chunk_overlap: int
) -> list[str]:
    """Split an overly long paragraph on sentence boundaries."""
    # Try splitting on sentence-ending punctuation
    sentences = re.split(r"(?<=[。！？\.!\?])", text)
    chunks: list[str] = []
    current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) <= chunk_size:
            current = current + sent if current else sent
        else:
            if current:
                chunks.append(current)
            # If a single sentence is too long, hard-cut
            if len(sent) > chunk_size:
                for i in range(0, len(sent), chunk_size):
                    chunks.append(sent[i:i + chunk_size])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks if chunks else [text]
