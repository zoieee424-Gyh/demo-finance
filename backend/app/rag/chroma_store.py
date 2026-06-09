"""
ChromaStore — Chroma vector database wrapper.

Uses Chroma PersistentClient with a configurable persist directory.
Each collection represents a single business domain (advisory, compliance,
education, risk, etc.).

Key operations:
  - get_or_create_collection(name)
  - add_chunks(chunks, collection_name)
  - semantic_search(query, collection_names, top_k)

Result format:
    [
        {
            "content": str,
            "score": float,        # 1 - distance (cosine)
            "metadata": {...},
            "chunk_id": str,
        },
        ...
    ]
"""
from __future__ import annotations

from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction


class ChromaStore:
    """Chroma vector store backed by PersistentClient.

    Usage:
        store = ChromaStore(persist_dir="data/chroma")
        store.add_chunks(chunks, "advisory_knowledge")
        results = store.search("如何配置资产", ["advisory_knowledge"], top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        embedding_provider: Any = None,
    ) -> None:
        """
        Args:
            persist_dir: Directory for Chroma's on-disk persistence.
            embedding_provider: An EmbeddingProvider-like object with an
                                embed() method. If None, created lazily.
        """
        self.persist_dir = persist_dir
        # Wrap the provider as a Chroma EmbeddingFunction
        self._embed_fn = _ProviderEmbeddingFunction(embedding_provider)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_provider = embedding_provider

    # ── Collection management ────────────────────────────────────

    def get_or_create_collection(self, name: str) -> Any:
        """Get an existing collection or create a new one.

        Cosines similarity is used as the distance metric.
        """
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def list_collections(self) -> list[dict]:
        """List all collections with document/vector counts."""
        info: list[dict] = []
        for col in self._client.list_collections():
            info.append({
                "name": col.name,
                "count": col.count(),
                "metadata": col.metadata,
            })
        return info

    def collection_info(self, name: str) -> dict | None:
        """Get info for a specific collection, or None if not found."""
        try:
            col = self._client.get_collection(name, embedding_function=self._embed_fn)
            return {
                "name": col.name,
                "count": col.count(),
                "metadata": col.metadata,
            }
        except Exception:
            return None

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True if it existed."""
        try:
            self._client.delete_collection(name)
            return True
        except Exception:
            return False

    def collection_exists(self, name: str) -> bool:
        """Return True if the collection already exists and has data."""
        info = self.collection_info(name)
        return info is not None and info.get("count", 0) > 0

    # ── Ingest ───────────────────────────────────────────────────

    def prepare_chunk_records(
        self, chunks: list[dict], collection_name: str
    ) -> list[dict]:
        """Prepare chunk records with computed Chroma IDs and sanitized metadata.

        Does NOT write to Chroma. Use this to obtain stable chroma_id values
        before upserting (e.g. for synchronising with a MySQL metadata store).

        Returns a list of dicts, each containing:
            chroma_id, chunk_index, content, metadata (sanitized),
            content_hash, title, token_count
        """
        records: list[dict] = []
        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            if not content.strip():
                continue
            meta = dict(chunk.get("metadata", {}))
            content_hash = meta.get("content_hash", f"chunk_{i}")
            chroma_id = f"{collection_name}_{content_hash}"
            clean_meta = _sanitize_metadata(meta)
            records.append({
                "chroma_id": chroma_id,
                "chunk_index": meta.get("chunk_index", i),
                "content": content,
                "metadata": clean_meta,
                "content_hash": content_hash,
                "title": meta.get("title", ""),
                "token_count": meta.get("token_count", len(content)),
            })
        return records

    def add_chunks(self, chunks: list[dict], collection_name: str) -> int:
        """Add chunks to a collection.

        Each chunk dict must have: content, metadata. Embeddings are generated
        by Chroma through the configured embedding function. When all chunks
        come from the same file_name, old chunks for that file are deleted
        first so re-ingesting an edited document does not leave stale vectors.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        records = self.prepare_chunk_records(chunks, collection_name)
        if not records:
            return 0

        col = self.get_or_create_collection(collection_name)
        self._delete_existing_file_chunks(col, chunks)

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []

        for rec in records:
            documents.append(rec["content"])
            metadatas.append(rec["metadata"])
            ids.append(rec["chroma_id"])

        col.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return len(documents)

    def _delete_existing_file_chunks(self, collection: Any, chunks: list[dict]) -> None:
        """Delete previous chunks for the same source file before replacement."""
        file_names = {
            chunk.get("metadata", {}).get("file_name")
            for chunk in chunks
            if chunk.get("metadata", {}).get("file_name")
        }
        if len(file_names) != 1:
            return

        file_name = next(iter(file_names))
        try:
            collection.delete(where={"file_name": file_name})
        except Exception:
            # Best-effort cleanup. The following upsert still keeps the current
            # ingest functional if Chroma has no prior matching rows.
            return

    # ── Search ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection_names: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Semantic search across one or more collections.

        Args:
            query: Search text.
            collection_names: List of collection names to search.
            top_k: Number of results to return per collection.

        Returns:
            Merged, scored, and sorted list of result dicts.
        """
        all_results: list[dict] = []

        for name in collection_names:
            try:
                col = self._client.get_collection(
                    name, embedding_function=self._embed_fn
                )
            except Exception:
                continue  # skip missing collections gracefully

            if col.count() == 0:
                continue

            raw = col.query(
                query_texts=[query],
                n_results=min(top_k, col.count()),
                include=["documents", "metadatas", "distances"],
            )

            documents = raw.get("documents", [[]])[0]
            metadatas = raw.get("metadatas", [[]])[0]
            distances = raw.get("distances", [[]])[0]

            for i, doc in enumerate(documents):
                distance = distances[i] if i < len(distances) else 1.0
                score = max(0.0, 1.0 - distance)  # cosine distance → similarity
                meta = metadatas[i] if i < len(metadatas) else {}

                all_results.append({
                    "content": doc,
                    "score": round(score, 4),
                    "metadata": meta,
                    "collection_name": name,
                    "title": meta.get("title", ""),
                    "source_type": meta.get("source_type", "knowledge_base"),
                    "chunk_index": meta.get("chunk_index", 0),
                })

        # Sort by score descending
        all_results.sort(key=lambda r: r["score"], reverse=True)
        return all_results[:top_k]


# ── Helpers ─────────────────────────────────────────────────────────


def _sanitize_metadata(meta: dict) -> dict:
    """Ensure all metadata values are Chroma-compatible types."""
    allowed = (str, int, float, bool)
    clean: dict[str, str | int | float | bool] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, allowed):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


class _ProviderEmbeddingFunction(EmbeddingFunction):
    """Bridge: converts an EmbeddingProvider-like object into a Chroma
    EmbeddingFunction so Chroma can call embed() on-demand."""

    def __init__(self, provider: Any = None) -> None:
        self._provider = provider

    def set_provider(self, provider: Any) -> None:
        self._provider = provider

    def _get_provider(self) -> Any:
        if self._provider is None:
            from app.rag.embedding_provider import EmbeddingProvider
            self._provider = EmbeddingProvider()
        return self._provider

    @staticmethod
    def name() -> str:
        """Return a stable Chroma embedding function name."""
        return "dashscope_text_embedding_v4"

    def get_config(self) -> dict:
        """Return Chroma-compatible embedding function config."""
        return {"provider": "dashscope", "model": "text-embedding-v4"}

    @staticmethod
    def build_from_config(config: dict) -> "_ProviderEmbeddingFunction":
        """Build embedding function from Chroma config."""
        return _ProviderEmbeddingFunction()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._get_provider().embed(input)
