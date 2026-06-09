"""
EmbeddingProvider — cloud embedding via DashScope (OpenAI-compatible API).

Default model: text-embedding-v4
API Key: DASHSCOPE_API_KEY environment variable
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1

Raises clear errors when the API key is not configured — never silently
degrades to fake vectors.
"""
from __future__ import annotations

import os
from typing import Any


class EmbeddingProvider:
    """Cloud embedding provider using DashScope's OpenAI-compatible endpoint.

    Usage:
        provider = EmbeddingProvider()
        vectors = provider.embed(["text1", "text2"])   # → [[float, ...], ...]
        single  = provider.embed_single("some text")    # → [float, ...]
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-v4",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or ""
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
        self.base_url = base_url
        self._client: Any = None  # Lazy-init OpenAI client

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _ensure_client(self) -> Any:
        """Lazy-init the OpenAI client, raising if key is missing."""
        if self._client is not None:
            return self._client
        if not self.is_configured:
            raise RuntimeError(
                "EmbeddingProvider: DASHSCOPE_API_KEY is not set. "
                "Set it via environment variable or pass api_key= explicitly."
            )
        try:
            from openai import OpenAI as OpenAIClient
        except ImportError:
            raise RuntimeError(
                "EmbeddingProvider: openai library is required. "
                "Install it with: pip install openai"
            )
        self._client = OpenAIClient(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        return self._client

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Embed multiple texts, batching automatically.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per API call.

        Returns:
            List of embedding vectors in the same order as texts.
        """
        if not texts:
            return []

        client = self._ensure_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(input=batch, model=self.model)
            # response.data is ordered the same as batch
            for emb_data in response.data:
                all_embeddings.append(list(emb_data.embedding))

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Embed a single text. Returns a flat list of floats."""
        results = self.embed([text])
        return results[0] if results else []


# ── Fake/Test provider ──────────────────────────────────────────────

class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake embedding provider for unit tests.

    Produces fixed-length vectors derived from text content (NOT random).
    The same text always yields the same vector, enabling reproducible tests.
    """

    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self._dim = dimension

    @property
    def is_configured(self) -> bool:
        return True

    def _ensure_client(self) -> Any:
        return self  # no real client needed

    def embed(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Generate deterministic pseudo-embeddings from text hash."""
        import hashlib
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            # Use hash bytes to seed a deterministic vector
            vec: list[float] = []
            for i in range(self._dim):
                b = h[i % len(h)]
                # Map byte 0-255 → -0.5 to 0.5
                vec.append((b / 255.0) - 0.5)
            # Normalize to unit length for cosine similarity
            norm = sum(v * v for v in vec) ** 0.5
            if norm > 0:
                vec = [v / norm for v in vec]
            results.append(vec)
        return results
