"""Embedding provider abstraction with local and API-based backends.

Supports:
- LocalEmbeddingProvider: uses sentence-transformers, no API key needed, Chinese-capable
- OpenAIEmbeddingProvider: wraps existing OpenAI LLMProvider.embed()
- DoubaoEmbeddingProvider: 火山方舟豆包 embedding API (2048-dim, OpenAI-compatible)

The factory `create_embedding_provider()` picks the right one based on config,
returning None if no embeddings available (system gracefully falls back to keyword search).
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from persona_agent.providers.base import LLMProvider


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the text."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch (default: one-by-one)."""
        return [await self.embed(t) for t in texts]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding via sentence-transformers (no API, supports Chinese).

    Lazy-loads the model on first call. If sentence-transformers is not
    installed, raises ImportError on first use.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._load_lock:
            if self._model is not None:
                return
            # Load in thread to avoid blocking event loop
            self._model = await asyncio.to_thread(self._load_model)

    def _load_model(self) -> Any:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        print(f"[embedding] loading model {self.model_name} on {self.device}...")
        model = SentenceTransformer(self.model_name, device=self.device)
        print(f"[embedding] model loaded")
        return model

    def _embed_sync(self, text: str) -> list[float]:
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in arr]

    async def embed(self, text: str) -> list[float]:
        await self._ensure_loaded()
        return await asyncio.to_thread(self._embed_sync, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_loaded()
        return await asyncio.to_thread(self._embed_batch_sync, texts)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Adapter that wraps an existing LLMProvider's embed() method."""

    def __init__(self, llm_provider: LLMProvider):
        self._llm = llm_provider

    async def embed(self, text: str) -> list[float]:
        return await self._llm.embed(text)


class DoubaoEmbeddingProvider(EmbeddingProvider):
    """火山方舟豆包 embedding API.

    Supports both text and multimodal (vision) embedding endpoints.
    Uses the multimodal endpoint (/embeddings/multimodal) by default,
    which works with both text-only and vision models.

    Requires ARK_API_KEY env var or explicit api_key parameter.
    'model' should be the endpoint ID (ep-xxx) from Volcengine console.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "doubao-embedding-large-text-240915",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    ):
        self._api_key = api_key or os.environ.get("ARK_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        client = self._get_client()
        resp = await client.post(
            f"{self._base_url}/embeddings/multimodal",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": [{"type": "text", "text": text}],
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Doubao embedding API error {resp.status_code}: {resp.text}")

        data = resp.json()
        # Multimodal response: data.data.embedding (single dict, not array)
        return data["data"]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (one API call per text for multimodal endpoint)."""
        # Multimodal endpoint accepts one input per call, so we parallelize
        tasks = [self.embed(t) for t in texts]
        return await asyncio.gather(*tasks)


def create_embedding_provider(
    kind: str = "local",
    model: str | None = None,
    llm_provider: LLMProvider | None = None,
    device: str = "cpu",
    api_key: str | None = None,
    base_url: str | None = None,
) -> EmbeddingProvider | None:
    """Factory for embedding providers.

    kind:
        "local"    -> LocalEmbeddingProvider (sentence-transformers)
        "openai"   -> OpenAIEmbeddingProvider (wraps llm_provider)
        "doubao"   -> DoubaoEmbeddingProvider (火山方舟, 2048-dim)
        "none"     -> returns None (keyword search only)

    Returns None if the requested provider can't be created.
    """
    if kind in ("none", "disabled", None):
        return None

    if kind == "local":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            print("[embedding] sentence-transformers not installed, falling back to keyword search")
            return None
        model_name = model or "BAAI/bge-small-zh-v1.5"
        return LocalEmbeddingProvider(model_name=model_name, device=device)

    if kind == "openai":
        if llm_provider is None:
            print("[embedding] openai backend requires llm_provider, skipping")
            return None
        return OpenAIEmbeddingProvider(llm_provider)

    if kind == "doubao":
        key = api_key or os.environ.get("ARK_API_KEY", "")
        if not key:
            print("[embedding] doubao backend requires ARK_API_KEY, skipping")
            return None
        model_name = model or "doubao-embedding-large-text-240915"
        url = base_url or "https://ark.cn-beijing.volces.com/api/v3"
        return DoubaoEmbeddingProvider(api_key=key, model=model_name, base_url=url)

    print(f"[embedding] unknown embedding kind: {kind}")
    return None
