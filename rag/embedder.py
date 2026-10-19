"""Embedding API — provider-agnostic wrapper around EmbedderClient.

Callers import embed_texts, embed_query, and close_embedder_client from here.
The concrete implementation is chosen by embedder_factory based on LLM_LLM_PROVIDER.
"""
from typing import List

from rag.embedder_base import EmbedderClient
from rag.embedder_factory import get_embedder

_client: EmbedderClient | None = None


def _get_client() -> EmbedderClient:
    global _client
    if _client is None:
        _client = get_embedder()
    return _client


async def close_embedder_client() -> None:
    """Close the shared embedder client. Call during application shutdown."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    return await _get_client().embed_texts(texts, batch_size=batch_size)


async def embed_query(text: str) -> List[float]:
    return await _get_client().embed_query(text)
