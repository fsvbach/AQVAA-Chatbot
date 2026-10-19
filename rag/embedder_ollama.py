"""Embedding via Ollama /api/embed endpoint."""
from typing import List

import httpx

from logger import get_logger
from rag.embedder_base import EmbedderClient

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder(EmbedderClient):

    def __init__(self, base_url: str, model: str) -> None:
        self._model = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

    async def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            logger.debug(
                "Embedding batch %d-%d of %d texts",
                batch_start,
                batch_start + len(batch) - 1,
                len(texts),
            )
            try:
                response = await self._client.post(
                    "/api/embed",
                    json={"model": self._model, "input": batch},
                )
            except httpx.ConnectError as exc:
                raise EmbeddingError(
                    f"Cannot reach Ollama at {self._client.base_url}. "
                    "Make sure Ollama is running."
                ) from exc

            if response.status_code != 200:
                raise EmbeddingError(
                    f"Ollama /api/embed returned {response.status_code}: {response.text}"
                )

            data = response.json()
            batch_embeddings = data.get("embeddings")
            if not batch_embeddings:
                raise EmbeddingError(
                    f"Ollama /api/embed response missing 'embeddings' key: {data}"
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def close(self) -> None:
        await self._client.aclose()
