"""Embedding via OpenAI-compatible /v1/embeddings endpoint (vLLM, OpenAI)."""
from typing import List

import httpx

from logger import get_logger
from rag.embedder_base import EmbedderClient

logger = get_logger(__name__)


class EmbeddingError(RuntimeError):
    pass


class OpenAIEmbedder(EmbedderClient):

    def __init__(self, base_url: str, model: str, api_key: str = "not_needed") -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120.0,
        )

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
                    "/embeddings",
                    json={"model": self._model, "input": batch},
                )
            except httpx.ConnectError as exc:
                raise EmbeddingError(
                    f"Cannot reach embedding server at {self._client.base_url}. "
                    "Make sure the server is running."
                ) from exc
            except httpx.ReadError:
                # Stale connection from pool — retry once with a fresh request.
                logger.warning("ReadError on embedding request — retrying once.")
                try:
                    response = await self._client.post(
                        "/embeddings",
                        json={"model": self._model, "input": batch},
                    )
                except httpx.ReadError as exc:
                    raise EmbeddingError("Embedding request failed after retry.") from exc

            if response.status_code != 200:
                raise EmbeddingError(
                    f"/v1/embeddings returned {response.status_code}: {response.text}"
                )

            data = response.json()
            batch_embeddings = [item["embedding"] for item in data.get("data", [])]
            if not batch_embeddings:
                raise EmbeddingError(
                    f"/v1/embeddings response missing 'data' key: {data}"
                )

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def close(self) -> None:
        await self._client.aclose()
