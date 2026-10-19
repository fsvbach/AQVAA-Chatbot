from abc import ABC, abstractmethod
from typing import List


class EmbedderClient(ABC):

    @abstractmethod
    async def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        pass

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_texts([text], batch_size=1)
        return results[0]

    @abstractmethod
    async def close(self) -> None:
        pass
