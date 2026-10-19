from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Literal

from llm.types import ChatChunk, ChatResponse, LLMOptions

FormatType = Literal["", "json"] | dict[str, Any] | None

class LLMClient(ABC):

    @abstractmethod
    def stream(self, messages: list[dict], metadata: dict | None = None, options: LLMOptions | None = None, format: FormatType = None) -> AsyncGenerator[ChatChunk, None]:
        pass

    @abstractmethod
    async def complete(self, messages: list[dict], options: LLMOptions | None = None, format: FormatType = None) -> ChatResponse:
        pass



