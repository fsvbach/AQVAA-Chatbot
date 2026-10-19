# llm/types.py
from dataclasses import dataclass, field
from typing import Any, TypedDict

@dataclass
class ChatChunk:
    content: str | None

@dataclass
class ChatResponse:
    content: str | None
    input_tokens: int
    output_tokens: int
    logprobs: list[Any] | None = field(default=None)  # list[ChatCompletionTokenLogprob] from openai SDK


class LLMOptions(TypedDict, total=False):
    temperature: float
    reasoning: bool
    top_p: float
    top_k: int
    logprobs:bool
    top_logprobs: int
