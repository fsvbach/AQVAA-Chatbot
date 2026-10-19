

from openinference.semconv.trace import SpanAttributes

from llm.types import ChatChunk, ChatResponse, LLMOptions
from llm.base import LLMClient, FormatType
import ollama
from logger import get_logger, get_tracer

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


def _set_input_messages(span, messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        span.set_attribute(f"llm.input_messages.{i}.message.role", msg.get("role", ""))
        span.set_attribute(f"llm.input_messages.{i}.message.content", msg.get("content", ""))


class OllamaClient(LLMClient):

    def __init__(self, base_url: str, model: str):
        logger.info("Creating Ollama client: base_url=%s, model=%s", base_url, model)
        self.client = ollama.AsyncClient(host=base_url)
        self.model = model

    async def stream(self, messages: list[dict], metadata: dict | None = None, options: LLMOptions | None = None, format: FormatType = None):
        with _tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model)
            _set_input_messages(span, messages)

            tokens: list[str] = []
            last_chunk = None
            async for chunk in await self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                format=format,
                options={"temperature": options["temperature"]} if options and "temperature" in options else None,
            ):
                last_chunk = chunk
                content = chunk.message.content or ""
                tokens.append(content)
                yield ChatChunk(content=content)

            full_output = "".join(tokens)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, full_output)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", full_output)

            if last_chunk:
                input_tokens = last_chunk.prompt_eval_count or 0
                output_tokens = last_chunk.eval_count or 0
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens)
                span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens)
                if metadata is not None:
                    metadata["tokens_input"] = input_tokens
                    metadata["tokens_output"] = output_tokens

    async def complete(self, messages: list[dict], options: LLMOptions | None = None, format: FormatType = None) -> ChatResponse:
        with _tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model)
            _set_input_messages(span, messages)

            response = await self.client.chat(
                model=self.model,
                messages=messages,
                format=format,
                options={"temperature": options["temperature"]} if options and "temperature" in options else None,
            )
            content = response.message.content or ""
            input_tokens = response.prompt_eval_count or 0
            output_tokens = response.eval_count or 0

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, content)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", content)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens)

            return ChatResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens)