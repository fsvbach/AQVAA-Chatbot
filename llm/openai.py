# llm/openai.py
from openai import AsyncOpenAI
from openinference.semconv.trace import SpanAttributes
from typing import cast
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from llm.types import ChatChunk, ChatResponse, LLMOptions
from llm.base import LLMClient, FormatType
from logger import get_logger, get_tracer

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


def _set_input_messages(span, messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        span.set_attribute(f"llm.input_messages.{i}.message.role", msg.get("role", ""))
        span.set_attribute(f"llm.input_messages.{i}.message.content", msg.get("content", ""))


class OpenAIClient(LLMClient):

    def __init__(self, base_url: str, api_key: str, model: str):
        logger.info("Creating OpenAI-compatible client: base_url=%s, model=%s", base_url, model)
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _extra_params(self, options: LLMOptions | None, format: FormatType) -> dict:
        extra = {}
        if isinstance(format, dict):
            extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "response","strict": True, "schema": format}}
        if options and "temperature" in options:
            extra["temperature"] = options["temperature"]
        if options and "top_p" in options:
            extra["top_p"] = options["top_p"]
        if options and options.get("reasoning"):
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        if options and "top_k" in options:
            extra.setdefault("extra_body", {})["top_k"] = options["top_k"]
        if options and options.get("logprobs"):
            extra["logprobs"] = True
        if options and "top_logprobs" in options:
            extra["top_logprobs"] = options["top_logprobs"]
        return extra

    async def stream(self, messages: list[dict], metadata: dict | None = None, options: LLMOptions | None = None, format: FormatType = None):
        with _tracer.start_as_current_span("llm.stream") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, self.model)
            _set_input_messages(span, messages)

            tokens: list[str] = []
            reasoning_tokens: list[str]=[]
            input_tokens = 0
            output_tokens = 0
            
            async with await self.client.chat.completions.create(
                messages=cast(list[ChatCompletionMessageParam], messages),
                model=self.model,
                stream=True,
                stream_options={"include_usage": True},
                **self._extra_params(options, format),
                ) as stream:
                async for chunk in stream:
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                    delta = chunk.choices[0].delta if chunk.choices else None
                    content = delta.content or "" if delta else ""
                    reasoning = getattr(delta, "reasoning", None) or ""
                    tokens.append(content)
                    reasoning_tokens.append(reasoning)
                    yield ChatChunk(content=content)
            full_output = "".join(tokens)
            full_reasoning = "".join(reasoning_tokens)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, full_output)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", full_output)
            if full_reasoning:
                span.set_attribute("llm.reasoning_content", full_reasoning)
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

            response: ChatCompletion = await self.client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                **self._extra_params(options, format),
            )
            message = response.choices[0].message
            content = message.content or ""
            if not content:
                # vLLM + Qwen3: when thinking is ON and JSON schema constrained decoding
                # is active, the model somehow places output inside the think block. vLLM routes
                # that to message.reasoning,so content is empty empty.
                # TODO: investigate this further and find a cleaner fix
                content = getattr(message, "reasoning", None) or ""
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            reasoning = getattr(message, "reasoning", None) or ""

            span.set_attribute(SpanAttributes.OUTPUT_VALUE, content)
            span.set_attribute("llm.output_messages.0.message.role", "assistant")
            span.set_attribute("llm.output_messages.0.message.content", content)
            if reasoning and reasoning != content:
                span.set_attribute("llm.reasoning_content", reasoning)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens)
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens)

            choice = response.choices[0]
            logprobs = choice.logprobs.content if choice.logprobs else None

            return ChatResponse(content=content, input_tokens=input_tokens, output_tokens=output_tokens, logprobs=logprobs)
