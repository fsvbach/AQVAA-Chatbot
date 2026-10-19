"""Query rewriting / contextualization for the retrieval pipeline.

Turns a (possibly elliptical) follow-up message into a single, self-contained
search query, resolving pronouns and references against the chat history and the
VAA statement the user is currently answering.

The chat history is embedded into the system prompt (not passed as real
conversation turns), so the model frames the task as a transformation — rewrite
the latest message — rather than a conversation to continue.

Falls back to the original user query on any error so that rewriting never
blocks a legitimate query.
"""
import asyncio
from typing import Literal

from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import StatusCode

from llm import factory
from config import settings, LLMRole
from logger import get_logger, get_tracer
from models.schemas import ChatHistoryMessage, ChatbotMode
from prompts.query_rewriting_prompt import ASSISTANT_CURRENT as _ASSISTANT_REWRITE_SYSTEM_PROMPT
from prompts.query_rewriting_prompt import INTERVIEW_CURRENT as _INTERVIEW_REWRITE_SYSTEM_PROMPT

logger = get_logger(__name__)
_tracer = get_tracer(__name__)

_REWRITE_TIMEOUT = 15.0

_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}


def _render_chat_history(chat_history: list[ChatHistoryMessage]) -> str:
    """Render the chat history into a transcript for embedding in the prompt."""
    if not chat_history:
        return "(no previous messages)"
    return "\n".join(
        f"{_ROLE_LABELS.get(m.role, m.role)}: {m.content}" for m in chat_history
    )


async def get_contextualized_user_query(
    user_query: str,
    question_text: str,
    chat_history: list[ChatHistoryMessage] = [],
    mode: ChatbotMode = "assistant"
) -> str:
    """Rewrite ``user_query`` into a self-contained retrieval query.

    Uses the VAA statement (``question_text``) and ``chat_history`` to resolve
    references. Returns the original ``user_query`` unchanged on any error or
    empty response.
    """
    with _tracer.start_as_current_span("query_rewriting") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, user_query)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, settings.profile(LLMRole.REWRITING).model)

        rewrite_system_prompt = _ASSISTANT_REWRITE_SYSTEM_PROMPT if mode == "assistant" else _INTERVIEW_REWRITE_SYSTEM_PROMPT

        system_prompt = rewrite_system_prompt.format(
            question_text=question_text,
            chat_history=_render_chat_history(chat_history),
        )
        span.set_attribute(SpanAttributes.LLM_SYSTEM, system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        try:
            client = factory.get_client(LLMRole.REWRITING)
            async with asyncio.timeout(_REWRITE_TIMEOUT):
                response = await client.complete(
                    messages=messages,
                    options={"reasoning": False},
                )
            rewritten = (response.content or "").strip()
            if not rewritten:
                raise ValueError("response.content is empty")

            logger.info("Query rewritten: %r -> %r", user_query, rewritten)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, rewritten)
            return rewritten
        except Exception as exc:
            logger.warning(
                "Query rewriting failed (%s: %s) — falling back to original query",
                type(exc).__name__,
                exc,
            )
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, description=type(exc).__name__)
            return user_query
